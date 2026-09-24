"""Train and evaluate a temporal plan model on immutable parent cohorts."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

import numpy as np
import pandas as pd
from plan_value_run import compact_targets, decision_metrics
from price_pilot import load_arrays, verify_files

from quant_research.plan_value import (
    apply_calibration,
    calibration_bins,
    compare_heads,
    fit_calibration,
    head_metrics,
)
from quant_research.storage import file_hash, utc_now, write_json


def read(path):
    return json.loads(path.read_text())


def freeze(args):
    prior, out = args.prior.resolve(), args.output.resolve()
    state = read(prior/'run-status.json')
    if state['status'] != 'completed' or not read(prior/'verification.json')['passed']:
        raise ValueError('Parent must be complete and verified')
    verify_files(prior, state['files'])
    base = Path(__file__).resolve().parents[1]
    for name in ['plan_value.py', 'plan_targets.py', 'price_strategy.py']:
        if file_hash(base/'src/quant_research'/name) != file_hash(prior/'code/src/quant_research'/name):
            raise ValueError('Plan definitions differ from parent')
    protocol = read(args.protocol)
    parent_protocol = read(prior/'protocol.json')
    if protocol['sealed_holdout_start'] != parent_protocol['sealed_holdout_start']:
        raise ValueError('Holdout boundary differs')
    fold = parent_protocol['fold_indices'][0]
    cfg = read(prior/f'fold-{fold:02d}'/'config.json')
    panel = Path(cfg['root'])/'panel-h5'
    if file_hash(panel/'manifest.json') != cfg['source_hashes']['panel_manifest']:
        raise ValueError('Panel manifest differs from parent')
    verify_files(panel, read(panel/'manifest.json')['files'])
    out.mkdir()
    write_json(out/'protocol.json', protocol)
    write_json(out/'experiment.json', {'created_at': utc_now(), 'prior': str(prior),
        'parent_status_sha256': file_hash(prior/'run-status.json'),
        'fold_indices': parent_protocol['fold_indices'], 'panel': str(panel),
        'panel_manifest_sha256': file_hash(panel/'manifest.json'), 'executable': False})
    shutil.copytree(base/'src/quant_research', out/'code/src/quant_research',
                    ignore=shutil.ignore_patterns('__pycache__'))
    (out/'code/scripts').mkdir()
    for name in ['plan_transformer_run.py', 'plan_value_run.py', 'price_pilot.py', 'model_quality_run.py']:
        shutil.copy2(base/'scripts'/name, out/'code/scripts'/name)
    shutil.copy2(base/'uv.lock', out/'code/uv.lock')
    write_json(out/'frozen-manifest.json', {str(f.relative_to(out)): file_hash(f)
        for f in out.rglob('*') if f.is_file()})
    print('Frozen', out, flush=True)


def fit(args):
    # Torch is isolated from any LightGBM native runtime in the supervisor.
    from quant_research.plan_transformer import predict, train
    out = args.output.resolve()
    verify_files(out, read(out/'frozen-manifest.json'))
    experiment, protocol = read(out/'experiment.json'), read(out/'protocol.json')
    if args.fold not in experiment['fold_indices']:
        raise ValueError('Unregistered fold')
    prior = Path(experiment['prior'])/f'fold-{args.fold:02d}'
    panel = Path(experiment['panel'])
    if file_hash(panel/'manifest.json') != experiment['panel_manifest_sha256']:
        raise ValueError('Panel manifest changed')
    values = np.load(panel/'values.npy', mmap_mode='r')
    dest = out/f'fold-{args.fold:02d}'
    dest.mkdir()
    write_json(dest/'worker.json', {'pid': os.getpid(), 'started_at': utc_now()})
    rows = {p: pd.read_parquet(prior/f'{p}-rows.parquet') for p in
            ['train', 'selection', 'calibration', 'evaluation']}
    tr, selection = load_arrays(prior/'train.npz'), load_arrays(prior/'selection.npz')
    model, mean, scale = train(values, rows['train'], tr, compact_targets(tr),
        rows['selection'], selection, compact_targets(selection), dest, protocol)
    del tr, selection
    cal = load_arrays(prior/'calibration.npz')
    raw_cal, prices_cal = predict(model, values, rows['calibration'], mean, scale)
    np.savez_compressed(dest/'raw-calibration.npz', **raw_cal)
    np.save(dest/'price-calibration.npy', prices_cal)
    calibration = fit_calibration(raw_cal, compact_targets(cal), rows['calibration'].date)
    write_json(dest/'calibration.json', calibration)
    del cal
    evaluation = load_arrays(prior/'evaluation.npz')
    raw, prices = predict(model, values, rows['evaluation'], mean, scale)
    np.save(dest/'price-evaluation.npy', prices)
    forecasts = {'transformer_raw': raw, 'transformer': apply_calibration(raw, calibration)}
    window = f'fold-{args.fold:02d}'
    for name, prediction in forecasts.items():
        np.savez_compressed(dest/f'{name}-evaluation.npz', **prediction)
        metrics, chosen = decision_metrics(rows['evaluation'], prediction, evaluation, window, name)
        metrics.to_csv(dest/f'{name}-decision-metrics.csv', index=False)
        chosen.to_parquet(dest/f'{name}-chosen.parquet', index=False)
    head_metrics(rows['evaluation'], compact_targets(evaluation), forecasts, window).to_csv(
        dest/'head-metrics.csv', index=False)
    calibration_bins(rows['evaluation'], compact_targets(evaluation), forecasts, window).to_csv(
        dest/'calibration-bins.csv', index=False)
    write_json(dest/'progress.json', {'status': 'completed', 'updated_at': utc_now()})
    write_json(dest/'completed.json', {'status': 'completed', 'updated_at': utc_now(), 'files': {
        str(f.relative_to(dest)): file_hash(f) for f in dest.rglob('*') if f.is_file()}})


def run(args):
    out = args.output.resolve()
    verify_files(out, read(out/'frozen-manifest.json'))
    experiment = read(out/'experiment.json')
    prior, panel = Path(experiment['prior']), Path(experiment['panel'])
    if file_hash(prior/'run-status.json') != experiment['parent_status_sha256']:
        raise ValueError('Parent changed')
    verify_files(prior, read(prior/'run-status.json')['files'])
    verify_files(panel, read(panel/'manifest.json')['files'])
    marker = out/'run-status.json'
    if marker.exists():
        raise FileExistsError('Preserve previous attempts')
    write_json(marker, {'status': 'running', 'pid': os.getpid(), 'started_at': utc_now()})
    metrics, decisions, bins = [], [], []
    try:
        for index, fold in enumerate(experiment['fold_indices']):
            name = f'fold-{fold:02d}'
            write_json(out/'progress.json', {'status': 'running', 'current': name,
                'completed_fits': index, 'total_fits': len(experiment['fold_indices']), 'updated_at': utc_now()})
            print('Starting', name, flush=True)
            with (out/f'{name}.log').open('w') as log:
                subprocess.run([sys.executable, str(out/'code/scripts/plan_transformer_run.py'),
                    'fit', '--output', str(out), '--fold', str(fold)],
                    env={**os.environ, 'PYTHONPATH': str(out/'code/src')},
                    stdout=log, stderr=subprocess.STDOUT, check=True)
            dest = out/name
            verify_files(dest, read(dest/'completed.json')['files'])
            metrics.append(pd.read_csv(dest/'head-metrics.csv'))
            bins.append(pd.read_csv(dest/'calibration-bins.csv'))
            decisions.extend(pd.read_csv(dest/f'{m}-decision-metrics.csv')
                             for m in ['transformer', 'transformer_raw'])
            print('Completed', name, flush=True)
        baseline = pd.read_csv(prior/'head-metrics.csv')
        baseline = baseline.loc[baseline.model.isin(['learned', 'learned_raw'])].replace(
            {'model': {'learned': 'baseline27', 'learned_raw': 'baseline27_raw'}})
        combined = pd.concat([baseline, *metrics], ignore_index=True)
        combined.to_csv(out/'head-metrics.csv', index=False)
        pd.concat(bins, ignore_index=True).to_csv(out/'calibration-bins.csv', index=False)
        pd.concat(decisions, ignore_index=True).to_csv(out/'decision-metrics.csv', index=False)
        comparisons = {}
        for suffix in ['', '_raw']:
            pair = combined.loc[combined.model.isin(['transformer'+suffix, 'baseline27'+suffix])].replace(
                {'model': {'transformer'+suffix: 'learned', 'baseline27'+suffix: 'empirical'}})
            comparisons['calibrated' if not suffix else 'raw'] = compare_heads(pair)
        write_json(out/'assessment.json', {'comparisons': comparisons,
            'reference': 'frozen learned27, not empirical constant',
            'training': {f: read(out/f'fold-{f:02d}'/'training-result.json') for f in experiment['fold_indices']},
            'executable': False, 'quality_promotion': False})
        write_json(out/'progress.json', {'status': 'completed', 'completed_fits': len(metrics), 'updated_at': utc_now()})
        write_json(marker, {'status': 'completed', 'finished_at': utc_now(), 'files': {
            str(f.relative_to(out)): file_hash(f) for f in out.rglob('*') if f.is_file()
            and f.name != 'run-status.json' and '__pycache__' not in f.parts}})
    except BaseException as exc:
        write_json(marker, {'status': 'failed', 'error': repr(exc), 'updated_at': utc_now()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'run', 'fit'])
    parser.add_argument('--prior', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--fold', type=int)
    args = parser.parse_args()
    globals()[args.stage](args)


if __name__ == '__main__':
    main()
