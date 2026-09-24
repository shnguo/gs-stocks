"""Freeze, run and compare temporal freshness and date-density ablations."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Resolve the matching frozen package when invoked from an artifact snapshot.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

import numpy as np
import pandas as pd
from model_quality_run import audit_window
from price_pilot import load_arrays, verify_files

from quant_research.model_quality import (
    daily_metrics,
    fit_volatility_baseline,
    volatility_scale,
    weighted_quantile,
)
from quant_research.rolling_price import (
    build_date_plans,
    paired_comparison,
    prediction_diagnostics,
)
from quant_research.storage import file_hash, utc_now, write_json


def read(path):
    return json.loads(path.read_text())


def freeze(args):
    root, out = args.root.resolve(), args.output.resolve()
    protocol = read(args.protocol)
    schedule = read(root/'schedule.json')
    calendar = read(root/'panel-h5/manifest.json')['dates']
    plans = {f'fold-{i:02d}': build_date_plans(calendar, schedule['development'][i],
        schedule['sealed_holdout_start'], protocol['evaluation_dates']) for i in protocol['fold_indices']}
    out.mkdir()
    write_json(out/'protocol.json', protocol)
    write_json(out/'experiment.json', {'created_at': utc_now(), 'root': str(root),
        'sealed_holdout_start': schedule['sealed_holdout_start'],
        'source_files': {n: file_hash(root/n) for n in ['schedule.json', 'development-acceptance.json']},
        'daily_strategy_ready': False, 'prior_quality_gate_overridden': False})
    for fold, arms in plans.items():
        for arm, plan in arms.items():
            write_json(out/f'{fold}-{arm}-dates.json', plan)
    base = Path(__file__).resolve().parents[1]
    shutil.copytree(base/'src/quant_research', out/'code/src/quant_research',
                    ignore=shutil.ignore_patterns('__pycache__'))
    (out/'code/scripts').mkdir()
    for name in ['rolling_price_run.py', 'model_quality_run.py', 'price_pilot.py']:
        shutil.copy2(base/'scripts'/name, out/'code/scripts'/name)
    shutil.copy2(base/'uv.lock', out/'code/uv.lock')
    write_json(out/'frozen-manifest.json', {str(f.relative_to(out)): file_hash(f)
        for f in out.rglob('*') if f.is_file()})
    print('Frozen', out, flush=True)


def run(args):
    out = args.output.resolve()
    verify_files(out, read(out/'frozen-manifest.json'))
    protocol, experiment = read(out/'protocol.json'), read(out/'experiment.json')
    root = Path(experiment['root'])
    verify_files(root, experiment['source_files'])
    marker = out/'run-status.json'
    if marker.exists():
        raise FileExistsError('Keep attempted runs immutable; use a new version')
    write_json(marker, {'status': 'running', 'pid': os.getpid(), 'started_at': utc_now()})
    env = {**os.environ, 'PYTHONPATH': str(out/'code/src')}
    names = read(root/'panel-h5/manifest.json')['feature_names']
    volatility_index = names.index('volatility_20')
    metrics, diagnostics, audits = [], [], {}
    try:
        for fold in protocol['fold_indices']:
            common_rows, common_targets = None, None
            window = f'fold-{fold:02d}'
            for arm in protocol['arms']:
                name, directory = f'{window}-{arm}', out/f'{window}-{arm}'
                write_json(out/'progress.json', {'status': 'running', 'current': name,
                    'completed_fits': len(audits), 'total_fits': len(protocol['arms'])*len(protocol['fold_indices']),
                    'updated_at': utc_now()})
                print('Preparing', name, flush=True)
                cmd = [sys.executable, str(out/'code/scripts/price_pilot.py'), 'prepare',
                    '--root', str(root), '--output', str(directory), '--fold', str(fold),
                    '--date-plan', str(out/f'{name}-dates.json'), '--iterations', str(protocol['iterations'])]
                with (out/f'{name}-prepare.log').open('w') as log:
                    subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                audit = audit_window(directory, protocol)
                if not audit['passed'] or audit['evaluation_label_overlaps']:
                    raise ValueError(f'Failed structural audit: {name}')
                write_json(directory/'data-audit.json', audit)
                rows = pd.read_parquet(directory/'evaluation-rows.parquet')
                data = load_arrays(directory/'evaluation.npz')
                keys = rows[['date', 'instrument_id']]
                if common_rows is None:
                    common_rows, common_targets = keys, data['targets'].copy()
                else:
                    pd.testing.assert_frame_equal(keys, common_rows)
                    np.testing.assert_array_equal(data['targets'], common_targets)
                with (out/f'{name}-tree.log').open('w') as log:
                    subprocess.run([sys.executable, str(directory/'code/price_pilot.py'), 'tree',
                        '--output', str(directory)], env={**env, 'PYTHONPATH': str(directory/'code')},
                        stdout=log, stderr=subprocess.STDOUT, check=True)
                stage = read(directory/'tree-status.json')
                if stage['status'] != 'completed':
                    raise ValueError('Incomplete tree fit')
                verify_files(directory, stage['files'])
                train = load_arrays(directory/'train.npz')
                train_rows = pd.read_parquet(directory/'train-rows.parquet')
                weights = 1/train_rows.groupby('date').date.transform('size').to_numpy()
                baseline = np.empty((5, 4, 3))
                for day in range(5):
                    for field in range(4):
                        baseline[day, field] = weighted_quantile(train['targets'][:, day, field], weights)
                scale = volatility_scale(train['x'], volatility_index, .0001)
                vb = fit_volatility_baseline(train['targets'], scale, train_rows.date)
                forecasts = {arm: np.load(directory/'lightgbm-evaluation.npy'),
                    arm+'_naive': np.broadcast_to(baseline, (len(rows), 5, 4, 3)),
                    arm+'_volatility': vb[None]*volatility_scale(data['x'], volatility_index, .0001)[:, None, None, None]}
                np.save(directory/'date-weighted-naive.npy', baseline)
                np.save(directory/'date-weighted-volatility.npy', vb)
                daily = daily_metrics(rows, data['targets'], forecasts, window)
                daily.to_csv(directory/'daily-metrics.csv', index=False)
                metrics.append(daily)
                diag, drift = prediction_diagnostics(rows, data['targets'], forecasts[arm],
                    train['x'], data['x'], names)
                diag['window'], diag['model'] = window, arm
                diag.to_csv(directory/'prediction-diagnostics.csv', index=False)
                drift.to_csv(directory/'feature-drift.csv', index=False)
                diagnostics.append(diag)
                audits[name] = audit
                write_json(directory/'completed-manifest.json', {str(f.relative_to(directory)): file_hash(f)
                    for f in directory.rglob('*') if f.is_file() and '__pycache__' not in f.parts})
                print('Completed', name, 'train rows', len(train_rows), 'eval rows', len(rows), flush=True)
        daily = pd.concat(metrics, ignore_index=True)
        daily.to_csv(out/'daily-metrics.csv', index=False)
        pd.concat(diagnostics, ignore_index=True).to_csv(out/'prediction-diagnostics.csv', index=False)
        summary = daily.groupby(['window', 'model'])[
            ['pinball', 'median_mae', 'all_ohlc_pinball', 'coverage80', 'width80']].mean()
        summary.to_csv(out/'window-metrics.csv')
        comparisons = {f'{a}_vs_{b}': paired_comparison(daily, a, b)
            for a, b in [('recent24', 'legacy24'), ('recent96', 'recent24')]
            + [(a, a+s) for a in protocol['arms'] for s in ['_naive', '_volatility']]}
        write_json(out/'assessment.json', {'comparisons': comparisons, 'data_audits': audits,
            'daily_strategy_ready': False, 'quality_promotion': False,
            'uncertainty': 'Historical development windows. Two-date block intervals conditional on chosen windows; no significance or profit claim.',
            'next_stage': 'preregister plan-conditional net-return development experiment; existing G2 status unchanged'})
        files = {str(f.relative_to(out)): file_hash(f) for f in out.rglob('*')
                 if f.is_file() and f.name not in ['run-status.json', 'progress.json'] and '__pycache__' not in f.parts}
        write_json(marker, {'status': 'completed', 'finished_at': utc_now(), 'files': files})
        write_json(out/'progress.json', {'status': 'completed', 'completed_fits': len(audits), 'updated_at': utc_now()})
    except BaseException as exc:
        write_json(marker, {'status': 'failed', 'error': repr(exc), 'updated_at': utc_now()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'run'])
    parser.add_argument('--root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path)
    args = parser.parse_args()
    globals()[args.stage](args)


if __name__ == '__main__':
    main()
