"""Compare native categorical context and stock identity on frozen plan cohorts."""
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

from quant_research.category_context import encode_categories, fit_dictionary, raw_categories
from quant_research.industry_context import read_snapshot
from quant_research.plan_value import (
    apply_calibration,
    calibration_bins,
    compare_heads,
    fit_calibration,
    fit_heads,
    head_metrics,
    predict_heads,
)
from quant_research.storage import file_hash, utc_now, write_json


def read(path):
    return json.loads(path.read_text())


def freeze(args):
    prior, out = args.prior.resolve(), args.output.resolve()
    state = read(prior/'run-status.json')
    if state['status'] != 'completed' or not read(prior/'verification.json')['passed']:
        raise ValueError('Parent experiment is not complete and verified')
    verify_files(prior, state['files'])
    base = Path(__file__).resolve().parents[1]
    industry = args.industry.resolve()
    industry_state = read(industry/'run-status.json')
    if industry_state['status'] != 'completed' or not read(industry/'verification.json')['passed']:
        raise ValueError('Industry source experiment is not verified')
    verify_files(industry, industry_state['files'])
    evidence = read(industry/'industry-snapshots.json')
    required = set()
    for fold in read(prior/'protocol.json')['fold_indices']:
        for dates in read(prior/f'fold-{fold:02d}'/'config.json')['dates'].values():
            required.update(dates)
    if set(evidence) != required:
        raise ValueError('Industry dates differ from category experiment')
    for day, item in evidence.items():
        age = (pd.Timestamp(day)-pd.Timestamp(item['source_date'])).days
        if not 0 <= age <= 6:
            raise ValueError('Invalid historical industry mapping')
        read_snapshot(item['directory'], item['source_date'])
    out.mkdir()
    write_json(out/'industry-snapshots.json', evidence)
    write_json(out/'protocol.json', read(args.protocol))
    write_json(out/'experiment.json', {'created_at': utc_now(), 'prior': str(prior),
        'parent_status_sha256': file_hash(prior/'run-status.json'),
        'training_protocol': read(prior/'protocol.json'), 'industry_source': str(industry),
        'industry_status_sha256': file_hash(industry/'run-status.json'), 'executable': False})
    shutil.copytree(base/'src/quant_research', out/'code/src/quant_research', ignore=shutil.ignore_patterns('__pycache__'))
    (out/'code/scripts').mkdir()
    for name in ['category_ablation_run.py', 'verify_category_ablation.py', 'plan_value_run.py', 'price_pilot.py', 'model_quality_run.py', 'verify_industry_decisions.py', 'verify_plan_prequential.py', 'verify_plan_asof_fit.py', 'verify_plan_risk_decision.py']:
        shutil.copy2(base/'scripts'/name, out/'code/scripts'/name)
    shutil.copy2(base/'uv.lock', out/'code/uv.lock')
    write_json(out/'frozen-manifest.json', {str(f.relative_to(out)): file_hash(f) for f in out.rglob('*') if f.is_file()})
    print('Frozen', out, flush=True)


def fit(args):
    out = args.output.resolve()
    verify_files(out, read(out/'frozen-manifest.json'))
    experiment, protocol = read(out/'experiment.json'), read(out/'protocol.json')
    prior = Path(experiment['prior'])/f'fold-{args.fold:02d}'
    dest = out/f'fold-{args.fold:02d}-{args.arm}'
    dest.mkdir()
    names = read(Path(read(prior/'config.json')['root'])/'panel-h5/manifest.json')['feature_names']
    columns = protocol['arms'][args.arm]
    write_json(dest/'features.json', {'base': names, 'added': columns,
        'categorical_indices': list(range(len(names), len(names)+len(columns))),
        'source': 'retrospective industry; identifier-derived exchange/code families, not certified historical board membership'})
    index = read(out/'industry-snapshots.json')
    dictionary = None
    train_ids = None
    def load(part):
        nonlocal dictionary, train_ids
        data = load_arrays(prior/f'{part}.npz')
        rows = pd.read_parquet(prior/f'{part}-rows.parquet')
        snapshots = {}
        for date in sorted(rows.date.unique()):
            entry = index[date]
            directory = Path(entry['directory'])
            if file_hash(directory/'manifest.json') != entry['manifest_sha256']:
                raise ValueError('Industry snapshot manifest changed')
            snapshots[date] = read_snapshot(directory, entry['source_date'])
        categories = raw_categories(rows, snapshots, protocol['max_snapshot_age_days'])
        if part == 'train':
            dictionary = fit_dictionary(categories[columns])
            train_ids = set(rows.instrument_id)
            write_json(dest/'dictionary.json', dictionary)
        if dictionary is None or train_ids is None:
            raise ValueError('Training dictionary must precede other partitions')
        context = encode_categories(categories[columns], dictionary)
        seen = rows.instrument_id.isin(train_ids).to_numpy()
        write_json(dest/f'{part}-category-coverage.json', dict(total_rows=len(rows),
            missing_or_unseen={k: int((context[:, i] == -1).sum()) for i,k in enumerate(columns)},
            seen_stock_rows=int(seen.sum()), unseen_stock_rows=int((~seen).sum())))
        np.save(dest/f'{part}-context.npy', context)
        pd.DataFrame({'seen_stock': seen, 'identity_exchange': categories.identity_exchange}).to_parquet(
            dest/f'{part}-cohorts.parquet', index=False)
        data['x'] = np.column_stack([data['x'], context])
        return data, rows
    train, tr = load('train')
    selection, sr = load('selection')
    training_protocol = {**experiment['training_protocol'],
                         'categorical_parameters': protocol['categorical_parameters']}
    metadata = fit_heads(train['x'], tr.date, compact_targets(train), selection['x'], sr.date,
                         compact_targets(selection), dest/'plan-models', training_protocol,
                         categorical_features=list(range(len(names), len(names)+len(columns))))
    write_json(dest/'models.json', metadata)
    del train, selection
    cal, cr = load('calibration')
    calibration = fit_calibration(predict_heads(cal['x'], dest/'plan-models', metadata), compact_targets(cal), cr.date)
    write_json(dest/'calibration.json', calibration)
    del cal
    evaluation, rows = load('evaluation')
    raw = predict_heads(evaluation['x'], dest/'plan-models', metadata)
    calibrated = apply_calibration(raw, calibration)
    forecasts = {args.arm+'_raw': raw, args.arm: calibrated}
    for model, predictions in forecasts.items():
        np.savez_compressed(dest/f'{model}-evaluation.npz', **predictions)
    window = f'fold-{args.fold:02d}'
    head_metrics(rows, compact_targets(evaluation), forecasts, window).to_csv(dest/'head-metrics.csv', index=False)
    calibration_bins(rows, compact_targets(evaluation), forecasts, window).to_csv(dest/'calibration-bins.csv', index=False)
    for model, predictions in forecasts.items():
        metrics, chosen = decision_metrics(rows, predictions, evaluation, window, model)
        metrics.to_csv(dest/f'{model}-decision-metrics.csv', index=False)
        chosen.to_parquet(dest/f'{model}-chosen.parquet', index=False)
    cohorts = pd.read_parquet(dest/'evaluation-cohorts.parquet')
    subgroup_metrics, subgroup_decisions = [], []
    baseline = {name: load_arrays(prior/f'{name}-evaluation.npz') for name in ['learned_raw', 'learned']}
    for cohort, mask in [('seen_stock', cohorts.seen_stock.to_numpy()),
                         ('unseen_stock', ~cohorts.seen_stock.to_numpy()),
                         *[(f'exchange_{e}', cohorts.identity_exchange.eq(e).to_numpy())
                           for e in sorted(cohorts.identity_exchange.unique())]]:
        if not mask.any():
            continue
        subset = {k: v[mask] for k,v in evaluation.items()}
        subset_rows = rows.loc[mask].reset_index(drop=True)
        predictions = {name: {h: values[mask] for h,values in pred.items()}
                       for name,pred in {**forecasts, **baseline}.items()}
        report = head_metrics(subset_rows, compact_targets(subset), predictions, window)
        report['cohort'] = cohort
        subgroup_metrics.append(report)
        for name, pred in predictions.items():
            report, _ = decision_metrics(subset_rows, pred, subset, window, name)
            report['cohort'] = cohort
            subgroup_decisions.append(report)
    pd.concat(subgroup_metrics, ignore_index=True).to_csv(dest/'subgroup-head-metrics.csv', index=False)
    pd.concat(subgroup_decisions, ignore_index=True).to_csv(dest/'subgroup-decision-metrics.csv', index=False)
    write_json(dest/'completed.json', {'status': 'completed', 'updated_at': utc_now(), 'files': {
        str(f.relative_to(dest)): file_hash(f) for f in dest.rglob('*') if f.is_file()}})


def run(args):
    out = args.output.resolve()
    verify_files(out, read(out/'frozen-manifest.json'))
    experiment, protocol = read(out/'experiment.json'), read(out/'protocol.json')
    prior = Path(experiment['prior'])
    if file_hash(prior/'run-status.json') != experiment['parent_status_sha256']:
        raise ValueError('Parent status changed')
    verify_files(prior, read(prior/'run-status.json')['files'])
    marker = out/'run-status.json'
    if marker.exists():
        raise FileExistsError('Preserve previous attempts')
    write_json(marker, {'status': 'running', 'pid': os.getpid(), 'started_at': utc_now()})
    metrics, bins, decisions, completed = [], [], [], 0
    try:
        for fold in experiment['training_protocol']['fold_indices']:
            for arm in protocol['arms']:
                name = f'fold-{fold:02d}-{arm}'
                write_json(out/'progress.json', {'status': 'running', 'current': name,
                    'completed_fits': completed, 'total_fits': len(experiment['training_protocol']['fold_indices'])*len(protocol['arms']), 'updated_at': utc_now()})
                print('Starting', name, flush=True)
                with (out/f'{name}.log').open('w') as log:
                    subprocess.run([sys.executable, str(out/'code/scripts/category_ablation_run.py'),
                        'fit', '--output', str(out), '--fold', str(fold), '--arm', arm],
                        env={**os.environ, 'PYTHONPATH': str(out/'code/src')}, stdout=log, stderr=subprocess.STDOUT, check=True)
                dest = out/name
                state = read(dest/'completed.json')
                if state['status'] != 'completed':
                    raise ValueError('Incomplete arm')
                verify_files(dest, state['files'])
                metrics.append(pd.read_csv(dest/'head-metrics.csv'))
                bins.append(pd.read_csv(dest/'calibration-bins.csv'))
                decisions.extend(pd.read_csv(dest/f'{m}-decision-metrics.csv') for m in [arm, arm+'_raw'])
                completed += 1
                print('Completed', name, flush=True)
        baseline = pd.read_csv(prior/'head-metrics.csv')
        baseline = baseline.loc[baseline.model.isin(['learned', 'learned_raw'])].replace(
            {'model': {'learned': 'baseline27', 'learned_raw': 'baseline27_raw'}})
        all_metrics = pd.concat([baseline, *metrics], ignore_index=True)
        all_metrics.to_csv(out/'head-metrics.csv', index=False)
        pd.concat(bins, ignore_index=True).to_csv(out/'calibration-bins.csv', index=False)
        pd.concat(decisions, ignore_index=True).to_csv(out/'decision-metrics.csv', index=False)
        comparisons = {}
        for arm in protocol['arms']:
            for suffix in ['', '_raw']:
                candidate, reference = arm+suffix, 'baseline27'+suffix
                paired = all_metrics.loc[all_metrics.model.isin([candidate, reference])].replace(
                    {'model': {candidate: 'learned', reference: 'empirical'}})
                comparisons[candidate+'_vs_'+reference] = compare_heads(paired)
        write_json(out/'assessment.json', {'comparisons': comparisons,
            'reference': 'frozen 27-feature baseline, same raw/calibration variant; all exposed development windows',
            'primary': 'context_categories_raw_vs_baseline27_raw',
            'stock_identity_scope': 'naturally unseen stock cohort; not a randomized held-out-stock test',
            'executable': False, 'quality_promotion': False})
        write_json(marker, {'status': 'completed', 'finished_at': utc_now(), 'files': {
            str(f.relative_to(out)): file_hash(f) for f in out.rglob('*') if f.is_file()
            and f.name not in ['run-status.json', 'progress.json'] and '__pycache__' not in f.parts}})
        write_json(out/'progress.json', {'status': 'completed', 'completed_fits': completed, 'updated_at': utc_now()})
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
    parser.add_argument('--arm')
    parser.add_argument('--industry', type=Path)
    args = parser.parse_args()
    globals()[args.stage](args)


if __name__ == '__main__':
    main()
