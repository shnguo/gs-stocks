"""Frozen fixed-plan conditional-return experiment; no recommendation promotion."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

# Resolve the matching frozen package when invoked from an artifact snapshot.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

import numpy as np
import pandas as pd
from model_quality_run import audit_window
from price_pilot import load_arrays, verify_files

from quant_research.plan_targets import plan_targets
from quant_research.plan_value import (
    HEADS,
    apply_calibration,
    baseline_heads,
    calibration_bins,
    choose_research_plan,
    compare_heads,
    fit_calibration,
    fit_heads,
    head_metrics,
    make_date_plan,
    predict_heads,
)
from quant_research.price_strategy import TradeAssumptions
from quant_research.storage import file_hash, utc_now, write_json


def read(path):
    return json.loads(path.read_text())


def compact_targets(labels):
    all_targets = plan_targets(labels)
    return {k: all_targets[k] for k in ['filled', 'conditional_net_return',
                                       'conditional_loss', 'conditional_downside']}


def freeze(args):
    root, out = args.root.resolve(), args.output.resolve()
    protocol, schedule = read(args.protocol), read(root/'schedule.json')
    if protocol['sealed_holdout_start'] != schedule['sealed_holdout_start']:
        raise ValueError('Holdout boundary changed')
    dates = read(root/'panel-h5/manifest.json')['dates']
    plans = {i: make_date_plan(dates, schedule['development'][i], protocol)
             for i in protocol['fold_indices']}
    out.mkdir()
    write_json(out/'protocol.json', protocol)
    write_json(out/'experiment.json', {'created_at': utc_now(), 'root': str(root),
        'source_files': {n: file_hash(root/n) for n in ['schedule.json', 'development-acceptance.json']},
        'executable': False, 'quality_promotion': False})
    for i, plan in plans.items():
        write_json(out/f'fold-{i:02d}-dates.json', plan)
    base = Path(__file__).resolve().parents[1]
    shutil.copytree(base/'src/quant_research', out/'code/src/quant_research',
                    ignore=shutil.ignore_patterns('__pycache__'))
    (out/'code/scripts').mkdir()
    for name in ['plan_value_run.py', 'price_pilot.py', 'model_quality_run.py']:
        shutil.copy2(base/'scripts'/name, out/'code/scripts'/name)
    shutil.copy2(base/'uv.lock', out/'code/uv.lock')
    write_json(out/'frozen-manifest.json', {str(f.relative_to(out)): file_hash(f)
        for f in out.rglob('*') if f.is_file()})
    print('Frozen', out, flush=True)


def decision_metrics(rows, prediction, labels, window, model):
    chosen = choose_research_plan(prediction)
    base = plan_targets(labels)
    a = TradeAssumptions()
    stress = plan_targets(labels, replace(a, commission=a.commission*2,
        minimum_fee=a.minimum_fee*2, sell_tax=a.sell_tax*2, slippage_bps=a.slippage_bps*2))
    records = []
    for date in sorted(rows.date.unique()):
        ids = np.flatnonzero(rows.date.eq(date).to_numpy() & (chosen >= 0))
        ix = (ids, chosen[ids])
        net, stressed, filled = base['scenario_order_net_return'][ix], stress['scenario_order_net_return'][ix], base['filled'][ix]
        records.append(dict(window=window, model=model, date=date,
            candidate_stock_rows=int(rows.date.eq(date).sum()), selected=len(ids),
            fill_unknown=int(np.isnan(filled).sum()), filled=int((filled == 1).sum()),
            return_unknown=int(np.isnan(net).sum()), ambiguous=int(base['ambiguous'][ix].sum()),
            known_selected_scenario_mean=float(np.nanmean(net)) if np.isfinite(net).any() else np.nan,
            stress_known_selected_scenario_mean=float(np.nanmean(stressed)) if np.isfinite(stressed).any() else np.nan,
            stress_return_unknown=int(np.isnan(stressed).sum())))
    selected = rows[['date', 'instrument_id']].copy()
    selected['plan_index'] = chosen
    for head in HEADS:
        selected[head] = np.where(chosen >= 0, prediction[head][np.arange(len(rows)), np.maximum(chosen, 0)], np.nan)
    selected['executable'] = False
    return pd.DataFrame(records), selected


def run(args):
    out = args.output.resolve()
    verify_files(out, read(out/'frozen-manifest.json'))
    experiment, protocol = read(out/'experiment.json'), read(out/'protocol.json')
    root = Path(experiment['root'])
    verify_files(root, experiment['source_files'])
    marker = out/'run-status.json'
    if marker.exists():
        raise FileExistsError('Run already attempted; preserve it and use a new version')
    write_json(marker, {'status': 'running', 'pid': os.getpid(), 'started_at': utc_now()})
    env = {**os.environ, 'PYTHONPATH': str(out/'code/src')}
    audits, metrics, decisions, bins = {}, [], [], []
    try:
        for fold in protocol['fold_indices']:
            name, directory = f'fold-{fold:02d}', out/f'fold-{fold:02d}'
            write_json(out/'progress.json', {'status': 'running', 'current': name,
                'completed_windows': len(audits), 'total_windows': len(protocol['fold_indices']), 'updated_at': utc_now()})
            print('Preparing', name, flush=True)
            with (out/f'{name}-prepare.log').open('w') as log:
                subprocess.run([sys.executable, str(out/'code/scripts/price_pilot.py'), 'prepare',
                    '--root', str(root), '--output', str(directory), '--fold', str(fold),
                    '--date-plan', str(out/f'{name}-dates.json')], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            audit = audit_window(directory, {'thresholds': {
                'min_known_day5_fraction': .90, 'min_exchange_known_fraction': .80}})
            if not audit['passed'] or audit['evaluation_label_overlaps']:
                raise ValueError('Data audit failed')
            write_json(directory/'data-audit.json', audit)
            train, selection = [load_arrays(directory/f'{p}.npz') for p in ['train', 'selection']]
            train_rows, select_rows = [pd.read_parquet(directory/f'{p}-rows.parquet') for p in ['train', 'selection']]
            metadata = fit_heads(train['x'], train_rows.date, compact_targets(train),
                selection['x'], select_rows.date, compact_targets(selection), directory/'plan-models', protocol)
            write_json(directory/'models.json', metadata)
            del train, selection
            cal = load_arrays(directory/'calibration.npz')
            cal_rows = pd.read_parquet(directory/'calibration-rows.parquet')
            calibrated = {}
            for model, raw in [('learned', predict_heads(cal['x'], directory/'plan-models', metadata)),
                               ('empirical', baseline_heads(len(cal_rows), metadata))]:
                calibrated[model] = fit_calibration(raw, compact_targets(cal), cal_rows.date)
            write_json(directory/'calibration.json', calibrated)
            del cal
            evaluation = load_arrays(directory/'evaluation.npz')
            rows = pd.read_parquet(directory/'evaluation-rows.parquet')
            raw = {'learned': predict_heads(evaluation['x'], directory/'plan-models', metadata),
                   'empirical': baseline_heads(len(rows), metadata)}
            forecasts = {**{m+'_raw': p for m, p in raw.items()},
                         **{m: apply_calibration(p, calibrated[m]) for m, p in raw.items()}}
            for model, prediction in forecasts.items():
                np.savez_compressed(directory/f'{model}-evaluation.npz', **prediction)
            result = head_metrics(rows, compact_targets(evaluation), forecasts, name)
            result.to_csv(directory/'head-metrics.csv', index=False)
            metrics.append(result)
            binned = calibration_bins(rows, compact_targets(evaluation), forecasts, name)
            binned.to_csv(directory/'calibration-bins.csv', index=False)
            bins.append(binned)
            for model in ['learned', 'empirical']:
                result, selected = decision_metrics(rows, forecasts[model], evaluation, name, model)
                selected.to_parquet(directory/f'{model}-chosen.parquet', index=False)
                result.to_csv(directory/f'{model}-decision-metrics.csv', index=False)
                decisions.append(result)
            audits[name] = audit
            write_json(directory/'completed-manifest.json', {str(f.relative_to(directory)): file_hash(f)
                for f in directory.rglob('*') if f.is_file() and '__pycache__' not in f.parts})
            print('Completed', name, flush=True)
        all_metrics = pd.concat(metrics, ignore_index=True)
        all_metrics.to_csv(out/'head-metrics.csv', index=False)
        pd.concat(bins, ignore_index=True).to_csv(out/'calibration-bins.csv', index=False)
        pd.concat(decisions, ignore_index=True).to_csv(out/'decision-metrics.csv', index=False)
        write_json(out/'assessment.json', {'data_audits': audits, 'quality_promotion': False,
            'head_comparisons': compare_heads(all_metrics),
            'executable': False, 'evaluation_status': 'development diagnostics pending independent review',
            'interpretation': 'Conditional mean is for resolved non-ambiguous fills. Selected scenario means exclude unknown returns and are not unconditional expected returns or portfolio results.'})
        write_json(marker, {'status': 'completed', 'finished_at': utc_now(), 'files': {
            str(f.relative_to(out)): file_hash(f) for f in out.rglob('*') if f.is_file()
            and f.name not in ['run-status.json', 'progress.json'] and '__pycache__' not in f.parts}})
        write_json(out/'progress.json', {'status': 'completed', 'completed_windows': len(audits), 'updated_at': utc_now()})
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
