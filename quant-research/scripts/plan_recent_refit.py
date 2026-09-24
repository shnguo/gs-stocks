"""Refit selected tree budgets on 96 dates including newly mature selection labels."""
import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from plan_risk_decision import (
    check,
    choose,
    contrasts,
    daily_metrics,
    load,
    outcomes,
    paired_metrics,
    read,
)
from plan_value_run import decision_metrics

from quant_research.plan_targets import plan_targets
from quant_research.plan_value import (
    HEADS,
    PROBABILITIES,
    compare_heads,
    date_weights,
    head_metrics,
    head_targets,
    predict_heads,
)
from quant_research.price_strategy import GRID, TradeAssumptions
from quant_research.storage import file_hash, utc_now, write_json


def plan_dates(spec):
    old, recent = spec['train'], spec['selection']
    if len(old) != 96 or len(recent) != 4 or len(set(old+recent)) != 100:
        raise ValueError('Expected 96 disjoint training and four selection dates')
    if old != sorted(old) or recent != sorted(recent) or old[-1] >= recent[0]:
        raise ValueError('Invalid source chronology')
    if spec['selection_label_end'] > spec['asof_signal_close']:
        raise ValueError('Selection labels not mature at prediction time')
    return old[4:]+recent


def frozen_context(out):
    check(out, read(out/'frozen-manifest.json'))
    e = read(out/'experiment.json')
    parent = Path(e['parent'])
    if file_hash(parent/'run-status.json') != e['parent_status_sha256']:
        raise ValueError('Forecast parent changed')
    if file_hash(parent/'verification.json') != e['parent_verification_sha256']:
        raise ValueError('Forecast parent verification changed')
    return e, parent, read(parent/'run-status.json')['files']


def cached(parent, manifest, day, asof, labels):
    directory = parent/'day-cache'/day
    prefix = str(directory.relative_to(parent))+'/'
    inp = read(directory/'input-manifest.json')
    if inp['date'] != day or day > asof or inp['first_requested_asof'] > asof:
        raise ValueError('Future input cache')
    anchors = ['input-manifest.json']
    if labels:
        label = read(directory/'label-manifest.json')
        # Check logical availability before opening any target arrays.
        if label['date'] != day or label['label_end'] > asof or label['first_requested_asof'] > asof:
            raise ValueError('Immature labels rejected before array access')
        if label['input_manifest_sha256'] != file_hash(directory/'input-manifest.json'):
            raise ValueError('Label input lineage differs')
        anchors.append('label-manifest.json')
    receipt = {prefix+n: manifest[prefix+n] for n in anchors}
    check(parent, receipt)
    check(directory, inp['files'])
    rows = pd.read_parquet(directory/'rows.parquet')
    data = load(directory/'inputs.npz')
    if labels:
        check(directory, label['files'])
        targets = load(directory/'labels.npz')
        np.testing.assert_array_equal(data['reference'], targets['reference'])
        data.update(targets)
    return rows, data, receipt


def freeze(args):
    parent, out = args.parent.resolve(), args.output.resolve()
    state = read(parent/'run-status.json')
    if state['status'] != 'completed' or not read(parent/'verification.json')['passed']:
        raise ValueError('Completed verified rolling parent required')
    check(parent, state['files'])
    schedule, training = read(parent/'schedule.json'), read(parent/'protocol.json')['training']
    new_schedule, count = {}, 0
    budgets = {h: [] for h in HEADS}
    for window, info in schedule.items():
        new_schedule[window] = {}
        for item in info['predictions']:
            day = item['date']
            spec = info['fits'][day]
            selected = plan_dates(spec)
            source = parent/window/'fits'/day
            meta = read(source/'models.json')
            for head, values in meta['heads'].items():
                budgets[head].extend(v.get('best_iteration', 0) for v in values)
            new_schedule[window][day] = dict(asof=day, train=selected, removed=spec['train'][:4],
                added=spec['selection'], latest_label_end=spec['selection_label_end'],
                forecast_label_end=spec['forecast_label_end'], source_models_sha256=file_hash(source/'models.json'),
                source_completion_sha256=file_hash(source/'raw-completed.json'))
            count += 1
    assert count == 48
    out.mkdir()
    write_json(out/'experiment.json', dict(parent=str(parent), parent_status_sha256=file_hash(parent/'run-status.json'),
        parent_verification_sha256=file_hash(parent/'verification.json'), prior=read(parent/'experiment.json')['prior'],
        registered_at=utc_now(), total_fits=count))
    write_json(out/'protocol.json', dict(protocol_id='plan-recent-refit-v1', training=training,
        hypothesis='After historical model selection, refit on 96 mature dates by replacing four oldest training dates with the four recent selection dates',
        iterations='Each head uses its already selected parent best_iteration, no additional early stopping or iteration search; constants use one tree if new labels have variation',
        separation='Selection labels may enter final fit only after that asof budget is fixed; all labels must end at or before signal close. The prediction horizon is strictly later.',
        comparison='Same 48 stock-date cohorts, 27 features, labels, seed, tree options, fixed plan choices and costs; no new capacity or calibration. Date count remains96; replacing dates changes recency/composition together.',
        forecasts='Raw only; existing parent calibration cannot be reused after model refit. A positive result needs separately registered historical prediction calibration.',
        interpretation='Exposed development experiment, not a G2/G3 pass or actual fills; no daily-model promotion',
        primary='Paired conditional net-return MSE vs raw rolling parent; all five heads and original/risk decisions retained',
        sealed_holdout_start=training['sealed_holdout_start'], quality_promotion=False))
    write_json(out/'schedule.json', new_schedule)
    write_json(out/'selected-budget-summary.json', {h: dict(models=len(v), at_cap=sum(x == training['iterations'] for x in v),
        at_one=sum(x == 1 for x in v), median=float(np.median(v))) for h, v in budgets.items()})
    base = Path(__file__).resolve().parents[1]
    shutil.copytree(base/'src/quant_research', out/'code/src/quant_research', ignore=shutil.ignore_patterns('__pycache__'))
    (out/'code/scripts').mkdir()
    for name in ['plan_recent_refit.py', 'plan_risk_decision.py', 'plan_value_run.py', 'price_pilot.py', 'model_quality_run.py']:
        shutil.copy2(base/'scripts'/name, out/'code/scripts'/name)
    shutil.copy2(base/'uv.lock', out/'code/uv.lock')
    write_json(out/'frozen-manifest.json', {str(p.relative_to(out)): file_hash(p) for p in out.rglob('*') if p.is_file()})
    print('Frozen recent-label refit', count, 'asof tasks', flush=True)


def fit(args):
    out = args.output.resolve()
    e, parent, manifest = frozen_context(out)
    protocol, schedule = read(out/'protocol.json'), read(out/'schedule.json')
    spec = schedule[args.window][args.asof]
    dest = out/args.window/'fits'/args.asof
    dest.mkdir(parents=True)
    source = parent/args.window/'fits'/args.asof
    assert file_hash(source/'models.json') == spec['source_models_sha256']
    assert file_hash(source/'raw-completed.json') == spec['source_completion_sha256']
    check(source, read(source/'raw-completed.json')['files'])
    budgets = read(source/'models.json')
    write_json(dest/'config.json', dict(window=args.window, asof=args.asof, schedule=spec,
        started_at=utc_now(), frozen_experiment_sha256=file_hash(out/'frozen-manifest.json')))
    parts, row_parts, receipts = [], [], {}
    for day in spec['train']:
        rows, data, receipt = cached(parent, manifest, day, args.asof, True)
        row_parts.append(rows)
        parts.append(data)
        receipts.update(receipt)
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    rows = pd.concat(row_parts, ignore_index=True)
    del parts, row_parts
    targets = head_targets(plan_targets(data))
    params = protocol['training']['tree_parameters']
    directory = dest/'models'
    directory.mkdir()
    metadata = {'heads': {}, 'scope': 'Raw final refit using only historically mature labels'}
    for head in HEADS:
        metadata['heads'][head] = []
        for j, selected in enumerate(budgets['heads'][head]):
            y = targets[head][:, j]
            known = np.isfinite(y)
            if known.sum() < 100:
                raise ValueError('Insufficient mature targets')
            weights = date_weights(rows.date, known)
            iterations = selected.get('best_iteration', 1)
            state = dict(plan=j, training_known=int(known.sum()), selected_iterations=iterations,
                parent_kind=selected['kind'], parent_best_iteration=selected.get('best_iteration'))
            if np.unique(y[known]).size == 1:
                state.update(kind='constant', value=float(np.average(y[known], weights=weights)))
            else:
                options = dict(n_estimators=iterations, learning_rate=params['learning_rate'], num_leaves=params['num_leaves'],
                    min_child_samples=params['min_child_samples'], reg_lambda=params['reg_lambda'], n_jobs=params['n_jobs'],
                    random_state=protocol['training']['seed'], verbosity=-1, deterministic=True, force_col_wise=True)
                model = lgb.LGBMClassifier(objective='binary', **options) if head in PROBABILITIES else lgb.LGBMRegressor(objective='regression', **options)
                model.fit(data['x'][known], y[known], sample_weight=weights)
                filename = f'{head}-plan{j:02d}.txt'
                model.booster_.save_model(str(directory/filename))
                state.update(kind='booster', file=filename, actual_iterations=model.booster_.current_iteration())
            metadata['heads'][head].append(state)
        print('Refitted head', head, args.window, args.asof, flush=True)
    write_json(dest/'models.json', metadata)
    del data, targets, rows
    rows, data, receipt = cached(parent, manifest, args.asof, args.asof, False)
    receipts.update(receipt)
    predicted = predict_heads(data['x'], directory, metadata)
    rows.to_parquet(dest/'prediction-rows.parquet', index=False)
    np.savez_compressed(dest/'raw.npz', **predicted)
    np.savez_compressed(dest/'choices.npz', original=choose(predicted), risk=choose(predicted, True))
    write_json(dest/'data-receipts.json', dict(asof=args.asof, cache_files=receipts,
        latest_training_label=spec['latest_label_end'], training_date_count=96,
        prediction_input_only=True, parent_models_sha256=spec['source_models_sha256']))
    write_json(dest/'completed.json', dict(status='completed', completed_at=utc_now(), files={
        str(p.relative_to(dest)): file_hash(p) for p in dest.rglob('*') if p.is_file()}))


def evaluate(out, e, parent, schedule):
    reports, decisions, risk_reports, risk_pairs = [], [], [], []
    prior = Path(e['prior'])
    for window, info in schedule.items():
        source = prior/window
        rows = pd.read_parquet(source/'evaluation-rows.parquet')
        merged = {h: [] for h in HEADS}
        for day in info:
            directory = out/window/'fits'/day
            check(directory, read(directory/'completed.json')['files'])
            actual = pd.read_parquet(directory/'prediction-rows.parquet')
            expected = rows[rows.date.eq(day)].reset_index(drop=True)
            pd.testing.assert_frame_equal(actual, expected[actual.columns])
            for h, values in load(directory/'raw.npz').items():
                merged[h].append(values)
        prediction = {h: np.concatenate(v) for h, v in merged.items()}
        np.savez_compressed(out/window/'recent_raw-evaluation.npz', **prediction)
        original = load(parent/window/'rolling_raw-evaluation.npz')
        forecast = dict(recent_raw=prediction, rolling_raw=original)
        choices = dict(fixed=np.full(len(rows), GRID.index((-.01, .03, .02))), cash=np.full(len(rows), -1))
        for name, pred in forecast.items():
            choices[name+'/original'] = choose(pred)
            choices[name+'/risk'] = choose(pred, True)
        for rule in ['original', 'risk']:
            saved = np.concatenate([load(out/window/'fits'/day/'choices.npz')[rule] for day in info])
            np.testing.assert_array_equal(saved, choices['recent_raw/'+rule])
        np.savez_compressed(out/window/'policy-choices.npz', **choices)
        data = load(source/'evaluation.npz')
        a = TradeAssumptions()
        base = plan_targets(data, a)
        stress = plan_targets(data, replace(a, commission=a.commission*2, minimum_fee=a.minimum_fee*2,
            sell_tax=a.sell_tax*2, slippage_bps=a.slippage_bps*2))
        values = {name: outcomes(chosen, base, stress) for name, chosen in choices.items()}
        for name, value in values.items():
            risk_reports.append(daily_metrics(rows, value, window, name))
        pairs = [('recent_raw/'+rule, ref) for rule in ['original', 'risk']
                 for ref in ['rolling_raw/'+rule, 'fixed', 'cash']]
        pairs.append(('recent_raw/risk', 'recent_raw/original'))
        for candidate, baseline in pairs:
            risk_pairs.append(paired_metrics(rows, values[candidate], values[baseline], window, candidate+'_vs_'+baseline))
        report = head_metrics(rows, plan_targets(data), forecast, window)
        report.to_csv(out/window/'head-metrics.csv', index=False)
        reports.append(report)
        for name, values in forecast.items():
            daily, chosen = decision_metrics(rows, values, data, window, name)
            daily.to_csv(out/window/f'{name}-decision-metrics.csv', index=False)
            chosen.to_parquet(out/window/f'{name}-chosen.parquet', index=False)
            decisions.append(daily)
    combined = pd.concat(reports, ignore_index=True)
    combined.to_csv(out/'head-metrics.csv', index=False)
    pd.concat(decisions, ignore_index=True).to_csv(out/'decision-metrics.csv', index=False)
    pair = combined.replace({'model': {'recent_raw': 'learned', 'rolling_raw': 'empirical'}})
    write_json(out/'assessment.json', dict(comparisons={'recent_raw_vs_rolling_raw': compare_heads(pair)},
        quality_promotion=False, executable=False))
    pd.concat(risk_reports, ignore_index=True).to_csv(out/'policy-daily.csv', index=False)
    paired = pd.concat(risk_pairs, ignore_index=True)
    paired.to_csv(out/'policy-paired.csv', index=False)
    write_json(out/'policy-contrasts.json', contrasts(paired))


def run(args):
    out = args.output.resolve()
    e, parent, manifest = frozen_context(out)
    schedule = read(out/'schedule.json')
    marker = out/'run-status.json'
    if marker.exists():
        raise FileExistsError('Preserve previous attempts')
    check(parent, manifest)
    write_json(marker, dict(status='running', pid=os.getpid(), started_at=utc_now()))
    try:
        completed = 0
        for window, info in schedule.items():
            for day in info:
                directory = out/window/'fits'/day
                write_json(out/'progress.json', dict(status='fitting', window=window, asof=day,
                    completed_fits=completed, total_fits=48, updated_at=utc_now()))
                if (directory/'completed.json').exists():
                    check(directory, read(directory/'completed.json')['files'])
                    assert read(directory/'config.json')['frozen_experiment_sha256'] == file_hash(out/'frozen-manifest.json')
                else:
                    with (out/f'{window}-{day}.log').open('x') as log:
                        subprocess.run([sys.executable, str(out/'code/scripts/plan_recent_refit.py'), 'fit', '--output', str(out),
                            '--window', window, '--asof', day], stdout=log, stderr=subprocess.STDOUT, check=True,
                            env={**os.environ, 'PYTHONPATH': str(out/'code/src')} )
                completed += 1
        write_json(out/'forecast-stage-completed.json', dict(completed_at=utc_now(), fits=completed,
            files={str(p.relative_to(out)): file_hash(p) for p in out.glob('fold-*/fits/*/completed.json')}))
        write_json(out/'progress.json', dict(status='evaluating', completed_fits=completed, total_fits=48, updated_at=utc_now()))
        evaluate(out, e, parent, schedule)
        check(out, read(out/'frozen-manifest.json'))
        write_json(marker, dict(status='completed', finished_at=utc_now(), files={
            str(p.relative_to(out)): file_hash(p) for p in out.rglob('*') if p.is_file()
            and p.name not in ['run-status.json', 'progress.json'] and '__pycache__' not in p.parts}))
        write_json(out/'progress.json', dict(status='completed', completed_fits=48, total_fits=48, updated_at=utc_now()))
    except BaseException as exc:
        write_json(marker, dict(status='failed', error=repr(exc), updated_at=utc_now()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'fit', 'run'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--window')
    parser.add_argument('--asof')
    args = parser.parse_args()
    {'freeze': freeze, 'fit': fit, 'run': run}[args.stage](args)


if __name__ == '__main__':
    main()
