"""Frozen risk-sensitive decision research on already verified model forecasts."""
import argparse
import json
import os
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.plan_targets import plan_targets
from quant_research.plan_value import HEADS
from quant_research.price_strategy import GRID, TradeAssumptions
from quant_research.storage import file_hash, utc_now, write_json


def read(path):
    return json.loads(path.read_text())


def load(path):
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def check(root, files):
    for name, expected in files.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root.resolve()) or file_hash(path) != expected:
            raise ValueError(f'Changed evidence: {path}')


def choose(prediction, risk=False):
    fill, resolution, mean = [np.asarray(prediction[h]) for h in HEADS[:3]]
    downside = np.asarray(prediction[HEADS[4]])
    if any(x.shape != fill.shape or not np.isfinite(x).all() for x in [fill, resolution, mean, downside]):
        raise ValueError('Complete paired prediction support is required')
    if fill.ndim != 2 or fill.shape[1] != len(GRID) or (downside < 0).any():
        raise ValueError('Invalid plan axes or downside')
    utility = mean-downside if risk else mean
    eligible = (fill >= .3) & (resolution >= .9) & (mean > 0) & (utility > 0)
    score = np.where(eligible, fill*utility, -np.inf)
    chosen = score.argmax(axis=1)
    chosen[~eligible.any(axis=1)] = -1
    return chosen


def outcomes(chosen, base, stress):
    """No action is known zero; selected unknowns never become zero."""
    if ((chosen < -1) | (chosen >= len(GRID))).any():
        raise ValueError('Invalid decision identity')
    selected = chosen >= 0
    ix = (np.arange(len(chosen)), np.maximum(chosen, 0))
    ambiguous = selected & base['ambiguous'][ix]
    net = np.where(selected, base['scenario_order_net_return'][ix], 0.)
    primary = np.where(ambiguous, np.nan, net)
    stressed = np.where(selected, stress['scenario_order_net_return'][ix], 0.)
    filled = np.where(selected, base['filled'][ix], 0.)
    return dict(selected=selected, ambiguous=ambiguous, filled=filled,
        primary=primary, stress=stressed)


def finite_mean(values):
    values = np.asarray(values, float)
    return float(values[np.isfinite(values)].mean()) if np.isfinite(values).any() else np.nan


def daily_metrics(rows, result, window, policy):
    records = []
    for day, ids in rows.groupby('date', sort=True).indices.items():
        selected = result['selected'][ids]
        net, stress = result['primary'][ids], result['stress'][ids]
        records.append(dict(window=window, date=day, policy=policy, candidates=len(ids),
            selected=int(selected.sum()), no_action=int((~selected).sum()),
            ambiguous=int(result['ambiguous'][ids].sum()),
            primary_known=int(np.isfinite(net).sum()), primary_unknown=int((~np.isfinite(net)).sum()),
            selected_primary_known=int((selected & np.isfinite(net)).sum()),
            stress_known=int(np.isfinite(stress).sum()), stress_unknown=int((~np.isfinite(stress)).sum()),
            selected_stress_known=int((selected & np.isfinite(stress)).sum()),
            filled=int((result['filled'][ids] == 1).sum()),
            primary_mean=finite_mean(net), stress_mean=finite_mean(stress),
            primary_downside=finite_mean(np.maximum(-net, 0)), stress_downside=finite_mean(np.maximum(-stress, 0))))
    return pd.DataFrame(records)


def paired_metrics(rows, candidate, baseline, window, comparison):
    records = []
    for day, ids in rows.groupby('date', sort=True).indices.items():
        for scenario in ['primary', 'stress']:
            a, b = candidate[scenario][ids], baseline[scenario][ids]
            known = np.isfinite(a) & np.isfinite(b)
            records.append(dict(window=window, date=day, comparison=comparison, scenario=scenario,
                candidates=len(ids), common_known=int(known.sum()),
                candidate_only_known=int((np.isfinite(a) & ~np.isfinite(b)).sum()),
                baseline_only_known=int((~np.isfinite(a) & np.isfinite(b)).sum()),
                both_unknown=int((~np.isfinite(a) & ~np.isfinite(b)).sum()),
                candidate_mean=finite_mean(a[known]), baseline_mean=finite_mean(b[known]),
                delta=finite_mean(a[known]-b[known]),
                candidate_downside=finite_mean(np.maximum(-a[known], 0)),
                baseline_downside=finite_mean(np.maximum(-b[known], 0))))
    return pd.DataFrame(records)


def contrasts(paired, repetitions=2000):
    result = {}
    for (comparison, scenario), frame in paired.groupby(['comparison', 'scenario'], sort=True):
        rng = np.random.default_rng(17)
        series = [part.sort_values('date').delta.to_numpy() for _, part in frame.groupby('window', sort=True)]
        means = [finite_mean(x) for x in series]
        draws = []
        for _ in range(repetitions):
            windows = []
            for values in series:
                starts = rng.integers(len(values), size=(len(values)+1)//2)
                indices = np.column_stack([starts, (starts+1) % len(values)]).ravel()[:len(values)]
                windows.append(finite_mean(values[indices]))
            draws.append(np.mean(windows))
        result[comparison+'/'+scenario] = dict(by_window=means, mean=float(np.mean(means)),
            positive_windows=int(sum(x > 0 for x in means)),
            conditional_two_date_block_95=np.quantile(draws, [.025, .975]).tolist())
    return result


def screening(daily, paired, effects, protocol):
    policy = protocol['primary_policy']
    checks = {}
    candidate = daily.loc[daily.policy.eq(policy)]
    checks['nontrivial_selection'] = bool(candidate.selected.sum()/candidate.candidates.sum() >= .01)
    checks['active_dates'] = bool((candidate.selected > 0).sum() >= 24)
    for scenario in ['primary', 'stress']:
        checks[scenario+'_selected_coverage'] = bool(all(
            part['selected_'+scenario+'_known'].sum()/max(part.selected.sum(), 1) >= .95
            for _, part in candidate.groupby('window')))
        for baseline in ['rolling_raw/original', 'fixed', 'cash']:
            comparison = policy+'_vs_'+baseline
            effect = effects[comparison+'/'+scenario]
            part = paired.loc[paired.comparison.eq(comparison) & paired.scenario.eq(scenario)]
            checks[comparison+'/'+scenario+'/positive_interval'] = bool(effect['conditional_two_date_block_95'][0] > 0)
            checks[comparison+'/'+scenario+'/stable_windows'] = bool(effect['positive_windows'] >= 4)
            checks[comparison+'/'+scenario+'/common_coverage'] = bool(all(
                p.common_known.sum()/p.candidates.sum() >= .95 for _, p in part.groupby('window')))
            if baseline == 'rolling_raw/original':
                difference = (part.candidate_downside-part.baseline_downside).groupby(part.window).mean().mean()
                checks[scenario+'/no_downside_increase'] = bool(difference <= 1e-12)
    return dict(checks=checks, passed=all(checks.values()), quality_promotion=False,
        interpretation='Prespecified development screen only; historical outcomes already exposed; does not pass G2, certify execution, or replace independent forward validation')


def freeze(args):
    out = args.output.resolve()
    parents = {}
    for name, directory in [('baseline', args.prior), ('rolling', args.rolling)]:
        root = directory.resolve()
        state = read(root/'run-status.json')
        if state['status'] != 'completed' or not read(root/'verification.json')['passed']:
            raise ValueError('Requires complete verified forecasting parents')
        check(root, state['files'])
        parents[name] = dict(root=str(root), status_sha256=file_hash(root/'run-status.json'),
            verification_sha256=file_hash(root/'verification.json'))
    assert Path(read(args.rolling/'experiment.json')['prior']).resolve() == args.prior.resolve()
    out.mkdir()
    models = ['baseline_raw', 'baseline_calibrated', 'rolling_raw', 'rolling_calibrated']
    pairs = [(m+'/risk', m+'/original') for m in models]
    pairs += [(m+'/'+rule, reference) for m in models for rule in ['original', 'risk'] for reference in ['fixed', 'cash']]
    pairs += [('fixed', 'cash')]
    write_json(out/'protocol.json', dict(protocol_id='plan-risk-decision-v1', registered_at=utc_now(),
        models=models, comparisons=pairs, primary_policy='rolling_raw/risk',
        fold_indices=read(args.prior/'protocol.json')['fold_indices'],
        sealed_holdout_start=read(args.prior/'protocol.json')['sealed_holdout_start'], grid=GRID,
        fixed_plan_index=GRID.index((-.01, .03, .02)), assumptions=asdict(TradeAssumptions()),
        original='fill>=0.3, resolution>=0.9, mean>0; maximize fill*mean; ties earliest grid plan',
        risk='Same gates, require mean-downside>0; maximize fill*(mean-downside); fixed downside penalty1, no parameter search',
        risk_interpretation='Conditional mean minus conditional expected downside is a research risk penalty, not unconditional expected return or a user account risk preference',
        unit='One same-notional hypothetical opportunity per stock-date; deliberate no action=0, selected unknown=NaN; all candidates retained; no portfolio/capital/overlap claim',
        primary_outcome='Net daily scenario excluding target-stop ambiguity; unknown remains missing',
        stress_outcome='Double all original cost components and retain conservative daily-bar ambiguity scenarios on unchanged choices',
        pairing='Same stock-date common finite outcomes; report all excluded and one-sided known counts; common-mask results do not identify missing outcomes',
        aggregation='Equal known dates per window, then six windows equal; 2000 circular two-date blocks; absent windows remain unknown',
        screen=dict(minimum_selected_fraction=.01, minimum_active_dates=24, minimum_selected_known_fraction_per_window=.95,
            minimum_common_fraction_per_window=.95, minimum_positive_windows=4,
            lower_95_delta_vs_original_fixed_cash_strictly_positive=True, mean_downside_vs_original_must_not_increase=True,
            both_primary_and_stress=True, primary_arm_reason='Raw rolling designated after observed calibration harm; exposed development hypothesis, not blind selection'),
        promotion=False, future_validation_required=True))
    write_json(out/'parents.json', parents)
    base = Path(__file__).resolve().parents[1]
    shutil.copytree(base/'src/quant_research', out/'code/src/quant_research', ignore=shutil.ignore_patterns('__pycache__'))
    (out/'code/scripts').mkdir()
    shutil.copy2(Path(__file__), out/'code/scripts/plan_risk_decision.py')
    shutil.copy2(base/'uv.lock', out/'code/uv.lock')
    write_json(out/'frozen-manifest.json', {str(p.relative_to(out)): file_hash(p) for p in out.rglob('*') if p.is_file()})


def run(args):
    out = args.output.resolve()
    check(out, read(out/'frozen-manifest.json'))
    protocol, parents = read(out/'protocol.json'), read(out/'parents.json')
    roots = {k: Path(v['root']) for k, v in parents.items()}
    for name, root in roots.items():
        assert file_hash(root/'run-status.json') == parents[name]['status_sha256']
        assert file_hash(root/'verification.json') == parents[name]['verification_sha256']
        check(root, read(root/'run-status.json')['files'])
    marker = out/'run-status.json'
    if marker.exists():
        raise FileExistsError('Preserve previous attempts')
    write_json(marker, dict(status='running', pid=os.getpid(), started_at=utc_now()))
    try:
        # Save all choices across all six windows before opening any outcomes.
        for fold in protocol['fold_indices']:
            window = f'fold-{fold:02d}'
            directory = out/window
            directory.mkdir()
            rows = pd.read_parquet(roots['baseline']/window/'evaluation-rows.parquet')
            assert rows.label_end.max() < protocol['sealed_holdout_start']
            rows[['date', 'instrument_id']].to_parquet(directory/'rows.parquet', index=False)
            choices = dict(fixed=np.full(len(rows), protocol['fixed_plan_index']), cash=np.full(len(rows), -1))
            for model in protocol['models']:
                name, version = model.split('_', 1)
                prefix = 'learned' if name == 'baseline' else 'rolling'
                suffix = '_raw' if version == 'raw' else ''
                pred = load(roots[name]/window/f'{prefix}{suffix}-evaluation.npz')
                for rule in ['original', 'risk']:
                    choices[model+'/'+rule] = choose(pred, rule == 'risk')
            np.savez_compressed(directory/'choices.npz', **choices)
        write_json(out/'choices-completed.json', dict(completed_at=utc_now(), files={
            str(p.relative_to(out)): file_hash(p) for p in out.glob('fold-*/*')}))
        reports, comparisons = [], []
        for fold in protocol['fold_indices']:
            window = f'fold-{fold:02d}'
            directory = out/window
            rows = pd.read_parquet(directory/'rows.parquet')
            data = load(roots['baseline']/window/'evaluation.npz')
            a = TradeAssumptions(**protocol['assumptions'])
            base = plan_targets(data, a)
            stress = plan_targets(data, replace(a, commission=a.commission*2, minimum_fee=a.minimum_fee*2,
                sell_tax=a.sell_tax*2, slippage_bps=a.slippage_bps*2))
            values = {name: outcomes(choice, base, stress) for name, choice in load(directory/'choices.npz').items()}
            for name, result in values.items():
                reports.append(daily_metrics(rows, result, window, name))
            for candidate, baseline in protocol['comparisons']:
                comparisons.append(paired_metrics(rows, values[candidate], values[baseline], window, candidate+'_vs_'+baseline))
            print('Evaluated risk decision window', window, flush=True)
        daily, paired = pd.concat(reports, ignore_index=True), pd.concat(comparisons, ignore_index=True)
        daily.to_csv(out/'daily.csv', index=False)
        paired.to_csv(out/'paired.csv', index=False)
        effects = contrasts(paired)
        write_json(out/'contrasts.json', effects)
        write_json(out/'assessment.json', screening(daily, paired, effects, protocol))
        check(out, read(out/'choices-completed.json')['files'])
        write_json(marker, dict(status='completed', finished_at=utc_now(), files={
            str(p.relative_to(out)): file_hash(p) for p in out.rglob('*') if p.is_file()
            and p.name != 'run-status.json' and '__pycache__' not in p.parts}))
    except BaseException as exc:
        write_json(marker, dict(status='failed', error=repr(exc), updated_at=utc_now()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'run'])
    for name in ['prior', 'rolling', 'output']:
        parser.add_argument('--'+name, type=Path, required=name == 'output')
    args = parser.parse_args()
    {'freeze': freeze, 'run': run}[args.stage](args)


if __name__ == '__main__':
    main()
