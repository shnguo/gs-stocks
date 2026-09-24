"""Independent reconstruction of frozen risk choices and paired diagnostics."""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from verify_plan_asof_fit import check, read

from quant_research.plan_targets import plan_targets
from quant_research.plan_value import HEADS
from quant_research.price_strategy import TradeAssumptions
from quant_research.storage import file_hash, utc_now, write_json


def load(path):
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def same_frame(expected, actual, keys):
    assert not expected.duplicated(keys).any() and not actual.duplicated(keys).any()
    pd.testing.assert_frame_equal(expected.sort_values(keys).reset_index(drop=True)[actual.columns],
        actual.sort_values(keys).reset_index(drop=True), check_dtype=False, rtol=1e-10, atol=1e-12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.experiment.resolve(), args.output.resolve()
    state, protocol, parents = [read(root/n) for n in ['run-status.json', 'protocol.json', 'parents.json']]
    assert state['status'] == 'completed'
    out.mkdir()
    source_hashes = {str(Path(__file__).resolve()): file_hash(Path(__file__))}
    source_hashes.update({str(p.resolve()): file_hash(p) for p in (Path(__file__).resolve().parents[1]/'src/quant_research').glob('*.py')})
    helper = Path(__file__).with_name('verify_plan_asof_fit.py')
    source_hashes[str(helper.resolve())] = file_hash(helper)
    write_json(out/'launch.json', dict(started_at=utc_now(), source_hashes=source_hashes,
        experiment=str(root), parent_status_sha256=file_hash(root/'run-status.json')))
    counts = dict(hashes=check(root, state['files']), choices=0, daily_rows=0, paired_rows=0, contrasts=0)
    roots = {k: Path(v['root']) for k, v in parents.items()}
    for name, directory in roots.items():
        assert file_hash(directory/'run-status.json') == parents[name]['status_sha256']
        assert file_hash(directory/'verification.json') == parents[name]['verification_sha256']
        counts['hashes'] += check(directory, read(directory/'run-status.json')['files'])
    choice_marker = read(root/'choices-completed.json')
    assert choice_marker['completed_at'] < state['finished_at']
    counts['hashes'] += check(root, choice_marker['files'])
    daily_records, paired_records = [], []
    for fold in protocol['fold_indices']:
        window = f'fold-{fold:02d}'
        parent = roots['baseline']/window
        rows = pd.read_parquet(parent/'evaluation-rows.parquet')
        pd.testing.assert_frame_equal(rows[['date', 'instrument_id']], pd.read_parquet(root/window/'rows.parquet'))
        assert rows.label_end.max() < protocol['sealed_holdout_start']
        recorded = load(root/window/'choices.npz')
        assert set(recorded) == {'fixed', 'cash'} | {m+'/'+r for m in protocol['models'] for r in ['original', 'risk']}
        for model in protocol['models']:
            name, version = model.split('_', 1)
            prefix = ('learned' if name == 'baseline' else 'rolling')+('_raw' if version == 'raw' else '')
            pred = load(roots[name]/window/f'{prefix}-evaluation.npz')
            fill, resolution, means, downside = [pd.DataFrame(pred[h]) for h in [HEADS[0], HEADS[1], HEADS[2], HEADS[4]]]
            for rule in ['original', 'risk']:
                utility = means-downside if rule == 'risk' else means
                scores = (fill*utility).where((fill >= .3) & (resolution >= .9) & (means > 0) & (utility > 0), -np.inf)
                choices = scores.idxmax(axis=1).where(scores.max(axis=1).gt(-np.inf), -1).to_numpy()
                np.testing.assert_array_equal(recorded[model+'/'+rule], choices)
                # The original baseline run persisted choices only after calibration.
                # All variants are independently reconstructed above; compare the
                # three original versions that also have parent choice artifacts.
                if rule == 'original' and not (name == 'baseline' and version == 'raw'):
                    previous = pd.read_parquet(roots[name]/window/f'{prefix}-chosen.parquet')
                    np.testing.assert_array_equal(previous.plan_index, choices)
        assert (recorded['fixed'] == 4).all() and protocol['grid'][4] == [-.01, .03, .02]
        assert (recorded['cash'] == -1).all()
        values = load(parent/'evaluation.npz')
        a = TradeAssumptions(**protocol['assumptions'])
        base = plan_targets(values, a)
        stress = plan_targets(values, replace(a, commission=a.commission*2, minimum_fee=a.minimum_fee*2,
            sell_tax=a.sell_tax*2, slippage_bps=a.slippage_bps*2))
        scenarios = {}
        for policy, chosen in recorded.items():
            data = rows[['date', 'instrument_id']].copy()
            data['selected'] = chosen >= 0
            data['primary'], data['stress'], data['filled'], data['ambiguous'] = 0., 0., 0., False
            ids = np.flatnonzero(chosen >= 0)
            ix = (ids, chosen[ids])
            data.loc[ids, 'primary'] = base['scenario_order_net_return'][ix]
            data.loc[ids, 'stress'] = stress['scenario_order_net_return'][ix]
            data.loc[ids, 'filled'] = base['filled'][ix]
            data.loc[ids, 'ambiguous'] = base['ambiguous'][ix]
            data.loc[data.ambiguous, 'primary'] = np.nan
            scenarios[policy] = data
            for day, part in data.groupby('date', sort=True):
                item = dict(window=window, date=day, policy=policy, candidates=len(part),
                    selected=int(part.selected.sum()), no_action=int((~part.selected).sum()),
                    ambiguous=int(part.ambiguous.sum()), filled=int(part.filled.eq(1).sum()))
                for scenario in ['primary', 'stress']:
                    net = part[scenario]
                    item.update({scenario+'_known': int(net.notna().sum()), scenario+'_unknown': int(net.isna().sum()),
                        'selected_'+scenario+'_known': int((part.selected & net.notna()).sum()),
                        scenario+'_mean': net.mean(), scenario+'_downside': (-net).clip(lower=0).mean()})
                daily_records.append(item)
            counts['choices'] += len(chosen)
        for candidate, baseline in protocol['comparisons']:
            for scenario in ['primary', 'stress']:
                combined = pd.DataFrame(dict(date=rows.date, a=scenarios[candidate][scenario], b=scenarios[baseline][scenario]))
                for day, part in combined.groupby('date', sort=True):
                    known = part.dropna()
                    paired_records.append(dict(window=window, date=day, comparison=candidate+'_vs_'+baseline,
                        scenario=scenario, candidates=len(part), common_known=len(known),
                        candidate_only_known=int((part.a.notna() & part.b.isna()).sum()),
                        baseline_only_known=int((part.a.isna() & part.b.notna()).sum()),
                        both_unknown=int((part.a.isna() & part.b.isna()).sum()), candidate_mean=known.a.mean(),
                        baseline_mean=known.b.mean(), delta=(known.a-known.b).mean(),
                        candidate_downside=(-known.a).clip(lower=0).mean(), baseline_downside=(-known.b).clip(lower=0).mean()))
        print('Verified risk choices and paired outcomes', window, flush=True)
    daily, paired = pd.DataFrame(daily_records), pd.DataFrame(paired_records)
    same_frame(daily, pd.read_csv(root/'daily.csv'), ['window', 'policy', 'date'])
    same_frame(paired, pd.read_csv(root/'paired.csv'), ['window', 'comparison', 'scenario', 'date'])
    counts['daily_rows'], counts['paired_rows'] = len(daily), len(paired)
    effects = read(root/'contrasts.json')
    assert len(effects) == len(protocol['comparisons'])*2
    for (comparison, scenario), frame in paired.groupby(['comparison', 'scenario']):
        report = effects[comparison+'/'+scenario]
        series = [group.sort_values('date').delta.to_numpy() for _, group in frame.groupby('window')]
        means = [pd.Series(x).mean() for x in series]
        np.testing.assert_allclose(report['by_window'], means, atol=1e-12)
        np.testing.assert_allclose(report['mean'], np.mean(means), atol=1e-12)
        assert report['positive_windows'] == sum(x > 0 for x in means)
        rng = np.random.default_rng(17)
        draws = []
        for _ in range(2000):
            parts = []
            for x in series:
                starts = rng.integers(0, len(x), size=(len(x)+1)//2)
                indices = [j for start in starts for j in [start, (start+1) % len(x)]][:len(x)]
                parts.append(float(np.nanmean(x[indices])))
            draws.append(np.mean(parts))
        np.testing.assert_allclose(report['conditional_two_date_block_95'], np.quantile(draws, [.025, .975]), atol=1e-12)
        counts['contrasts'] += 1
    selected = daily[daily.policy.eq('rolling_raw/risk')]
    gates = dict(nontrivial_selection=bool(selected.selected.sum() >= .01*selected.candidates.sum()),
        active_dates=bool(selected.selected.gt(0).sum() >= 24))
    for scenario in ['primary', 'stress']:
        gates[scenario+'_selected_coverage'] = all(p['selected_'+scenario+'_known'].sum() >= .95*max(p.selected.sum(), 1)
            for _, p in selected.groupby('window'))
        for reference in ['rolling_raw/original', 'fixed', 'cash']:
            name = 'rolling_raw/risk_vs_'+reference
            part = paired[paired.comparison.eq(name) & paired.scenario.eq(scenario)]
            report = effects[name+'/'+scenario]
            gates[name+'/'+scenario+'/positive_interval'] = report['conditional_two_date_block_95'][0] > 0
            gates[name+'/'+scenario+'/stable_windows'] = report['positive_windows'] >= 4
            gates[name+'/'+scenario+'/common_coverage'] = all(g.common_known.sum() >= .95*g.candidates.sum() for _, g in part.groupby('window'))
            if reference == 'rolling_raw/original':
                gates[scenario+'/no_downside_increase'] = (part.candidate_downside-part.baseline_downside).groupby(part.window).mean().mean() <= 1e-12
    assessment = read(root/'assessment.json')
    assert assessment['checks'] == gates and assessment['passed'] == all(gates.values())
    assert assessment['quality_promotion'] is False and protocol['promotion'] is False
    for path, expected in source_hashes.items():
        assert file_hash(Path(path)) == expected
    counts['hashes'] += check(root, state['files'])
    write_json(out/'verification.json', dict(passed=True, checked_at=utc_now(), counts=counts,
        scope='All saved plan choices independently reconstructed; daily and common-stock-date outcome statistics, all block contrasts and all screening gates checked; previously verified daily scenario labels reused, not live fills or account returns'))
    print(counts, flush=True)


if __name__ == '__main__':
    main()
