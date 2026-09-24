"""Verify all recent-label refits, prediction errors and paired policy outcomes."""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from verify_industry_decisions import verify_decisions
from verify_plan_prequential import verify_comparisons, verify_decision_assembly
from verify_plan_risk_decision import same_frame
from verify_recent_refit_fit import audit, check, independent_choices, load, read

from quant_research.plan_targets import plan_targets
from quant_research.plan_value import HEADS, PROBABILITIES, head_targets
from quant_research.price_strategy import TradeAssumptions
from quant_research.storage import utc_now, write_json


def policy_check(rows, choices, base, stress, window):
    daily, paired, frames = [], [], {}
    for policy, chosen in choices.items():
        frame = pd.DataFrame(dict(date=rows.date, selected=chosen >= 0, primary=0., stress=0., filled=0., ambiguous=False))
        selected = np.flatnonzero(chosen >= 0)
        ix = (selected, chosen[selected])
        frame.loc[selected, 'primary'] = base['scenario_order_net_return'][ix]
        frame.loc[selected, 'stress'] = stress['scenario_order_net_return'][ix]
        frame.loc[selected, 'filled'] = base['filled'][ix]
        frame.loc[selected, 'ambiguous'] = base['ambiguous'][ix]
        frame.loc[frame.ambiguous, 'primary'] = np.nan
        frames[policy] = frame
        for day, part in frame.groupby('date'):
            item = dict(window=window, date=day, policy=policy, candidates=len(part), selected=int(part.selected.sum()),
                no_action=int((~part.selected).sum()), ambiguous=int(part.ambiguous.sum()), filled=int(part.filled.eq(1).sum()))
            for scenario in ['primary', 'stress']:
                values = part[scenario]
                item.update({scenario+'_known': int(values.notna().sum()), scenario+'_unknown': int(values.isna().sum()),
                    'selected_'+scenario+'_known': int((part.selected & values.notna()).sum()),
                    scenario+'_mean': values.mean(), scenario+'_downside': (-values).clip(lower=0).mean()})
            daily.append(item)
    pairs = [('recent_raw/'+rule, ref) for rule in ['original', 'risk'] for ref in ['rolling_raw/'+rule, 'fixed', 'cash']]
    pairs.append(('recent_raw/risk', 'recent_raw/original'))
    for candidate, baseline in pairs:
        for scenario in ['primary', 'stress']:
            frame = pd.DataFrame(dict(date=rows.date, a=frames[candidate][scenario], b=frames[baseline][scenario]))
            for day, part in frame.groupby('date'):
                known = part.dropna()
                paired.append(dict(window=window, date=day, comparison=candidate+'_vs_'+baseline, scenario=scenario,
                    candidates=len(part), common_known=len(known), candidate_only_known=int((part.a.notna() & part.b.isna()).sum()),
                    baseline_only_known=int((part.a.isna() & part.b.notna()).sum()), both_unknown=int((part.a.isna() & part.b.isna()).sum()),
                    candidate_mean=known.a.mean(), baseline_mean=known.b.mean(), delta=(known.a-known.b).mean(),
                    candidate_downside=(-known.a).clip(lower=0).mean(), baseline_downside=(-known.b).clip(lower=0).mean()))
    return daily, paired


def verify_policy_contrasts(paired, recorded):
    assert len(recorded) == 14
    for (comparison, scenario), part in paired.groupby(['comparison', 'scenario']):
        report = recorded[comparison+'/'+scenario]
        series = [group.sort_values('date').delta.to_numpy() for _, group in part.groupby('window')]
        means = [pd.Series(x).mean() for x in series]
        np.testing.assert_allclose(report['by_window'], means, atol=1e-12)
        np.testing.assert_allclose(report['mean'], np.mean(means), atol=1e-12)
        assert report['positive_windows'] == sum(x > 0 for x in means)
        rng, draws = np.random.default_rng(17), []
        for _ in range(2000):
            sample = []
            for values in series:
                starts = rng.integers(len(values), size=(len(values)+1)//2)
                indices = [j for start in starts for j in [start, (start+1) % len(values)]][:len(values)]
                sample.append(np.nanmean(values[indices]))
            draws.append(np.mean(sample))
        np.testing.assert_allclose(report['conditional_two_date_block_95'], np.quantile(draws, [.025, .975]), atol=1e-12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    root = parser.parse_args().output.resolve()
    state, e, schedule = [read(root/n) for n in ['run-status.json', 'experiment.json', 'schedule.json']]
    assert state['status'] == 'completed' and not (root/'verification.json').exists()
    counts = dict(hashes=check(root, state['files']), fits=0, restored_values=0, training_dates=0, choices=0,
        heads=0, head_metric_rows=0, decision_dates=0, decision_rows=0, policy_daily_rows=0, policy_paired_rows=0)
    parent, prior = Path(e['parent']), Path(e['prior'])
    counts['hashes'] += check(parent, read(parent/'run-status.json')['files'])
    from quant_research.storage import file_hash
    assert file_hash(prior/'run-status.json') == read(parent/'experiment.json')['parent_status_sha256']
    counts['hashes'] += check(prior, read(prior/'run-status.json')['files'])
    final = read(root/'forecast-stage-completed.json')
    assert final['fits'] == 48
    expected = {f'{window}/fits/{day}/completed.json' for window, info in schedule.items() for day in info}
    assert set(final['files']) == expected
    counts['hashes'] += check(root, final['files'])
    target_cache, head_reports, decision_reports, policy_daily, policy_paired = {}, [], [], [], []
    for window, info in schedule.items():
        for day in info:
            for key, value in audit(root, window, day, target_cache).items():
                counts[key] += value
            assert read(root/window/'fits'/day/'completed.json')['completed_at'] <= final['completed_at']
            counts['fits'] += 1
        rows = pd.read_parquet(prior/window/'evaluation-rows.parquet')
        assert list(info) == sorted(rows.date.unique())
        prediction = load(root/window/'recent_raw-evaluation.npz')
        for head in HEADS:
            expected = np.concatenate([load(root/window/'fits'/day/'raw.npz')[head] for day in info])
            np.testing.assert_array_equal(prediction[head], expected)
        original = load(parent/window/'rolling_raw-evaluation.npz')
        forecasts = dict(recent_raw=prediction, rolling_raw=original)
        values = load(prior/window/'evaluation.npz')
        base = plan_targets(values)
        a = TradeAssumptions()
        stress = plan_targets(values, replace(a, commission=a.commission*2, minimum_fee=a.minimum_fee*2,
            sell_tax=a.sell_tax*2, slippage_bps=a.slippage_bps*2))
        targets = head_targets(base)
        records = []
        for model, pred in forecasts.items():
            for day, ids in rows.groupby('date').indices.items():
                for head in HEADS:
                    y, p = targets[head][ids].ravel(), pred[head][ids].ravel()
                    known = np.isfinite(y)
                    observed = pd.Series(y[known], dtype=float)
                    expected = pd.Series(p[known], dtype=float)
                    item = dict(window=window, model=model, date=day, head=head, total_plan_rows=len(y), known_rows=int(known.sum()),
                        mae=(observed-expected).abs().mean(), mse=((observed-expected)**2).mean(),
                        mean_actual=observed.mean(), mean_prediction=expected.mean())
                    if head in PROBABILITIES:
                        probability = expected.clip(1e-6, 1-1e-6)
                        item.update(brier=item['mse'], log_loss=(-observed*np.log(probability)-(1-observed)*np.log1p(-probability)).mean())
                    records.append(item)
            daily = pd.read_csv(root/window/f'{model}-decision-metrics.csv')
            count, dates = verify_decisions(rows, pred, base, stress, pd.read_parquet(root/window/f'{model}-chosen.parquet'), daily)
            counts['decision_rows'] += count
            counts['decision_dates'] += dates
            decision_reports.append(daily)
        report = pd.DataFrame(records)
        same_frame(report, pd.read_csv(root/window/'head-metrics.csv'), ['window', 'model', 'date', 'head'])
        head_reports.append(report)
        counts['head_metric_rows'] += len(report)
        choices = load(root/window/'policy-choices.npz')
        assert set(choices) == {'fixed', 'cash'} | {m+'/'+r for m in forecasts for r in ['original', 'risk']}
        assert (choices['fixed'] == 4).all() and (choices['cash'] == -1).all()
        for name, pred in forecasts.items():
            for rule in ['original', 'risk']:
                np.testing.assert_array_equal(choices[name+'/'+rule], independent_choices(pred, rule == 'risk'))
        daily, paired = policy_check(rows, choices, base, stress, window)
        policy_daily.extend(daily)
        policy_paired.extend(paired)
        print('Verified recent-label window', window, flush=True)
    combined = pd.concat(head_reports, ignore_index=True)
    same_frame(combined, pd.read_csv(root/'head-metrics.csv'), ['window', 'model', 'date', 'head'])
    verify_decision_assembly(decision_reports, pd.read_csv(root/'decision-metrics.csv'))
    verify_comparisons(combined, read(root/'assessment.json'))
    daily, paired = pd.DataFrame(policy_daily), pd.DataFrame(policy_paired)
    same_frame(daily, pd.read_csv(root/'policy-daily.csv'), ['window', 'policy', 'date'])
    same_frame(paired, pd.read_csv(root/'policy-paired.csv'), ['window', 'comparison', 'scenario', 'date'])
    verify_policy_contrasts(paired, read(root/'policy-contrasts.json'))
    counts['policy_daily_rows'], counts['policy_paired_rows'] = len(daily), len(paired)
    assert counts['fits'] == 48 and counts['heads'] == 48*60
    counts['hashes'] += check(root, state['files'])
    write_json(root/'verification.json', dict(passed=True, checked_at=utc_now(), counts=counts,
        scope='All48 refits,96 mature dates per fit, unchanged selected budgets and training options, all heads on24 rows, prediction assembly and choices, all head metrics and original/risk/fixed/cash outcomes and paired contrasts; no account or prospective profitability claim'))
    print(counts, flush=True)


if __name__ == '__main__':
    main()
