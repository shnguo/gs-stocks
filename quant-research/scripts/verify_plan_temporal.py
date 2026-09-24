"""Independently aggregate frozen targets and reconcile with original metrics."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.plan_targets import plan_targets
from quant_research.plan_value import HEADS
from quant_research.storage import file_hash, utc_now, write_json


def read(p):
    return json.loads(p.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output.resolve()
    state, source, protocol = [read(out/n) for n in ['run-status.json','parent.json','protocol.json']]
    if state['status'] != 'completed' or (out/'verification.json').exists():
        raise ValueError('Requires completed, not previously verified results')
    prior = Path(source['root'])
    assert file_hash(prior/'run-status.json') == source['status_sha256']
    counts = dict(hashes=0, partition_plan_dates=0, offset_plan_dates=0, original_metric_reconciliations=0)
    for root in [out, prior]:
        for name, expected in read(root/'run-status.json')['files'].items():
            p = (root/name).resolve()
            if not p.is_relative_to(root) or file_hash(p) != expected:
                raise ValueError('Artifact changed')
            counts['hashes'] += 1
    daily = pd.read_csv(out/'partition-daily.csv')
    effects = pd.read_csv(out/'offset-effects.csv')
    ages = pd.read_csv(out/'partition-ages.csv')
    budgets = pd.read_csv(out/'tree-budget.csv')
    for fold in protocol['fold_indices']:
        parent = prior/f'fold-{fold:02d}'
        cfg, metadata = read(parent/'config.json'), read(parent/'models.json')
        calendar = read(Path(cfg['root'])/'panel-h5/manifest.json')['dates']
        for part in protocol['partitions']:
            rows = pd.read_parquet(parent/f'{part}-rows.parquet')
            assert sorted(rows.date.unique()) == cfg['dates'][part]
            assert rows.label_end.max() < protocol['sealed_holdout_start']
            if part != 'evaluation':
                age = ages.loc[ages.window.eq(parent.name) & ages.partition.eq(part)].iloc[0]
                first = min(cfg['dates']['evaluation'])
                assert age.last_signal == rows.date.max() and age.last_label_end == rows.label_end.max()
                assert age.first_evaluation_signal == first
                assert age.signal_age_calendar_days == (pd.Timestamp(first)-pd.Timestamp(rows.date.max())).days
                assert age.signal_age_sessions == calendar.index(first)-calendar.index(rows.date.max())
                assert age.label_age_sessions == calendar.index(first)-calendar.index(rows.label_end.max())
                following = protocol['partitions'][protocol['partitions'].index(part)+1]
                assert rows.label_end.max() < min(cfg['dates'][following])
            with np.load(parent/f'{part}.npz') as z:
                targets = plan_targets({k:z[k] for k in ['reference','future','valid','upper','lower']})
            grouped = pd.DataFrame(targets['conditional_net_return']).groupby(rows.date)
            actual = {'conditional_mean':grouped.mean(), 'conditional_std':grouped.std(ddof=0),
                'conditional_q10':grouped.quantile(.1), 'conditional_q90':grouped.quantile(.9),
                'conditional_known':grouped.count(),
                'conditional_loss_rate':pd.DataFrame(targets['conditional_loss']).groupby(rows.date).mean(),
                'filled':pd.DataFrame(targets['filled'] == 1).groupby(rows.date).sum(),
                'fill_unknown':pd.DataFrame(np.isnan(targets['filled'])).groupby(rows.date).sum(),
                'ambiguous':pd.DataFrame(targets['ambiguous']).groupby(rows.date).sum()}
            saved = daily.loc[daily.window.eq(parent.name) & daily.partition.eq(part)]
            assert len(saved) == len(cfg['dates'][part])*12
            for key, expected in actual.items():
                value = saved.pivot(index='date', columns='plan', values=key)
                np.testing.assert_allclose(value, expected, rtol=1e-10, atol=1e-12, equal_nan=True)
            np.testing.assert_array_equal(saved.groupby('date').stock_rows.first(), rows.groupby('date').size())
            if part == 'train':
                np.testing.assert_allclose(actual['conditional_mean'].mean().to_numpy(), metadata['baseline'][HEADS[2]], atol=1e-12)
            if part == 'calibration':
                correction = read(parent/'calibration.json')['empirical'][HEADS[2]]
                implied = np.asarray(metadata['baseline'][HEADS[2]])+np.array([s['offset'] for s in correction])
                np.testing.assert_allclose(actual['conditional_mean'].mean().to_numpy(), implied, atol=1e-12)
            if part == 'evaluation':
                with np.load(parent/'learned_raw-evaluation.npz') as z:
                    raw = z[HEADS[2]]
                with np.load(parent/'learned-evaluation.npz') as z:
                    corrected = z[HEADS[2]]
                y = targets['conditional_net_return']
                for date, ids in rows.groupby('date').indices.items():
                    for j in range(12):
                        known = np.isfinite(y[ids,j])
                        yy, pp, cc = y[ids,j][known], raw[ids,j][known], corrected[ids,j][known]
                        s = effects.loc[effects.window.eq(parent.name) & effects.date.eq(date) & effects.plan.eq(j)].iloc[0]
                        assert s.known_rows == len(yy)
                        np.testing.assert_allclose(cc-pp, s.offset, atol=1e-12)
                        expected = [np.mean(yy), np.mean(pp), np.mean(cc), np.mean((pp-yy)**2), np.mean((cc-yy)**2)] if len(yy) else [np.nan]*5
                        for key,value in zip(['actual_mean','raw_mean','corrected_mean','raw_mse','corrected_mse'],expected):
                            np.testing.assert_allclose(s[key],value,rtol=1e-10,atol=1e-12,equal_nan=True)
                        np.testing.assert_allclose(s.mse_change,s.corrected_mse-s.raw_mse,atol=1e-12)
                        np.testing.assert_allclose(s.mse_change,s.offset_cross_term+s.offset_square_term,atol=1e-12)
                        counts['offset_plan_dates'] += 1
                # Reconcile pooled-plan original daily MSE, not an unweighted-plan approximation.
                original = pd.read_csv(parent/'head-metrics.csv')
                for date in rows.date.unique():
                    s = effects.loc[effects.window.eq(parent.name) & effects.date.eq(date)]
                    for model,col in [('learned_raw','raw_mse'),('learned','corrected_mse')]:
                        m = original.loc[original.model.eq(model) & original.date.eq(date) & original['head'].eq(HEADS[2])].iloc[0]
                        np.testing.assert_allclose(np.average(s[col],weights=s.known_rows),m.mse,atol=1e-12)
                        counts['original_metric_reconciliations'] += 1
            counts['partition_plan_dates'] += len(saved)
            del targets
        for head in HEADS:
            for j,spec in enumerate(metadata['heads'][head]):
                row = budgets.loc[budgets.window.eq(parent.name) & budgets['head'].eq(head) & budgets.plan.eq(j)].iloc[0]
                assert row.kind == spec['kind']
                assert row.best_iteration == spec.get('best_iteration')
                assert row.registered_cap == read(prior/'protocol.json')['iterations']
                assert row.best_at_cap == (spec.get('best_iteration') == row.registered_cap)
        print('Verified temporal window', fold, flush=True)
    summary = daily.groupby(['window','partition','plan']).conditional_mean.agg(['mean','count'])
    pd.testing.assert_frame_equal(summary.reset_index(),pd.read_csv(out/'partition-plan-means.csv'),check_exact=False,rtol=1e-10,atol=1e-12)
    aggregate = summary['mean'].groupby(['window','partition']).mean().unstack('partition').reset_index()
    pd.testing.assert_frame_equal(aggregate,pd.read_csv(out/'partition-means.csv'),check_names=False,check_exact=False,rtol=1e-10,atol=1e-12)
    values = effects.groupby('window')[['raw_mse','corrected_mse','mse_change','offset_cross_term','offset_square_term']].mean().reset_index()
    pd.testing.assert_frame_equal(values,pd.read_csv(out/'offset-window-means.csv'),check_exact=False,rtol=1e-10,atol=1e-12)
    write_json(out/'verification.json',dict(passed=True,checked_at=utc_now(),counts=counts,
        scope='All partition statistics via pandas, date ages, scalar-offset effects, training empirical means, calibration empirical means and original evaluation MSE reconciliation; no causal or execution proof'))
    print(counts,flush=True)


if __name__ == '__main__':
    main()
