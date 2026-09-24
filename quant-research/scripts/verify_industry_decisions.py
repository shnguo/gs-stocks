"""Supplement the industry feature audit with calibrator and decision checks."""
import argparse
from dataclasses import replace
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from verify_plan_asof_fit import check, read
from verify_plan_prequential import (
    calibrator_check,
    independent_choice,
    load,
    mean,
    verify_comparisons,
)

from quant_research.plan_targets import plan_targets
from quant_research.plan_value import HEADS, PROBABILITIES, head_targets
from quant_research.price_strategy import TradeAssumptions
from quant_research.storage import file_hash, utc_now, write_json


def restore(directory, x, metadata):
    result = {}
    for head in HEADS:
        columns = []
        for spec in metadata['heads'][head]:
            if spec['kind'] == 'constant':
                value = np.full(len(x), spec['value'])
            else:
                value = lgb.Booster(model_file=str(directory/'plan-models'/spec['file'])).predict(x, num_threads=1)
            if head in PROBABILITIES:
                value = np.clip(value, 0, 1)
            elif head == HEADS[4]:
                value = np.maximum(value, 0)
            columns.append(value)
        result[head] = np.column_stack(columns)
    return result


def verify_decisions(rows, pred, base, stress, recorded, daily):
    chosen = independent_choice(pred)
    pd.testing.assert_frame_equal(recorded[['date', 'instrument_id']], rows[['date', 'instrument_id']])
    np.testing.assert_array_equal(recorded.plan_index, chosen)
    for head in HEADS:
        value = np.where(chosen >= 0, pred[head][np.arange(len(rows)), np.maximum(chosen, 0)], np.nan)
        np.testing.assert_allclose(recorded[head], value, atol=1e-12, equal_nan=True)
    assert not recorded.executable.any()
    assert len(daily) == rows.date.nunique() and not daily.date.duplicated().any()
    for date in rows.date.unique():
        ids = np.flatnonzero(rows.date.eq(date).to_numpy() & (chosen >= 0))
        ix = (ids, chosen[ids])
        filled, net = base['filled'][ix], base['scenario_order_net_return'][ix]
        stressed = stress['scenario_order_net_return'][ix]
        expected = dict(candidate_stock_rows=int(rows.date.eq(date).sum()), selected=len(ids),
            fill_unknown=np.isnan(filled).sum(), filled=(filled == 1).sum(),
            return_unknown=np.isnan(net).sum(), ambiguous=base['ambiguous'][ix].sum(),
            known_selected_scenario_mean=mean(net), stress_known_selected_scenario_mean=mean(stressed),
            stress_return_unknown=np.isnan(stressed).sum())
        item = daily.loc[daily.date.eq(date)].iloc[0]
        for key, value in expected.items():
            np.testing.assert_allclose(item[key], value, atol=1e-12, equal_nan=True)
    return len(chosen), len(daily)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.experiment.resolve(), args.output.resolve()
    state, experiment, protocol = [read(root/n) for n in ['run-status.json', 'experiment.json', 'protocol.json']]
    assert state['status'] == 'completed' and read(root/'verification.json')['passed']
    out.mkdir()
    counts = dict(hashes=check(root, state['files']), calibrated_states=0, restored_values=0,
        chosen_rows=0, decision_dates=0, best_at_cap=0, tree_heads=0)
    prior = Path(experiment['prior'])
    assert file_hash(prior/'run-status.json') == experiment['parent_status_sha256']
    counts['hashes'] += check(prior, read(prior/'run-status.json')['files'])
    source_paths = [Path(__file__), Path(__file__).with_name('verify_plan_prequential.py'),
        Path(__file__).with_name('verify_plan_asof_fit.py')]
    source_hashes = {str(p.resolve()): file_hash(p) for p in source_paths}
    source_hashes.update({str(p.resolve()): file_hash(p) for p in (Path(__file__).resolve().parents[1]/'src/quant_research').glob('*.py')})
    write_json(out/'launch.json', dict(started_at=utc_now(), experiment=str(root), source_hashes=source_hashes,
        parent_status_sha256=file_hash(root/'run-status.json'), feature_verification_sha256=file_hash(root/'verification.json')))
    reports = []
    for fold in experiment['training_protocol']['fold_indices']:
        parent = prior/f'fold-{fold:02d}'
        rows = pd.read_parquet(parent/'evaluation-rows.parquet')
        cal_rows = pd.read_parquet(parent/'calibration-rows.parquet')
        data, cal_data = load(parent/'evaluation.npz'), load(parent/'calibration.npz')
        assert rows.label_end.max() < experiment['training_protocol']['sealed_holdout_start']
        base = plan_targets(data)
        assumptions = TradeAssumptions()
        stress = plan_targets(data, replace(assumptions, commission=assumptions.commission*2,
            minimum_fee=assumptions.minimum_fee*2, sell_tax=assumptions.sell_tax*2, slippage_bps=assumptions.slippage_bps*2))
        for arm in protocol['arms']:
            directory = root/f'fold-{fold:02d}-{arm}'
            metadata = read(directory/'models.json')
            raw, corrected = [load(directory/f'{name}-evaluation.npz') for name in [arm+'_raw', arm]]
            sample = np.unique(np.linspace(0, len(rows)-1, 24, dtype=int))
            x = np.column_stack([data['x'], np.load(directory/'evaluation-context.npy')])
            restored = restore(directory, x[sample], metadata)
            for head in HEADS:
                np.testing.assert_allclose(restored[head], raw[head][sample], rtol=1e-12, atol=1e-12)
                counts['restored_values'] += restored[head].size
                for spec in metadata['heads'][head]:
                    if spec['kind'] != 'constant':
                        assert 1 <= spec['best_iteration'] <= experiment['training_protocol']['iterations']
                        counts['tree_heads'] += 1
                        counts['best_at_cap'] += spec['best_iteration'] == experiment['training_protocol']['iterations']
            cal_x = np.column_stack([cal_data['x'], np.load(directory/'calibration-context.npy')])
            counts['calibrated_states'] += calibrator_check(restore(directory, cal_x, metadata),
                head_targets(plan_targets(cal_data)), cal_rows.date.to_numpy(), read(directory/'calibration.json'), raw, corrected)
            for name, pred in [(arm, corrected), (arm+'_raw', raw)]:
                daily = pd.read_csv(directory/f'{name}-decision-metrics.csv')
                assert daily.model.eq(name).all() and daily.window.eq(f'fold-{fold:02d}').all()
                chosen, dates = verify_decisions(rows, pred, base, stress,
                    pd.read_parquet(directory/f'{name}-chosen.parquet'), daily)
                counts['chosen_rows'] += chosen
                counts['decision_dates'] += dates
                reports.append(daily)
        print('Verified industry decisions', fold, flush=True)
    pd.testing.assert_frame_equal(pd.concat(reports, ignore_index=True), pd.read_csv(root/'decision-metrics.csv'))
    assessment = read(root/'assessment.json')
    adapted = {'comparisons': {}}
    for arm, heads in assessment['comparisons'].items():
        adapted['comparisons'][arm+'_vs_baseline27'] = {
            head: {('learned_minus_empirical' if k == 'candidate_minus_baseline27' else k): v for k, v in item.items()}
            for head, item in heads.items()}
    verify_comparisons(pd.read_csv(root/'head-metrics.csv'), adapted)
    for p, expected in source_hashes.items():
        assert file_hash(Path(p)) == expected
    counts['hashes'] += check(root, state['files'])
    write_json(out/'verification.json', dict(passed=True, checked_at=utc_now(), counts=counts,
        scope='All 360 saved heads on 24 rows each; all calibration states independently refitted; all selected rows and daily cost/unknown diagnostics; all paired two-date block comparisons. Complements parent context audit; does not certify historical publication times or actual fills.'))
    print(counts, flush=True)


if __name__ == '__main__':
    main()
