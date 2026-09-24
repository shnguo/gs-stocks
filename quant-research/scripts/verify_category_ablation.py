"""Rebuild categorical dictionaries, restore models, and audit matched outcomes."""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from verify_industry_decisions import restore, verify_decisions
from verify_plan_asof_fit import check, read
from verify_plan_prequential import (
    calibrator_check,
    load,
    verify_comparisons,
    verify_decision_assembly,
)

from quant_research.industry_context import read_snapshot
from quant_research.plan_targets import plan_targets
from quant_research.plan_value import HEADS, head_targets
from quant_research.price_strategy import TradeAssumptions
from quant_research.storage import file_hash, utc_now, write_json


def independent_categories(rows, index):
    records = []
    for day, group in rows.groupby('date', sort=False):
        receipt = index[day]
        assert 0 <= (pd.Timestamp(day)-pd.Timestamp(receipt['source_date'])).days <= 6
        directory = Path(receipt['directory'])
        assert file_hash(directory/'manifest.json') == receipt['manifest_sha256']
        frame = read_snapshot(directory, receipt['source_date'])
        mapping = {}
        for r in frame.itertuples():
            if isinstance(r.industry, str) and r.industry and isinstance(r.industryClassification, str) and r.industryClassification:
                mapping[r.instrument_id] = json.dumps([r.industryClassification, r.industry], ensure_ascii=False)
        for i, r in group.iterrows():
            _, exchange, code = r.instrument_id.split('.')
            if exchange == 'xshg':
                family = 'sh_68' if code[:3] in ['688', '689'] else 'sh_other_a'
            elif exchange == 'xshe':
                family = 'sz_30' if code[:3] in ['300', '301'] else 'sz_other_a'
            else:
                assert exchange == 'xbse'
                family = 'bj_identity'
            records.append(dict(position=i, industry_category=mapping.get(r.instrument_id),
                identity_exchange=exchange, identity_code_family=family, stock_identity=r.instrument_id))
    return pd.DataFrame(records).set_index('position').sort_index().reset_index(drop=True)


def verify_metrics(rows, targets, forecasts, recorded):
    count = 0
    assert len(recorded) == rows.date.nunique()*len(forecasts)*len(HEADS)
    for name, pred in forecasts.items():
        for day in rows.date.unique():
            mask = rows.date.eq(day).to_numpy()
            for head in HEADS:
                y, p = targets[head][mask].ravel(), pred[head][mask].ravel()
                assert np.isfinite(p).all()
                known = np.isfinite(y)
                match = recorded.loc[recorded.model.eq(name) & recorded.date.eq(day) & recorded['head'].eq(head)]
                assert len(match) == 1
                row = match.iloc[0]
                assert row.known_rows == known.sum() and row.total_plan_rows == len(y)
                for key, value in dict(mse=np.mean((y[known]-p[known])**2) if known.any() else np.nan,
                    mae=np.mean(abs(y[known]-p[known])) if known.any() else np.nan,
                    mean_actual=np.mean(y[known]) if known.any() else np.nan,
                    mean_prediction=np.mean(p[known]) if known.any() else np.nan).items():
                    np.testing.assert_allclose(row[key], value, rtol=1e-10, atol=1e-12, equal_nan=True)
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    root = parser.parse_args().output.resolve()
    if (root/'verification.json').exists():
        raise FileExistsError('Preserve completed verification')
    state, exp, protocol = [read(root/n) for n in ['run-status.json', 'experiment.json', 'protocol.json']]
    assert state['status'] == 'completed'
    counts = dict(hashes=check(root, state['files']), category_values=0, saved_predictions=0,
                  calibration_states=0, head_date_rows=0, chosen_rows=0, cost_dates=0)
    prior = Path(exp['prior'])
    assert file_hash(prior/'run-status.json') == exp['parent_status_sha256']
    counts['hashes'] += check(prior, read(prior/'run-status.json')['files'])
    index = read(root/'industry-snapshots.json')
    reports = []
    for fold in exp['training_protocol']['fold_indices']:
        parent = prior/f'fold-{fold:02d}'
        parts = {}
        for part in ['train', 'selection', 'calibration', 'evaluation']:
            rows = pd.read_parquet(parent/f'{part}-rows.parquet')
            assert rows.label_end.max() < exp['training_protocol']['sealed_holdout_start']
            parts[part] = (rows, independent_categories(rows, index))
        data, cal_data = load(parent/'evaluation.npz'), load(parent/'calibration.npz')
        rows, categories = parts['evaluation']
        targets = head_targets(plan_targets(data))
        base = plan_targets(data)
        a = TradeAssumptions()
        stress = plan_targets(data, replace(a, commission=a.commission*2, minimum_fee=a.minimum_fee*2,
                                            sell_tax=a.sell_tax*2, slippage_bps=a.slippage_bps*2))
        baseline = {n: load(parent/f'{n}-evaluation.npz') for n in ['learned_raw', 'learned']}
        for arm, columns in protocol['arms'].items():
            directory = root/f'fold-{fold:02d}-{arm}'
            expected_dict = {c: sorted(set(parts['train'][1][c].dropna())) for c in columns}
            assert read(directory/'dictionary.json') == expected_dict
            train_ids = set(parts['train'][0].instrument_id)
            for part, (part_rows, cats) in parts.items():
                expected = np.column_stack([cats[c].map({v:i for i,v in enumerate(expected_dict[c])}).fillna(-1).to_numpy()
                                            for c in columns])
                np.testing.assert_array_equal(np.load(directory/f'{part}-context.npy'), expected)
                counts['category_values'] += expected.size
                cohorts = pd.read_parquet(directory/f'{part}-cohorts.parquet')
                seen = part_rows.instrument_id.isin(train_ids).to_numpy()
                np.testing.assert_array_equal(cohorts.seen_stock, seen)
                np.testing.assert_array_equal(cohorts.identity_exchange, cats.identity_exchange)
                assert read(directory/f'{part}-category-coverage.json') == dict(total_rows=len(cats),
                    missing_or_unseen={c:int((expected[:,i] == -1).sum()) for i,c in enumerate(columns)},
                    seen_stock_rows=int(seen.sum()), unseen_stock_rows=int((~seen).sum()))
            metadata = read(directory/'models.json')
            x = np.column_stack([data['x'], np.load(directory/'evaluation-context.npy')])
            sample = np.unique(np.linspace(0, len(rows)-1, 24, dtype=int))
            raw, calibrated = load(directory/f'{arm}_raw-evaluation.npz'), load(directory/f'{arm}-evaluation.npz')
            restored = restore(directory, x[sample], metadata)
            for head in HEADS:
                np.testing.assert_allclose(restored[head], raw[head][sample], atol=1e-12, rtol=1e-12)
                counts['saved_predictions'] += restored[head].size
                for spec in metadata['heads'][head]:
                    if spec['kind'] == 'booster':
                        booster = lgb.Booster(model_file=str(directory/'plan-models'/spec['file']))
                        assert booster.params['categorical_feature'] == list(range(27, 27+len(columns)))
                        for key,value in protocol['categorical_parameters'].items():
                            assert booster.params[key] == value
                        assert 1 <= spec['best_iteration'] <= exp['training_protocol']['iterations']
            cal_x = np.column_stack([cal_data['x'], np.load(directory/'calibration-context.npy')])
            counts['calibration_states'] += calibrator_check(restore(directory, cal_x, metadata),
                head_targets(plan_targets(cal_data)), parts['calibration'][0].date.to_numpy(),
                read(directory/'calibration.json'), raw, calibrated)
            forecasts = {arm:calibrated, arm+'_raw':raw}
            counts['head_date_rows'] += verify_metrics(rows, targets, forecasts, pd.read_csv(directory/'head-metrics.csv'))
            for name, pred in forecasts.items():
                daily = pd.read_csv(directory/f'{name}-decision-metrics.csv')
                chosen, dates = verify_decisions(rows, pred, base, stress,
                    pd.read_parquet(directory/f'{name}-chosen.parquet'), daily)
                counts['chosen_rows'] += chosen
                counts['cost_dates'] += dates
                reports.append(daily)
            groups = pd.read_parquet(directory/'evaluation-cohorts.parquet')
            subgroup = pd.read_csv(directory/'subgroup-head-metrics.csv')
            for cohort, mask in [('seen_stock', groups.seen_stock.to_numpy()),
                ('unseen_stock', ~groups.seen_stock.to_numpy()),
                *[(f'exchange_{e}', groups.identity_exchange.eq(e).to_numpy()) for e in sorted(groups.identity_exchange.unique())]]:
                if mask.any():
                    counts['head_date_rows'] += verify_metrics(rows.loc[mask].reset_index(drop=True),
                        {h:y[mask] for h,y in targets.items()},
                        {n:{h:v[mask] for h,v in pred.items()} for n,pred in {**forecasts, **baseline}.items()},
                        subgroup.loc[subgroup.cohort.eq(cohort)])
            print('Verified category arm', fold, arm, flush=True)
    verify_decision_assembly(reports, pd.read_csv(root/'decision-metrics.csv'))
    verify_comparisons(pd.read_csv(root/'head-metrics.csv'), read(root/'assessment.json'))
    counts['hashes'] += check(root, state['files'])
    write_json(root/'verification.json', dict(passed=True, verified_at=utc_now(), counts=counts,
        scope='All categorical inputs and train-only dictionaries, all 720 heads sampled after reload, all calibrators refitted, subgroup head errors, main choices/cost dates and block comparisons. Subgroup cost tables not independently recomputed; retrospective sources do not certify historical board status, publication time or profitability.'))
    print(counts, flush=True)


if __name__ == '__main__':
    main()
