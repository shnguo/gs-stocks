"""Verify frozen plan-value predictions, labels and date-level arithmetic."""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from quant_research.plan_targets import plan_targets
from quant_research.plan_value import HEADS, PROBABILITIES, head_targets
from quant_research.price_strategy import TradeAssumptions, candidate_plans, trade_diagnostic
from quant_research.storage import file_hash, utc_now, write_json


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out, counts = args.output.resolve(), dict(scenario_plan_checks=0, prediction_values=0, date_head_metrics=0)
    state = read(out/'run-status.json')
    if state['status'] != 'completed':
        raise ValueError('Run is incomplete')
    for name, expected in state['files'].items():
        p = (out/name).resolve()
        if not p.is_relative_to(out) or file_hash(p) != expected:
            raise ValueError(f'Artifact changed: {name}')
    protocol = read(out/'protocol.json')
    for fold in protocol['fold_indices']:
        p = out/f'fold-{fold:02d}'
        rows = pd.read_parquet(p/'evaluation-rows.parquet')
        partitions = [pd.read_parquet(p/f'{part}-rows.parquet') for part in ['train', 'selection', 'calibration', 'evaluation']]
        for a, b in zip(partitions, partitions[1:]):
            assert a.label_end.max() < b.date.min()
        assert rows.label_end.max() < protocol['sealed_holdout_start']
        with np.load(p/'evaluation.npz') as archive:
            labels = {k: archive[k] for k in archive.files}
        outcomes = plan_targets(labels)
        sample = np.unique(np.linspace(0, len(rows)-1, 24, dtype=int))
        for i in sample:
            for j, plan in enumerate(candidate_plans(labels['reference'][i], TradeAssumptions())):
                expected = trade_diagnostic(labels['future'][i], labels['valid'][i], plan,
                    TradeAssumptions(), labels['upper'][i], labels['lower'][i])
                value = outcomes['scenario_order_net_return'][i, j]
                if expected['net_return'] is None:
                    assert np.isnan(value)
                else:
                    np.testing.assert_allclose(value, expected['net_return'], atol=1e-12)
                assert outcomes['ambiguous'][i, j] == expected['ambiguous']
                counts['scenario_plan_checks'] += 1
        with np.load(p/'learned_raw-evaluation.npz') as archive:
            raw = {h: archive[h] for h in HEADS}
        metadata, calibration = read(p/'models.json'), read(p/'calibration.json')['learned']
        for head in HEADS:
            for j in [0, 5, 11]:
                spec = metadata['heads'][head][j]
                if spec['kind'] == 'constant':
                    predicted = np.full(len(sample), spec['value'])
                else:
                    predicted = lgb.Booster(model_file=str(p/'plan-models'/spec['file'])).predict(labels['x'][sample], num_threads=1)
                if head in PROBABILITIES:
                    predicted = np.clip(predicted, 0, 1)
                elif head == HEADS[4]:
                    predicted = np.maximum(predicted, 0)
                np.testing.assert_allclose(predicted, raw[head][sample, j], atol=1e-12)
                counts['prediction_values'] += len(sample)
        with np.load(p/'learned-evaluation.npz') as archive:
            prediction = {h: archive[h] for h in HEADS}
        for head in HEADS:
            for j, spec in enumerate(calibration[head]):
                if spec['kind'] == 'constant':
                    expected = np.full(len(rows), spec['value'])
                elif spec['kind'] == 'offset':
                    expected = raw[head][:, j]+spec['offset']
                else:
                    prob = np.clip(raw[head][:, j], 1e-6, 1-1e-6)
                    logit = np.clip(spec['slope']*np.log(prob/(1-prob))+spec['intercept'], -40, 40)
                    expected = 1/(1+np.exp(-logit))
                if head == HEADS[4]:
                    expected = np.maximum(expected, 0)
                np.testing.assert_allclose(expected, prediction[head][:, j], atol=1e-12)
        target = head_targets(outcomes)
        table = pd.read_csv(p/'head-metrics.csv')
        for date in sorted(rows.date.unique()):
            ids = rows.date.eq(date).to_numpy()
            for head in HEADS:
                y, pred = target[head][ids].ravel(), prediction[head][ids].ravel()
                known = np.isfinite(y)
                row = table.loc[table.model.eq('learned') & table.date.eq(date) & table['head'].eq(head)].iloc[0]
                assert row.known_rows == known.sum()
                np.testing.assert_allclose(np.mean((y[known]-pred[known])**2), row.mse, atol=1e-12)
                counts['date_head_metrics'] += 1
        fill, resolution, mean = [prediction[h] for h in HEADS[:3]]
        eligible = (fill >= .3) & (resolution >= .9) & (mean > 0)
        index = np.argmax(np.where(eligible, fill*mean, -np.inf), axis=1)
        index[~eligible.any(axis=1)] = -1
        selected = pd.read_parquet(p/'learned-chosen.parquet')
        np.testing.assert_equal(selected.plan_index, index)
        assert not selected.executable.any()
    write_json(out/'verification.json', {'passed': True, 'verified_at': utc_now(),
        'hashes_checked': len(state['files']), **counts,
        'scope': 'All artifact hashes and learned calibration/selection; all learned date-head MSEs; sampled scalar scenarios and saved boosters. Not proof of real fills, PIT completeness or profitability.'})
    print('Verified', counts)


if __name__ == '__main__':
    main()
