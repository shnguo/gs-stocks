"""Independently recompute market context, booster samples and all head-date errors."""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from quant_research.plan_targets import plan_targets
from quant_research.plan_value import HEADS, PROBABILITIES, head_targets
from quant_research.storage import file_hash, utc_now, write_json


def read(path):
    return json.loads(path.read_text())


def load(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def independent_context(x, rows, names):
    fields = ['return_1', 'return_5', 'return_20', 'volatility_20', 'price_mean_20', 'volume_mean_5']
    source = x[:, [names.index(k) for k in fields]].astype(float)
    out = np.zeros((len(x), 12))
    for day in sorted(rows.date.unique()):
        ids = np.flatnonzero(rows.date.to_numpy() == day)
        s = source[ids]
        means = s[:, :4].mean(axis=0)
        out[ids, :4] = means
        out[ids, 4] = np.sqrt(np.mean((s[:, 0]-means[0])**2))
        out[ids, 5:8] = np.mean(s[:, [0, 4, 5]] > 0, axis=0)
        out[ids, 8:] = s[:, :4]-means
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output.resolve()
    state, experiment, protocol = [read(out/n) for n in ['run-status.json', 'experiment.json', 'protocol.json']]
    if state['status'] != 'completed':
        raise ValueError('Incomplete experiment')
    for name, expected in state['files'].items():
        path = (out/name).resolve()
        if not path.is_relative_to(out) or file_hash(path) != expected:
            raise ValueError(f'Changed file: {name}')
    prior = Path(experiment['prior'])
    if file_hash(prior/'run-status.json') != experiment['parent_status_sha256']:
        raise ValueError('Changed baseline')
    for name, expected in read(prior/'run-status.json')['files'].items():
        if file_hash(prior/name) != expected:
            raise ValueError(f'Changed baseline input: {name}')
    count = dict(context_values=0, booster_values=0, date_head_errors=0)
    for fold in experiment['training_protocol']['fold_indices']:
        parent = prior/f'fold-{fold:02d}'
        cfg = read(parent/'config.json')
        names = read(Path(cfg['root'])/'panel-h5/manifest.json')['feature_names']
        for part in ['train', 'selection', 'calibration', 'evaluation']:
            x = load(parent/f'{part}.npz')['x']
            rows = pd.read_parquet(parent/f'{part}-rows.parquet')
            context = independent_context(x, rows, names)
            for arm, columns in protocol['arms'].items():
                recorded = np.load(out/f'fold-{fold:02d}-{arm}'/f'{part}-context.npy')
                np.testing.assert_allclose(recorded, context[:, columns], rtol=1e-5, atol=2e-7)
                count['context_values'] += recorded.size
        data = load(parent/'evaluation.npz')
        target = head_targets(plan_targets(data))
        sample = np.unique(np.linspace(0, len(rows)-1, 24, dtype=int))
        for arm in protocol['arms']:
            directory = out/f'fold-{fold:02d}-{arm}'
            x = np.column_stack([data['x'], np.load(directory/'evaluation-context.npy')])
            raw, calibrated = [load(directory/f'{m}-evaluation.npz') for m in [arm+'_raw', arm]]
            metadata, calibration = read(directory/'models.json'), read(directory/'calibration.json')
            for head in HEADS:
                for j in [0, 5, 11]:
                    m = metadata['heads'][head][j]
                    predicted = np.full(len(sample), m['value']) if m['kind'] == 'constant' else lgb.Booster(
                        model_file=str(directory/'plan-models'/m['file'])).predict(x[sample], num_threads=1)
                    if head in PROBABILITIES:
                        predicted = np.clip(predicted, 0, 1)
                    elif head == HEADS[4]:
                        predicted = np.maximum(predicted, 0)
                    np.testing.assert_allclose(predicted, raw[head][sample, j], atol=1e-12)
                    count['booster_values'] += len(sample)
                for j, spec in enumerate(calibration[head]):
                    if spec['kind'] == 'constant':
                        expected = np.full(len(rows), spec['value'])
                    elif spec['kind'] == 'offset':
                        expected = raw[head][:, j]+spec['offset']
                    else:
                        p = np.clip(raw[head][:, j], 1e-6, 1-1e-6)
                        z = np.clip(spec['slope']*np.log(p/(1-p))+spec['intercept'], -40, 40)
                        expected = 1/(1+np.exp(-z))
                    if head == HEADS[4]:
                        expected = np.maximum(expected, 0)
                    np.testing.assert_allclose(expected, calibrated[head][:, j], atol=1e-12)
            metrics = pd.read_csv(directory/'head-metrics.csv')
            for date in rows.date.unique():
                ids = rows.date.eq(date).to_numpy()
                for head in HEADS:
                    for model, predicted in [(arm, calibrated), (arm+'_raw', raw)]:
                        y, p = target[head][ids].ravel(), predicted[head][ids].ravel()
                        known = np.isfinite(y)
                        recorded = metrics.loc[metrics.date.eq(date) & metrics.model.eq(model) & metrics['head'].eq(head)].iloc[0]
                        assert recorded.known_rows == known.sum()
                        np.testing.assert_allclose(np.mean((y[known]-p[known])**2), recorded.mse, atol=1e-12)
                        count['date_head_errors'] += 1
    write_json(out/'verification.json', {'passed': True, 'verified_at': utc_now(),
        'artifact_hashes': len(state['files']), **count,
        'scope': 'All context values, raw/calibrated head-date MSE and calibration values; sampled saved boosters. Not real-fill or profitability proof.'})
    print(count)


if __name__ == '__main__':
    main()
