"""Audit common calibration support, fitted corrections, errors and decisions."""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from quant_research.plan_targets import plan_targets
from quant_research.plan_value import HEADS, PROBABILITIES, head_targets
from quant_research.storage import file_hash, utc_now, write_json


def read(p):
    return json.loads(p.read_text())


def load(p):
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output.resolve()
    state, e, protocol = [read(out/n) for n in ['run-status.json', 'experiment.json', 'protocol.json']]
    if state['status'] != 'completed':
        raise ValueError('Common calibration is incomplete')
    if (out/'verification.json').exists():
        raise FileExistsError('Preserve verification evidence')
    roots = {k: Path(e[k]) for k in ['kronos', 'prior', 'transformer']}
    checked = dict(hashes=0, restored_tree_values=0, fitted_calibrators=0, daily_errors=0, chosen_rows=0)
    for root in [out, *roots.values()]:
        for name, expected in read(root/'run-status.json')['files'].items():
            p = (root/name).resolve()
            if not p.is_relative_to(root) or file_hash(p) != expected:
                raise ValueError(f'Changed evidence: {p}')
            checked['hashes'] += 1
    for fold in protocol['fold_indices']:
        dest = out/f'fold-{fold:02d}'
        raw, rows, targets = {}, {}, {}
        for part in ['calibration', 'evaluation']:
            rows[part] = pd.read_parquet(roots['kronos']/f'fold-{fold:02d}-{part}-rows.parquet')
            ids = rows[part].parent_row_index.to_numpy()
            source = load(roots['prior']/f'fold-{fold:02d}'/f'{part}.npz')
            labels = {h: a[ids] for h, a in source.items()}
            targets[part] = head_targets(plan_targets(labels))
            raw[part] = {m: load(dest/f'{m}-raw-{part}.npz') for m in ['kronos', 'lightgbm', 'transformer']}
            native = load(roots['kronos']/f'fold-{fold:02d}'/f'raw-{part}.npz')
            name = 'raw-calibration.npz' if part == 'calibration' else 'transformer_raw-evaluation.npz'
            temporal = load(roots['transformer']/f'fold-{fold:02d}'/name)
            for h in HEADS:
                np.testing.assert_array_equal(raw[part]['kronos'][h], native[h])
                np.testing.assert_array_equal(raw[part]['transformer'][h], temporal[h][ids])
            if part == 'calibration':
                metadata = read(roots['prior']/f'fold-{fold:02d}'/'models.json')
                sample = np.unique(np.linspace(0, len(ids)-1, 48, dtype=int))
                for h in HEADS:
                    for j, spec in enumerate(metadata['heads'][h]):
                        pred = np.full(len(sample), spec['value']) if spec['kind'] == 'constant' else lgb.Booster(
                            model_file=str(roots['prior']/f'fold-{fold:02d}'/'plan-models'/spec['file'])).predict(source['x'][ids[sample]], num_threads=1)
                        if h in PROBABILITIES:
                            pred = np.clip(pred, 0, 1)
                        elif h == HEADS[4]:
                            pred = np.maximum(pred, 0)
                        np.testing.assert_allclose(pred, raw[part]['lightgbm'][h][sample, j], rtol=1e-12, atol=1e-12)
                        checked['restored_tree_values'] += len(sample)
            else:
                tree = load(roots['prior']/f'fold-{fold:02d}'/'learned_raw-evaluation.npz')
                for h in HEADS:
                    np.testing.assert_array_equal(tree[h][ids], raw[part]['lightgbm'][h])
        dates = rows['calibration'].date.to_numpy()
        common = {h: np.logical_and.reduce([np.isfinite(p[h]) for p in raw['calibration'].values()]) for h in HEADS}
        saved_mask = load(dest/'common-calibration-forecast-mask.npz')
        for h in HEADS:
            np.testing.assert_array_equal(common[h], saved_mask[h])
        forecasts = {}
        for name in raw['calibration']:
            forecasts[name] = load(dest/f'{name}-evaluation.npz')
            calibration = read(dest/f'{name}-calibration.json')
            if name == 'kronos':
                native_calibration = read(roots['kronos']/f'fold-{fold:02d}'/'calibration.json')
                for h in HEADS:
                    np.testing.assert_array_equal(common[h], np.isfinite(raw['calibration'][name][h]))
                    for actual, original in zip(calibration[h], native_calibration[h]):
                        assert actual.keys() == original.keys()
                        for key in actual:
                            if isinstance(actual[key], (float, int)):
                                np.testing.assert_allclose(actual[key], original[key], rtol=1e-6, atol=1e-8)
                            else:
                                assert actual[key] == original[key]
            for h in HEADS:
                for j, spec in enumerate(calibration[h]):
                    y, p = targets['calibration'][h][:, j], raw['calibration'][name][h][:, j]
                    known = common[h][:, j] & np.isfinite(y)
                    days, inverse, counts = np.unique(dates[known], return_inverse=True, return_counts=True)
                    assert spec['known_rows'] == int(known.sum()) and spec['known_dates'] == len(days)
                    if len(days) < protocol['min_calibration_dates']:
                        assert spec['kind'] == 'identity'
                    else:
                        w = 1/counts[inverse]
                        expected_kind = 'offset'
                        if h in PROBABILITIES:
                            expected_kind = 'constant' if np.unique(y[known]).size < 2 or np.std(p[known]) < 1e-12 else 'sigmoid'
                        assert spec['kind'] == expected_kind
                        if spec['kind'] == 'constant':
                            np.testing.assert_allclose(spec['value'], np.average(y[known], weights=w), atol=1e-12)
                        elif spec['kind'] == 'offset':
                            np.testing.assert_allclose(spec['offset'], np.average(y[known]-p[known], weights=w), atol=1e-12)
                        else:
                            assert spec['kind'] == 'sigmoid'
                            prob = np.clip(p[known], 1e-6, 1-1e-6)
                            fitted = LogisticRegression(C=1., solver='lbfgs', random_state=17)
                            fitted.fit(np.log(prob/(1-prob))[:, None], y[known], sample_weight=w*len(w)/w.sum())
                            np.testing.assert_allclose([spec['slope'], spec['intercept']],
                                [fitted.coef_[0, 0], fitted.intercept_[0]], rtol=1e-10, atol=1e-10)
                    pr = raw['evaluation'][name][h][:, j]
                    expected = pr.copy()
                    good = np.isfinite(pr)
                    if spec['kind'] == 'constant':
                        expected[good] = spec['value']
                    elif spec['kind'] == 'offset':
                        expected[good] += spec['offset']
                    elif spec['kind'] == 'sigmoid':
                        q = np.clip(pr[good], 1e-6, 1-1e-6)
                        z = np.clip(spec['slope']*np.log(q/(1-q))+spec['intercept'], -40, 40)
                        expected[good] = 1/(1+np.exp(-z))
                    if h == HEADS[4]:
                        expected = np.maximum(expected, 0)
                    np.testing.assert_allclose(expected, forecasts[name][h][:, j], rtol=1e-7, atol=1e-8, equal_nan=True)
                    np.testing.assert_array_equal(good, np.isfinite(forecasts[name][h][:, j]))
                    checked['fitted_calibrators'] += 1
            fill, resolution, mean = [forecasts[name][h] for h in HEADS[:3]]
            eligible = (fill >= .3) & (resolution >= .9) & (mean > 0)
            choice = np.where(eligible, fill*mean, -np.inf).argmax(1)
            choice[~eligible.any(1)] = -1
            saved = pd.read_parquet(dest/f'{name}-chosen.parquet')
            np.testing.assert_array_equal(choice, saved.plan_index)
            checked['chosen_rows'] += len(choice)
        metrics = pd.read_csv(dest/'head-metrics.csv')
        for day in sorted(rows['evaluation'].date.unique()):
            mask = rows['evaluation'].date.eq(day).to_numpy()
            for h in HEADS:
                y = targets['evaluation'][h][mask].ravel()
                available = np.logical_and.reduce([np.isfinite(p[h][mask].ravel()) for p in forecasts.values()])
                known = available & np.isfinite(y)
                for name, pred in forecasts.items():
                    record = metrics.loc[metrics.date.eq(day) & metrics.model.eq(name) & metrics['head'].eq(h)].iloc[0]
                    assert record.common_available == available.sum() and record.common_known == known.sum()
                    mse = np.mean((y[known]-pred[h][mask].ravel()[known])**2) if known.any() else np.nan
                    np.testing.assert_allclose(record.mse, mse, rtol=1e-10, atol=1e-12, equal_nan=True)
                    checked['daily_errors'] += 1
    write_json(out/'verification.json', dict(passed=True, checked_at=utc_now(), counts=checked,
        scope='Common support, sampled saved-tree predictions, all reused raw predictions, all fitted calibration states and corrected values, all daily MSE and plan selections; not execution proof'))
    print(checked)


if __name__ == '__main__':
    main()
