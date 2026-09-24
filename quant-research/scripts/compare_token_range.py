"""Frozen-path five-day extrema, range, touch and identifiable order diagnostics."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from compare_token_kronos_paths import ART, KR, ROOTS, legal, read, sha

PARENT = ART / 'token-kronos-path-comparison-20260914-v1'
OUT = ART / 'token-range-comparison-20260914-v1'
TARGETS = ['maximum', 'minimum', 'range']


def extrema(paths, reference):
    """Price extrema/range in percentage points relative to signal close."""
    high, low = paths[..., 1].max(-1), paths[..., 2].min(-1)
    return np.stack([(high/reference-1)*100, (low/reference-1)*100,
                     (high-low)/reference*100], -1)


def order_class(paths):
    """0 all global lows precede all highs; 1 reverse; 2 unresolved from daily bars."""
    hi, lo = paths[..., 1], paths[..., 2]
    h, low_mask = hi == hi.max(-1, keepdims=True), lo == lo.min(-1, keepdims=True)
    days = np.arange(paths.shape[-2])
    first_h, last_h = np.where(h, days, 99).min(-1), np.where(h, days, -1).max(-1)
    first_l = np.where(low_mask, days, 99).min(-1)
    last_l = np.where(low_mask, days, -1).max(-1)
    return np.where(last_l < first_h, 0, np.where(last_h < first_l, 1, 2))


def historical_paths(history, identity, samples=32):
    """Sample contiguous 5-day OHLC log-return blocks; remove past mean close drift."""
    x = np.asarray(history, np.float64)
    if x.shape != (60, 4) or not np.isfinite(x).all() or (x <= 0).any():
        raise ValueError('Expected 60 valid positive adjusted OHLC bars')
    relative = np.log(x[1:] / x[:-1, 3:4])
    relative -= relative[:, 3].mean()
    seed = int.from_bytes(hashlib.sha256(f'17:{identity}'.encode()).digest()[:8], 'little')
    starts = np.random.default_rng(seed).integers(0, len(relative)-5+1, size=samples)
    blocks = relative[starts[:, None] + np.arange(5)]
    previous = np.concatenate([np.zeros((samples, 1)), blocks[:, :-1, 3].cumsum(1)], 1)
    return x[-1, 3] * np.exp(previous[..., None] + blocks), starts


def score(samples, actual):
    v = np.asarray(samples, np.float64)
    q10, median, q90 = np.quantile(v, [.1, .5, .9])
    crps = np.abs(v-actual).mean() - np.abs(v[:, None]-v).mean()/2
    n = len(v)
    alternative = np.abs(v-actual).mean() - np.sum((2*np.arange(1, n+1)-n-1)*np.sort(v))/n**2
    np.testing.assert_allclose(crps, alternative, atol=1e-10, rtol=1e-10)
    interval_score = q90-q10 + 10*max(q10-actual, 0) + 10*max(actual-q90, 0)
    return dict(mae=abs(median-actual), crps=crps, coverage80=float(q10 <= actual <= q90),
                width80=q90-q10, interval_score80=interval_score, bias=median-actual)


def block_interval(values):
    values = np.asarray(values)
    rng = np.random.default_rng(17)
    means = []
    for _ in range(2000):
        starts = rng.integers(0, len(values), size=(len(values)+1)//2)
        indices = np.column_stack([starts, (starts+1) % len(values)]).ravel()[:len(values)]
        means.append(values[indices].mean())
    return dict(mean=float(values.mean()), conditional95=np.quantile(means, [.025, .975]).tolist())


def main():
    OUT.mkdir()
    protocol = dict(scope='Post-hoc diagnostics on fixed common cohort; no fit or neural resampling',
        primary_descriptive_targets=TARGETS, horizon=5, units='percentage points of signal close',
        weighting='Equal stock rows within date, then equal dates', min_legal_paths=8,
        known_labels='All five future OHLC bars must be known and valid; no partial extrema',
        baseline='60 historical adjusted OHLC bars; 59 previous-close log-return vectors; remove mean close log drift; sample 32 contiguous five-day blocks with replacement; seed17 hashed with stock/date',
        touch_levels=[.02, .05], order='Global low strictly before global high; reverse; daily ambiguity',
        intervals='2000 two-date circular block resamples; descriptive, no multiplicity adjustment',
        sealed_holdout_accessed=False, prediction_quality_promotion=False)
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    assert read(PARENT / 'completed.json')['passed']
    for p, expected in read(PARENT / 'completed.json')['files'].items():
        assert sha(PARENT / p) == expected, p
    sources = read(PARENT / 'sources.json')
    for p, expected in sources.items():
        assert sha(Path(p)) == expected, p
    print('All inherited sources and comparison files verified', flush=True)
    joint = pd.read_parquet(PARENT / 'matched-rows.parquet')
    paths = {}
    for name, root in ROOTS.items():
        with np.load(root / 'forecast/paths.npz') as z:
            paths[name] = z['paths'][joint[name].to_numpy()].astype(np.float64)
    kp = np.full((len(pd.read_parquet(KR / 'fold-15-evaluation-rows.parquet')), 32, 5, 6), np.nan)
    for p in sorted((KR / 'fold-15').glob('evaluation-paths-*.npz')):
        with np.load(p) as z:
            kp[z['row_indices']] = z['paths']
    paths['kronos_small'] = kp[joint.kronos_small.to_numpy()]
    root = ROOTS['decoder_3720k']
    with np.load(root / 'evaluation.npz') as z:
        actual, known, reference = (z['future'][joint.data_index.to_numpy()],
                                   z['valid'][joint.data_index.to_numpy()],
                                   z['last'][joint.data_index.to_numpy(), 3])
    masks = {name: legal(p) for name, p in paths.items()}
    common = np.logical_and.reduce([v.sum(1) >= 8 for v in masks.values()])
    np.testing.assert_array_equal(common, joint.all_models_eight_legal_paths)
    eligible = common & known.all(1)
    selected = joint.loc[eligible].copy().reset_index(drop=True)
    for name in paths:
        paths[name], masks[name] = paths[name][eligible], masks[name][eligible]
    actual, reference = actual[eligible], reference[eligible].astype(np.float64)
    expected = extrema(actual, reference)
    actual_order = order_class(actual)
    folder = Path(read(root / 'experiment.json')['inputs'])
    manifest = read(folder / 'manifest.json')
    raw = np.load(folder / 'values.npy', mmap_mode='r')
    si = {s: i for i,s in enumerate(manifest['instruments'])}
    di = {d: i for i,d in enumerate(manifest['dates'])}
    bp, draws = [], []
    for row in selected.itertuples():
        t = di[row.date]
        h = raw[si[row.instrument_id], t-59:t+1].astype(np.float64)
        x = h[:, :4] * h[:, 6:7] / h[-1, 6]
        b, starts = historical_paths(x, f'{row.instrument_id}:{row.date}')
        bp.append(b)
        draws.append(starts)
    baseline = np.stack(bp)
    # Add unused volume/amount zeros solely to apply the same OHLC validity function.
    baseline = np.concatenate([baseline, np.zeros((*baseline.shape[:-1], 2))], -1)
    assert legal(baseline).all()
    paths['historical_volatility'] = baseline
    paths['persistence'] = np.broadcast_to(reference[:, None, None, None], (len(reference), 32, 5, 6)).copy()
    masks.update({name: legal(paths[name]) for name in ['historical_volatility', 'persistence']})
    np.savez_compressed(OUT / 'baseline-paths.npz', paths=baseline, block_starts=np.stack(draws))
    selected.assign(actual_order=actual_order).to_parquet(OUT / 'rows.parquet', index=False)
    records, touch, order, quantiles = [], [], [], []
    for name, p in paths.items():
        for i in range(len(selected)):
            price_samples = p[i, masks[name][i]]
            values = extrema(price_samples, reference[i])
            common_keys = dict(model=name, date=selected.iloc[i].date,
                               instrument_id=selected.iloc[i].instrument_id)
            for j,target in enumerate(TARGETS):
                records.append(dict(**common_keys, target=target, **score(values[:, j], expected[i,j])))
                q = np.quantile(values[:, j], [.1,.5,.9])
                quantiles.append(dict(**common_keys, target=target, actual=expected[i,j],
                                      q10=q[0], q50=q[1], q90=q[2]))
            for threshold in [.02, .05]:
                for direction, j in [('up',0), ('down',1)]:
                    y = expected[i,j] >= threshold*100 if j==0 else expected[i,j] <= -threshold*100
                    hit = values[:,j] >= threshold*100 if j==0 else values[:,j] <= -threshold*100
                    prob = hit.mean()
                    touch.append(dict(**common_keys, direction=direction, threshold=threshold,
                                      brier=(prob-float(y))**2, probability=prob, actual=float(y)))
            classes = order_class(price_samples)
            probs = np.bincount(classes, minlength=3)/len(classes)
            truth = np.eye(3)[actual_order[i]]
            order.append(dict(**common_keys, brier=float(np.sum((probs-truth)**2)),
                              accuracy=float(np.argmax(probs)==actual_order[i]),
                              predicted_low_first=probs[0], predicted_high_first=probs[1],
                              predicted_ambiguous=probs[2], actual_ambiguous=float(actual_order[i]==2)))
    frame = pd.DataFrame(records)
    frame.to_parquet(OUT / 'row-metrics.parquet', index=False)
    pd.DataFrame(quantiles).to_parquet(OUT / 'row-predictions.parquet', index=False)
    fields = ['mae','crps','coverage80','width80','interval_score80','bias']
    daily = frame.groupby(['model','date','target'])[fields].mean().reset_index()
    daily.to_csv(OUT / 'daily-metrics.csv', index=False)
    summary = daily.groupby(['model','target'])[fields].mean().reset_index()
    summary.to_csv(OUT / 'metrics.csv', index=False)
    td = pd.DataFrame(touch).groupby(['model','date','direction','threshold'])[
        ['brier','probability','actual']].mean().reset_index()
    td.to_csv(OUT / 'daily-touch.csv', index=False)
    td.groupby(['model','direction','threshold'])[['brier','probability','actual']].mean().to_csv(OUT / 'touch.csv')
    od = pd.DataFrame(order).groupby(['model','date']).mean(numeric_only=True).reset_index()
    od.to_csv(OUT / 'daily-order.csv', index=False)
    od.groupby('model').mean(numeric_only=True).to_csv(OUT / 'order.csv')
    comparisons = []
    for target in TARGETS:
        for metric in ['mae','crps']:
            table = daily.loc[daily.target==target].pivot(index='date', columns='model', values=metric)
            for against in ['kronos_small','historical_volatility']:
                comparisons.append(dict(target=target, metric=metric, model='decoder_3720k', against=against,
                    relative_difference=float(table.decoder_3720k.mean()/table[against].mean()-1),
                    **block_interval((table.decoder_3720k-table[against]).to_numpy())))
    result = dict(matched_rows=len(joint), all_models_available=int(common.sum()),
                  fully_known_common_rows=len(selected), dates=sorted(selected.date.unique()),
                  observed_order_counts=np.bincount(actual_order,minlength=3).tolist(),
                  metrics=summary.to_dict('records'), paired_comparisons=comparisons,
                  verification='All inherited hashes and cohort checks passed; every row CRPS matched pairwise and sorted formulas; only complete five-day labels used',
                  limitations=['Post-hoc single development window with eight dates',
                    'Path legality conditions remain; same-day order is explicitly unresolved',
                    'Extrema and range do not imply obtainable trading profit',
                    'Historical block baseline assumes recent centered returns are representative',
                    'Kronos pretrained zero-shot vs locally trained predictors; checkpoint pretraining timeline unverified',
                    'No final holdout, strategy optimization or execution backtest'])
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2))
    sources[str(Path(__file__))] = sha(Path(__file__))
    sources[str(PARENT / 'completed.json')] = sha(PARENT / 'completed.json')
    (OUT / 'sources.json').write_text(json.dumps(sources, indent=2))
    (OUT / 'completed.json').write_text(json.dumps(dict(passed=True,
        files={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}), indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
