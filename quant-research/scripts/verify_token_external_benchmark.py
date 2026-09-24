"""Independent raw-path, historical-bootstrap and paired-score verification."""
import hashlib

import numpy as np
import pandas as pd
from token_external_benchmark import FIELDS, INPUT, PRIOR, ROOT, SEEDS, checked, read
from token_history_run import DATA, Dataset
from verify_token_pipeline_audit import independent_rows, legal
from verify_token_profile_transfer import verify_run

from quant_research.storage import file_hash, utc_now, write_json


def verify_inputs():
    rows = pd.read_parquet(ROOT / 'rows.parquet')
    ids = np.load(ROOT / 'row-ids.npy')
    all_rows = pd.read_parquet(DATA / 'rows.parquet')
    pd.testing.assert_frame_equal(rows, all_rows.iloc[ids].reset_index(drop=True))
    eligible = all_rows[(all_rows.date >= '2025-04-01') & (all_rows.date < '2025-07-30') & (all_rows.label_end < '2025-07-30')]
    np.testing.assert_array_equal(ids, eligible.groupby('date').head(32).row_id)
    assert rows.date.nunique() == 76 and len(rows) == 2432
    with np.load(ROOT / 'inputs.npz') as z:
        inputs = {k: z[k] for k in z.files}
    source = np.load(INPUT / 'values.npy', mmap_mode='r')
    for i, row in enumerate(rows.itertuples()):
        h = source[row.stock_index, row.date_index-59:row.date_index+1]
        np.testing.assert_array_equal(h, inputs['history'][i])
    x = inputs['history'][..., :6].astype(np.float32)
    factors = inputs['history'][..., 6] / inputs['history'][:, -1:, 6]
    x[..., :4] = x[..., :4] * factors[..., None]
    mean = np.mean(x, axis=1, keepdims=True)
    scale = np.std(x, axis=1, keepdims=True) + 1e-5
    for name, value in [('mean', mean), ('scale', scale), ('x', np.clip((x-mean)/scale, -5, 5))]:
        np.testing.assert_array_equal(inputs[name], value)
    calendar = np.load(DATA / 'calendar-stamps.npy')
    for i, row in enumerate(rows.itertuples()):
        np.testing.assert_array_equal(inputs['past'][i], calendar[row.date_index-59:row.date_index+1])
        np.testing.assert_array_equal(inputs['future'][i], calendar[row.date_index+1:row.date_index+6])
    assert rows.label_end.max() < '2025-07-30' < read(ROOT / 'protocol.json')['sealed_holdout_start']
    return ids, rows, inputs


def verify_historical(rows, inputs):
    saved = np.load(ROOT / 'historical/paths.npy', mmap_mode='r')
    draws = np.load(ROOT / 'historical/block-starts.npy')
    assert saved.shape == (2432, 64, 5, 6) and legal(saved).all()
    for i, row in enumerate(rows.itertuples()):
        h = inputs['history'][i]
        prices = h[:, :4].astype(float) * h[:, 6:7] / h[-1, 6]
        changes = np.log(prices[1:] / prices[:-1, 3:4])
        changes -= np.mean(changes[:, 3])
        digest = hashlib.sha256(f'17:{row.instrument_id}:{row.date}'.encode()).digest()
        expected = np.random.default_rng(int.from_bytes(digest[:8], 'little')).integers(0, 55, 64)
        np.testing.assert_array_equal(draws[i], expected)
        out = np.empty((64, 5, 4), float)
        previous = np.full(64, prices[-1, 3])
        for day in range(5):
            out[:, day] = previous[:, None] * np.exp(changes[expected+day])
            previous = out[:, day, 3]
        np.testing.assert_allclose(saved[i, :, :, :4], out, atol=1e-10, rtol=1e-12)
        np.testing.assert_array_equal(saved[i, :, :, 4:], 0.)
    return saved


def main():
    checked()
    assert read(ROOT / 'run-completed.json')['passed']
    ids, rows, inputs = verify_inputs()
    data = Dataset()
    baseline = verify_historical(rows, inputs)
    kronos = np.load(ROOT / 'kronos/paths.npy', mmap_mode='r')
    assert kronos.shape == baseline.shape
    lineage = read(ROOT / 'kronos/lineage.json')
    assert lineage['passed'] and lineage['reloaded_exact_paths'] == 256
    assert read(ROOT / 'kronos/input-parity.json')['history_tokens'] == 291840
    for name, digest in lineage['chunks'].items():
        p = ROOT / 'kronos/chunks' / name
        assert file_hash(p) == digest
        start = int(p.stem)
        with np.load(p) as z:
            np.testing.assert_array_equal(z['row_ids'], ids[start:start+4])
            np.testing.assert_array_equal(z['paths'], kronos[start:start+4])
            np.testing.assert_array_equal(z['valid'], legal(z['paths']))
    frames = []
    cov = pd.read_csv(ROOT / 'results/coverage.csv')
    for name, paths in [('kronos', kronos), ('historical', baseline)]:
        f = independent_rows(paths, ids, data.rows, data.a['future'], data.a['valid'], data.a['last'])
        f['row_id'] = ids[f.local_row.to_numpy()]
        frames.append(f.assign(variant=name))
        for h in [2, 5]:
            masks = legal(paths[:, :, :h])
            known = data.a['valid'][ids, :h].all(1)
            expected = cov[(cov.variant == name) & (cov.horizon == h)].iloc[0]
            assert expected.inputs == len(ids) and expected.known == known.sum()
            assert expected.usable == (known & (masks.sum(1) >= 16)).sum()
            assert expected.valid_paths == masks.sum() and expected.total_paths == masks.size
        print('Independent pairwise CRPS and raw scores:', name, len(f), flush=True)
    profiles = read(PRIOR / 'profiles.json')
    for seed in SEEDS:
        f = verify_run(data, ids, 'refreshed', seed, profiles)
        for kind in ['raw', 'calibrated']:
            frames.append(f[f.variant == f'refreshed_{kind}'].assign(variant=f'ours_{seed}_{kind}'))
        print('Verified reused self-built paths, scores and calibration:', seed, flush=True)
    fields = ['variant', 'local_row', 'row_id', 'date', 'horizon', 'target'] + FIELDS
    all_scores = pd.concat(frames, ignore_index=True)[fields]
    order = ['variant', 'row_id', 'horizon', 'target']
    def equal(a, b, sort):
        pd.testing.assert_frame_equal(a.sort_values(sort).reset_index(drop=True)[b.columns],
            b.sort_values(sort).reset_index(drop=True), check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-10)
    equal(all_scores, pd.read_parquet(ROOT / 'results/all-scores.parquet'), order)
    shared = None
    for _, g in all_scores.groupby('variant'):
        keys = set(zip(g.row_id, g.horizon, g.target))
        shared = keys if shared is None else shared & keys
    use = [(r, h, t) in shared for r, h, t in zip(all_scores.row_id, all_scores.horizon, all_scores.target)]
    common = all_scores.loc[use]
    equal(common, pd.read_parquet(ROOT / 'results/common.parquet'), order)
    records = []
    for (name, h, target), g in common.groupby(['variant', 'horizon', 'target']):
        dates, codes = np.unique(g.date, return_inverse=True)
        counts = np.bincount(codes)
        values = {k: np.bincount(codes, weights=g[k]) / counts for k in FIELDS}
        for i, date in enumerate(dates):
            records.append(dict(variant=name, horizon=h, target=target, date=date, **{k: v[i] for k, v in values.items()}))
    daily = pd.DataFrame(records)
    for kind in ['raw', 'calibrated']:
        samples = [daily[daily.variant == f'ours_{seed}_{kind}'].sort_values(['date', 'horizon', 'target']).reset_index(drop=True) for seed in SEEDS]
        avg = samples[0].copy()
        for field in FIELDS:
            avg[field] = sum(s[field] for s in samples)/3
        avg['variant'] = f'ours_mean_{kind}'
        daily = pd.concat([daily, avg], ignore_index=True)
    equal(daily, pd.read_csv(ROOT / 'results/daily.csv'), ['variant', 'date', 'horizon', 'target'])
    summary = daily.groupby(['variant', 'horizon', 'target'])[FIELDS].mean().reset_index()
    equal(summary, pd.read_csv(ROOT / 'results/summary.csv'), ['variant', 'horizon', 'target'])
    pairs = pd.read_csv(ROOT / 'results/paired.csv', dtype={'seed': str})
    assert len(pairs) == 576
    for row in pairs.itertuples():
        candidate = f'ours_{row.seed}_{row.kind}'
        targets = ['maximum', 'minimum'] if row.target == 'high_low' else [row.target]
        sub = daily[(daily.horizon == row.horizon) & daily.target.isin(targets)]
        x = sub[sub.variant == candidate].groupby('date')[row.metric].mean().sort_index()
        b = sub[sub.variant == row.against].groupby('date')[row.metric].mean().sort_index()
        np.testing.assert_array_equal(x.index, b.index)
        delta = (x-b).to_numpy()
        rng = np.random.default_rng(314159)
        starts = rng.integers(len(delta), size=(2000, int(np.ceil(len(delta)/10))))
        ix = np.stack([(starts+i) % len(delta) for i in range(10)], axis=-1).reshape(2000, -1)[:, :len(delta)]
        low, high = np.quantile(delta[ix].mean(1), [.025, .975])
        np.testing.assert_allclose([row.baseline, row.candidate, row.delta, row.relative_change, row.ci_low, row.ci_high],
            [b.mean(), x.mean(), delta.mean(), x.mean()/b.mean()-1, low, high], atol=1e-10, rtol=1e-10)
        assert row.dates == 76
    for done in ROOT.rglob('completed.json'):
        meta = read(done)
        assert meta['passed']
        for name, digest in meta['files'].items():
            assert file_hash(done.parent / name) == digest
    write_json(ROOT / 'independent-verification.json', dict(passed=True, at=utc_now(), raw_and_calibrated_score_records=len(all_scores),
        common_score_records=len(common), paired_estimates_and_intervals=len(pairs), historical_paths_reconstructed=155648,
        native_reloaded_paths=256, source_hashes_match=True, all_history_inputs_match=True,
        calibrated_crps_not_claimed=True, sealed_holdout_opened=False))
    print('Independent external benchmark verification passed', flush=True)


if __name__ == '__main__':
    main()
