"""Independently verify frozen-profile transfer, including raw distributions."""
from pathlib import Path

import numpy as np
import pandas as pd
from token_history_run import DATA, Dataset
from token_profile_transfer_run import FIELDS, PRIOR, ROOT, SEEDS, read
from verify_token_pipeline_audit import independent_rows, legal

from quant_research.storage import file_hash, utc_now, write_json


def verify_run(data, ids, owner, seed, profiles):
    folder = ROOT / 'forecasts' / owner / f'seed{seed}'
    paths = np.load(folder / 'paths.npy', mmap_mode='r')
    assert paths.shape == (len(ids), 64, 5, 6)
    proof = read(folder / 'lineage.json')
    assert proof['first_batch_exact_replay']
    assert proof['checkpoint_sha256'] == profiles[owner]['checkpoints'][str(seed)]['sha256']
    assert proof['decoder_sha256'] == profiles['decoder']['sha256']
    for name, digest in proof['chunks'].items():
        path = folder / 'chunks' / name
        assert file_hash(path) == digest
        start = int(path.stem)
        with np.load(path) as z:
            np.testing.assert_array_equal(z['row_ids'], ids[start:start + 4])
            np.testing.assert_array_equal(z['paths'], paths[start:start + len(z['row_ids'])])
            np.testing.assert_array_equal(z['valid'], legal(z['paths']))
            assert z['s1'].shape == z['s2'].shape == (len(z['row_ids']), 64, 5)
    dest = ROOT / 'scores' / owner / f'seed{seed}'
    stored_q = pd.read_parquet(dest / 'quantiles.parquet')
    raw = independent_rows(paths, ids, data.rows, data.a['future'], data.a['valid'], data.a['last'])
    keys = ['local_row', 'horizon', 'target']
    np.testing.assert_allclose(raw.sort_values(keys).crps,
        stored_q.sort_values(keys).crps, rtol=1e-10, atol=1e-10)
    ref = data.a['last'][ids, 3].astype(float)
    rows = []
    coverage = pd.read_csv(dest / 'coverage.csv').set_index('horizon')
    for h in [2, 5]:
        masks = legal(paths[:, :, :h])
        known = data.a['valid'][ids, :h].all(1)
        usable = known & (masks.sum(1) >= 16)
        cov = coverage.loc[h]
        assert cov.known == known.sum() and cov.usable == usable.sum()
        assert cov.valid_paths == masks.sum() and cov.total_paths == masks.size and cov.inputs == len(ids)
        for start in range(0, len(ids), 64):
            indices = np.arange(start, min(start + 64, len(ids)))
            indices = indices[usable[indices]]
            if not len(indices):
                continue
            p = paths[indices, :, :h].astype(float)
            high, low = p[..., 1].max(-1), p[..., 2].min(-1)
            values = np.stack([(high / ref[indices, None] - 1) * 100,
                               (low / ref[indices, None] - 1) * 100,
                               (high - low) / ref[indices, None] * 100], -1)
            values = np.where(masks[indices, :, None], values, np.nan)
            q = np.nanquantile(values, [.1, .5, .9], axis=1)
            actual = data.a['future'][ids[indices], :h].astype(float)
            high, low = actual[..., 1].max(-1), actual[..., 2].min(-1)
            actual = np.stack([(high / ref[indices] - 1) * 100, (low / ref[indices] - 1) * 100,
                               (high - low) / ref[indices] * 100], -1)
            for i, position in enumerate(indices):
                for j, target in enumerate(['maximum', 'minimum', 'range']):
                    rows.append(dict(owner=owner, seed=seed, local_row=int(position), row_id=int(ids[position]),
                        date=data.rows.iloc[ids[position]].date, horizon=h, target=target, actual=actual[i, j],
                        lower=q[0, i, j], median=q[1, i, j], upper=q[2, i, j], draws=int(masks[position].sum())))
    found = pd.DataFrame(rows)
    found = found.merge(raw[keys + ['crps']], on=keys, validate='one_to_one')
    a, b = [f.sort_values(keys).reset_index(drop=True) for f in [found, stored_q]]
    pd.testing.assert_frame_equal(a[b.columns], b, check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-10)
    fitted = read(Path(profiles[owner]['fit_path']))
    assert file_hash(Path(profiles[owner]['fit_path'])) == profiles[owner]['fit_sha256']
    params = {(p['seed'], p['horizon'], p['target']): p for p in fitted['parameters']}
    assert len(params) == 18
    all_rows = []
    for (h, target), group in found.groupby(['horizon', 'target']):
        param = params[(seed, h, target)]
        lo, mid, hi = [group[k].to_numpy() for k in ['lower', 'median', 'upper']]
        y = group.actual.to_numpy()
        for variant in ['raw', 'calibrated']:
            left, right = lo.copy(), hi.copy()
            if variant == 'calibrated':
                left = np.maximum(profiles['fit_settings']['lower_support'][target],
                    mid - param['scale'] * np.maximum(mid - lo, param['epsilon']))
                right = mid + param['scale'] * np.maximum(hi - mid, param['epsilon'])
            out = group.drop(columns=['lower', 'median', 'upper', 'crps']).copy()
            out['variant'] = f'{owner}_{variant}'
            out['lower'], out['median'], out['upper'] = left, mid, right
            out['mae'] = np.abs(mid - y)
            out['coverage80'] = ((left <= y) & (y <= right)).astype(float)
            out['width80'] = right - left
            out['interval_score80'] = right - left + 10 * np.where(y < left, left - y, 0) + 10 * np.where(y > right, y - right, 0)
            out['lower_miss'], out['upper_miss'] = (y < left).astype(float), (y > right).astype(float)
            out['crps'] = group.crps.to_numpy() if variant == 'raw' else np.nan
            all_rows.append(out)
    output = pd.concat(all_rows, ignore_index=True)
    saved = pd.read_parquet(dest / 'scores.parquet')
    sort = keys + ['variant']
    a, b = [f.sort_values(sort).reset_index(drop=True) for f in [output, saved]]
    pd.testing.assert_frame_equal(a[b.columns], b, check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-10)
    for field in ['median', 'mae']:
        both = output.pivot(index=keys, columns='variant', values=field)
        np.testing.assert_array_equal(both[f'{owner}_raw'], both[f'{owner}_calibrated'])
    assert output.loc[output.variant.str.endswith('_calibrated'), 'crps'].isna().all()
    frozen = read(dest / 'frozen-fit.json')
    assert frozen['fit_sha256'] == profiles[owner]['fit_sha256']
    assert frozen['checkpoint_sha256'] == profiles[owner]['checkpoints'][str(seed)]['sha256']
    return output


def verify_aggregation(frames, cfg):
    matched = []
    keys = ['row_id', 'horizon', 'target']
    for seed in SEEDS:
        old = frames[('old', seed)]
        new = frames[('refreshed', seed)]
        old_index = pd.MultiIndex.from_frame(old[old.variant == 'old_raw'][keys])
        new_index = pd.MultiIndex.from_frame(new[new.variant == 'refreshed_raw'][keys])
        shared = old_index.intersection(new_index)
        for frame in [old, new]:
            eligible = pd.MultiIndex.from_frame(frame[keys]).isin(shared)
            matched.append(frame[eligible])
    common = pd.concat(matched, ignore_index=True)
    expected = pd.read_parquet(ROOT / 'results/common.parquet')
    order = ['seed', 'row_id', 'horizon', 'target', 'variant']
    pd.testing.assert_frame_equal(common.sort_values(order).reset_index(drop=True)[expected.columns],
        expected.sort_values(order).reset_index(drop=True), check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-10)
    daily_rows = []
    for (seed, variant, h, target), group in common.groupby(['seed', 'variant', 'horizon', 'target']):
        dates, codes = np.unique(group.date, return_inverse=True)
        counts = np.bincount(codes)
        values = {k: np.bincount(codes, weights=group[k]) / counts for k in FIELDS}
        for i, date in enumerate(dates):
            daily_rows.append(dict(seed=seed, variant=variant, horizon=h, target=target, date=date,
                **{k: float(v[i]) for k, v in values.items()}))
    daily = pd.DataFrame(daily_rows)
    expected = pd.read_csv(ROOT / 'results/daily.csv')
    order = ['seed', 'variant', 'horizon', 'target', 'date']
    pd.testing.assert_frame_equal(daily.sort_values(order).reset_index(drop=True)[expected.columns],
        expected.sort_values(order).reset_index(drop=True), check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-10)
    paired = pd.read_csv(ROOT / 'results/paired.csv', dtype={'seed': str})
    for row in paired.itertuples():
        baseline, candidate = cfg['comparisons'][row.comparison]
        sub = daily[(daily.horizon == row.horizon) & daily.variant.isin([baseline, candidate])]
        if row.target == 'high_low':
            sub = sub[sub.target.isin(['maximum', 'minimum'])]
        elif row.target != 'mean':
            sub = sub[sub.target == row.target]
        if row.seed != 'mean':
            sub = sub[sub.seed == int(row.seed)]
        by_date = sub.pivot_table(index='date', columns='variant', values=row.metric, aggfunc='mean').sort_index()
        b, x = by_date[baseline].to_numpy(), by_date[candidate].to_numpy()
        delta = x - b
        rng = np.random.default_rng(314159)
        starts = rng.integers(len(delta), size=(2000, int(np.ceil(len(delta) / 10))))
        indices = np.stack([(starts + offset) % len(delta) for offset in range(10)], axis=-1).reshape(2000, -1)[:, :len(delta)]
        low, high = np.quantile(np.mean(delta[indices], axis=1), [.025, .975])
        np.testing.assert_allclose([row.baseline, row.candidate, row.delta, row.relative_change, row.ci_low, row.ci_high],
            [b.mean(), x.mean(), delta.mean(), x.mean() / b.mean() - 1, low, high], rtol=1e-10, atol=1e-10)
        assert row.dates == cfg['evaluation']['signal_dates']
        if row.comparison == 'refreshed_calibration' and row.metric == 'mae':
            assert row.delta == row.ci_low == row.ci_high == 0
    assert len(paired) == 360
    return dict(common_rows=len(common), daily_rows=len(daily), paired_estimates_and_intervals=len(paired))


def main():
    assert read(ROOT / 'run-completed.json')['passed']
    cfg, profiles = read(ROOT / 'protocol.json'), read(ROOT / 'profiles.json')
    manifest = read(ROOT / 'source-manifest.json')
    assert file_hash(ROOT / 'protocol.json') == manifest['protocol_sha256']
    assert file_hash(ROOT / 'profiles.json') == manifest['profiles_sha256']
    for name, digest in manifest['sources'].items():
        assert file_hash(Path(name)) == digest, name
    for name, digest in manifest['code'].items():
        assert file_hash(ROOT / name) == digest, name
    for name in ['rows.parquet', 's1.npy', 's2.npy', 'valid.npy', 'future.npy', 'last.npy', 'mean.npy', 'scale.npy', 'calendar-stamps.npy']:
        assert file_hash(DATA / name) == read(DATA / 'completed.json')['files'][name]
    original = read(PRIOR / 'research-profile.json')['checkpoints']['2024h2']
    assert profiles['refreshed']['fit_sha256'] == original['fit_sha256']
    assert profiles['refreshed']['fit_path'] == original['fit_path']
    for p in original['predictors']:
        assert profiles['refreshed']['checkpoints'][str(p['seed'])]['sha256'] == p['sha256']
    data = Dataset()
    ids = np.load(ROOT / 'row-ids.npy')
    start, end = cfg['evaluation']['dates']
    eligible = data.rows[data.rows.date.ge(start) & data.rows.date.lt(end) & data.rows.label_end.lt(end)]
    expected = eligible.groupby('date').head(32).row_id.to_numpy()
    np.testing.assert_array_equal(ids, expected)
    assert len(ids) == 2432 and len(np.unique(ids)) == len(ids)
    assert data.rows.iloc[ids].label_end.max() < end < cfg['sealed_holdout_start']
    frames, counts = {}, {}
    for seed in SEEDS:
        for owner in ['refreshed', 'old']:
            frame = verify_run(data, ids, owner, seed, profiles)
            frames[(owner, seed)] = frame
            counts[f'{owner}_seed{seed}'] = len(frame)
            print('Verified', owner, seed, len(frame), 'score records', flush=True)
    results = verify_aggregation(frames, cfg)
    for done in ROOT.rglob('completed.json'):
        meta = read(done)
        assert meta['passed']
        for name, digest in meta['files'].items():
            assert file_hash(done.parent / name) == digest
    write_json(ROOT / 'independent-verification.json', dict(passed=True, at=utc_now(), run_score_records=counts,
        **results, all_source_hashes_match=True, six_exact_replays=True, medians_exactly_preserved=True,
        predictor_refitted=False, calibration_refitted=False, calibrated_crps_not_claimed=True,
        sealed_holdout_opened=False))
    print('Independent frozen-profile transfer verification passed', flush=True)


if __name__ == '__main__':
    main()
