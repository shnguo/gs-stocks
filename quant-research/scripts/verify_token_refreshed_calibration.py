"""Independent raw-path quantiles, calibration fits, scores and pipeline checks."""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import verify_token_interval_calibration as audit
from token_history_run import DATA, Dataset
from token_refreshed_calibration_run import (
    DECODER,
    FIELDS,
    OLD,
    REFRESH,
    ROOT,
    SEEDS,
    predictor,
    read,
)
from verify_token_pipeline_audit import legal

from quant_research.storage import file_hash, utc_now, write_json


def quantiles(data, window, part, cfg):
    stage = cfg['windows'][window][part]
    dest = ROOT / window / 'intervals' / stage['name']
    rows = []
    for seed in SEEDS:
        folder = (ROOT / window / f'calibration-forecasts/seed{seed}' if part == 'calibration'
                  else predictor(window, seed) / 'forecast')
        ids = np.load(folder / 'row-ids.npy')
        split = np.load(REFRESH / window / ('selection-ids.npy' if part == 'calibration' else 'evaluation-ids.npy'))
        expected = data.rows.iloc[split].groupby('date').head(stage['rows_per_date']).row_id.to_numpy()
        np.testing.assert_array_equal(ids, expected)
        assert data.rows.iloc[ids].date.min() >= stage['dates'][0]
        assert data.rows.iloc[ids].label_end.max() < stage['dates'][1] < cfg['sealed_holdout_start']
        paths = np.load(folder / 'paths.npy', mmap_mode='r')
        ref = data.a['last'][ids, 3].astype(float)
        for h in [2, 5]:
            masks = legal(paths[:, :, :h])
            usable = data.a['valid'][ids, :h].all(1) & (masks.sum(1) >= 16)
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
                y = np.stack([(high / ref[indices] - 1) * 100, (low / ref[indices] - 1) * 100,
                              (high - low) / ref[indices] * 100], -1)
                for i, position in enumerate(indices):
                    for j, target in enumerate(['maximum', 'minimum', 'range']):
                        rows.append(dict(stage=stage['name'], seed=seed, row_id=int(ids[position]),
                            date=data.rows.iloc[ids[position]].date, horizon=h, target=target,
                            lower=q[0, i, j], median=q[1, i, j], upper=q[2, i, j], actual=y[i, j],
                            draws=int(masks[position].sum())))
        proof = read(folder / 'lineage.json')
        assert proof['first_batch_exact_replay']
        assert proof['checkpoint_sha256'] == file_hash(predictor(window, seed) / 'training/best.pt')
        assert proof['decoder_sha256'] == file_hash(DECODER)
        if part == 'calibration':
            for name, digest in proof['chunks'].items():
                path = folder / 'chunks' / name
                assert file_hash(path) == digest
                start = int(path.stem)
                with np.load(path) as z:
                    np.testing.assert_array_equal(z['row_ids'], ids[start:start + 4])
                    np.testing.assert_array_equal(z['paths'], paths[start:start + len(z['row_ids'])])
                    np.testing.assert_array_equal(z['valid'], legal(z['paths']))
    found = pd.DataFrame(rows)
    expected = pd.read_parquet(dest / 'quantiles.parquet')
    keys = ['seed', 'row_id', 'horizon', 'target']
    a, b = [f.sort_values(keys).reset_index(drop=True) for f in [found, expected]]
    pd.testing.assert_frame_equal(a[keys + ['stage', 'date', 'draws']], b[keys + ['stage', 'date', 'draws']], check_dtype=False)
    np.testing.assert_allclose(a[['lower', 'median', 'upper', 'actual']], b[['lower', 'median', 'upper', 'actual']], rtol=1e-10, atol=1e-10)
    sources = read(dest / 'source.json')
    assert file_hash(dest / 'quantiles.parquet') == sources['quantiles_sha256']
    for name, digest in sources['files'].items():
        assert file_hash(Path(name)) == digest
    return found


def matched_pipeline(window, cfg):
    stage = cfg['windows'][window]['evaluation']['name']
    old = pd.read_parquet(OLD / 'intervals' / stage / 'scores.parquet')
    new = pd.read_parquet(ROOT / window / 'intervals' / stage / 'scores.parquet')
    keys = ['seed', 'row_id', 'horizon', 'target']
    old = old[old.variant == 'calibrated'].set_index(keys).sort_index()
    new = new[new.variant == 'calibrated'].set_index(keys).sort_index()
    assert old.index.is_unique and new.index.is_unique
    index = old.index.intersection(new.index)
    pd.testing.assert_series_equal(old.loc[index, 'date'], new.loc[index, 'date'])
    np.testing.assert_array_equal(old.loc[index, 'actual'], new.loc[index, 'actual'])
    expected = pd.concat([old.loc[index].assign(variant='old_calibrated'),
                          new.loc[index].assign(variant='refreshed_calibrated')]).reset_index()
    stored = pd.read_parquet(ROOT / window / 'end-to-end/common.parquet')
    order = keys + ['variant']
    pd.testing.assert_frame_equal(expected.sort_values(order).reset_index(drop=True)[stored.columns],
        stored.sort_values(order).reset_index(drop=True), check_dtype=False, check_exact=False, atol=1e-10, rtol=1e-10)
    records = []
    for (seed, variant, horizon, target), group in expected.groupby(['seed', 'variant', 'horizon', 'target']):
        dates, codes = np.unique(group.date, return_inverse=True)
        counts = np.bincount(codes)
        values = {f: np.bincount(codes, weights=group[f]) / counts for f in FIELDS}
        for i, date in enumerate(dates):
            records.append(dict(stage=stage, seed=seed, variant=variant, date=date, horizon=horizon,
                target=target, **{f: float(v[i]) for f, v in values.items()}))
    daily = pd.DataFrame(records)
    saved = pd.read_csv(ROOT / window / 'end-to-end/daily.csv')
    order = ['seed', 'variant', 'date', 'horizon', 'target']
    pd.testing.assert_frame_equal(daily.sort_values(order).reset_index(drop=True)[saved.columns],
        saved.sort_values(order).reset_index(drop=True), check_dtype=False, check_exact=False, atol=1e-10, rtol=1e-10)
    # Check row-level point errors against the exact earlier refresh cohort.
    for seed in SEEDS:
        prior = pd.read_parquet(predictor(window, seed) / 'comparison/common.parquet')
        ids = np.load(predictor(window, seed) / 'forecast/row-ids.npy')
        prior['row_id'] = ids[prior.local_row.to_numpy(int)]
        prior['variant'] = prior.variant.map(dict(old='old_calibrated', refreshed='refreshed_calibrated'))
        a = prior.set_index(['row_id', 'horizon', 'target', 'variant']).mae.sort_index()
        b = stored[stored.seed == seed].set_index(['row_id', 'horizon', 'target', 'variant']).mae.sort_index()
        pd.testing.assert_series_equal(a, b, check_names=False, check_exact=False, atol=1e-10, rtol=1e-10)
    return daily.assign(window=window), len(expected)


def verify_pairs(sources):
    paired = pd.read_csv(ROOT / 'results/paired.csv', dtype={'seed': str})
    prior = pd.read_csv(REFRESH / 'results/paired.csv', dtype={'seed': str})
    for row in paired.itertuples():
        data = sources[row.comparison]
        sub = data[(data.window == row.window) & (data.horizon == row.horizon)]
        if row.target == 'high_low':
            sub = sub[sub.target.isin(['maximum', 'minimum'])]
        elif row.target != 'mean':
            sub = sub[sub.target == row.target]
        if row.seed != 'mean':
            sub = sub[sub.seed == int(row.seed)]
        values = sub.pivot_table(index='date', columns='variant', values=row.metric, aggfunc='mean').sort_index()
        names = ('raw', 'calibrated') if row.comparison == 'calibration' else ('old_calibrated', 'refreshed_calibrated')
        b, x = [values[n].to_numpy() for n in names]
        delta = x - b
        rng = np.random.default_rng(314159)
        starts = rng.integers(len(delta), size=(2000, int(np.ceil(len(delta) / 10))))
        indices = np.stack([(starts + offset) % len(delta) for offset in range(10)], axis=-1).reshape(2000, -1)[:, :len(delta)]
        low, high = np.quantile(np.mean(delta[indices], axis=1), [.025, .975])
        np.testing.assert_allclose([row.baseline, row.candidate, row.delta, row.relative_change, row.ci_low, row.ci_high],
            [b.mean(), x.mean(), delta.mean(), x.mean() / b.mean() - 1, low, high], atol=1e-10, rtol=1e-10)
        if row.metric == 'mae':
            if row.comparison == 'calibration':
                assert row.delta == 0 and row.ci_low == 0 and row.ci_high == 0
            else:
                before = prior[(prior.window == row.window) & (prior.horizon == row.horizon)
                    & (prior.target == row.target) & (prior.seed == row.seed) & (prior.metric == 'mae')].iloc[0]
                np.testing.assert_allclose([row.baseline, row.candidate, row.relative_change],
                    [before.old, before.refreshed, before.relative_change], atol=1e-10, rtol=1e-10)
    assert len(paired) == 640
    return len(paired)


def main():
    assert read(ROOT / 'run-completed.json')['passed']
    cfg = read(ROOT / 'protocol.json')
    manifest = read(ROOT / 'source-manifest.json')
    assert file_hash(ROOT / 'protocol.json') == manifest['protocol_sha256']
    for name, digest in manifest['sources'].items():
        assert file_hash(Path(name)) == digest, name
    for name, digest in manifest['code'].items():
        assert file_hash(ROOT / name) == digest, name
    consumed = ['rows.parquet', 's1.npy', 's2.npy', 'future.npy', 'valid.npy', 'mean.npy', 'scale.npy', 'last.npy', 'calendar-stamps.npy']
    for name in consumed:
        assert file_hash(DATA / name) == read(DATA / 'completed.json')['files'][name]
    frozen = read(ROOT / 'fits-frozen.json')
    data = Dataset()
    checks, daily, combined = {}, [], []
    for window in cfg['windows']:
        audit.ROOT = ROOT / window
        settings = read(audit.ROOT / 'protocol.json')
        fitted = read(audit.ROOT / 'fit.json')
        assert file_hash(audit.ROOT / 'fit.json') == frozen['fits'][window]
        for seed in SEEDS:
            assert fitted['checkpoint_sha256'][str(seed)] == file_hash(predictor(window, seed) / 'training/best.pt')
        assert fitted['decoder_sha256'] == file_hash(DECODER) and fitted['sampling'] == cfg['sampling']
        assert settings['calibration']['dates'][1] <= settings['evaluation']['dates'][0]
        for part in ['calibration', 'evaluation']:
            frame = quantiles(data, window, part, cfg)
            if part == 'calibration':
                audit.check_fit(frame, settings, fitted)
            else:
                path = audit.ROOT / 'intervals' / settings[part]['name'] / 'quantiles.parquet'
                assert path.stat().st_mtime >= datetime.fromisoformat(frozen['at']).timestamp()
            d, n = audit.check_scores(frame, settings, fitted)
            checks[f'{window}_{part}'] = dict(quantiles=len(frame), scores=n)
            if part == 'evaluation':
                daily.append(d.assign(window=window))
            print('Verified', window, part, checks[f'{window}_{part}'], flush=True)
        d, n = matched_pipeline(window, cfg)
        combined.append(d)
        checks[f'{window}_end_to_end'] = dict(matched_rows=n, prior_point_MAE_preserved=True)
    sources = dict(calibration=pd.concat(daily), end_to_end=pd.concat(combined))
    for name, frame in sources.items():
        path = ROOT / 'results' / ('calibration-daily.csv' if name == 'calibration' else 'end-to-end-daily.csv')
        order = ['window', 'seed', 'variant', 'date', 'horizon', 'target']
        expected = pd.read_csv(path)
        pd.testing.assert_frame_equal(frame.sort_values(order).reset_index(drop=True)[expected.columns],
            expected.sort_values(order).reset_index(drop=True), check_dtype=False, check_exact=False, atol=1e-10, rtol=1e-10)
    checks['paired_estimates_and_intervals'] = verify_pairs(sources)
    for done in ROOT.rglob('completed.json'):
        obj = read(done)
        assert obj['passed']
        for name, digest in obj['files'].items():
            assert file_hash(done.parent / name) == digest
    write_json(ROOT / 'independent-verification.json', dict(passed=True, at=utc_now(), checks=checks,
        factors=36, all_source_hashes_match=True, calibration_only=True, medians_exactly_unchanged=True,
        prior_point_MAE_improvements_preserved=True, sealed_holdout_opened=False))
    print('Independent verification passed', flush=True)


if __name__ == '__main__':
    main()
