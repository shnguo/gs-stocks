"""Independent raw-path arithmetic and aggregation verification; no model inference."""
import numpy as np
import pandas as pd
from token_seed43_diagnostic import DATA, INPUT, MID, ROOT, TRANSFER, checked, read

from quant_research.storage import file_hash, utc_now, write_json


def equal(a, b, keys):
    pd.testing.assert_frame_equal(a[b.columns].sort_values(keys).reset_index(drop=True),
        b.sort_values(keys).reset_index(drop=True), check_dtype=False, check_exact=False,
        rtol=1e-9, atol=1e-10)


def interval(x):
    x = np.asarray(x)
    starts = np.random.default_rng(314159).integers(len(x), size=(2000, (len(x)+9)//10))
    positions = np.stack([(starts+j) % len(x) for j in range(10)], axis=-1).reshape(2000, -1)[:, :len(x)]
    return np.quantile(x[positions].mean(1), [.025, .975])


def main():
    checked()
    for name, expected in read(ROOT / 'run-completed.json')['files'].items():
        assert file_hash(ROOT / name) == expected
    f = pd.read_parquet(ROOT / 'rows.parquet')
    original = pd.read_parquet(DATA / 'rows.parquet')
    y, known, last = [np.load(DATA / f'{name}.npy', mmap_mode='r') for name in ['future', 'valid', 'last']]
    expected_rows, support = [], []
    entries = pd.DataFrame(read(ROOT / 'inventory.json'))
    for (window, seed), items in entries.groupby(['window', 'seed']):
        ids = np.load(items.iloc[0].ids)
        paths = {item.owner: np.load(item.paths, mmap_mode='r') for item in items.itertuples()}
        assert original.iloc[ids].label_end.max() < '2025-08-07'
        for h in [2, 5]:
            masks = {}
            for owner, p in paths.items():
                q = p[:, :, :h]
                masks[owner] = (np.isfinite(q).all(-1) & (q[..., :4] > 0).all(-1)
                    & (q[..., 4:] >= 0).all(-1) & (q[..., 1] >= q[..., 0])
                    & (q[..., 1] >= q[..., 2]) & (q[..., 1] >= q[..., 3])
                    & (q[..., 2] <= q[..., 0]) & (q[..., 2] <= q[..., 3])).all(-1)
            common = known[ids, :h].all(1)
            for m in masks.values():
                common &= m.sum(1) >= 16
            support.append(dict(window=window, seed=seed, horizon=h, inputs=len(ids), known=known[ids, :h].all(1).sum(),
                                common_rows=common.sum(), dates=original.iloc[ids[common]].date.nunique()))
            ix = np.flatnonzero(common)
            for owner, p in paths.items():
                for start in range(0, len(ix), 64):
                    i = ix[start:start+64]
                    ref = last[ids[i], 3].astype(float)
                    high = 100*(p[i, :, :h, 1].astype(float).max(-1)/ref[:, None]-1)
                    low = 100*(p[i, :, :h, 2].astype(float).min(-1)/ref[:, None]-1)
                    mask = masks[owner][i]
                    mh = np.nanmedian(np.where(mask, high, np.nan), axis=1)
                    ml = np.nanmedian(np.where(mask, low, np.nan), axis=1)
                    yh = 100*(y[ids[i], :h, 1].max(-1)/ref-1)
                    yl = 100*(y[ids[i], :h, 2].min(-1)/ref-1)
                    midpoint = (high+low)/2
                    median = np.nanmedian(np.where(mask, midpoint, np.nan), axis=1)
                    span_median = np.nanmedian(np.where(mask, high-low, np.nan), axis=1)
                    n = mask.sum(1)
                    first = np.where(mask, np.abs(midpoint-(yh+yl)[:, None]/2), 0).sum(1)/n
                    pair_mask = mask[:, :, None] & mask[:, None, :]
                    pair_distance = np.abs(midpoint[:, :, None]-midpoint[:, None, :])
                    score = first-np.where(pair_mask, pair_distance, 0).sum((1, 2))/(2*n*n)
                    ce, we = ((mh-yh)+(ml-yl))/2, ((mh-yh)-(ml-yl))/2
                    part = pd.DataFrame(dict(window=window, seed=seed, owner=owner, row_id=ids[i], local_row=i,
                        date=original.iloc[ids[i]].date.to_numpy(), instrument_id=original.iloc[ids[i]].instrument_id.to_numpy(), horizon=h,
                        high_median=mh, low_median=ml, actual_high=yh, actual_low=yl,
                        center=(mh+ml)/2, half_width=(mh-ml)/2, center_error=ce, half_width_error=we,
                        center_mae=abs(ce), half_width_mae=abs(we), center_mse=ce**2, half_width_mse=we**2,
                        high_mae=abs(mh-yh), low_mae=abs(ml-yl), endpoint_mae=np.maximum(abs(ce), abs(we)),
                        endpoint_mse=ce**2+we**2, path_center_median=median, path_center_mae=abs(median-(yh+yl)/2),
                        path_center_bias=median-(yh+yl)/2, path_center_crps=score, path_range_mae=abs(span_median-yh+yl),
                        legal_fraction=n/p.shape[1]))
                    expected_rows.append(part)
        print('Verified paths', window, seed, flush=True)
    independent = pd.concat(expected_rows, ignore_index=True)
    keys = ['window', 'seed', 'owner', 'row_id', 'horizon']
    equal(f, independent, keys)
    equal(pd.DataFrame(support), pd.read_csv(ROOT / 'support.csv'), ['window', 'seed', 'horizon'])
    # Input features reconstructed by simple-return/log1p arithmetic using exact signal-prefix coordinates.
    features = pd.read_parquet(ROOT / 'input-features.parquet')
    raw = np.load(INPUT / 'values.npy', mmap_mode='r')
    for row in features.itertuples():
        meta = original.iloc[row.row_id]
        q = raw[int(meta.stock_index), int(meta.date_index)-20:int(meta.date_index)+1].astype(float)
        adjusted = q[:, 3]*q[:, 6]
        returns = adjusted[1:]/adjusted[:-1]-1
        np.testing.assert_allclose(row.past_volatility, np.log1p(returns).std()*100, atol=1e-10)
        np.testing.assert_allclose(row.past_return, (adjusted[-1]/adjusted[0]-1)*100, atol=1e-10)
    h1 = features[features.row_id.isin(np.load(TRANSFER / 'calibration-ids.npy'))]
    edges = np.quantile(h1.past_volatility, [1/3, 2/3])
    np.testing.assert_allclose(edges, read(ROOT / 'feature-thresholds.json')['volatility_terciles'])
    assert features.volatility_group.tolist() == np.select([features.past_volatility <= edges[0], features.past_volatility <= edges[1]], ['low', 'middle'], default='high').tolist()
    assert features.trend_group.tolist() == np.where(features.past_return < 0, 'negative', 'nonnegative').tolist()
    equal(f[['row_id', *features.columns[1:]]].drop_duplicates(), features[features.row_id.isin(f.row_id)], ['row_id'])
    # Reconcile prior reports at the row level on their original cohorts.
    old = pd.read_parquet(MID / 'initial-reference/rows.parquet').replace({'arm': {'initial': 'current'}}).rename(columns={'arm': 'owner'})
    current = f[f.window == '2024h2']
    j = current.merge(old, on=['owner', 'seed', 'row_id', 'horizon'], validate='one_to_one', suffixes=('', '_old'))
    assert len(j) == len(old) == len(current)
    np.testing.assert_allclose(j.endpoint_mae, j.endpoint_mae_old, atol=1e-10)
    np.testing.assert_allclose(j.path_center_crps, j.center_crps, atol=1e-10)
    old = pd.read_parquet(TRANSFER / 'results/common.parquet')
    old = old[old.variant.str.endswith('_raw') & old.target.isin(['maximum', 'minimum'])]
    old = old.groupby(['seed', 'owner', 'row_id', 'horizon']).mae.mean().rename('endpoint_mae').reset_index()
    equal(f[f.window == '2025transfer'], old, ['seed', 'owner', 'row_id', 'horizon'])
    # Every paired row, arithmetic contribution, aggregate and conditional interval.
    a = pd.read_parquet(ROOT / 'attribution-rows.parquet')
    metrics = [c.removesuffix('_delta') for c in a if c.endswith('_delta')]
    for comparison, g in a.groupby('comparison'):
        new, base = comparison.split('_vs_')
        idx = ['window', 'seed', 'horizon', 'row_id']
        b, n = [f[f.owner == owner].set_index(idx).sort_index() for owner in [base, new]]
        g = g.set_index(idx).sort_index()
        pd.testing.assert_index_equal(b.index, g.index)
        for metric in metrics:
            np.testing.assert_allclose(g[metric+'_delta'], n[metric]-b[metric], atol=1e-10)
        bc, bw, nc, nw = [abs(x).to_numpy() for x in [b.center_error, b.half_width_error, n.center_error, n.half_width_error]]
        mixed_center, mixed_width = np.maximum(nc, bw), np.maximum(bc, nw)
        cc = .5*((mixed_center-b.endpoint_mae)+(n.endpoint_mae-mixed_width))
        wc = .5*((mixed_width-b.endpoint_mae)+(n.endpoint_mae-mixed_center))
        np.testing.assert_allclose(g.center_contribution, cc, atol=1e-10)
        np.testing.assert_allclose(g.width_contribution, wc, atol=1e-10)
        np.testing.assert_allclose(cc+wc, g.endpoint_mae_delta, atol=1e-10)
    daily = pd.read_csv(ROOT / 'daily.csv')
    dims = ['window', 'seed', 'owner', 'horizon']
    equal(f.groupby(dims+['date'])[metrics].mean().reset_index(), daily, dims+['date'])
    equal(daily.groupby(dims)[metrics].mean().reset_index(), pd.read_csv(ROOT / 'summary.csv'), dims)
    ad = pd.read_csv(ROOT / 'attribution-daily.csv')
    dims = ['window', 'seed', 'horizon', 'comparison']
    numeric = [c for c in ad if c not in dims+['date']]
    equal(a.groupby(dims+['date'])[numeric].mean().reset_index(), ad, dims+['date'])
    grouped = {k: g for k, g in a.groupby(dims)}
    for row in pd.read_csv(ROOT / 'paired.csv').itertuples():
        g = grouped[(row.window, row.seed, row.horizon, row.comparison)]
        col = row.metric+'_delta' if row.metric in metrics else row.metric
        values = g.groupby('date')[col].mean().sort_index()
        np.testing.assert_allclose([row.delta, row.ci_low, row.ci_high], [values.mean(), *interval(values)], atol=1e-10)
        b, n = g.groupby('date')[['base_mae', 'new_mae']].mean().mean()
        np.testing.assert_allclose([row.base_mae, row.new_mae, row.mae_change_pct], [b, n, 100*(n/b-1)], atol=1e-10)
    for row in pd.read_csv(ROOT / 'concentration.csv').itertuples():
        g = grouped[(row.window, row.seed, row.horizon, row.comparison)]
        values = g.groupby('date').endpoint_mae_delta.mean().sort_index()
        stock = g.groupby('instrument_id').endpoint_mae_delta.mean()
        assert row.dates == len(values) and row.stocks == len(stock) and row.rows == len(g)
        assert row.positive_dates == sum(values > 0) and row.positive_stocks == sum(stock > 0)
        np.testing.assert_allclose([row.dates_fraction, row.stocks_fraction, row.stock_median_support],
                                  [(values > 0).mean(), (stock > 0).mean(), g.groupby('instrument_id').size().median()])
        ordered = np.sort(values.to_numpy())
        sums = np.convolve(values, np.ones(10), 'valid')
        first = int(np.argmax(sums))
        np.testing.assert_allclose([row.total_delta, row.after_top1_removed, row.after_top5_removed, row.after_worst10_removed, row.top5_net_share],
            [values.mean(), ordered[:-1].mean(), ordered[:-5].mean(), np.delete(values.to_numpy(), np.arange(first, first+10)).mean(), ordered[-5:].sum()/values.sum()], atol=1e-10)
        assert row.worst10_start == values.index[first] and row.worst10_end == values.index[first+9]
    cuts = pd.read_csv(ROOT / 'cuts.csv')
    for key, g in grouped.items():
        observed = cuts.set_index(dims).sort_index().loc[key].reset_index()
        for cut in ['month', 'instrument_id', 'volatility_group', 'trend_group']:
            c = observed[observed.cut == cut].set_index('group').sort_index()
            count = g.groupby('date').row_id.transform('size')
            weighted = g.assign(contribution=g.endpoint_mae_delta/count/g.date.nunique())
            aggregate = weighted.groupby(cut).contribution.sum().sort_index()
            np.testing.assert_allclose(c.global_delta_contribution, aggregate, atol=1e-10)
            np.testing.assert_allclose(aggregate.sum(), g.groupby('date').endpoint_mae_delta.mean().mean(), atol=1e-10)
            expected = g.groupby([cut, 'date'])[numeric].mean().groupby(cut).mean().sort_index()
            np.testing.assert_allclose(c[numeric], expected[numeric], atol=1e-10)
            assert c.rows.tolist() == g.groupby(cut).size().sort_index().tolist()
            assert c.dates.tolist() == g.groupby(cut).date.nunique().sort_index().tolist()
            np.testing.assert_allclose(c.positive_row_fraction, g.assign(positive=g.endpoint_mae_delta > 0).groupby(cut).positive.mean().sort_index())
    common = f.groupby(['window', 'horizon', 'row_id']).owner.count().eq(9)
    common = common[common].reset_index()[['window', 'horizon', 'row_id']]
    shared = f.merge(common, on=['window', 'horizon', 'row_id'], validate='many_to_one')
    keys = ['window', 'seed', 'horizon', 'owner']
    expected = shared.groupby(keys+['date'])[metrics].mean().groupby(keys).mean().reset_index()
    equal(expected, pd.read_csv(ROOT / 'all-seed-summary.csv'), keys)
    equal(common.groupby(['window', 'horizon']).size().rename('rows').reset_index(), pd.read_csv(ROOT / 'all-seed-support.csv'), ['window', 'horizon'])
    selection = pd.read_parquet(MID / 'selection-scores/rows.parquet')
    keys = ['seed', 'arm', 'horizon']
    expected = selection.groupby(keys+['date'])[['endpoint_mae', 'center_mae', 'center_crps', 'range_mae']].mean().groupby(keys).mean().reset_index()
    equal(expected, pd.read_csv(ROOT / 'original-selection-summary.csv'), keys)
    # The original loss-selection number remains unchanged.
    selection_mean = expected.groupby('arm').center_crps.mean()
    np.testing.assert_allclose([selection_mean['ce'], selection_mean['midpoint']],
        [read(MID / 'selection.json')['ce_center_crps'], read(MID / 'selection.json')['midpoint_center_crps']], atol=1e-10)
    checked()
    write_json(ROOT / 'independent-verification.json', dict(passed=True, at=utc_now(), raw_path_rows=len(f),
        attribution_rows=len(a), paired_estimates=len(pd.read_csv(ROOT / 'paired.csv')), cuts=len(cuts),
        input_features=len(features), source_forecasts=len(entries), prior_results_reconciled=True,
        original_selection_reconciled=True, all_seed_cohorts_verified=True, source_hashes_unchanged=True))
    print('Independent verification passed', flush=True)


if __name__ == '__main__':
    main()
