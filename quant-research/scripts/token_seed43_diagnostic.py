"""Saved-path diagnostic; no model loading, training, or prediction selection."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.range_decomposition import attribute_mae_change, point_components
from quant_research.storage import file_hash, utc_now, write_json

BASE = Path('/Users/guo/Documents/stocks/quant-research')
ART = BASE / 'artifacts'
ROOT = ART / 'token-seed43-diagnostic-20260915-v1'
DATA = ART / 'token-history-data-20260915-v1'
MID = ART / 'token-midpoint-loss-20260915-v1'
TRANSFER = ART / 'token-midpoint-transfer-20260915-v1'
REFRESH = ART / 'token-training-refresh-20260915-v1'
CAL = ART / 'token-refreshed-calibration-20260915-v1'
INPUT = ART / 'kronos-inputs-20260914-v3'
SEEDS = [17, 29, 43]
OWNERS = ['current', 'ce', 'midpoint']
WINDOWS = ['2024h1', '2024h2', '2025transfer']
METRICS = ['endpoint_mae', 'high_mae', 'low_mae', 'center_error', 'half_width_error',
           'center_mae', 'half_width_mae', 'path_center_mae', 'path_center_crps',
           'path_center_bias', 'path_range_mae', 'legal_fraction']
COMPARISONS = [('current', 'ce'), ('ce', 'midpoint'), ('current', 'midpoint')]


def read(path):
    return json.loads(path.read_text())


def inventory():
    result = []
    for window in WINDOWS:
        ids = (TRANSFER / 'calibration-ids.npy' if window == '2024h1' else
               MID / 'evaluation-ids.npy' if window == '2024h2' else TRANSFER / 'evaluation-ids.npy')
        for seed in SEEDS:
            for owner in OWNERS:
                if window == '2024h1':
                    folder = (CAL / f'2024h2/calibration-forecasts/seed{seed}' if owner == 'current'
                              else TRANSFER / f'calibration/forecasts/{owner}/seed{seed}')
                elif window == '2024h2':
                    folder = (REFRESH / f'2024h2/predictors/seed{seed}/dense_3720k_equal/forecast'
                              if owner == 'current' else MID / f'{owner}/seed{seed}/evaluation')
                else:
                    folder = TRANSFER / f'forecasts/{owner}/seed{seed}'
                result.append(dict(window=window, seed=seed, owner=owner,
                                   paths=str(folder / 'paths.npy'), ids=str(ids)))
    return result


def prepare():
    assert not ROOT.exists(), ROOT
    sources = {}
    def bind(p, expected=None):
        p = Path(p)
        value = file_hash(p)
        if expected is not None:
            assert value == expected, p
        sources[str(p)] = value
    for parent in [MID, TRANSFER, CAL, REFRESH]:
        p = parent / 'final-verification.json'
        assert read(p)['passed']
        bind(p)
    for item in inventory():
        p = Path(item['paths'])
        receipt = read(p.parent / 'completed.json')
        assert receipt['passed']
        bind(p, receipt['files']['paths.npy'])
        bind(p.parent / 'completed.json')
        bind(p.parent / 'lineage.json')
        bind(item['ids'])
        if (p.parent / 'row-ids.npy').exists():
            np.testing.assert_array_equal(np.load(item['ids']), np.load(p.parent / 'row-ids.npy'))
            bind(p.parent / 'row-ids.npy')
    manifest = read(DATA / 'completed.json')['files']
    for name in ['rows.parquet', 'future.npy', 'last.npy', 'valid.npy']:
        bind(DATA / name, manifest[name])
    for name in ['values.npy', 'manifest.json']:
        p = INPUT / name
        bind(p, read(DATA / 'sources.json')[str(p)])
    for p in [MID / 'selection-scores/rows.parquet', MID / 'selection.json',
              MID / 'selection-ids.npy', MID / 'initial-reference/rows.parquet',
              TRANSFER / 'results/common.parquet', TRANSFER / 'results/midpoint-rows.parquet',
              TRANSFER / 'profiles.json']:
        bind(p)
    for seed in SEEDS:
        for owner in OWNERS:
            folder = (REFRESH / f'2024h2/predictors/seed{seed}/dense_3720k_equal/training'
                      if owner == 'current' else MID / f'{owner}/seed{seed}/training')
            for name in ['history.json', 'summary.json', 'best.pt' if owner == 'current' else 'final.pt']:
                bind(folder / name, read(folder / 'completed.json')['files'][name])
    ROOT.mkdir()
    write_json(ROOT / 'sources.json', sources)
    write_json(ROOT / 'inventory.json', inventory())
    write_json(ROOT / 'protocol.json', dict(at=utc_now(), question='Locate seed 43 regression and check pre-transfer warnings',
        windows=WINDOWS, owners=OWNERS, seeds=SEEDS, horizons=[2, 5], minimum_legal_paths=16,
        cohort='Three-owner intersection within each seed/window/horizon; all-nine-model intersection sensitivity.',
        units='Percentage points of signal close', aggregation='Equal stocks within date, equal dates; never combine seed paths.',
        components='Center and half-width implied by endpoint medians; separately score actual per-path midpoint.',
        attribution='Two-order arithmetic replacement attribution; not causal or an executable mixed forecast.',
        cuts=['month', 'instrument', 'past 20-return volatility tercile', 'past 20-day trend sign'],
        input_features='Adjusted past closes from the exact model input array, through signal date only. Volatility thresholds from all 1792 H1 inputs, input-weighted; no future outcomes.',
        concentration='Positive-date fraction, equal-date stock contributions, removal of largest positive 1/5 dates and worst contiguous 10-date block; descriptive only.',
        uncertainty='2000 circular 10-date block bootstrap replicates, seed 314159, pointwise conditional intervals; no multiple-testing claims.',
        pretransfer='Inspect original 896-input selection table and full 1792-input calibration window separately; both reused development data.',
        constraints='No training, inference, calibration, checkpoint/seed selection, default change, or sealed holdout use.',
        limitations='Already inspected development periods and one fixed Monte Carlo draw set; initialization and sampling effects cannot be separated. Tokenizer pretraining dates unverified.'))
    (ROOT / 'code').mkdir()
    for path in [Path(__file__), BASE / 'src/quant_research/range_decomposition.py']:
        shutil.copy2(path, ROOT / 'code' / path.name)
    write_json(ROOT / 'prepared.json', dict(passed=True, at=utc_now(), files={
        str(p.relative_to(ROOT)): file_hash(p) for p in ROOT.rglob('*') if p.is_file()}))


def checked():
    for p, expected in read(ROOT / 'sources.json').items():
        assert file_hash(Path(p)) == expected, p
    for p, expected in read(ROOT / 'prepared.json')['files'].items():
        assert file_hash(ROOT / p) == expected, p


def legal(p):
    return (np.isfinite(p).all(-1) & (p[..., :4] > 0).all(-1) & (p[..., 4:] >= 0).all(-1)
            & (p[..., 1] >= p[..., [0, 2, 3]].max(-1))
            & (p[..., 2] <= p[..., [0, 1, 3]].min(-1))).all(-1)


def crps(x, y):
    x = np.sort(x)
    return np.abs(x - y).mean() - np.sum((2 * np.arange(len(x)) - len(x) + 1) * x) / len(x)**2


def ci(x):
    x = np.asarray(x)
    rng = np.random.default_rng(314159)
    idx = (rng.integers(len(x), size=(2000, (len(x) + 9) // 10, 1)) + np.arange(10)) % len(x)
    return np.quantile(x[idx.reshape(2000, -1)[:, :len(x)]].mean(1), [.025, .975])


def input_features(rows, ids):
    values = np.load(INPUT / 'values.npy', mmap_mode='r')
    result = []
    for rid in ids:
        row = rows.iloc[rid]
        history = values[int(row.stock_index), int(row.date_index)-20:int(row.date_index)+1].astype(float)
        close = history[:, 3] * history[:, 6] / history[-1, 6]
        assert len(close) == 21 and np.isfinite(close).all() and (close > 0).all()
        result.append(dict(row_id=int(rid), past_volatility=float(np.std(np.diff(np.log(close)), ddof=0)*100),
                           past_return=float((close[-1]/close[0]-1)*100)))
    out = pd.DataFrame(result)
    h1 = out[out.row_id.isin(np.load(TRANSFER / 'calibration-ids.npy'))]
    edges = np.quantile(h1.past_volatility, [1/3, 2/3])
    out['volatility_group'] = pd.cut(out.past_volatility, [-np.inf, *edges, np.inf], labels=['low', 'middle', 'high']).astype(str)
    out['trend_group'] = np.where(out.past_return < 0, 'negative', 'nonnegative')
    write_json(ROOT / 'feature-thresholds.json', dict(source='2024h1 input-only', rows=len(h1), volatility_terciles=edges.tolist()))
    return out


def run():
    checked()
    assert not (ROOT / 'rows.parquet').exists()
    rows = pd.read_parquet(DATA / 'rows.parquet')
    future, known, last = [np.load(DATA / f'{n}.npy', mmap_mode='r') for n in ['future', 'valid', 'last']]
    entries = inventory()
    ids_union = np.unique(np.concatenate([np.load(x['ids']) for x in entries]))
    features = input_features(rows, ids_union)
    features.to_parquet(ROOT / 'input-features.parquet', index=False)
    records, support = [], []
    for window in WINDOWS:
        ids = np.load(next(x['ids'] for x in entries if x['window'] == window))
        assert rows.iloc[ids].label_end.max() < '2025-08-07'
        for seed in SEEDS:
            paths = {x['owner']: np.load(x['paths'], mmap_mode='r') for x in entries if x['window'] == window and x['seed'] == seed}
            for h in [2, 5]:
                masks = {owner: legal(p[:, :, :h]) for owner, p in paths.items()}
                common = known[ids, :h].all(1)
                for mask in masks.values():
                    common &= mask.sum(1) >= 16
                support.append(dict(window=window, seed=seed, horizon=h, inputs=len(ids), known=int(known[ids, :h].all(1).sum()),
                                    common_rows=int(common.sum()), dates=rows.iloc[ids[common]].date.nunique()))
                for owner, p in paths.items():
                    for i in np.flatnonzero(common):
                        rid = int(ids[i])
                        ref = float(last[rid, 3])
                        bars = p[i, masks[owner][i], :h].astype(float)
                        high = (bars[..., 1].max(-1)/ref-1)*100
                        low = (bars[..., 2].min(-1)/ref-1)*100
                        target = future[rid, :h].astype(float)
                        yh, yl = (target[:, 1].max()/ref-1)*100, (target[:, 2].min()/ref-1)*100
                        mh, ml = np.median(high), np.median(low)
                        center, span = (high+low)/2, high-low
                        yc, yw = (yh+yl)/2, yh-yl
                        c = point_components(mh, ml, yh, yl)
                        records.append(dict(window=window, seed=seed, owner=owner, row_id=rid, local_row=int(i), horizon=h,
                            date=rows.iloc[rid].date, instrument_id=rows.iloc[rid].instrument_id,
                            high_median=mh, low_median=ml, actual_high=yh, actual_low=yl,
                            high_mae=abs(mh-yh), low_mae=abs(ml-yl), **c,
                            path_center_median=np.median(center), path_center_mae=abs(np.median(center)-yc),
                            path_center_bias=np.median(center)-yc, path_center_crps=crps(center, yc),
                            path_range_mae=abs(np.median(span)-yw), legal_fraction=float(masks[owner][i].mean())))
            print('Scored', window, seed, flush=True)
    f = pd.DataFrame(records).merge(features, on='row_id', validate='many_to_one')
    f['month'] = f.date.str[:7]
    f.to_parquet(ROOT / 'rows.parquet', index=False)
    pd.DataFrame(support).to_csv(ROOT / 'support.csv', index=False)
    daily = f.groupby(['window', 'seed', 'owner', 'horizon', 'date'])[METRICS].mean().reset_index()
    daily.to_csv(ROOT / 'daily.csv', index=False)
    daily.groupby(['window', 'seed', 'owner', 'horizon'])[METRICS].mean().reset_index().to_csv(ROOT / 'summary.csv', index=False)
    paired, attribution = [], []
    keys = ['window', 'seed', 'horizon', 'row_id']
    for base, new in COMPARISONS:
        b, n = [f[f.owner == x].set_index(keys).sort_index() for x in [base, new]]
        pd.testing.assert_index_equal(b.index, n.index)
        a = attribute_mae_change(b.center_error, b.half_width_error, n.center_error, n.half_width_error)
        out = n[['date', 'month', 'instrument_id', 'volatility_group', 'trend_group']].copy()
        for metric in METRICS:
            out[metric + '_delta'] = n[metric]-b[metric]
        out['center_contribution'], out['width_contribution'] = a['center'], a['half_width']
        out['base_mae'], out['new_mae'] = b.endpoint_mae, n.endpoint_mae
        out['comparison'] = f'{new}_vs_{base}'
        attribution.append(out.reset_index())
        for (window, seed, horizon), g in out.reset_index().groupby(['window', 'seed', 'horizon']):
            d = g.groupby('date').mean(numeric_only=True).sort_index()
            for metric in METRICS + ['center_contribution', 'width_contribution']:
                values = d[metric + '_delta' if metric in METRICS else metric]
                lo, hi = ci(values)
                paired.append(dict(window=window, seed=seed, horizon=horizon, comparison=f'{new}_vs_{base}',
                    metric=metric, delta=values.mean(), ci_low=lo, ci_high=hi,
                    base_mae=d.base_mae.mean(), new_mae=d.new_mae.mean(), mae_change_pct=(d.new_mae.mean()/d.base_mae.mean()-1)*100))
    a = pd.concat(attribution, ignore_index=True)
    a.to_parquet(ROOT / 'attribution-rows.parquet', index=False)
    pd.DataFrame(paired).to_csv(ROOT / 'paired.csv', index=False)
    dims = ['window', 'seed', 'horizon', 'comparison']
    columns = ['endpoint_mae_delta', 'high_mae_delta', 'low_mae_delta', 'center_contribution', 'width_contribution', 'base_mae', 'new_mae']
    cut_records, concentration = [], []
    for key, g in a.groupby(dims):
        identity = dict(zip(dims, key))
        d = g.groupby('date')[columns].mean().sort_index()
        delta = d.endpoint_mae_delta
        for cut in ['month', 'instrument_id', 'volatility_group', 'trend_group']:
            counts = g.groupby('date').size()
            weighted = g.assign(weight=g.date.map(1/counts)/len(counts))
            for label, sub in weighted.groupby(cut):
                local = sub.groupby('date')[columns].mean().mean()
                cut_records.append(dict(**identity, cut=cut, group=label, rows=len(sub), dates=sub.date.nunique(),
                    **local.to_dict(), global_delta_contribution=float((sub.endpoint_mae_delta*sub.weight).sum()),
                    positive_row_fraction=float((sub.endpoint_mae_delta > 0).mean())))
        stocks = g.groupby('instrument_id').endpoint_mae_delta.mean()
        largest = delta.nlargest(5)
        blocks = delta.rolling(10).sum().dropna()
        worst_end = delta.index.get_loc(blocks.idxmax())
        remaining = delta.drop(delta.index[worst_end-9:worst_end+1])
        concentration.append(dict(**identity, dates=len(delta), rows=len(g), stocks=len(stocks),
            positive_dates=int((delta > 0).sum()), positive_stocks=int((stocks > 0).sum()),
            dates_fraction=float((delta > 0).mean()), stocks_fraction=float((stocks > 0).mean()),
            stock_median_support=float(g.groupby('instrument_id').size().median()),
            total_delta=delta.mean(), after_top1_removed=delta.drop(largest.index[:1]).mean(),
            after_top5_removed=delta.drop(largest.index).mean(), after_worst10_removed=remaining.mean(),
            top5_net_share=largest.sum()/delta.sum() if delta.sum() != 0 else np.nan,
            worst10_start=delta.index[worst_end-9], worst10_end=delta.index[worst_end]))
    pd.DataFrame(cut_records).to_csv(ROOT / 'cuts.csv', index=False)
    pd.DataFrame(concentration).to_csv(ROOT / 'concentration.csv', index=False)
    a.groupby(dims+['date'])[columns].mean().reset_index().to_csv(ROOT / 'attribution-daily.csv', index=False)
    # Same inputs for all nine models isolate the small seed-dependent cohort differences.
    counts = f.groupby(['window', 'horizon', 'row_id']).size().rename('models').reset_index()
    shared = counts[counts.models == 9][['window', 'horizon', 'row_id']]
    shared_rows = f.merge(shared, validate='many_to_one', on=['window', 'horizon', 'row_id'])
    shared_daily = shared_rows.groupby(['window', 'seed', 'horizon', 'owner', 'date'])[METRICS].mean().reset_index()
    shared_daily.groupby(['window', 'seed', 'horizon', 'owner'])[METRICS].mean().reset_index().to_csv(ROOT / 'all-seed-summary.csv', index=False)
    shared.groupby(['window', 'horizon']).size().rename('rows').reset_index().to_csv(ROOT / 'all-seed-support.csv', index=False)
    # Original selection forecasts used eight inputs per date and a different draw assignment.
    old = pd.read_parquet(MID / 'selection-scores/rows.parquet')
    old.groupby(['seed', 'arm', 'horizon', 'date'])[['endpoint_mae', 'center_mae', 'center_crps', 'range_mae']].mean().groupby(
        ['seed', 'arm', 'horizon']).mean().reset_index().to_csv(ROOT / 'original-selection-summary.csv', index=False)
    histories = []
    for seed in SEEDS:
        for owner in OWNERS:
            folder = (REFRESH / f'2024h2/predictors/seed{seed}/dense_3720k_equal/training'
                      if owner == 'current' else MID / f'{owner}/seed{seed}/training')
            summary = read(folder / 'summary.json')
            for entry in read(folder / 'history.json'):
                histories.append(dict(seed=seed, owner=owner, selected_original_epoch=summary.get('selected_epoch'), **entry))
    pd.DataFrame(histories).to_csv(ROOT / 'training-history.csv', index=False)
    write_json(ROOT / 'run-completed.json', dict(passed=True, at=utc_now(), rows=len(f), attribution_rows=len(a),
        paired_estimates=len(paired), source_forecasts=len(entries), training_started=False, inference_started=False,
        files={p.name: file_hash(p) for p in ROOT.iterdir() if p.suffix in ['.parquet', '.csv']}))
    print('Completed', len(f), 'scores;', len(a), 'attributions;', len(paired), 'paired estimates', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run'])
    args = parser.parse_args()
    prepare() if args.action == 'prepare' else run()
