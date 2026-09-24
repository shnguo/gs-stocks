"""Independent artifact checks and exact checkpoint/path replay for feature experiments."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from token_features_run import feature_schema
from token_history_run import DATA, Dataset
from tokenizer_reconstruction_run import load_decoder

from quant_research.free_features import replay
from quant_research.storage import file_hash, utc_now, write_json
from quant_research.token_features import fit_normalizer, history_features
from quant_research.token_transformer import (
    decode_paths,
    generate_tokens,
    restore_model,
    valid_bars,
)

BASE = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())



def reference_indicators(raw):
    """Independent pandas implementation, including calendar-gap resets."""
    raw = np.asarray(raw, dtype=np.float64)
    output = np.full((len(raw), 6), np.nan)
    adjusted = raw[:, :4]*raw[:, 6:7]
    ok = valid_bars(raw[:, :6]) & np.isfinite(raw[:, 6]) & (raw[:, 6] > 0)
    positions = np.flatnonzero(ok)
    for indices in np.split(positions, np.flatnonzero(np.diff(positions) != 1)+1):
        if not len(indices):
            continue
        price = adjusted[indices]
        c, h, low = [pd.Series(price[:, i]) for i in [3, 1, 2]]
        mean, std = c.rolling(20).mean(), c.rolling(20).std(ddof=0)
        width = 4*std
        boll = (c-mean+2*std)/width.where(std > mean.abs()*1e-12)
        dif = c.ewm(span=12, adjust=False, min_periods=12).mean()-c.ewm(span=26, adjust=False, min_periods=26).mean()
        dea = dif.ewm(span=9, adjust=False, min_periods=9).mean()
        hi, lo = h.rolling(9).max(), low.rolling(9).min()
        rsv = (c-lo)/(hi-lo).where(hi-lo > hi.abs()*1e-12)
        k = d = .5
        ks, ds = [], []
        for value in rsv:
            if not np.isfinite(value):
                k = d = .5
                ks.append(np.nan)
                ds.append(np.nan)
            else:
                k = (2*k+value)/3
                d = (2*d+k)/3
                ks.append(k)
                ds.append(d)
        output[indices] = np.column_stack([boll, width/mean, dif/c, (dif-dea)/c, ks, ds])
    return output.astype(np.float32)


def verify_indicators(root, cfg, data):
    from quant_research.token_indicators import indicator_features

    for p, expected in read(root / 'live-state.json').items():
        assert file_hash(Path(p)) == expected, 'Live workflow changed during the experiment'
    raw = np.load(BASE / cfg['price_input'] / 'values.npy', mmap_mode='r')
    checked = 0
    for part in ['training', 'selection', 'evaluation']:
        ids = np.load(root / f'{part}-ids.npy')
        np.testing.assert_array_equal(ids, np.load(BASE / cfg['cohort_source'] / f'{part}-ids.npy'))
        rows = data.rows.iloc[ids]
        assert rows.date.min() >= cfg[part][0] and rows.label_end.max() < cfg[part][1]
        assert len(np.unique(ids)) == len(ids)
        values = np.load(root / f'{part}-features.npy', mmap_mode='r')
        for i in np.linspace(0, len(ids)-1, 128, dtype=int):
            row = rows.iloc[i]
            history = raw[row.stock_index, row.date_index-59:row.date_index+1].copy()
            expected = reference_indicators(history)
            np.testing.assert_allclose(values[i], expected, rtol=1e-5, atol=1e-7, equal_nan=True)
            # Change every later bar AND factor. No earlier feature may move.
            changed = history.copy()
            changed[40:, :4] *= 3
            changed[40:, 6] *= 2
            np.testing.assert_array_equal(values[i, :40], indicator_features(changed[None])[0, :40])
            checked += 1
    center, scale = fit_normalizer(np.load(root / 'training-features.npy', mmap_mode='r'))
    with np.load(root / 'normalizer.npz') as z:
        np.testing.assert_array_equal(center, z['center'])
        np.testing.assert_array_equal(scale, z['scale'])
    return checked, center, scale

def verify(root):
    meta = read(root / 'completed.json')
    if not meta['passed'] or meta['live_promoted']:
        raise ValueError('Expected completed isolated research')
    for name, expected in meta['files'].items():
        assert file_hash(root / name) == expected, name
    for name, expected in read(root / 'sources.json').items():
        assert file_hash(Path(name)) == expected, name
    cfg = read(root / 'protocol.json')
    schema, groups = feature_schema(cfg)
    parity_path = root.parent / (root.name + '-backward-parity.json')
    parity = read(parity_path) if parity_path.exists() else None
    if parity is not None:
        assert file_hash(BASE / 'artifacts/daily-token-live-v1/runs/2026-09-15/manifest.json') == parity['live_manifest_sha256']
    data = Dataset()
    calendar = read(DATA / 'calendar.json')
    if cfg.get('feature_family') == 'indicators':
        raw_rows, center, scale = verify_indicators(root, cfg, data)
    else:
        features = pd.read_parquet(root / 'daily-features.parquet')
        # Reconstruct random historical joins from date-keyed features, not array offsets.
        for part in ['training', 'selection', 'evaluation']:
            ids = np.load(root / f'{part}-ids.npy')
            rows = data.rows.iloc[ids]
            assert rows.date.min() >= cfg[part][0] and rows.label_end.max() < cfg[part][1]
            assert len(np.unique(ids)) == len(ids)
            x = np.load(root / f'{part}-features.npy', mmap_mode='r')
            probe = np.linspace(0, len(ids)-1, 128, dtype=int)
            np.testing.assert_array_equal(x[probe], history_features(rows.iloc[probe], features, calendar))
        training = np.load(root / 'training-features.npy', mmap_mode='r')
        center, scale = fit_normalizer(training)
        with np.load(root / 'normalizer.npz') as z:
            np.testing.assert_array_equal(z['center'], center)
            np.testing.assert_array_equal(z['scale'], scale)
        # Raw provider replay and arithmetic cross-check on captures spanning the archive.
        source_root = BASE / cfg['source']
        index = read(source_root / 'index.json')
        days = sorted(index)
        prices = np.load(DATA / 'quotes.npy', mmap_mode='r')
        stocks = pd.Index(read(DATA / 'instruments.json'))
        features = features.set_index(['instrument_id', 'date'])
        raw_rows = 0
        for j in np.linspace(0, len(days)-1, 4, dtype=int):
            day = days[j]
            raw = replay(source_root / index[day]['capture'])
            for row in raw.itertuples():
                key = (row.instrument_id, row.date)
                if key not in features.index:
                    continue
                i, t = stocks.get_loc(row.instrument_id), calendar.index(row.date)
                reference = prices[i, t]
                expected_t = expected_ep = np.nan
                if np.isfinite(reference[3]) and reference[3] > 0 and abs(reference[3] - row.close) <= .010001:
                    if np.isfinite(row.float_shares) and row.float_shares > 0 and np.isfinite(reference[4]) and reference[4] >= 0:
                        expected_t = np.log1p(reference[4] / row.float_shares * 100)
                    if np.isfinite(row.pe_ttm) and abs(row.pe_ttm) > 1e-6:
                        expected_ep = 1 / row.pe_ttm
                np.testing.assert_allclose(features.loc[key, ['log_turnover_pct', 'earnings_yield']].to_numpy(float),
                                           [expected_t, expected_ep], equal_nan=True, atol=1e-12, rtol=1e-12)
                raw_rows += 1
    torch.set_num_threads(4)
    decoder = load_decoder(read(root / 'profiles.json')['decoder']['path'], cfg['device'])
    ids = np.load(root / 'evaluation-ids.npy')
    x = np.load(root / 'evaluation-features.npy', mmap_mode='r')
    replayed, ranking_checked = 0, 0
    independent = []
    for seed in cfg['seeds']:
        ranks = pd.read_csv(root / f'seed{seed}-reference-ranking.csv')
        for variant in cfg['variants']:
            dest = root / f'seed{seed}' / variant
            model, saved = restore_model(dest / 'model.pt', cfg['device'])
            assert tuple(model.config.auxiliary_features) == groups[variant]
            assert saved['auxiliary_source_mode'] == cfg.get('source_mode', 'retrospective_research')
            columns = [schema.index(name) for name in groups[variant]]
            aux = torch.from_numpy(np.array(x[:cfg['forecast_batch']][..., columns])).to(cfg['device']) if columns else None
            if columns:
                np.testing.assert_array_equal(model.auxiliary_center.cpu().numpy(), center[columns])
                np.testing.assert_array_equal(model.auxiliary_scale.cpu().numpy(), scale[columns])
            batch = ids[:cfg['forecast_batch']]
            a, b, stamps, _ = data.tensors(batch, cfg['device'])
            pair = generate_tokens(model, a[:, :60], b[:, :60], stamps[:, :60], stamps[:, 60:],
                                   samples=cfg['samples'], seed=17, top_p=1., history_auxiliary=aux)
            p, _ = decode_paths(decoder, pair, data.a['mean'][batch], data.a['scale'][batch], 5)
            paths = np.load(dest / 'paths.npy', mmap_mode='r')
            np.testing.assert_array_equal(paths[:len(batch)], p)
            replayed += len(batch)
            for r in ranks[ranks.variant == variant].itertuples():
                i = r.local_row
                legal = valid_bars(paths[i]).all(-1)
                q = paths[i, legal].astype(float)
                counts = [(q[:, 1:, 1].argmax(1) == d).sum() for d in range(4)]
                chosen = counts.index(max(counts)) + 1
                buy, sell = np.quantile(q[:, 0, 2], .5), np.quantile(q[:, chosen, 1], .5)
                assert r.sell_offset == chosen
                np.testing.assert_allclose([r.buy_reference, r.sell_reference, r.predicted, r.actual_extrema_scenario],
                    [buy, sell, sell/buy-1-cfg['cost'], data.a['future'][ids[i], chosen, 1]/data.a['future'][ids[i], 0, 2]-1-cfg['cost']],
                    rtol=1e-12, atol=1e-12)
                ranking_checked += 1
            for h in [2, 5]:
                for i in range(len(ids)):
                    q = paths[i, valid_bars(paths[i, :, :h]).all(-1), :h].astype(float)
                    if len(q) < 16 or not data.a['valid'][ids[i], :h].all():
                        continue
                    ref = float(data.a['last'][ids[i], 3])
                    actual = data.a['future'][ids[i], :h]
                    hi, lo = np.median(q[:, :, 1].max(1)), np.median(q[:, :, 2].min(1))
                    independent.append(dict(seed=seed, variant=variant, local_row=i, horizon=h,
                        maximum=abs(hi-actual[:, 1].max())/ref*100,
                        minimum=abs(lo-actual[:, 2].min())/ref*100))
            del model
    independent = pd.DataFrame(independent)
    # All arms must use the same valid row cohort within each horizon.
    count = independent.groupby(['seed', 'local_row', 'horizon']).variant.nunique()
    keys = count[count == len(cfg['variants'])].reset_index()[['seed', 'local_row', 'horizon']]
    independent = independent.merge(keys, on=['seed', 'local_row', 'horizon'])
    independent['date'] = data.rows.iloc[ids[independent.local_row]].date.to_numpy()
    own = independent.groupby(['seed', 'variant', 'date', 'horizon'])[['maximum', 'minimum']].mean().stack().rename('mae')
    own.index = own.index.set_names(['seed', 'variant', 'date', 'horizon', 'target'])
    stored = pd.read_csv(root / 'daily-scores.csv')
    stored = stored[stored.target.isin(['maximum', 'minimum'])].set_index(['seed', 'variant', 'date', 'horizon', 'target']).mae
    np.testing.assert_allclose(own.sort_index(), stored.sort_index(), atol=1e-6, rtol=1e-6)
    # Independently aggregate the checked per-stock ranking into daily top groups.
    ranking_daily = []
    for seed in cfg['seeds']:
        frame = pd.read_csv(root / f'seed{seed}-reference-ranking.csv')
        for (variant, day), group in frame.groupby(['variant', 'date']):
            selected = group.sort_values(['predicted', 'instrument_id'], ascending=[False, True]).head(math.ceil(len(group)*.2))
            ranking_daily.append(dict(seed=seed, variant=variant, date=day,
                return_mae_pp=float(np.abs(group.predicted-group.actual_extrema_scenario).mean()*100),
                top_extrema_scenario_pct=float(selected.actual_extrema_scenario.mean()*100),
                top_lift_pp=float((selected.actual_extrema_scenario.mean()-group.actual_extrema_scenario.mean())*100)))
    computed_ranking = pd.DataFrame(ranking_daily).set_index(['seed', 'variant', 'date']).sort_index()
    stored_ranking = pd.read_csv(root / 'daily-ranking.csv').set_index(['seed', 'variant', 'date']).sort_index()
    np.testing.assert_allclose(computed_ranking, stored_ranking[computed_ranking.columns], atol=1e-10, rtol=1e-10)
    paired_ranking = []
    for variant in cfg['variants'][1:]:
        a = computed_ranking.xs(variant, level='variant').groupby('date').mean()
        b = computed_ranking.xs('baseline', level='variant').groupby('date').mean()
        for metric in computed_ranking.columns:
            delta = (b[metric]-a[metric]) if metric == 'return_mae_pp' else (a[metric]-b[metric])
            if not np.isfinite(delta).all():
                raise ValueError('Ranking dates are not paired')
            rng = np.random.default_rng(314159)
            n = len(delta)
            block = min(cfg['bootstrap_block_dates'], n)
            bootstrap = []
            for _ in range(cfg['bootstrap_replicates']):
                starts = rng.integers(0, n, size=math.ceil(n/block))
                ix = ((starts[:, None]+np.arange(block)) % n).ravel()[:n]
                bootstrap.append(float(delta.to_numpy()[ix].mean()))
            paired_ranking.append(dict(variant=variant, metric=metric, improvement_pp=float(delta.mean()),
                improved_date_fraction=float((delta > 1e-10).mean()), dates=n, tie_tolerance_pp=1e-10,
                bootstrap_95_low=float(np.quantile(bootstrap, .025)), bootstrap_95_high=float(np.quantile(bootstrap, .975))))
    write_json(root / 'verified-ranking-comparison.json', paired_ranking)
    result = dict(passed=True, at=utc_now(), feature_family=cfg.get('feature_family', 'fundamental'), independently_checked_feature_samples=raw_rows, exact_replay_inputs=replayed,
                  reference_ranking_rows=ranking_checked, independent_daily_extrema_scores=len(own),
                  verifier_sha256=file_hash(Path(__file__)), ranking_comparison_sha256=file_hash(root / 'verified-ranking-comparison.json'),
                  no_live_promotion=True, live_manifest_unchanged=(root / 'live-state.json').exists() or parity is not None, verified_completed_sha256=file_hash(root / 'completed.json'))
    write_json(root / 'verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=BASE / 'artifacts/token-features-20260916-v1')
    verify(parser.parse_args().root)
