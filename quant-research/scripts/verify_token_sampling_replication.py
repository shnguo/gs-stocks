"""Independent saved-path metrics, cohorts, seed schedule, chunks and reload replay."""
import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from token_history_run import Dataset
from token_sampling_replication import (
    DATA,
    DIAG,
    DRAWS,
    METRICS,
    OWNERS,
    ROOT,
    SEEDS,
    checked,
    folder,
    read,
)
from tokenizer_reconstruction_run import load_decoder, verify

from quant_research.storage import file_hash, utc_now, write_json
from quant_research.token_transformer import decode_paths, generate_tokens, restore_model


def legal(p):
    return (np.isfinite(p).all(-1) & (p[..., :4] > 0).all(-1) & (p[..., 4:] >= 0).all(-1)
        & (p[..., 1] >= p[..., 0]) & (p[..., 1] >= p[..., 2]) & (p[..., 1] >= p[..., 3])
        & (p[..., 2] <= p[..., 0]) & (p[..., 2] <= p[..., 3])).all(-1)


def equal(a, b, keys):
    pd.testing.assert_frame_equal(a[b.columns].sort_values(keys).reset_index(drop=True),
        b.sort_values(keys).reset_index(drop=True), check_exact=False, check_dtype=False, atol=1e-10, rtol=1e-9)


def replay():
    checked()
    torch.set_num_threads(4)
    data = Dataset()
    ids = np.load(ROOT / 'row-ids.npy')
    profiles = read(ROOT / 'profiles.json')
    total_chunks = replays = 0
    rng_ranges = [{draw+start for start in range(0, len(ids), 4)} for draw in DRAWS]
    for i, left in enumerate(rng_ranges):
        for right in rng_ranges[i+1:]:
            assert not left & right
    for draw in DRAWS:
        for seed in SEEDS:
            for owner in OWNERS:
                dest = folder(draw, seed, owner)
                verify(dest)
                lineage = read(dest / 'lineage.json')
                checkpoint = profiles[owner]['checkpoints'][str(seed)]
                assert lineage['checkpoint_sha256'] == checkpoint['sha256']
                assert lineage['decoder_sha256'] == profiles['decoder']['sha256']
                assert lineage['generation_seed'] == draw and lineage['samples'] == 64
                assert lineage['sampling'] == dict(temperature=1., top_p=1., top_k=0)
                assert set(lineage['chunks']) == {f'{i:06d}.npz' for i in range(0, len(ids), 4)}
                paths = np.load(dest / 'paths.npy', mmap_mode='r')
                assert paths.shape == (2432, 64, 5, 6)
                for name, digest in lineage['chunks'].items():
                    path = dest / 'chunks' / name
                    assert file_hash(path) == digest
                    offset = int(path.stem)
                    with np.load(path) as z:
                        np.testing.assert_array_equal(z['row_ids'], ids[offset:offset+4])
                        np.testing.assert_array_equal(z['paths'], paths[offset:offset+4])
                        np.testing.assert_array_equal(z['valid'], legal(z['paths']))
                        assert z['s1'].shape == z['s2'].shape == (4, 64, 5)
                        for key in ['s1', 's2']:
                            assert ((z[key] >= 0) & (z[key] < 1024)).all()
                    total_chunks += 1
                model, saved = restore_model(Path(checkpoint['path']), 'mps')
                assert saved['dataset_manifest_sha256'] == file_hash(DATA / 'completed.json')
                model.eval()
                decoder = load_decoder(Path(profiles['decoder']['path']))
                # First and last batches cover both ends of the RNG schedule for new replicates.
                for start in ([0] if draw == 17 else [0, len(ids)-4]):
                    batch = ids[start:start+4]
                    a, b, s, _ = data.tensors(batch, 'mps')
                    tokens = generate_tokens(model, a[:, :60], b[:, :60], s[:, :60], s[:, 60:],
                        samples=64, seed=draw+start, temperature=1., top_p=1., top_k=0)
                    p, v = decode_paths(decoder, tokens, data.a['mean'][batch], data.a['scale'][batch], 5)
                    with np.load(dest / 'chunks' / f'{start:06d}.npz') as z:
                        np.testing.assert_array_equal(p, z['paths'])
                        np.testing.assert_array_equal(v, z['valid'])
                        for key, token in zip(['s1', 's2'], tokens):
                            np.testing.assert_array_equal(z[key], token[:, :, -5:].cpu().numpy())
                    replays += 1
                del model, decoder
                gc.collect()
                torch.mps.empty_cache()
                print('Reload and chunks passed', draw, seed, owner, flush=True)
    checked()
    write_json(ROOT / 'replay-verification.json', dict(passed=True, at=utc_now(), chunks=total_chunks,
        exact_replays=replays, replayed_paths=replays*256, disjoint_draw_seed_ranges=True))


def arithmetic():
    checked()
    scoring = read(ROOT / 'scoring-completed.json')
    assert scoring['passed']
    for name, digest in scoring['files'].items():
        assert file_hash(ROOT / name) == digest
    ids = np.load(ROOT / 'row-ids.npy')
    meta = pd.read_parquet(DATA / 'rows.parquet')
    assert meta.iloc[ids].date.nunique() == 76 and meta.iloc[ids].label_end.max() == '2025-07-29' < '2025-08-07'
    future, known, last = [np.load(DATA / f'{name}.npy', mmap_mode='r') for name in ['future', 'valid', 'last']]
    records, coverage = [], []
    for draw in DRAWS:
        for seed in SEEDS:
            for owner in OWNERS:
                p = np.load(folder(draw, seed, owner) / 'paths.npy', mmap_mode='r')
                for h in [2, 5]:
                    mask = legal(p[:, :, :h])
                    use = known[ids, :h].all(1) & (mask.sum(1) >= 16)
                    coverage.append(dict(draw=draw, seed=seed, owner=owner, horizon=h, inputs=len(ids),
                        known=known[ids, :h].all(1).sum(), usable=use.sum(), legal_paths=mask.sum(), total_paths=mask.size))
                    selected = np.flatnonzero(use)
                    for start in range(0, len(selected), 64):
                        ix = selected[start:start+64]
                        ref = last[ids[ix], 3].astype(float)
                        high = 100*(p[ix, :, :h, 1].astype(float).max(-1)/ref[:, None]-1)
                        low = 100*(p[ix, :, :h, 2].astype(float).min(-1)/ref[:, None]-1)
                        valid = mask[ix]
                        mh = np.nanquantile(np.where(valid, high, np.nan), .5, axis=1)
                        ml = np.nanquantile(np.where(valid, low, np.nan), .5, axis=1)
                        yh = 100*(future[ids[ix], :h, 1].max(-1)/ref-1)
                        yl = 100*(future[ids[ix], :h, 2].min(-1)/ref-1)
                        center, span = (high+low)/2, high-low
                        mid = np.nanquantile(np.where(valid, center, np.nan), .5, axis=1)
                        width = np.nanquantile(np.where(valid, span, np.nan), .5, axis=1)
                        n = valid.sum(1)
                        first = np.where(valid, abs(center-(yh+yl)[:, None]/2), 0).sum(1)/n
                        distances = abs(center[:, :, None]-center[:, None, :])
                        pairs = valid[:, :, None] & valid[:, None, :]
                        crps = first-np.where(pairs, distances, 0).sum((1, 2))/(2*n**2)
                        ce = ((mh-yh)+(ml-yl))/2
                        we = ((mh-yh)-(ml-yl))/2
                        records.append(pd.DataFrame(dict(draw=draw, seed=seed, owner=owner, horizon=h,
                            row_id=ids[ix], local_row=ix, date=meta.iloc[ids[ix]].date.to_numpy(),
                            high_median=mh, low_median=ml, actual_high=yh, actual_low=yl,
                            endpoint_mae=np.maximum(abs(ce), abs(we)), high_mae=abs(mh-yh), low_mae=abs(ml-yl),
                            center_bias=ce, half_width_bias=we, midpoint_mae=abs(mid-(yh+yl)/2), midpoint_crps=crps,
                            range_mae=abs(width-yh+yl), legal_fraction=n/64)))
                print('Raw arithmetic passed', draw, seed, owner, flush=True)
    f = pd.concat(records, ignore_index=True)
    keys = ['draw', 'seed', 'owner', 'horizon', 'row_id']
    assert not f.duplicated(keys).any()
    equal(f, pd.read_parquet(ROOT / 'all-scores.parquet'), keys)
    equal(pd.DataFrame(coverage), pd.read_csv(ROOT / 'coverage.csv'), keys[:-1])
    old = pd.read_parquet(DIAG / 'rows.parquet').query('window == "2025transfer"')
    old = old.rename(columns={'center_error': 'center_bias', 'half_width_error': 'half_width_bias',
        'path_center_mae': 'midpoint_mae', 'path_center_crps': 'midpoint_crps', 'path_range_mae': 'range_mae'})
    old = old[['seed', 'owner', 'horizon', 'row_id', *METRICS]].assign(draw=17)
    equal(f.merge(old[keys], on=keys, validate='one_to_one'), old, keys)
    cohorts = {}
    for (draw, seed, h), g in f.groupby(['draw', 'seed', 'horizon']):
        owners = [set(x.row_id) for _, x in g.groupby('owner')]
        cohorts[(draw, seed, h)] = set.intersection(*owners)
    all27 = {h: set.intersection(*[v for k, v in cohorts.items() if k[2] == h]) for h in [2, 5]}
    primary = f[f.apply(lambda r: r.row_id in all27[r.horizon], axis=1)].assign(cohort='all27')
    local = f[f.apply(lambda r: r.row_id in cohorts[(r.draw, r.seed, r.horizon)], axis=1)].assign(cohort='within_draw_seed')
    expected = pd.concat([primary, local], ignore_index=True)
    equal(expected, pd.read_parquet(ROOT / 'common.parquet'), ['cohort', *keys])
    support = expected.groupby(['cohort', 'draw', 'seed', 'horizon']).row_id.nunique().rename('rows').reset_index()
    equal(support, pd.read_csv(ROOT / 'support.csv'), ['cohort', 'draw', 'seed', 'horizon'])
    # Build all averages directly from independently reconstructed rows.
    by_date = expected.groupby(['cohort', 'draw', 'seed', 'owner', 'horizon', 'date'])[METRICS].mean().reset_index()
    draw_average = by_date.groupby(['cohort', 'seed', 'owner', 'horizon', 'date'])[METRICS].mean().reset_index().assign(draw=0)
    combined = pd.concat([by_date, draw_average], ignore_index=True)
    seed_average = combined.groupby(['cohort', 'draw', 'owner', 'horizon', 'date'])[METRICS].mean().reset_index().assign(seed=0)
    combined = pd.concat([combined, seed_average], ignore_index=True)
    keys = ['cohort', 'draw', 'seed', 'owner', 'horizon', 'date']
    equal(combined, pd.read_csv(ROOT / 'daily.csv'), keys)
    equal(combined.groupby(keys[:-1])[METRICS].mean().reset_index(), pd.read_csv(ROOT / 'summary.csv'), keys[:-1])
    grouped = {k: g for k, g in combined.groupby(['cohort', 'draw', 'seed', 'horizon'])}
    estimates = pd.read_csv(ROOT / 'paired.csv')
    for row in estimates.itertuples():
        g = grouped[(row.cohort, row.draw, row.seed, row.horizon)]
        new, base = row.comparison.split('_vs_')
        a = g[g.owner == base].sort_values('date')[row.metric].to_numpy()
        b = g[g.owner == new].sort_values('date')[row.metric].to_numpy()
        assert len(a) == len(b) == row.dates == 76
        starts = np.random.default_rng(314159).integers(len(a), size=(2000, (len(a)+9)//10))
        index = np.stack([(starts+j) % len(a) for j in range(10)], axis=-1).reshape(2000, -1)[:, :len(a)]
        lo, hi = np.quantile((b-a)[index].mean(1), [.025, .975])
        np.testing.assert_allclose([row.baseline, row.candidate, row.delta, row.ci_low, row.ci_high],
            [a.mean(), b.mean(), (b-a).mean(), lo, hi], atol=1e-10, rtol=1e-9)
        if row.metric not in ['center_bias', 'half_width_bias']:
            np.testing.assert_allclose(row.relative_change_pct, (b.mean()/a.mean()-1)*100, atol=1e-10)
        else:
            assert np.isnan(row.relative_change_pct)
    assert len(estimates) == 2*4*4*2*3*len(METRICS)
    original_cohorts = {h: set.intersection(*[cohorts[(17, s, h)] for s in SEEDS]) for h in [2, 5]}
    checked()
    write_json(ROOT / 'arithmetic-verification.json', dict(passed=True, at=utc_now(), score_rows=len(f),
        common_score_rows=len(expected), paired_estimates=len(estimates), prior_score_rows=len(old),
        all27_rows={str(h): len(v) for h, v in all27.items()},
        original_all9_rows={str(h): len(v) for h, v in original_cohorts.items()},
        newly_excluded={str(h): len(original_cohorts[h]-all27[h]) for h in [2, 5]},
        seed_and_draw_averaging_verified=True, sealed_holdout_used=False))
    print('Independent arithmetic passed', len(f), 'scores;', len(estimates), 'estimates', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['replay', 'arithmetic'])
    globals()[parser.parse_args().action]()
