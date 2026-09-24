"""Independent first-run token watchlist, chronology and source verification."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from quant_research.daily_loop import read, verify
from quant_research.kronos_ranker import timestamps
from quant_research.storage import file_hash, utc_now, write_json
from quant_research.token_transformer import (
    decode_paths,
    generate_tokens,
    load_tokenizer,
    restore_model,
)


def legal(p):
    return (np.isfinite(p).all(-1) & (p[..., :4] > 0).all(-1) & (p[..., 4:] >= 0).all(-1)
        & (p[..., 1] >= p[..., 0]) & (p[..., 1] >= p[..., 2]) & (p[..., 1] >= p[..., 3])
        & (p[..., 2] <= p[..., 0]) & (p[..., 2] <= p[..., 3])).all(-1)


def main(run):
    run = Path(run)
    verify(run)
    cfg = read(run / 'binding.json')['config']
    info = read(run / 'run.json')
    input_info = read(run / 'inputs/input.json')
    verify(run / 'inputs')
    source = Path(input_info['source'])
    verify(source / 'snapshot')
    verify(source / 'panel-h5')
    assert file_hash(source / 'snapshot/manifest.json') == input_info['snapshot_sha256']
    panel = read(source / 'panel-h5/manifest.json')
    dates, signal = panel['dates'], info['signal_date']
    cutoff = dates.index(signal)
    assert info['horizon_dates'] == dates[cutoff+1:cutoff+6]
    train = pd.read_parquet(run / 'inputs/training-rows.parquet')
    assert len(train) == input_info['training_rows'] and train.label_end.max() <= signal
    mature = train[train.role != 'historical_replay']
    for r in mature.itertuples():
        assert dates.index(r.label_end) == dates.index(r.date)+5
    rows = pd.read_parquet(run / 'inputs/forecast-rows.parquet')
    bars = pd.read_parquet(source / 'snapshot/bars.parquet')
    groups = {s: g.set_index('date') for s, g in bars.groupby('instrument_id')}
    training_arrays = {k: np.load(run / f'inputs/train-{k}.npy', mmap_mode='r') for k in
                       ['future', 'mean', 'scale', 'last', 'stamps', 's1', 's2', 'valid']}
    assert training_arrays['valid'].all()
    for r in mature.itertuples():
        t = dates.index(r.date)
        g = groups[r.instrument_id].reindex(dates[t-59:t+6])
        q = g[['open', 'high', 'low', 'close', 'volume', 'amount', 'factor']].to_numpy(float)
        assert len(q) == 65 and np.isfinite(q).all()
        assert np.isclose(q[59:, 6], q[59, 6], rtol=1e-8, atol=0).all()
        assert all(g[k].iloc[59:].nunique() == 1 for k in ['sequence_id', 'label_sequence_id'])
        values = q[:60, :6].astype(np.float32)
        values[:, :4] *= q[:60, 6:7]/q[59:60, 6:7]
        np.testing.assert_array_equal(training_arrays['future'][r.Index], q[60:, :6])
        np.testing.assert_allclose(training_arrays['mean'][r.Index, 0], values.mean(0), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(training_arrays['scale'][r.Index, 0], values.std(0)+1e-5, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(training_arrays['last'][r.Index], q[59, :6])
        np.testing.assert_array_equal(training_arrays['stamps'][r.Index], timestamps(dates[t-59:t+6]))
    old = Path(cfg['historical_replay'])
    replay_ids = np.load(run / 'inputs/historical-replay-ids.npy')
    replay_positions = np.flatnonzero(train.role.eq('historical_replay'))
    old_rows = pd.read_parquet(old / 'rows.parquet').iloc[replay_ids]
    assert old_rows.label_end.max() < '2024-01-01'
    for key in ['future', 'mean', 'scale', 'last', 's1', 's2', 'valid']:
        expected = np.load(old / f'{key}.npy', mmap_mode='r')[replay_ids]
        np.testing.assert_array_equal(training_arrays[key][replay_positions], expected)
    mean, scale, last = [np.load(run / f'inputs/forecast-{k}.npy') for k in ['mean', 'scale', 'last']]
    for i, row in enumerate(rows.itertuples()):
        block = groups[row.instrument_id].reindex(dates[cutoff-59:cutoff+1])
        raw = block[['open', 'high', 'low', 'close', 'volume', 'amount', 'factor']].to_numpy(float)
        assert np.isfinite(raw).all() and (raw[:, 6] > 0).all() and legal(raw[:, :6][None]).all()
        values = raw[:, :6].astype(np.float32)
        values[:, :4] *= raw[:, 6:7]/raw[-1:, 6:7]
        np.testing.assert_allclose(mean[i, 0], values.mean(0), atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(scale[i, 0], values.std(0)+1e-5, atol=1e-6, rtol=1e-6)
        np.testing.assert_array_equal(last[i], raw[-1, :6])
    a, b = [np.load(run / f'inputs/forecast-{k}.npy') for k in ['s1', 's2']]
    stamps = np.load(run / 'inputs/forecast-stamps.npy')
    np.testing.assert_array_equal(stamps, timestamps(dates[cutoff-59:cutoff+6]))
    profiles = read(run / 'binding.json')
    torch.set_num_threads(4)
    decoder = load_tokenizer(Path(cfg['tokenizer_bundle']), cfg['device'])
    decoder.load_state_dict(torch.load(profiles['decoder']['path'], map_location='cpu', weights_only=True)['state_dict'])
    reconstructed = []
    quantiles = []
    chunks = replays = 0
    for seed in cfg['seeds']:
        model_path = run / f'models/seed{seed}/model.pt'
        assert file_hash(model_path) == info['checkpoints'][str(seed)]['sha256']
        assert file_hash(Path(profiles['checkpoints'][str(seed)]['path'])) == profiles['checkpoints'][str(seed)]['sha256']
        training = read(run / f'models/seed{seed}/training.json')
        assert training['rows'] == len(train) and training['training_labels_through'] <= signal and training['reload_exact']
        paths = np.load(run / f'forecasts/seed{seed}/paths.npy', mmap_mode='r')
        assert paths.shape == (len(rows), 64, 5, 6)
        for path in sorted((run / f'forecasts/seed{seed}/chunks').glob('*.npz')):
            start = int(path.stem)
            with np.load(path) as z:
                np.testing.assert_array_equal(z['paths'], paths[start:start+len(z['paths'])])
                np.testing.assert_array_equal(z['valid'], legal(z['paths']))
            chunks += 1
        for i, row in enumerate(rows.itertuples()):
            masks = {h: legal(paths[i, :, :h]) for h in [2, 5]}
            if any(mask.sum() < 16 for mask in masks.values()):
                continue
            for h, mask in masks.items():
                x = paths[i, mask, :h].astype(float)
                entry = x[:, 0, 2] if info.get('ranking_method') == 't_low_to_future_high' else x[:, 0, 0]
                exit_price = x[:, 1:, 1].max(1) if info.get('ranking_method') == 't_low_to_future_high' else x[:, -1, 3]
                net = (exit_price-entry)/entry-cfg['round_trip_cost_scenario']
                qs = np.quantile(net, [.1, .5, .9])
                reconstructed.append(dict(instrument_id=row.instrument_id, name=row.name, seed=seed, horizon=h,
                    entry_median=np.median(entry), exit_median=np.median(exit_price), expected_net_return=net.mean(), positive_fraction=np.count_nonzero(net > 0)/len(net),
                    return_q10=qs[0], return_q50=qs[1], return_q90=qs[2],
                    upside_median=np.quantile((x[:, :, 1].max(1)-entry)/entry, .5),
                    downside_q10=np.quantile((x[:, :, 2].min(1)-entry)/entry, .1), valid_paths=len(x),
                    high_median=np.quantile(x[:, :, 1].max(1), .5), low_median=np.quantile(x[:, :, 2].min(1), .5)))
            q = np.quantile(paths[i, masks[5]].astype(float), [.1, .5, .9], axis=0)
            for d in range(5):
                for j, name in enumerate(['open', 'high', 'low', 'close', 'volume', 'amount']):
                    quantiles.append(dict(instrument_id=row.instrument_id, seed=seed, day=d, field=name, q10=q[0, d, j], q50=q[1, d, j], q90=q[2, d, j]))
        model, _ = restore_model(model_path, cfg['device'])
        for start in [0, ((len(rows)-1)//4)*4]:
            count = min(4, len(rows)-start)
            aa, bb = [torch.as_tensor(x[start:start+count], dtype=torch.long, device=cfg['device']) for x in [a, b]]
            stamp = torch.as_tensor(stamps, dtype=torch.long, device=cfg['device'])[None].expand(count, -1, -1)
            tokens = generate_tokens(model, aa, bb, stamp[:, :60], stamp[:, 60:], samples=64,
                seed=cfg['sampling_seed']+start, temperature=1., top_p=1., top_k=0)
            p, _ = decode_paths(decoder, tokens, mean[start:start+count], scale[start:start+count], 5)
            np.testing.assert_array_equal(p, paths[start:start+count])
            with np.load(run / f'forecasts/seed{seed}/chunks/{start:06d}.npz') as z:
                for key, x in zip(['s1', 's2'], tokens):
                    np.testing.assert_array_equal(z[key], x[:, :, -5:].cpu().numpy())
            replays += 1
    f = pd.DataFrame(reconstructed)
    saved = pd.read_parquet(run / 'model-scores.parquet')
    keys = ['instrument_id', 'seed', 'horizon']
    pd.testing.assert_frame_equal(f[saved.columns].sort_values(keys).reset_index(drop=True), saved.sort_values(keys).reset_index(drop=True),
        check_dtype=False, check_exact=False, atol=1e-10, rtol=1e-9)
    saved = pd.read_parquet(run / 'daily-quantiles.parquet')
    keys = ['instrument_id', 'seed', 'day', 'field']
    pd.testing.assert_frame_equal(pd.DataFrame(quantiles)[saved.columns].sort_values(keys).reset_index(drop=True),
        saved.sort_values(keys).reset_index(drop=True), check_dtype=False)
    common = f.groupby(['instrument_id', 'horizon']).seed.transform('nunique').eq(3)
    f = f[common & f.horizon.eq(5)]
    expected = f.groupby('instrument_id').expected_net_return.mean().sort_values(ascending=False)
    ranks = pd.read_csv(run / 'ranking.csv')
    assert ranks.instrument_id.tolist() == expected.index.tolist()
    np.testing.assert_allclose(ranks.expected_net_return, expected, atol=1e-10)
    assert ranks['rank'].tolist() == list(range(1, len(ranks)+1))
    summary = dict(passed=True, at=utc_now(), forecast_inputs=len(rows), training_rows=len(train),
        training_labels_through=train.label_end.max(), model_score_rows=len(reconstructed), quantile_rows=len(quantiles),
        ranked=len(ranks), chunks=chunks, exact_replays=replays, models=3, source_chronology_verified=True,
        no_orders=True, prospective=info['prospective'], run_manifest_sha256=file_hash(run / 'manifest.json'))
    write_json(run.parent.parent / 'verification' / (run.name+'.json'), summary)
    print(summary, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    main(p.parse_args().run)
