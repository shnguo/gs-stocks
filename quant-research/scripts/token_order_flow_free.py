"""Matched Transformer continuation using free size-tier net flow features."""
import argparse
import fcntl
import gc
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from token_ranking_incremental import grouped_batches
from tokenizer_reconstruction_run import load_decoder

from quant_research.daily_loop import digest, read, verify, write_manifest
from quant_research.daily_token import FIELDS, future_known, panel
from quant_research.kronos_ranker import timestamps
from quant_research.midpoint_loss import sampled_midpoint_loss
from quant_research.order_flow_features import FEATURES, expand_model, size_features
from quant_research.return_ranking_loss import sampled_return_ranking_loss
from quant_research.storage import file_hash, utc_now, write_json
from quant_research.token_cached_inference import decode_cached, generate_cached
from quant_research.token_features import fit_normalizer
from quant_research.token_history import loss_parts
from quant_research.token_indicators import FEATURES as INDICATORS
from quant_research.token_indicators import indicator_features
from quant_research.token_transformer import (
    checkpoint_payload,
    forecast_auxiliary,
    load_tokenizer,
    normalize_training,
    restore_model,
    valid_bars,
)

BASE = Path(__file__).resolve().parents[1]


def prepare(root, cfg):
    source = BASE/cfg['price_source']
    summary = read(root/'source-summary.json')
    if summary['stocks'] != cfg['stocks']:
        raise ValueError('Incomplete source cohort')
    files = [source/'snapshot/manifest.json', source/'panel-h5/manifest.json',
             root/'protocol.json', root/'source-summary.json', root/'universe.json']
    profiles = read(BASE/cfg['parent']/'profiles.json')
    files += [Path(profiles['decoder']['path'])]
    if file_hash(files[-1]) != profiles['decoder']['sha256']:
        raise ValueError('Decoder changed')
    for seed in cfg['seeds']:
        parent = BASE/cfg['parent']/f'seed{seed}'/'indicators_ranking'
        verify(parent, 'trained.json')
        files.append(parent/'model.pt')
    code = [Path(__file__), BASE/'scripts/finalize_order_flow_free.py',
            BASE/'scripts/collect_order_flow_free.py', BASE/'scripts/finalize_token_ranking.py',
            BASE/'scripts/finalize_token_three_ideas.py', BASE/'scripts/verify_token_ranking.py',
            *sorted((BASE/'src/quant_research').glob('*.py'))]
    binding = dict(config=cfg, sources={str(p): file_hash(p) for p in files},
                   code={str(p): file_hash(p) for p in code})
    if (root/'binding.json').exists() and read(root/'binding.json') != binding:
        raise ValueError('Changed experiment inputs or code; use new output')
    write_json(root/'binding.json', binding)
    dest = root/'inputs'
    if (dest/'manifest.json').exists():
        verify(dest)
        return
    dest.mkdir(exist_ok=True)
    for p in code:
        out = root/'code'/p.relative_to(BASE)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
    paths = [BASE/'configs/daily-token-v1.json', BASE/'configs/token-ranking-daily-v1.json',
             BASE/'artifacts/daily-token-live-v1/current.json', BASE/'artifacts/daily-token-live-v1/latest.json',
             BASE/'artifacts/token-ranking-daily-v1/current.json']
    write_json(root/'live-state.json', {str(p): file_hash(p) for p in paths if p.exists()})
    meta, bars, _ = panel(source)
    dates = [d for d in meta['dates'] if d <= meta['price_data_through']]
    symbols = read(root/'universe.json')['symbols']
    histories, flow = {}, {}
    for symbol in symbols:
        folder = root/'source'/symbol
        if file_hash(folder/'manifest.json') != summary['source_manifests'][symbol]:
            raise ValueError('Changed flow capture')
        verify(folder)
        frame = pd.read_parquet(folder/'flows.parquet') if (folder/'flows.parquet').exists() else pd.DataFrame(columns=['date', *FEATURES])
        if 'small_net' not in frame:
            from quant_research.order_flow_features import TIERS
            frame = pd.DataFrame(columns=['date', *TIERS])
        g = bars[bars.instrument_id.eq(symbol)].set_index('date').reindex(dates)
        histories[symbol] = g
        flow[symbol] = size_features(frame, g.amount, dates, cfg['flow_lag_sessions']).to_numpy()
    records = []
    raw, target, known, extra = [], [], [], []
    for part in ['training', 'evaluation']:
        begin, boundary = cfg[part]
        chosen_days = [d for d in dates if begin <= d < boundary][::cfg['date_stride']]
        for day in chosen_days:
            t = dates.index(day)
            if t < 59 or t+5 >= len(dates) or (part == 'training' and dates[t+5] >= boundary):
                continue
            for symbol in symbols:
                g = histories[symbol]
                h = g.iloc[t-59:t+1][FIELDS].to_numpy(float)
                if not np.isfinite(h).all() or not valid_bars(h[:, :6]).all() or (h[:, 6] <= 0).any() or (h[-1, 4:6] <= 0).any():
                    continue
                full = future_known(g, t, t+5)
                if part == 'training' and not full:
                    continue
                raw.append(h)
                target.append(g.iloc[t+1:t+6][FIELDS[:6]].to_numpy(float))
                known.append([full]*5)
                extra.append(flow[symbol][t])
                records.append(dict(instrument_id=symbol, date=day, label_end=dates[t+5],
                    part=part, date_index=t, row_id=len(records)))
    rows = pd.DataFrame(records)
    if any(rows[rows.part.eq(p)].date.nunique() < 15 for p in ['training', 'evaluation']):
        raise ValueError('Insufficient independent signal dates')
    raw, target, known, extra = np.stack(raw), np.stack(target), np.array(known), np.stack(extra)
    ix = np.flatnonzero(rows.part.eq('training'))
    center, scale = fit_normalizer(extra[ix, None])
    if np.isfinite(extra[ix]).mean() < .4:
        raise ValueError('Insufficient size-flow training coverage')
    np.savez(dest/'normalizer.npz', center=center, scale=scale)
    rows.to_parquet(dest/'rows.parquet', index=False)
    write_json(dest/'calendar.json', dates)
    tokenizer = load_tokenizer(BASE/cfg['bundle'], cfg['device'])
    parts = {k: [] for k in ['s1', 's2', 'mean', 'scale', 'stamps', 'valid', 'features', 'volatility']}
    parity_rows = 0
    for start in range(0, len(rows), 256):
        sl = slice(start, start+256)
        h, y, mask = raw[sl], target[sl], known[sl]
        x, valid, mean, std = normalize_training(h, y, mask)
        with torch.inference_mode():
            a, b = tokenizer.encode(torch.tensor(x, device=cfg['device']), half=True)
            # Verify forecast prefix does not depend on any future OHLCVA label.
            hidden = x.copy()
            hidden[:, 60:] = 0
            aa, bb = tokenizer.encode(torch.tensor(hidden, device=cfg['device']), half=True)
            torch.testing.assert_close(a[:, :60], aa[:, :60], atol=0, rtol=0)
            torch.testing.assert_close(b[:, :60], bb[:, :60], atol=0, rtol=0)
        parity_rows += len(h)
        aux = np.full((len(h), 60, len(INDICATORS)+len(FEATURES)), np.nan, np.float32)
        aux[:, :, :len(INDICATORS)] = indicator_features(h)
        aux[:, -1, len(INDICATORS):] = extra[sl]
        adjusted = h[:, -21:, 3]*h[:, -21:, 6]
        values = dict(s1=a.cpu().numpy(), s2=b.cpu().numpy(), valid=valid, mean=mean, scale=std,
            stamps=np.stack([timestamps(dates[t-59:t+6]) for t in rows.iloc[sl].date_index]),
            features=aux, volatility=np.diff(np.log(adjusted), axis=1).std(1)*100)
        for key, value in values.items():
            parts[key].append(value)
    for key, values in parts.items():
        np.save(dest/f'{key}.npy', np.concatenate(values))
    np.save(dest/'future.npy', target)
    np.save(dest/'last.npy', raw[:, -1, :6])
    # Used by the independent feature audit; exact adjusted histories retained.
    np.save(dest/'raw-history.npy', raw)
    write_json(dest/'coverage.json', dict(rows=len(rows), prefix_parity_rows=parity_rows,
        parts={p: dict(rows=int(rows.part.eq(p).sum()), dates=int(rows[rows.part.eq(p)].date.nunique()),
                       first=rows[rows.part.eq(p)].date.min(), last=rows[rows.part.eq(p)].date.max(),
                       label_end=rows[rows.part.eq(p)].label_end.max()) for p in ['training', 'evaluation']},
        flow_feature_coverage={n: float(np.isfinite(extra[:, j]).mean()) for j, n in enumerate(FEATURES)}))
    write_manifest(dest)
    del tokenizer
    gc.collect()
    torch.mps.empty_cache()
    print('Prepared', read(dest/'coverage.json'), flush=True)


def arrays(root):
    return {p.stem: np.load(p, mmap_mode='r') for p in (root/'inputs').glob('*.npy')}


def tensors(a, ids, arm, device):
    ts = [torch.tensor(np.array(a[k][ids]), dtype=torch.long, device=device) for k in ['s1', 's2', 'stamps']]
    aux = np.array(a['features'][ids])
    if arm == 'control':
        aux[:, :, len(INDICATORS):] = np.nan
    return *ts, torch.tensor(np.array(a['valid'][ids]), device=device), torch.tensor(aux, device=device)


def train(root, cfg, seed, arm, decoder):
    dest = root/f'seed{seed}'/arm/'training'
    if (dest/'manifest.json').exists():
        verify(dest)
        return
    dest.mkdir(parents=True, exist_ok=True)
    a = arrays(root)
    rows = pd.read_parquet(root/'inputs/rows.parquet')
    training = rows[rows.part.eq('training')].reset_index(drop=True)
    parent = BASE/cfg['parent']/f'seed{seed}'/'indicators_ranking/model.pt'
    torch.manual_seed(seed)
    original, saved = restore_model(parent, cfg['device'])
    if saved['trained_labels_through'] >= training.date.min():
        raise ValueError('Parent checkpoint overlaps training/evaluation period')
    with np.load(root/'inputs/normalizer.npz') as norm:
        model = expand_model(original, norm['center'], norm['scale'])
    model.eval()
    original.eval()
    aa, bb, stamp, valid, aux = tensors(a, training.row_id.iloc[:2].to_numpy(), arm, cfg['device'])
    with torch.inference_mode():
        old = original.forecast_logits(aa[:, :-1], bb[:, :-1], stamp[:, :-1], aa[:, 1:], 59,
            auxiliary=forecast_auxiliary(aux[:, :, :len(INDICATORS)], 60, 64))
        new = model.forecast_logits(aa[:, :-1], bb[:, :-1], stamp[:, :-1], aa[:, 1:], 59,
            auxiliary=forecast_auxiliary(aux, 60, 64))
        for x, y in zip(old, new):
            torch.testing.assert_close(x, y, atol=3e-6, rtol=3e-6)
    del original
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=.01)
    first_epoch, history, exposure = 0, [], []
    if (dest/'resume.pt').exists():
        checkpoint = torch.load(dest/'resume.pt', map_location=cfg['device'], weights_only=True)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        first_epoch, history, exposure = checkpoint['epoch'], checkpoint['history'], checkpoint['exposure']
    for epoch in range(first_epoch, cfg['epochs']):
        torch.manual_seed(seed+epoch)
        model.train()
        losses = []
        for step, local in enumerate(grouped_batches(training, seed+epoch, cfg['batch_size'])):
            ids = training.row_id.to_numpy()[local]
            exposure.append(ids.tolist())
            aa, bb, stamp, valid, aux = tensors(a, ids, arm, cfg['device'])
            optimizer.zero_grad(set_to_none=True)
            ce = loss_parts(model, aa, bb, stamp, valid, 60, history_auxiliary=aux).mean()
            ce.backward()
            mid, _ = sampled_midpoint_loss(model, decoder, aa[:8], bb[:8], stamp[:8],
                a['mean'][ids[:8]], a['scale'][ids[:8]], a['future'][ids[:8]], a['valid'][ids[:8]],
                a['last'][ids[:8], 3], seed=seed*100000+epoch*1000+step, samples=4, history_auxiliary=aux[:8])
            (cfg['midpoint_loss_weight']*mid).backward()
            local_rows = training.iloc[local].reset_index(drop=True)
            ri = local_rows.groupby('date', sort=False).head(4).index.to_numpy()[:16]
            rids = ids[ri]
            rank, _ = sampled_return_ranking_loss(model, decoder, aa[ri], bb[ri], stamp[ri],
                a['mean'][rids], a['scale'][rids], a['future'][rids], a['valid'][rids], local_rows.iloc[ri].date.to_numpy(),
                seed=seed*1000000+epoch*1000+step, history_auxiliary=aux[ri], cost=cfg['cost'], **cfg['return_ranking_loss'])
            rank.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not all(torch.isfinite(v) for v in [ce, mid, rank, grad]):
                raise ValueError('Nonfinite training')
            optimizer.step()
            losses.append(float(ce.detach()))
        history.append(dict(epoch=epoch+1, mean_ce=float(np.mean(losses)), at=utc_now()))
        torch.save(dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                        epoch=epoch+1, history=history, exposure=exposure), dest/'resume.tmp')
        (dest/'resume.tmp').replace(dest/'resume.pt')
        write_json(root/'progress.json', dict(stage='training', seed=seed, arm=arm, **history[-1]))
        print('Trained epoch', seed, arm, history[-1], flush=True)
    model.eval().cpu()
    torch.save(checkpoint_payload(model, seed=seed, arm=arm, parent_sha256=file_hash(parent),
        trained_labels_through=training.label_end.max(), trained_signal_through=training.date.max(),
        protocol_sha256=file_hash(root/'protocol.json')), dest/'model.pt')
    loaded, _ = restore_model(dest/'model.pt')
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, loaded.state_dict()[k], atol=0, rtol=0)
    write_json(dest/'training.json', dict(history=history, rows=len(training), epochs=cfg['epochs'],
        parameters=sum(p.numel() for p in model.parameters()), exposure_sha256=digest(exposure),
        parent_sha256=file_hash(parent), reload_exact=True, warm_start_parity=True))
    write_manifest(dest)
    del model, loaded, optimizer
    gc.collect()
    torch.mps.empty_cache()


def forecast(root, cfg, seed, arm, decoder):
    dest = root/f'seed{seed}'/arm/'forecast'
    if (dest/'manifest.json').exists():
        verify(dest)
        return
    dest.mkdir(parents=True, exist_ok=True)
    a = arrays(root)
    rows = pd.read_parquet(root/'inputs/rows.parquet')
    ids = rows.loc[rows.part.eq('evaluation'), 'row_id'].to_numpy()
    parent = dest.parent/'training/model.pt'
    model, _ = restore_model(parent, cfg['device'])
    for start in range(0, len(ids), cfg['chunk_rows']):
        chunk = dest/f'{start:07d}'
        if (chunk/'manifest.json').exists():
            verify(chunk)
            identity = read(chunk/'identity.json')
            if identity != dict(model_sha256=file_hash(parent), ids=ids[start:start+cfg['chunk_rows']].tolist()):
                raise ValueError('Changed forecast chunk identity')
            continue
        chunk.mkdir(exist_ok=True)
        paths = []
        for offset in range(start, min(start+cfg['chunk_rows'], len(ids)), cfg['forecast_batch']):
            ix = ids[offset:min(offset+cfg['forecast_batch'], start+cfg['chunk_rows'])]
            aa, bb, stamp, _, aux = tensors(a, ix, arm, cfg['device'])
            tokens = generate_cached(model, aa[:, :60], bb[:, :60], stamp[:, :60], stamp[:, 60:],
                samples=cfg['samples'], seed=17+offset, temperature=1., top_p=1., top_k=0, history_auxiliary=aux)
            decoded, _ = decode_cached(decoder, tokens, a['mean'][ix], a['scale'][ix], 5)
            paths.append(decoded)
        np.save(chunk/'paths.npy', np.concatenate(paths))
        write_json(chunk/'identity.json', dict(model_sha256=file_hash(parent), ids=ids[start:start+cfg['chunk_rows']].tolist()))
        write_manifest(chunk)
        write_json(root/'progress.json', dict(stage='forecast', seed=seed, arm=arm,
                   complete=min(start+cfg['chunk_rows'], len(ids)), total=len(ids), at=utc_now()))
        print('Forecast', seed, arm, min(start+cfg['chunk_rows'], len(ids)), len(ids), utc_now(), flush=True)
    write_manifest(dest)
    del model
    gc.collect()
    torch.mps.empty_cache()


def run(config):
    cfg = read(config)
    root = BASE/cfg['output']
    torch.set_num_threads(4)
    with (root/'experiment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(root, cfg)
        spec = read(BASE/cfg['parent']/'profiles.json')['decoder']
        decoder = load_decoder(spec['path'], cfg['device'])
        for seed in cfg['seeds']:
            for arm in cfg['variants']:
                train(root, cfg, seed, arm, decoder)
                forecast(root, cfg, seed, arm, decoder)
        from finalize_order_flow_free import finalize
        finalize(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=BASE/'configs/token-order-flow-free-v1.json')
    run(parser.parse_args().config)
