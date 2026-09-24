"""Incremental token-model updates and immutable five-session ranked watchlists."""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .daily_loop import digest, read, verify, write_manifest
from .kronos_ranker import timestamps
from .midpoint_loss import sampled_midpoint_loss
from .storage import file_hash, utc_now, write_json
from .token_history import loss_parts, stratified_pool
from .token_transformer import (
    checkpoint_payload,
    decode_paths,
    generate_tokens,
    load_tokenizer,
    normalize_training,
    normalized_history,
    restore_model,
    valid_bars,
)

FIELDS = ['open', 'high', 'low', 'close', 'volume', 'amount', 'factor']


def future_known(frame, start, stop):
    """Mature five-session labels, no changed factor or identity across the boundary."""
    segment = frame.iloc[start:stop+1]
    if len(segment) != 6:
        return False
    q = segment[FIELDS].to_numpy(float)
    if not np.isfinite(q).all() or not valid_bars(q[:, :6]).all() or (q[:, 4:6] <= 0).any():
        return False
    if not np.isclose(q[:, 6], q[0, 6], atol=0, rtol=1e-8).all():
        return False
    for field in ['sequence_id', 'label_sequence_id']:
        if segment[field].isna().any() or segment[field].eq('').any() or segment[field].nunique() != 1:
            return False
    return not segment.source_trade_status.eq(0).any()


def panel(source):
    source = Path(source)
    verify(source / 'snapshot')
    meta = read(source / 'panel-h5/manifest.json')
    bars = pd.read_parquet(source / 'snapshot/bars.parquet')
    instruments = pd.read_parquet(source / 'snapshot/instruments.parquet').set_index('instrument_id')
    if bars.duplicated(['instrument_id', 'date']).any():
        raise ValueError('Duplicate daily prices')
    return meta, bars, instruments


def prepare_inputs(source, dest, cfg, previous):
    if (dest / 'manifest.json').exists():
        verify(dest)
        return
    meta, bars, instruments = panel(source)
    dates = meta['dates']
    signal = meta['price_data_through']
    end = dates.index(signal)
    future_dates = dates[end+1:end+6]
    if len(future_dates) != 5 or bars.date.max() != signal:
        raise ValueError('Incomplete close or forward trading calendar')
    if end < 64:
        raise ValueError('Insufficient history for mature token labels')
    start = max(59, end-5-cfg['initial_recent_dates']+1)
    candidates, histories, forecasts, ledger = [], {}, [], []
    for symbol, group in bars.groupby('instrument_id', sort=True):
        g = group.set_index('date').reindex(dates)
        raw = g[FIELDS].to_numpy(float)
        good = np.isfinite(raw).all(1) & (raw[:, 6] > 0) & valid_bars(raw[:, :6])
        is_current = symbol in instruments.index and bool(instruments.loc[symbol, 'current_member'])
        current_valid = bool(good[end-59:end+1].all() and (raw[end, 4:6] > 0).all())
        if is_current:
            ledger.append(dict(instrument_id=symbol, eligible=current_valid,
                reason='ready' if current_valid else 'missing_or_unverified_60_session_history'))
        if is_current and current_valid:
            forecasts.append((symbol, raw[end-59:end+1].copy()))
        for t in range(start, end-4):
            if good[t-59:t+1].all() and future_known(g, t, t+5):
                candidates.append(dict(instrument_id=symbol, date=dates[t], time=t, label_end=dates[t+5]))
        histories[symbol] = raw
    available = pd.DataFrame(candidates)
    if available.empty or not forecasts:
        raise ValueError('No eligible training or forecast inputs')
    prior_date = previous.get('trained_signal_through', '') if previous else ''
    chosen = []
    for day, group in available.groupby('date'):
        if prior_date:
            if day <= prior_date:
                continue
            chosen.extend(group.index.tolist())
        else:
            positions = stratified_pool(np.arange(len(group)), group.instrument_id.tolist(), day, cfg['initial_rows_per_date'])
            chosen.extend(group.index.to_numpy()[positions].tolist())
    if prior_date:
        older = available[available.date <= prior_date]
        if len(older):
            # Stable hash-order sample spreads replay over stocks and dates.
            order = older.apply(lambda r: digest([signal, r.date, r.instrument_id]), axis=1).sort_values().index
            chosen.extend(order[:cfg['recent_replay_rows']].tolist())
    chosen = list(dict.fromkeys(chosen))
    if not chosen:
        raise ValueError('No newly mature labels or replay data')
    selected = available.loc[chosen].reset_index(drop=True)
    selected['role'] = np.where(selected.date > prior_date, 'newly_mature' if prior_date else 'initial_recent', 'recent_replay')
    dest.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(Path(cfg['tokenizer_bundle']), cfg['device'])
    parts = {key: [] for key in ['s1', 's2', 'stamps', 'valid', 'mean', 'scale', 'future', 'last']}
    for offset in range(0, len(selected), 256):
        sub = selected.iloc[offset:offset+256]
        raw = np.stack([histories[r.instrument_id][r.time-59:r.time+1] for r in sub.itertuples()])
        target = np.stack([histories[r.instrument_id][r.time+1:r.time+6, :6] for r in sub.itertuples()])
        x, valid, mean, scale = normalize_training(raw, target, np.ones((len(raw), 5), bool))
        stamp = np.stack([timestamps(dates[r.time-59:r.time+6]) for r in sub.itertuples()])
        with torch.inference_mode():
            a, b = tokenizer.encode(torch.as_tensor(x, device=cfg['device']), half=True)
        for key, value in dict(s1=a.cpu().numpy(), s2=b.cpu().numpy(), stamps=stamp, valid=valid,
                               mean=mean, scale=scale, future=target, last=raw[:, -1, :6]).items():
            parts[key].append(value)
    train = {k: np.concatenate(v) for k, v in parts.items()}
    # Old replay is restricted to the original pre-2024 training partition.
    old = Path(cfg['historical_replay'])
    old_manifest = read(old / 'completed.json')['files']
    for name in ['rows.parquet', '2024h2-train.npy', 's1.npy', 's2.npy', 'valid.npy', 'future.npy',
                 'mean.npy', 'scale.npy', 'last.npy', 'calendar-stamps.npy']:
        if file_hash(old / name) != old_manifest[name]:
            raise ValueError('Changed historical replay source: '+name)
    old_rows = pd.read_parquet(old / 'rows.parquet')
    eligible = np.load(old / '2024h2-train.npy')
    known = np.load(old / 'valid.npy', mmap_mode='r')
    eligible = eligible[known[eligible].all(1)]
    rng = np.random.default_rng(int(digest(signal)[:8], 16))
    replay = rng.choice(eligible, min(cfg['historical_replay_rows'], len(eligible)), replace=False)
    assert old_rows.iloc[replay].label_end.max() < '2024-01-01' <= signal
    old_stamps = np.load(old / 'calendar-stamps.npy')
    t = old_rows.iloc[replay].date_index.to_numpy(int)
    for key in train:
        values = old_stamps[t[:, None]+np.arange(-59, 6)] if key == 'stamps' else np.load(old / f'{key}.npy', mmap_mode='r')[replay]
        train[key] = np.concatenate([train[key], values])
    selected = pd.concat([selected, old_rows.iloc[replay][['instrument_id', 'date', 'label_end']].assign(role='historical_replay')], ignore_index=True)
    assert selected.label_end.max() <= signal
    for key, value in train.items():
        np.save(dest / f'train-{key}.npy', value)
    selected.to_parquet(dest / 'training-rows.parquet', index=False)
    np.save(dest / 'historical-replay-ids.npy', replay)
    # Prediction eligibility uses current history only, never future labels.
    symbols, raw = zip(*forecasts)
    raw = np.stack(raw)
    x, mean, scale = normalized_history(raw)
    token_a, token_b = [], []
    for offset in range(0, len(x), 256):
        with torch.inference_mode():
            a, b = tokenizer.encode(torch.as_tensor(x[offset:offset+256], device=cfg['device']), half=True)
        token_a.append(a.cpu().numpy())
        token_b.append(b.cpu().numpy())
    for key, value in dict(s1=np.concatenate(token_a), s2=np.concatenate(token_b), mean=mean, scale=scale,
                            last=raw[:, -1, :6], stamps=timestamps(dates[end-59:end+6])).items():
        np.save(dest / f'forecast-{key}.npy', value)
    pd.DataFrame(dict(instrument_id=symbols, name=[instruments.loc[s, 'name'] for s in symbols])).to_parquet(dest / 'forecast-rows.parquet', index=False)
    pd.DataFrame(ledger).to_parquet(dest / 'coverage.parquet', index=False)
    write_json(dest / 'input.json', dict(signal_date=signal, horizon_dates=future_dates, inputs=len(symbols),
        training_rows=len(selected), roles=selected.role.value_counts().to_dict(), training_dates=int(selected.date.nunique()),
        new_signal_through=available.date.max(), observed_labels_through=selected.label_end.max(),
        source=str(Path(source).resolve()), snapshot_sha256=file_hash(Path(source) / 'snapshot/manifest.json'),
        historical_manifest_sha256=file_hash(old / 'completed.json'), created_at=utc_now()))
    write_manifest(dest)
    del tokenizer
    gc.collect()
    torch.mps.empty_cache() if cfg['device'] == 'mps' else None


def load_arrays(root, prefix):
    return {p.stem.removeprefix(prefix+'-'): np.load(p, mmap_mode='r') for p in root.glob(prefix+'-*.npy')}


def update_model(initial, dest, arrays, cfg, seed, decoder, input_meta):
    if (dest / 'manifest.json').exists():
        verify(dest)
        return
    dest.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    model, saved = restore_model(Path(initial), cfg['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=.01)
    if 'daily_optimizer' in saved:
        optimizer.load_state_dict(saved['daily_optimizer'])
        for group in optimizer.param_groups:
            group['lr'] = cfg['learning_rate']
    day = input_meta['signal_date']
    order = np.random.default_rng(seed+int(day.replace('-', ''))).permutation(len(arrays['s1']))
    losses = []
    model.train()
    for start in range(0, len(order), cfg['training_batch']):
        ix = order[start:start+cfg['training_batch']]
        a, b, stamps = [torch.as_tensor(np.array(arrays[k][ix]), dtype=torch.long, device=cfg['device']) for k in ['s1', 's2', 'stamps']]
        valid = torch.as_tensor(np.array(arrays['valid'][ix]), device=cfg['device'])
        optimizer.zero_grad(set_to_none=True)
        loss = loss_parts(model, a, b, stamps, valid, 60).mean()
        loss.backward()
        if cfg['midpoint_loss_weight']:
            k = min(8, len(ix))
            risk, _ = sampled_midpoint_loss(model, decoder, a[:k], b[:k], stamps[:k],
                arrays['mean'][ix[:k]], arrays['scale'][ix[:k]], arrays['future'][ix[:k]],
                arrays['valid'][ix[:k]], arrays['last'][ix[:k], 3], seed=seed*1000000+start+int(day.replace('-', '')), samples=4)
            (cfg['midpoint_loss_weight']*risk).backward()
            if not torch.isfinite(risk):
                raise ValueError('Nonfinite midpoint loss')
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise ValueError('Nonfinite daily training update')
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    payload = checkpoint_payload(model, daily_optimizer=optimizer.state_dict(), seed=seed,
        trained_labels_through=input_meta['observed_labels_through'], trained_signal_through=input_meta['new_signal_through'],
        parent_sha256=file_hash(Path(initial)), daily_config_sha256=digest(cfg))
    payload['state_dict'] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(payload, dest / 'model.pt')
    # Independent reload on CPU must reproduce both forecast heads exactly.
    model.cpu()
    loaded, _ = restore_model(dest / 'model.pt', 'cpu')
    loaded.eval()
    args = [torch.as_tensor(np.array(arrays[k][:2]), dtype=torch.long) for k in ['s1', 's2', 'stamps']]
    with torch.inference_mode():
        x = model.forecast_logits(args[0][:, :-1], args[1][:, :-1], args[2][:, :-1], args[0][:, 1:], 59)
        y = loaded.forecast_logits(args[0][:, :-1], args[1][:, :-1], args[2][:, :-1], args[0][:, 1:], 59)
    for left, right in zip(x, y):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    write_json(dest / 'training.json', dict(seed=seed, rows=len(order), epochs=1, mean_batch_ce=float(np.mean(losses)),
        parent_checkpoint=str(initial), parent_sha256=file_hash(Path(initial)), reload_exact=True,
        training_labels_through=input_meta['observed_labels_through'], finished_at=utc_now()))
    write_manifest(dest)
    del model, loaded, optimizer
    gc.collect()
    torch.mps.empty_cache() if cfg['device'] == 'mps' else None


def forecast_model(checkpoint, dest, arrays, cfg, seed, decoder):
    if (dest / 'manifest.json').exists():
        verify(dest)
        return
    dest.mkdir(parents=True, exist_ok=True)
    chunks = dest / 'chunks'
    chunks.mkdir(exist_ok=True)
    model, _ = restore_model(checkpoint, cfg['device'])
    model.eval()
    n, size = len(arrays['s1']), cfg['forecast_batch']
    for start in range(0, n, size):
        path = chunks / f'{start:06d}.npz'
        if path.exists():
            continue
        end = min(start+size, n)
        a, b = [torch.as_tensor(np.array(arrays[k][start:end]), dtype=torch.long, device=cfg['device']) for k in ['s1', 's2']]
        stamp = torch.as_tensor(np.array(arrays['stamps']), dtype=torch.long, device=cfg['device'])[None].expand(end-start, -1, -1)
        tokens = generate_tokens(model, a, b, stamp[:, :60], stamp[:, 60:], samples=cfg['samples_per_model'],
            seed=cfg['sampling_seed']+start, temperature=1., top_p=1., top_k=0)
        paths, valid = decode_paths(decoder, tokens, arrays['mean'][start:end], arrays['scale'][start:end], 5)
        with path.with_suffix('.tmp').open('wb') as f:
            np.savez_compressed(f, paths=paths, valid=valid, s1=tokens[0][:, :, -5:].cpu().numpy(), s2=tokens[1][:, :, -5:].cpu().numpy())
        path.with_suffix('.tmp').replace(path)
        if start % 512 == 0:
            print(dict(stage='forecast', seed=seed, inputs=end, total=n, at=utc_now()), flush=True)
    out = np.lib.format.open_memmap(dest / 'paths.npy', mode='w+', dtype=np.float32, shape=(n, cfg['samples_per_model'], 5, 6))
    for start in range(0, n, size):
        with np.load(chunks / f'{start:06d}.npz') as z:
            out[start:start+len(z['paths'])] = z['paths']
    out.flush()
    write_json(dest / 'forecast.json', dict(seed=seed, checkpoint_sha256=file_hash(checkpoint), inputs=n,
        samples=cfg['samples_per_model'], sampling_seed=cfg['sampling_seed'], finished_at=utc_now()))
    write_manifest(dest)
    del model, out
    gc.collect()
    torch.mps.empty_cache() if cfg['device'] == 'mps' else None


OPEN_CLOSE = 'open_to_close'
LOW_HIGH = 't_low_to_future_high'


def scenario_prices(paths, method=OPEN_CLOSE):
    if method == OPEN_CLOSE:
        return paths[..., 0, 0], paths[..., -1, 3]
    if method == LOW_HIGH:
        if paths.shape[-2] < 2:
            raise ValueError('Selling requires at least T+1')
        return paths[..., 0, 2], paths[..., 1:, 1].max(axis=-1)
    raise ValueError('Unknown ranking method: '+str(method))


def scenario_description(method):
    if method == LOW_HIGH:
        return 'Ideal-timing potential: buy at T low and sell at the highest high from T+1 through T+4; extrema are not guaranteed fills'
    if method == OPEN_CLOSE:
        return 'T open to T+4 close cost scenario; not actual fills'
    raise ValueError('Unknown ranking method: '+str(method))


def path_statistics(paths, cost, method=OPEN_CLOSE):
    """All returns are scenarios from future opening prices, not executable fills."""
    records = []
    for h in [2, 5]:
        mask = valid_bars(paths[:, :h]).all(-1)
        x = paths[mask, :h].astype(float)
        if len(x) < 16:
            return None
        entry, exit_price = scenario_prices(x, method)
        net = exit_price/entry-1-cost
        upside = x[:, :, 1].max(1)/entry-1
        downside = x[:, :, 2].min(1)/entry-1
        records.append(dict(horizon=h, entry_median=float(np.median(entry)), exit_median=float(np.median(exit_price)), expected_net_return=float(net.mean()), positive_fraction=float((net > 0).mean()),
            return_q10=float(np.quantile(net, .1)), return_q50=float(np.median(net)), return_q90=float(np.quantile(net, .9)),
            upside_median=float(np.median(upside)), downside_q10=float(np.quantile(downside, .1)),
            valid_paths=int(mask.sum()), high_median=float(np.median(x[:, :, 1].max(1))), low_median=float(np.median(x[:, :, 2].min(1)))))
    return records


def rank_paths(run, cfg, destination=None):
    destination = Path(destination) if destination is not None else run
    destination.mkdir(parents=True, exist_ok=True)
    rows = pd.read_parquet(run / 'inputs/forecast-rows.parquet')
    records, quantiles = [], []
    for seed in cfg['seeds']:
        paths = np.load(run / f'forecasts/seed{seed}/paths.npy', mmap_mode='r')
        for i, row in enumerate(rows.itertuples()):
            stats = path_statistics(paths[i], cfg['round_trip_cost_scenario'], cfg.get('ranking_method', OPEN_CLOSE))
            if stats is None:
                continue
            for item in stats:
                records.append(dict(instrument_id=row.instrument_id, name=row.name, seed=seed, **item))
            legal = valid_bars(paths[i]).all(-1)
            q = np.quantile(paths[i, legal].astype(float), [.1, .5, .9], axis=0)
            for day in range(5):
                for field, name in enumerate(FIELDS[:6]):
                    quantiles.append(dict(instrument_id=row.instrument_id, seed=seed, day=day, field=name,
                        q10=q[0, day, field], q50=q[1, day, field], q90=q[2, day, field]))
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError('No usable forecast distributions')
    frame.to_parquet(destination / 'model-scores.parquet', index=False)
    pd.DataFrame(quantiles).to_parquet(destination / 'daily-quantiles.parquet', index=False)
    common = frame.groupby(['instrument_id', 'horizon']).seed.transform('nunique').eq(len(cfg['seeds']))
    f = frame[common]
    metrics = list(f.select_dtypes(include='number').columns.difference(['seed', 'horizon']))
    out = f.groupby(['instrument_id', 'name', 'horizon'])[metrics].mean().reset_index()
    disagreement = f.groupby(['instrument_id', 'horizon']).expected_net_return.std(ddof=0).rename('model_disagreement')
    out = out.merge(disagreement, on=['instrument_id', 'horizon'], validate='one_to_one')
    ranks = out[out.horizon == 5].sort_values(['expected_net_return', 'instrument_id'], ascending=[False, True]).copy()
    ranks['rank'] = np.arange(1, len(ranks)+1)
    ranks.to_csv(destination / 'ranking.csv', index=False)
    out.to_parquet(destination / 'summary.parquet', index=False)
    return ranks


def review_prior(store, source):
    meta, bars, _ = panel(source)
    now = meta['price_data_through']
    groups = {s: g.set_index('date') for s, g in bars.groupby('instrument_id')}
    summaries = []
    for p in sorted((store / 'runs').glob('*/run.json')):
        info = read(p)
        if (store / 'reference-publications' / (p.parent.name+'.json')).exists():
            # Canonical scorecards evaluate the actually published reference-price ranking.
            continue
        if info['horizon_dates'][-1] > now or not info['prospective']:
            continue
        destination = store / 'reviews' / p.parent.name / digest([now, file_hash(Path(source) / 'snapshot/manifest.json')])[:16]
        if (destination / 'manifest.json').exists():
            verify(destination)
            summaries.append(read(destination / 'review.json'))
            continue
        ranks = pd.read_csv(p.parent / 'ranking.csv')
        scored = []
        for r in ranks.itertuples():
            g = groups.get(r.instrument_id)
            if g is None:
                continue
            dates = [info['signal_date'], *info['horizon_dates']]
            x = g.reindex(dates)
            if not future_known(x, 0, 5):
                continue
            entry, exit_price = scenario_prices(x.iloc[1:][FIELDS[:6]].to_numpy(float), info.get('ranking_method', OPEN_CLOSE))
            realized = exit_price/entry-1-info['cost_scenario']
            scored.append(dict(instrument_id=r.instrument_id, rank=r.rank,
                expected=r.expected_net_return, realized_net_scenario=float(realized)))
        result = pd.DataFrame(scored)
        destination.mkdir(parents=True, exist_ok=True)
        result.to_csv(destination / 'rows.csv', index=False)
        top = result[result['rank'] <= info['top_n']] if len(result) else result
        summary = dict(run=p.parent.name, known=len(result), total=len(ranks), unknown=len(ranks)-len(result),
            observed_through=now, top_known=len(top),
            top_mean_net_scenario=float(top.realized_net_scenario.mean()) if len(top) else None,
            top_positive_fraction=float((top.realized_net_scenario > 0).mean()) if len(top) else None,
            pool_mean_net_scenario=float(result.realized_net_scenario.mean()) if len(result) else None,
            rank_ic=float(result.expected.rank().corr(result.realized_net_scenario.rank())) if len(result)>2 and result.expected.nunique()>1 and result.realized_net_scenario.nunique()>1 else None,
            ranking_method=info.get('ranking_method', OPEN_CLOSE), interpretation=scenario_description(info.get('ranking_method', OPEN_CLOSE)))
        write_json(destination / 'review.json', summary)
        write_manifest(destination)
        summaries.append(summary)
    return summaries


def cycle(source, store, cfg):
    """Call under the operator's process lock. Publish pointers only after completion."""
    store, source = Path(store), Path(source)
    meta = read(source / 'panel-h5/manifest.json')
    signal = meta['price_data_through']
    if pd.Timestamp(signal+'T15:00:00', tz='Asia/Shanghai') > pd.Timestamp(utc_now()):
        raise ValueError('Cannot train from an unfinished close')
    run = store / 'runs' / signal
    if (run / 'manifest.json').exists():
        verify(run)
        if read(run / 'run.json')['config_sha256'] != digest(cfg):
            raise ValueError('This close already has a completed run under another config')
        publish_state(store, run)
        review_prior(store, source)
        return run
    state_path = store / 'current.json'
    previous = read(state_path) if state_path.exists() else None
    if previous and previous['signal_date'] >= signal:
        raise ValueError('Non-monotone daily model update')
    profiles = read(Path(cfg['initial_profiles']))
    initial = previous['checkpoints'] if previous else profiles[cfg['initial_owner']]['checkpoints']
    run.mkdir(parents=True, exist_ok=True)
    binding = dict(config=cfg, source=str(source.resolve()), source_sha256=file_hash(source / 'snapshot/manifest.json'),
        checkpoints=initial, decoder=profiles['decoder'], previous_state=previous, started_at=utc_now())
    if (run / 'binding.json').exists():
        existing = read(run / 'binding.json')
        for key in ['config', 'source', 'source_sha256', 'checkpoints', 'decoder']:
            if existing[key] != binding[key]:
                raise ValueError('Resume inputs or model lineage changed')
        binding = existing
    else:
        write_json(run / 'binding.json', binding)
    for item in initial.values():
        if file_hash(Path(item['path'])) != item['sha256']:
            raise ValueError('Changed parent model')
    if file_hash(Path(profiles['decoder']['path'])) != profiles['decoder']['sha256']:
        raise ValueError('Changed decoder')
    prepare_inputs(source, run / 'inputs', cfg, previous)
    info = read(run / 'inputs/input.json')
    arrays = load_arrays(run / 'inputs', 'train')
    decoder = load_tokenizer(Path(cfg['tokenizer_bundle']), cfg['device'])
    adapted = torch.load(profiles['decoder']['path'], map_location='cpu', weights_only=True)
    decoder.load_state_dict(adapted['state_dict'])
    decoder.eval().requires_grad_(False)
    checkpoints = {}
    for seed in cfg['seeds']:
        dest = run / f'models/seed{seed}'
        update_model(initial[str(seed)]['path'], dest, arrays, cfg, seed, decoder, info)
        checkpoint = dest / 'model.pt'
        checkpoints[str(seed)] = dict(path=str(checkpoint.resolve()), sha256=file_hash(checkpoint))
        print(dict(stage='updated', seed=seed, rows=info['training_rows'], at=utc_now()), flush=True)
    forecast = load_arrays(run / 'inputs', 'forecast')
    for seed in cfg['seeds']:
        forecast_model(Path(checkpoints[str(seed)]['path']), run / f'forecasts/seed{seed}', forecast, cfg, seed, decoder)
    ranks = rank_paths(run, cfg)
    published = utc_now()
    prospective = pd.Timestamp(published) < pd.Timestamp(info['horizon_dates'][0]+'T09:15:00', tz='Asia/Shanghai')
    run_info = dict(signal_date=signal, horizon_dates=info['horizon_dates'], published_at=published,
        prospective=prospective, mode='post_close_watchlist' if prospective else 'late_research_watchlist',
        forecast_inputs=info['inputs'], ranked=len(ranks), top_n=cfg['top_n'], cost_scenario=cfg['round_trip_cost_scenario'],
        no_orders=True, ranking_method=cfg.get('ranking_method', OPEN_CLOSE), config_sha256=digest(cfg), checkpoints=checkpoints, trained_signal_through=info['new_signal_through'])
    write_json(run / 'run.json', run_info)
    (run / 'report.md').write_text(watchlist_report(run_info, ranks))
    write_manifest(run)
    # Reusing a verified run cannot apply the same day's gradient updates twice.
    publish_state(store, run)
    review_prior(store, source)
    return run


def watchlist_report(info, ranks):
    method = info.get('ranking_method', OPEN_CLOSE)
    report = '# Daily token-model watchlist\n\n'
    report += f'Completed close: {info["signal_date"]}. Forecast sessions: '+', '.join(info['horizon_dates'])+'.\n\n'
    report += f'Published: {info["published_at"]}. Mode: {info["mode"]}. Ranked {len(ranks)} stocks from {info["forecast_inputs"]} eligible histories.\n\n'
    report += scenario_description(method)+'. Average each valid path return within each model, then weight all three models equally. Subtract the '+f'{info["cost_scenario"]:.2%}'+' round-trip cost scenario. Positive-path frequencies are model-implied, not calibrated real-world win rates.\n\n'
    report += '| Rank | Code | Name | Mean net potential | T entry median | Exit median | Positive paths | Return q10 | Low-side q10 | Model disagreement |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n'
    for r in ranks.head(info['top_n']).itertuples():
        report += f'| {r.rank} | {r.instrument_id} | {r.name} | {r.expected_net_return:.2%} | {r.entry_median:.2f} | {r.exit_median:.2f} | {r.positive_fraction:.1%} | {r.return_q10:.2%} | {r.downside_q10:.2%} | {r.model_disagreement:.2%} |\n'
    report += '\nEntry, exit and return quantiles average per-model quantiles; dividing displayed medians does not reproduce the mean path return. A five-day high on T is excluded from the exit window. The two-day version exits at T+1 high when using ideal timing. These are research scenarios, not order prices or realized trading profits. Policy/news and execution constraints are not modeled.\n' if method == LOW_HIGH else '\nResearch scenarios, not orders or realized trading profits.\n'
    return report


def publish_state(store, run):
    """Recover the pointer if interrupted after completing a run; never regress it."""
    info = read(run / 'run.json')
    source = read(run / 'binding.json')['source']
    state_path = store / 'current.json'
    if state_path.exists() and read(state_path)['signal_date'] > info['signal_date']:
        return
    write_json(state_path, dict(signal_date=info['signal_date'], trained_signal_through=info['trained_signal_through'],
        checkpoints=info['checkpoints'], source=source, run=str(run.resolve()),
        manifest_sha256=file_hash(run / 'manifest.json'), published_at=info['published_at']))
    write_json(store / 'latest.json', dict(run=str(run.resolve()), report=str((run / 'report.md').resolve()), signal_date=info['signal_date']))
