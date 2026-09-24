"""Manual, isolated matched incremental bridge; never publishes live model state."""
import argparse
import fcntl
import gc
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from token_ranking_shadow import prepare_shadow_features
from tokenizer_reconstruction_run import load_decoder

from quant_research.daily_loop import digest, read, verify, write_manifest
from quant_research.daily_token import FIELDS, load_arrays, panel
from quant_research.kronos_ranker import timestamps
from quant_research.midpoint_loss import sampled_midpoint_loss
from quant_research.return_ranking_loss import reference_prices, sampled_return_ranking_loss
from quant_research.storage import file_hash, utc_now, write_json
from quant_research.token_cached_inference import decode_cached, generate_cached
from quant_research.token_history import loss_parts, stratified_pool
from quant_research.token_indicators import FEATURES, indicator_features
from quant_research.token_transformer import (
    checkpoint_payload,
    forecast_auxiliary,
    normalized_history,
    restore_model,
)

BASE = Path(__file__).resolve().parents[1]
ARMS = ('baseline', 'indicators_ranking')


def grouped_batches(rows, seed, limit=256):
    """Each row once, bounded batches, retain same-date peers where possible."""
    rng = np.random.default_rng(seed)
    groups = [rng.permutation(g.index.to_numpy()) for _, g in rows.groupby('date', sort=True)]
    pending = []
    for pos in rng.permutation(len(groups)):
        group = groups[pos]
        for start in range(0, len(group), limit):
            chunk = group[start:start+limit].tolist()
            if pending and len(pending)+len(chunk) > limit:
                yield np.array(pending, dtype=int)
                pending = []
            pending.extend(chunk)
    if pending:
        yield np.array(pending, dtype=int)


def validate_rows(rows, arrays, info, previous=None):
    if len(rows) != len(arrays['s1']) or rows[['instrument_id', 'date']].duplicated().any():
        raise ValueError('Training row identity mismatch')
    if rows.label_end.max() > info['signal_date'] or rows.label_end.max() != info['observed_labels_through']:
        raise ValueError('Training labels exceed the observed close')
    if rows.date.max() != info['new_signal_through'] or not arrays['valid'].all():
        raise ValueError('Only complete mature training labels are supported')
    if previous is None:
        if set(rows.role) != {'initial_recent', 'historical_replay'}:
            raise ValueError('This bridge requires the initial daily update, not a later resumed cycle')
    else:
        roles = set(rows.role)
        if not roles <= {'newly_mature', 'recent_replay', 'historical_replay'} or 'newly_mature' not in roles:
            raise ValueError('Continuation requires newly mature labels and recognized replay roles')
        if info['signal_date'] <= previous['signal_date'] or info['new_signal_through'] <= previous['new_signal_through']:
            raise ValueError('Repeated or non-monotone training date')
        if (rows.loc[rows.role.eq('newly_mature'), 'date'] <= previous['new_signal_through']).any():
            raise ValueError('Previously consumed labels marked newly mature')
        if (rows.loc[rows.role.ne('newly_mature'), 'date'] > previous['new_signal_through']).any():
            raise ValueError('Unseen signal dates mislabeled replay')


def training_features(run, cfg):
    root = run/'inputs'
    info = read(root/'input.json')
    rows = pd.read_parquet(root/'training-rows.parquet')
    arrays = load_arrays(root, 'train')
    previous = read(Path(cfg['parent_root'])/'prepared/input.json') if cfg.get('parent_root') else None
    validate_rows(rows, arrays, info, previous)
    source = Path(info['source'])
    meta, bars, _ = panel(source)
    if file_hash(source/'snapshot/manifest.json') != info['snapshot_sha256'] or meta['price_data_through'] != info['signal_date']:
        raise ValueError('Changed daily snapshot')
    groups = {s: g.set_index('date') for s, g in bars.groupby('instrument_id')}
    old = BASE/cfg['historical_replay']
    if file_hash(old/'completed.json') != info['historical_manifest_sha256']:
        raise ValueError('Changed historical replay manifest')
    verify(old, 'completed.json')
    prices = BASE/cfg['price_input']
    lineage = read(old/'sources.json')
    if file_hash(prices/'values.npy') != lineage[str((prices/'values.npy').resolve())]:
        raise ValueError('Changed historical prices')
    raw_old = np.load(prices/'values.npy', mmap_mode='r')
    old_arrays = {k: np.load(old/f'{k}.npy', mmap_mode='r') for k in ['future', 'last']}
    old_dates = read(old/'calendar.json')
    old_meta = read(prices/'manifest.json')
    if old_dates != old_meta['dates'] or read(old/'instruments.json') != old_meta['instruments']:
        raise ValueError('Historical price axes differ')
    old_rows = pd.read_parquet(old/'rows.parquet').iloc[np.load(root/'historical-replay-ids.npy')]
    replay_positions = np.flatnonzero(rows.role.eq('historical_replay'))
    pd.testing.assert_frame_equal(rows.iloc[replay_positions][['instrument_id', 'date', 'label_end']].reset_index(drop=True),
        old_rows[['instrument_id', 'date', 'label_end']].reset_index(drop=True))
    replay = dict(zip(replay_positions, old_rows.itertuples()))
    out = np.empty((len(rows), 60, len(FEATURES)), np.float32)
    feature_batches = [g.index.to_numpy()[start:start+256] for _, g in rows.groupby('role', sort=False)
                       for start in range(0, len(g), 256)]
    for ix in feature_batches:
        histories, futures, stamps = [], [], []
        for i in ix:
            r = rows.iloc[i]
            if i in replay:
                prior = replay[i]
                t, dates = prior.date_index, old_dates
                history = raw_old[prior.stock_index, t-59:t+1].copy()
                target = raw_old[prior.stock_index, t+1:t+6, :6].copy()
            else:
                dates = meta['dates']
                t = dates.index(r.date)
                values = groups[r.instrument_id].reindex(dates[t-59:t+6])[FIELDS].to_numpy(float)
                # Match the original training builder's column-major history views.
                history, target = values[:60], values[60:, :6]
            if dates[t+5] != r.label_end:
                raise ValueError('Label calendar mismatch')
            histories.append(history)
            futures.append(target)
            stamps.append(timestamps(dates[t-59:t+6]))
        raw = np.stack(histories)
        _, mean, scale = normalized_history(raw)
        sl = ix
        last = raw[:, -1, :6]
        future = np.stack(futures)
        if rows.iloc[ix[0]].role == 'historical_replay':
            future = future.astype(old_arrays['future'].dtype)
            last = last.astype(old_arrays['last'].dtype)
        for name, actual in [('mean', mean), ('scale', scale), ('future', future),
                             ('last', last), ('stamps', np.stack(stamps))]:
            expected = arrays[name][sl]
            if name == 'future' and rows.iloc[ix[0]].role == 'historical_replay':
                # Archived histories are float32; replay labels retain their original float64 prices.
                expected = expected.astype(raw_old.dtype)
            np.testing.assert_array_equal(actual, expected, err_msg=name)
        out[sl] = indicator_features(raw)
    return out


def live_state(run):
    store = run.parent.parent
    paths = [BASE/'configs/daily-token-v1.json', store/'current.json', store/'latest.json', run/'manifest.json']
    paths += sorted((run/'models').glob('*/model.pt'))
    if (store/'current.json').exists():
        paths += [Path(v['path']) for v in read(store/'current.json')['checkpoints'].values()]
    return {str(p): file_hash(p) for p in paths if p.exists()}


def prepare(run, root, cfg):
    verify(run)
    experiment = BASE/cfg['experiment']
    verify(experiment, 'completed.json')
    if cfg.get('parent_root'):
        verify(Path(cfg['parent_root']))
    binding = dict(config=cfg, daily_manifest=file_hash(run/'manifest.json'),
        parent_manifest=file_hash(Path(cfg['parent_root'])/'manifest.json') if cfg.get('parent_root') else None,
        experiment_manifest=file_hash(experiment/'completed.json'),
        code={str(p.relative_to(BASE)): file_hash(p) for p in [Path(__file__), *sorted((BASE/'src/quant_research').glob('*.py')),
            BASE/'scripts/token_ranking_shadow.py', BASE/'scripts/tokenizer_reconstruction_run.py']})
    if (root/'prepared/manifest.json').exists():
        verify(root/'prepared')
        if read(root/'prepared/binding.json') != binding:
            raise ValueError('Changed protocol, source or implementation; use a new output')
        return
    dest = root/'prepared'
    if dest.exists():
        raise ValueError('Partial preparation retained; use a new output')
    dest.mkdir(parents=True)
    write_json(dest/'binding.json', binding)
    write_json(dest/'live-state.json', live_state(run))
    features = training_features(run, cfg)
    np.save(dest/'training-features.npy', features)
    forecasts = load_arrays(run/'inputs', 'forecast')
    rows = pd.read_parquet(run/'inputs/forecast-rows.parquet')
    signal = read(run/'inputs/input.json')['signal_date']
    ids = stratified_pool(np.arange(len(rows)), rows.instrument_id.tolist(), signal, cfg['forecast_inputs']) if cfg['forecast_inputs'] else np.arange(len(rows))
    aux, _ = prepare_shadow_features(run, {k: v if k == 'stamps' else v[ids] for k, v in forecasts.items()}, rows.iloc[ids])
    np.save(dest/'forecast-ids.npy', ids)
    np.save(dest/'forecast-features.npy', aux)
    rows.iloc[ids].reset_index(drop=True).to_parquet(dest/'forecast-rows.parquet', index=False)
    shutil.copy2(run/'inputs/training-rows.parquet', dest/'training-rows.parquet')
    write_json(dest/'input.json', read(run/'inputs/input.json'))
    write_manifest(dest)


def parent_checkpoint(cfg, seed, arm):
    if cfg.get('parent_root'):
        return Path(cfg['parent_root'])/f'seed{seed}'/arm/'training/model.pt'
    return BASE/cfg['experiment']/f'seed{seed}'/arm/'model.pt'


def optimizer_steps(state):
    return {str(key): int(value['step'].item()) for key, value in state['state'].items() if 'step' in value}


def train(run, root, cfg, seed, arm, decoder):
    dest = root/f'seed{seed}'/arm
    if (dest/'training/manifest.json').exists():
        verify(dest/'training')
        return
    output = dest/'training'
    output.mkdir(parents=True, exist_ok=True)
    parent = parent_checkpoint(cfg, seed, arm)
    torch.manual_seed(seed)
    model, saved = restore_model(parent, cfg['device'])
    info = read(root/'prepared/input.json')
    if cfg.get('parent_root'):
        previous = read(Path(cfg['parent_root'])/'prepared/input.json')
        if saved['trained_labels_through'] != previous['observed_labels_through'] or saved['trained_signal_through'] != previous['new_signal_through']:
            raise ValueError('Continuation checkpoint cutoff differs from parent inputs')
    elif saved['trained_labels_through'] >= '2024-01-01':
        raise ValueError('Matched parent must retain the pre-2024 cutoff')
    expected = () if arm == 'baseline' else FEATURES
    if tuple(model.config.auxiliary_features) != expected:
        raise ValueError('Unexpected feature schema')
    arrays = load_arrays(run/'inputs', 'train')
    rows = pd.read_parquet(root/'prepared/training-rows.parquet')
    features = np.load(root/'prepared/training-features.npy', mmap_mode='r')
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=.01)
    if cfg.get('parent_root'):
        optimizer.load_state_dict(saved['daily_optimizer'])
        for group in optimizer.param_groups:
            group['lr'] = cfg['learning_rate']
    optimizer_before = optimizer_steps(optimizer.state_dict())
    losses, risk_stats, batches = [], [], []
    model.train()
    for step, ix in enumerate(grouped_batches(rows, seed+int(info['signal_date'].replace('-', '')), cfg['training_batch'])):
        batches.append(ix.tolist())
        a, b, stamps = [torch.as_tensor(np.array(arrays[k][ix]), dtype=torch.long, device=cfg['device']) for k in ['s1', 's2', 'stamps']]
        known = torch.as_tensor(np.array(arrays['valid'][ix]), device=cfg['device'])
        aux = torch.as_tensor(np.array(features[ix]), device=cfg['device']) if expected else None
        optimizer.zero_grad(set_to_none=True)
        ce = loss_parts(model, a, b, stamps, known, 60, history_auxiliary=aux).mean()
        ce.backward()
        k = min(8, len(ix))
        mid, _ = sampled_midpoint_loss(model, decoder, a[:k], b[:k], stamps[:k],
            arrays['mean'][ix[:k]], arrays['scale'][ix[:k]], arrays['future'][ix[:k]], arrays['valid'][ix[:k]],
            arrays['last'][ix[:k], 3], seed=seed*100000+step+(int(info['signal_date'].replace('-', '')) if cfg.get('parent_root') else 0), samples=4, history_auxiliary=None if aux is None else aux[:k])
        (cfg['midpoint_loss_weight']*mid).backward()
        if expected:
            local = rows.iloc[ix].reset_index(drop=True)
            # At most four peers per date and sixteen rows per batch, including sparse replay dates.
            ri = local.groupby('date', sort=False).head(4).index.to_numpy()[:16]
            ids = ix[ri]
            risk, stats = sampled_return_ranking_loss(model, decoder, a[ri], b[ri], stamps[ri],
                arrays['mean'][ids], arrays['scale'][ids], arrays['future'][ids], arrays['valid'][ids],
                local.iloc[ri].date.to_numpy(), seed=seed*1000000+step+(int(info['signal_date'].replace('-', '')) if cfg.get('parent_root') else 0),
                history_auxiliary=aux[ri], cost=cfg['cost'], **cfg['return_ranking_loss'])
            risk.backward()
            if not torch.isfinite(risk):
                raise ValueError('Nonfinite return/ranking loss')
            risk_stats.append(stats)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not all(torch.isfinite(x) for x in [ce, mid, norm]):
            raise ValueError('Nonfinite incremental loss or gradients')
        optimizer.step()
        losses.append(float(ce.detach()))
        if step % 10 == 0:
            print('Update', seed, arm, step, losses[-1], utc_now(), flush=True)
    model.eval().cpu()
    payload = checkpoint_payload(model, daily_optimizer=optimizer.state_dict(), seed=seed, arm=arm,
        trained_labels_through=info['observed_labels_through'], trained_signal_through=info['new_signal_through'],
        parent_sha256=file_hash(parent), incremental_protocol_sha256=digest(cfg))
    torch.save(payload, output/'model.pt')
    loaded, _ = restore_model(output/'model.pt', 'cpu')
    loaded.eval()
    args = [torch.as_tensor(np.array(arrays[k][:2]), dtype=torch.long) for k in ['s1', 's2', 'stamps']]
    aux = forecast_auxiliary(torch.tensor(np.array(features[:2])) if expected else None, 60, 64)
    with torch.inference_mode():
        call = args[0][:, :-1], args[1][:, :-1], args[2][:, :-1], args[0][:, 1:], 59
        for x, y in zip(model.forecast_logits(*call, auxiliary=aux), loaded.forecast_logits(*call, auxiliary=aux)):
            torch.testing.assert_close(x, y, atol=0, rtol=0)
    write_json(output/'training.json', dict(rows=len(rows), epochs=1, steps=len(batches), batches_sha256=digest(batches),
        mean_ce=float(np.mean(losses)), risk=risk_stats, reload_exact=True, parent_sha256=file_hash(parent),
        labels_through=info['observed_labels_through'], finished_at=utc_now(),
        optimizer_steps_before=optimizer_before, optimizer_steps_after=optimizer_steps(optimizer.state_dict()),
        optimizer='continued AdamW for both matched arms' if cfg.get('parent_root') else 'fresh AdamW for both matched arms'))
    write_manifest(output)
    del model, loaded, optimizer
    gc.collect()
    if cfg['device'] == 'mps':
        torch.mps.empty_cache()


def forecast(run, root, cfg, seed, arm, decoder):
    dest = root/f'seed{seed}'/arm/'forecast'
    if (dest/'manifest.json').exists():
        verify(dest)
        return
    dest.mkdir(parents=True, exist_ok=True)
    model, _ = restore_model(dest.parent/'training/model.pt', cfg['device'])
    arrays = load_arrays(run/'inputs', 'forecast')
    ids = np.load(root/'prepared/forecast-ids.npy')
    features = np.load(root/'prepared/forecast-features.npy')
    paths = np.lib.format.open_memmap(dest/'paths.npy', mode='w+', shape=(len(ids), cfg['samples'], 5, 6), dtype=np.float32)
    for start in range(0, len(ids), cfg['forecast_batch']):
        ix = ids[start:start+cfg['forecast_batch']]
        a, b = [torch.tensor(np.array(arrays[k][ix]), dtype=torch.long, device=cfg['device']) for k in ['s1', 's2']]
        stamps = torch.tensor(np.array(arrays['stamps']), dtype=torch.long, device=cfg['device'])[None].expand(len(ix), -1, -1)
        aux = torch.tensor(features[start:start+len(ix)], device=cfg['device']) if model.config.auxiliary_features else None
        pairs = generate_cached(model, a, b, stamps[:, :60], stamps[:, 60:], samples=cfg['samples'], seed=17+start,
            temperature=1., top_p=1., top_k=0, history_auxiliary=aux)
        paths[start:start+len(ix)], _ = decode_cached(decoder, pairs, arrays['mean'][ix], arrays['scale'][ix], 5)
        if start % 1024 == 0:
            print('Forecast progress', seed, arm, start+len(ix), len(ids), utc_now(), flush=True)
    paths.flush()
    write_json(dest/'forecast.json', dict(checkpoint_sha256=file_hash(dest.parent/'training/model.pt'), inputs=len(ids), at=utc_now()))
    write_manifest(dest)
    print('Forecast complete', seed, arm, len(ids), utc_now(), flush=True)
    del model
    gc.collect()
    if cfg['device'] == 'mps':
        torch.mps.empty_cache()


def report(root, run, cfg):
    rows = pd.read_parquet(root/'prepared/forecast-rows.parquet')
    info = read(root/'prepared/input.json')
    summaries, ranks = [], {}
    for arm in ARMS:
        paths = [np.load(root/f'seed{s}'/arm/'forecast/paths.npy', mmap_mode='r') for s in cfg['seeds']]
        records = []
        for i, row in enumerate(rows.itertuples()):
            ref = reference_prices(np.stack([p[i] for p in paths]), cfg['cost'], cfg['minimum_legal_paths'])
            if ref is not None:
                records.append(dict(instrument_id=row.instrument_id, name=row.name, expected_net_return=ref['predicted'],
                    buy_reference_price=ref['buy_reference'], sell_reference_price=ref['sell_reference'],
                    buy_date=info['horizon_dates'][0], sell_reference_date=info['horizon_dates'][ref['sell_offset']]))
        if not records:
            raise ValueError('No legal forecasts')
        frame = pd.DataFrame(records).sort_values(['expected_net_return', 'instrument_id'], ascending=[False, True])
        frame['rank'] = np.arange(1, len(frame)+1)
        frame.to_csv(root/f'{arm}-ranking.csv', index=False)
        ranks[arm] = frame
        summaries.append(dict(arm=arm, ranked=len(frame), top20_expected_net_return_pct=float(frame.head(20).expected_net_return.mean()*100)))
    overlap = len(set(ranks[ARMS[0]].head(20).instrument_id) & set(ranks[ARMS[1]].head(20).instrument_id))
    ranks[ARMS[0]].merge(ranks[ARMS[1]], on='instrument_id', suffixes=('_baseline', '_candidate')).to_csv(root/'comparison.csv', index=False)
    for seed in cfg['seeds']:
        a, b = [read(root/f'seed{seed}'/arm/'training/training.json') for arm in ARMS]
        if a['batches_sha256'] != b['batches_sha256'] or a['rows'] != b['rows']:
            raise ValueError('Unequal training exposure')
    if live_state(run) != read(root/'prepared/live-state.json'):
        raise ValueError('Live state changed during isolated comparison')
    finished = utc_now()
    prospective = bool(cfg.get('parent_root') and pd.Timestamp(finished) < pd.Timestamp(info['horizon_dates'][0]+'T09:15:00', tz='Asia/Shanghai'))
    result = dict(signal=info['signal_date'], labels_through=info['observed_labels_through'], summaries=summaries,
        inputs=len(rows), training_rows=info['training_rows'], top20_overlap=overlap, completed_at=finished,
        prospective=prospective, status='prospective' if prospective else 'late_workflow_validation_only', main_unchanged=True)
    write_json(root/'result.json', result)
    lines = ['# Matched incremental bridge', '',
        f'Signal close: {info["signal_date"]}. Both arms trained through labels available on {info["observed_labels_through"]}.', '',
        f'Each of six fits uses the same {info["training_rows"]:,} rows, one epoch, LR {cfg["learning_rate"]}, {"continued AdamW" if cfg.get("parent_root") else "fresh AdamW"}, matched date-grouped batches and fixed feature normalization. All checkpoints passed exact CPU reload.', '',
        f'{len(rows)} stocks ({"full eligible daily universe" if not cfg["forecast_inputs"] else "exchange/hash sample"}); {cfg["samples"]} paths per stock/model. Top20 overlap: {overlap}/20.', '',
        ('Timely prospective comparison; outcomes pending.' if prospective else 'Late workflow validation only; excluded from prospective scoring.')+' Forecast-implied returns do not measure accuracy or realized gains. Main model and report pointers are unchanged.', '',
        '| Arm | Ranked | Mean forecast-implied Top20 net return |', '| --- | ---: | ---: |']
    for r in summaries:
        lines.append(f'| {r["arm"]} | {r["ranked"]} | {r["top20_expected_net_return_pct"]:.2f}% |')
    lines += ['', '## Candidate references', '', '| Rank | Stock | Expected net return | Buy date | Buy price | Sell date | Sell price |',
              '| ---: | --- | ---: | --- | ---: | --- | ---: |']
    for r in ranks[ARMS[1]].head(20).itertuples():
        lines.append(f'| {r.rank} | {r.name} ({r.instrument_id}) | {r.expected_net_return:.2%} | {r.buy_date} | {r.buy_reference_price:.4f} | {r.sell_reference_date} | {r.sell_reference_price:.4f} |')
    lines += ['', 'Daily extrema are theoretical references, not guaranteed fills. Both parents retain the matched historical experiment; the operational live model has a different training path. This bridge validates equal incremental exposure, not isolated loss causality or future effectiveness. No automatic candidate promotion or recurring execution.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    if cfg.get('parent_root'):
        write_json(root/'publication.json', dict(signal_date=info['signal_date'], horizon_dates=info['horizon_dates'],
            completed_at=finished, prospective=prospective, cost=cfg['cost'], top_n=20,
            ranking_files={'baseline': 'baseline-ranking.csv', 'candidate': 'indicators_ranking-ranking.csv'}))
    write_manifest(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=BASE/'configs/token-ranking-incremental-v1.json')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    cfg = read(args.config)
    run, root = BASE/cfg['daily_run'], BASE/cfg['output']
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent/(root.name+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root/'manifest.json').exists():
            verify(root)
            if read(root/'prepared/binding.json')['config'] != cfg:
                raise ValueError('Completed protocol differs')
            print('Reused', root/'report.md')
            return
        prepare(run, root, cfg)
        if args.prepare_only:
            return
        torch.set_num_threads(4)
        decoder_spec = read(BASE/cfg['experiment']/'profiles.json')['decoder']
        if file_hash(Path(decoder_spec['path'])) != decoder_spec['sha256']:
            raise ValueError('Changed decoder')
        decoder = load_decoder(decoder_spec['path'], cfg['device'])
        for seed in cfg['seeds']:
            for arm in ARMS:
                train(run, root, cfg, seed, arm, decoder)
                forecast(run, root, cfg, seed, arm, decoder)
        report(root, run, cfg)
        print(root/'report.md')


if __name__ == '__main__':
    main()
