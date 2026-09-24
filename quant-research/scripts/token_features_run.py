"""Manual, isolated auxiliary-feature experiments; never schedules or promotes."""
import argparse
import fcntl
import gc
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from token_history_run import DATA, Dataset
from tokenizer_reconstruction_run import load_decoder

from quant_research.forecast_audit import common_scores, score_paths
from quant_research.midpoint_loss import sampled_midpoint_loss
from quant_research.storage import file_hash, utc_now, write_json
from quant_research.token_features import (
    FEATURES,
    GROUPS,
    daily_features,
    fit_normalizer,
    history_features,
)
from quant_research.token_history import loss_parts, partition_indices
from quant_research.token_transformer import (
    checkpoint_payload,
    decode_paths,
    forecast_auxiliary,
    generate_tokens,
    restore_model,
    valid_bars,
    with_auxiliary,
)

BASE = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def verify_manifest(root, name):
    for file, expected in read(root / name)['files'].items():
        if file_hash(root / file) != expected:
            raise ValueError(f'Changed experiment evidence: {file}')



def feature_schema(cfg):
    family = cfg.get('feature_family', 'fundamental')
    if family == 'fundamental':
        return FEATURES, GROUPS
    if family == 'indicators':
        from quant_research.token_indicators import FEATURES as names
        from quant_research.token_indicators import GROUPS as groups
        return names, groups
    if family == 'market_context':
        from quant_research.token_market_context import FEATURES as names
        return names, {'context_ranking': names}
    raise ValueError('Unknown auxiliary feature family')


def prepare_indicators(root, cfg):
    from quant_research.token_indicators import indicator_features
    from quant_research.token_transformer import normalized_history

    cohort = BASE / cfg['cohort_source']
    verify_manifest(cohort, 'completed.json')
    prior = read(cohort / 'protocol.json')
    # Same cohort and budget as the preceding bounded study; never choose rows
    # after observing an indicator or forecast outcome.
    for key in ['training', 'selection', 'evaluation', 'train_per_date',
                'selection_per_date', 'evaluation_per_date', 'epochs', 'batch_size',
                'learning_rate', 'midpoint_loss_weight', 'samples', 'forecast_batch', 'cost', 'seeds']:
        if cfg[key] != prior[key]:
            raise ValueError(f'Indicator protocol must preserve matched {key}')
    root.mkdir(parents=True)
    write_json(root / 'protocol.json', cfg)
    lineage = {}
    def check(path, expected=None):
        h = file_hash(path)
        if expected is not None and h != expected:
            raise ValueError(f'Changed source: {path}')
        lineage[str(path.resolve())] = h
    profiles = read(BASE / cfg['profiles'])
    check(BASE / cfg['profiles'])
    for item in [profiles['decoder'], *[profiles[cfg['owner']]['checkpoints'][str(seed)] for seed in cfg['seeds']]]:
        check(Path(item['path']), item['sha256'])
    data = Dataset()
    meta = read(DATA / 'completed.json')
    for name in ['rows.parquet', 's1.npy', 's2.npy', 'valid.npy', 'future.npy', 'mean.npy', 'scale.npy',
                 'last.npy', 'calendar-stamps.npy', 'calendar.json', 'instruments.json']:
        check(DATA / name, meta['files'][name])
    check(DATA / 'completed.json')
    check(cohort / 'completed.json')
    input_root = BASE / cfg['price_input']
    im = read(input_root / 'manifest.json')
    check(input_root / 'manifest.json')
    check(input_root / 'values.npy', read(DATA / 'sources.json')[str((input_root / 'values.npy').resolve())])
    check(DATA / 'sources.json')
    if im['dates'] != read(DATA / 'calendar.json') or im['instruments'] != read(DATA / 'instruments.json'):
        raise ValueError('Indicator price axes differ from token data')
    raw = np.load(input_root / 'values.npy', mmap_mode='r')
    names, _ = feature_schema(cfg)
    coverage, counts = {}, {}
    for part in ['training', 'selection', 'evaluation']:
        check(cohort / f'{part}-ids.npy')
        ids = np.load(cohort / f'{part}-ids.npy')
        np.save(root / f'{part}-ids.npy', ids)
        rows = data.rows.iloc[ids]
        if rows.date.min() < cfg[part][0] or rows.label_end.max() >= cfg[part][1]:
            raise ValueError('Label embargo failed')
        values = np.lib.format.open_memmap(root / f'{part}-features.npy', mode='w+',
            dtype=np.float32, shape=(len(ids), 60, len(names)))
        for start in range(0, len(ids), 512):
            selected = ids[start:start+512]
            ss, tt = data.rows.iloc[selected][['stock_index', 'date_index']].to_numpy(int).T
            history = raw[ss[:, None], tt[:, None] + np.arange(-59, 1)].copy()
            _, mean, scale = normalized_history(history)
            np.testing.assert_array_equal(mean, data.a['mean'][selected])
            np.testing.assert_array_equal(scale, data.a['scale'][selected])
            values[start:start+len(selected)] = indicator_features(history)
        values.flush()
        coverage[part] = {name: dict(history_observed_fraction=float(np.isfinite(values[:, :, i]).mean()),
            signal_observed_fraction=float(np.isfinite(values[:, -1, i]).mean())) for i, name in enumerate(names)}
        if part == 'training':
            center, scale = fit_normalizer(values)
            np.savez(root / 'normalizer.npz', center=center, scale=scale)
        counts[part] = dict(rows=len(ids), dates=int(rows.date.nunique()), first=rows.date.min(),
            last=rows.date.max(), label_end=rows.label_end.max())
    write_json(root / 'coverage.json', coverage)
    write_json(root / 'splits.json', counts)
    write_json(root / 'profiles.json', profiles)
    write_json(root / 'sources.json', lineage)
    write_json(root / 'live-state.json', {str(p.resolve()): file_hash(p) for p in
        [BASE / 'artifacts/daily-token-live-v1/latest.json', BASE / 'configs/daily-token-v1.json',
         BASE / 'artifacts/daily-token-live-v1/runs/2026-09-15/manifest.json']})
    shutil.copytree(BASE / 'src', root / 'code/src', ignore=shutil.ignore_patterns('__pycache__'))
    (root / 'code/scripts').mkdir()
    for name in ['token_features_run.py', 'token_history_run.py', 'tokenizer_reconstruction_run.py']:
        shutil.copy2(BASE / 'scripts' / name, root / 'code/scripts' / name)
    write_json(root / 'prepared.json', dict(at=utc_now(), files={str(p.relative_to(root)): file_hash(p)
        for p in root.rglob('*') if p.is_file()}))
    print('Prepared indicator histories', counts, flush=True)

def prepare(root, cfg):
    if (root / 'prepared.json').exists():
        if read(root / 'protocol.json') != cfg:
            raise ValueError('Existing output belongs to a different protocol')
        verify_manifest(root, 'prepared.json')
        return
    if root.exists():
        raise ValueError('Incomplete preparation; use a new output path to retain evidence')
    if cfg.get('feature_family') == 'indicators':
        prepare_indicators(root, cfg)
        return
    root.mkdir(parents=True)
    write_json(root / 'protocol.json', cfg)
    profiles = read(BASE / cfg['profiles'])
    lineage = {}
    for item in [profiles['decoder'], *[profiles[cfg['owner']]['checkpoints'][str(seed)] for seed in cfg['seeds']]]:
        if file_hash(Path(item['path'])) != item['sha256']:
            raise ValueError('Initial checkpoint hash mismatch')
        lineage[item['path']] = item['sha256']
    meta = read(DATA / 'completed.json')
    for name in ['rows.parquet', 's1.npy', 's2.npy', 'valid.npy', 'future.npy', 'mean.npy',
                 'scale.npy', 'last.npy', 'calendar-stamps.npy', 'calendar.json', 'quotes.npy', 'instruments.json']:
        path = DATA / name
        if file_hash(path) != meta['files'][name]:
            raise ValueError(f'Historical source changed: {path}')
        lineage[str(path)] = meta['files'][name]
    data = Dataset()
    calendar = read(DATA / 'calendar.json')
    source_root = BASE / cfg['source']
    index = read(source_root / 'index.json')
    sources = []
    for day, item in sorted(index.items()):
        path = source_root / 'daily' / (day + '.parquet')
        capture = source_root / item['capture'] / 'manifest.json'
        for p, h in [(path, item['parquet_sha256']), (capture, item['capture_sha256'])]:
            if file_hash(p) != h:
                raise ValueError(f'Feature capture changed: {p}')
            lineage[str(p)] = h
        frame = pd.read_parquet(path)
        if len(frame) != item['rows'] or not frame.date.eq(day).all():
            raise ValueError('Feature capture row/date mismatch')
        sources.append(frame)
    source = pd.concat(sources, ignore_index=True)
    splits, counts = {}, {}
    for part, limit in [('training', cfg['train_per_date']), ('selection', cfg['selection_per_date']), ('evaluation', cfg['evaluation_per_date'])]:
        ids = partition_indices(data.rows, calendar, *cfg[part])
        rows = data.rows.iloc[ids]
        rows = rows[rows.date.isin(index)].groupby('date', sort=True).head(limit)
        ids = rows.row_id.to_numpy(int)
        # Shared label eligibility; auxiliary missingness never filters the cohort.
        ids = ids[data.a['valid'][ids].any(1)]
        if not len(ids):
            raise ValueError(f'Empty {part} cohort')
        rows = data.rows.iloc[ids]
        splits[part] = ids
        np.save(root / f'{part}-ids.npy', ids)
        counts[part] = dict(rows=len(ids), dates=int(rows.date.nunique()), first=rows.date.min(),
                            last=rows.date.max(), label_end=rows.label_end.max())
    all_ids = np.unique(np.concatenate(list(splits.values())))
    wanted = set(data.rows.iloc[all_ids].instrument_id)
    source = source[source.instrument_id.isin(wanted)]
    instruments = pd.Index(read(DATA / 'instruments.json'))
    si, ti = instruments.get_indexer(source.instrument_id), pd.Index(calendar).get_indexer(source.date)
    inside = (si >= 0) & (ti >= 0)
    source = source.loc[inside].reset_index(drop=True)
    quotes = np.load(DATA / 'quotes.npy', mmap_mode='r')[si[inside], ti[inside]]
    bars = source[['instrument_id', 'date']].assign(close=quotes[:, 3], volume=quotes[:, 4])
    features = daily_features(bars, source, calendar, retrospective=True)
    features.to_parquet(root / 'daily-features.parquet', index=False)
    coverage = {}
    for part, ids in splits.items():
        history = history_features(data.rows.iloc[ids], features, calendar)
        np.save(root / f'{part}-features.npy', history)
        coverage[part] = {name: dict(history_observed_fraction=float(np.isfinite(history[:, :, i]).mean()),
                                    signal_observed_fraction=float(np.isfinite(history[:, -1, i]).mean()))
                          for i, name in enumerate(FEATURES)}
        if part == 'training':
            center, scale = fit_normalizer(history)
            np.savez(root / 'normalizer.npz', center=center, scale=scale)
    write_json(root / 'splits.json', counts)
    write_json(root / 'coverage.json', coverage)
    write_json(root / 'profiles.json', profiles)
    write_json(root / 'sources.json', lineage)
    code = root / 'code'
    shutil.copytree(BASE / 'src', code / 'src', ignore=shutil.ignore_patterns('__pycache__'))
    (code / 'scripts').mkdir()
    for name in ['token_features_run.py', 'token_history_run.py', 'tokenizer_reconstruction_run.py']:
        shutil.copy2(BASE / 'scripts' / name, code / 'scripts' / name)
    write_json(root / 'prepared.json', dict(at=utc_now(), files={str(p.relative_to(root)): file_hash(p)
        for p in root.rglob('*') if p.is_file()}))
    print('Prepared', counts, flush=True)


def auxiliary(features, index, names, device, schema=FEATURES):
    if not names:
        return None
    columns = [schema.index(name) for name in names]
    return torch.as_tensor(np.array(features[index][..., columns]), device=device)


@torch.inference_mode()
def ce_score(model, data, ids, features, cfg):
    model.eval()
    scores = []
    for start in range(0, len(ids), cfg['batch_size']):
        ix = np.arange(start, min(start + cfg['batch_size'], len(ids)))
        aux = auxiliary(features, ix, model.config.auxiliary_features, cfg['device'], feature_schema(cfg)[0])
        scores.extend(loss_parts(model, *data.tensors(ids[ix], cfg['device']), 60,
                                 history_auxiliary=aux).mean(1).cpu().numpy())
    return float(pd.DataFrame(dict(date=data.rows.iloc[ids].date.to_numpy(), score=scores)).groupby('date').score.mean().mean())


def train(root, cfg, data, seed, variant, decoder):
    dest = root / f'seed{seed}' / variant
    if (dest / 'trained.json').exists():
        verify_manifest(dest, 'trained.json')
        return
    dest.mkdir(parents=True, exist_ok=True)
    profiles = read(root / 'profiles.json')
    initial = profiles[cfg['owner']]['checkpoints'][str(seed)]
    model, _ = restore_model(initial['path'], cfg['device'])
    schema, groups = feature_schema(cfg)
    names = groups[variant]
    torch.manual_seed(seed)
    if names:
        columns = [schema.index(name) for name in names]
        with np.load(root / 'normalizer.npz') as z:
            model = with_auxiliary(model, names, z['center'][columns], z['scale'][columns])
    # Reset training RNG after branch construction: matched dropout stream across arms.
    torch.manual_seed(seed)
    train_ids, selection_ids = [np.load(root / f'{p}-ids.npy') for p in ['training', 'selection']]
    train_features, selection_features = [np.load(root / f'{p}-features.npy', mmap_mode='r') for p in ['training', 'selection']]
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=.01)
    history = []
    start_time = time.monotonic()
    for epoch in range(cfg['epochs']):
        order = np.random.default_rng(seed + epoch).permutation(len(train_ids))
        model.train()
        losses = []
        for start in range(0, len(order), cfg['batch_size']):
            ix = order[start:start + cfg['batch_size']]
            ids = train_ids[ix]
            a, b, stamps, known = data.tensors(ids, cfg['device'])
            aux = auxiliary(train_features, ix, names, cfg['device'], schema)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_parts(model, a, b, stamps, known, 60, history_auxiliary=aux).mean()
            loss.backward()
            if cfg['midpoint_loss_weight']:
                k = min(8, len(ids))
                risk, _ = sampled_midpoint_loss(model, decoder, a[:k], b[:k], stamps[:k],
                    data.a['mean'][ids[:k]], data.a['scale'][ids[:k]], data.a['future'][ids[:k]],
                    data.a['valid'][ids[:k]], data.a['last'][ids[:k], 3], seed=seed*100000 + epoch*10000 + start,
                    samples=4, history_auxiliary=None if aux is None else aux[:k])
                (cfg['midpoint_loss_weight'] * risk).backward()
                if not torch.isfinite(risk):
                    raise ValueError('Nonfinite auxiliary objective')
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise ValueError('Nonfinite feature training')
            optimizer.step()
            losses.append(float(loss.detach()))
        score = ce_score(model, data, selection_ids, selection_features, cfg)
        history.append(dict(epoch=epoch + 1, train_ce=float(np.mean(losses)), selection_ce=score,
                            seconds=time.monotonic() - start_time))
        write_json(dest / 'progress.json', dict(seed=seed, variant=variant, **history[-1]))
        print('Training', seed, variant, history[-1], flush=True)
    model.eval().cpu()
    payload = checkpoint_payload(model, parent_sha256=initial['sha256'], auxiliary_source_mode=cfg.get('source_mode', 'retrospective_research'),
        feature_protocol_sha256=file_hash(root / 'protocol.json'), trained_signal_through=read(root / 'splits.json')['training']['last'],
        trained_labels_through=read(root / 'splits.json')['training']['label_end'], seed=seed)
    torch.save(payload, dest / 'model.pt')
    loaded, _ = restore_model(dest / 'model.pt')
    loaded.eval()
    a, b, stamps, _ = data.tensors(selection_ids[:2], 'cpu')
    aux = auxiliary(selection_features, np.arange(2), names, 'cpu', schema)
    with torch.inference_mode():
        args = (a[:, :-1], b[:, :-1], stamps[:, :-1], a[:, 1:], 59)
        padded = forecast_auxiliary(aux, 60, 64)
        for x, y in zip(model.forecast_logits(*args, auxiliary=padded), loaded.forecast_logits(*args, auxiliary=padded)):
            torch.testing.assert_close(x, y, atol=0, rtol=0)
    write_json(dest / 'training.json', dict(history=history, parameters=sum(p.numel() for p in model.parameters()),
        features=list(names), unique_training_rows=len(train_ids), examples=len(train_ids)*cfg['epochs'],
        reload_exact=True, retrospective=True))
    write_json(dest / 'trained.json', dict(files={p: file_hash(dest / p) for p in ['model.pt', 'training.json']}))
    del model, loaded, optimizer
    gc.collect()
    if cfg['device'] == 'mps':
        torch.mps.empty_cache()


def forecast(root, cfg, data, seed, variant, decoder):
    dest = root / f'seed{seed}' / variant
    if (dest / 'forecast.json').exists():
        verify_manifest(dest, 'forecast.json')
        return
    model, _ = restore_model(dest / 'model.pt', cfg['device'])
    ids = np.load(root / 'evaluation-ids.npy')
    features = np.load(root / 'evaluation-features.npy', mmap_mode='r')
    paths = np.lib.format.open_memmap(dest / 'paths.npy', mode='w+', dtype=np.float32,
                                     shape=(len(ids), cfg['samples'], 5, 6))
    for start in range(0, len(ids), cfg['forecast_batch']):
        ix = np.arange(start, min(start + cfg['forecast_batch'], len(ids)))
        a, b, stamps, _ = data.tensors(ids[ix], cfg['device'])
        aux = auxiliary(features, ix, model.config.auxiliary_features, cfg['device'], feature_schema(cfg)[0])
        pairs = generate_tokens(model, a[:, :60], b[:, :60], stamps[:, :60], stamps[:, 60:],
            samples=cfg['samples'], seed=17 + start, top_p=1., history_auxiliary=aux)
        paths[ix], _ = decode_paths(decoder, pairs, data.a['mean'][ids[ix]], data.a['scale'][ids[ix]], 5)
        if start % 256 == 0:
            print('Forecast', seed, variant, start, len(ids), flush=True)
    paths.flush()
    write_json(dest / 'forecast.json', dict(files={'paths.npy': file_hash(dest / 'paths.npy'),
        'model.pt': file_hash(dest / 'model.pt')}, inputs=len(ids), samples=cfg['samples'], at=utc_now()))
    del model, paths
    gc.collect()
    if cfg['device'] == 'mps':
        torch.mps.empty_cache()


def ranking_rows(paths, future, known, rows, cost):
    records = []
    for i, sample in enumerate(paths):
        legal = valid_bars(sample).all(-1)
        if not known[i].all() or legal.sum() < 16:
            continue
        x = sample[legal].astype(float)
        day = int(np.bincount(x[:, 1:, 1].argmax(1), minlength=4).argmax()) + 1
        buy, sell = float(np.median(x[:, 0, 2])), float(np.median(x[:, day, 1]))
        predicted = sell / buy - 1 - cost
        actual = float(future[i, day, 1] / future[i, 0, 2] - 1 - cost)
        records.append(dict(local_row=i, date=rows.iloc[i].date, instrument_id=rows.iloc[i].instrument_id,
                            sell_offset=day, buy_reference=buy, sell_reference=sell,
                            predicted=predicted, actual_extrema_scenario=actual, absolute_error=abs(predicted-actual)))
    return pd.DataFrame(records)


def report(root, cfg, data):
    schema, _ = feature_schema(cfg)
    ids = np.load(root / 'evaluation-ids.npy')
    rows = data.rows.iloc[ids].reset_index(drop=True)
    daily_all, coverage_all, rank_daily = [], [], []
    for seed in cfg['seeds']:
        records, rankings = {}, []
        for variant in cfg['variants']:
            paths = np.load(root / f'seed{seed}' / variant / 'paths.npy', mmap_mode='r')
            records[variant], coverage = score_paths(paths, data.a['future'][ids], data.a['valid'][ids],
                data.a['last'][ids, 3], rows.date.to_numpy())
            coverage_all.append(coverage.assign(seed=seed, variant=variant))
            rankings.append(ranking_rows(paths, data.a['future'][ids], data.a['valid'][ids], rows, cfg['cost']).assign(variant=variant))
        _, daily, _ = common_scores(records)
        daily_all.append(daily.assign(seed=seed))
        rank = pd.concat(rankings, ignore_index=True)
        common = rank.groupby('local_row').variant.nunique()
        rank = rank[rank.local_row.isin(common[common == len(cfg['variants'])].index)]
        rank.to_csv(root / f'seed{seed}-reference-ranking.csv', index=False)
        for (variant, day), g in rank.groupby(['variant', 'date']):
            top = g.sort_values(['predicted', 'instrument_id'], ascending=[False, True]).head(max(1, math.ceil(len(g)*.2)))
            rank_daily.append(dict(seed=seed, variant=variant, date=day, rows=len(g), top_rows=len(top),
                return_mae_pp=float(g.absolute_error.mean()*100), top_extrema_scenario_pct=float(top.actual_extrema_scenario.mean()*100),
                top_lift_pp=float((top.actual_extrema_scenario.mean()-g.actual_extrema_scenario.mean())*100)))
    daily = pd.concat(daily_all, ignore_index=True)
    if daily.empty or not rank_daily:
        raise ValueError('No common forecast cohort')
    daily['window'] = np.where(daily.date < '2025-01-01', '2024H2', '2025H1+July')
    ranking = pd.DataFrame(rank_daily)
    daily.to_csv(root / 'daily-scores.csv', index=False)
    pd.concat(coverage_all).to_csv(root / 'forecast-coverage.csv', index=False)
    ranking.to_csv(root / 'daily-ranking.csv', index=False)
    summary = daily.groupby(['variant', 'horizon', 'target']).mae.mean().unstack(['horizon', 'target'])
    base = daily[daily.variant == 'baseline'].drop(columns='variant')
    comparisons = []
    for variant in cfg['variants'][1:]:
        paired = daily[daily.variant == variant].merge(base, on=['seed', 'date', 'horizon', 'target', 'window'], suffixes=('', '_baseline'))
        primary = paired[paired.target.isin(['maximum', 'minimum'])].copy()
        primary['improvement'] = primary.mae_baseline - primary.mae
        by_date = primary.groupby('date').improvement.mean().sort_index()
        rng = np.random.default_rng(314159)
        n, block = len(by_date), min(cfg['bootstrap_block_dates'], len(by_date))
        draws = []
        for _ in range(cfg['bootstrap_replicates']):
            start = rng.integers(0, n, size=math.ceil(n/block))
            indices = ((start[:, None] + np.arange(block)) % n).ravel()[:n]
            draws.append(float(by_date.to_numpy()[indices].mean()))
        scenarios = primary.groupby(['window', 'horizon', 'target']).improvement.mean()
        comparisons.append(dict(variant=variant, endpoint_mae_improvement_pp=float(by_date.mean()),
            date_win_fraction=float((by_date > 0).mean()), scenarios_improved=int((scenarios > 0).sum()),
            scenarios=len(scenarios), bootstrap_95_low=float(np.quantile(draws, .025)),
            bootstrap_95_high=float(np.quantile(draws, .975))))
    write_json(root / 'comparison.json', comparisons)
    metrics = ranking.groupby('variant')[['return_mae_pp', 'top_extrema_scenario_pct', 'top_lift_pp']].mean()
    lines = ['# Token auxiliary-feature comparison', '',
        'Completed isolated retrospective research; the live daily model and report pointer are unchanged.', '',
        f'Same parent checkpoint, cohorts, {cfg["epochs"]} continuation epochs, CE plus {cfg["midpoint_loss_weight"]} midpoint loss, decoder and {cfg["samples"]}-path sampler across all {len(cfg["variants"])} arms. Normalization uses training histories only; future auxiliary inputs are always missing.', '',
        '## Cohorts', '', json.dumps(read(root / 'splits.json'), ensure_ascii=False, indent=2), '',
        '## High/low error', '', 'MAE in percentage points of the signal close; lower is better. Stock-date cohorts are paired across all arms.', '',
        '| Variant | 2-day high | 2-day low | 5-day high | 5-day low |', '| --- | ---: | ---: | ---: | ---: |']
    for variant in cfg['variants']:
        v = summary.loc[variant]
        lines.append(f'| {variant} | {v[2,"maximum"]:.4f} | {v[2,"minimum"]:.4f} | {v[5,"maximum"]:.4f} | {v[5,"minimum"]:.4f} |')
    lines += ['', '## Breadth of improvement', '', '| Variant | Mean MAE improvement, pp | Dates improved | Scenarios improved | Block-bootstrap 95% interval, pp |', '| --- | ---: | ---: | ---: | --- |']
    for x in comparisons:
        lines.append(f'| {x["variant"]} | {x["endpoint_mae_improvement_pp"]:+.4f} | {x["date_win_fraction"]:.1%} | {x["scenarios_improved"]}/{x["scenarios"]} | [{x["bootstrap_95_low"]:+.4f}, {x["bootstrap_95_high"]:+.4f}] |')
    lines += ['', 'Positive improvement means lower error. Date win fraction describes this sample, not the probability of future effectiveness. Scenarios are two calendar periods × two horizons × two extrema. A mixed subgroup or interval spanning zero does not automatically negate a broad improvement.', '',
        '## Reference-price ranking', '', 'Rank by median T low and median high on the modal predicted exit date; evaluate the actual high on that frozen date. Top 20% of the sampled stock cohort; extrema prices are not guaranteed fills and these figures are not realized returns.', '',
        '| Variant | Return forecast MAE, pp | Top-group extrema scenario | Top-group lift over cohort, pp |', '| --- | ---: | ---: | ---: |']
    for variant in cfg['variants']:
        v = metrics.loc[variant]
        lines.append(f'| {variant} | {v.return_mae_pp:.4f} | {v.top_extrema_scenario_pct:.2f}% | {v.top_lift_pp:+.4f} |')
    lines += ['', '## Input coverage', '', '| Feature | Training history observed | Evaluation signal observed |', '| --- | ---: | ---: |']
    coverage = read(root / 'coverage.json')
    for name in schema:
        lines.append(f'| {name} | {coverage["training"][name]["history_observed_fraction"]:.1%} | {coverage["evaluation"][name]["signal_observed_fraction"]:.1%} |')
    lines += ['', '## Limits', '', cfg['limitations'], '',
        cfg.get('feature_note', 'Rolling features require every session in their window. Sparse captures remain missing; the fifth available record is never treated as five trading days. Historical source revisions and unresolved tokenizer pretraining overlap prevent an unbiased prospective performance claim.'), '',
        'Full per-date error, raw legal-path coverage, selected exit dates and reference prices are retained in the adjacent CSV files.', '']
    (root / 'report.md').write_text('\n'.join(lines))
    write_json(root / 'completed.json', dict(at=utc_now(), passed=True, live_promoted=False,
        files={str(p.relative_to(root)): file_hash(p) for p in root.rglob('*') if p.is_file() and p.name != 'completed.json'}))
    print('Completed', root / 'report.md', flush=True)


def run(cfg, prepare_only=False):
    root = BASE / cfg['output']
    if (root / 'completed.json').exists():
        if read(root / 'protocol.json') != cfg:
            raise ValueError('Protocol differs from completed run')
        verify_manifest(root, 'completed.json')
        print('Already completed', root / 'report.md')
        return
    prepare(root, cfg)
    if prepare_only:
        return
    for name, expected in read(root / 'sources.json').items():
        if file_hash(Path(name)) != expected:
            raise ValueError(f'Changed source: {name}')
    torch.set_num_threads(4)
    data = Dataset()
    decoder = load_decoder(read(root / 'profiles.json')['decoder']['path'], cfg['device'])
    for seed in cfg['seeds']:
        for variant in cfg['variants']:
            train(root, cfg, data, seed, variant, decoder)
            forecast(root, cfg, data, seed, variant, decoder)
    report(root, cfg, data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=BASE / 'configs/token-features-v1.json')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    cfg = read(args.config)
    root = BASE / cfg['output']
    root.parent.mkdir(parents=True, exist_ok=True)
    with root.with_suffix('.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('This feature experiment is already running') from None
        run(cfg, args.prepare_only)


if __name__ == '__main__':
    main()
