"""Bounded, manual three-arm reference-return experiment; no live promotion."""
import argparse
import fcntl
import gc
import math
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from token_features_run import auxiliary, ce_score, feature_schema, read, verify_manifest
from token_history_run import DATA, Dataset
from tokenizer_reconstruction_run import load_decoder

from quant_research.midpoint_loss import sampled_midpoint_loss
from quant_research.return_ranking_loss import reference_prices, sampled_return_ranking_loss
from quant_research.storage import file_hash, utc_now, write_json
from quant_research.token_cached_inference import decode_cached, generate_cached
from quant_research.token_features import fit_normalizer
from quant_research.token_history import loss_parts, partition_indices
from quant_research.token_indicators import FEATURES, indicator_features
from quant_research.token_transformer import (
    checkpoint_payload,
    forecast_auxiliary,
    normalized_history,
    restore_model,
    with_auxiliary,
)

BASE = Path(__file__).resolve().parents[1]
PARTS = ['training', 'selection', 'evaluation']


def seal(root, name, paths=None, **extra):
    paths = paths or [p for p in root.rglob('*') if p.is_file() and p.name != name]
    write_json(root/name, dict(at=utc_now(), **extra,
        files={str(p.relative_to(root)): file_hash(p) for p in paths}))


def prepare(root, cfg):
    if (root/'prepared.json').exists():
        if read(root/'protocol.json') != cfg:
            raise ValueError('Protocol changed; use a new experiment directory')
        verify_manifest(root, 'prepared.json')
        return
    if root.exists():
        raise ValueError('Partial preparation retained; choose a new output directory')
    prior = BASE/cfg['cohort_source']
    verify_manifest(prior, 'completed.json')
    data = Dataset()
    root.mkdir(parents=True)
    write_json(root/'protocol.json', cfg)
    lineage = {}
    def check(path, expected=None):
        path = Path(path)
        actual = file_hash(path)
        if expected is not None and actual != expected:
            raise ValueError(f'Changed source {path}')
        lineage[str(path.resolve())] = actual
    profiles = read(BASE/cfg['profiles'])
    check(BASE/cfg['profiles'])
    for item in [profiles['decoder'], *[profiles[cfg['owner']]['checkpoints'][str(s)] for s in cfg['seeds']]]:
        check(item['path'], item['sha256'])
    check(prior/'completed.json')
    meta = read(DATA/'completed.json')
    for name in ['rows.parquet', 's1.npy', 's2.npy', 'valid.npy', 'future.npy', 'mean.npy', 'scale.npy',
                 'last.npy', 'calendar-stamps.npy', 'calendar.json', 'instruments.json']:
        check(DATA/name, meta['files'][name])
    check(DATA/'completed.json')
    source = BASE/cfg['price_input']
    im = read(source/'manifest.json')
    check(source/'manifest.json')
    check(source/'values.npy', read(DATA/'sources.json')[str((source/'values.npy').resolve())])
    check(DATA/'sources.json')
    dates = read(DATA/'calendar.json')
    if im['dates'] != dates or im['instruments'] != read(DATA/'instruments.json'):
        raise ValueError('Raw price axes differ from token data')
    raw = np.load(source/'values.npy', mmap_mode='r')
    counts, coverage = {}, {}
    for part in PARTS:
        check(prior/f'{part}-ids.npy')
        original = np.load(prior/f'{part}-ids.npy')
        if part == 'training':
            ids = original.copy()  # Fixed original training budget.
        else:
            pool = data.rows.iloc[partition_indices(data.rows, dates, *cfg[part])]
            # Expand by the archived, outcome-independent exchange/hash order.
            pool = pool[pool.date.isin(data.rows.iloc[original].date.unique())]
            ids = pool.groupby('date', sort=True).head(cfg['evaluation_per_date']).row_id.to_numpy(int)
        rows = data.rows.iloc[ids]
        if rows.date.min() < cfg[part][0] or rows.label_end.max() >= cfg[part][1]:
            raise ValueError('Five-session label embargo failed')
        np.save(root/f'{part}-ids.npy', ids)
        features = np.lib.format.open_memmap(root/f'{part}-features.npy', mode='w+',
            shape=(len(ids), 60, len(FEATURES)), dtype=np.float32)
        vols = []
        for start in range(0, len(ids), 512):
            ix = ids[start:start+512]
            ss, tt = data.rows.iloc[ix][['stock_index', 'date_index']].to_numpy(int).T
            history = raw[ss[:, None], tt[:, None]+np.arange(-59, 1)].copy()
            _, mean, scale = normalized_history(history)
            np.testing.assert_array_equal(mean, data.a['mean'][ix])
            np.testing.assert_array_equal(scale, data.a['scale'][ix])
            features[start:start+len(ix)] = indicator_features(history)
            close = history[:, -21:, 3].astype(float)*history[:, -21:, 6]
            vols.extend(np.diff(np.log(close), axis=1).std(axis=1, ddof=0)*100)
        features.flush()
        np.save(root/f'{part}-volatility.npy', np.asarray(vols))
        if part == 'training':
            center, scale = fit_normalizer(features)
            np.savez(root/'normalizer.npz', center=center, scale=scale)
        coverage[part] = {name: float(np.isfinite(features[:, -1, j]).mean()) for j, name in enumerate(FEATURES)}
        per_date = rows.groupby('date').size()
        counts[part] = dict(rows=len(ids), dates=int(rows.date.nunique()), first=rows.date.min(),
            last=rows.date.max(), label_end=rows.label_end.max(), per_date_min=int(per_date.min()),
            per_date_max=int(per_date.max()), full_label_rows=int(data.a['valid'][ids].all(1).sum()))
    write_json(root/'splits.json', counts)
    write_json(root/'coverage.json', coverage)
    write_json(root/'profiles.json', profiles)
    write_json(root/'sources.json', lineage)
    live = BASE/'artifacts/daily-token-live-v1'
    write_json(root/'live-state.json', {str(p): file_hash(p) for p in
        [BASE/'configs/daily-token-v1.json', live/'latest.json', live/'current.json', live/'runs/2026-09-15/manifest.json']})
    shutil.copytree(BASE/'src', root/'code/src', ignore=shutil.ignore_patterns('__pycache__'))
    (root/'code/scripts').mkdir()
    for name in ['token_ranking_run.py', 'token_features_run.py', 'token_history_run.py', 'tokenizer_reconstruction_run.py']:
        shutil.copy2(BASE/'scripts'/name, root/'code/scripts'/name)
    seal(root, 'prepared.json')
    print('Prepared', counts, flush=True)


def date_batches(rows, seed, epoch, dates_per_batch=4):
    rng = np.random.default_rng(seed+epoch)
    groups = [rng.permutation(g.index.to_numpy()) for _, g in rows.groupby('date', sort=True)]
    order = rng.permutation(len(groups))
    for start in range(0, len(order), dates_per_batch):
        group = [groups[i] for i in order[start:start+dates_per_batch]]
        yield np.concatenate(group)


def train(root, cfg, data, seed, arm, decoder):
    dest = root/f'seed{seed}'/arm
    if (dest/'trained.json').exists():
        verify_manifest(dest, 'trained.json')
        return
    dest.mkdir(parents=True, exist_ok=True)
    initial = read(root/'profiles.json')[cfg['owner']]['checkpoints'][str(seed)]
    model, _ = restore_model(initial['path'], cfg['device'])
    schema = feature_schema(cfg)[0]
    names = () if arm == 'baseline' else schema
    torch.manual_seed(seed)
    if names:
        with np.load(root/'normalizer.npz') as z:
            model = with_auxiliary(model, names, z['center'], z['scale'])
    torch.manual_seed(seed)
    ids, selected = [np.load(root/f'{p}-ids.npy') for p in ['training', 'selection']]
    features, selected_features = [np.load(root/f'{p}-features.npy', mmap_mode='r') for p in ['training', 'selection']]
    rows = data.rows.iloc[ids].reset_index(drop=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=.01)
    history, risk_rows = [], set()
    started = time.monotonic()
    for epoch in range(cfg['epochs']):
        model.train()
        losses, risks = [], []
        for step, ix in enumerate(date_batches(rows, seed, epoch, cfg['dates_per_batch'])):
            selected_ids = ids[ix]
            a, b, stamps, known = data.tensors(selected_ids, cfg['device'])
            aux = auxiliary(features, ix, names, cfg['device'], schema)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_parts(model, a, b, stamps, known, 60, history_auxiliary=aux).mean()
            loss.backward()
            k = min(8, len(ix))
            risk, _ = sampled_midpoint_loss(model, decoder, a[:k], b[:k], stamps[:k],
                data.a['mean'][selected_ids[:k]], data.a['scale'][selected_ids[:k]], data.a['future'][selected_ids[:k]],
                data.a['valid'][selected_ids[:k]], data.a['last'][selected_ids[:k], 3],
                seed=seed*100000+epoch*10000+step, samples=4, history_auxiliary=None if aux is None else aux[:k])
            (cfg['midpoint_loss_weight']*risk).backward()
            if arm in ('indicators_ranking', 'context_ranking'):
                local = rows.iloc[ix].reset_index(drop=True)
                full = data.a['valid'][selected_ids].all(1)
                rix = local[full].groupby('date', sort=False).head(cfg['risk_rows_per_date']).index.to_numpy()
                # Date groups with insufficient complete labels cannot form a pair.
                if len(rix):
                    ri = selected_ids[rix]
                    ranking, stats = sampled_return_ranking_loss(model, decoder, a[rix], b[rix], stamps[rix],
                        data.a['mean'][ri], data.a['scale'][ri], data.a['future'][ri], data.a['valid'][ri],
                        local.iloc[rix].date.to_numpy(), seed=seed*1000000+epoch*10000+step,
                        history_auxiliary=aux[rix], cost=cfg['cost'], **cfg['return_ranking_loss'])
                    ranking.backward()
                    if not torch.isfinite(ranking):
                        raise ValueError('Nonfinite reference-return loss')
                    risks.append(stats)
                    risk_rows.update(ri.tolist())
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(loss) or not torch.isfinite(risk) or not torch.isfinite(norm):
                raise ValueError('Nonfinite training objective or gradients')
            optimizer.step()
            losses.append(float(loss.detach()))
        score = ce_score(model, data, selected, selected_features, cfg)
        entry = dict(epoch=epoch+1, train_ce=float(np.mean(losses)), selection_ce=score,
            seconds=time.monotonic()-started, risk={k: float(np.mean([r[k] for r in risks])) for k in risks[0]} if risks else {})
        history.append(entry)
        write_json(dest/'progress.json', entry)
        print('Training', seed, arm, entry, flush=True)
    model.eval().cpu()
    payload = checkpoint_payload(model, parent_sha256=initial['sha256'], seed=seed,
        trained_signal_through=read(root/'splits.json')['training']['last'],
        trained_labels_through=read(root/'splits.json')['training']['label_end'],
        ranking_protocol_sha256=file_hash(root/'protocol.json'), arm=arm)
    torch.save(payload, dest/'model.pt')
    loaded, _ = restore_model(dest/'model.pt')
    loaded.eval()
    a, b, stamps, _ = data.tensors(selected[:2], 'cpu')
    aux = auxiliary(selected_features, np.arange(2), names, 'cpu', schema)
    with torch.inference_mode():
        args = a[:, :-1], b[:, :-1], stamps[:, :-1], a[:, 1:], 59
        padded = forecast_auxiliary(aux, 60, 64)
        for x, y in zip(model.forecast_logits(*args, auxiliary=padded), loaded.forecast_logits(*args, auxiliary=padded)):
            torch.testing.assert_close(x, y, atol=0, rtol=0)
    write_json(dest/'training.json', dict(history=history, parameters=sum(p.numel() for p in model.parameters()),
        unique_training_rows=len(ids), examples=len(ids)*cfg['epochs'], reference_loss_unique_rows=len(risk_rows),
        reload_exact=True, final_epoch=cfg['epochs'], parent=initial, features=names))
    seal(dest, 'trained.json', [dest/'model.pt', dest/'training.json'])
    del model, loaded, optimizer
    gc.collect()
    if cfg['device'] == 'mps':
        torch.mps.empty_cache()


def forecast(root, cfg, data, seed, arm, decoder, part):
    model_root = root/f'seed{seed}'/arm
    dest = model_root/part
    if (dest/'forecast.json').exists():
        verify_manifest(dest, 'forecast.json')
        return
    dest.mkdir(parents=True, exist_ok=True)
    model, _ = restore_model(model_root/'model.pt', cfg['device'])
    ids = np.load(root/f'{part}-ids.npy')
    features = np.load(root/f'{part}-features.npy', mmap_mode='r')
    paths = np.lib.format.open_memmap(dest/'paths.npy', mode='w+', dtype=np.float32,
        shape=(len(ids), cfg['samples'], 5, 6))
    for start in range(0, len(ids), cfg['forecast_batch']):
        ix = np.arange(start, min(start+cfg['forecast_batch'], len(ids)))
        a, b, stamps, _ = data.tensors(ids[ix], cfg['device'])
        aux = auxiliary(features, ix, model.config.auxiliary_features, cfg['device'], FEATURES)
        pairs = generate_cached(model, a[:, :60], b[:, :60], stamps[:, :60], stamps[:, 60:],
            samples=cfg['samples'], seed=17+start, temperature=1., top_p=1., top_k=0, history_auxiliary=aux)
        paths[ix], _ = decode_cached(decoder, pairs, data.a['mean'][ids[ix]], data.a['scale'][ids[ix]], 5)
        if start % 2048 == 0:
            print('Forecast', part, seed, arm, start, len(ids), flush=True)
    paths.flush()
    seal(dest, 'forecast.json', [dest/'paths.npy'], checkpoint_sha256=file_hash(model_root/'model.pt'),
         inputs=len(ids), samples=cfg['samples'])
    del model, paths
    gc.collect()
    if cfg['device'] == 'mps':
        torch.mps.empty_cache()


def ranking_records(paths, future, known, rows, volatility, cfg):
    result, coverage = [], []
    for i in range(len(rows)):
        ref = reference_prices(np.stack([p[i] for p in paths]), cfg['cost'], cfg['minimum_legal_paths'])
        coverage.append(dict(local_row=i, forecast_eligible=ref is not None, label_known=bool(known[i].all())))
        if ref is None or not known[i].all():
            continue
        actual = future[i, ref['sell_offset'], 1]/future[i, 0, 2]-1-cfg['cost']
        result.append(dict(local_row=i, instrument_id=rows.iloc[i].instrument_id, date=rows.iloc[i].date,
            **{k: v for k, v in ref.items() if k != 'legal_paths'}, actual_extrema_scenario=float(actual),
            absolute_error=abs(ref['predicted']-actual), volatility_pp=float(volatility[i]),
            adverse_excursion_pct=float((future[i, :ref['sell_offset']+1, 2].min()/future[i, 0, 2]-1)*100)))
    return pd.DataFrame(result), pd.DataFrame(coverage)


def daily_metrics(rank, top_n=20):
    output = []
    for (arm, seed, day), g in rank.groupby(['arm', 'seed', 'date']):
        if len(g) < top_n:
            continue
        top = g.sort_values(['predicted', 'instrument_id'], ascending=[False, True]).head(top_n)
        exchange = top.instrument_id.str.split('.').str[1].value_counts(normalize=True)
        vcut = g.volatility_pp.quantile(.8)
        output.append(dict(arm=arm, seed=seed, date=day, rows=len(g), top_rows=len(top),
            return_mae_pp=float(g.absolute_error.mean()*100),
            top_extrema_scenario_pct=float(top.actual_extrema_scenario.mean()*100),
            top_lift_pp=float((top.actual_extrema_scenario.mean()-g.actual_extrema_scenario.mean())*100),
            rank_ic=float(g.predicted.rank().corr(g.actual_extrema_scenario.rank())),
            top_q10_pct=float(top.actual_extrema_scenario.quantile(.1)*100),
            top_adverse_excursion_pct=float(top.adverse_excursion_pct.mean()),
            top_high_volatility_fraction=float((top.volatility_pp >= vcut).mean()),
            largest_exchange_fraction=float(exchange.max()),
            top_prediction_bias_pp=float((top.predicted-top.actual_extrema_scenario).mean()*100)))
    return pd.DataFrame(output)


def summarize(root, cfg, data, part):
    ids = np.load(root/f'{part}-ids.npy')
    rows = data.rows.iloc[ids].reset_index(drop=True)
    volatility = np.load(root/f'{part}-volatility.npy')
    results, coverage = [], []
    for arm in cfg['variants']:
        arrays = {seed: np.load(root/f'seed{seed}'/arm/part/'paths.npy', mmap_mode='r') for seed in cfg['seeds']}
        for seed in [*cfg['seeds'], 'ensemble']:
            paths = list(arrays.values()) if seed == 'ensemble' else [arrays[seed]]
            rank, cov = ranking_records(paths, data.a['future'][ids], data.a['valid'][ids], rows, volatility, cfg)
            results.append(rank.assign(arm=arm, seed=str(seed)))
            coverage.append(cov.assign(arm=arm, seed=str(seed)))
    ranks = pd.concat(results, ignore_index=True)
    pd.concat(coverage, ignore_index=True).to_csv(root/f'{part}-coverage.csv', index=False)
    ranks.to_csv(root/f'{part}-all-ranking.csv', index=False)
    # Same stock-date cohort across every arm AND seed, including the ensemble.
    counts = ranks.groupby('local_row').size()
    expected = len(cfg['variants'])*(len(cfg['seeds'])+1)
    common = counts[counts == expected].index
    ranks = ranks[ranks.local_row.isin(common)]
    ranks.to_csv(root/f'{part}-paired-ranking.csv', index=False)
    daily = daily_metrics(ranks, cfg['top_n'])
    if daily.empty:
        raise ValueError('No dates with 20 paired forecasts')
    daily.to_csv(root/f'{part}-daily.csv', index=False)
    metrics = daily.groupby(['arm', 'seed']).mean(numeric_only=True).reset_index()
    metrics.to_csv(root/f'{part}-summary.csv', index=False)
    return daily, metrics


def comparisons(daily, cfg):
    out = []
    metric_names = ['top_extrema_scenario_pct', 'top_lift_pp', 'return_mae_pp', 'top_q10_pct', 'rank_ic']
    for seed in [str(s) for s in cfg['seeds']]+['ensemble']:
        subset = daily[daily.seed == seed]
        for left, right in [('indicators', 'baseline'), ('indicators_ranking', 'baseline'), ('indicators_ranking', 'indicators')]:
            a, b = [subset[subset.arm == arm].set_index('date').sort_index() for arm in [left, right]]
            if not a.index.equals(b.index):
                raise ValueError('Unpaired dates')
            for metric in metric_names:
                delta = (a[metric]-b[metric])*(-1 if metric == 'return_mae_pp' else 1)
                values = delta.to_numpy()
                rng = np.random.default_rng(314159)
                n, block = len(values), min(cfg['bootstrap_block_dates'], len(values))
                samples = []
                for _ in range(cfg['bootstrap_replicates']):
                    starts = rng.integers(0, n, size=math.ceil(n/block))
                    indices = ((starts[:, None]+np.arange(block)) % n).ravel()[:n]
                    samples.append(values[indices].mean())
                out.append(dict(seed=seed, arm=left, comparator=right, metric=metric, improvement=float(delta.mean()),
                    dates=n, date_win_fraction=float((delta > 1e-10).mean()),
                    bootstrap_95=[float(np.quantile(samples, .025)), float(np.quantile(samples, .975))]))
    return out


def choose_shadow(metrics, cfg):
    # Frozen decision rule: candidate with highest selection-ensemble Top20 lift.
    # Shadow-only nomination does not change the main model or assert efficacy.
    candidates = metrics[(metrics.seed == 'ensemble') & (metrics.arm != 'baseline')]
    candidate = candidates.sort_values(['top_lift_pp', 'arm'], ascending=[False, True]).iloc[0]
    return str(candidate.arm)


def finalize(root, cfg, data):
    sd, sm = summarize(root, cfg, data, 'selection')
    # Nomination is written BEFORE evaluation reporting and uses selection only.
    nominated = choose_shadow(sm, cfg)
    selection = dict(arm=nominated, rule=cfg['nomination_rule'], at=utc_now(),
        selection_summary_sha256=file_hash(root/'selection-summary.csv'), live_promoted=False)
    write_json(root/'nomination.json', selection)
    ed, em = summarize(root, cfg, data, 'evaluation')
    comp = comparisons(ed, cfg)
    write_json(root/'comparison.json', comp)
    for part, frame in [('selection', sd), ('evaluation', ed)]:
        frame.assign(period=np.where(frame.date < '2025-01-01', '2024', '2025')).groupby(
            ['arm', 'seed', 'period']).mean(numeric_only=True).to_csv(root/f'{part}-periods.csv')
        frame.groupby(['arm', 'seed']).top_extrema_scenario_pct.quantile(.1).to_csv(root/f'{part}-poor-date-decile.csv')
    lines = ['# Reference-return and ranking loss comparison', '',
        'Matched historical development experiment. Top20 from up to 192 sampled stocks per date; not a whole-market test or executed return.', '',
        'The published score is sell reference / buy reference - 1 - 0.25%. Exit date is chosen from forecast paths before outcomes; it is never replaced by the best observed exit date.', '',
        '## Protocol', '', f'{len(cfg["seeds"])} seeds × three arms × {cfg["epochs"]} fixed epochs. Identical training rows, grouped batches, initialization within seed, decoder and 64-path inference. Selection CE does not stop training.', '',
        'Baseline: CE + 0.05 sampled midpoint loss. Indicators: identical loss plus six causal BOLL/MACD/KDJ inputs. Indicators + ranking: adds 0.01 return Huber and 0.01 within-date pair ranking cost, normalized by 5 percentage points. Two independent bags of eight real decoded paths provide score-function gradients; no soft token or separate ranking head.', '',
        '## Evaluation ensemble', '',
        '| Arm | Return MAE, pp | Top20 extrema scenario | Lift over paired universe, pp | Top20 10th percentile | High-volatility share |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in em[em.seed == 'ensemble'].itertuples():
        lines.append(f'| {r.arm} | {r.return_mae_pp:.4f} | {r.top_extrema_scenario_pct:.2f}% | {r.top_lift_pp:+.4f} | {r.top_q10_pct:.2f}% | {r.top_high_volatility_fraction:.1%} |')
    lines += ['', '## Paired ensemble differences', '', '| Arm vs comparator | Metric | Improvement | Dates improved | Date-block 95% interval |', '| --- | --- | ---: | ---: | --- |']
    for x in comp:
        if x['seed'] == 'ensemble':
            lo, hi = x['bootstrap_95']
            lines.append(f'| {x["arm"]} vs {x["comparator"]} | {x["metric"]} | {x["improvement"]:+.4f} | {x["date_win_fraction"]:.1%} | [{lo:+.4f}, {hi:+.4f}] |')
    lines += ['', f'Selection-only shadow nomination: **{nominated}**. This is a candidate for prospective comparison, not automatic promotion.', '',
        'See per-seed summaries, calendar-period results, poor-date deciles, forecast availability, prediction bias and concentration in the adjacent CSV files. Positive improvement means better; return MAE uses reversed sign. Date frequency is descriptive, not probability of future success. Mixed metrics and intervals crossing zero do not automatically reject a candidate.', '',
        '## Limitations', '', cfg['limitations'], '',
        'The return/ranking objective uses each bag\'s own selected exit day, so outcomes depend on the forecast action. Pairwise targets and medians are detached; whole-bag log probabilities carry gradients. Independent-bag baselines exclude the current draw. Illegal bags are penalized, not silently removed from the training loss. Only complete observed five-day labels enter that loss.', '',
        'Daily main checkpoints and report pointers remain unchanged. Candidate checkpoints stay frozen for shadow forecasts; they are older than the incrementally updated live model, so that operational comparison also measures training-recency differences. Only forecasts completed before T opens qualify as prospective. Subsequent labels must mature before review.', '']
    (root/'report.md').write_text('\n'.join(lines))
    for path, expected in read(root/'live-state.json').items():
        if file_hash(Path(path)) != expected:
            raise ValueError(f'Live state changed during experiment: {path}')
    seal(root, 'completed.json', passed=True, live_promoted=False)
    shadow = dict(version='token-ranking-shadow-v1', experiment=str(root), experiment_sha256=file_hash(root/'completed.json'),
        arm=nominated, checkpoints={str(seed): dict(path=str(root/f'seed{seed}'/nominated/'model.pt'),
            sha256=file_hash(root/f'seed{seed}'/nominated/'model.pt')) for seed in cfg['seeds']},
        decoder=read(root/'profiles.json')['decoder'], samples=cfg['samples'], forecast_batch=cfg['forecast_batch'],
        cost=cfg['cost'], top_n=20, minimum_legal_paths=cfg['minimum_legal_paths'], device=cfg['device'],
        store=str(BASE/'artifacts/token-ranking-shadow-v1'), fixed_weights=True)
    write_json(root.parent/(root.name+'-shadow.json'), shadow)
    print('Completed', root/'report.md', 'shadow', nominated, flush=True)


def run(cfg, prepare_only=False):
    root = BASE/cfg['output']
    if (root/'completed.json').exists():
        if read(root/'protocol.json') != cfg:
            raise ValueError('Completed protocol differs')
        verify_manifest(root, 'completed.json')
        print('Already completed', root/'report.md', flush=True)
        return
    prepare(root, cfg)
    if prepare_only:
        return
    for path, expected in read(root/'sources.json').items():
        if file_hash(Path(path)) != expected:
            raise ValueError(f'Changed source {path}')
    torch.set_num_threads(4)
    data = Dataset()
    decoder = load_decoder(read(root/'profiles.json')['decoder']['path'], cfg['device'])
    for seed in cfg['seeds']:
        for arm in cfg['variants']:
            train(root, cfg, data, seed, arm, decoder)
            for part in ['selection', 'evaluation']:
                forecast(root, cfg, data, seed, arm, decoder, part)
    finalize(root, cfg, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=BASE/'configs/token-ranking-v2.json')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    cfg = read(args.config)
    root = BASE/cfg['output']
    root.parent.mkdir(parents=True, exist_ok=True)
    with root.with_suffix('.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Ranking experiment is already running') from None
        run(cfg, args.prepare_only)
