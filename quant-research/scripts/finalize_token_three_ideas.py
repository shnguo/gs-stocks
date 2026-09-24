"""Date-streamed reporting, frozen Top20 and independent raw-path checks."""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from finalize_token_ranking import freeze_and_score
from token_features_run import read, verify_manifest
from token_ranking_run import seal
from verify_token_ranking import independent_stats

from quant_research.return_ranking_loss import reference_prices
from quant_research.storage import file_hash, utc_now, write_json


def path_rows(root, cfg, seed, arm, indices):
    """Read only requested rows from immutable, independently hashed chunks."""
    destination = root/'forecasts'/f'seed{seed}'/arm
    out = np.empty((len(indices), cfg['samples'], 5, 6), np.float32)
    chunk = indices//cfg['chunk_rows']*cfg['chunk_rows']
    for start in np.unique(chunk):
        mask = chunk == start
        values = np.load(destination/f'{start:07d}.npy', mmap_mode='r')
        out[mask] = values[indices[mask]-start]
    return out


def compare(daily, cfg):
    rows = []
    metrics = ['return_mae_pp', 'top_extrema_scenario_pct', 'top_lift_pp', 'top_q10_pct',
               'rank_ic', 'top_adverse_excursion_pct', 'top_prediction_bias_pp']
    for seed in [str(s) for s in cfg['seeds']]+['ensemble']:
        for candidate, control in cfg['comparisons']:
            a, b = [daily[(daily.seed == seed) & (daily.arm == arm)].set_index('date').sort_index()
                    for arm in [candidate, control]]
            if not a.index.equals(b.index):
                raise ValueError('Unpaired evaluation dates')
            for metric in metrics:
                delta = a[metric]-b[metric]
                if metric == 'return_mae_pp':
                    delta = -delta
                elif metric == 'top_prediction_bias_pp':
                    delta = b[metric].abs()-a[metric].abs()
                delta = delta.dropna()
                values = delta.to_numpy()
                n = len(values)
                if not n:
                    continue
                block = min(cfg['bootstrap_block_dates'], n)
                rng = np.random.default_rng(314159)
                starts = rng.integers(0, n, (cfg['bootstrap_replicates'], math.ceil(n/block)))
                ix = ((starts[:, :, None]+np.arange(block)) % n).reshape(len(starts), -1)[:, :n]
                means = values[ix].mean(1)
                rows.append(dict(seed=seed, candidate=candidate, control=control, metric=metric,
                    improvement=float(values.mean()), median_improvement=float(np.median(values)),
                    dates=n, dates_improved=int((values > 1e-10).sum()),
                    date_win_fraction=float((values > 1e-10).mean()),
                    bootstrap_95=np.quantile(means, [.025, .975]).tolist()))
    return rows


def summarize_daily(daily, dest, name):
    daily.to_csv(dest/f'{name}-daily.csv', index=False)
    summary = daily.groupby(['arm', 'seed']).mean(numeric_only=True).reset_index()
    summary.to_csv(dest/f'{name}-summary.csv', index=False)
    daily.assign(period=np.where(daily.date < '2025-01-01', '2024H2', '2025H1_July')).groupby(
        ['arm', 'seed', 'period']).mean(numeric_only=True).to_csv(dest/f'{name}-periods.csv')
    daily.groupby(['arm', 'seed']).top_extrema_scenario_pct.quantile(.1).to_csv(dest/f'{name}-poor-date-decile.csv')
    ensemble = daily[daily.seed == 'ensemble']
    full = ensemble.groupby('date').top_unknown.sum().eq(0)
    complete = daily[daily.date.isin(full[full].index)]
    complete.groupby(['arm', 'seed']).mean(numeric_only=True).to_csv(dest/f'{name}-complete-top20-dates.csv')
    return summary


def finalize(root):
    cfg = read(root/'protocol.json')
    verify_manifest(root, 'prepared.json')
    binding = file_hash(root/'prepared.json')
    for seed in cfg['seeds']:
        for arm in cfg['variants']:
            dest = root/'forecasts'/f'seed{seed}'/arm
            verify_manifest(dest, 'completed.json')
            if read(dest/'completed.json')['identity']['prepared_sha256'] != binding:
                raise ValueError('Forecast input identity mismatch')
            for marker in sorted(dest.glob('[0-9]*.json')):
                record = read(marker)
                if file_hash(marker.with_suffix('.npy')) != record['sha256']:
                    raise ValueError('Forecast paths changed')
    dataset = root/'evaluation-inputs'
    rows = pd.read_parquet(dataset/'rows.parquet')
    future = np.load(dataset/'future.npy', mmap_mode='r')
    valid = np.load(dataset/'valid.npy', mmap_mode='r')
    volatility = np.load(dataset/'volatility.npy', mmap_mode='r')
    dest = root/'validation'
    dest.mkdir(exist_ok=True)
    for mode in ['native', 'paired']:
        (dest/mode).mkdir(exist_ok=True)
    native_days, paired_days, coverage = [], [], []
    raw_checks, frozen_checks = 0, 0
    for day, group in rows.groupby('date', sort=True):
        indices = group.row_id.to_numpy(int)
        known = valid[indices].all(1)
        targets = np.asarray(future[indices])
        frames = []
        for arm in cfg['variants']:
            arrays = {str(s): path_rows(root, cfg, s, arm, indices) for s in cfg['seeds']}
            stats = {s: independent_stats(p, cfg['minimum_legal_paths']) for s, p in arrays.items()}
            for seed in [str(s) for s in cfg['seeds']]+['ensemble']:
                selected = list(stats) if seed == 'ensemble' else [seed]
                used = [stats[s] for s in selected]
                eligible = np.logical_and.reduce([x[0] for x in used])
                exit_day = np.mean([x[1] for x in used], axis=0).argmax(1)+1
                buy = np.mean([x[2] for x in used], axis=0)
                sell = np.mean([x[3] for x in used], axis=0)[np.arange(len(indices)), exit_day-1]
                predicted = sell/buy-1-cfg['cost']
                # Freeze forecast-only membership, rank and selected exit BEFORE labels.
                f = pd.DataFrame(dict(local_row=indices, instrument_id=group.instrument_id.to_numpy(),
                    date=day, arm=arm, seed=seed, sell_offset=exit_day, buy_reference=buy,
                    sell_reference=sell, predicted=predicted, volatility_pp=volatility[indices]))
                f = f[eligible].sort_values(['predicted', 'instrument_id'], ascending=[False, True])
                f['rank'] = np.arange(1, len(f)+1)
                frozen = f[['local_row', 'rank', 'sell_offset']].copy()
                outcome = targets[np.arange(len(indices)), exit_day, 1]/targets[:, 0, 2]-1-cfg['cost']
                adverse = np.array([(targets[i, :exit_day[i]+1, 2].min()/targets[i, 0, 2]-1)*100
                                    if known[i] else np.nan for i in range(len(indices))])
                f['label_known'] = known[f.index]
                f['actual_extrema_scenario'] = np.where(known, outcome, np.nan)[f.index]
                f['absolute_error'] = np.abs(predicted-np.where(known, outcome, np.nan))[f.index]
                f['adverse_excursion_pct'] = adverse[f.index]
                pd.testing.assert_frame_equal(frozen, f[['local_row', 'rank', 'sell_offset']])
                frames.append(f)
                coverage.append(dict(date=day, arm=arm, seed=seed, input_rows=len(indices),
                    forecast_eligible=int(eligible.sum()), future_labels_known=int(known.sum())))
                # Scalar production formula versus the independent vectorized statistics.
                probes = np.unique(np.concatenate([np.linspace(0, len(indices)-1, 16, dtype=int), f.head(20).index]))
                for i in probes:
                    ref = reference_prices(np.stack([arrays[s][i] for s in selected]), cfg['cost'], cfg['minimum_legal_paths'])
                    if ref is None:
                        assert not eligible[i]
                    else:
                        assert eligible[i] and ref['sell_offset'] == exit_day[i]
                        np.testing.assert_allclose([buy[i], sell[i], predicted[i]],
                            [ref[k] for k in ['buy_reference', 'sell_reference', 'predicted']], rtol=1e-12, atol=1e-12)
                    raw_checks += 1
        frame = pd.concat(frames, ignore_index=True)
        native, nd = freeze_and_score(frame, cfg['top_n'])
        counts = frame.groupby('local_row').size()
        shared = counts[counts == len(cfg['variants'])*(len(cfg['seeds'])+1)].index
        paired, pdaily = freeze_and_score(frame[frame.local_row.isin(shared)], cfg['top_n'])
        for mode, rankings, daily in [('native', native, nd), ('paired', paired, pdaily)]:
            hidden = rankings.copy()
            hidden['label_known'] = False
            for name in ['actual_extrema_scenario', 'absolute_error', 'adverse_excursion_pct']:
                hidden[name] = np.nan
            reranked, _ = freeze_and_score(hidden, cfg['top_n'])
            keys = ['arm', 'seed', 'local_row']
            pd.testing.assert_series_equal(rankings.set_index(keys)['rank'].sort_index(),
                                           reranked.set_index(keys)['rank'].sort_index())
            for key, g in rankings.groupby(['arm', 'seed', 'date']):
                metric = daily.set_index(['arm', 'seed', 'date']).loc[key]
                top = g.sort_values('rank').head(cfg['top_n'])
                visible = top[top.label_known]
                assert metric.top_known == len(visible) and metric.top_unknown == cfg['top_n']-len(visible)
                np.testing.assert_allclose(metric.top_extrema_scenario_pct,
                    visible.actual_extrema_scenario.mean()*100, rtol=1e-12, equal_nan=True)
            rankings.to_parquet(dest/mode/f'{day}.parquet', index=False)
            frozen_checks += len(rankings)
        native_days.append(nd)
        paired_days.append(pdaily)
        write_json(root/'progress.json', dict(stage='auditing', date=day, at=utc_now()))
        print('Audited', day, len(indices), 'paired', len(shared), flush=True)
    native = pd.concat(native_days, ignore_index=True)
    paired = pd.concat(paired_days, ignore_index=True)
    native_summary = summarize_daily(native, dest, 'native')
    paired_summary = summarize_daily(paired, dest, 'paired')
    pd.DataFrame(coverage).to_csv(dest/'forecast-coverage.csv', index=False)
    comparisons = compare(paired, cfg)
    write_json(dest/'paired-comparisons.json', comparisons)
    write_json(dest/'native-comparisons.json', compare(native, cfg))
    write_json(dest/'verification.json', dict(passed=True, at=utc_now(), raw_path_reference_checks=raw_checks,
        frozen_rank_checks=frozen_checks, unknown_top20_not_replaced=True, live_promoted=False,
        source_cohort_rows=len(rows), evaluation_dates=int(rows.date.nunique()),
        all_three_ideas_measured=True, source_input_parity=read(root/'input-parity.json')))
    lines = ['# Three-idea validation: full archived universe', '',
        'The three questions are return/ranking loss, full-universe Top20 robustness, and added index/market/sector context. All four frozen arms are evaluated on the same 71 historical dates and 378,079 input-eligible stock-date rows, with three seeds and 64 paths per model.', '',
        'Earlier baseline, indicator and ranking checkpoints are reused. The context arm receives the same original parent, training rows, four epochs and losses as the ranking control. Its 18 extra features appear only at the signal-close token. Unknown sectors remain missing and the target stock is excluded from sector peers.', '',
        '## Paired ensemble results', '',
        '| Arm | Return MAE (pp) | Top20 observed scenario | Top20 q10 | Known Top20 | High-volatility share |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in paired_summary[paired_summary.seed == 'ensemble'].itertuples():
        lines.append(f'| {r.arm} | {r.return_mae_pp:.4f} | {r.top_extrema_scenario_pct:.2f}% | {r.top_q10_pct:.2f}% | {r.top_known:.2f}/20 | {r.top_high_volatility_fraction:.1%} |')
    lines += ['', '## Improvement magnitude and frequency', '',
        '| Candidate vs control | Metric | Mean improvement | Dates improved | Date-block 95% interval |',
        '| --- | --- | ---: | ---: | --- |']
    for r in comparisons:
        if r['seed'] == 'ensemble':
            lo, hi = r['bootstrap_95']
            lines.append(f'| {r["candidate"]} vs {r["control"]} | {r["metric"]} | {r["improvement"]:+.4f} | {r["dates_improved"]}/{r["dates"]} ({r["date_win_fraction"]:.1%}) | [{lo:+.4f}, {hi:+.4f}] |')
    lines += ['', 'Positive improvements are favorable; MAE and absolute bias reverse sign. Date frequencies describe this historical sample, not the probability of future effectiveness. An interval crossing zero does not by itself negate an improvement.', '',
        'Native-universe, individual-seed, calendar-period, poor-date-decile and complete-Top20-date sensitivity results accompany this report. Paired results use stocks forecast-eligible in every arm and seed. Native results retain each arm’s own forecast eligibility. Missing outcomes never change membership, rank or exit date; observed Top20 means can still be biased by informative missingness.', '',
        'The expected score remains sell reference / buy reference − 1 − 0.25%. The actual scenario uses the observed high on the forecast-selected day divided by the observed T low, not the best observed future day. These are daily-extrema scenarios, not executed profits.', '',
        '## Native ensemble sensitivity', '', native_summary[native_summary.seed == 'ensemble'].to_csv(index=False), '',
        '## Limitations', '', cfg['limitations'], '',
        'No live model or report pointer was changed. Evaluation did not tune the architecture, losses, checkpoint epoch or features.', '']
    (dest/'report.md').write_text('\n'.join(lines))
    for path, expected in read(root/'live-state.json').items():
        if file_hash(Path(path)) != expected:
            raise ValueError(f'Protected live state changed: {path}')
    seal(dest, 'completed.json', passed=True)
    write_json(root/'completed.json', dict(passed=True, at=utc_now(), all_three_ideas_validated=True,
        prepared_sha256=binding, validation_manifest_sha256=file_hash(dest/'completed.json'),
        live_promoted=False, report=str(dest/'report.md')))
    print('Completed', dest/'report.md', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    finalize(parser.parse_args().root)
