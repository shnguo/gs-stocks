"""Record immutable candidate forecasts beside a completed manual daily run."""
import argparse
import fcntl
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tokenizer_reconstruction_run import load_decoder

from quant_research.daily_loop import digest, read, verify, write_manifest
from quant_research.daily_token import FIELDS, future_known, load_arrays, panel
from quant_research.return_ranking_loss import reference_prices
from quant_research.storage import file_hash, utc_now, write_json
from quant_research.token_cached_inference import decode_cached, generate_cached
from quant_research.token_indicators import FEATURES, indicator_features
from quant_research.token_transformer import (
    normalized_history,
    restore_model,
)


def is_prospective(completed_at, first_day, baseline_prospective):
    stamp = pd.Timestamp(completed_at)
    if stamp.tzinfo is None:
        raise ValueError('Timezone-aware completion time required')
    return bool(baseline_prospective and stamp < pd.Timestamp(first_day+'T09:15:00', tz='Asia/Shanghai'))


def review_shadow(store, source):
    """Review only immutable, timely forecasts after all five labels mature."""
    meta, bars, _ = panel(source)
    observed = meta['price_data_through']
    by_symbol = {s: g.set_index('date') for s, g in bars.groupby('instrument_id')}
    results = []
    for run in sorted((store/'runs').glob('*')):
        if not (run/'manifest.json').exists():
            continue
        verify(run)
        info = read(run/'publication.json')
        if not info['prospective']:
            results.append(dict(signal=run.name, status='late_excluded'))
            continue
        if info['horizon_dates'][-1] > observed:
            results.append(dict(signal=run.name, status='pending_labels', mature_after=info['horizon_dates'][-1]))
            continue
        destination = store/'reviews'/run.name/file_hash(Path(source)/'snapshot/manifest.json')[:16]
        if (destination/'manifest.json').exists():
            verify(destination)
            results.append(read(destination/'review.json'))
            continue
        known_rows, summaries = [], []
        dates = [info['signal_date'], *info['horizon_dates']]
        ranking_files = info.get('ranking_files', {'current': 'current-ranking.csv', 'candidate': 'candidate-ranking.csv'})
        for arm, filename in ranking_files.items():
            ranking = pd.read_csv(run/filename)
            for row in ranking.itertuples():
                frame = by_symbol.get(row.instrument_id)
                if frame is None:
                    continue
                block = frame.reindex(dates)
                if not future_known(block, 0, 5):
                    continue
                price = block[FIELDS].to_numpy(float)
                day = dates.index(row.sell_reference_date)
                actual = price[day, 1]/price[1, 2]-1-info['cost']
                known_rows.append(dict(arm=arm, instrument_id=row.instrument_id, rank=row.rank,
                    expected=row.expected_net_return, actual_extrema_scenario=float(actual),
                    absolute_error=abs(row.expected_net_return-actual)))
            subset = [r for r in known_rows if r['arm'] == arm]
            top = [r for r in subset if r['rank'] <= info['top_n']]
            summaries.append(dict(arm=arm, total=len(ranking), known=len(subset), unknown=len(ranking)-len(subset),
                top_known=len(top), top_expected=min(info['top_n'], len(ranking)),
                return_mae_pp=float(np.mean([r['absolute_error'] for r in subset])*100) if subset else None,
                top_extrema_scenario_pct=float(np.mean([r['actual_extrema_scenario'] for r in top])*100) if top else None))
        destination.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(known_rows).to_csv(destination/'rows.csv', index=False)
        result = dict(signal=run.name, status='reviewed', observed_through=observed, metrics=summaries,
            source_manifest_sha256=file_hash(Path(source)/'snapshot/manifest.json'),
            publication_sha256=file_hash(run/'manifest.json'), at=utc_now(),
            interpretation='Frozen Top20 and exit dates; actual extrema scenarios, not executed returns. Missing labels are reported, not replaced by lower-ranked stocks.')
        write_json(destination/'review.json', result)
        write_manifest(destination)
        results.append(result)
    write_json(store/'review-latest.json', dict(at=utc_now(), observed_through=observed, results=results))
    return results


def prepare_shadow_features(run, arrays, rows):
    input_info = read(run/'inputs/input.json')
    source = Path(input_info['source'])
    meta, bars, _ = panel(source)
    if file_hash(source/'snapshot/manifest.json') != input_info['snapshot_sha256']:
        raise ValueError('Daily source snapshot changed')
    signal = input_info['signal_date']
    if meta['price_data_through'] != signal or bars.date.max() != signal:
        raise ValueError('Shadow source must stop at the daily signal close')
    end = meta['dates'].index(signal)
    history_dates = meta['dates'][end-59:end+1]
    groups = {s: g.set_index('date') for s, g in bars.groupby('instrument_id')}
    output = np.empty((len(rows), 60, len(FEATURES)), np.float32)
    for start in range(0, len(rows), 512):
        symbols = rows.instrument_id.iloc[start:start+512]
        # Match prepare_inputs: each raw history is copied in C order before stacking.
        # NumPy's float32 reduction order can differ for a Fortran-layout view.
        raw = np.stack([groups[s].reindex(history_dates)[FIELDS].to_numpy(float).copy() for s in symbols])
        _, mean, scale = normalized_history(raw)
        np.testing.assert_array_equal(mean, arrays['mean'][start:start+len(symbols)])
        np.testing.assert_array_equal(scale, arrays['scale'][start:start+len(symbols)])
        output[start:start+len(symbols)] = indicator_features(raw)
    return output, source


def record(run, cfg):
    run = Path(run).resolve()
    verify(run)
    info = read(run/'run.json')
    signal = info['signal_date']
    store = Path(cfg['store'])
    destination = store/'runs'/signal
    if (destination/'manifest.json').exists():
        verify(destination)
        binding = read(destination/'binding.json')
        if binding['config_sha256'] != digest(cfg) or binding['daily_manifest_sha256'] != file_hash(run/'manifest.json'):
            raise ValueError('Existing shadow run has a different candidate or daily source')
        review_shadow(store, Path(read(run/'inputs/input.json')['source']))
        return destination/'report.md'
    experiment = Path(cfg['experiment'])
    if file_hash(experiment/'completed.json') != cfg['experiment_sha256']:
        raise ValueError('Changed experiment evidence')
    evidence = cfg.get('nomination_evidence')
    if not evidence:
        raise ValueError('Complete the independently verified frozen-Top20 nomination first')
    evidence_path = Path(evidence['path'])
    if file_hash(evidence_path) != evidence['sha256']:
        raise ValueError('Changed candidate nomination evidence')
    verify(evidence_path.parent, 'completed.json')
    if read(evidence_path.parent/'nomination.json')['arm'] != cfg['arm']:
        raise ValueError('Candidate differs from the selection-only nomination')
    if cfg['cost'] != info['cost_scenario']:
        raise ValueError('Candidate and current return costs differ')
    publication = run.parent.parent/'reference-publications'/(signal+'.json')
    pointer = read(publication)
    current_view = Path(pointer['report'])
    verify(current_view)
    if file_hash(current_view/'manifest.json') != pointer['manifest_sha256']:
        raise ValueError('Current publication changed')
    delivery = read(current_view/'delivery.json')
    if delivery['run_manifest_sha256'] != file_hash(run/'manifest.json') or delivery['ranking_method'] != 't_low_to_future_high':
        raise ValueError('Current publication uses another run or ranking formula')
    binding = dict(config_sha256=digest(cfg), daily_manifest_sha256=file_hash(run/'manifest.json'),
        current_publication_sha256=pointer['manifest_sha256'], code_sha256=file_hash(Path(__file__)))
    destination.mkdir(parents=True, exist_ok=True)
    if (destination/'binding.json').exists() and read(destination/'binding.json') != binding:
        raise ValueError('Partial shadow run binding differs')
    write_json(destination/'binding.json', binding)
    write_json(destination/'config.json', cfg)
    arrays = load_arrays(run/'inputs', 'forecast')
    rows = pd.read_parquet(run/'inputs/forecast-rows.parquet')
    features, source = prepare_shadow_features(run, arrays, rows)
    np.save(destination/'history-features.npy', features)
    decoder_spec = cfg['decoder']
    if file_hash(Path(decoder_spec['path'])) != decoder_spec['sha256']:
        raise ValueError('Changed decoder')
    torch.set_num_threads(4)
    decoder = load_decoder(decoder_spec['path'], cfg['device'])
    for seed, spec in cfg['checkpoints'].items():
        if file_hash(Path(spec['path'])) != spec['sha256']:
            raise ValueError('Changed candidate checkpoint')
        output = destination/f'seed{seed}'
        if (output/'manifest.json').exists():
            verify(output)
            continue
        output.mkdir(exist_ok=True)
        model, saved = restore_model(spec['path'], cfg['device'])
        if tuple(model.config.auxiliary_features) != FEATURES or saved['trained_labels_through'] > signal:
            raise ValueError('Candidate feature schema or training cutoff differs')
        paths = np.lib.format.open_memmap(output/'paths.npy', mode='w+', dtype=np.float32,
            shape=(len(rows), cfg['samples'], 5, 6))
        for start in range(0, len(rows), cfg['forecast_batch']):
            end = min(start+cfg['forecast_batch'], len(rows))
            a, b = [torch.as_tensor(np.array(arrays[k][start:end]), device=cfg['device'], dtype=torch.long) for k in ['s1', 's2']]
            stamps = torch.as_tensor(np.array(arrays['stamps']), device=cfg['device'], dtype=torch.long)[None].expand(end-start, -1, -1)
            aux = torch.as_tensor(features[start:end], device=cfg['device'])
            pairs = generate_cached(model, a, b, stamps[:, :60], stamps[:, 60:], samples=cfg['samples'],
                seed=17+start, temperature=1., top_p=1., top_k=0, history_auxiliary=aux)
            paths[start:end], _ = decode_cached(decoder, pairs, arrays['mean'][start:end], arrays['scale'][start:end], 5)
            if start % 2048 == 0:
                print('Shadow', signal, seed, start, len(rows), flush=True)
        paths.flush()
        write_json(output/'forecast.json', dict(checkpoint=spec, at=utc_now(), inputs=len(rows)))
        write_manifest(output)
        del model, paths
        gc.collect()
        if cfg['device'] == 'mps':
            torch.mps.empty_cache()
    models = [np.load(destination/f'seed{seed}/paths.npy', mmap_mode='r') for seed in cfg['checkpoints']]
    records = []
    for i, row in enumerate(rows.itertuples()):
        ref = reference_prices(np.stack([m[i] for m in models]), cfg['cost'], cfg['minimum_legal_paths'])
        if ref is not None:
            records.append(dict(instrument_id=row.instrument_id, name=row.name,
                expected_net_return=ref['predicted'], buy_reference_price=ref['buy_reference'],
                sell_reference_price=ref['sell_reference'], buy_date=info['horizon_dates'][0],
                sell_reference_date=info['horizon_dates'][ref['sell_offset']], sell_date_frequency=ref['sell_date_frequency']))
    if not records:
        raise ValueError('No legal candidate forecasts')
    candidate = pd.DataFrame(records).sort_values(['expected_net_return', 'instrument_id'], ascending=[False, True])
    candidate['rank'] = np.arange(1, len(candidate)+1)
    candidate.to_csv(destination/'candidate-ranking.csv', index=False)
    current = pd.read_csv(current_view/'ranking.csv')
    current.to_csv(destination/'current-ranking.csv', index=False)
    comparison = current.merge(candidate, on='instrument_id', suffixes=('_current', '_candidate'), validate='one_to_one')
    comparison.to_csv(destination/'comparison.csv', index=False)
    finished = utc_now()
    prospective = is_prospective(finished, info['horizon_dates'][0], delivery['prospective'])
    publication_info = dict(signal_date=signal, horizon_dates=info['horizon_dates'], completed_at=finished,
        prospective=prospective, arm=cfg['arm'], top_n=cfg['top_n'], cost=cfg['cost'], inputs=len(rows),
        candidate_ranked=len(candidate), current_ranked=len(current), shared_ranked=len(comparison),
        current_run=str(run), candidate_frozen=True, candidate_training_recency_differs=True)
    write_json(destination/'publication.json', publication_info)
    lines = ['# Manual candidate comparison', '', f'Signal close: {signal}. Candidate: {cfg["arm"]}.', '',
        'Prospective, completed before the first forecast opening auction.' if prospective else 'Late reconstruction: excluded from prospective performance.', '',
        'Both rankings use sell reference / buy reference - 1 - costs, with the forecast-selected exit date. Values are extrema scenarios, not executed profits.', '',
        'Candidate weights are frozen historical research checkpoints; the current model is incrementally updated. This operational comparison also reflects their different training recency.', '',
        '| Rank | Stock | Expected net return | Buy date | Buy reference | Sell date | Sell reference |',
        '| ---: | --- | ---: | --- | ---: | --- | ---: |']
    for r in candidate.head(cfg['top_n']).itertuples():
        lines.append(f'| {r.rank} | {r.name} ({r.instrument_id}) | {r.expected_net_return:.2%} | {r.buy_date} | {r.buy_reference_price:.4f} | {r.sell_reference_date} | {r.sell_reference_price:.4f} |')
    overlap = len(set(current.head(cfg['top_n']).instrument_id) & set(candidate.head(cfg['top_n']).instrument_id))
    lines += ['', f'Top20 overlap with the current model: {overlap}/20.', '',
        f'[Current published report]({current_view / "report.md"}) · [Full side-by-side comparison]({destination / "comparison.csv"})', '',
        'Full current/candidate rankings are retained alongside this report. Outcomes remain pending until the five-day labels mature.']
    (destination/'report.md').write_text('\n'.join(lines)+'\n')
    write_manifest(destination)
    review_shadow(store, source)
    return destination/'report.md'


def run_shadow(run, config):
    cfg = read(config)
    store = Path(cfg['store'])
    store.mkdir(parents=True, exist_ok=True)
    with (store/'shadow.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Candidate comparison already running') from None
        return record(run, cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, help='Existing verified daily run')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--review-source', type=Path, help='Review earlier forecasts using this verified source; do not forecast')
    args = parser.parse_args()
    if args.review_source:
        cfg = read(args.config)
        print(review_shadow(Path(cfg['store']), args.review_source))
    elif args.run:
        print(run_shadow(args.run, args.config))
    else:
        parser.error('--run or --review-source is required')
