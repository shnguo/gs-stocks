"""Manual, resumable full-universe validation of three frozen research ideas."""
import argparse
import fcntl
import gc
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from token_features_run import auxiliary, read, verify_manifest
from token_history_run import BUNDLE, DATA, Dataset
from token_ranking_run import seal, train
from tokenizer_reconstruction_run import load_decoder

from quant_research.industry_context import read_snapshot
from quant_research.storage import file_hash, utc_now, write_json
from quant_research.token_cached_inference import decode_cached, generate_cached
from quant_research.token_features import fit_normalizer
from quant_research.token_history import future_labels
from quant_research.token_indicators import FEATURES as INDICATORS
from quant_research.token_market_context import (
    CONTEXT,
    FEATURES,
    contextual_history,
    dated_industries,
    signal_context,
)
from quant_research.token_transformer import (
    load_tokenizer,
    normalized_history,
    restore_model,
    valid_bars,
)

BASE = Path(__file__).resolve().parents[1]


def index_history(directory, dates):
    meta, records = [read(directory/name) for name in ['manifest.json', 'records.json']]
    if (meta['status'] != 'completed' or meta['query'] != dict(api='daily', code='sh.000300',
            start_date='2020-01-01', end_date='2025-07-29') or not records['pagination_complete']):
        raise ValueError('Incomplete or incorrectly scoped index capture')
    for spec in meta['raw_files']:
        if file_hash(directory/spec['file']) != spec['sha256']:
            raise ValueError('Changed index raw evidence')
    if file_hash(directory/'records.json') != meta['records_sha256']:
        raise ValueError('Changed index records')
    frame = pd.DataFrame(records['rows'], columns=records['fields'])
    if frame.date.duplicated().any() or not frame.code.eq('sh.000300').all():
        raise ValueError('Duplicate or foreign index records')
    bars = frame[['open', 'high', 'low', 'close', 'volume', 'amount']].astype(float).to_numpy()
    if not valid_bars(bars).all() or not frame.adjustflag.eq('3').all():
        raise ValueError('Invalid unadjusted index bars')
    expected = [d for d in dates if '2020-01-01' <= d <= '2025-07-29']
    if sorted(frame.date) != expected:
        raise ValueError('Index history does not exactly cover the research exchange calendar')
    return pd.Series(bars[:, 3], index=frame.date).reindex(dates).to_numpy()


def cohort_rows(coords, data, ids, dates, symbols):
    """Expand the already frozen dates before any future-label availability check."""
    wanted = data.rows.iloc[ids].date_index.unique()
    coords = np.asarray(coords[np.isin(coords[:, 1], wanted)])
    rows = pd.DataFrame(coords, columns=['stock_index', 'date_index'])
    rows['instrument_id'] = np.asarray(symbols)[rows.stock_index]
    rows['date'] = np.asarray(dates)[rows.date_index]
    rows = rows.sort_values(['date', 'instrument_id']).reset_index(drop=True)
    if rows.duplicated(['stock_index', 'date_index']).any():
        raise ValueError('Duplicate inherited input cohort')
    rows['label_end'] = np.asarray(dates)[rows.date_index+5]
    rows['row_id'] = np.arange(len(rows))
    return rows


def prepare(root, cfg):
    if (root/'prepared.json').exists():
        if read(root/'protocol.json') != cfg:
            raise ValueError('Protocol changed')
        verify_manifest(root, 'prepared.json')
        return
    # Preparation can be rerun deterministically after interruption; forecasts
    # cannot start until the complete dataset and lineage have been sealed.
    root.mkdir(parents=True, exist_ok=True)
    if (root/'protocol.json').exists() and read(root/'protocol.json') != cfg:
        raise ValueError('Use a new output for a new protocol')
    write_json(root/'protocol.json', cfg)
    prior = BASE/cfg['prior']
    verify_manifest(prior, 'prepared.json')
    data = Dataset()
    dates, symbols = [read(DATA/f'{name}.json') for name in ['calendar', 'instruments']]
    lineage = {}
    def check(path, expected=None):
        value = file_hash(path)
        if expected is not None and value != expected:
            raise ValueError(f'Changed source {path}')
        lineage[str(path.resolve())] = value
    historical = read(DATA/'completed.json')['files']
    for name in ['rows.parquet', 'all-eligible-coordinates.npy', 'quotes.npy', 'priced.npy',
                 'sequence_id.npy', 'label_sequence_id.npy', 'actions.npy', 'calendar.json',
                 'instruments.json', 'calendar-stamps.npy', *[f'{k}.npy' for k in data.a]]:
        check(DATA/name, historical[name])
    for path, expected in read(prior/'sources.json').items():
        check(Path(path), expected)
    check(prior/'completed.json')
    profiles = read(prior/'profiles.json')
    write_json(root/'profiles.json', profiles)
    for item in [profiles['decoder'], *profiles[cfg['owner']]['checkpoints'].values()]:
        check(Path(item['path']), item['sha256'])
    for seed in cfg['seeds']:
        for arm in cfg['variants'][:-1]:
            source = prior/f'seed{seed}'/arm
            verify_manifest(source, 'trained.json')
            check(source/'model.pt', read(source/'trained.json')['files']['model.pt'])
    raw_path = BASE/cfg['price_input']/'values.npy'
    check(raw_path, read(DATA/'sources.json')[str(raw_path.resolve())])
    raw = np.load(raw_path, mmap_mode='r')
    coords = np.load(DATA/'all-eligible-coordinates.npy', mmap_mode='r')
    arrays = {k: np.load(DATA/f'{k}.npy', mmap_mode='r') for k in
              ['quotes', 'priced', 'sequence_id', 'label_sequence_id', 'actions']}
    industry_root = BASE/cfg['industry_source']
    target, captures = [read(industry_root/name) for name in ['target-source-map.json', 'verified-snapshots.json']]
    for name in ['target-source-map.json', 'verified-snapshots.json', 'source-completion-verification.json']:
        check(industry_root/name)
    index_root = BASE/cfg['index_source']
    index = index_history(index_root, dates)
    for path in index_root.iterdir():
        if path.is_file():
            check(path)
    write_json(root/'index-verification.json', dict(passed=True, matched_sessions=int(np.isfinite(index).sum()),
        no_forward_fill=True, source=str(index_root), information_vintage='retrospective_reconstructed'))
    source_ids = {part: np.load(prior/f'{part}-ids.npy') for part in ['training', 'selection', 'evaluation']}
    features = {}
    for part in ['training', 'selection']:
        np.save(root/f'{part}-ids.npy', source_ids[part])
        features[part] = np.lib.format.open_memmap(root/f'{part}-features.npy', mode='w+',
            dtype=np.float32, shape=(len(source_ids[part]), 60, len(FEATURES)))
    eval_rows = cohort_rows(coords, data, source_ids['evaluation'], dates, symbols)
    if (len(eval_rows) != 378079 or eval_rows.date.nunique() != 71
            or eval_rows.label_end.max() >= cfg['evaluation'][1]):
        raise ValueError('Frozen full-universe cohort or embargo changed')
    dataset = root/'evaluation-inputs'
    dataset.mkdir(exist_ok=True)
    eval_rows.to_parquet(dataset/'rows.parquet', index=False)
    n = len(eval_rows)
    specs = dict(s1=(np.int16, (n, 65)), s2=(np.int16, (n, 65)), valid=(bool, (n, 5)),
        future=(np.float64, (n, 5, 6)), mean=(np.float32, (n, 1, 6)), scale=(np.float32, (n, 1, 6)),
        last=(np.float32, (n, 6)), indicators=(np.float32, (n, 60, len(INDICATORS))),
        context=(np.float32, (n, len(CONTEXT))), volatility=(np.float64, (n,)))
    outputs = {k: np.lib.format.open_memmap(dataset/f'{k}.npy', mode='w+', dtype=dtype, shape=shape)
               for k, (dtype, shape) in specs.items()}
    tokenizer = load_tokenizer(BUNDLE, cfg['device'])
    sampled = {p: data.rows.iloc[ids].reset_index(drop=True) for p, ids in source_ids.items()}
    needed = sorted(set().union(*[set(r.date) for r in sampled.values()]))
    grouped = pd.DataFrame(coords, columns=['stock_index', 'date_index'])
    grouped = grouped[grouped.date_index.isin([dates.index(d) for d in needed])].groupby('date_index')
    eval_groups = eval_rows.groupby('date').indices
    coverage, parity = [], []
    for number, day in enumerate(needed):
        t = dates.index(day)
        ss = grouped.get_group(t).stock_index.to_numpy(int)
        h = raw[ss[:, None], t+np.arange(-59, 1)].copy()
        snapshot = read_snapshot(Path(captures[target[day]]), target[day])
        for name in ['manifest.json', 'records.json']:
            check(Path(captures[target[day]])/name)
        industries = dated_industries(snapshot, np.asarray(symbols)[ss], day, cfg['maximum_industry_age_days'])
        context = signal_context(h, industries, index[t-20:t+1], cfg['minimum_industry_peers'])
        positions = pd.Series(np.arange(len(ss)), index=ss)
        coverage.append(dict(date=day, input_cohort=len(ss), industry_source_date=target[day],
            industry_known=int((industries != '').sum()), sector_peers_available=int(np.isfinite(context[:, 10]).sum())))
        for part in ['training', 'selection']:
            local = np.flatnonzero(sampled[part].date.to_numpy() == day)
            if len(local):
                ix = positions.loc[sampled[part].iloc[local].stock_index].to_numpy()
                features[part][local] = contextual_history(h[ix], context[ix])
                # Preserve the old six-feature branch exactly.
                original = np.load(prior/f'{part}-features.npy', mmap_mode='r')
                np.testing.assert_array_equal(features[part][local, :, :len(INDICATORS)], original[local])
        if day in eval_groups:
            local = eval_groups[day]
            source_positions = positions.loc[eval_rows.iloc[local].stock_index].to_numpy()
            for start in range(0, len(local), cfg['encoding_batch']):
                ix = local[start:start+cfg['encoding_batch']]
                hi = source_positions[start:start+len(ix)]
                history = h[hi].copy()
                normalized, mean, scale = normalized_history(history)
                # Encode a fixed unknown future suffix; the prefix is causal.
                padded = np.concatenate([normalized, np.zeros((len(ix), 5, 6), np.float32)], axis=1)
                with torch.inference_mode():
                    a, b = tokenizer.encode(torch.as_tensor(padded, device=cfg['device']), half=True)
                tokens = [a.cpu().numpy(), b.cpu().numpy()]
                # Archive has sampled prefixes encoded with real future labels;
                # exact shared-stock token parity detects any future leakage.
                selected = sampled['evaluation'][sampled['evaluation'].date.eq(day)]
                joined = selected.merge(eval_rows.iloc[ix][['stock_index', 'row_id']], on='stock_index', suffixes=('_old', '_new'))
                for k, token in zip(['s1', 's2'], tokens):
                    probe = joined.row_id_new.to_numpy()-ix[0]
                    np.testing.assert_array_equal(token[probe, :60], data.a[k][joined.row_id_old.to_numpy(), :60])
                    outputs[k][ix] = token
                future, known = future_labels(*[arrays[k] for k in ['quotes', 'priced', 'sequence_id', 'label_sequence_id', 'actions']],
                    ss[hi], np.full(len(ix), t))
                known &= np.logical_and.accumulate(valid_bars(future), axis=1)
                expanded = contextual_history(history, context[hi])
                close = history[:, -21:, 3].astype(float)*history[:, -21:, 6]
                for k, value in dict(mean=mean, scale=scale, last=history[:, -1, :6], future=future,
                        valid=known, indicators=expanded[:, :, :len(INDICATORS)], context=context[hi],
                        volatility=np.diff(np.log(close), axis=1).std(1)*100).items():
                    outputs[k][ix] = value
                if len(joined):
                    old = joined.row_id_old.to_numpy()
                    new = joined.row_id_new.to_numpy()
                    for k in ['mean', 'scale', 'future', 'valid', 'last']:
                        np.testing.assert_array_equal(outputs[k][new], data.a[k][old])
                    parity.extend(old.tolist())
        if number % 10 == 0 or number == len(needed)-1:
            progress = dict(stage='preparing', completed_dates=number+1, total_dates=len(needed), date=day, at=utc_now())
            write_json(root/'progress.json', progress)
            print(progress, flush=True)
    if len(parity) != len(source_ids['evaluation']) or len(set(parity)) != len(parity):
        raise ValueError('Incomplete historical input parity')
    for v in [*features.values(), *outputs.values()]:
        v.flush()
    center, scale = fit_normalizer(features['training'])
    np.savez(root/'normalizer.npz', center=center, scale=scale)
    pd.DataFrame(coverage).to_csv(root/'context-coverage.csv', index=False)
    splits = {}
    for part in ['training', 'selection', 'evaluation']:
        frame = eval_rows if part == 'evaluation' else sampled[part]
        splits[part] = dict(rows=len(frame), dates=int(frame.date.nunique()), first=frame.date.min(),
            last=frame.date.max(), label_end=frame.label_end.max())
    write_json(root/'splits.json', splits)
    write_json(root/'input-parity.json', dict(passed=True, rows=len(parity),
        exact_tokens_with_unknown_future=True, exact_normalization=True, exact_labels=True,
        inherited_input_eligibility_only=True))
    write_json(root/'sources.json', lineage)
    protected = [BASE/'configs/daily-token-v1.json', BASE/'artifacts/daily-token-live-v1/current.json',
                 BASE/'artifacts/daily-token-live-v1/latest.json']
    write_json(root/'live-state.json', {str(p): file_hash(p) for p in protected})
    shutil.copytree(BASE/'src', root/'code/src', dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
    (root/'code/scripts').mkdir(parents=True, exist_ok=True)
    for name in ['token_three_ideas.py', 'token_ranking_run.py', 'token_features_run.py',
                 'token_history_run.py', 'tokenizer_reconstruction_run.py', 'finalize_token_three_ideas.py']:
        shutil.copy2(BASE/'scripts'/name, root/'code/scripts'/name)
    seal(root, 'prepared.json', [p for p in root.rglob('*') if p.is_file() and p.name != 'progress.json'], passed=True)
    del tokenizer
    gc.collect()
    torch.mps.empty_cache()


class EvaluationDataset(Dataset):
    def __init__(self, root):
        self.root = root/'evaluation-inputs'
        self.rows = pd.read_parquet(self.root/'rows.parquet')
        self.stamps = np.load(DATA/'calendar-stamps.npy')
        self.a = {k: np.load(self.root/f'{k}.npy', mmap_mode='r') for k in
                  ['s1', 's2', 'valid', 'future', 'mean', 'scale', 'last']}
        self.indicators = np.load(self.root/'indicators.npy', mmap_mode='r')
        self.context = np.load(self.root/'context.npy', mmap_mode='r')

    def features(self, ix, names, device):
        if not names:
            return None
        values = np.full((len(ix), 60, len(FEATURES)), np.nan, np.float32)
        values[:, :, :len(INDICATORS)] = self.indicators[ix]
        values[:, -1, len(INDICATORS):] = self.context[ix]
        return auxiliary(values, np.arange(len(ix)), names, device, FEATURES)


def model_path(root, cfg, seed, arm):
    parent = root if arm == 'context_ranking' else BASE/cfg['prior']
    return parent/f'seed{seed}'/arm/'model.pt'


def forecast(root, cfg, data, seed, arm, decoder):
    dest = root/'forecasts'/f'seed{seed}'/arm
    dest.mkdir(parents=True, exist_ok=True)
    checkpoint = model_path(root, cfg, seed, arm)
    identity = dict(checkpoint_sha256=file_hash(checkpoint), prepared_sha256=file_hash(root/'prepared.json'),
        samples=cfg['samples'], forecast_batch=cfg['forecast_batch'], chunk_rows=cfg['chunk_rows'])
    if (dest/'completed.json').exists():
        if read(dest/'completed.json')['identity'] != identity:
            raise ValueError('Forecast identity changed')
        verify_manifest(dest, 'completed.json')
        return
    model, _ = restore_model(checkpoint, cfg['device'])
    n = len(data.rows)
    manifests = []
    for chunk_start in range(0, n, cfg['chunk_rows']):
        end = min(chunk_start+cfg['chunk_rows'], n)
        path = dest/f'{chunk_start:07d}.npy'
        marker = path.with_suffix('.json')
        if marker.exists():
            receipt = read(marker)
            if (receipt['identity'] != identity or receipt['start'] != chunk_start
                    or receipt['end'] != end or file_hash(path) != receipt['sha256']):
                raise ValueError('Changed completed forecast chunk')
        else:
            temporary = path.with_suffix('.partial.npy')
            paths = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32,
                shape=(end-chunk_start, cfg['samples'], 5, 6))
            for start in range(chunk_start, end, cfg['forecast_batch']):
                ix = np.arange(start, min(start+cfg['forecast_batch'], end))
                a, b, stamps, _ = data.tensors(ix, cfg['device'])
                aux = data.features(ix, model.config.auxiliary_features, cfg['device'])
                pairs = generate_cached(model, a[:, :60], b[:, :60], stamps[:, :60], stamps[:, 60:],
                    samples=cfg['samples'], seed=17+start, temperature=1., top_p=1., top_k=0, history_auxiliary=aux)
                paths[ix-chunk_start], _ = decode_cached(decoder, pairs, data.a['mean'][ix], data.a['scale'][ix], 5)
            paths.flush()
            del paths
            temporary.replace(path)
            write_json(marker, dict(identity=identity, start=chunk_start, end=end, sha256=file_hash(path), at=utc_now()))
        manifests.append(marker)
        progress = dict(stage='forecasting', seed=seed, arm=arm, rows=end, total=n, at=utc_now())
        write_json(root/'progress.json', progress)
        print(progress, flush=True)
    seal(dest, 'completed.json', manifests, identity=identity, inputs=n)
    del model
    gc.collect()
    torch.mps.empty_cache()


def run(cfg, prepare_only=False):
    root = BASE/cfg['output']
    prepare(root, cfg)
    if prepare_only:
        return
    for path, expected in read(root/'sources.json').items():
        if file_hash(Path(path)) != expected:
            raise ValueError(f'Changed frozen source {path}')
    torch.set_num_threads(4)
    decoder = load_decoder(Path(read(root/'profiles.json')['decoder']['path']), cfg['device'])
    decoder.requires_grad_(False).eval()
    original = Dataset()
    for seed in cfg['seeds']:
        train(root, cfg, original, seed, 'context_ranking', decoder)
    del original
    data = EvaluationDataset(root)
    # Complete context and matched ranking controls first, then broader controls.
    for arm in ['context_ranking', 'indicators_ranking', 'indicators', 'baseline']:
        for seed in cfg['seeds']:
            forecast(root, cfg, data, seed, arm, decoder)
    from finalize_token_three_ideas import finalize
    finalize(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/token-three-ideas-v1.json')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    with open(BASE/'artifacts/token-three-ideas.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.write(json.dumps(dict(pid=os.getpid(), started_at=utc_now())))
        lock.flush()
        torch.set_num_threads(4)
        run(read(BASE/args.config), args.prepare_only)
