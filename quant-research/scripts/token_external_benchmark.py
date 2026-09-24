"""Frozen external comparison on the inherited 76-date development cohort."""
import argparse
import gc
import importlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from compare_token_range import historical_paths
from token_history_run import DATA, Dataset
from tokenizer_reconstruction_run import finish, verify

from quant_research.forecast_audit import score_paths
from quant_research.storage import file_hash, utc_now, write_json
from quant_research.token_transformer import load_tokenizer, normalized_history, valid_bars

BASE = Path('/Users/guo/Documents/stocks/quant-research')
ART = BASE / 'artifacts'
ROOT = ART / 'token-external-benchmark-20260915-v1'
PRIOR = ART / 'token-profile-transfer-20260915-v1'
BUNDLE = ART / 'kronos-comparison-20260910-v1'
WEIGHTS = ART / 'kronos-base-weights-20260914-v1'
INPUT = ART / 'kronos-inputs-20260914-v3'
SEEDS = [17, 29, 43]
FIELDS = ['mae', 'crps', 'coverage80', 'width80', 'interval_score80']
SHA = 'abff193acab6db1a0368e9773e75799d11403b6d054ee6d5f0a11aeabc5f4b83'


def read(p):
    return json.loads(p.read_text())


def prepare():
    assert not ROOT.exists()
    sources = {}
    for p, h in read(PRIOR / 'final-verification.json')['files'].items():
        if Path(p).is_relative_to(ART):
            assert file_hash(Path(p)) == h, p
    for p in [PRIOR / 'final-verification.json', PRIOR / 'row-ids.npy', PRIOR / 'profiles.json', DATA / 'completed.json']:
        sources[str(p)] = file_hash(p)
    for name in ['rows.parquet', 'future.npy', 'valid.npy', 'last.npy', 'mean.npy', 'scale.npy', 's1.npy', 's2.npy', 'calendar-stamps.npy']:
        p = DATA / name
        assert file_hash(p) == read(DATA / 'completed.json')['files'][name]
        sources[str(p)] = file_hash(p)
    for p in [INPUT / 'values.npy', INPUT / 'manifest.json']:
        assert file_hash(p) == read(DATA / 'sources.json')[str(p)]
        sources[str(p)] = file_hash(p)
    assert file_hash(WEIGHTS / 'model.safetensors') == SHA
    for name in ['model.safetensors', 'config.json', 'README.md']:
        sources[str(WEIGHTS / name)] = file_hash(WEIGHTS / name)
    provenance = read(BUNDLE / 'pretrained-provenance.json')
    for name in ['upstream/model/__init__.py', 'upstream/model/kronos.py', 'upstream/model/module.py', 'upstream/LICENSE', 'tokenizer/config.json', 'tokenizer/model.safetensors']:
        p = BUNDLE / name
        assert file_hash(p) == provenance['files'][name]
        sources[str(p)] = file_hash(p)
    for seed in SEEDS:
        for part in ['forecasts', 'scores']:
            d = PRIOR / part / 'refreshed' / f'seed{seed}'
            verify(d)
            for name, h in read(d / 'completed.json')['files'].items():
                sources[str(d / name)] = h
    ROOT.mkdir()
    cfg = dict(protocol_id='token-external-benchmark-v1', created=utc_now(),
        evaluation=read(PRIOR / 'protocol.json')['evaluation'], horizons=[2, 5],
        samples=64, lookback=60, batch_rows=4, seed=17, temperature=1., top_p=1., top_k=0,
        minimum_legal_paths=16, model='NeoQuasar/Kronos-base', revision='2b554741eca47781b64468546e77fef3e85130e6',
        native_decoder=True, native_inference=True, sample_count=1,
        baseline='59 previous-close OHLC log-return vectors from 60 historical bars; remove mean close drift; sample 64 contiguous five-day blocks with replacement; hash seed 17 and stock/date identity',
        cohort='Intersection of all three self-built seeds, Kronos Base and historical volatility, separately at each horizon; known labels and at least 16 legal paths. Raw/calibrated self-built share this same cohort.',
        aggregation='Average stocks within date, then targets and training seeds, then dates; self-built seed mean averages scores, not paths; external forecast occurs once, not three independent replicas.',
        comparison='Raw MAE/CRPS/interval score against both external baselines; calibrated intervals secondary and explicitly unequal calibration treatment; no fit or retuning.',
        bootstrap=dict(replicates=2000, block_dates=10, seed=314159), sealed_holdout_start='2025-08-07',
        limitations='Previously examined development period. Pretrained predictor/tokenizer exact training-date coverage unresolved. Different full pipelines, not an isolated architecture effect. No profitability or production promotion claim.')
    write_json(ROOT / 'protocol.json', cfg)
    data = Dataset()
    ids = np.load(PRIOR / 'row-ids.npy')
    rows = data.rows.iloc[ids].reset_index(drop=True)
    assert len(rows) == 2432 and rows.date.nunique() == 76 and rows.label_end.max() < '2025-07-30'
    raw = np.load(INPUT / 'values.npy', mmap_mode='r')
    s, t = rows[['stock_index', 'date_index']].to_numpy(int).T
    history = raw[s[:, None], t[:, None] + np.arange(-59, 1)].copy()
    x, mean, scale = normalized_history(history)
    np.testing.assert_array_equal(mean, data.a['mean'][ids])
    np.testing.assert_array_equal(scale, data.a['scale'][ids])
    np.testing.assert_array_equal(history[:, -1, :6].astype(np.float32), data.a['last'][ids])
    stamps = data.stamps[t[:, None] + np.arange(-59, 6)].astype(np.float32)
    np.savez_compressed(ROOT / 'inputs.npz', history=history, x=x, mean=mean, scale=scale, past=stamps[:, :60], future=stamps[:, 60:])
    np.save(ROOT / 'row-ids.npy', ids)
    rows.to_parquet(ROOT / 'rows.parquet', index=False)
    shutil.copytree(PRIOR / 'code/src', ROOT / 'code/src', ignore=shutil.ignore_patterns('__pycache__'))
    (ROOT / 'code/scripts').mkdir()
    for name in ['token_external_benchmark.py', 'token_history_run.py', 'tokenizer_reconstruction_run.py', 'compare_token_range.py', 'compare_token_kronos_paths.py']:
        shutil.copy2(BASE / 'scripts' / name, ROOT / 'code/scripts' / name)
    write_json(ROOT / 'sources.json', sources)
    write_json(ROOT / 'prepared.json', dict(passed=True, at=utc_now(), files={str(p.relative_to(ROOT)): file_hash(p)
        for p in ROOT.rglob('*') if p.is_file()}))
    print('Prepared 2432 inputs across 76 dates; full history/statistics match', flush=True)


def checked():
    for name, h in read(ROOT / 'prepared.json')['files'].items():
        assert file_hash(ROOT / name) == h, name
    for name, h in read(ROOT / 'sources.json').items():
        assert file_hash(Path(name)) == h, name


def load():
    from safetensors.torch import load_model
    tokenizer = load_tokenizer(BUNDLE, 'mps')
    model = importlib.import_module('model').Kronos(**read(WEIGHTS / 'config.json'))
    load_model(model, str(WEIGHTS / 'model.safetensors'), strict=True)
    assert sum(p.numel() for p in model.parameters()) == 102310592
    return tokenizer, model.eval().requires_grad_(False).to('mps')


def generate(tokenizer, model, data, start, end):
    torch.manual_seed(17 + start)
    np.random.seed(17 + start)
    inference = importlib.import_module('model.kronos').auto_regressive_inference
    with torch.inference_mode():
        x, past, future = [torch.from_numpy(np.repeat(data[k][start:end], 64, axis=0)).to('mps') for k in ['x', 'past', 'future']]
        out = inference(tokenizer, model, x, past, future, max_context=512, pred_len=5,
            clip=5, T=1., top_k=0, top_p=1., sample_count=1, verbose=False)
    return out[:, -5:].reshape(end-start, 64, 5, 6) * data['scale'][start:end, None] + data['mean'][start:end, None]


def forecast():
    checked()
    torch.set_num_threads(4)
    dest = ROOT / 'kronos'
    if (dest / 'completed.json').exists():
        verify(dest)
        return
    dest.mkdir(exist_ok=True)
    chunks = dest / 'chunks'
    chunks.mkdir(exist_ok=True)
    with np.load(ROOT / 'inputs.npz') as z:
        inputs = {k: z[k] for k in ['x', 'mean', 'scale', 'past', 'future']}
    ids = np.load(ROOT / 'row-ids.npy')
    tokenizer, model = load()
    write_json(ROOT / 'runtime.json', dict(python=sys.version, torch=str(torch.__version__), numpy=np.__version__,
        device='mps', parameters=102310592, pid=os.getpid(), at=utc_now()))
    data = Dataset()
    # History-only encoding must agree with the frozen 65-bar training prefix.
    with torch.inference_mode():
        for start in range(0, len(ids), 128):
            pair = tokenizer.encode(torch.from_numpy(inputs['x'][start:start+128]).to('mps'), half=True)
            for k, value in zip(['s1', 's2'], pair):
                np.testing.assert_array_equal(value.cpu().numpy(), data.a[k][ids[start:start+128], :60])
    write_json(dest / 'input-parity.json', dict(passed=True, encoded_histories=len(ids), history_tokens=2*len(ids)*60))
    begun = time.monotonic()
    for start in range(0, len(ids), 4):
        p = chunks / f'{start:06d}.npz'
        if not p.exists():
            paths = generate(tokenizer, model, inputs, start, min(start+4, len(ids)))
            with p.with_suffix('.tmp').open('wb') as f:
                np.savez_compressed(f, paths=paths, row_ids=ids[start:start+4], valid=valid_bars(paths).all(-1))
            p.with_suffix('.tmp').replace(p)
        if start % 64 == 0:
            progress = dict(inputs=start+4, total=len(ids), elapsed_seconds=time.monotonic()-begun, at=utc_now())
            write_json(dest / 'progress.json', progress)
            print(progress, flush=True)
    output = np.lib.format.open_memmap(dest / 'paths.npy', mode='w+', dtype=np.float32, shape=(len(ids), 64, 5, 6))
    for start in range(0, len(ids), 4):
        with np.load(chunks / f'{start:06d}.npz') as z:
            np.testing.assert_array_equal(z['row_ids'], ids[start:start+4])
            output[start:start+4] = z['paths']
    output.flush()
    del model, tokenizer
    gc.collect()
    torch.mps.empty_cache()
    tokenizer, model = load()
    replay = generate(tokenizer, model, inputs, 0, 4)
    np.testing.assert_array_equal(output[:4], replay)
    write_json(dest / 'lineage.json', dict(passed=True, weight_sha256=file_hash(WEIGHTS / 'model.safetensors'),
        reloaded_exact_paths=256, native_decoder=True, elapsed_seconds=time.monotonic()-begun,
        chunks={p.name: file_hash(p) for p in chunks.iterdir()}))
    finish(dest)
    print('Kronos native forecast and exact reload passed', flush=True)


def baseline():
    dest = ROOT / 'historical'
    if (dest / 'completed.json').exists():
        verify(dest)
        return
    dest.mkdir()
    rows = pd.read_parquet(ROOT / 'rows.parquet')
    with np.load(ROOT / 'inputs.npz') as z:
        history = z['history']
    paths, draws = [], []
    for row, h in zip(rows.itertuples(), history):
        adjusted = h[:, :4].astype(float) * h[:, 6:7] / h[-1, 6]
        p, starts = historical_paths(adjusted, f'{row.instrument_id}:{row.date}', samples=64)
        paths.append(p)
        draws.append(starts)
    p = np.stack(paths)
    p = np.concatenate([p, np.zeros((*p.shape[:-1], 2))], -1)
    assert valid_bars(p).all()
    np.save(dest / 'paths.npy', p)
    np.save(dest / 'block-starts.npy', np.stack(draws))
    finish(dest)


def evaluate():
    checked()
    data = Dataset()
    ids = np.load(ROOT / 'row-ids.npy')
    records, coverage = [], []
    for name in ['kronos', 'historical']:
        verify(ROOT / name)
        paths = np.load(ROOT / name / 'paths.npy', mmap_mode='r')
        frame, cov = score_paths(paths, data.a['future'][ids], data.a['valid'][ids], data.a['last'][ids, 3], data.rows.iloc[ids].date.to_numpy())
        frame['row_id'] = ids[frame.local_row.to_numpy()]
        records.append(frame.assign(variant=name))
        coverage.append(cov.assign(variant=name))
    for seed in SEEDS:
        d = PRIOR / 'scores/refreshed' / f'seed{seed}'
        frame = pd.read_parquet(d / 'scores.parquet')
        for kind in ['raw', 'calibrated']:
            f = frame[frame.variant == f'refreshed_{kind}'].copy()
            f['variant'] = f'ours_{seed}_{kind}'
            records.append(f)
        coverage.append(pd.read_csv(d / 'coverage.csv').assign(variant=f'ours_{seed}'))
    fields = ['variant', 'local_row', 'row_id', 'date', 'horizon', 'target'] + FIELDS
    all_rows = pd.concat(records, ignore_index=True)[fields]
    common = all_rows[all_rows.groupby(['row_id', 'horizon', 'target']).variant.transform('nunique').eq(8)].copy()
    dest = ROOT / 'results'
    dest.mkdir(exist_ok=True)
    all_rows.to_parquet(dest / 'all-scores.parquet', index=False)
    common.to_parquet(dest / 'common.parquet', index=False)
    pd.concat(coverage, ignore_index=True).to_csv(dest / 'coverage.csv', index=False)
    daily = common.groupby(['variant', 'date', 'horizon', 'target'])[FIELDS].mean().reset_index()
    for kind in ['raw', 'calibrated']:
        mean = daily[daily.variant.isin([f'ours_{s}_{kind}' for s in SEEDS])].groupby(['date', 'horizon', 'target'])[FIELDS].mean().reset_index()
        daily = pd.concat([daily, mean.assign(variant=f'ours_mean_{kind}')], ignore_index=True)
    daily.to_csv(dest / 'daily.csv', index=False)
    daily.groupby(['variant', 'horizon', 'target'])[FIELDS].mean().reset_index().to_csv(dest / 'summary.csv', index=False)
    paired = []
    for against in ['kronos', 'historical']:
        for kind in ['raw', 'calibrated']:
            for seed in [*SEEDS, 'mean']:
                name = f'ours_{seed}_{kind}'
                for h in [2, 5]:
                    for target in ['maximum', 'minimum', 'range', 'high_low']:
                        sub = daily[(daily.horizon == h) & daily.variant.isin([against, name])]
                        sub = sub[sub.target.isin(['maximum', 'minimum'] if target == 'high_low' else [target])]
                        for metric in (FIELDS if kind == 'raw' else [f for f in FIELDS if f != 'crps']):
                            p = sub.groupby(['date', 'variant'])[metric].mean().unstack().sort_index()
                            b, x = p[against].to_numpy(), p[name].to_numpy()
                            rng = np.random.default_rng(314159)
                            idx = (rng.integers(len(p), size=(2000, int(np.ceil(len(p)/10)), 1)) + np.arange(10)) % len(p)
                            idx = idx.reshape(2000, -1)[:, :len(p)]
                            low, high = np.quantile((x-b)[idx].mean(1), [.025, .975])
                            paired.append(dict(against=against, kind=kind, seed=seed, horizon=h, target=target,
                                metric=metric, baseline=b.mean(), candidate=x.mean(), delta=(x-b).mean(),
                                relative_change=x.mean()/b.mean()-1, ci_low=low, ci_high=high, dates=len(p)))
    pd.DataFrame(paired).to_csv(dest / 'paired.csv', index=False)
    finish(dest)
    write_json(ROOT / 'run-completed.json', dict(passed=True, at=utc_now(), inputs=2432, signal_dates=76,
        new_kronos_paths=155648, new_historical_paths=155648, reused_self_built_seeds=3,
        training=False, calibration_refit=False, sealed_holdout_opened=False))
    print('External comparison scored', len(common), 'common records', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'forecast', 'baseline', 'evaluate'])
    globals()[parser.parse_args().stage]()
