"""Repeat frozen checkpoints with two additional forecast RNG seeds."""
import argparse
import gc
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from token_history_run import DATA, Dataset
from tokenizer_reconstruction_run import finish, load_decoder, verify

from quant_research.forecast_audit import empirical_score
from quant_research.storage import file_hash, utc_now, write_json
from quant_research.token_transformer import (
    decode_paths,
    generate_tokens,
    restore_model,
    valid_bars,
)

BASE = Path('/Users/guo/Documents/stocks/quant-research')
ROOT = BASE / 'artifacts/token-sampling-replication-20260915-v1'
PRIOR = BASE / 'artifacts/token-midpoint-transfer-20260915-v1'
DIAG = BASE / 'artifacts/token-seed43-diagnostic-20260915-v1'
SEEDS = [17, 29, 43]
DRAWS = [17, 100017, 200017]
OWNERS = ['current', 'ce', 'midpoint']
SAMPLING = dict(temperature=1., top_p=1., top_k=0)
METRICS = ['endpoint_mae', 'high_mae', 'low_mae', 'center_bias', 'half_width_bias',
           'midpoint_mae', 'midpoint_crps', 'range_mae', 'legal_fraction']


def read(path):
    return json.loads(path.read_text())


def folder(draw, seed, owner):
    return (PRIOR if draw == 17 else ROOT / f'draw{draw}') / 'forecasts' / owner / f'seed{seed}'


def prepare():
    assert not ROOT.exists()
    assert read(DIAG / 'final-verification.json')['passed']
    sources = {}
    def bind(path, digest=None):
        path = Path(path)
        h = file_hash(path)
        if digest is not None:
            assert h == digest, path
        sources[str(path)] = h
    profiles = read(PRIOR / 'profiles.json')
    for p in [PRIOR / 'profiles.json', PRIOR / 'final-verification.json',
              PRIOR / 'evaluation-ids.npy', DIAG / 'final-verification.json', DIAG / 'rows.parquet']:
        bind(p)
    for owner in OWNERS:
        for seed in SEEDS:
            item = profiles[owner]['checkpoints'][str(seed)]
            bind(item['path'], item['sha256'])
            f = folder(17, seed, owner)
            verify(f)
            for name in ['paths.npy', 'lineage.json', 'completed.json']:
                bind(f / name)
    bind(profiles['decoder']['path'], profiles['decoder']['sha256'])
    manifest = read(DATA / 'completed.json')
    bind(DATA / 'completed.json')
    for name in ['rows.parquet', 's1.npy', 's2.npy', 'valid.npy', 'future.npy', 'mean.npy',
                 'scale.npy', 'last.npy', 'calendar-stamps.npy', 'calendar.json']:
        bind(DATA / name, manifest['files'][name])
    data = Dataset()
    ids = np.load(PRIOR / 'evaluation-ids.npy')
    r = data.rows.iloc[ids]
    assert len(ids) == 2432 and r.date.nunique() == 76 and r.groupby('date').size().eq(32).all()
    assert r.date.min() == '2025-04-01' and r.date.max() == '2025-07-22' and r.label_end.max() == '2025-07-29'
    ROOT.mkdir()
    np.save(ROOT / 'row-ids.npy', ids)
    write_json(ROOT / 'profiles.json', profiles)
    write_json(ROOT / 'sources.json', sources)
    write_json(ROOT / 'protocol.json', dict(at=utc_now(), predictors=SEEDS, generation_seeds=DRAWS,
        new_generation_seeds=DRAWS[1:], owners=OWNERS, inputs=2432, dates=76, rows_per_date=32,
        horizons=[2, 5], lookback=60, samples=64, batch_rows=4, sampling=SAMPLING,
        rng='CPU generator seed = generation seed + four-row batch offset; same schedule for all checkpoints; disjoint seed ranges across replicates.',
        checkpoint_policy='Exact frozen originals and two continuation arms; no training, model selection or calibration fitting.',
        primary_cohort='Known labels and >=16 legal paths in all 27 owner/training-seed/draw combinations, same cohort for every comparison.',
        sensitivity='Also report three-owner common cohorts within each training seed and draw; disclose exclusions relative to original all-nine cohort.',
        primary='Reproducibility of signed center movement and high/low MAE, especially seed 43 midpoint vs CE and vs current; retain all seeds.',
        metrics=METRICS, aggregation='Equal stocks within date, equal dates, equal draw replicates; seed means average scores, never prices or sampled paths.',
        uncertainty='2000 circular ten-date bootstrap replicates, RNG314159; conditional on fixed weights and three draw sets. Pointwise intervals; draw min/max are descriptive, not a confidence interval.',
        no_adaptive_stopping=True, sealed_holdout_start='2025-08-07',
        limitations='Already examined development period, three draw replicates, shared calendar and weights, unknown tokenizer pretraining dates; no trading interpretation.'))
    shutil.copytree(BASE / 'src', ROOT / 'code/src', ignore=shutil.ignore_patterns('__pycache__'))
    (ROOT / 'code/scripts').mkdir()
    for name in ['token_sampling_replication.py', 'token_history_run.py', 'tokenizer_reconstruction_run.py']:
        shutil.copy2(BASE / 'scripts' / name, ROOT / 'code/scripts' / name)
    write_json(ROOT / 'prepared.json', dict(passed=True, at=utc_now(), files={
        str(p.relative_to(ROOT)): file_hash(p) for p in ROOT.rglob('*') if p.is_file()}))
    print('Prepared 18 new forecast sets, 9 reused; 2,801,664 new five-day paths.', flush=True)


def checked():
    for path, sha in read(ROOT / 'sources.json').items():
        assert file_hash(Path(path)) == sha, path
    for path, sha in read(ROOT / 'prepared.json')['files'].items():
        assert file_hash(ROOT / path) == sha, path


def forecast(data, ids, profiles, draw, seed, owner):
    dest = folder(draw, seed, owner)
    if (dest / 'completed.json').exists():
        verify(dest)
        return
    dest.mkdir(parents=True, exist_ok=True)
    chunks = dest / 'chunks'
    chunks.mkdir(exist_ok=True)
    model, saved = restore_model(Path(profiles[owner]['checkpoints'][str(seed)]['path']), 'mps')
    assert saved['dataset_manifest_sha256'] == file_hash(DATA / 'completed.json')
    model.eval()
    decoder = load_decoder(Path(profiles['decoder']['path']))
    started = time.monotonic()
    for start in range(0, len(ids), 4):
        path = chunks / f'{start:06d}.npz'
        if path.exists():
            continue
        batch = ids[start:start+4]
        a, b, stamps, _ = data.tensors(batch, 'mps')
        pairs = generate_tokens(model, a[:, :60], b[:, :60], stamps[:, :60], stamps[:, 60:],
                                samples=64, seed=draw+start, **SAMPLING)
        paths, valid = decode_paths(decoder, pairs, data.a['mean'][batch], data.a['scale'][batch], 5)
        with path.with_suffix('.tmp').open('wb') as out:
            np.savez_compressed(out, row_ids=batch, paths=paths, valid=valid,
                s1=pairs[0][:, :, -5:].cpu().numpy(), s2=pairs[1][:, :, -5:].cpu().numpy())
        path.with_suffix('.tmp').replace(path)
        if start % 256 == 0:
            progress = dict(at=utc_now(), draw=draw, seed=seed, owner=owner, inputs=start+len(batch), total=len(ids),
                            elapsed_seconds=time.monotonic()-started)
            write_json(ROOT / 'progress.json', progress)
            print(progress, flush=True)
    output = np.lib.format.open_memmap(dest / 'paths.npy', mode='w+', dtype=np.float32, shape=(len(ids), 64, 5, 6))
    for start in range(0, len(ids), 4):
        with np.load(chunks / f'{start:06d}.npz') as z:
            np.testing.assert_array_equal(z['row_ids'], ids[start:start+4])
            output[start:start+4] = z['paths']
    output.flush()
    write_json(dest / 'lineage.json', dict(at=utc_now(), checkpoint_sha256=profiles[owner]['checkpoints'][str(seed)]['sha256'],
        decoder_sha256=profiles['decoder']['sha256'], row_ids_sha256=file_hash(ROOT / 'row-ids.npy'),
        sampling=SAMPLING, samples=64, generation_seed=draw, batch_rows=4, elapsed_seconds=time.monotonic()-started,
        chunks={p.name: file_hash(p) for p in chunks.iterdir()}))
    finish(dest)
    del model, decoder, output
    gc.collect()
    torch.mps.empty_cache()
    print('Completed forecast', draw, seed, owner, flush=True)


def run():
    checked()
    torch.set_num_threads(4)
    write_json(ROOT / 'runtime.json', dict(at=utc_now(), pid=os.getpid(), torch=str(torch.__version__), device='mps'))
    data = Dataset()
    ids = np.load(ROOT / 'row-ids.npy')
    profiles = read(ROOT / 'profiles.json')
    # All seeds and arms finish under the fixed protocol regardless of intermediate outcomes.
    for draw in DRAWS[1:]:
        for seed in SEEDS:
            for owner in OWNERS:
                forecast(data, ids, profiles, draw, seed, owner)
    checked()
    write_json(ROOT / 'generation-completed.json', dict(passed=True, at=utc_now(), new_forecast_sets=18,
        reused_forecast_sets=9, new_paths=18*len(ids)*64, training_runs=0, calibration_fits=0))


def interval(x):
    x = np.asarray(x)
    rng = np.random.default_rng(314159)
    ix = (rng.integers(len(x), size=(2000, (len(x)+9)//10, 1))+np.arange(10)) % len(x)
    return np.quantile(x[ix.reshape(2000, -1)[:, :len(x)]].mean(1), [.025, .975])


def paired(daily):
    result = []
    dims = ['cohort', 'draw', 'seed', 'horizon']
    for key, g in daily.groupby(dims):
        for base, new in [('current', 'ce'), ('ce', 'midpoint'), ('current', 'midpoint')]:
            for metric in METRICS:
                p = g.pivot(index='date', columns='owner', values=metric).sort_index()
                delta = p[new]-p[base]
                lo, hi = interval(delta)
                result.append(dict(zip(dims, key), comparison=f'{new}_vs_{base}', metric=metric,
                    baseline=p[base].mean(), candidate=p[new].mean(), delta=delta.mean(), ci_low=lo, ci_high=hi,
                    relative_change_pct=(p[new].mean()/p[base].mean()-1)*100 if metric not in ['center_bias', 'half_width_bias'] else np.nan,
                    dates=len(p)))
    return pd.DataFrame(result)


def score():
    checked()
    assert read(ROOT / 'generation-completed.json')['passed']
    data = Dataset()
    ids = np.load(ROOT / 'row-ids.npy')
    records, coverage = [], []
    for draw in DRAWS:
        for seed in SEEDS:
            for owner in OWNERS:
                dest = folder(draw, seed, owner)
                verify(dest)
                p = np.load(dest / 'paths.npy', mmap_mode='r')
                for h in [2, 5]:
                    mask = valid_bars(p[:, :, :h]).all(-1)
                    known = data.a['valid'][ids, :h].all(1)
                    use = known & (mask.sum(1) >= 16)
                    coverage.append(dict(draw=draw, seed=seed, owner=owner, horizon=h, inputs=len(ids), known=int(known.sum()),
                        usable=int(use.sum()), legal_paths=int(mask.sum()), total_paths=int(mask.size)))
                    for i in np.flatnonzero(use):
                        rid = ids[i]
                        ref = float(data.a['last'][rid, 3])
                        x = p[i, mask[i], :h].astype(float)
                        high, low = (x[..., 1].max(-1)/ref-1)*100, (x[..., 2].min(-1)/ref-1)*100
                        y = data.a['future'][rid, :h].astype(float)
                        yh, yl = (y[:, 1].max()/ref-1)*100, (y[:, 2].min()/ref-1)*100
                        mh, ml = np.median(high), np.median(low)
                        mid = (high+low)/2
                        mid_score = empirical_score(mid[:, None], np.array([(yh+yl)/2]))
                        records.append(dict(draw=draw, seed=seed, owner=owner, horizon=h, row_id=int(rid), local_row=int(i),
                            date=data.rows.iloc[rid].date, high_median=mh, low_median=ml, actual_high=yh, actual_low=yl,
                            endpoint_mae=(abs(mh-yh)+abs(ml-yl))/2, high_mae=abs(mh-yh), low_mae=abs(ml-yl),
                            center_bias=(mh+ml-yh-yl)/2, half_width_bias=(mh-ml-yh+yl)/2,
                            midpoint_mae=mid_score['mae'][0], midpoint_crps=mid_score['crps'][0],
                            range_mae=abs(np.median(high-low)-yh+yl), legal_fraction=mask[i].mean()))
                print('Scored', draw, seed, owner, flush=True)
    frame = pd.DataFrame(records)
    frame.to_parquet(ROOT / 'all-scores.parquet', index=False)
    pd.DataFrame(coverage).to_csv(ROOT / 'coverage.csv', index=False)
    frame['models'] = frame.groupby(['row_id', 'horizon']).owner.transform('size')
    primary = frame[frame.models == 27].drop(columns='models').assign(cohort='all27')
    n = frame.groupby(['draw', 'seed', 'horizon', 'row_id']).owner.transform('size')
    local = frame[n == 3].drop(columns='models').assign(cohort='within_draw_seed')
    common = pd.concat([primary, local], ignore_index=True)
    common.to_parquet(ROOT / 'common.parquet', index=False)
    daily = common.groupby(['cohort', 'draw', 'seed', 'owner', 'horizon', 'date'])[METRICS].mean().reset_index()
    mean_draw = daily.groupby(['cohort', 'seed', 'owner', 'horizon', 'date'])[METRICS].mean().reset_index().assign(draw=0)
    daily = pd.concat([daily, mean_draw], ignore_index=True)
    mean_seed = daily.groupby(['cohort', 'draw', 'owner', 'horizon', 'date'])[METRICS].mean().reset_index().assign(seed=0)
    daily = pd.concat([daily, mean_seed], ignore_index=True)
    assert (daily.groupby(['cohort', 'draw', 'seed', 'owner', 'horizon']).size() == 76).all()
    daily.to_csv(ROOT / 'daily.csv', index=False)
    paired(daily).to_csv(ROOT / 'paired.csv', index=False)
    daily.groupby(['cohort', 'draw', 'seed', 'owner', 'horizon'])[METRICS].mean().reset_index().to_csv(ROOT / 'summary.csv', index=False)
    common.groupby(['cohort', 'draw', 'seed', 'horizon']).row_id.nunique().rename('rows').reset_index().to_csv(ROOT / 'support.csv', index=False)
    checked()
    write_json(ROOT / 'scoring-completed.json', dict(passed=True, at=utc_now(), raw_score_rows=len(frame),
        common_rows=len(common), files={p.name: file_hash(p) for p in ROOT.iterdir() if p.suffix in ['.parquet', '.csv']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run', 'score'])
    globals()[parser.parse_args().action]()
