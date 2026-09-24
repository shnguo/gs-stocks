"""Evaluate two frozen predictor/calibration vintages on a later development block."""
import gc
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tokenizer_reconstruction_run as core
import torch
from token_history_run import DATA, Dataset

from quant_research.forecast_audit import TARGETS, empirical_score, extrema
from quant_research.storage import file_hash, utc_now, write_json
from quant_research.token_intervals import apply_scale, interval_metrics
from quant_research.token_transformer import (
    decode_paths,
    generate_tokens,
    restore_model,
    valid_bars,
)

BASE = Path('/Users/guo/Documents/stocks/quant-research')
ART = BASE / 'artifacts'
ROOT = ART / 'token-profile-transfer-20260915-v1'
PRIOR = ART / 'token-refreshed-calibration-20260915-v1'
OLD = ART / 'token-interval-calibration-20260915-v1'
ORIGINAL = ART / 'tokenizer-reconstruction-20260915-v1'
SEEDS = [17, 29, 43]
SAMPLING = dict(temperature=1., top_p=1., top_k=0)
FIELDS = ['mae', 'coverage80', 'width80', 'interval_score80', 'lower_miss', 'upper_miss', 'crps']


def read(path):
    return json.loads(path.read_text())


def initialize():
    cfg = read(BASE / 'configs/token-profile-transfer-v1.json')
    if (ROOT / 'protocol.json').exists():
        assert read(ROOT / 'protocol.json') == cfg
        return
    sources = {}
    for prior in [PRIOR, OLD]:
        meta = read(prior / 'final-verification.json')
        assert meta['passed']
        for name, digest in meta['files'].items():
            if Path(name).is_relative_to(ART):
                assert file_hash(Path(name)) == digest, name
        sources[str(prior / 'final-verification.json')] = file_hash(prior / 'final-verification.json')
    profile = read(PRIOR / 'research-profile.json')
    sources[str(PRIOR / 'research-profile.json')] = file_hash(PRIOR / 'research-profile.json')
    chosen = profile['checkpoints'][cfg['selected_vintage']]
    refreshed = {str(p['seed']): dict(path=p['path'], sha256=p['sha256']) for p in chosen['predictors']}
    old = {}
    for seed in SEEDS:
        checkpoint = ORIGINAL / f'predictors/seed{seed}/dense_3720k_equal/training/best.pt'
        proof = read(OLD / f'calibration2023h1/predictors/seed{seed}/dense_3720k_equal/decoder-forecast/verification.json')
        assert proof['predictor_sha256'] == file_hash(checkpoint)
        old[str(seed)] = dict(path=str(checkpoint), sha256=file_hash(checkpoint))
    profiles = dict(decoder=profile['decoder'], sampling=cfg['sampling'], fit_settings=profile['calibration'],
        old=dict(checkpoints=old, fit_path=str(OLD / 'fit.json'), fit_sha256=file_hash(OLD / 'fit.json'),
            training_end='2022-01-01', calibration_dates=['2023-01-01', '2023-07-01']),
        refreshed=dict(checkpoints=refreshed, fit_path=chosen['fit_path'], fit_sha256=chosen['fit_sha256'],
            training_end='2024-01-01', calibration_dates=chosen['calibration_period']))
    for owner in ['old', 'refreshed']:
        p = profiles[owner]
        assert file_hash(Path(p['fit_path'])) == p['fit_sha256']
        sources[p['fit_path']] = p['fit_sha256']
        assert p['training_end'] < p['calibration_dates'][1] <= cfg['evaluation']['dates'][0]
        for item in p['checkpoints'].values():
            assert file_hash(Path(item['path'])) == item['sha256']
            sources[item['path']] = item['sha256']
    sources[profiles['decoder']['path']] = profiles['decoder']['sha256']
    assert file_hash(Path(profiles['decoder']['path'])) == profiles['decoder']['sha256']
    sources[str(DATA / 'completed.json')] = file_hash(DATA / 'completed.json')
    sources[str(DATA / cfg['evaluation']['split'])] = file_hash(DATA / cfg['evaluation']['split'])
    ROOT.mkdir()
    write_json(ROOT / 'protocol.json', cfg)
    write_json(ROOT / 'profiles.json', profiles)
    shutil.copytree(PRIOR / 'code/src', ROOT / 'code/src', ignore=shutil.ignore_patterns('__pycache__'))
    (ROOT / 'code/scripts').mkdir()
    for name in ['token_profile_transfer_run.py', 'tokenizer_reconstruction_run.py', 'token_history_run.py']:
        shutil.copy2(BASE / 'scripts' / name, ROOT / 'code/scripts' / name)
    write_json(ROOT / 'source-manifest.json', dict(at=utc_now(), sources=sources,
        protocol_sha256=file_hash(ROOT / 'protocol.json'), profiles_sha256=file_hash(ROOT / 'profiles.json'),
        code={str(p.relative_to(ROOT)): file_hash(p) for p in (ROOT / 'code').rglob('*.py')}))


def preflight(data, cfg):
    ids = np.load(DATA / cfg['evaluation']['split'])
    ids = data.rows.iloc[ids].groupby('date').head(cfg['evaluation']['rows_per_date']).row_id.to_numpy(int)
    r = data.rows.iloc[ids]
    assert len(ids) == cfg['evaluation']['inputs_per_seed']
    assert r.date.nunique() == cfg['evaluation']['signal_dates']
    assert r.groupby('date').size().eq(cfg['evaluation']['rows_per_date']).all()
    assert r.date.min() >= cfg['evaluation']['dates'][0]
    assert r.label_end.max() < cfg['evaluation']['dates'][1] < cfg['sealed_holdout_start']
    np.save(ROOT / 'row-ids.npy', ids)
    write_json(ROOT / 'preflight.json', dict(passed=True, at=utc_now(), inputs=len(ids), dates=int(r.date.nunique()),
        first_signal=str(r.date.min()), last_signal=str(r.date.max()), last_label=str(r.label_end.max()),
        refitted=False, recalibrated=False, sealed_holdout_opened=False))
    return ids


def forecast(data, ids, owner, seed, profiles):
    dest = ROOT / 'forecasts' / owner / f'seed{seed}'
    if (dest / 'completed.json').exists():
        core.verify(dest)
        return
    dest.mkdir(parents=True, exist_ok=True)
    chunks = dest / 'chunks'
    chunks.mkdir(exist_ok=True)
    checkpoint = Path(profiles[owner]['checkpoints'][str(seed)]['path'])
    model, saved = restore_model(checkpoint, 'mps')
    assert saved['dataset_manifest_sha256'] == file_hash(DATA / 'completed.json')
    model.eval()
    decoder = core.load_decoder(Path(profiles['decoder']['path']))
    for start in range(0, len(ids), 4):
        batch = ids[start:start + 4]
        path = chunks / f'{start:06d}.npz'
        if path.exists():
            continue
        a, b, stamps, _ = data.tensors(batch, 'mps')
        pairs = generate_tokens(model, a[:, :60], b[:, :60], stamps[:, :60], stamps[:, 60:],
            samples=64, seed=17 + start, **SAMPLING)
        paths, valid = decode_paths(decoder, pairs, data.a['mean'][batch], data.a['scale'][batch], 5)
        with path.with_suffix('.tmp').open('wb') as f:
            np.savez_compressed(f, row_ids=batch, paths=paths, valid=valid,
                s1=pairs[0][:, :, -5:].cpu().numpy(), s2=pairs[1][:, :, -5:].cpu().numpy())
        path.with_suffix('.tmp').replace(path)
        if start % 512 == 0:
            progress = dict(owner=owner, seed=seed, inputs=start + len(batch), total=len(ids), at=utc_now())
            write_json(dest / 'progress.json', progress)
            print(progress, flush=True)
    output = np.lib.format.open_memmap(dest / 'paths.npy', mode='w+', dtype=np.float32, shape=(len(ids), 64, 5, 6))
    for start in range(0, len(ids), 4):
        with np.load(chunks / f'{start:06d}.npz') as z:
            np.testing.assert_array_equal(z['row_ids'], ids[start:start + 4])
            output[start:start + len(z['row_ids'])] = z['paths']
    output.flush()
    a, b, stamps, _ = data.tensors(ids[:4], 'mps')
    pairs = generate_tokens(model, a[:, :60], b[:, :60], stamps[:, :60], stamps[:, 60:], samples=64, seed=17, **SAMPLING)
    paths, valid = decode_paths(decoder, pairs, data.a['mean'][ids[:4]], data.a['scale'][ids[:4]], 5)
    with np.load(chunks / '000000.npz') as z:
        np.testing.assert_array_equal(z['paths'], paths)
        np.testing.assert_array_equal(z['valid'], valid)
        for key, value in zip(['s1', 's2'], pairs):
            np.testing.assert_array_equal(z[key], value[:, :, -5:].cpu().numpy())
    write_json(dest / 'lineage.json', dict(checkpoint_sha256=file_hash(checkpoint), decoder_sha256=profiles['decoder']['sha256'],
        sampling=SAMPLING, samples=64, generation_seed=17, first_batch_exact_replay=True,
        chunks={p.name: file_hash(p) for p in chunks.iterdir()}))
    core.finish(dest)
    del model, decoder, output
    gc.collect()
    torch.mps.empty_cache()


def score(data, ids, owner, seed, profiles):
    dest = ROOT / 'scores' / owner / f'seed{seed}'
    if (dest / 'completed.json').exists():
        core.verify(dest)
        return pd.read_parquet(dest / 'scores.parquet')
    dest.mkdir(parents=True, exist_ok=True)
    paths = np.load(ROOT / 'forecasts' / owner / f'seed{seed}/paths.npy', mmap_mode='r')
    ref = data.a['last'][ids, 3].astype(float)
    fitted = read(Path(profiles[owner]['fit_path']))
    assert file_hash(Path(profiles[owner]['fit_path'])) == profiles[owner]['fit_sha256']
    params = {(p['seed'], p['horizon'], p['target']): p for p in fitted['parameters']}
    quantiles, raw_metrics, coverage = [], [], []
    for h in [2, 5]:
        legal = valid_bars(paths[:, :, :h]).all(-1)
        known = data.a['valid'][ids, :h].all(1)
        usable = known & (legal.sum(1) >= 16)
        truth = extrema(data.a['future'][ids, :h].astype(float), ref)
        coverage.append(dict(horizon=h, inputs=len(ids), known=int(known.sum()), usable=int(usable.sum()),
            valid_paths=int(legal.sum()), total_paths=int(legal.size)))
        for i in np.flatnonzero(usable):
            values = extrema(paths[i, legal[i], :h].astype(float), ref[i])
            q = np.quantile(values, [.1, .5, .9], axis=0)
            metrics = empirical_score(values, truth[i])
            for j, target in enumerate(TARGETS):
                quantiles.append(dict(owner=owner, seed=seed, local_row=int(i), row_id=int(ids[i]),
                    date=data.rows.iloc[ids[i]].date, horizon=h, target=target, actual=truth[i, j],
                    lower=q[0, j], median=q[1, j], upper=q[2, j], draws=int(legal[i].sum())))
                raw_metrics.append(dict(crps=metrics['crps'][j]))
    frame = pd.DataFrame(quantiles)
    frame['crps'] = pd.DataFrame(raw_metrics).crps
    frame.to_parquet(dest / 'quantiles.parquet', index=False)
    outputs = []
    for (h, target), group in frame.groupby(['horizon', 'target']):
        q = group[['lower', 'median', 'upper']].to_numpy()
        p = params[(seed, h, target)]
        adjusted = apply_scale(q, p['scale'], profiles['fit_settings']['lower_support'][target], p['epsilon'])
        np.testing.assert_array_equal(adjusted[:, 1], q[:, 1])
        for variant, values in [('raw', q), ('calibrated', adjusted)]:
            out = group.drop(columns=['lower', 'median', 'upper', 'crps']).copy()
            out['variant'] = f'{owner}_{variant}'
            out[['lower', 'median', 'upper']] = values
            for field, value in interval_metrics(values, group.actual.to_numpy()).items():
                out[field] = value
            out['crps'] = group.crps.to_numpy() if variant == 'raw' else np.nan
            outputs.append(out)
    output = pd.concat(outputs, ignore_index=True)
    output.to_parquet(dest / 'scores.parquet', index=False)
    pd.DataFrame(coverage).to_csv(dest / 'coverage.csv', index=False)
    write_json(dest / 'frozen-fit.json', dict(fit_sha256=profiles[owner]['fit_sha256'],
        checkpoint_sha256=profiles[owner]['checkpoints'][str(seed)]['sha256'], median_exactly_unchanged=True))
    core.finish(dest)
    return output


def aggregate(frames, cfg):
    all_rows = pd.concat(frames, ignore_index=True)
    keys = ['seed', 'row_id', 'horizon', 'target']
    n = all_rows.groupby(keys).variant.transform('nunique')
    common = all_rows[n == 4].copy()
    dest = ROOT / 'results'
    dest.mkdir(exist_ok=True)
    common.to_parquet(dest / 'common.parquet', index=False)
    daily = common.groupby(['seed', 'variant', 'date', 'horizon', 'target'])[FIELDS].mean().reset_index()
    daily.to_csv(dest / 'daily.csv', index=False)
    records = []
    for comparison, (baseline, candidate) in cfg['comparisons'].items():
        metrics = ['crps'] if comparison == 'raw_distribution' else ['mae', 'coverage80', 'width80', 'interval_score80']
        for h in [2, 5]:
            for target in ['maximum', 'minimum', 'range', 'high_low', 'mean']:
                sub = daily[(daily.horizon == h) & daily.variant.isin([baseline, candidate])]
                if target == 'high_low':
                    sub = sub[sub.target.isin(['maximum', 'minimum'])]
                elif target != 'mean':
                    sub = sub[sub.target == target]
                for metric in metrics:
                    paired = sub.groupby(['seed', 'date', 'variant'])[metric].mean().unstack('variant')
                    for seed in SEEDS + ['mean']:
                        p = (paired if seed == 'mean' else paired.loc[[seed]]).groupby('date').mean()
                        b, x = p[baseline].to_numpy(), p[candidate].to_numpy()
                        rng = np.random.default_rng(314159)
                        idx = (rng.integers(len(p), size=(2000, int(np.ceil(len(p) / 10)), 1)) + np.arange(10)) % len(p)
                        idx = idx.reshape(2000, -1)[:, :len(p)]
                        low, high = np.quantile((x - b)[idx].mean(1), [.025, .975])
                        records.append(dict(comparison=comparison, horizon=h, target=target, metric=metric,
                            seed=seed, baseline=b.mean(), candidate=x.mean(), delta=(x - b).mean(),
                            relative_change=x.mean() / b.mean() - 1, ci_low=low, ci_high=high, dates=len(p)))
    pd.DataFrame(records).to_csv(dest / 'paired.csv', index=False)
    core.finish(dest)


def main():
    initialize()
    cfg = read(ROOT / 'protocol.json')
    profiles = read(ROOT / 'profiles.json')
    manifest = read(ROOT / 'source-manifest.json')
    assert file_hash(ROOT / 'protocol.json') == manifest['protocol_sha256']
    assert file_hash(ROOT / 'profiles.json') == manifest['profiles_sha256']
    for name, digest in manifest['sources'].items():
        assert file_hash(Path(name)) == digest, name
    for name, digest in manifest['code'].items():
        assert file_hash(ROOT / name) == digest, name
    if (ROOT / 'run-completed.json').exists():
        for done in ROOT.rglob('completed.json'):
            core.verify(done.parent)
        return
    torch.set_num_threads(4)
    data = Dataset()
    ids = preflight(data, cfg)
    frames = []
    for seed in SEEDS:
        for owner in ['refreshed', 'old']:
            forecast(data, ids, owner, seed, profiles)
            frames.append(score(data, ids, owner, seed, profiles))
    aggregate(frames, cfg)
    write_json(ROOT / 'run-completed.json', dict(passed=True, at=utc_now(), generation_runs=6,
        inputs_per_run=len(ids), sampled_paths=len(ids) * 64 * 6, trained=False, recalibrated=False,
        sealed_holdout_opened=False))


if __name__ == '__main__':
    main()
