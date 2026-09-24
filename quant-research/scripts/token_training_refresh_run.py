"""Bounded expanding-history refresh comparison with immutable source reuse."""
import gc
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tokenizer_reconstruction_run as core
import torch

from quant_research.forecast_audit import common_scores, score_paths
from quant_research.storage import file_hash, utc_now, write_json
from quant_research.token_transformer import decode_paths, generate_tokens, restore_model

BASE = Path('/Users/guo/Documents/stocks/quant-research')
ROOT = BASE / 'artifacts/token-training-refresh-20260915-v1'
ART = BASE / 'artifacts'
ORIGINAL = ART / 'tokenizer-reconstruction-20260915-v1'
CONFIRM = ART / 'token-pipeline-confirmation-20260915-v1'
CALIBRATION = ART / 'token-interval-calibration-20260915-v1'
DECODER = ORIGINAL / 'decoder-training/price_consistency/best-qualified.pt'
SEEDS = [17, 29, 43]
SAMPLING = dict(temperature=1., top_p=1., top_k=0)


def read(path):
    return json.loads(path.read_text())


def old_forecast(window, seed):
    parent = (ART / 'tokenizer-temporal-20260915-v1' if window == '2023h2'
              else CALIBRATION / 'evaluation2024h2')
    return parent / f'predictors/seed{seed}/dense_3720k_equal/decoder-forecast'


def reused_training(window, seed):
    if window == '2023h2':
        return CONFIRM / f'seed{seed}/dense_3720k_equal/training'
    if seed == 17:
        return ART / 'token-history-experiment-20260915-v1/2024h2/dense_3720k_equal/training'
    return None


def initialize():
    cfg = read(BASE / 'configs/token-training-refresh-v1.json')
    if (ROOT / 'protocol.json').exists():
        assert read(ROOT / 'protocol.json') == cfg
        return
    ROOT.mkdir()
    write_json(ROOT / 'protocol.json', cfg)
    shutil.copytree(ORIGINAL / 'code', ROOT / 'code')
    shutil.copy2(BASE / 'scripts/token_training_refresh_run.py', ROOT / 'code/scripts/token_training_refresh_run.py')
    sources = {str(DECODER): file_hash(DECODER), str(core.DATA / 'completed.json'): file_hash(core.DATA / 'completed.json')}
    for window in cfg['windows']:
        for seed in SEEDS:
            old = old_forecast(window, seed)
            core.verify(old)
            for name in ['completed.json', 'row-ids.npy', 'adapted-paths.npy']:
                sources[str(old / name)] = file_hash(old / name)
            source = reused_training(window, seed)
            if source:
                core.verify(source)
                dest = ROOT / window / f'predictors/seed{seed}/dense_3720k_equal/training'
                dest.mkdir(parents=True)
                for name in ['best.pt', 'summary.json', 'history.json', 'reload-reference.npz', 'visited-row-ids.npy']:
                    shutil.copy2(source / name, dest / name)
                    sources[str(source / name)] = file_hash(source / name)
                write_json(dest / 'reuse.json', dict(source=str(source), refitted=False))
                core.finish(dest)
            if window == '2023h2':
                cached = CONFIRM / f'seed{seed}/dense_3720k_equal/evaluation/sampling'
                core.verify(cached)
                settings = read(cached / 'settings.json')
                assert settings['sampling'] == SAMPLING and settings['samples'] == 64 and settings['seed'] == 17
                assert file_hash(source / 'best.pt') == settings['checkpoint_sha256']
                for name, digest in settings['chunks'].items():
                    path = cached / 'chunks' / name
                    assert file_hash(path) == digest
                for name in ['settings.json', 'completed.json', 'row-ids.npy']:
                    sources[str(cached / name)] = file_hash(cached / name)
    write_json(ROOT / 'source-manifest.json', dict(at=utc_now(), sources=sources,
        protocol=file_hash(ROOT / 'protocol.json'),
        code={str(p.relative_to(ROOT)): file_hash(p) for p in (ROOT / 'code').rglob('*.py')}))


def dataset(cfg, window, seed):
    settings = read(BASE / 'configs/tokenizer-temporal-v1.json')
    settings['fold'] = cfg['windows'][window]
    core.OUT = ROOT / window
    core.OUT.mkdir(exist_ok=True)
    data = core.ReconstructionData(settings, seed)
    core.history.OUT = core.OUT / 'predictors'
    return data


def forecast(data, window, seed):
    root = ROOT / window / f'predictors/seed{seed}/dense_3720k_equal'
    dest = root / 'forecast'
    if (dest / 'completed.json').exists():
        core.verify(dest)
        return
    dest.mkdir(exist_ok=True)
    chunks = dest / 'chunks'
    chunks.mkdir(exist_ok=True)
    ids = data.selected(None, 'evaluation', 32)
    np.testing.assert_array_equal(ids, np.load(old_forecast(window, seed) / 'row-ids.npy'))
    np.save(dest / 'row-ids.npy', ids)
    model, saved = restore_model(root / 'training/best.pt', 'mps')
    assert saved['dataset_manifest_sha256'] == file_hash(core.DATA / 'completed.json')
    model.eval()
    decoder = core.load_decoder(DECODER)
    cached = CONFIRM / f'seed{seed}/dense_3720k_equal/evaluation/sampling'
    if window == '2023h2':
        np.testing.assert_array_equal(ids, np.load(cached / 'row-ids.npy'))
        cached_hashes = read(cached / 'settings.json')['chunks']
    for start in range(0, len(ids), 4):
        batch = ids[start:start + 4]
        path = chunks / f'{start:06d}.npz'
        if path.exists():
            continue
        a, b, stamps, _ = data.tensors(batch, 'mps')
        if window == '2023h2':
            source = cached / 'chunks' / path.name
            assert file_hash(source) == cached_hashes[path.name]
            with np.load(source) as z:
                np.testing.assert_array_equal(batch, z['row_ids'])
                pairs = [torch.cat([x[:, None, :60].expand(-1, 64, -1),
                    torch.as_tensor(z[k], device='mps', dtype=torch.long)], dim=-1)
                    for x, k in zip([a, b], ['s1', 's2'])]
        else:
            pairs = generate_tokens(model, a[:, :60], b[:, :60], stamps[:, :60], stamps[:, 60:],
                samples=64, seed=17 + start, **SAMPLING)
        paths, valid = decode_paths(decoder, pairs, data.a['mean'][batch], data.a['scale'][batch], 5)
        with path.with_suffix('.tmp').open('wb') as f:
            np.savez_compressed(f, row_ids=batch, paths=paths, valid=valid,
                s1=pairs[0][:, :, -5:].cpu().numpy(), s2=pairs[1][:, :, -5:].cpu().numpy())
        path.with_suffix('.tmp').replace(path)
        if start % 512 == 0:
            progress = dict(window=window, seed=seed, inputs=start + len(batch), total=len(ids), at=utc_now())
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
        for key, value in zip(['s1', 's2'], pairs):
            np.testing.assert_array_equal(z[key], value[:, :, -5:].cpu().numpy())
        np.testing.assert_array_equal(z['paths'], paths)
        np.testing.assert_array_equal(z['valid'], valid)
    write_json(dest / 'lineage.json', dict(checkpoint=str(root / 'training/best.pt'),
        checkpoint_sha256=file_hash(root / 'training/best.pt'), decoder_sha256=file_hash(DECODER),
        sampling=SAMPLING, samples=64, seed=17, first_batch_exact_replay=True,
        cached_tokens_reused=window == '2023h2',
        chunks={p.name: file_hash(p) for p in chunks.iterdir()}))
    core.finish(dest)
    del model, decoder, output
    gc.collect()
    torch.mps.empty_cache()


def score(data, window, seed):
    root = ROOT / window / f'predictors/seed{seed}/dense_3720k_equal'
    ids = np.load(root / 'forecast/row-ids.npy')
    mapping = dict(old=np.load(old_forecast(window, seed) / 'adapted-paths.npy', mmap_mode='r'),
                   refreshed=np.load(root / 'forecast/paths.npy', mmap_mode='r'))
    frames, coverage = {}, []
    for variant, paths in mapping.items():
        frames[variant], cov = score_paths(paths, data.a['future'][ids], data.a['valid'][ids],
            data.a['last'][ids, 3].astype(float), data.rows.iloc[ids].date.to_numpy(),
            horizons=(2, 5), minimum_fraction=.25)
        coverage.append(cov.assign(variant=variant))
    common, daily, metrics = common_scores(frames)
    dest = root / 'comparison'
    dest.mkdir(exist_ok=True)
    common.to_parquet(dest / 'common.parquet', index=False)
    daily.to_csv(dest / 'daily.csv', index=False)
    metrics.to_csv(dest / 'metrics.csv', index=False)
    pd.concat(coverage).to_csv(dest / 'coverage.csv', index=False)
    core.finish(dest)
    return daily.assign(window=window, seed=seed)


def aggregate(daily):
    dest = ROOT / 'results'
    dest.mkdir(exist_ok=True)
    daily.to_csv(dest / 'daily.csv', index=False)
    records = []
    for window in ['2023h2', '2024h2']:
        for horizon in [2, 5]:
            for target in ['maximum', 'minimum', 'range', 'high_low', 'mean']:
                sub = daily[(daily.window == window) & (daily.horizon == horizon)]
                if target == 'high_low':
                    sub = sub[sub.target.isin(['maximum', 'minimum'])]
                elif target != 'mean':
                    sub = sub[sub.target == target]
                for metric in ['mae', 'crps', 'coverage80', 'width80', 'interval_score80']:
                    paired = sub.groupby(['seed', 'date', 'variant'])[metric].mean().unstack('variant')
                    for seed in SEEDS + ['mean']:
                        p = (paired if seed == 'mean' else paired.loc[[seed]]).groupby('date').mean()
                        delta = (p.refreshed - p.old).to_numpy()
                        rng = np.random.default_rng(314159)
                        indices = (rng.integers(len(p), size=(2000, int(np.ceil(len(p) / 10)), 1)) + np.arange(10)) % len(p)
                        indices = indices.reshape(2000, -1)[:, :len(p)]
                        lo, hi = np.quantile(delta[indices].mean(1), [.025, .975])
                        records.append(dict(window=window, horizon=horizon, target=target, seed=seed, metric=metric,
                            old=p.old.mean(), refreshed=p.refreshed.mean(), delta=delta.mean(),
                            relative_change=p.refreshed.mean() / p.old.mean() - 1,
                            ci_low=lo, ci_high=hi, dates=len(p), date_win_fraction=float((delta < -1e-12).mean())))
    pd.DataFrame(records).to_csv(dest / 'paired.csv', index=False)
    core.finish(dest)


def main():
    initialize()
    cfg = read(ROOT / 'protocol.json')
    manifest = read(ROOT / 'source-manifest.json')
    assert file_hash(ROOT / 'protocol.json') == manifest['protocol']
    for name, digest in {**manifest['sources'], **{str(ROOT / k): v for k, v in manifest['code'].items()}}.items():
        assert file_hash(Path(name)) == digest, name
    torch.set_num_threads(4)
    for seed in [29, 43]:
        data = dataset(cfg, '2024h2', seed)
        core.history.train(data, f'seed{seed}', 'dense_3720k', False)
        del data
        gc.collect()
    all_daily = []
    for window in cfg['windows']:
        data = dataset(cfg, window, 17)
        for seed in SEEDS:
            forecast(data, window, seed)
            all_daily.append(score(data, window, seed))
    aggregate(pd.concat(all_daily, ignore_index=True))
    write_json(ROOT / 'run-completed.json', dict(passed=True, at=utc_now(), new_training_runs=2,
        reused_training_runs=4, forecast_runs=6, sealed_holdout_opened=False))


if __name__ == '__main__':
    main()
