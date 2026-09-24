"""Calibrate each refreshed checkpoint on its own earlier validation forecasts."""
import gc
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import token_interval_calibration_run as calibration
import tokenizer_reconstruction_run as core
import torch
from token_history_run import DATA, Dataset

from quant_research.forecast_audit import TARGETS, extrema
from quant_research.storage import file_hash, utc_now, write_json
from quant_research.token_transformer import (
    decode_paths,
    generate_tokens,
    restore_model,
    valid_bars,
)

BASE = Path('/Users/guo/Documents/stocks/quant-research')
ART = BASE / 'artifacts'
ROOT = ART / 'token-refreshed-calibration-20260915-v1'
REFRESH = ART / 'token-training-refresh-20260915-v1'
OLD = ART / 'token-interval-calibration-20260915-v1'
DECODER = ART / 'tokenizer-reconstruction-20260915-v1/decoder-training/price_consistency/best-qualified.pt'
SEEDS = [17, 29, 43]
SAMPLING = dict(temperature=1., top_p=1., top_k=0)
FIELDS = calibration.FIELDS


def read(path):
    return json.loads(path.read_text())


def predictor(window, seed):
    return REFRESH / window / f'predictors/seed{seed}/dense_3720k_equal'


def cached_tokens(window):
    if window == '2023h2':
        return ART / 'token-pipeline-confirmation-20260915-v1/sampler-selection/full_distribution'
    return ART / 'token-pipeline-audit-20260915-v1/sampling/2024h2/full_distribution'


def initialize():
    cfg = read(BASE / 'configs/token-refreshed-calibration-v1.json')
    if (ROOT / 'protocol.json').exists():
        assert read(ROOT / 'protocol.json') == cfg
        return
    sources = {}
    for prior in [REFRESH, OLD]:
        final = read(prior / 'final-verification.json')
        assert final['passed']
        for name, digest in final['files'].items():
            if Path(name).is_relative_to(ART):
                assert file_hash(Path(name)) == digest, name
        sources[str(prior / 'final-verification.json')] = file_hash(prior / 'final-verification.json')
    ROOT.mkdir()
    write_json(ROOT / 'protocol.json', cfg)
    shutil.copytree(REFRESH / 'code/src', ROOT / 'code/src', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(BASE / 'src/quant_research/token_intervals.py', ROOT / 'code/src/quant_research/token_intervals.py')
    (ROOT / 'code/scripts').mkdir()
    for name in ['token_refreshed_calibration_run.py', 'token_interval_calibration_run.py',
                 'tokenizer_reconstruction_run.py', 'token_history_run.py']:
        shutil.copy2(BASE / 'scripts' / name, ROOT / 'code/scripts' / name)
    sources[str(DECODER)] = file_hash(DECODER)
    sources[str(DATA / 'completed.json')] = file_hash(DATA / 'completed.json')
    for window, periods in cfg['windows'].items():
        dest = ROOT / window
        dest.mkdir()
        write_json(dest / 'protocol.json', dict(**periods, fit=cfg['fit']))
        cache = cached_tokens(window)
        core.verify(cache)
        settings = read(cache / 'settings.json')
        assert settings['sampling'] == SAMPLING and settings['samples'] == 64 and settings['seed'] == 17
        assert settings['checkpoint_sha256'] == file_hash(predictor(window, 17) / 'training/best.pt')
        for name, digest in settings['chunks'].items():
            assert file_hash(cache / 'chunks' / name) == digest
        for name in ['settings.json', 'row-ids.npy', 'completed.json']:
            sources[str(cache / name)] = file_hash(cache / name)
        for part in ['selection', 'evaluation']:
            path = REFRESH / window / f'{part}-ids.npy'
            sources[str(path)] = file_hash(path)
        for seed in SEEDS:
            parent = predictor(window, seed)
            core.verify(parent / 'training')
            core.verify(parent / 'forecast')
            for name in ['training/best.pt', 'training/completed.json', 'forecast/paths.npy',
                         'forecast/row-ids.npy', 'forecast/lineage.json', 'forecast/completed.json']:
                sources[str(parent / name)] = file_hash(parent / name)
        path = OLD / 'intervals' / periods['evaluation']['name'] / 'scores.parquet'
        sources[str(path)] = file_hash(path)
    write_json(ROOT / 'source-manifest.json', dict(at=utc_now(), sources=sources,
        protocol_sha256=file_hash(ROOT / 'protocol.json'),
        code={str(p.relative_to(ROOT)): file_hash(p) for p in (ROOT / 'code').rglob('*.py')}))


def selected(data, window, part, count):
    ids = np.load(REFRESH / window / f'{part}-ids.npy')
    return data.rows.iloc[ids].groupby('date').head(count).row_id.to_numpy(int)


def forecast(data, window, seed):
    dest = ROOT / window / f'calibration-forecasts/seed{seed}'
    if (dest / 'completed.json').exists():
        core.verify(dest)
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    chunks = dest / 'chunks'
    chunks.mkdir(exist_ok=True)
    ids = selected(data, window, 'selection', 16)
    np.save(dest / 'row-ids.npy', ids)
    checkpoint = predictor(window, seed) / 'training/best.pt'
    model, saved = restore_model(checkpoint, 'mps')
    assert saved['dataset_manifest_sha256'] == file_hash(DATA / 'completed.json')
    model.eval()
    decoder = core.load_decoder(DECODER)
    cached = cached_tokens(window)
    if seed == 17:
        np.testing.assert_array_equal(ids, np.load(cached / 'row-ids.npy'))
        hashes = read(cached / 'settings.json')['chunks']
    for start in range(0, len(ids), 4):
        batch = ids[start:start + 4]
        path = chunks / f'{start:06d}.npz'
        if path.exists():
            continue
        a, b, stamps, _ = data.tensors(batch, 'mps')
        if seed == 17:
            source = cached / 'chunks' / path.name
            assert file_hash(source) == hashes[path.name]
            with np.load(source) as z:
                np.testing.assert_array_equal(z['row_ids'], batch)
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
        if start % 256 == 0:
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
        np.testing.assert_array_equal(z['paths'], paths)
        np.testing.assert_array_equal(z['valid'], valid)
        for name, value in zip(['s1', 's2'], pairs):
            np.testing.assert_array_equal(z[name], value[:, :, -5:].cpu().numpy())
    write_json(dest / 'lineage.json', dict(checkpoint=str(checkpoint), checkpoint_sha256=file_hash(checkpoint),
        decoder_sha256=file_hash(DECODER), sampling=SAMPLING, samples=64, generation_seed=17,
        cached_tokens_reused=seed == 17, first_batch_exact_replay=True,
        chunks={p.name: file_hash(p) for p in chunks.iterdir()}))
    core.finish(dest)
    del model, decoder, output
    gc.collect()
    torch.mps.empty_cache()
    return dest


def extract(data, window, part, cfg):
    stage = cfg['windows'][window][part]
    dest = ROOT / window / 'intervals' / stage['name']
    dest.mkdir(parents=True, exist_ok=True)
    records, sources = [], {}
    for seed in SEEDS:
        folder = (ROOT / window / f'calibration-forecasts/seed{seed}' if part == 'calibration'
                  else predictor(window, seed) / 'forecast')
        core.verify(folder)
        ids = np.load(folder / 'row-ids.npy')
        rows = data.rows.iloc[ids]
        assert rows.date.min() >= stage['dates'][0]
        assert rows.label_end.max() < stage['dates'][1] < cfg['sealed_holdout_start']
        assert rows.groupby('date').size().eq(stage['rows_per_date']).all()
        paths = np.load(folder / 'paths.npy', mmap_mode='r')
        ref = data.a['last'][ids, 3].astype(float)
        for horizon in [2, 5]:
            legal = valid_bars(paths[:, :, :horizon]).all(-1)
            known = data.a['valid'][ids, :horizon].all(1)
            truth = extrema(data.a['future'][ids, :horizon].astype(float), ref)
            for i in np.flatnonzero(known & (legal.sum(1) >= 16)):
                values = extrema(paths[i, legal[i], :horizon].astype(float), ref[i])
                q = np.quantile(values, [.1, .5, .9], axis=0)
                for j, target in enumerate(TARGETS):
                    records.append(dict(stage=stage['name'], seed=seed, row_id=int(ids[i]), date=rows.iloc[i].date,
                        horizon=horizon, target=target, lower=q[0, j], median=q[1, j], upper=q[2, j],
                        actual=truth[i, j], draws=int(legal[i].sum())))
        for name in ['paths.npy', 'row-ids.npy', 'lineage.json', 'completed.json']:
            sources[str(folder / name)] = file_hash(folder / name)
    frame = pd.DataFrame(records)
    frame.to_parquet(dest / 'quantiles.parquet', index=False)
    write_json(dest / 'source.json', dict(files=sources, quantiles_sha256=file_hash(dest / 'quantiles.parquet')))
    return frame


def paired_results(daily, label, baseline, candidate):
    records = []
    for window in ['2023h2', '2024h2']:
        for horizon in [2, 5]:
            for target in ['maximum', 'minimum', 'range', 'high_low', 'mean']:
                sub = daily[(daily.window == window) & (daily.horizon == horizon)]
                if target == 'high_low':
                    sub = sub[sub.target.isin(['maximum', 'minimum'])]
                elif target != 'mean':
                    sub = sub[sub.target == target]
                for metric in ['mae', 'coverage80', 'width80', 'interval_score80']:
                    pair = sub.groupby(['seed', 'date', 'variant'])[metric].mean().unstack('variant')
                    for seed in SEEDS + ['mean']:
                        p = (pair if seed == 'mean' else pair.loc[[seed]]).groupby('date').mean()
                        b, x = p[baseline].to_numpy(), p[candidate].to_numpy()
                        rng = np.random.default_rng(314159)
                        idx = (rng.integers(len(p), size=(2000, int(np.ceil(len(p) / 10)), 1)) + np.arange(10)) % len(p)
                        idx = idx.reshape(2000, -1)[:, :len(p)]
                        low, high = np.quantile((x - b)[idx].mean(1), [.025, .975])
                        records.append(dict(comparison=label, window=window, horizon=horizon, target=target,
                            metric=metric, seed=seed, baseline=b.mean(), candidate=x.mean(), delta=(x - b).mean(),
                            relative_change=x.mean() / b.mean() - 1, ci_low=low, ci_high=high, dates=len(p)))
    return pd.DataFrame(records)


def end_to_end(window, cfg):
    stage = cfg['windows'][window]['evaluation']['name']
    old = pd.read_parquet(OLD / 'intervals' / stage / 'scores.parquet')
    old = old[old.variant == 'calibrated'].copy()
    old['variant'] = 'old_calibrated'
    new = pd.read_parquet(ROOT / window / 'intervals' / stage / 'scores.parquet')
    new = new[new.variant == 'calibrated'].copy()
    new['variant'] = 'refreshed_calibrated'
    all_rows = pd.concat([old, new], ignore_index=True)
    keys = ['seed', 'row_id', 'horizon', 'target']
    count = all_rows.groupby(keys).variant.transform('nunique')
    common = all_rows[count == 2].copy()
    dest = ROOT / window / 'end-to-end'
    dest.mkdir(exist_ok=True)
    common.to_parquet(dest / 'common.parquet', index=False)
    daily = common.groupby(['stage', 'seed', 'variant', 'date', 'horizon', 'target'])[FIELDS].mean().reset_index()
    daily.to_csv(dest / 'daily.csv', index=False)
    core.finish(dest)
    return daily.assign(window=window)


def main():
    initialize()
    cfg = read(ROOT / 'protocol.json')
    manifest = read(ROOT / 'source-manifest.json')
    assert file_hash(ROOT / 'protocol.json') == manifest['protocol_sha256']
    for name, digest in manifest['sources'].items():
        assert file_hash(Path(name)) == digest, name
    for name, digest in manifest['code'].items():
        assert file_hash(ROOT / name) == digest, name
    torch.set_num_threads(4)
    data = Dataset()
    fits = {}
    for window in cfg['windows']:
        for seed in SEEDS:
            forecast(data, window, seed)
        frame = extract(data, window, 'calibration', cfg)
        calibration.ROOT = ROOT / window
        settings = read(calibration.ROOT / 'protocol.json')
        fitted = calibration.fit(frame, settings)
        fitted['checkpoint_sha256'] = {str(seed): file_hash(predictor(window, seed) / 'training/best.pt') for seed in SEEDS}
        fitted['decoder_sha256'] = file_hash(DECODER)
        fitted['sampling'] = cfg['sampling']
        write_json(calibration.ROOT / 'fit.json', fitted)
        fits[window] = file_hash(calibration.ROOT / 'fit.json')
        calibration.evaluate(frame, fitted, settings, calibration.ROOT / 'intervals' / settings['calibration']['name'])
    write_json(ROOT / 'fits-frozen.json', dict(at=utc_now(), fits=fits, factors=36,
        fitted_only_from_corresponding_H1=True, frozen_before_this_run_evaluation_scoring=True))
    daily, combined = [], []
    for window in cfg['windows']:
        calibration.ROOT = ROOT / window
        settings = read(calibration.ROOT / 'protocol.json')
        assert file_hash(calibration.ROOT / 'fit.json') == fits[window]
        frame = extract(data, window, 'evaluation', cfg)
        dest = calibration.ROOT / 'intervals' / settings['evaluation']['name']
        d = calibration.evaluate(frame, read(calibration.ROOT / 'fit.json'), settings, dest)
        daily.append(d.assign(window=window))
        combined.append(end_to_end(window, cfg))
    dest = ROOT / 'results'
    dest.mkdir(exist_ok=True)
    daily, combined = pd.concat(daily), pd.concat(combined)
    daily.to_csv(dest / 'calibration-daily.csv', index=False)
    combined.to_csv(dest / 'end-to-end-daily.csv', index=False)
    pd.concat([paired_results(daily, 'calibration', 'raw', 'calibrated'),
               paired_results(combined, 'end_to_end', 'old_calibrated', 'refreshed_calibrated')]).to_csv(dest / 'paired.csv', index=False)
    core.finish(dest)
    write_json(ROOT / 'run-completed.json', dict(passed=True, at=utc_now(), factors=36,
        neural_network_training_runs=0, new_token_generation_runs=4, cached_token_sets_redecoded=2,
        evaluation_forecasts_reused=6, sealed_holdout_opened=False))


if __name__ == '__main__':
    main()
