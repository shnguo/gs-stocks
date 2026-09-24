"""Independent scores, paired estimates, split boundaries and checkpoint checks."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from token_history_run import Dataset
from token_training_refresh_run import (
    BASE,
    CALIBRATION,
    DECODER,
    ORIGINAL,
    ROOT,
    SEEDS,
    old_forecast,
    read,
)
from verify_token_pipeline_audit import compare_group, legal, verify_aggregation

from quant_research.storage import file_hash, utc_now, write_json
from quant_research.token_transformer import restore_model


def main():
    assert read(ROOT / 'run-completed.json')['passed']
    data = Dataset()
    torch.set_num_threads(4)
    data_root = BASE / 'artifacts/token-history-data-20260915-v1'
    data_manifest = read(data_root / 'completed.json')
    consumed = ['s1.npy', 's2.npy', 'valid.npy', 'future.npy', 'mean.npy', 'scale.npy',
                'last.npy', 'calendar-stamps.npy', 'rows.parquet', 'protocol.json']
    for name in consumed:
        assert file_hash(data_root / name) == data_manifest['files'][name], name
    protocol = read(ROOT / 'protocol.json')
    manifest = read(ROOT / 'source-manifest.json')
    assert file_hash(ROOT / 'protocol.json') == manifest['protocol']
    for name, digest in manifest['sources'].items():
        assert file_hash(Path(name)) == digest, name
    for name, digest in manifest['code'].items():
        assert file_hash(ROOT / name) == digest, name
    for done in ROOT.rglob('completed.json'):
        obj = read(done)
        assert obj['passed']
        for name, digest in obj['files'].items():
            assert file_hash(done.parent / name) == digest, done.parent / name
    checks, all_daily = [], []
    for window, fold in protocol['windows'].items():
        split = {}
        for part in ['train', 'selection', 'evaluation']:
            ids = np.load(ROOT / window / f'{part}-ids.npy')
            start, end = fold[part]
            expected = np.flatnonzero(data.rows.date.ge(start) & data.rows.date.lt(end) & data.rows.label_end.lt(end))
            np.testing.assert_array_equal(ids, expected)
            assert data.rows.iloc[ids].label_end.max() < end < protocol['sealed_holdout_start']
            split[part] = ids
        for seed in SEEDS:
            root = ROOT / window / f'predictors/seed{seed}/dense_3720k_equal'
            summary = read(root / 'training/summary.json')
            history = read(root / 'training/history.json')
            best = min(history, key=lambda x: x['selection_ce'])
            assert best['epoch'] == summary['selected_epoch']
            assert abs(best['selection_ce'] - summary['best_selection_ce']) < 1e-12
            visited = np.load(root / 'training/visited-row-ids.npy')
            assert np.isin(visited, split['train']).all()
            assert data.a['valid'][visited].any(1).all()
            assert len(visited) == summary['unique_examples_visited']
            assert summary['parameters'] == 3720448 and summary['horizon_weights'] is None
            model, saved = restore_model(root / 'training/best.pt', 'cpu')
            model.eval()
            assert saved['config'] == data.cfg['models']['dense_3720k']
            assert saved['dataset_manifest_sha256'] == file_hash(BASE / 'artifacts/token-history-data-20260915-v1/completed.json')
            with np.load(root / 'training/reload-reference.npz') as z, torch.inference_mode():
                assert np.isin(z['ids'], split['selection']).all()
                a, b, stamps, _ = data.tensors(z['ids'], 'cpu')
                logits = model.forecast_logits(a[:, :-1], b[:, :-1], stamps[:, :-1], a[:, 1:], 59)
                np.testing.assert_array_equal(logits[0].numpy(), z['coarse'])
                np.testing.assert_array_equal(logits[1].numpy(), z['fine'])
            del model
            ids = np.load(root / 'forecast/row-ids.npy')
            expected = data.rows.iloc[split['evaluation']].groupby('date').head(32).row_id.to_numpy()
            np.testing.assert_array_equal(ids, expected)
            np.testing.assert_array_equal(ids, np.load(old_forecast(window, seed) / 'row-ids.npy'))
            assert len(np.unique(ids)) == len(ids)
            mapping = dict(old=np.load(old_forecast(window, seed) / 'adapted-paths.npy', mmap_mode='r'),
                           refreshed=np.load(root / 'forecast/paths.npy', mmap_mode='r'))
            lineage = read(root / 'forecast/lineage.json')
            old_lineage = read(old_forecast(window, seed) / 'verification.json')
            assert old_lineage['predictor_sha256'] == file_hash(ORIGINAL / f'predictors/seed{seed}/dense_3720k_equal/training/best.pt')
            assert old_lineage['decoder_sha256'] == lineage['decoder_sha256'] == file_hash(DECODER)
            assert lineage['first_batch_exact_replay']
            assert file_hash(Path(lineage['checkpoint'])) == lineage['checkpoint_sha256']
            for name, digest in lineage['chunks'].items():
                path = root / 'forecast/chunks' / name
                assert file_hash(path) == digest
                start = int(path.stem)
                with np.load(path) as z:
                    np.testing.assert_array_equal(z['row_ids'], ids[start:start + 4])
                    np.testing.assert_array_equal(z['paths'], mapping['refreshed'][start:start + len(z['row_ids'])])
            common = pd.read_parquet(root / 'comparison/common.parquet')
            daily = pd.read_csv(root / 'comparison/daily.csv')
            metrics = pd.read_csv(root / 'comparison/metrics.csv')
            n = compare_group(mapping, ids, common, data.rows, data.a['future'], data.a['valid'], data.a['last'])
            verify_aggregation(common, daily, metrics)
            coverage = pd.read_csv(root / 'comparison/coverage.csv')
            for v, paths in mapping.items():
                for h in [2, 5]:
                    mask = legal(paths[:, :, :h])
                    known = data.a['valid'][ids, :h].all(1)
                    usable = known & (mask.sum(1) >= 16)
                    cov = coverage[(coverage.variant == v) & (coverage.horizon == h)].iloc[0]
                    assert cov.usable == usable.sum() and cov.known == known.sum()
                    assert cov.valid_paths == mask.sum() and cov.total_paths == mask.size
            checks.append(dict(window=window, seed=seed, score_rows=n, inputs=len(ids),
                dates=int(data.rows.iloc[ids].date.nunique()), selected_epoch=summary['selected_epoch'],
                cpu_reload_exact=True, lineage_verified=True, raw_scores_verified=True))
            all_daily.append(daily.assign(window=window, seed=seed))
            print('Verified', checks[-1], flush=True)
    daily = pd.concat(all_daily, ignore_index=True)
    actual = pd.read_csv(ROOT / 'results/daily.csv')
    pd.testing.assert_frame_equal(daily, actual, check_dtype=False, atol=1e-12, rtol=1e-12)
    paired = pd.read_csv(ROOT / 'results/paired.csv', dtype={'seed': str})
    for row in paired.itertuples():
        subset = daily[(daily.window == row.window) & (daily.horizon == row.horizon)]
        if row.target == 'high_low':
            subset = subset[subset.target.isin(['maximum', 'minimum'])]
        elif row.target != 'mean':
            subset = subset[subset.target == row.target]
        if row.seed != 'mean':
            subset = subset[subset.seed == int(row.seed)]
        values = subset.pivot_table(index='date', columns='variant', values=row.metric, aggfunc='mean').sort_index()
        baseline, candidate = values.old.to_numpy(), values.refreshed.to_numpy()
        delta = candidate - baseline
        rng = np.random.default_rng(314159)
        starts = rng.integers(len(delta), size=(2000, int(np.ceil(len(delta) / 10))))
        estimates = []
        for block_starts in starts:
            idx = np.concatenate([(int(i) + np.arange(10)) % len(delta) for i in block_starts])[:len(delta)]
            estimates.append(delta[idx].mean())
        low, high = np.percentile(estimates, [2.5, 97.5])
        np.testing.assert_allclose([row.old, row.refreshed, row.delta, row.relative_change, row.ci_low, row.ci_high, row.date_win_fraction],
            [baseline.mean(), candidate.mean(), delta.mean(), candidate.mean() / baseline.mean() - 1,
             low, high, (delta < -1e-12).mean()], atol=1e-10, rtol=1e-10)
    prior = 0
    for name, digest in read(CALIBRATION / 'final-verification.json')['files'].items():
        if Path(name).is_relative_to(BASE / 'artifacts'):
            assert file_hash(Path(name)) == digest, name
            prior += 1
    write_json(ROOT / 'independent-verification.json', dict(passed=True, at=utc_now(), runs=checks,
        paired_contrasts=len(paired), prior_calibration_artifact_files_unchanged=prior,
        all_source_hashes_match=True, consumed_dataset_files_verified=len(consumed),
        calibration_coefficients_reused=False, sealed_holdout_opened=False))
    print('Independent verification passed:', len(paired), 'contrasts', flush=True)


if __name__ == '__main__':
    main()
