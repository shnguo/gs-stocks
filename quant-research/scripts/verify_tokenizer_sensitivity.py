"""Independently check scores after retaining identical legal token trajectories."""
import numpy as np
import pandas as pd
from tokenizer_forecast_sensitivity import DATA, ROOT
from verify_token_pipeline_audit import independent_rows, legal, verify_aggregation

from quant_research.storage import file_hash, utc_now, write_json


def main():
    dest = ROOT / 'common-draw-sensitivity'
    rows = pd.read_parquet(DATA / 'rows.parquet')
    arrays = {k: np.load(DATA / f'{k}.npy', mmap_mode='r') for k in ['future', 'valid', 'last']}
    daily = pd.read_csv(dest / 'daily.csv')
    metrics = pd.read_csv(dest / 'metrics.csv')
    coverage = pd.read_csv(dest / 'coverage.csv')
    checked = 0
    for seed in [17, 29, 43]:
        folder = ROOT / f'predictors/seed{seed}/dense_3720k_equal/decoder-forecast'
        ids = np.load(folder / 'row-ids.npy')
        paths = {name: np.load(folder / f'{name}-paths.npy') for name in ['frozen', 'adapted']}
        frames = []
        for horizon in [2, 5]:
            shared = legal(paths['frozen'][:, :, :horizon]) & legal(paths['adapted'][:, :, :horizon])
            known = arrays['valid'][ids, :horizon].all(1)
            expected = coverage[(coverage.seed == seed) & (coverage.horizon == horizon)].iloc[0]
            assert int(expected.known) == int(known.sum())
            assert int(expected.usable) == int((known & (shared.sum(1) >= 16)).sum())
            assert int(expected.common_legal_paths) == int(shared.sum())
            assert int(expected.total_paths) == shared.size
            for name, values in paths.items():
                masked = values.copy()
                masked[~shared] = np.nan
                scores = independent_rows(masked, ids, rows, arrays['future'], arrays['valid'], arrays['last'])
                frames.append(scores[scores.horizon == horizon].assign(variant=name))
        frame = pd.concat(frames, ignore_index=True)
        verify_aggregation(frame, daily[daily.seed == seed], metrics[metrics.seed == seed])
        checked += len(frame)
    write_json(dest / 'independent-verification.json', dict(
        passed=True, at=utc_now(), row_target_scores=checked,
        method='Explicit pairwise CRPS, independently computed OHLC masks and date means',
        files={name: file_hash(dest / name) for name in ['daily.csv', 'metrics.csv', 'coverage.csv', 'completed.json']}))
    print('Common-draw independent verification passed:', checked, flush=True)


if __name__ == '__main__':
    main()
