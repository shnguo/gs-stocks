"""Independent output arithmetic, cohort, calendar and saved-booster verification."""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from quant_research.storage import file_hash, utc_now, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    state = json.loads((out/'run-status.json').read_text())
    if state['status'] != 'completed':
        raise ValueError('Cannot verify incomplete run')
    for name, expected in state['files'].items():
        path = (out/name).resolve()
        if not path.is_relative_to(out) or file_hash(path) != expected:
            raise ValueError(f'Hash mismatch {name}')
    protocol = json.loads((out/'protocol.json').read_text())
    metrics = pd.read_csv(out/'daily-metrics.csv')
    verified_rows, checked_dates = 0, 0
    for fold in protocol['fold_indices']:
        base_keys, base_targets = None, None
        for arm in protocol['arms']:
            p = out/f'fold-{fold:02d}-{arm}'
            cfg = json.loads((p/'config.json').read_text())
            partitions = [pd.read_parquet(p/f'{s}-rows.parquet') for s in
                          ['train', 'selection', 'calibration', 'evaluation']]
            for a, b in zip(partitions, partitions[1:]):
                assert a.label_end.max() < b.date.min()
            rows = partitions[-1]
            date_ends = rows.groupby('date').label_end.max().sort_index()
            assert all(a <= b for a, b in zip(date_ends.iloc[:-1], date_ends.index[1:]))
            assert rows.label_end.max() < cfg['sealed_holdout_start']
            with np.load(p/'evaluation.npz') as archive:
                x, y = archive['x'], archive['targets']
            keys = rows[['date', 'instrument_id']]
            if base_keys is None:
                base_keys, base_targets = keys, y.copy()
            else:
                pd.testing.assert_frame_equal(base_keys, keys)
                np.testing.assert_array_equal(base_targets, y)
            prediction = np.load(p/'lightgbm-evaluation.npy')
            indices = np.unique(np.linspace(0, len(x)-1, 30, dtype=int))
            restored = np.column_stack([lgb.Booster(model_file=str(
                p/f'lightgbm-models/d5-close-q{q}.txt')).predict(x[indices], num_threads=1)
                for q in [.1, .5, .9]])
            np.testing.assert_allclose(np.sort(restored, axis=1), prediction[indices, 4, 3], atol=1e-12)
            verified_rows += len(indices)
            for date, group in rows.groupby('date'):
                ids = group.index.to_numpy()
                actual = y[ids, 4, 3]
                good = np.isfinite(actual)
                q = prediction[ids, 4, 3][good]
                residual = actual[good, None]-q
                loss = np.maximum(residual*np.array([.1, .5, .9]),
                                  residual*np.array([-.9, -.5, -.1])).mean()
                values = dict(pinball=loss, median_mae=np.abs(residual[:, 1]).mean(),
                    coverage80=((actual[good] >= q[:, 0]) & (actual[good] <= q[:, 2])).mean(),
                    width80=(q[:, 2]-q[:, 0]).mean())
                recorded = metrics.loc[(metrics.window == f'fold-{fold:02d}') &
                                       (metrics.model == arm) & (metrics.date == date)].iloc[0]
                for name, value in values.items():
                    np.testing.assert_allclose(value, recorded[name], atol=1e-12)
                checked_dates += 1
    write_json(out/'verification.json', {'verified_at': utc_now(), 'passed': True,
        'hashes_checked': len(state['files']), 'saved_booster_rows_reproduced': verified_rows,
        'model_dates_recalculated': checked_dates, 'cohort_and_partition_checks': True,
        'scope': 'Full recorded artifact hashes; all learned-model date metrics for day5 close; sample of saved day5-close boosters. Not a full execution or PIT audit.'})
    print('Verified', len(state['files']), 'hashes;', checked_dates, 'model-dates;', verified_rows, 'booster rows')


if __name__ == '__main__':
    main()
