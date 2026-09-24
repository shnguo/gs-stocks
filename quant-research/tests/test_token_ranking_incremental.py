import numpy as np
import pandas as pd
import pytest
from token_ranking_incremental import grouped_batches, validate_rows


def test_matched_batches_cover_each_row_once_and_keep_date_peers():
    rows = pd.DataFrame({'date': ['a']*128 + ['b']*128 + ['c']*12 + ['d']*400})
    batches = list(grouped_batches(rows, 17, 256))
    assert all(len(b) <= 256 for b in batches)
    np.testing.assert_array_equal(np.sort(np.concatenate(batches)), np.arange(len(rows)))
    for day in ['a', 'b', 'c']:
        assert sum(day in set(rows.iloc[b].date) for b in batches) == 1
    for left, right in zip(batches, grouped_batches(rows, 17, 256), strict=True):
        np.testing.assert_array_equal(left, right)
    assert [b.tolist() for b in batches] != [b.tolist() for b in grouped_batches(rows, 29, 256)]


def fixture_rows():
    rows = pd.DataFrame(dict(instrument_id=['x', 'y'], date=['2026-09-08', '2023-12-01'],
        label_end=['2026-09-15', '2023-12-08'], role=['initial_recent', 'historical_replay']))
    arrays = dict(s1=np.zeros((2, 65)), valid=np.ones((2, 5), bool))
    info = dict(signal_date='2026-09-15', observed_labels_through='2026-09-15', new_signal_through='2026-09-08')
    return rows, arrays, info


def test_maturity_cutoff_and_identity_fail_closed():
    rows, arrays, info = fixture_rows()
    validate_rows(rows, arrays, info)
    rows.loc[0, 'label_end'] = '2026-09-16'
    with pytest.raises(ValueError, match='observed close'):
        validate_rows(rows, arrays, info)
    rows, arrays, info = fixture_rows()
    arrays['valid'][1, 3] = False
    with pytest.raises(ValueError, match='complete mature'):
        validate_rows(rows, arrays, info)
    rows, arrays, info = fixture_rows()
    rows.loc[1] = rows.loc[0]
    with pytest.raises(ValueError, match='identity'):
        validate_rows(rows, arrays, info)


def test_bridge_does_not_silently_skip_intervening_updates():
    rows, arrays, info = fixture_rows()
    rows.loc[0, 'role'] = 'newly_mature'
    with pytest.raises(ValueError, match='initial daily update'):
        validate_rows(rows, arrays, info)
