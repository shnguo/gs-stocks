import numpy as np
import pandas as pd
import pytest

from quant_research.token_market_context import (
    CONTEXT,
    contextual_history,
    dated_industries,
    signal_context,
)


def histories(n=8):
    x = np.ones((n, 60, 7), float)
    for i in range(n):
        x[i, :, 0] = x[i, :, 3] = 10*np.exp(np.arange(60)*(.001+i*.0001))
    x[:, :, 1] = x[:, :, 3]*1.01
    x[:, :, 2] = x[:, :, 3]*.99
    x[:, :, 4] = 100
    x[:, :, 5] = 1000
    return x


def test_independent_context_returns_breadth_and_liquidity():
    x = histories()
    x[:, -1, 5] = 2000
    index = np.arange(100., 121.)
    result = signal_context(x, ['sector']*8, index)
    expected = x[:, -1, 3]/x[:, -6, 3]-1
    assert result[0, 1] == pytest.approx(120/115-1)
    assert result[0, 4] == pytest.approx(expected.mean())
    assert result[0, 6] == 1
    assert result[0, 9] == pytest.approx(np.log(2))
    assert result[0, 11] == pytest.approx(expected[1:].mean())
    assert result[0, 14] == pytest.approx(np.log(2))
    assert result[0, 15] == pytest.approx(np.log(8))
    assert result[0, 17] == pytest.approx(expected[0]-expected[1:].mean())


def test_sector_is_external_and_unknown_members_do_not_drop_stocks():
    x = histories()
    groups = ['sector']*7+['']
    index = np.ones(21)*100
    before = signal_context(x, groups, index)
    x[0, -1, :4] *= 1.5
    after = signal_context(x, groups, index)
    np.testing.assert_allclose(before[0, 10:15], after[0, 10:15])
    assert np.isnan(after[-1, 10:15]).all() and after[-1, 15] == 0
    sparse = signal_context(x, ['s']*4+['']*4, index)
    assert np.isnan(sparse[:4, 10:15]).all()
    assert len(sparse) == 8


def test_split_adjustment_and_signal_only_context():
    x = histories()
    index = np.ones(21)*100
    ctx = signal_context(x, ['s']*8, index)
    split = x.copy()
    split[:, :30, :4] /= 2
    split[:, :30, 6] *= 2
    np.testing.assert_allclose(ctx, signal_context(split, ['s']*8, index))
    features = contextual_history(x, ctx)
    assert np.isnan(features[:, :-1, -len(CONTEXT):]).all()
    np.testing.assert_array_equal(features[:, -1, -len(CONTEXT):], ctx)


def test_dated_membership_rejects_future_stale_and_mixed_snapshots():
    frame = pd.DataFrame(dict(instrument_id=['a', 'b'], requested_date=['2024-01-01']*2,
        updateDate=['2023-12-31']*2, industry=['one', 'two']))
    assert dated_industries(frame, ['b', 'unknown'], '2024-01-05').tolist() == ['two', '']
    for day in ['2023-12-31', '2024-01-08']:
        with pytest.raises(ValueError, match='future-dated or stale'):
            dated_industries(frame, ['a'], day)
    frame.loc[0, 'updateDate'] = '2024-01-02'
    with pytest.raises(ValueError, match='future-dated or stale'):
        dated_industries(frame, ['a'], '2024-01-05')


def test_missing_index_is_not_forward_filled():
    index = np.ones(21)
    index[-2] = np.nan
    with pytest.raises(ValueError, match='index closes'):
        signal_context(histories(), ['s']*8, index)
