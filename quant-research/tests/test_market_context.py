import numpy as np
import pandas as pd
import pytest

from quant_research.market_context import MARKET_FEATURES, market_context


def test_market_context_uses_same_signal_cohort_only():
    names = ['return_1', 'return_5', 'return_20', 'volatility_20', 'price_mean_20', 'volume_mean_5']
    x = np.array([[.02, .03, .05, .02, .01, .1], [-.01, .01, .03, .04, -.01, -.1],
                  [.04, .1, .2, .05, .1, .2], [.06, .1, .2, .05, .1, .2]])
    rows = pd.DataFrame(dict(date=['2024-01-02']*2+['2024-01-03']*2, instrument_id=['a', 'b']*2))
    before = market_context(x, rows, names, min_cohort=2)
    assert before.shape == (4, len(MARKET_FEATURES))
    assert before[0, 0] == pytest.approx(.005)
    assert before[0, 5] == .5
    assert before[0, 8] == pytest.approx(.015)
    x[2:] *= 100
    after = market_context(x, rows, names, min_cohort=2)
    np.testing.assert_array_equal(before[:2], after[:2])
    shuffled = market_context(x[::-1], rows.iloc[::-1], names, min_cohort=2)
    np.testing.assert_allclose(shuffled[::-1], after)


def test_market_context_rejects_partial_or_duplicated_cohort():
    names = ['return_1', 'return_5', 'return_20', 'volatility_20', 'price_mean_20', 'volume_mean_5']
    rows = pd.DataFrame(dict(date=['2024-01-02']*2, instrument_id=['a', 'b']))
    with pytest.raises(ValueError, match='Insufficient'):
        market_context(np.ones((2, 6)), rows, names)
    rows.instrument_id = 'a'
    with pytest.raises(ValueError, match='Duplicate'):
        market_context(np.ones((2, 6)), rows, names, min_cohort=2)
