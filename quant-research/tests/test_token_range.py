"""Check extrema versus terminal price, chronology, and baseline invariants."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from compare_token_range import extrema, historical_paths, order_class, score


def test_extrema_uses_entire_window_not_terminal_bar():
    p = np.array([[10, 12, 9, 10], [10, 11, 8, 10], [10, 11, 9, 10]], float)
    np.testing.assert_allclose(extrema(p, 10), [20,-20,40])


def test_same_extrema_opposite_order_and_daily_ambiguity():
    low = [10, 10.5, 9, 10]
    high = [10, 12, 9.5, 10]
    both = [10, 12, 9, 10]
    quiet = [10, 10.5, 9.5, 10]
    np.testing.assert_array_equal(order_class(np.array([[low,high],[high,low],[both,quiet]])),[0,1,2])
    assert order_class(np.array([low,high,low])) == 2


def test_historical_baseline_reproducible_scale_equivariant_and_ohlc_valid():
    close = 10*np.exp(np.sin(np.arange(60))*.02)
    bars = np.column_stack([close,close*1.02,close*.98,close])
    a, ix = historical_paths(bars,'stock:date')
    b, jx = historical_paths(bars*7,'stock:date')
    np.testing.assert_array_equal(ix,jx)
    np.testing.assert_allclose(a*7,b)
    np.testing.assert_array_equal(historical_paths(bars,'stock:date')[0],a)
    assert ix.min() >= 0 and ix.max() <= 54
    assert (a[...,1] >= a.max(-1)).all() and (a[...,2] <= a.min(-1)).all()


def test_flat_history_produces_flat_baseline():
    p, _ = historical_paths(np.full((60,4),10.),'flat')
    np.testing.assert_array_equal(p,np.full((32,5,4),10.))


def test_known_crps_and_interval_score_penalize_widening():
    assert score([0,2],1)['crps'] == .5
    assert score([2,2],1)['crps'] == 1
    assert score([-10,10],0)['interval_score80'] > score([-1,1],0)['interval_score80']
