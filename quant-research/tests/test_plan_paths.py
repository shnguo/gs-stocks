import numpy as np
import pytest

from quant_research.plan_paths import apply_sparse_calibration, calibrate_sparse, path_heads
from quant_research.plan_value import HEADS


def test_no_fill_missing_paths_and_ambiguous_return_are_distinct():
    # Constant no-fill path above all limits, a filled path, and malformed paths.
    paths = np.tile([101., 102., 100.5, 101., 10., 1000.], (4, 16, 5, 1))
    paths[1] = np.tile([98., 99., 97., 98., 10., 1000.], (16, 5, 1))
    paths[2] = np.nan
    paths[3] = paths[1]
    paths[3, :, 1:, 1] = 110.
    paths[3, :, 1:, 2] = 90.
    pred, coverage = path_heads(paths, np.full(4, 100.))
    np.testing.assert_equal(pred[HEADS[0]][0], 0)
    assert np.isnan(pred[HEADS[2]][0]).all()
    assert np.isnan(pred[HEADS[0]][2]).all()
    assert np.isfinite(pred[HEADS[2]][1]).any()
    assert np.isnan(pred[HEADS[2]][3]).all()
    assert (coverage['ambiguous_paths'][3] == 16).all()
    assert not coverage['valid_paths'][2].any()
    assert (pred[HEADS[1]][3] == 0).all()


def test_insufficient_samples_cannot_become_zero_or_calibrated_constants():
    paths = np.tile([98., 99., 97., 98., 10., 1000.], (2, 16, 5, 1))
    paths[1, 7:] = np.nan
    pred, _ = path_heads(paths, np.full(2, 100.))
    calibration = {h: [dict(kind='constant', value=.2) for _ in range(12)] for h in HEADS}
    cal = apply_sparse_calibration(pred, calibration)
    for h in HEADS:
        np.testing.assert_equal(np.isfinite(cal[h]), np.isfinite(pred[h]))
        assert np.isnan(cal[h][1]).all()
    with pytest.raises(ValueError):
        path_heads(paths, np.full(2, 100.), min_valid=17)


def test_sparse_calibration_insufficient_dates_preserves_identity():
    paths = np.tile([98., 99., 97., 98., 10., 1000.], (2, 16, 5, 1))
    pred, _ = path_heads(paths, np.full(2, 100.))
    net = np.zeros((2, 12))
    outcomes = dict(filled=np.ones_like(net), conditional_net_return=net,
        conditional_loss=net, conditional_downside=net)
    calibration = calibrate_sparse(pred, outcomes, ['a', 'b'])
    restored = apply_sparse_calibration(pred, calibration)
    for h in HEADS:
        np.testing.assert_equal(pred[h], restored[h])
