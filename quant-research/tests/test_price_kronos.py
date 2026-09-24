import numpy as np

from quant_research.price_kronos import path_quantiles


def test_paths_are_not_mean_predictions_and_invalid_paths_are_counted():
    paths = np.full((1, 16, 5, 6), 10.)
    paths[0, :8, :, :4] = 9.
    paths[0, 8:, :, :4] = 11.
    q, valid = path_quantiles(paths, np.array([10.]))
    assert valid.sum() == 16
    assert q[0, 4, 3, 0] < 0 < q[0, 4, 3, 2]
    paths[0, :9, :, 1] = 1.  # Impossible high below low/open/close.
    q, valid = path_quantiles(paths, np.array([10.]))
    assert valid.sum() == 7
    assert np.isnan(q).all()
