import numpy as np
import pytest

from quant_research.range_decomposition import attribute_mae_change, point_components


def test_translation_and_width_errors_have_different_endpoint_effects():
    p=point_components([14,14,12], [10,6,10], [12,12,12], [8,8,8])
    np.testing.assert_array_equal(p['center_error'], [2,0,1])
    np.testing.assert_array_equal(p['half_width_error'], [0,2,-1])
    np.testing.assert_array_equal(p['endpoint_mae'], [2,2,1])
    np.testing.assert_array_equal(p['endpoint_mse'], p['center_mse']+p['half_width_mse'])
    np.testing.assert_array_equal(p['endpoint_mae'], np.maximum(p['center_mae'], p['half_width_mae']))


def test_attribution_handles_nonadditive_interaction_and_cancellation():
    out=attribute_mae_change([2,2], [1,1], [1,1], [2,0])
    np.testing.assert_allclose(out['center'], [-.5,-1])
    np.testing.assert_allclose(out['half_width'], [.5,0])
    np.testing.assert_allclose(out['total'], [0,-1])
    np.testing.assert_allclose(out['center']+out['half_width'],out['total'])


def test_endpoint_medians_do_not_define_median_path_range_or_center():
    high=np.array([10.,11.,20.])
    low=np.array([0.,9.,8.])
    p=point_components(np.median(high),np.median(low),12,7)
    assert 2*p['half_width'] == 3
    assert np.median(high-low) == 10
    assert p['center'] == 9.5
    assert np.median((high+low)/2) == 10


def test_reject_invalid_endpoints():
    with pytest.raises(ValueError,match='High'):
        point_components(2,3,4,1)
    with pytest.raises(ValueError,match='Finite'):
        point_components(np.nan,1,4,1)
