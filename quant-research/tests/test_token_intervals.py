import numpy as np
import pytest

from quant_research.token_intervals import apply_scale, fit_scale, interval_metrics


def test_date_weighting_is_invariant_to_within_date_replication():
    q=np.tile([1.,2.,3.],(4,1))
    y=np.array([2.,3.,5.,6.])
    d=np.array(['a','a','b','b'])
    expected=fit_scale(q,y,d)['scale']
    ids=np.array([0,1,0,1,0,1,2,3])
    assert fit_scale(q[ids],y[ids],d[ids])['scale']==expected
    assert fit_scale(q[::-1],y[::-1],d[::-1])['scale']==expected


def test_known_empirical_scale_and_median_preservation():
    q=np.tile([1.,2.,3.],(5,1))
    y=np.array([2.,3.,4.,5.,9.])
    fit=fit_scale(q,y,np.arange(5))
    assert fit['scale']==3
    out=apply_scale(q,fit['scale'],0.)
    np.testing.assert_array_equal(out[:,1],q[:,1])
    np.testing.assert_array_equal(out[:,0],0.)
    np.testing.assert_array_equal(out[:,2],5.)
    assert interval_metrics(out,y)['coverage80'].mean()==.8


def test_support_intersection_keeps_coverage_and_improves_score():
    q=np.array([[1.,2.,3.]])
    y=np.array([4.])
    unrestricted=apply_scale(q,4.,-100.)
    supported=apply_scale(q,4.,0.)
    a,b=interval_metrics(unrestricted,y),interval_metrics(supported,y)
    np.testing.assert_array_equal(a['coverage80'],b['coverage80'])
    assert b['interval_score80'][0]<a['interval_score80'][0]
    np.testing.assert_array_equal(a['mae'],b['mae'])


def test_degenerate_quantiles_and_bad_inputs():
    q=np.ones((5,3))
    y=np.arange(5.)
    fit=fit_scale(q,y,np.arange(5))
    assert np.isfinite(fit['scale']) and fit['degenerate_halfwidths']==5
    assert interval_metrics(apply_scale(q,fit['scale'],0),y)['coverage80'].mean()>=.8
    with pytest.raises(ValueError):
        fit_scale(q,np.full(5,np.nan),np.arange(5))
    with pytest.raises(ValueError):
        fit_scale(q[:,::-1]+[2,0,0],y,np.arange(5))
    with pytest.raises(ValueError):
        apply_scale(q,.5,0)


def test_calibration_does_not_narrow_and_scores_penalize_width():
    q=np.tile([1.,2.,3.],(5,1))
    y=np.full(5,2.)
    assert fit_scale(q,y,np.arange(5))['scale']==1
    baseline=interval_metrics(q,y)
    expanded=interval_metrics(apply_scale(q,2.,0),y)
    np.testing.assert_array_equal(baseline['mae'],expanded['mae'])
    assert expanded['interval_score80'].mean()>baseline['interval_score80'].mean()
