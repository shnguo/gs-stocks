import numpy as np
import pandas as pd
import pytest

from quant_research.forecast_audit import (
    choose_sampler,
    common_scores,
    empirical_score,
    score_paths,
)


def test_distribution_score_matches_explicit_expectations():
    x=np.array([[1.,2.,3.],[2.,4.,5.],[4.,5.,9.]])
    y=np.array([2.5,4.5,6.])
    got=empirical_score(x,y)
    expected=np.abs(x-y).mean(0)-np.abs(x[:,None]-x[None,:]).mean((0,1))/2
    np.testing.assert_allclose(got['crps'],expected)


def test_later_invalid_bars_cannot_remove_valid_two_day_forecasts():
    p=np.ones((1,8,5,6))*10
    p[...,1]=11
    p[...,2]=9
    p[:,:,4,1]=1
    truth=np.ones((1,5,6))*10
    truth[...,1]=11
    truth[...,2]=9
    frame,cov=score_paths(p,truth,np.ones((1,5),bool),np.array([10.]),['x'])
    assert set(frame.horizon)=={2}
    assert cov.set_index('horizon').usable.to_dict()=={2:1,5:0}


def test_common_cohort_and_dates_are_equal_weighted():
    fields=dict(mae=1.,crps=1.,coverage80=1.,width80=1.,interval_score80=1.,bias=1.)
    a=pd.DataFrame([dict(local_row=i,date=day,horizon=2,target='maximum',**fields) for i,day in enumerate(['a','a','b'])])
    a.loc[2,['mae','crps']]=7
    b=a.copy()
    b.loc[1,'crps']=999
    b=b.drop(index=1)
    common,_,means=common_scores({'a':a,'b':b})
    assert set(common.local_row)=={0,2}
    assert means.crps.tolist()==[4.,4.]


def test_sampler_choice_rejects_test_data_and_coverage_regression():
    rows=[]
    cov=[]
    for name,error,coverage in [('baseline',1.,1.),('good',.9,1.),('bad',.8,.5)]:
        rows.append(dict(partition='selection',fold='f',variant=name,mae=error,crps=error,
            coverage80=.7,width80=1,interval_score80=error,bias=0))
        cov.append(dict(partition='selection',fold='f',variant=name,usable_fraction=coverage))
    config=dict(maximum_relative_mae_increase=.01,maximum_usable_coverage_drop=.01,
        require_interval_score_noninferior=True,minimum_relative_improvement=.01,fallback='baseline')
    frame,coverage=pd.DataFrame(rows),pd.DataFrame(cov)
    assert choose_sampler(frame,coverage,config)['selected']=='good'
    frame['partition']='evaluation'
    with pytest.raises(ValueError,match='validation'):
        choose_sampler(frame,coverage,config)
