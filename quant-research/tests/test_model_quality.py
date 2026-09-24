import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_research.model_quality import (
    assess,
    daily_metrics,
    fit_volatility_baseline,
    label_overlaps,
    volatility_scale,
)


def protocol():
    return json.loads((Path(__file__).parents[1]/'configs/model-quality-v1.json').read_text())


def test_equal_label_boundary_is_disjoint_but_later_end_overlaps():
    rows=pd.DataFrame({'date':['2022-03-01','2022-03-08'],
                       'label_end':['2022-03-08','2022-03-15']})
    assert not label_overlaps(rows)
    rows.loc[0,'label_end']='2022-03-09'
    assert label_overlaps(rows)==[{'signal':'2022-03-01','label_end':'2022-03-09','next_signal':'2022-03-08'}]


def qualifying_metrics():
    p=protocol()
    records=[]
    for fold in p['fold_indices']:
        for model in p['models']:
            for day in range(12):
                learned=model in p['learned_candidates']
                records.append(dict(window=f'fold-{fold:02d}',model=model,date=str(day),
                    pinball=.09 if learned else .1,all_ohlc_pinball=.09 if learned else .1,
                    median_mae=.1 if learned else .12,coverage80=.8,width80=.2,
                    direction_accuracy=.6,prediction_fraction=1.,input_rows=100,known_rows=95,scored_rows=95))
    audit={f'fold-{i:02d}':{'passed':True} for i in p['fold_indices']}
    return pd.DataFrame(records),p,audit


def test_gate_only_advances_to_decision_experiment():
    daily,p,audit=qualifying_metrics()
    _,result=assess(daily,p,audit)
    assert all(r['passed'] for r in result.values())
    assert all(r['next_step']=='decision_value_experiment' for r in result.values())
    assert not any(r['daily_strategy_ready'] for r in result.values())


@pytest.mark.parametrize('defect',['missing_window','overwide','selective_predictions','data_failure','weak_second_baseline'])
def test_gate_rejects_invalid_or_uncompetitive_evidence(defect):
    daily,p,audit=qualifying_metrics()
    mask=daily.model.eq('lightgbm')
    if defect=='missing_window':
        daily=daily.loc[~(mask & daily.window.eq('fold-02'))]
    elif defect=='overwide':
        daily.loc[mask,'coverage80']=.99
    elif defect=='selective_predictions':
        daily.loc[mask,'prediction_fraction']=.9
    elif defect=='data_failure':
        audit['fold-02']['passed']=False
    else:
        daily.loc[daily.model.eq('volatility'),'pinball']=.08
    _,result=assess(daily,p,audit)
    assert not result['lightgbm']['passed']
    json.dumps(result,allow_nan=False)


def test_metrics_preserve_unknowns_and_measure_width():
    rows=pd.DataFrame({'date':['d']*3})
    y=np.zeros((3,5,4))
    y[0,4,3]=np.nan
    q=np.zeros((3,5,4,3))
    q[...,0],q[...,2]=-1.,1.
    q[1]=np.nan
    r=daily_metrics(rows,y,{'a':q},'w').iloc[0]
    assert r.known_rows==2 and r.scored_rows==1
    assert r.coverage80==1 and r.width80==2
    assert r.prediction_fraction==pytest.approx(2/3)


def test_volatility_baseline_uses_date_weights_and_rescales():
    # A one-row date receives half the weight, independent of the 100-row date.
    targets=np.ones((101,5,4))
    targets[-1]=3
    scales=np.full(101,2.)
    baseline=fit_volatility_baseline(targets,scales,['a']*100+['b'])
    assert baseline[4,3,2]==1.5
    assert baseline[4,3,2]*2==3
    np.testing.assert_array_equal(volatility_scale(np.array([[0.,0.],[0.,.02]]),1,.0001),[.0001,.02])
    with pytest.raises(ValueError):
        volatility_scale(np.array([[0.,np.nan]]),1,.0001)
