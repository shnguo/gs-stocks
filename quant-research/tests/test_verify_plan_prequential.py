import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from verify_plan_prequential import calibrator_check, independent_choice, verify_decision_assembly

from quant_research.plan_value import HEADS, apply_calibration, fit_calibration, head_targets


def test_independent_calibration_refit_detects_changed_offset():
    rng=np.random.default_rng(41)
    dates=np.repeat(['2024-01-01','2024-01-08','2024-01-15','2024-01-22'],6)
    filled=rng.integers(0,2,size=(24,12)).astype(float)
    net=np.where(filled==1,rng.normal(0,.03,(24,12)),np.nan)
    outcome=dict(filled=filled,conditional_net_return=net,
        conditional_loss=np.where(np.isfinite(net),(net<0).astype(float),np.nan),
        conditional_downside=np.where(np.isfinite(net),np.maximum(-net,0),np.nan))
    prediction={h:rng.uniform(.2,.8,(24,12)) if h in [HEADS[0],HEADS[1],HEADS[3]] else rng.normal(0,.01,(24,12)) for h in HEADS}
    raw={h:v[:3].copy() for h,v in prediction.items()}
    state=fit_calibration(prediction,outcome,dates)
    corrected=apply_calibration(raw,state)
    assert calibrator_check(prediction,head_targets(outcome),dates,state,raw,corrected)==60
    state[HEADS[2]][0]['offset']+=.01
    with pytest.raises(AssertionError):
        calibrator_check(prediction,head_targets(outcome),dates,state,raw,corrected)


def test_choice_reconstruction_retains_abstention():
    pred={h:np.full((2,12),v) for h,v in zip(HEADS,[.5,.95,.02,.4,.01])}
    pred[HEADS[2]][0,3]=.1
    pred[HEADS[1]][1]=.8
    assert independent_choice(pred).tolist()==[3,-1]


def test_decision_assembly_aligns_keys_but_rejects_changed_values():
    frame = pd.DataFrame(dict(window=['fold-01']*2, model=['rolling', 'rolling_raw'],
        date=['2024-01-01']*2, selected=[3, 5], return_unknown=[1, 2]))
    combined = frame.iloc[::-1].reset_index(drop=True)
    verify_decision_assembly([frame], combined)
    combined.loc[0, 'return_unknown'] = 0
    with pytest.raises(AssertionError):
        verify_decision_assembly([frame], combined)


def test_decision_assembly_rejects_duplicate_or_missing_keys():
    frame = pd.DataFrame(dict(window=['fold-01']*2, model=['rolling', 'rolling_raw'],
        date=['2024-01-01']*2, selected=[3, 5]))
    for changed in [frame.iloc[:1], pd.concat([frame.iloc[:1]]*2, ignore_index=True)]:
        with pytest.raises(AssertionError):
            verify_decision_assembly([frame], changed)
