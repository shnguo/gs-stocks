"""Calibration audit must respect neural storage precision without hiding errors."""

import subprocess
import sys
from pathlib import Path


def test_float32_rounding_passes_but_changed_forecast_fails():
    code = r"""
import sys
import numpy as np
sys.path.insert(0,sys.argv[1])
from verify_plan_prequential import calibrator_check
from quant_research.plan_value import HEADS,fit_calibration,apply_calibration,head_targets
n=16
dates=np.repeat(['a','b','c','d'],4)
net=np.repeat(np.linspace(-.02,.04,n)[:,None],12,axis=1)
filled=np.ones((n,12));filled[0]=np.nan;net[0]=np.nan
outcomes=dict(filled=filled,conditional_net_return=net,
 conditional_loss=np.where(np.isfinite(net),net<0,np.nan),
 conditional_downside=np.where(np.isfinite(net),np.maximum(-net,0),np.nan))
raw={h:np.repeat(np.linspace(.1,.9,n,dtype=np.float32)[:,None],12,axis=1) for h in HEADS}
for h in [HEADS[2],HEADS[4]]:raw[h]*=.03
states=fit_calibration(raw,outcomes,dates)
corrected=apply_calibration(raw,states)
assert calibrator_check(raw,head_targets(outcomes),dates,states,raw,corrected)==60
bad={h:v.copy() for h,v in corrected.items()};bad[HEADS[2]][0,0]+=.001
try:calibrator_check(raw,head_targets(outcomes),dates,states,raw,bad)
except AssertionError:pass
else:raise AssertionError('Materially wrong float32 forecast accepted')
"""
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    subprocess.run([sys.executable, "-c", code, str(scripts)], check=True, timeout=60)
