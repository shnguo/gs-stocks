"""Isolated neural context regression and checkpoint compatibility checks."""

import subprocess
import sys


def test_signal_context_masks_training_scaler_and_reload(tmp_path):
    code = r"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from quant_research.plan_transformer import (PlanTransformer, fit_context_scaler,
    normalize_context, target_arrays, train, predict)
torch.set_num_threads(1)
rng=np.random.default_rng(19)
x=np.array([[1,np.nan,5],[3,np.nan,5]],np.float32)
m,s,n=fit_context_scaler(x)
np.testing.assert_array_equal(m,[2,0,5]);np.testing.assert_array_equal(n,[2,0,2])
v=normalize_context(np.array([[np.nan,999,5]],np.float32),m,s)
assert np.isfinite(v).all() and v[0,0]==0 and v[0,3]==1
np.testing.assert_array_equal(m,[2,0,5])
try:normalize_context(np.array([[np.inf,1,1]],np.float32),m,s)
except ValueError:pass
else:raise AssertionError('Infinite values accepted')
nrows=8
net=np.repeat(np.linspace(-.03,.04,nrows)[:,None],12,axis=1);net[0]=np.nan
outcomes=dict(filled=np.ones((nrows,12)),conditional_net_return=net,
 conditional_loss=np.where(np.isfinite(net),net<0,np.nan),
 conditional_downside=np.where(np.isfinite(net),np.maximum(-net,0),np.nan))
rows=pd.DataFrame(dict(stock_index=np.arange(nrows)%2,date_index=np.full(nrows,60),date=['a']*4+['b']*4))
_,_,center,scale,prior=target_arrays(outcomes,rows.date)
torch.manual_seed(17);baseline=PlanTransformer(center,scale,prior,width=8)
torch.manual_seed(17);enhanced=PlanTransformer(center,scale,prior,width=8,context_features=3)
for k,v in baseline.state_dict().items():torch.testing.assert_close(v,enhanced.state_dict()[k],rtol=0,atol=0)
# The base architecture still loads its exact old-shaped checkpoint.
restored=PlanTransformer(center,scale,prior,width=8);restored.load_state_dict(baseline.state_dict(),strict=True)
values=rng.normal(size=(2,70,27)).astype(np.float32)
context=rng.normal(size=(nrows,3)).astype(np.float32);context[0,:]=np.nan
selection_context=context.copy();selection_context[1,0]=10000
prices=rng.normal(0,.02,(nrows,5,4)).astype(np.float32)
config=dict(seed=17,device='cpu',width=8,context_features=3,learning_rate=.001,
 batch_size=4,max_epochs=2,min_epochs=2,patience=1,min_delta=1e9,price_auxiliary_weight=.1)
out=Path(sys.argv[1])
model,mean,std=train(values,rows,{'targets':prices},outcomes,rows,{'targets':prices},outcomes,
 out,config,train_context=context,select_context=selection_context)
saved=np.load(out/'scalers.npz')
em,es,en=fit_context_scaler(context)
np.testing.assert_array_equal(saved['context_mean'],em)
np.testing.assert_array_equal(saved['context_scale'],es)
np.testing.assert_array_equal(saved['context_count'],en)
state=torch.load(out/'transformer.pt',weights_only=True)
reloaded=PlanTransformer(center,scale,prior,width=8,context_features=3)
reloaded.load_state_dict(state['state_dict'])
a,_=predict(model,values,rows,mean,std,context=context,context_mean=em,context_scale=es)
b,_=predict(reloaded,values,rows,mean,std,context=context,context_mean=em,context_scale=es)
for h in a:np.testing.assert_array_equal(a[h],b[h])
assert np.isfinite(np.stack(list(a.values()))).all()
try:predict(model,values,rows,mean,std)
except ValueError:pass
else:raise AssertionError('Missing context accepted')
"""
    subprocess.run([sys.executable, "-c", code, str(tmp_path)], check=True, timeout=90)
