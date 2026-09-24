"""Diagnostics must preserve optimization; configuration must affect real modules."""

import subprocess
import sys


def test_diagnostics_preserve_training_and_report_components(tmp_path):
    code = r"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from quant_research.plan_transformer import PlanTransformer, target_arrays, train, plan_loss, plan_loss_components
torch.set_num_threads(1)
rng=np.random.default_rng(44); n=8
net=np.broadcast_to(np.linspace(-.02,.03,n)[:,None],(n,12)).copy();net[0]=np.nan
outcomes=dict(filled=np.ones((n,12)),conditional_net_return=net,
 conditional_loss=np.where(np.isfinite(net),net<0,np.nan),
 conditional_downside=np.where(np.isfinite(net),np.maximum(-net,0),np.nan))
rows=pd.DataFrame(dict(stock_index=np.arange(n)%2,date_index=np.full(n,60),date=['a']*4+['b']*4))
y,w,center,scale,prior=target_arrays(outcomes,rows.date)
z=torch.tensor(rng.normal(size=y.shape),dtype=torch.float32,requires_grad=True)
loss=plan_loss(z,torch.from_numpy(y),torch.from_numpy(w))
parts=plan_loss_components(z,torch.from_numpy(y),torch.from_numpy(w))
torch.testing.assert_close(loss,parts.mean())
loss.backward();assert (z.grad[torch.from_numpy(w==0)]==0).all()
model=PlanTransformer(center,scale,prior,width=8,context_features=3,dropout=.3)
for layer in model.layers:
 assert layer.self_attn.dropout==.3
 assert all(m.p==.3 for m in layer.modules() if isinstance(m,torch.nn.Dropout))
values=rng.normal(size=(2,70,27)).astype(np.float32)
context=rng.normal(size=(n,3)).astype(np.float32)
prices=rng.normal(0,.02,(n,5,4)).astype(np.float32)
cfg=dict(seed=17,device='cpu',width=8,context_features=3,learning_rate=.001,
 weight_decay=.1,dropout=.3,batch_size=4,max_epochs=2,min_epochs=2,patience=1,
 min_delta=1e9,price_auxiliary_weight=0.)
states=[]
for diagnostics in [False,True]:
 out=Path(sys.argv[1])/str(diagnostics);out.mkdir()
 trained,_,_=train(values,rows,{'targets':prices},outcomes,rows,{'targets':prices},outcomes,
  out,dict(cfg,diagnostics=diagnostics),train_context=context,select_context=context)
 states.append({k:v.clone() for k,v in trained.state_dict().items()})
for k in states[0]:torch.testing.assert_close(states[0][k],states[1][k],rtol=0,atol=0)
out=Path(sys.argv[1])/'True'
initial=json.loads((out/'initial-diagnostics.json').read_text())
assert initial['epoch']==0 and not initial['checkpoint_eligible'] and initial['probe_ids']==list(range(n))
log=json.loads((out/'training-log.json').read_text())
for row in log:
 assert len(row['training_head_losses'])==5
 np.testing.assert_allclose(row['training_total_objective'],np.mean(list(row['training_head_losses'].values())),rtol=1e-6)
 assert row['training_probe_net_mse']>=0
 np.testing.assert_allclose(row['selection_net_mse'],np.mean(row['selection_plan_net_mse']),rtol=1e-6)
last=torch.load(out/'last-checkpoint.pt',weights_only=False)
assert last['optimizer']['param_groups'][0]['weight_decay']==.1
"""
    subprocess.run([sys.executable, "-c", code, str(tmp_path)], check=True, timeout=90)


def test_selection_uses_only_predeclared_challengers():
    code = r"""
import sys
sys.path.insert(0,'scripts')
from transformer_optimization import choose_challenger
assert choose_challenger(dict(reference=.001, lower_lr=.003, regularized=.004, no_price_aux=.002))=='no_price_aux'
try:choose_challenger(dict(reference=1, lower_lr=0))
except ValueError:pass
else:raise AssertionError('Incomplete screen accepted')
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)
