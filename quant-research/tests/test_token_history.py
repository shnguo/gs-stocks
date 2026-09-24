import numpy as np
import pandas as pd
import pytest
import torch

from quant_research.token_history import (
    date_weights,
    epoch_sample,
    partition_indices,
    reduce_horizon_losses,
    stratified_pool,
)


def test_partition_purges_future_labels_not_historical_context():
    dates=pd.bdate_range('2024-01-01',periods=20).strftime('%Y-%m-%d').tolist()
    rows=pd.DataFrame({'date':dates,'date_index':np.arange(20)})
    ids=partition_indices(rows,dates,dates[0],dates[10],5)
    np.testing.assert_array_equal(ids,np.arange(5))
    assert all(dates[i+5]<dates[10] for i in ids)


def test_balanced_epoch_visits_all_dates_and_rotates_stock_pool():
    groups={'a':np.arange(192),'b':np.arange(192,384)}
    orders=[epoch_sample(groups,e,32,17) for e in range(1,7)]
    assert all(len(x)==64 and (x<192).sum()==32 for x in orders)
    np.testing.assert_array_equal(np.unique(np.concatenate(orders)),np.arange(384))
    np.testing.assert_array_equal(orders[0],epoch_sample(groups,1,32,17))


def test_date_weights_prevent_large_cross_section_from_dominating():
    rows=pd.DataFrame({'date':['a','a','a','b']})
    w=date_weights(rows,np.arange(4))
    assert w[:3].sum()==pytest.approx(w[3])


def test_stratified_pool_has_no_outcome_argument_and_covers_exchanges():
    symbols=['cn.xshg.a','cn.xshe.b','cn.xbse.c','cn.xshg.d','cn.xshe.e','cn.xbse.f']
    chosen=stratified_pool(np.arange(6),symbols,'2024-01-01',3)
    assert {symbols[i].split('.')[1] for i in chosen}=={'xshg','xshe','xbse'}


def test_equal_horizon_weights_match_original_and_mask_gradient():
    error=torch.tensor([[1.,2.,3.,4.,5.],[2.,4.,100.,100.,100.]],requires_grad=True)
    valid=torch.tensor([[1,1,1,1,1],[1,1,0,0,0]],dtype=torch.bool)
    loss=reduce_horizon_losses(error,valid)
    torch.testing.assert_close(loss,torch.tensor([3.,3.]))
    torch.testing.assert_close(loss,reduce_horizon_losses(error,valid,[1]*5))
    weighted=reduce_horizon_losses(error,valid,[1,.8,.6,.4,.2])
    torch.testing.assert_close(weighted,torch.tensor([7/3,5.2/1.8]))
    weighted.sum().backward()
    torch.testing.assert_close(error.grad[1,2:],torch.zeros(3))
    assert error.grad[0,0] == pytest.approx(5*error.grad[0,4].item())


def test_invalid_or_zero_horizon_weights_rejected():
    v=torch.ones((1,5),dtype=torch.bool)
    e=torch.ones((1,5))
    for w in [[0]*5,[1,-1,1,1,1],[1,float('nan'),1,1,1],[1,1]]:
        with pytest.raises(ValueError):
            reduce_horizon_losses(e,v,w)


def test_new_equal_loss_matches_existing_heads_and_all_parameter_gradients():
    from quant_research.kronos_ranker import timestamps
    from quant_research.token_history import loss_parts
    from quant_research.token_transformer import TokenConfig, TokenTransformer, token_loss
    torch.manual_seed(17)
    model=TokenTransformer(TokenConfig(width=16,layers=1,heads=4,s1_bits=3,s2_bits=3,max_context=16,dropout=0)).double()
    a,b=torch.randint(0,8,(3,9)),torch.randint(0,8,(3,9))
    stamps=torch.from_numpy(timestamps(pd.bdate_range('2025-01-01',periods=9)))[None].expand(3,-1,-1)
    valid=torch.tensor([[1,1,1,1,1],[1,1,0,0,0],[0,0,0,0,0]],dtype=torch.bool)
    old,parts=token_loss(model,a,b,stamps,valid,4)
    old.backward()
    grads={k:p.grad.clone() for k,p in model.named_parameters()}
    model.zero_grad()
    new=loss_parts(model,a,b,stamps,valid,4,[1]*5)
    torch.testing.assert_close(new,parts,rtol=0,atol=0)
    new[:2].mean().backward()
    for k,p in model.named_parameters():
        torch.testing.assert_close(p.grad,grads[k],rtol=1e-10,atol=1e-12)


def test_future_label_barriers_censor_entire_suffix():
    from quant_research.token_history import future_labels
    q=np.ones((5,6,7))
    q[...,:4]=10
    q[...,1]=11
    q[...,2]=9
    priced=np.ones((5,6),bool)
    seq=np.zeros((5,6),int)
    label=seq.copy()
    action=np.zeros((5,6),bool)
    q[0,3,6]=2
    priced[1,2]=False
    seq[2,4:]=1
    label[3,1:]=1
    action[4,5]=True
    _,valid=future_labels(q,priced,seq,label,action,np.arange(5),np.zeros(5,int))
    np.testing.assert_array_equal(valid,[[1,1,0,0,0],[1,0,0,0,0],[1,1,1,0,0],[0,0,0,0,0],[1,1,1,1,0]])
