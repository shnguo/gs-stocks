import itertools

import pytest
import torch

from quant_research.midpoint_loss import midpoint_rewards, score_function_loss


def test_score_function_matches_exact_population_crps_gradient():
    theta=torch.tensor(.3,dtype=torch.float64,requires_grad=True)
    probs=torch.stack([1-theta.sigmoid(),theta.sigmoid()])
    support=torch.tensor([-1.,3.],dtype=torch.float64)
    target=torch.tensor([[.5]],dtype=torch.float64)
    exact=(probs*(support-.5).abs()).sum()-.5*(probs[:,None]*probs[None,:]*(support[:,None]-support[None,:]).abs()).sum()
    expected=torch.autograd.grad(exact,theta,retain_graph=True)[0]
    estimate=theta*0
    for indices in itertools.product(range(2),repeat=2):
        ix=torch.tensor(indices)
        values=support[ix][None,:,None]
        reward,_=midpoint_rewards(values,target)
        logp=probs[ix].log()[None,:,None]
        term=score_function_loss(logp,reward,torch.ones((1,1),dtype=torch.bool))
        estimate=estimate+probs[ix].prod().detach()*term
    found=torch.autograd.grad(estimate,theta)[0]
    torch.testing.assert_close(found,expected,atol=1e-12,rtol=1e-12)


def test_draw_permutation_preserves_objective_and_rewards():
    x=torch.tensor([[[-1.,0.],[2.,4.],[3.,5.]]])
    y=torch.tensor([[1.,2.]])
    rewards,score=midpoint_rewards(x,y)
    order=torch.tensor([2,0,1])
    shuffled,other=midpoint_rewards(x[:,order],y)
    torch.testing.assert_close(shuffled,rewards[:,order])
    torch.testing.assert_close(other,score)
    assert not rewards.requires_grad


def test_unknown_horizon_carries_no_gradient():
    logp=torch.ones((2,3,2),requires_grad=True)
    rewards=torch.tensor([[[1.,20.],[2.,30.],[3.,40.]],[[9.,9.],[9.,9.],[9.,9.]]])
    loss=score_function_loss(logp,rewards,torch.tensor([[True,False],[False,False]]))
    loss.backward()
    assert (logp.grad[0,:,0]!=0).all()
    assert (logp.grad[:,:,1]==0).all() and (logp.grad[1]==0).all()


def test_loss_checks_sampling_and_validity():
    with pytest.raises(ValueError,match='At least two'):
        midpoint_rewards(torch.ones(1,1,2),torch.ones(1,2))
    with pytest.raises(ValueError,match='Known'):
        score_function_loss(torch.ones(1,2,2),torch.ones(1,2,2),torch.zeros(1,2,dtype=torch.bool))
