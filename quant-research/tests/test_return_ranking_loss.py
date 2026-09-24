import itertools

import numpy as np
import pytest
import torch

from quant_research.return_ranking_loss import bag_contributions, bag_score_loss, reference_prices


def test_joint_pair_score_gradient_matches_exact_enumeration():
    # Two independent stock actions; target also depends on the selected exit
    # action. Compare exact joint expectation with the two-bag LOO estimator.
    theta = torch.tensor([.2, -.4], dtype=torch.float64, requires_grad=True)
    prob = theta.sigmoid()
    actions = list(itertools.product(range(2), repeat=2))
    costs, contributions, probabilities, logps = [], [], [], []
    for action in actions:
        x = np.array(action)
        pred = np.array([.03, .02])+x*.05
        actual = np.array([.08, .01])+x*np.array([-.01, .02])
        c, stats = bag_contributions(np.repeat(pred[:, None], 2, 1),
            np.repeat(actual[:, None], 2, 1), np.ones((2, 2), bool), np.ones(2, bool), ['d', 'd'])
        p = torch.where(torch.tensor(action).bool(), prob, 1-prob)
        costs.append(stats['objective'])
        contributions.append(c[:, 0])
        probabilities.append(p.prod())
        logps.append(p.log())
    exact = sum(p*c for p, c in zip(probabilities, costs))
    expected = torch.autograd.grad(exact, theta, retain_graph=True)[0]
    estimator = theta.sum()*0
    for first, second in itertools.product(range(4), repeat=2):
        estimator = estimator+(probabilities[first]*probabilities[second]).detach()*bag_score_loss(
            torch.stack([logps[first], logps[second]], 1),
            np.stack([contributions[first], contributions[second]], 1))
    observed = torch.autograd.grad(estimator, theta)[0]
    torch.testing.assert_close(observed, expected, atol=1e-12, rtol=1e-12)


def test_no_cross_date_pairs_or_missing_label_gradient():
    pred = np.array([[.1, .2], [.2, .1], [np.nan, np.nan]])
    contrib, stats = bag_contributions(pred, np.zeros((3, 2)), np.ones((3, 2), bool),
                                      [True, True, False], ['a', 'b', 'a'])
    assert stats['pairs'] == 0
    np.testing.assert_array_equal(contrib[2], 0)
    lp = torch.ones((3, 2), requires_grad=True)
    bag_score_loss(lp, contrib).backward()
    assert (lp.grad[2] == 0).all()


def test_invalid_bags_penalized_and_ties_carry_no_ranking_preference():
    c, stats = bag_contributions(np.zeros((2, 2)), np.zeros((2, 2)),
        np.ones((2, 2), bool), [True, True], ['a', 'a'])
    assert stats['pair_loss'] == 0
    invalid, bad = bag_contributions(np.zeros((2, 2)), np.zeros((2, 2)),
        [[True, False], [True, True]], [True, True], ['a', 'a'])
    assert bad['objective'] > stats['objective'] and invalid[0, 1] > c[0, 1]


def test_reference_uses_equal_models_and_selected_day_not_path_maximum():
    x = np.ones((2, 4, 5, 6))*10
    x[..., 1] = 12
    x[..., 2] = 8
    x[0, :, 1, 1] = 14
    x[1, :, 2, 1] = 20
    r = reference_prices(x, .0025, 2)
    assert r['sell_offset'] == 1  # Equal vote, earliest day.
    assert r['buy_reference'] == 8
    assert r['sell_reference'] == 13  # Mean of 14 and 12, NOT 14 and 20.
    assert r['predicted'] == 13/8-1-.0025
    x[1, :3, 0, 2] = -1
    assert reference_prices(x, minimum_paths=2) is None


def test_reference_price_scale_invariance_and_argument_validation():
    x = np.ones((1, 8, 5, 6))*10
    x[..., 1] = 12
    x[..., 2] = 8
    assert reference_prices(x, minimum_paths=4)['predicted'] == reference_prices(x*9, minimum_paths=4)['predicted']
    with pytest.raises(ValueError):
        bag_score_loss(torch.zeros(2, 1), np.zeros((2, 1)))


def test_sampled_objective_backpropagates_without_future_generation_inputs(monkeypatch):
    import quant_research.return_ranking_loss as module
    from quant_research.token_transformer import TokenConfig, TokenTransformer
    torch.manual_seed(19)
    model=TokenTransformer(TokenConfig(width=8,heads=2,layers=1,s1_bits=2,s2_bits=2))
    a,b=torch.randint(4,(2,65)),torch.randint(4,(2,65))
    stamps=torch.zeros(2,65,5,dtype=torch.long)
    observed=[]
    def generation(model,aa,bb,past,future,**kwargs):
        assert aa.shape==bb.shape==(2,60)
        assert past.shape==(2,60,5) and future.shape==(2,5,5)
        observed.append((aa.clone(),bb.clone()))
        rng=torch.Generator().manual_seed(kwargs['seed'])
        pairs=[]
        for x in [aa,bb]:
            tail=torch.randint(4,(2,kwargs['samples'],5),generator=rng)
            pairs.append(torch.cat([x[:,None].expand(-1,kwargs['samples'],-1),tail],-1))
        return tuple(pairs)
    def decoder(_,pairs,mean,scale,horizon):
        values=pairs[0][:,:,-5:].numpy()
        out=np.ones((*values.shape,6))*10
        out[...,1]=11+values/10
        out[...,2]=9+values/20
        return out,np.ones(values.shape[:2],bool)
    monkeypatch.setattr(module,'generate_tokens',generation)
    monkeypatch.setattr(module,'decode_paths',decoder)
    future=np.ones((2,5,6))*10
    future[:,:,1]=np.array([[11,12,13,14,15],[11,11,11,11,11]])
    future[:,:,2]=9
    loss,stats=module.sampled_return_ranking_loss(model,None,a,b,stamps,
        np.zeros((2,1,6)),np.ones((2,1,6)),future,np.ones((2,5),bool),['d','d'],seed=123)
    loss.backward()
    assert stats['pairs']==1 and model.training
    assert model.coarse_head.weight.grad.abs().sum()>0
    assert model.fine_head.weight.grad.abs().sum()>0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    module.sampled_return_ranking_loss(model,None,a,b,stamps,np.zeros((2,1,6)),np.ones((2,1,6)),
        future*2,np.ones((2,5),bool),['d','d'],seed=123)
    for first,second in zip(observed[0],observed[1]):
        torch.testing.assert_close(first,second,atol=0,rtol=0)
