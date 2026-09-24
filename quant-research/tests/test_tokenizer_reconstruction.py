import numpy as np
import pytest
import torch

from quant_research.tokenizer_reconstruction import prefix_mean, reconstruction_loss

CFG=dict(normalized_huber_beta=1.,extrema_huber_beta_pp=1.,extrema_weight=.25,ohlc_order_weight=.25)


def fixture():
    y=torch.tensor([[[10.,11.,9.,10.,100.,1000.]]*5])
    mean=torch.zeros(1,1,6)
    scale=torch.ones(1,1,6)
    return y,mean,scale,torch.tensor([10.])


def test_exact_legal_reconstruction_has_zero_loss():
    y,m,s,r=fixture()
    for objective in ['reconstruction','price_consistency']:
        got=reconstruction_loss(y,y,torch.ones(1,5,dtype=torch.bool),m,s,r,objective,CFG)
        assert got.item()==0


def test_unknown_suffix_cannot_change_loss_or_receive_gradient():
    y,m,s,r=fixture()
    known=torch.tensor([[True,True,False,False,False]])
    pred=(y+.3).requires_grad_()
    a=reconstruction_loss(pred,y,known,m,s,r,'price_consistency',CFG)
    target=y.clone()
    target[:,2:]=float('nan')
    changed=pred.detach().clone()
    changed[:,2:]=1e20
    changed.requires_grad_()
    b=reconstruction_loss(changed,target,known,m,s,r,'price_consistency',CFG)
    torch.testing.assert_close(a,b)
    b.sum().backward()
    assert torch.isfinite(changed.grad).all()
    assert torch.count_nonzero(changed.grad[:,2:])==0


def test_price_loss_detects_and_pushes_down_an_invalid_low():
    y,m,s,r=fixture()
    p=y.clone()
    p[...,2]=12.
    p.requires_grad_()
    known=torch.ones(1,5,dtype=torch.bool)
    base=reconstruction_loss(p,y,known,m,s,r,'reconstruction',CFG)
    shaped=reconstruction_loss(p,y,known,m,s,r,'price_consistency',CFG)
    assert shaped.item()>base.item()
    shaped.sum().backward()
    assert (p.grad[...,2]>0).all()


def test_nonprefix_and_empty_rows_are_rejected():
    for known in [[False]*5,[True,False,True,False,False]]:
        with pytest.raises(ValueError):
            prefix_mean(torch.ones(1,5),torch.tensor([known]))


def test_extrema_objective_is_invariant_to_common_price_units():
    y,m,s,r=fixture()
    p=y+.2
    known=torch.ones(1,5,dtype=torch.bool)
    a=reconstruction_loss(p,y,known,m,s,r,'price_consistency',CFG)
    factor=torch.tensor([10.,10.,10.,10.,1.,1.])[None,None]
    b=reconstruction_loss(p,y*factor,known,m*factor,s*factor,r*10,'price_consistency',CFG)
    np.testing.assert_allclose(a.detach(),b.detach(),rtol=1e-5)


def test_real_decoder_update_preserves_tokens_and_causality():
    import importlib.util
    from pathlib import Path

    from quant_research.token_transformer import load_tokenizer
    from quant_research.tokenizer_reconstruction import enable_decoder_training

    if not importlib.util.find_spec('safetensors'):
        pytest.skip('Requires pinned tokenizer runtime')
    torch.set_num_threads(4)
    bundle=Path(__file__).parents[1]/'artifacts/kronos-comparison-20260910-v1'
    tokenizer=load_tokenizer(bundle,'cpu')
    params=enable_decoder_training(tokenizer)
    frozen={k:v.detach().clone() for k,v in tokenizer.named_parameters() if not v.requires_grad}
    before_head=tokenizer.head.weight.detach().clone()
    torch.manual_seed(3)
    raw=torch.randn(2,65,6)
    with torch.no_grad():
        codes=tokenizer.encode(raw,half=True)
        prefix=tokenizer.encode(raw[:,:60],half=True)
    for a,b in zip(codes,prefix):
        torch.testing.assert_close(a[:,:60],b,atol=0,rtol=0)
    optimizer=torch.optim.AdamW(params,lr=1e-4)
    y,m,s,r=fixture()
    decoded=tokenizer.decode(codes,half=True)[:,-5:]
    loss=reconstruction_loss(decoded,y.expand(2,-1,-1),torch.ones(2,5,dtype=torch.bool),
        m.expand(2,-1,-1),s.expand(2,-1,-1),r.expand(2),'price_consistency',CFG).mean()
    loss.backward()
    optimizer.step()
    assert not torch.equal(before_head,tokenizer.head.weight)
    for name,p in tokenizer.named_parameters():
        if name in frozen:
            torch.testing.assert_close(p,frozen[name],rtol=0,atol=0)
            assert p.grad is None
    with torch.no_grad():
        encoded=tokenizer.encode(raw,half=True)
        for a,b in zip(codes,encoded):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        full=tokenizer.decode(codes,half=True)
        earlier=tokenizer.decode([c[:,:60] for c in codes],half=True)
        torch.testing.assert_close(full[:,:60],earlier,rtol=1e-5,atol=1e-5)
