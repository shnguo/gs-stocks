"""Score-function gradients for actual sampled midpoint distributions.

The frozen discrete sampler/decoder stays on its real inference path. Only the
sampled sequence log-probabilities are differentiated; no soft-code decoding.
"""
import numpy as np
import torch
from torch.nn import functional as F

from .token_transformer import decode_paths, forecast_auxiliary, generate_tokens


def midpoint_rewards(values, actual):
    """Unbiased population-CRPS score-function reward with an LOO baseline.

values: [row, independent draw, horizon]; actual: [row, horizon].
The leave-one-out absolute-error baseline does not depend on its own draw.
"""
    if values.ndim!=3 or actual.shape!=(values.shape[0],values.shape[2]) or values.shape[1]<2:
        raise ValueError('At least two independent draws and matching targets required')
    if not torch.isfinite(values).all() or not torch.isfinite(actual).all():
        raise ValueError('Finite midpoint draws and targets required')
    k=values.shape[1]
    absolute=(values-actual[:,None]).abs()
    distance=(values[:,:,None]-values[:,None,:]).abs().sum(2)/(k-1)
    baseline=(absolute.sum(1,keepdim=True)-absolute)/(k-1)
    reward=(absolute-distance-baseline).detach()
    crps=absolute.mean(1)-distance.mean(1)/2
    return reward,crps.detach()


def score_function_loss(log_probabilities, rewards, known, weights=None):
    if log_probabilities.shape!=rewards.shape or known.shape!=(rewards.shape[0],rewards.shape[2]):
        raise ValueError('Loss and validity shapes differ')
    if known.dtype!=torch.bool or not known.any():
        raise ValueError('Known midpoint horizons required')
    w=known.to(log_probabilities.dtype)
    per_row=((log_probabilities*rewards.detach()).mean(1)*w).sum(1)/w.sum(1).clamp_min(1)
    row_w=known.any(1).to(per_row.dtype)
    if weights is not None:
        if weights.shape!=row_w.shape or not torch.isfinite(weights).all() or (weights<0).any():
            raise ValueError('Invalid row weights')
        row_w=row_w*weights
    return (per_row*row_w).sum()/row_w.sum().clamp_min(1e-12)


def sampled_midpoint_loss(model,decoder,a,b,stamps,mean,scale,future,valid,reference,*,seed,samples=4,weights=None,history_auxiliary=None):
    """Free-running generation sees history/calendar only; labels enter rewards only."""
    pairs=generate_tokens(model,a[:,:60],b[:,:60],stamps[:,:60],stamps[:,60:],
        samples=samples,seed=seed,temperature=1.,top_p=1.,top_k=0,history_auxiliary=history_auxiliary)
    paths,legal=decode_paths(decoder,pairs,mean,scale,5)
    ref=np.asarray(reference,dtype=float)
    future=np.asarray(future,dtype=float)
    values,actual,known=[],[],[]
    for h in [2,5]:
        hi=paths[:,:,:h,1].astype(float).max(-1)
        lo=paths[:,:,:h,2].astype(float).min(-1)
        values.append(((hi+lo)/(2*ref[:,None])-1)*100)
        ok=np.asarray(valid)[:,:h].all(1)
        target=((future[:,:h,1].max(1)+future[:,:h,2].min(1))/(2*ref)-1)*100
        actual.append(np.where(ok,target,0.))
        known.append(ok)
    device=next(model.parameters()).device
    values=torch.as_tensor(np.stack(values,-1),device=device,dtype=torch.float32)
    actual=torch.as_tensor(np.stack(actual,-1),device=device,dtype=torch.float32)
    known=torch.as_tensor(np.stack(known,-1),device=device,dtype=torch.bool)
    rewards,crps=midpoint_rewards(values,actual)
    # Clone inference tensors before they are saved for the differentiable embedding pass.
    s1,s2=[x.reshape(-1,x.shape[-1]).clone() for x in pairs]
    repeated=stamps.repeat_interleave(samples,0)
    training=model.training
    model.eval()
    try:
        coarse,fine=model.forecast_logits(s1[:,:-1],s2[:,:-1],repeated[:,:-1],s1[:,1:],59,
            auxiliary=forecast_auxiliary(None if history_auxiliary is None else history_auxiliary.repeat_interleave(samples,0),60,s1.shape[1]-1))
        logp=F.log_softmax(coarse,-1).gather(-1,s1[:,60:,None]).squeeze(-1)
        logp=logp+F.log_softmax(fine,-1).gather(-1,s2[:,60:,None]).squeeze(-1)
        logp=logp.reshape(len(a),samples,5)
        prefix=torch.stack([logp[:,:,:2].sum(-1),logp.sum(-1)],-1)
        loss=score_function_loss(prefix,rewards,known,weights)
    finally:
        model.train(training)
    return loss,dict(crps=float((crps*known).sum()/known.sum()),legal_fraction=float(legal.mean()),
        reward_abs=float(rewards.abs().mean()),samples=samples,rows=len(a))
