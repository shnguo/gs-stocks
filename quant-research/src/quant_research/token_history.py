"""Chronological daily-window sampling and masked horizon-weighted losses."""
from __future__ import annotations

import hashlib

import numpy as np
import torch

from .token_transformer import forecast_auxiliary


def partition_indices(rows, dates, start, boundary, horizon=5):
    t = rows.date_index.to_numpy(int)
    mature = t + horizon < len(dates)
    ends = np.asarray(dates)[np.minimum(t+horizon, len(dates)-1)]
    mask = rows.date.ge(start).to_numpy() & rows.date.lt(boundary).to_numpy()
    return np.flatnonzero(mask & mature & (ends < boundary))


def future_labels(quotes, priced, sequence, label_sequence, actions, stock, time, horizon=5):
    offsets = time[:,None]+np.arange(horizon+1)
    s = stock[:,None]
    q = quotes[s,offsets]
    valid = priced[s,offsets].copy()
    valid &= np.isclose(q[...,6],q[:,:1,6],rtol=1e-8,atol=0)
    for seq in [sequence,label_sequence]:
        codes = seq[s,offsets]
        valid &= (codes == codes[:,:1]) & (codes >= 0)
    valid[:,1:] &= ~actions[s,offsets][:,1:]
    return q[:,1:,:6].copy(), np.logical_and.accumulate(valid,axis=1)[:,1:]


def stock_order(ids, symbols, day, seed=17):
    return sorted(ids, key=lambda i: hashlib.sha256(f'{seed}:{day}:{symbols[i]}'.encode()).digest())


def stratified_pool(ids, symbols, day, count, seed=17):
    """Outcome-independent exchange round-robin ordering with no invented membership."""
    ordered = stock_order(ids, symbols, day, seed)
    if any(len(symbols[i].split('.')) != 3 or symbols[i].split('.')[1] not in ['xshg', 'xshe', 'xbse'] for i in ordered):
        raise ValueError('Unrecognized canonical instrument identifier')
    groups = [[i for i in ordered if symbols[i].split('.')[1] == ex]
              for ex in ['xshg', 'xshe', 'xbse']]
    result = []
    for level in range(max(map(len, groups), default=0)):
        for group in groups:
            if level < len(group):
                result.append(group[level])
        if len(result) >= count:
            break
    return np.asarray(result[:count], dtype=np.int64)


def epoch_sample(groups, epoch, per_date, seed):
    """Every date contributes equally; cycle through each known-label stock pool."""
    chosen = []
    for day, ids in sorted(groups.items()):
        row_seed = int(hashlib.sha256(f'{seed}:{day}'.encode()).hexdigest()[:8], 16)
        order = np.random.default_rng(row_seed).permutation(ids)
        count = min(per_date, len(order))
        positions = (np.arange(count)+(epoch-1)*per_date) % len(order)
        chosen.extend(order[positions])
    return np.random.default_rng(seed+epoch).permutation(np.asarray(chosen, np.int64))


def date_weights(rows, ids):
    counts = rows.iloc[ids].date.value_counts()
    weights = rows.iloc[ids].date.map(1/counts).to_numpy(np.float32)
    return weights * len(weights)/weights.sum()


def reduce_horizon_losses(errors, valid, horizon_weights=None):
    """Per-row normalized loss; unknown suffix carries neither loss nor weight."""
    if errors.shape != valid.shape or valid.dtype != torch.bool:
        raise ValueError('Forecast loss/mask shape mismatch')
    if (valid[:, 1:] & ~valid[:, :-1]).any():
        raise ValueError('Invalid known prefix')
    if horizon_weights is None:
        weights = torch.ones(errors.shape[1], device=errors.device, dtype=errors.dtype)
    else:
        weights = torch.as_tensor(horizon_weights, device=errors.device, dtype=errors.dtype)
        if (weights.shape != (errors.shape[1],) or not torch.isfinite(weights).all()
                or (weights < 0).any() or not bool(weights.sum() > 0)):
            raise ValueError('Invalid horizon weights')
    effective = valid.to(errors.dtype)*weights
    denominator = effective.sum(1)
    return (torch.where(valid, errors, 0)*effective).sum(1)/denominator.clamp_min(1e-12)


def loss_parts(model, s1, s2, stamps, valid, lookback, horizon_weights=None, *, history_auxiliary=None):
    if s1.shape[1] != lookback+valid.shape[1]:
        raise ValueError('Target length differs')
    coarse, fine = model.forecast_logits(s1[:,:-1],s2[:,:-1],stamps[:,:-1],s1[:,1:],lookback-1,
        auxiliary=forecast_auxiliary(history_auxiliary, lookback, s1.shape[1]-1))
    parts = []
    for logits, target in [(coarse,s1[:,lookback:]),(fine,s2[:,lookback:])]:
        error = torch.nn.functional.cross_entropy(logits.transpose(1,2),target,reduction='none')
        parts.append(reduce_horizon_losses(error,valid,horizon_weights))
    return torch.stack(parts,-1)
