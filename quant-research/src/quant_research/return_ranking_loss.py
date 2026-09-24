"""Train the published reference-price functional through discrete token samples.

Independent bags use actual decoded paths, medians and the modal peak day.
Score-function gradients differentiate bag log probabilities, never medians or
argmax. The auxiliary objective is a finite-bag surrogate (8 paths during
training versus 64 during reporting), not a differentiable price head.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from .token_transformer import decode_paths, forecast_auxiliary, generate_tokens, valid_bars


def reference_prices(models, cost=.0025, minimum_paths=16):
    """Equal-model medians/frequencies, identical to daily reference publication.

    models: [model, draw, five days, six fields]. Return None if ANY model
    lacks enough legal paths. Ties in peak-day votes select the earliest day.
    """
    models = np.asarray(models, dtype=float)
    if models.ndim != 4 or models.shape[2:] != (5, 6) or minimum_paths < 1 or cost < 0:
        raise ValueError('Expected models of five-day OHLCVA paths and valid cost')
    buys, sells, votes, counts = [], [], [], []
    for paths in models:
        x = paths[valid_bars(paths).all(-1)]
        if len(x) < minimum_paths:
            return None
        buys.append(np.median(x[:, 0, 2]))
        sells.append(np.median(x[:, 1:, 1], axis=0))
        votes.append(np.bincount(x[:, 1:, 1].argmax(1), minlength=4)/len(x))
        counts.append(len(x))
    frequency = np.mean(votes, axis=0)
    selected = int(frequency.argmax())
    buy, sell = float(np.mean(buys)), float(np.mean(sells, axis=0)[selected])
    return dict(buy_reference=buy, sell_reference=sell, sell_offset=selected+1,
                predicted=sell/buy-1-cost, sell_date_frequency=float(frequency[selected]),
                legal_paths=counts)


def bag_contributions(predicted, actual, usable, known, dates, *, scale_pp=5.,
                      return_weight=.01, ranking_weight=.01, invalid_penalty=10.):
    """Detached cost attributable to each row/bag for an unbiased score estimator.

    Pair losses appear in BOTH participating rows' contributions because each
    row's sampled action affects the pair. The reported objective counts each
    pair once. Pairs only compare stocks sharing a signal date. Actual returns
    use each bag's selected exit date, frozen before observing future prices.
    """
    predicted, actual = np.asarray(predicted, float), np.asarray(actual, float)
    usable, known = np.asarray(usable, bool), np.asarray(known, bool)
    dates = np.asarray(dates)
    if (predicted.ndim != 2 or predicted.shape != actual.shape or usable.shape != predicted.shape
            or known.shape != (len(predicted),) or dates.shape != known.shape
            or predicted.shape[1] < 2 or not known.any() or scale_pp <= 0
            or min(return_weight, ranking_weight, invalid_penalty) < 0):
        raise ValueError('Invalid return/ranking bag inputs')
    if not np.isfinite(predicted[usable & known[:, None]]).all() or not np.isfinite(actual[usable & known[:, None]]).all():
        raise ValueError('Finite observed bag returns required')
    # Missing labels are not replaced with zero-return supervision.
    okay = usable & known[:, None]
    residual = np.where(okay, (predicted-actual)*100/scale_pp, 0.)
    absolute = np.abs(residual)
    huber = np.where(absolute <= 1, .5*residual**2, absolute-.5)
    huber = np.where(okay, huber, invalid_penalty)*known[:, None]
    contributions = return_weight*huber/known.sum()
    pairs = [(i, j) for i in range(len(known)) for j in range(i+1, len(known))
             if known[i] and known[j] and dates[i] == dates[j]]
    pair_costs = []
    for i, j in pairs:
        valid = okay[i] & okay[j]
        gap = np.where(valid, (actual[i]-actual[j])*100/scale_pp, 0.)
        delta = np.where(valid, (predicted[i]-predicted[j])*100/scale_pp, 0.)
        # Tied outcomes do not carry an arbitrary preference; large gaps have
        # at most unit weight. softplus is stable for extreme decoded returns.
        loss = np.logaddexp(0., -np.sign(gap)*delta)*np.minimum(np.abs(gap), 1.)
        loss = np.where(valid, loss, invalid_penalty)
        contributions[i] += ranking_weight*loss/len(pairs)
        contributions[j] += ranking_weight*loss/len(pairs)
        pair_costs.append(loss)
    mean_return = float(huber.sum(0).mean()/known.sum())
    mean_pair = float(np.mean(pair_costs)) if pair_costs else 0.
    return contributions, dict(return_huber=mean_return, pair_loss=mean_pair,
        objective=return_weight*mean_return+ranking_weight*mean_pair,
        pairs=len(pairs), usable_bag_fraction=float(okay[known].mean()))


def bag_score_loss(log_probabilities, contributions):
    """Each bag's baseline uses other independent bags, not its own draws."""
    values = torch.as_tensor(contributions, dtype=log_probabilities.dtype,
                             device=log_probabilities.device).detach()
    if values.shape != log_probabilities.shape or values.ndim != 2 or values.shape[1] < 2:
        raise ValueError('At least two matching independent bags required')
    if not torch.isfinite(values).all() or not torch.isfinite(log_probabilities).all():
        raise ValueError('Nonfinite bag loss')
    baseline = (values.sum(1, keepdim=True)-values)/(values.shape[1]-1)
    return ((values-baseline)*log_probabilities).sum(0).mean()


def sampled_return_ranking_loss(model, decoder, a, b, stamps, mean, scale, future,
                               valid, dates, *, seed, bags=2, paths_per_bag=8,
                               history_auxiliary=None, cost=.0025, **objective):
    if bags < 2 or paths_per_bag < 4:
        raise ValueError('Require two independent bags and at least four paths')
    samples = bags*paths_per_bag
    pairs = generate_tokens(model, a[:, :60], b[:, :60], stamps[:, :60], stamps[:, 60:],
        samples=samples, seed=seed, temperature=1., top_p=1., top_k=0,
        history_auxiliary=history_auxiliary)
    paths, _ = decode_paths(decoder, pairs, mean, scale, 5)
    grouped = paths.reshape(len(a), bags, paths_per_bag, 5, 6)
    predicted, actual = np.zeros((len(a), bags)), np.zeros((len(a), bags))
    usable = np.zeros_like(predicted, dtype=bool)
    known = np.asarray(valid, bool).all(1)
    for i in range(len(a)):
        for bag in range(bags):
            ref = reference_prices(grouped[i, bag][None], cost, max(2, paths_per_bag//2))
            if ref is not None:
                usable[i, bag] = True
                predicted[i, bag] = ref['predicted']
                if known[i]:
                    actual[i, bag] = future[i, ref['sell_offset'], 1]/future[i, 0, 2]-1-cost
    values, stats = bag_contributions(predicted, actual, usable, known, dates, **objective)
    s1, s2 = [x.reshape(-1, x.shape[-1]).clone() for x in pairs]
    repeated = stamps.repeat_interleave(samples, 0)
    training = model.training
    model.eval()
    try:
        coarse, fine = model.forecast_logits(s1[:, :-1], s2[:, :-1], repeated[:, :-1], s1[:, 1:], 59,
            auxiliary=forecast_auxiliary(None if history_auxiliary is None else history_auxiliary.repeat_interleave(samples, 0), 60, 64))
        logp = F.log_softmax(coarse, -1).gather(-1, s1[:, 60:, None]).squeeze(-1)
        logp = logp+F.log_softmax(fine, -1).gather(-1, s2[:, 60:, None]).squeeze(-1)
        logp = logp.reshape(len(a), bags, paths_per_bag, 5).sum((-1, -2))
        loss = bag_score_loss(logp, values)
    finally:
        model.train(training)
    return loss, stats
