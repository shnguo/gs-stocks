"""Causal, price-scale-invariant indicators from the decoder's existing history.

Fixed conventions: BOLL(20, 2), population standard deviation; MACD(12,26,9),
first-observation-seeded recursive EMA, histogram DIF-DEA; KDJ(9,3,3), K/D
initialized to 50, expressed on a 0..1 scale. J is omitted as redundant.
No values before warm-up, no cross-gap fill, and no history outside the input.
"""
from __future__ import annotations

import numpy as np

from .token_transformer import valid_bars

BOLL = ("boll_position_20", "boll_width_20")
MACD = ("macd_dif_12_26_relative", "macd_hist_12_26_9_relative")
KDJ = ("kdj_k_9_3", "kdj_d_9_3_3")
FEATURES = BOLL + MACD + KDJ
GROUPS = {"baseline": (), "boll": BOLL, "macd": MACD, "kdj": KDJ, "all": FEATURES}


def _ema(values, span):
    """Recursive EMA; reset state on any missing session and mask warm-up."""
    out = np.full(values.shape, np.nan, np.float64)
    state = np.full(len(values), np.nan)
    count = np.zeros(len(values), int)
    alpha = 2. / (span + 1)
    for t in range(values.shape[1]):
        v = values[:, t]
        ok = np.isfinite(v)
        state = np.where(ok, np.where(count > 0, alpha*v + (1-alpha)*state, v), np.nan)
        count = np.where(ok, count + 1, 0)
        out[:, t] = np.where(count >= span, state, np.nan)
    return out


def indicator_features(raw):
    """[row, history, OHLCVA+factor] -> [row, history, six continuous features].

    Inputs must preserve trading-calendar positions. Missing/invalid bars break
    rolling windows and EMA/KDJ state. Adjustment uses each known bar's factor;
    the outputs are invariant to a common price/factor rescaling.
    """
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 3 or raw.shape[-1] != 7 or raw.shape[1] < 1:
        raise ValueError("Expected historical OHLCVA plus adjustment factor")
    if np.isinf(raw).any():
        raise ValueError("Infinite indicator input")
    valid = valid_bars(raw[..., :6]) & np.isfinite(raw[..., 6]) & (raw[..., 6] > 0)
    price = raw[..., :4] * raw[..., 6:7]
    if np.isinf(price).any():
        raise ValueError("Adjusted price overflow")
    price = np.where(valid[..., None], price, np.nan)
    high, low, close = price[..., 1], price[..., 2], price[..., 3]
    output = np.full((*raw.shape[:2], len(FEATURES)), np.nan, np.float64)
    n = raw.shape[1]
    for t in range(19, n):
        window = close[:, t-19:t+1]
        complete = np.isfinite(window).all(1)
        mean, std = window.mean(1), window.std(1, ddof=0)
        width = 4*std
        np.divide(close[:, t] - (mean-2*std), width, out=output[:, t, 0],
                  where=complete & (std > np.abs(mean)*1e-12))
        np.divide(width, mean, out=output[:, t, 1], where=complete & (mean > 0))
    dif = _ema(close, 12) - _ema(close, 26)
    signal = _ema(dif, 9)
    output[..., 2] = dif / close
    output[..., 3] = (dif-signal) / close
    k, d = np.full(len(raw), .5), np.full(len(raw), .5)
    for t in range(8, n):
        hi, lo = high[:, t-8:t+1].max(1), low[:, t-8:t+1].min(1)
        span = hi-lo
        ok = np.isfinite(hi) & np.isfinite(lo) & (span > np.abs(hi)*1e-12)
        rsv = np.full(len(raw), np.nan)
        np.divide(close[:, t]-lo, span, out=rsv, where=ok)
        k = np.where(ok, (2*k+rsv)/3, .5)
        d = np.where(ok, (2*d+k)/3, .5)
        output[:, t, 4] = np.where(ok, k, np.nan)
        output[:, t, 5] = np.where(ok, d, np.nan)
    return output.astype(np.float32)


def predict_indicator_paths(model, tokenizer, raw, past_stamps, future_stamps, **sampling):
    """Inference adapter: build exactly the same features used during training."""
    from .token_transformer import predict_paths

    names = model.config.auxiliary_features
    if names and tuple(names) not in GROUPS.values():
        raise ValueError("Checkpoint is not a supported indicator variant")
    history = None if not names else indicator_features(raw)[..., [FEATURES.index(k) for k in names]]
    return predict_paths(model, tokenizer, raw, past_stamps, future_stamps,
                         history_auxiliary=history, **sampling)
