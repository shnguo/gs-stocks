"""Causal trend-stage features and forecast-only reranking utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .token_transformer import valid_bars

FEATURES = (
    "return_5", "return_10", "return_20", "ma5_distance", "ma10_distance",
    "ma20_distance", "ma5_slope", "ma5_atr", "volume_ratio_20",
    "pullback_5", "range_10", "early_stage_score", "extension_risk",
    "extension_flags", "overextended", "right_side",
)


def market_segment(symbol):
    """Map a canonical A-share identifier to a matching segment."""
    exchange, code = symbol.split(".")[-2:]
    if exchange == "xbse":
        return "XBSE"
    if exchange == "xshg" and code.startswith(("688", "689")):
        return "STAR"
    if exchange == "xshe" and code.startswith(("300", "301")):
        return "CHINEXT"
    return "MAIN"


def matching_history_controls(raw):
    """Compute forecast-time volatility and liquidity controls from history."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 3 or raw.shape[1] < 21 or raw.shape[2] != 7:
        raise ValueError("Expected at least 21 OHLCVA plus factor observations")
    okay = valid_bars(raw[..., :6]) & np.isfinite(raw[..., 6]) & (raw[..., 6] > 0)
    close = np.where(okay, raw[..., 3]*raw[..., 6], np.nan)
    amount = np.where(okay, raw[..., 5], np.nan)
    log_return = np.diff(np.log(close[:, -21:]), axis=1)
    volatility = np.std(log_return, axis=1, ddof=1)
    mean_amount = np.mean(amount[:, -20:], axis=1)
    log_amount = np.where(np.isfinite(mean_amount) & (mean_amount > 0),
                          np.log(mean_amount), np.nan)
    complete = okay[:, -21:].all(axis=1)
    return pd.DataFrame({
        "historical_volatility_20": np.where(complete, volatility, np.nan),
        "log_amount_20": np.where(complete, log_amount, np.nan),
    })


def _triangle(values, low, peak, high):
    values = np.asarray(values, dtype=float)
    if not low < peak < high:
        raise ValueError("Triangle bounds must be strictly increasing")
    left = (values-low)/(peak-low)
    right = (high-values)/(high-peak)
    return np.clip(np.minimum(left, right), 0., 1.)


def history_features(raw):
    """Return causal features from sixty OHLCVA plus factor observations.

    The final input bar is the signal close. Every output is invariant to a
    common price or factor rescaling. Incomplete rows remain missing and score
    zero rather than receiving imputed evidence.
    """
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 3 or raw.shape[1] < 21 or raw.shape[2] != 7:
        raise ValueError("Expected at least 21 OHLCVA plus factor observations")
    if np.isinf(raw).any():
        raise ValueError("Infinite history input")
    okay = valid_bars(raw[..., :6]) & np.isfinite(raw[..., 6]) & (raw[..., 6] > 0)
    price = np.where(okay[..., None], raw[..., :4]*raw[..., 6:7], np.nan)
    high, low, close = price[..., 1], price[..., 2], price[..., 3]
    volume = np.where(okay, raw[..., 4], np.nan)

    def mean_last(n):
        return np.mean(close[:, -n:], axis=1)

    last = close[:, -1]
    ma5, ma10, ma20 = mean_last(5), mean_last(10), mean_last(20)
    old_ma5 = np.mean(close[:, -10:-5], axis=1)
    previous = close[:, -15:-1]
    tr = np.maximum.reduce([
        high[:, -14:]-low[:, -14:],
        np.abs(high[:, -14:]-previous),
        np.abs(low[:, -14:]-previous),
    ])
    atr14 = np.mean(tr, axis=1)
    prior_volume = np.mean(volume[:, -21:-1], axis=1)

    ma5_atr = np.full(len(raw), np.nan)
    volume_ratio = np.full(len(raw), np.nan)
    np.divide(last-ma5, atr14, out=ma5_atr, where=np.isfinite(atr14) & (atr14 > 0))
    np.divide(volume[:, -1], prior_volume, out=volume_ratio,
              where=np.isfinite(prior_volume) & (prior_volume > 0))
    result = pd.DataFrame({
        "return_5": last/close[:, -6]-1,
        "return_10": last/close[:, -11]-1,
        "return_20": last/close[:, -21]-1,
        "ma5_distance": last/ma5-1,
        "ma10_distance": last/ma10-1,
        "ma20_distance": last/ma20-1,
        "ma5_slope": ma5/old_ma5-1,
        "ma5_atr": ma5_atr,
        "volume_ratio_20": volume_ratio,
        "pullback_5": last/np.max(high[:, -5:], axis=1)-1,
        "range_10": np.max(high[:, -10:], axis=1)/np.min(low[:, -10:], axis=1)-1,
    })
    right = (result.ma5_distance >= 0) & (result.ma5_slope > 0)
    components = np.stack([
        _triangle(result.ma5_atr, 0., .25, .8),
        _triangle(result.ma10_distance, -.02, .015, .04),
        _triangle(result.ma20_distance, -.03, .02, .06),
        _triangle(result.return_10, -.02, .04, .10),
        _triangle(result.return_20, -.05, .05, .15),
        _triangle(result.volume_ratio_20, .8, 1.3, 3.),
        _triangle(result.pullback_5, -.08, -.03, .01),
    ], axis=1)
    complete = np.isfinite(result.to_numpy()).all(axis=1)
    result["early_stage_score"] = np.where(complete & right, components.mean(axis=1), 0.)
    flags = np.stack([
        result.ma5_atr > .8,
        result.ma10_distance > .04,
        result.ma20_distance > .06,
        result.return_10 > .10,
        (result.volume_ratio_20 > 3.) & (result.pullback_5 > -.02),
    ], axis=1)
    result["extension_flags"] = flags.sum(axis=1).astype(np.int8)
    result["extension_risk"] = result.extension_flags/flags.shape[1]
    result["overextended"] = result.extension_flags >= 2
    result["right_side"] = right
    result.loc[~complete, ["early_stage_score", "extension_risk"]] = 0.
    result.loc[~complete, "extension_flags"] = 0
    result.loc[~complete, ["overextended", "right_side"]] = False
    return result[list(FEATURES)]


def path_execution_statistics(models, sell_offset, cost=.0025, minimum_paths=16):
    """Aggregate model medians for an executable entry and downside estimate.

    models is a list of arrays shaped row, draw, five days, six fields. Entry
    uses the first forecast session open. Exit uses the same forecast-selected
    future-high day as the reference publication.
    """
    if not models or cost < 0 or minimum_paths < 1:
        raise ValueError("Models, nonnegative cost and legal-path minimum required")
    sell_offset = np.asarray(sell_offset, dtype=int)
    n = len(sell_offset)
    if sell_offset.shape != (n,) or np.any((sell_offset < 1) | (sell_offset > 4)):
        raise ValueError("Sell offsets must select T+1 through T+4")
    entry, sell, downside, eligible = [], [], [], np.ones(n, dtype=bool)
    for paths in models:
        paths = np.asarray(paths, dtype=float)
        if paths.shape[0] != n or paths.ndim != 4 or paths.shape[2:] != (5, 6):
            raise ValueError("Expected row, draw, five-day OHLCVA paths")
        model_entry = np.full(n, np.nan)
        model_sell = np.full(n, np.nan)
        model_downside = np.full(n, np.nan)
        legal = valid_bars(paths).all(axis=-1)
        for i in range(n):
            x = paths[i, legal[i]]
            if len(x) < minimum_paths:
                eligible[i] = False
                continue
            offset = sell_offset[i]
            e = x[:, 0, 0]
            model_entry[i] = np.median(e)
            model_sell[i] = np.median(x[:, offset, 1])
            model_downside[i] = np.median(x[:, :offset+1, 2].min(axis=1)/e-1)
        entry.append(model_entry)
        sell.append(model_sell)
        downside.append(model_downside)
    entry = np.mean(entry, axis=0)
    sell = np.mean(sell, axis=0)
    downside = np.mean(downside, axis=0)
    eligible &= np.isfinite(entry) & np.isfinite(sell) & np.isfinite(downside) & (entry > 0)
    expected = sell/entry-1-cost
    return pd.DataFrame({
        "execution_entry": entry,
        "execution_sell": sell,
        "predicted_execution_return": expected,
        "predicted_downside": downside,
        "execution_forecast_eligible": eligible,
    })


def rerank(frame, weights):
    """Create four forecast-only rankings with fixed, scale-free weights."""
    required = {"instrument_id", "predicted", "predicted_execution_return",
                "predicted_downside", "early_stage_score", "extension_risk"}
    if required-set(frame):
        raise ValueError("Missing reranking inputs: "+str(sorted(required-set(frame))))
    frame = frame.copy()
    strengths = {
        "return_strength": frame.predicted.rank(method="average", pct=True),
        "execution_strength": frame.predicted_execution_return.rank(method="average", pct=True),
        "downside_strength": frame.predicted_downside.rank(method="average", pct=True),
    }
    for name, value in strengths.items():
        frame[name] = value
    definitions = {
        "control": frame.return_strength,
        "overheat_penalty": frame.return_strength-weights["overheat_penalty"]*frame.extension_risk,
        "early_stage": weights["early_return"]*frame.return_strength+
                       weights["early_quality"]*frame.early_stage_score,
        "combined": weights["combined_return"]*frame.return_strength+
                    weights["combined_execution"]*frame.execution_strength+
                    weights["combined_early"]*frame.early_stage_score+
                    weights["combined_downside"]*frame.downside_strength,
    }
    outputs = []
    for arm, score in definitions.items():
        out = frame.copy()
        out["arm"] = arm
        out["ranking_score"] = score
        out = out.sort_values(["ranking_score", "instrument_id"], ascending=[False, True])
        out["rank"] = np.arange(1, len(out)+1)
        outputs.append(out)
    return pd.concat(outputs, ignore_index=True)


def select_start_stage(frame, criteria):
    """Filter out extended moves, then rank eligible setups by forecast return."""
    required = {
        "instrument_id", "reference_price_net_return", "return_5", "return_10",
        "return_20", "ma20_distance", "volume_ratio_20", "pullback_5",
        "early_stage_score", "extension_flags", "right_side",
    }
    missing = required-set(frame)
    if missing:
        raise ValueError("Missing start-stage inputs: "+str(sorted(missing)))
    bounds = [
        ("return_5", "min_return_5", "max_return_5"),
        ("return_10", "min_return_10", "max_return_10"),
        ("return_20", "min_return_20", "max_return_20"),
        ("ma20_distance", "min_ma20_distance", "max_ma20_distance"),
        ("volume_ratio_20", "min_volume_ratio_20", "max_volume_ratio_20"),
        ("pullback_5", "min_pullback_5", "max_pullback_5"),
    ]
    selected = frame.copy()
    eligible = selected.right_side.astype(bool)
    eligible &= selected.extension_flags.le(criteria["max_extension_flags"])
    eligible &= selected.early_stage_score.ge(criteria["min_early_stage_score"])
    for column, lower, upper in bounds:
        eligible &= selected[column].between(criteria[lower], criteria[upper], inclusive="both")
    selected["start_stage_eligible"] = eligible
    selected = selected.loc[eligible].sort_values(
        ["reference_price_net_return", "early_stage_score", "instrument_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    selected["start_stage_rank"] = np.arange(1, len(selected)+1)
    return selected
