"""Backtest 'pullback within an uptrend' baskets on the US index union.

Usage: uv run python scripts/pullback_check.py configs/pullback-check-v1.json
"""
import argparse
import importlib.util
import json
import time
import warnings
from functools import cache
from pathlib import Path

import numpy as np

from quant_research.baskets import BasketBook
from quant_research.factors import _past_return, forward_returns, percentile, physics_momentum
from quant_research.storage import write_json

BASE = Path(__file__).resolve().parents[1]


@cache
def load_script(name):
    spec = importlib.util.spec_from_file_location(name, BASE / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recent_jump_down(panel, t, lookback, recent, jump_sigma):
    """True where the last `recent` sessions hold a jump-down day, with median and robust
    sigma taken from the `lookback` window (same rule as physics_momentum)."""
    window = slice(t - lookback, t + 1)
    close, seq = panel.close[window], panel.sequence[window]
    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r = np.log(close[1:] / close[:-1])
        r = np.where((seq[1:] == seq[:-1]) & (seq[1:] >= 0) & np.isfinite(r), r, np.nan)
        median = np.nanmedian(r, axis=0)
        sigma = 1.4826 * np.nanmedian(np.abs(r - median), axis=0)
        sigma = np.where(sigma > 0, sigma, np.nanstd(r, axis=0))
        tail = r[-recent:]
        return (np.isfinite(tail) & (tail - median < -jump_sigma * sigma)).any(axis=0)


def pullback_picks(panel, t, drift, mask, cfg):
    """Every basket for one signal date; also returns the inputs for the screen."""
    uptrend = load_script("uptrend_check")
    ok = mask & np.isfinite(drift)
    top = ok & (percentile(drift, ok) > 0.8)
    ma = uptrend.moving_average(panel, t, cfg["ma_days"], cfg["ma_min_share"])
    below = np.isfinite(ma) & (panel.close[t] < ma)
    above = np.isfinite(ma) & (panel.close[t] > ma)
    r1m = _past_return(panel, t, cfg["pullback_days"], 0, int(0.7 * cfg["pullback_days"]))
    r5 = _past_return(panel, t, cfg["recent_days"], 0, cfg["recent_days"] - 1)
    gap = recent_jump_down(panel, t, 252, cfg["pullback_days"], cfg["jump_sigma"])
    low_1m = percentile(r1m, mask & np.isfinite(r1m)) <= cfg["pool_bottom_share"]
    hi, mid, lo = cfg["depth_cuts"]
    pull = top & below
    picks = {
        "trend_all": top, "trend_above_ma50": top & above, "pullback_ma50": pull,
        "pullback_1m": top & low_1m,
        "depth_0_10": top & (r1m <= hi) & (r1m > mid), "depth_10_20": top & (r1m <= mid) & (r1m > lo),
        "depth_20_plus": top & (r1m <= lo),
        "pullback_gap": pull & gap, "pullback_gentle": pull & ~gap,
        "pullback_stabilized": pull & (r5 > 0), "pullback_still_falling": pull & (r5 <= 0),
    }
    inputs = {"ma50": ma, "below_ma50": below, "return_1m": r1m, "return_5d": r5, "gap_down_1m": gap, "trend_top": top}
    return picks, inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    cfg = json.loads(parser.parse_args().config.read_text())
    output = BASE / cfg["output"]
    if output.exists():
        raise FileExistsError(f"Use a new output path: {output}")
    started = time.time()
    us = json.loads((BASE / cfg["us_config"]).read_text())
    panel, months, source_hash = load_script("physics_robustness").us_months(us)
    m = us["momentum"]
    book = BasketBook(cfg["baskets"], us["cost_one_way"])
    long_h = max(cfg["horizon_months"])
    for i, (date, following, mask, _) in enumerate(months):
        t = panel.index(date)
        returns, _ = forward_returns(panel, date, following)
        long_returns = None
        if i + long_h - 1 < len(months):
            long_returns, _ = forward_returns(panel, date, months[i + long_h - 1][1])
            long_returns = np.where(mask, long_returns, np.nan)
        drift = np.where(mask, physics_momentum(panel, date, m["lookback"], m["skip"], m["min_traded"],
                                                cfg["jump_sigma"])["drift"], np.nan)
        picks, _ = pullback_picks(panel, t, drift, mask, cfg)
        valid = mask & np.isfinite(returns) & np.isfinite(drift)
        if valid.sum() < 100:
            continue
        book.add(date, np.asarray(panel.ids), returns, valid, picks, long_returns)
    output.mkdir(parents=True)
    book.frame().to_csv(output / "baskets.csv", index=False)
    results = book.summary(cfg["baskets"], us["bootstrap_block_months"], us["bootstrap_repetitions"])
    write_json(output / "results.json", {"version": cfg["version"], "config": cfg, "source_manifest_sha256": source_hash,
                                         "baskets": results, "elapsed_seconds": round(time.time() - started, 1)})
    print(f"wrote {output} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
