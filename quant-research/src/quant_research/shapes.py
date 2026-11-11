"""Chart-shape scores built from drift (daily moves with jump days removed).

rising_now    drift over the recent period, latest month included
turning_up    fell over the 189 sessions before the recent period, rising during it
long_uptrend  12-1 drift, with stocks still rising over the recent period ranked first

Each shape returns (score, basket): a cross-sectional score (higher = closer to the shape)
and the boolean set of stocks that match it. Shared by the backtest and the live screen.
"""
from __future__ import annotations

import numpy as np

from .factors import PricePanel, composite, percentile, physics_momentum


def window_drift(panel: PricePanel, date: str, lookback: int, skip: int, sigma: float,
                 min_share: float) -> np.ndarray:
    return physics_momentum(panel, date, lookback, skip, int(np.ceil((lookback - skip) * min_share)),
                            sigma)["drift"]


def shape_scores(panel: PricePanel, date: str, mask: np.ndarray, drift_12_1: np.ndarray,
                 cfg: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{"<shape>_<period>": (score, basket)} for every shape and period in cfg."""
    sigma, share, prior_len = cfg["jump_sigma"], cfg["min_traded_share"], cfg["prior_sessions"]
    keep = lambda x: np.where(mask, x, np.nan)  # noqa: E731
    base_rank = percentile(drift_12_1, mask)
    top_fifth = base_rank > 0.8
    out = {}
    for name, sessions in cfg["periods"].items():
        recent = keep(window_drift(panel, date, sessions, 0, sigma, share))
        prior = keep(window_drift(panel, date, sessions + prior_len, sessions, sigma, share))

        rising = percentile(recent, mask)
        out[f"rising_now_{name}"] = (rising, rising > 0.8)

        ok = mask & np.isfinite(prior) & np.isfinite(recent)
        turn = np.where(ok, composite([-prior, recent], ok, 2), np.nan)
        out[f"turning_up_{name}"] = (turn, ok & (prior < 0) & (recent > 0))

        up = np.isfinite(base_rank) & np.isfinite(recent)
        gated = np.where(up, base_rank + (recent > 0), np.nan)
        out[f"long_uptrend_{name}"] = (gated, up & top_fifth & (recent > 0))
    return out
