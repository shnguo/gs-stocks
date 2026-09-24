"""Exact-date auxiliary inputs for the market-token decoder.

No forward fill, current share-count substitution, or future financial values.
Retrospective provider captures are explicitly research-only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TURNOVER = (
    "log_turnover_pct", "turnover_change_1", "turnover_change_5",
    "turnover_relative_5", "turnover_relative_20",
)
VALUATION = ("earnings_yield", "earnings_yield_change_5", "earnings_yield_change_20", "negative_pe")
FEATURES = TURNOVER + VALUATION
GROUPS = {"baseline": (), "turnover": TURNOVER, "valuation": VALUATION, "both": FEATURES}


def daily_features(bars, source, calendar, *, retrospective=False, cutoff_time="16:30"):
    """Return exact calendar features. Volume and float_shares must both be shares.

    For prospective inputs, source.available_at must certify the value vintage,
    not merely describe the collection date. Missing provenance is rejected.
    """
    keys = ["instrument_id", "date"]
    dates = list(calendar)
    if not dates or dates != sorted(set(dates)):
        raise ValueError("A unique ordered trading calendar is required")
    for frame in [bars, source]:
        if frame.duplicated(keys).any() or frame[keys].isna().any().any():
            raise ValueError("Duplicate or missing feature keys")
        if not frame.date.isin(dates).all():
            raise ValueError("Feature date outside calendar")
    s = source.copy()
    if not retrospective:
        if "available_at" not in s or s.available_at.isna().any():
            raise ValueError("Point-in-time available_at evidence is required")
        if any(pd.Timestamp(value).tzinfo is None for value in s.available_at):
            raise ValueError("available_at requires an explicit timezone")
        available = pd.to_datetime(s.available_at, utc=True, errors="raise")
        cutoffs = pd.to_datetime(s.date + " " + cutoff_time).dt.tz_localize("Asia/Shanghai").dt.tz_convert("UTC")
        s.loc[available > cutoffs, ["close", "float_shares", "pe_ttm"]] = np.nan
    joined = bars[keys + ["close", "volume"]].merge(
        s[keys + ["close", "float_shares", "pe_ttm"]], on=keys, how="left",
        suffixes=("", "_source"), validate="one_to_one",
    )
    raw = joined[["close", "volume", "close_source", "float_shares", "pe_ttm"]].to_numpy(float)
    if np.isinf(raw).any():
        raise ValueError("Infinite feature source")
    match = (np.isfinite(raw[:, 0]) & (raw[:, 0] > 0) & np.isfinite(raw[:, 2])
             & (np.abs(raw[:, 0] - raw[:, 2]) <= .010001))
    volume, shares, pe = raw[:, 1], raw[:, 3], raw[:, 4]
    turnover = np.full(len(raw), np.nan)
    ok = match & np.isfinite(volume) & (volume >= 0) & np.isfinite(shares) & (shares > 0)
    turnover[ok] = 100 * volume[ok] / shares[ok]
    ep = np.full(len(raw), np.nan)
    ok = match & np.isfinite(pe) & (np.abs(pe) > 1e-6)
    ep[ok] = 1 / pe[ok]
    joined["turnover"] = turnover
    joined["ep"] = ep
    joined["negative_pe"] = np.where(match & np.isfinite(pe), (pe < 0).astype(float), np.nan)
    records = []
    for stock, group in joined.groupby("instrument_id", sort=False):
        # Reindex on the actual trading calendar: shift(5) means five sessions,
        # never the fifth sparse provider observation.
        expanded = group.set_index("date").reindex(dates)
        t, e = expanded.turnover, expanded.ep
        out = pd.DataFrame(index=expanded.index)
        out["log_turnover_pct"] = np.log1p(t)
        out["turnover_change_1"] = t - t.shift(1)
        out["turnover_change_5"] = t - t.shift(5)
        for n in [5, 20]:
            mean = t.rolling(n, min_periods=n).mean()
            out[f"turnover_relative_{n}"] = t / mean.where(mean > 0) - 1
        out["earnings_yield"] = e
        out["earnings_yield_change_5"] = e - e.shift(5)
        out["earnings_yield_change_20"] = e - e.shift(20)
        out["negative_pe"] = expanded.negative_pe
        out = out.reindex(group.date)
        out["instrument_id"] = stock
        out.index.name = "date"
        records.append(out.reset_index())
    if not records:
        return pd.DataFrame(columns=keys + list(FEATURES))
    return pd.concat(records, ignore_index=True)[keys + list(FEATURES)]


def history_features(rows, features, calendar, lookback=60):
    """Gather features ending at each signal date, preserving source gaps."""
    keys = ["instrument_id", "date"]
    if features.duplicated(keys).any():
        raise ValueError("Duplicate feature key")
    date_index = pd.Index(calendar).get_indexer(rows.date)
    if (date_index < lookback - 1).any():
        raise ValueError("Insufficient feature calendar history")
    dates = np.asarray(calendar)[date_index[:, None] + np.arange(1-lookback, 1)]
    requested = pd.MultiIndex.from_arrays([
        np.repeat(rows.instrument_id.to_numpy(), lookback), dates.ravel()
    ], names=keys)
    values = features.set_index(keys).reindex(requested)[list(FEATURES)].to_numpy(np.float32)
    return values.reshape(len(rows), lookback, len(FEATURES))


def fit_normalizer(history):
    """Fit on training histories only; unsupported columns stay masked, scale one."""
    x = np.asarray(history, dtype=np.float64).reshape(-1, history.shape[-1])
    if np.isinf(x).any():
        raise ValueError("Infinite training features")
    available = np.isfinite(x)
    count = available.sum(0)
    center = np.where(available, x, 0).sum(0) / np.maximum(count, 1)
    variance = np.where(available, (x - center)**2, 0).sum(0) / np.maximum(count, 1)
    scale = np.sqrt(variance)
    scale = np.where(scale > 1e-6, scale, 1.)
    return center.astype(np.float32), scale.astype(np.float32)
