"""Past-only, date-weighted empirical widening; no exchangeability guarantee."""
import numpy as np


def fit_intervals(predictions, targets, dates, min_rows=100, min_dates=5):
    dates = np.asarray(dates)
    if predictions.shape != (*targets.shape, 3) or targets.shape[1:] != (5, 4):
        raise ValueError("Expected aligned five-day OHLC quantiles")
    offsets = np.full((5, 4), np.nan)
    counts = np.zeros((5, 4), int)
    for day in range(5):
        for field in range(4):
            q, y = predictions[:, day, field], targets[:, day, field]
            valid = np.isfinite(q).all(1) & np.isfinite(y)
            ds = dates[valid]
            counts[day, field] = valid.sum()
            if valid.sum() < min_rows or len(np.unique(ds)) < min_dates:
                continue
            # Each calibration signal date has the same total weight.
            score = np.maximum(q[valid, 0] - y[valid], y[valid] - q[valid, 2])
            weight = np.array([1 / np.sum(ds == d) for d in ds])
            order = np.argsort(score, kind="stable")
            cdf = np.cumsum(weight[order]) / weight.sum()
            index = min(np.searchsorted(cdf, .8), len(order) - 1)
            offsets[day, field] = max(0., float(score[order[index]]))
    return {"offsets": offsets, "rows": counts,
            "method": "date_equal_empirical_80pct_nonnegative_CQR_score",
            "coverage_guarantee": False}


def apply_intervals(predictions, offsets):
    out = predictions.copy()
    offsets = np.asarray(offsets)
    if offsets.shape != (5, 4) or (offsets[np.isfinite(offsets)] < 0).any():
        raise ValueError("Invalid widening offsets")
    out[..., 0] -= offsets
    out[..., 2] += offsets
    out[:, ~np.isfinite(offsets)] = np.nan
    return out
