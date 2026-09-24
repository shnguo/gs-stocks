"""Learn a residual return head from mature, previously generated forecasts.

This head changes expected scenario returns, never OHLCVA paths or exit dates.
All returns are fractions; regression features and residual targets use pp.
"""
from __future__ import annotations

import numpy as np

FEATURES = ('predicted_pp', 'volatility_pp', 'return_volatility_interaction')


def features(frame):
    p = frame.predicted.to_numpy(float) * 100
    v = frame.volatility_pp.to_numpy(float)
    x = np.column_stack((p, v, p * v))
    if not np.isfinite(x).all() or (v < 0).any():
        raise ValueError('Finite predictions and nonnegative volatility required')
    return x


def mature_rows(frame, as_of):
    """Conservative embargo: the entire five-day label must end BEFORE signal."""
    if frame[['instrument_id', 'date']].duplicated().any():
        raise ValueError('Duplicate stock/date training rows')
    if frame[['date', 'label_end']].isna().any().any():
        raise ValueError('Missing label dates')
    if (frame.label_end <= frame.date).any():
        raise ValueError('Label end must follow signal date')
    return frame[frame.date.lt(as_of) & frame.label_end.lt(as_of) & frame.label_known].copy()


def fit_head(frame, as_of, *, ridge=0.01, min_dates=20):
    """Date-balanced ridge on actual-minus-predicted; shrink toward raw forecast.

    Equal dates prevent changing stock coverage from implicitly weighting regimes.
    No feature clipping, volatility exclusion, or fixed return subtraction.
    """
    if not np.isfinite(ridge) or ridge <= 0 or min_dates < 2:
        raise ValueError('Positive ridge and at least two training dates required')
    train = mature_rows(frame, as_of)
    dates = train.date.nunique()
    if dates < min_dates:
        raise ValueError('Insufficient mature dates')
    x = features(train)
    y = (train.actual_extrema_scenario - train.predicted).to_numpy(float) * 100
    if not np.isfinite(y).all():
        raise ValueError('Known labels must be finite')
    w = 1.0 / train.groupby('date').date.transform('size').to_numpy() / dates
    center = np.sum(w[:, None] * x, axis=0)
    scale = np.sqrt(np.sum(w[:, None] * (x - center) ** 2, axis=0))
    scale = np.where(scale > 1e-12, scale, 1.0)
    z = np.column_stack((np.ones(len(x)), (x - center) / scale))
    penalty = np.diag([0.0, ridge, ridge, ridge])
    coefficients = np.linalg.solve(z.T @ (w[:, None] * z) + penalty, z.T @ (w * y))
    return dict(version='return-calibration-v1', features=list(FEATURES),
                center=center.tolist(), scale=scale.tolist(), coefficients=coefficients.tolist(),
                as_of=as_of, trained_labels_through=train.label_end.max(),
                training_first=train.date.min(), training_last=train.date.max(),
                training_rows=len(train), training_dates=int(dates), ridge=ridge)


def predict_head(frame, head):
    if head['version'] != 'return-calibration-v1' or head['features'] != list(FEATURES):
        raise ValueError('Incompatible return head')
    if len(frame) and (frame.date.min() < head['as_of'] or
                       frame.date.min() <= head['trained_labels_through']):
        raise ValueError('Return head trained on unavailable labels')
    center, scale, coef = [np.asarray(head[k], float) for k in ('center', 'scale', 'coefficients')]
    if (center.shape != (3,) or scale.shape != (3,) or coef.shape != (4,) or
            not np.isfinite(np.concatenate((center, scale, coef))).all() or (scale <= 0).any()):
        raise ValueError('Invalid return head parameters')
    z = (features(frame) - center) / scale
    return frame.predicted.to_numpy(float) + (coef[0] + z @ coef[1:]) / 100


def rank_head(frame, head):
    """Rank using only forecast fields. Outcomes are optional and never accessed."""
    if frame.date.nunique() != 1 or frame.instrument_id.duplicated().any():
        raise ValueError('One unique cross-section required')
    result = frame.copy()
    result['raw_predicted'] = result.predicted
    result['predicted'] = predict_head(frame, head)
    if not np.isfinite(result.predicted).all():
        raise ValueError('Nonfinite calibrated forecast')
    result = result.sort_values(['predicted', 'instrument_id'], ascending=[False, True])
    result['rank'] = np.arange(1, len(result) + 1)
    return result


def volatility_pp(close, adjustment):
    """Same 20 adjusted close-to-close returns used by the frozen experiment."""
    prices = np.asarray(close, float) * np.asarray(adjustment, float)
    if prices.shape != (21,) or not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError('Exactly 21 positive adjusted closing prices required')
    return float(np.diff(np.log(prices)).std(ddof=0) * 100)
