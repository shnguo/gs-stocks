"""Signal-close market state from the available historical stock-input cohort.

This is an equal-weight research cohort, not an official exchange index. No
outcome, future availability, or present-day industry mapping is consumed.
"""
import numpy as np
import pandas as pd

MARKET_FEATURES = (
    'cohort_mean_return_1', 'cohort_mean_return_5', 'cohort_mean_return_20',
    'cohort_mean_stock_volatility_20', 'cohort_dispersion_return_1',
    'cohort_advancing_fraction', 'cohort_above_ma20_fraction',
    'cohort_volume_expanding_fraction', 'relative_to_cohort_return_1',
    'relative_to_cohort_return_5', 'relative_to_cohort_return_20',
    'relative_to_cohort_volatility_20',
)


def market_context(x, rows, feature_names, min_cohort=50):
    if x.shape != (len(rows), len(feature_names)) or not np.isfinite(x).all():
        raise ValueError('Invalid signal features')
    if rows.duplicated(['date', 'instrument_id']).any():
        raise ValueError('Duplicate cohort member')
    needed = ['return_1', 'return_5', 'return_20', 'volatility_20', 'price_mean_20', 'volume_mean_5']
    if any(name not in feature_names for name in needed):
        raise ValueError('Missing market source features')
    source = pd.DataFrame(x[:, [feature_names.index(name) for name in needed]], columns=needed)
    source['date'] = rows.date.to_numpy()
    counts = source.groupby('date').date.transform('size')
    if counts.min() < min_cohort or min_cohort < 2:
        raise ValueError('Insufficient historical cohort')
    result = np.empty((len(rows), len(MARKET_FEATURES)), dtype=np.float32)
    for date, positions in source.groupby('date', sort=True).indices.items():
        values = source.iloc[positions]
        average = values[needed[:4]].mean().to_numpy()
        common = [*average, values.return_1.std(ddof=0),
                  (values.return_1 > 0).mean(), (values.price_mean_20 > 0).mean(),
                  (values.volume_mean_5 > 0).mean()]
        result[positions, :8] = common
        result[positions, 8:] = values[needed[:4]].to_numpy()-average
    return result
