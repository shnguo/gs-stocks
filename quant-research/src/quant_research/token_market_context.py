"""Signal-close cross-stock context; no future labels or current industry backfill."""
import numpy as np
import pandas as pd

from .token_indicators import FEATURES as INDICATORS
from .token_indicators import indicator_features
from .token_transformer import valid_bars

CONTEXT = (
    'csi300_return_1', 'csi300_return_5', 'csi300_return_20',
    'market_return_1', 'market_return_5', 'market_return_20',
    'market_advancing_fraction', 'market_above_ma20_fraction',
    'market_return_dispersion_1', 'market_log_amount_relative_5',
    'industry_peer_return_1', 'industry_peer_return_5', 'industry_peer_return_20',
    'industry_peer_advancing_fraction', 'industry_peer_log_amount_relative_5',
    'industry_log_peer_count', 'stock_minus_market_return_5', 'stock_minus_industry_return_5',
)
FEATURES = INDICATORS + CONTEXT


def dated_industries(snapshot, symbols, signal, max_age_days=6):
    """Only bounded, earlier dated snapshots; preserve unknown sector membership."""
    if snapshot.instrument_id.duplicated().any() or snapshot.requested_date.nunique() != 1:
        raise ValueError('Duplicate or mixed industry snapshot')
    age = (pd.Timestamp(signal)-pd.to_datetime(snapshot.requested_date)).dt.days
    if (age < 0).any() or (age > max_age_days).any() or (snapshot.updateDate > snapshot.requested_date).any():
        raise ValueError('Industry snapshot is future-dated or stale')
    return snapshot.set_index('instrument_id').industry.reindex(symbols).fillna('').to_numpy()


def signal_context(histories, industries, index_close, min_peers=5):
    """All input-eligible stocks at one close, with self excluded from sector peers.

    histories: [stocks, 60, OHLCVA+factor]; index_close: last 21 calendar sessions.
    Official index returns and research-cohort aggregates are named separately.
    All fields are signal-day covariates, not backfilled onto earlier token positions.
    """
    x = np.asarray(histories)
    if x.ndim != 3 or x.shape[1:] != (60, 7) or len(x) < 2 or len(industries) != len(x):
        raise ValueError('Expected a complete signal-date stock cohort')
    if min_peers < 2 or not valid_bars(x[..., :6]).all() or not np.isfinite(x[..., 6]).all() or (x[..., 6] <= 0).any():
        raise ValueError('Invalid context histories or peer threshold')
    index = np.asarray(index_close, float)
    if index.shape != (21,) or not np.isfinite(index).all() or (index <= 0).any():
        raise ValueError('Complete positive dated index closes required')
    close = x[..., 3].astype(float)*x[..., 6]
    returns = np.column_stack([close[:, -1]/close[:, -1-n]-1 for n in [1, 5, 20]])
    amount = x[..., 5].astype(float)
    reference_amount = amount[:, -6:-1].mean(1)
    current_amount = amount[:, -1]
    out = np.full((len(x), len(CONTEXT)), np.nan, np.float32)
    out[:, :3] = [index[-1]/index[-1-n]-1 for n in [1, 5, 20]]
    market = returns.mean(0)
    out[:, 3:6] = market
    out[:, 6] = (returns[:, 0] > 0).mean()
    out[:, 7] = (close[:, -1] > close[:, -20:].mean(1)).mean()
    out[:, 8] = returns[:, 0].std(ddof=0)
    if reference_amount.sum() > 0 and current_amount.sum() > 0:
        out[:, 9] = np.log(current_amount.sum()/reference_amount.sum())
    out[:, 15] = 0
    out[:, 16] = returns[:, 1]-market[1]
    labels = np.asarray(industries, str)
    for group in sorted(set(labels)-{''}):
        ix = np.flatnonzero(labels == group)
        peers = len(ix)-1
        out[ix, 15] = np.log1p(peers)
        if peers < min_peers:
            continue
        averages = (returns[ix].sum(0)-returns[ix])/peers
        out[ix, 10:13] = averages
        positives = (returns[ix, 0] > 0).astype(float)
        out[ix, 13] = (positives.sum()-positives)/peers
        current = current_amount[ix].sum()-current_amount[ix]
        previous = reference_amount[ix].sum()-reference_amount[ix]
        okay = (current > 0) & (previous > 0)
        out[ix[okay], 14] = np.log(current[okay]/previous[okay])
        out[ix, 17] = returns[ix, 1]-averages[:, 1]
    return out


def contextual_history(histories, context):
    if context.shape != (len(histories), len(CONTEXT)) or np.isinf(context).any():
        raise ValueError('Signal context schema mismatch')
    result = np.full((len(histories), 60, len(FEATURES)), np.nan, np.float32)
    result[:, :, :len(INDICATORS)] = indicator_features(histories)
    result[:, -1, len(INDICATORS):] = context
    return result
