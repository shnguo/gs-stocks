"""Causal daily size-tier NET flow inputs, with missingness preserved."""
import numpy as np
import pandas as pd

TIERS = ('small_net', 'medium_net', 'large_net', 'super_large_net')
FEATURES = tuple(f'{tier}_turnover_{n}' for n in (1, 5, 20) for tier in TIERS)


def parse_eastmoney(content, symbol):
    import json
    obj = json.loads(content)
    data = obj.get('data')
    if data is None:
        return pd.DataFrame(columns=['date', 'main_net', *TIERS])
    if data.get('code') != symbol:
        raise ValueError('Flow symbol mismatch')
    records = []
    for line in data.get('klines', []):
        cells = line.split(',')
        if len(cells) < 6:
            raise ValueError('Truncated flow row')
        values = [np.nan if s in ('-', '', 'null') else float(s) for s in cells[1:6]]
        if np.isinf(values).any():
            raise ValueError('Infinite flow')
        records.append(dict(zip(['date', 'main_net', *TIERS], [cells[0], *values])))
    frame = pd.DataFrame(records, columns=['date', 'main_net', *TIERS])
    if frame.date.duplicated().any():
        raise ValueError('Duplicate flow dates')
    if len(frame) and not pd.to_datetime(frame.date).dt.strftime('%Y-%m-%d').equals(frame.date):
        raise ValueError('Invalid flow date')
    return frame.sort_values('date').reset_index(drop=True)


def size_features(flow, amount, calendar, lag=1):
    """Net CNY divided by turnover CNY; rolling sums use exact trading sessions.

    Positive and negative net values are retained. They cannot reconstruct
    gross active buy/sell amounts. Any missing observation breaks a window.
    """
    if lag < 1 or calendar != sorted(set(calendar)) or flow.date.duplicated().any():
        raise ValueError('Require unique calendar, flow dates, and positive lag')
    frame = flow.set_index('date').reindex(calendar)
    traded = amount.reindex(calendar).astype(float)
    traded = traded.where(np.isfinite(traded) & traded.gt(0))
    out = pd.DataFrame(index=calendar)
    for window in (1, 5, 20):
        denominator = traded.rolling(window, min_periods=window).sum()
        for tier in TIERS:
            values = frame[tier].astype(float)
            if np.isinf(values).any():
                raise ValueError('Infinite flow feature')
            out[f'{tier}_turnover_{window}'] = (values.rolling(window, min_periods=window).sum()/denominator).shift(lag)
    return out[list(FEATURES)].astype(np.float32)


def expand_model(model, center, scale):
    """Preserve existing indicator weights; new value/mask columns start at zero."""
    from dataclasses import replace

    import torch

    from .token_indicators import FEATURES as INDICATORS
    from .token_transformer import TokenTransformer

    if tuple(model.config.auxiliary_features) != INDICATORS:
        raise ValueError('Expected existing indicator/ranking parent')
    old_count, count = len(INDICATORS), len(INDICATORS)+len(FEATURES)
    device = next(model.parameters()).device
    expanded = TokenTransformer(replace(model.config, auxiliary_features=INDICATORS+FEATURES)).to(device)
    state = expanded.state_dict()
    for name, value in model.state_dict().items():
        if name not in ('auxiliary_center', 'auxiliary_scale', 'auxiliary_projection.0.weight'):
            state[name] = value.clone()
    state['auxiliary_center'][:old_count] = model.auxiliary_center
    state['auxiliary_scale'][:old_count] = model.auxiliary_scale
    state['auxiliary_center'][old_count:] = torch.as_tensor(center, device=device)
    state['auxiliary_scale'][old_count:] = torch.as_tensor(scale, device=device)
    weights = state['auxiliary_projection.0.weight']
    weights.zero_()
    original = model.auxiliary_projection[0].weight
    weights[:, :old_count] = original[:, :old_count]
    weights[:, count:count+old_count] = original[:, old_count:]
    expanded.load_state_dict(state)
    return expanded.train(model.training)
