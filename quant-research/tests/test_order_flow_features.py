import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from collect_order_flow_free import listed_on

from quant_research.order_flow_features import (
    FEATURES,
    TIERS,
    expand_model,
    parse_eastmoney,
    size_features,
)
from quant_research.token_indicators import FEATURES as INDICATORS
from quant_research.token_transformer import TokenConfig, TokenTransformer


def test_parser_keeps_signed_net_and_rejects_wrong_symbol_duplicates():
    obj = dict(data=dict(code='600036', klines=['2026-04-01,3,-1,-2,1,2']))
    frame = parse_eastmoney(json.dumps(obj), '600036')
    assert frame.iloc[0].small_net == -1 and frame.iloc[0].super_large_net == 2
    with pytest.raises(ValueError, match='symbol'):
        parse_eastmoney(json.dumps(obj), '000001')
    obj['data']['klines'] *= 2
    with pytest.raises(ValueError, match='Duplicate'):
        parse_eastmoney(json.dumps(obj), '600036')
    assert parse_eastmoney('{"data":null}', '600036').empty


def test_rolling_calendar_lag_and_missingness_cannot_leak_signal_or_future():
    dates = pd.bdate_range('2026-03-01', periods=30).strftime('%Y-%m-%d').tolist()
    flow = pd.DataFrame(dict(date=dates, **{n: np.arange(30, dtype=float)+1 for n in TIERS}))
    amount = pd.Series(100., index=dates)
    a = size_features(flow, amount, dates)
    assert a.loc[dates[20], 'small_net_turnover_20'] == pytest.approx(210/2000)
    assert a.loc[dates[20], 'small_net_turnover_1'] == pytest.approx(.2)
    poisoned = flow.copy()
    poisoned.loc[20:, list(TIERS)] = 100000.
    pd.testing.assert_frame_equal(a.iloc[:21], size_features(poisoned, amount, dates).iloc[:21])
    missing = size_features(flow.drop(index=10), amount, dates)
    assert np.isnan(missing.loc[dates[20], 'small_net_turnover_20'])
    assert np.isfinite(missing.loc[dates[20], 'small_net_turnover_5'])
    with pytest.raises(ValueError, match='positive lag'):
        size_features(flow, amount, dates, lag=0)
    assert size_features(flow, amount*0, dates).isna().all().all()


def test_equal_size_expansion_preserves_parent_and_control_has_no_flow_gradient():
    torch.manual_seed(17)
    parent = TokenTransformer(TokenConfig(width=16, heads=2, layers=1, s1_bits=2, s2_bits=2,
        auxiliary_features=INDICATORS, dropout=0)).eval()
    torch.nn.init.normal_(parent.auxiliary_projection[-1].weight, std=.05)
    expanded = expand_model(parent, np.zeros(len(FEATURES)), np.ones(len(FEATURES))).eval()
    tokens = torch.zeros((2, 5), dtype=torch.long)
    stamps = torch.zeros((2, 5, 5), dtype=torch.long)
    indicator = torch.randn(2, 5, len(INDICATORS))
    features = torch.cat([indicator, torch.randn(2, 5, len(FEATURES))], dim=-1)
    with torch.no_grad():
        old = parent.encode_history(tokens, tokens, stamps, indicator)
        new = expanded.encode_history(tokens, tokens, stamps, features)
        torch.testing.assert_close(new, old, atol=1e-6, rtol=1e-6)
    expanded.encode_history(tokens, tokens, stamps, features)[..., 0].sum().backward()
    assert expanded.auxiliary_projection[0].weight.grad[:, len(INDICATORS):len(INDICATORS)+len(FEATURES)].abs().sum() > 0
    expanded.zero_grad()
    features[:, :, len(INDICATORS):] = torch.nan
    expanded.encode_history(tokens, tokens, stamps, features)[..., 0].sum().backward()
    assert expanded.auxiliary_projection[0].weight.grad[:, len(INDICATORS):len(INDICATORS)+len(FEATURES)].abs().sum() == 0


def test_empty_delisting_date_is_active_and_future_listing_is_excluded():
    for end in ['', None, '2026-06-01']:
        assert listed_on(SimpleNamespace(listed_at='2000-01-01', delisted_at=end), '2026-05-15')
    assert not listed_on(SimpleNamespace(listed_at='2000-01-01', delisted_at='2026-05-15'), '2026-05-15')
    assert not listed_on(SimpleNamespace(listed_at='2026-05-16', delisted_at=''), '2026-05-15')
