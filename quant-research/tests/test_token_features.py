from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from quant_research.kronos_ranker import timestamps
from quant_research.token_features import FEATURES, daily_features, fit_normalizer, history_features
from quant_research.token_history import loss_parts
from quant_research.token_transformer import (
    TokenConfig,
    TokenTransformer,
    checkpoint_payload,
    forecast_auxiliary,
    generate_tokens,
    restore_model,
    with_auxiliary,
)


def fixture():
    days = pd.bdate_range('2023-01-02', periods=30).strftime('%Y-%m-%d').tolist()
    bars = pd.DataFrame(dict(instrument_id='cn.xshg.600000', date=days, close=10., volume=100.))
    source = bars.drop(columns='volume').assign(float_shares=1000., pe_ttm=20.)
    return days, bars, source


def test_exact_calendar_changes_no_sparse_fill_and_price_or_denominator_mismatch():
    days, bars, source = fixture()
    source.loc[5, 'pe_ttm'] = 10.
    source.loc[10, 'float_shares'] = 500.
    features = daily_features(bars, source, days, retrospective=True)
    assert features.loc[5, 'earnings_yield_change_5'] == pytest.approx(.05)
    assert features.loc[10, 'turnover_change_5'] == 10
    assert features.loc[10, 'turnover_relative_5'] == pytest.approx(20/12-1)
    sparse = daily_features(bars, source.drop(index=5), days, retrospective=True)
    assert np.isnan(sparse.loc[10, 'turnover_change_5'])
    assert np.isnan(sparse.loc[9, 'turnover_relative_5'])
    source.loc[12, 'close'] = 11
    source.loc[13, 'float_shares'] = 0
    source.loc[14, 'pe_ttm'] = 0
    source.loc[15, 'pe_ttm'] = -10
    features = daily_features(bars, source, days, retrospective=True)
    assert features.loc[12, list(FEATURES)].isna().all()
    assert np.isnan(features.loc[13, 'log_turnover_pct'])
    assert np.isnan(features.loc[14, 'earnings_yield'])
    assert features.loc[15, 'negative_pe'] == 1
    assert features.loc[15, 'earnings_yield'] == pytest.approx(-.1)


def test_asof_guard_history_alignment_and_future_source_invariance():
    days, bars, source = fixture()
    with pytest.raises(ValueError, match='available_at'):
        daily_features(bars, source, days)
    source['available_at'] = source.date + 'T08:00:00'
    with pytest.raises(ValueError, match='timezone'):
        daily_features(bars, source, days)
    source['available_at'] = source.date + 'T08:00:00Z'
    source.loc[20, 'available_at'] = days[20] + 'T09:00:00Z'
    features = daily_features(bars, source, days)
    assert features.loc[20, list(FEATURES)].isna().all()
    rows = bars.iloc[[19]][['instrument_id', 'date']]
    a = history_features(rows, features, days, 10)
    source.loc[20:, 'pe_ttm'] = 1000.
    b = history_features(rows, daily_features(bars, source, days), days, 10)
    np.testing.assert_array_equal(a, b)
    center, scale = fit_normalizer(a)
    assert np.isfinite(center).all() and (scale > 0).all()


def setup_model():
    torch.manual_seed(17)
    base = TokenTransformer(TokenConfig(width=16, layers=1, heads=4, s1_bits=3, s2_bits=3, max_context=16, dropout=0)).eval()
    model = with_auxiliary(base, ['turnover', 'ep'], [0., 0.], [1., 1.]).eval()
    a, b = torch.randint(0, 8, (2, 8)), torch.randint(0, 8, (2, 8))
    stamps = torch.from_numpy(timestamps(pd.bdate_range('2023-01-02', periods=8)))[None].expand(2, -1, -1)
    features = torch.randn(2, 5, 2)
    return base, model, a, b, stamps, features


def test_exact_warm_start_gradient_reload_and_feature_sensitivity(tmp_path):
    base, model, a, b, stamps, features = setup_model()
    padded = forecast_auxiliary(features, 5, 7)
    for x, y in zip(base.forecast_logits(a[:, :-1], b[:, :-1], stamps[:, :-1], a[:, 1:], 4),
                    model.forecast_logits(a[:, :-1], b[:, :-1], stamps[:, :-1], a[:, 1:], 4, auxiliary=padded)):
        torch.testing.assert_close(x, y, atol=0, rtol=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    loss_parts(model, a, b, stamps, torch.ones(2, 3, dtype=torch.bool), 5, history_auxiliary=features).mean().backward()
    assert model.auxiliary_projection[-1].weight.grad.abs().sum() > 0
    optimizer.step()
    before = model.forecast_logits(a[:, :-1], b[:, :-1], stamps[:, :-1], a[:, 1:], 4, auxiliary=padded)
    changed = model.forecast_logits(a[:, :-1], b[:, :-1], stamps[:, :-1], a[:, 1:], 4, auxiliary=padded + 10)
    assert not torch.allclose(before[0], changed[0])
    torch.save(checkpoint_payload(model), tmp_path/'model.pt')
    restored, _ = restore_model(tmp_path/'model.pt')
    restored.eval()
    after = restored.forecast_logits(a[:, :-1], b[:, :-1], stamps[:, :-1], a[:, 1:], 4, auxiliary=padded)
    for x, y in zip(before, after):
        torch.testing.assert_close(x, y, atol=0, rtol=0)
    with pytest.raises(ValueError, match='required'):
        restored.decode_s1(a, b, stamps)


def test_future_auxiliary_rejected_generation_masks_and_causality():
    base, model, a, b, stamps, features = setup_model()
    with pytest.raises(ValueError, match='lookback'):
        loss_parts(model, a, b, stamps, torch.ones(2, 3, dtype=torch.bool), 5,
                   history_auxiliary=torch.randn(2, 8, 2))
    torch.nn.init.normal_(model.auxiliary_projection[-1].weight)
    full = torch.randn(2, 8, 2)
    first = model.decode_s1(a, b, stamps, full)[0]
    changed = full.clone()
    changed[:, 4:] += 100
    second = model.decode_s1(a, b, stamps, changed)[0]
    torch.testing.assert_close(first[:, :4], second[:, :4])
    seen = []
    handle = model.auxiliary_projection.register_forward_pre_hook(lambda m, args: seen.append(args[0].detach().clone()))
    pairs = generate_tokens(model, a[:, :5], b[:, :5], stamps[:, :5], stamps[:, 5:],
                            samples=2, history_auxiliary=features)
    handle.remove()
    assert pairs[0].shape == (2, 2, 8)
    assert len(seen) == 3
    assert seen[-1][:, 5:, :].eq(0).all()  # Values AND availability flags are zero.
    with pytest.raises(ValueError, match='Baseline'):
        base.decode_s1(a, b, stamps, full)
    with pytest.raises(ValueError, match='Invalid auxiliary'):
        TokenTransformer(replace(base.config, auxiliary_features=('a', 'a')))


def test_feature_rollout_matches_teacher_forced_prefix_with_context_truncation():
    torch.manual_seed(11)
    model = TokenTransformer(TokenConfig(width=16, layers=1, heads=4, max_context=5,
        s1_bits=3, s2_bits=3, dropout=0, auxiliary_features=("ep",))).eval()
    torch.nn.init.normal_(model.auxiliary_projection[-1].weight)
    a, b = torch.randint(0, 8, (2, 5)), torch.randint(0, 8, (2, 5))
    stamps = torch.from_numpy(timestamps(pd.bdate_range('2023-01-02', periods=8)))[None].expand(2, -1, -1)
    history = torch.randn(2, 5, 1)
    observed = []
    original = model.decode_s1
    def record(s1, s2, time, auxiliary=None):
        value = original(s1, s2, time, auxiliary)
        observed.append((s1.clone(), s2.clone(), time.clone(), auxiliary.clone(), value[0].clone()))
        return value
    model.decode_s1 = record
    generate_tokens(model, a, b, stamps[:, :5], stamps[:, 5:], samples=1, history_auxiliary=history)
    for step, (s1, s2, time, auxiliary, logits) in enumerate(observed):
        assert s1.shape[1] == 5
        expected = torch.cat([history, history.new_full((2, step, 1), torch.nan)], 1)[:, -5:]
        torch.testing.assert_close(auxiliary, expected, equal_nan=True)
        with torch.inference_mode():
            torch.testing.assert_close(logits, original(s1, s2, time, expected)[0])


def test_raw_ohlcva_predict_interface_accepts_numpy_features():
    from test_token_transformer import DummyTokenizer, raw_inputs

    from quant_research.token_transformer import predict_paths

    base, _, _, _, _, _ = setup_model()
    model = with_auxiliary(base, ["ep"], [0.], [1.])
    raw = raw_inputs(length=6)
    stamps = timestamps(pd.bdate_range("2023-01-02", periods=11))[None].repeat(2, 0)
    values = np.full((2, 6, 1), .05, np.float32)
    result = predict_paths(model, DummyTokenizer(), raw, stamps[:, :6], stamps[:, 6:],
                           samples=2, history_auxiliary=values)
    assert result['paths'].shape == (2, 2, 5, 6)
