import numpy as np
import pandas as pd
import pytest

from quant_research.token_indicators import FEATURES, indicator_features


def bars(length=60, n=2):
    rng = np.random.default_rng(19)
    c = 10*np.exp(rng.normal(0, .02, (n, length)).cumsum(1))
    out = np.ones((n, length, 7), np.float64)
    out[..., 0] = out[..., 3] = c
    out[..., 1] = c*1.03
    out[..., 2] = c*.97
    out[..., 4:6] = 100
    return out


def independent(raw):
    raw = np.asarray(raw, dtype=np.float64)
    p = raw[:, :4]*raw[:, 6:7]
    c, h, low = pd.Series(p[:, 3]), pd.Series(p[:, 1]), pd.Series(p[:, 2])
    mean, std = c.rolling(20).mean(), c.rolling(20).std(ddof=0)
    dif = c.ewm(span=12, adjust=False, min_periods=12).mean()-c.ewm(span=26, adjust=False, min_periods=26).mean()
    dea = dif.ewm(span=9, adjust=False, min_periods=9).mean()
    hi, lo = h.rolling(9).max(), low.rolling(9).min()
    rsv = (c-lo)/(hi-lo)
    k = d = .5
    ks, ds = [], []
    for r in rsv:
        if not np.isfinite(r):
            ks.append(np.nan)
            ds.append(np.nan)
            k = d = .5
        else:
            k = (2*k+r)/3
            d = (2*d+k)/3
            ks.append(k)
            ds.append(d)
    return np.column_stack([(c-(mean-2*std))/(4*std),4*std/mean,dif/c,(dif-dea)/c,ks,ds])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_indicator_formulas_match_independent_pandas_reference(dtype):
    raw = bars().astype(dtype)
    actual = indicator_features(raw)
    assert actual.shape == (2, 60, len(FEATURES))
    for i in range(2):
        np.testing.assert_allclose(actual[i], independent(raw[i]), equal_nan=True, atol=1e-7, rtol=1e-6)
    assert np.isnan(actual[:, :19, :2]).all()
    assert np.isnan(actual[:, :25, 2]).all()
    assert np.isnan(actual[:, :33, 3]).all()
    assert np.isnan(actual[:, :8, 4:]).all()
    assert np.isfinite(actual[:, 33:]).all()


def test_causal_prefix_and_price_adjustment_invariance():
    raw = bars()
    before = indicator_features(raw)
    changed = raw.copy()
    changed[:, 40:, :4] *= 2
    changed[:, 40:, 6] = 3
    np.testing.assert_array_equal(before[:, :40], indicator_features(changed)[:, :40])
    np.testing.assert_array_equal(before[:, :40], indicator_features(raw[:, :40]))
    # Synthetic 2-for-1 split, perfectly offset by the historical factor.
    split = raw.copy()
    split[:, :30, :4] *= 2
    split[:, 30:, 6] *= 2
    np.testing.assert_allclose(before, indicator_features(split), equal_nan=True, rtol=1e-6, atol=1e-7)
    scaled = raw.copy()
    scaled[..., :4] *= 100
    np.testing.assert_allclose(before, indicator_features(scaled), equal_nan=True, rtol=1e-6, atol=1e-7)


def test_missing_calendar_bar_resets_recursion_and_rolling_warmup():
    raw = bars(90)
    raw[:, 35] = np.nan
    x = indicator_features(raw)
    assert np.isnan(x[:, 35]).all()
    assert np.isnan(x[:, 36:55, :2]).all()
    assert np.isnan(x[:, 36:61, 2]).all()
    assert np.isnan(x[:, 36:69, 3]).all()
    assert np.isnan(x[:, 36:44, 4:]).all()
    np.testing.assert_array_equal(x[:, 36:], indicator_features(raw[:, 36:]))


def test_flat_market_undefined_ratios_masked_without_infinity():
    raw = bars()
    raw[..., :4] = 10
    x = indicator_features(raw)
    assert not np.isinf(x).any()
    assert np.isnan(x[..., 0]).all()
    assert np.isnan(x[..., 4:]).all()
    np.testing.assert_array_equal(x[:, 19:, 1], 0)
    np.testing.assert_array_equal(x[:, 33:, 2:4], 0)
    with pytest.raises(ValueError, match='OHLCVA'):
        indicator_features(raw[..., :6])
    raw[0, 0, 3] = np.inf
    with pytest.raises(ValueError, match='Infinite'):
        indicator_features(raw)


def test_inference_adapter_uses_checkpoint_feature_order():
    from test_token_transformer import DummyTokenizer, tiny

    from quant_research.kronos_ranker import timestamps
    from quant_research.token_indicators import BOLL, predict_indicator_paths
    from quant_research.token_transformer import predict_paths, with_auxiliary

    raw = bars(12)
    base = tiny()
    model = with_auxiliary(base, BOLL, [0., 0.], [1., 1.])
    stamps = timestamps(pd.bdate_range('2023-01-02', periods=15))[None].repeat(2, 0)
    actual = predict_indicator_paths(model, DummyTokenizer(), raw, stamps[:, :12], stamps[:, 12:], samples=2)
    expected = predict_paths(model, DummyTokenizer(), raw, stamps[:, :12], stamps[:, 12:],
        samples=2, history_auxiliary=indicator_features(raw)[..., :2])
    for k in actual:
        np.testing.assert_array_equal(actual[k], expected[k])
    assert actual['paths'].shape == (2, 2, 3, 6)
