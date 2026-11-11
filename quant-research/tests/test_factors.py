import json

import numpy as np
import pandas as pd
import pytest

from quant_research.factors import (
    composite,
    eligible,
    evaluate,
    forward_returns,
    group_percentile,
    load_valuation,
    momentum,
    month_end_dates,
    neutralize,
    performance,
    physics_factors,
    physics_momentum,
    price_panel,
    quantile_labels,
    rank_ic,
    residualize,
    reversal,
    value_factors,
)
from quant_research.storage import file_hash


def make_bars(days=300, stocks=4, start="2020-01-01"):
    dates = pd.bdate_range(start, periods=days).strftime("%Y-%m-%d")
    rows = []
    for s in range(stocks):
        for t, d in enumerate(dates):
            price = 10 * (1 + 0.001 * (s + 1)) ** t
            rows.append(dict(instrument_id=f"cn.xshg.60000{s}", date=d, open=price, high=price * 1.01,
                             low=price * 0.99, close=price, volume=1000.0, factor=1.0,
                             source_trade_status=1.0, source_is_st=0.0,
                             sequence_id=f"cn.xshg.60000{s}"))
    return pd.DataFrame(rows), list(dates)


def instruments(stocks=4, listed="2010-01-01", exchange="XSHG"):
    return pd.DataFrame(dict(instrument_id=[f"cn.xshg.60000{s}" for s in range(stocks)],
                             exchange=exchange, listed_at=listed))


def test_momentum_skips_last_month_and_uses_adjusted_prices():
    bars, dates = make_bars()
    bars.loc[bars.instrument_id == "cn.xshg.600000", "factor"] = 2.0  # constant factor: no effect
    panel = price_panel(bars)
    date = dates[260]
    value = momentum(panel, date, lookback=252, skip=21, min_traded=200)
    expected = (1.001 ** (260 - 21)) / (1.001 ** (260 - 252)) - 1
    assert value[0] == pytest.approx(expected)
    assert np.isnan(momentum(panel, dates[100], 252, 21, 200)).all()


def test_factor_adjustment_removes_dividend_gap():
    bars, dates = make_bars(days=40, stocks=1)
    raw = bars.copy()
    # A 10% cash dividend: raw prices drop, the backward factor rises to compensate.
    after = raw.date >= dates[20]
    for col in ["open", "high", "low", "close"]:
        raw.loc[after, col] = raw.loc[after, col] / 1.1
    raw.loc[after, "factor"] = 1.1
    adjusted = price_panel(raw)
    clean = price_panel(bars)
    assert reversal(adjusted, dates[30], 21, 15)[0] == pytest.approx(reversal(clean, dates[30], 21, 15)[0])


def test_returns_never_span_sequence_break():
    bars, dates = make_bars()
    later = (bars.instrument_id == "cn.xshg.600001") & (bars.date >= dates[200])
    bars.loc[later, "sequence_id"] = "cn.xshg.600001/action"
    panel = price_panel(bars)
    assert np.isnan(momentum(panel, dates[260], 252, 21, 200)[1])
    assert np.isfinite(momentum(panel, dates[260], 252, 21, 200)[0])
    _, reason = forward_returns(panel, dates[190], dates[210])
    assert reason[1] == 4 and reason[0] == 0


def test_eligibility_excludes_st_unknown_new_suspended_and_exchange():
    bars, dates = make_bars(stocks=4)
    date = dates[50]
    bars.loc[(bars.instrument_id == "cn.xshg.600000") & (bars.date == date), "source_is_st"] = 1.0
    bars.loc[(bars.instrument_id == "cn.xshg.600001") & (bars.date == date), "source_is_st"] = np.nan
    bars.loc[(bars.instrument_id == "cn.xshg.600002") & (bars.date == date), "volume"] = 0.0
    panel = price_panel(bars)
    meta = instruments()
    assert eligible(panel, date, meta, min_listing_days=365, exclude_exchanges=[]).tolist() == \
        [False, False, False, True]
    meta.loc[3, "listed_at"] = dates[0]
    assert not eligible(panel, date, meta, min_listing_days=365, exclude_exchanges=[])[3]
    assert not eligible(panel, date, instruments(exchange="XBSE"), min_listing_days=365,
                        exclude_exchanges=["XBSE"]).any()


def test_forward_return_next_open_entry_limit_up_and_delisting():
    bars, dates = make_bars(stocks=3)
    signal, following = dates[100], dates[120]
    entry, exit_ = dates[101], dates[121]
    # Stock 1 opens at a one-price limit up: not buyable.
    i1 = (bars.instrument_id == "cn.xshg.600001") & (bars.date == entry)
    bars.loc[i1, ["open", "high", "low", "close"]] = bars.loc[i1, "close"].iloc[0] * 1.1
    # Stock 2 stops trading after dates[110]: valued at its last traded close.
    gone = (bars.instrument_id == "cn.xshg.600002") & (bars.date > dates[110])
    bars = bars.loc[~gone]
    panel = price_panel(bars)
    returns, reason = forward_returns(panel, signal, following)
    price = lambda s, t: 10 * (1 + 0.001 * (s + 1)) ** t  # noqa: E731
    assert returns[0] == pytest.approx(price(0, 121) / price(0, 101) - 1)
    assert reason[1] == 2 and np.isnan(returns[1])
    assert returns[2] == pytest.approx(price(2, 110) / price(2, 101) - 1)
    assert entry in panel.dates and exit_ in panel.dates


def test_quantiles_rank_ic_and_evaluation():
    rng = np.random.default_rng(0)
    cohorts = []
    ids = np.array([f"s{i}" for i in range(500)])
    for m in range(24):
        f = rng.normal(size=500)
        r = 0.02 * f + rng.normal(scale=0.05, size=500)
        cohorts.append({"date": f"2020-{m:02d}", "ids": ids, "f": f, "returns": r})
    labels = quantile_labels(cohorts[0]["f"], cohorts[0]["returns"], 5)
    assert np.bincount(labels)[1:].tolist() == [100] * 5
    assert cohorts[0]["f"][labels == 5].min() > cohorts[0]["f"][labels == 4].max()
    out = evaluate(cohorts, "f", 5, 0.0015, 6, 200)
    assert out["rank_ic_mean"] > 0.2 and out["rank_ic_positive_share"] == 1
    q = out["quantile_mean_monthly"]
    assert q["q5"] > q["q3"] > q["q1"]
    assert out["long_short_net"]["mean_monthly"] < out["long_short"]["mean_monthly"]
    assert out["periods"] == 24 and len(out["daily"]) == 24
    assert rank_ic(np.arange(100.0), np.arange(100.0)) == pytest.approx(1)


def test_composite_and_neutralize():
    mask = np.ones(6, bool)
    a = np.array([1, 2, 3, 4, 5, np.nan])
    b = np.array([5, 4, 3, 2, 1, np.nan])
    c = composite([a, b], mask, 2)
    assert np.isnan(c[5]) and np.allclose(c[:5], c[0])
    rng = np.random.default_rng(1)
    industry = np.array(["bank", "tech"] * 100, dtype=object)
    size = rng.normal(size=200)
    values = (industry == "bank") * 5 + 2 * size + rng.normal(scale=0.1, size=200)
    resid = neutralize(values, industry, size)
    assert abs(np.corrcoef(resid, pd.Series(size).rank())[0, 1]) < 1e-9
    assert abs(resid[industry == "bank"].mean() - resid[industry == "tech"].mean()) < 1e-9
    industry[:10] = None
    assert np.isnan(neutralize(values, industry, size)[:10]).all()


def test_value_factors_require_matching_close():
    bars, dates = make_bars(stocks=2)
    panel = price_panel(bars)
    date = dates[10]
    close = bars.loc[bars.date == date].set_index("instrument_id").close
    valuation = pd.DataFrame(dict(
        instrument_id=["cn.xshg.600000", "cn.xshg.600001"], date=date,
        close=[close["cn.xshg.600000"], close["cn.xshg.600001"] + 1],
        pe_ttm=[10.0, 5.0], pb=[2.0, 1.0], ps_ttm=[0.0, 1.0], pcf_ttm=[-4.0, 1.0],
        float_cap=[1e9, 1e9]))
    v = value_factors(panel, date, valuation)
    assert v.loc["cn.xshg.600000", "earnings_yield"] == pytest.approx(0.1)
    assert np.isnan(v.loc["cn.xshg.600000", "sales_yield"])
    assert v.loc["cn.xshg.600000", "cashflow_yield"] == pytest.approx(-0.25)
    assert v.loc["cn.xshg.600001"].drop("valuation_matched").isna().all()


def test_load_valuation_checks_hashes_and_seal(tmp_path):
    capture = tmp_path / "captures" / "d1"
    capture.mkdir(parents=True)
    rows = [dict(instrument_id="cn.xshg.600000", date="2020-01-02", close=1.0, pe_ttm=1.0)]
    (capture / "records.json").write_text(json.dumps(rows))
    manifest = dict(status="completed", files={"records.json": file_hash(capture / "records.json")})
    (capture / "manifest.json").write_text(json.dumps(manifest))
    index = {"2020-01-02": dict(capture="captures/d1", rows=1,
                                capture_sha256=file_hash(capture / "manifest.json")),
             "2030-01-02": dict(capture="missing", rows=1, capture_sha256="x")}
    (tmp_path / "index.json").write_text(json.dumps(index))
    pruned = tmp_path / "captures" / "d0"
    pruned.mkdir()
    (pruned / "manifest.json").write_text("{}")
    index["2020-01-01"] = dict(capture="captures/d0", rows=1, capture_sha256="y")
    (tmp_path / "index.json").write_text(json.dumps(index))
    frame, lineage, missing = load_valuation(tmp_path, "2025-08-07")
    assert len(frame) == 1 and len(lineage) == 1 and missing == ["2020-01-01"]
    (capture / "records.json").write_text(json.dumps(rows * 2))
    with pytest.raises(ValueError):
        load_valuation(tmp_path, "2025-08-07")


def test_month_end_and_performance():
    assert month_end_dates(["2020-01-02", "2020-01-31", "2020-02-03", "2020-02-28"]) == \
        ["2020-01-31", "2020-02-28"]
    p = performance(np.array([0.1, -0.5, 0.2]))
    assert p["max_drawdown"] == pytest.approx(-0.5)
    assert p["annual_return"] == pytest.approx((1.1 * 0.5 * 1.2) ** 4 - 1)


def test_physics_momentum_splits_jumps_from_drift_and_measures_mass():
    rng = np.random.default_rng(7)
    dates = list(pd.bdate_range("2020-01-01", periods=300).strftime("%Y-%m-%d"))
    steady = rng.normal(0.001, 0.01, 300)
    jumpy = rng.normal(0.0, 0.01, 300)
    jumpy[120] = 0.3
    rows = []
    for name, r, volume in [("a", steady, 1000.0), ("b", jumpy, 1000.0), ("c", steady, 10000.0)]:
        price = 10 * np.exp(np.cumsum(r))
        rows += [dict(instrument_id=name, date=d, open=p, high=p, low=p, close=p, volume=volume,
                      factor=1.0, sequence_id=name) for d, p in zip(dates, price)]
    panel = price_panel(pd.DataFrame(rows))
    out = physics_momentum(panel, dates[260], 252, 21, 200)
    window = np.log(panel.close[8:240] / panel.close[8:240][0])[-1]
    assert out["total"] == pytest.approx(window)
    assert abs(out["impulse"][0]) < 0.05 and out["impulse"][1] == pytest.approx(0.3, abs=0.03)
    assert out["drift"] == pytest.approx(out["total"] - out["impulse"])
    assert out["mass"][2] == pytest.approx(10 * out["mass"][0])
    daily = np.diff(np.log(panel.close[8:240, 1]))
    ret = np.expm1(out["total"][1])
    assert out["smooth"][1] == pytest.approx(ret * (1 - np.sign(ret) * ((daily < 0).mean() - (daily > 0).mean())))
    mask = np.ones(3, dtype=bool)
    f = physics_factors(out, mask, np.expm1(out["total"]), np.zeros(3))
    # Same velocity, ten times the mass: a larger move in the same direction.
    assert np.sign(f["momentum_mass"][2]) == np.sign(f["momentum_mass"][0])
    assert abs(f["momentum_mass"][2]) > abs(f["momentum_mass"][0])
    assert np.isnan(physics_momentum(panel, dates[100])["drift"]).all()


def test_residualize_removes_control_and_keeps_independent_part():
    rng = np.random.default_rng(3)
    control, extra = rng.normal(size=400), rng.normal(size=400)
    mask = np.ones(400, dtype=bool)
    mask[:10] = False
    out = residualize(control + extra, [control], mask)
    assert np.isnan(out[:10]).all()
    assert abs(rank_ic(out, control)) < 0.1 and rank_ic(out, extra) > 0.5


def test_group_percentile_and_industry_residual():
    values = np.array([1.0, 2.0, 3.0, 10.0, 20.0, np.nan])
    groups = np.array(["a", "a", "a", "b", "b", "b"], dtype=object)
    out = group_percentile(values, groups, np.ones(6, dtype=bool))
    assert out[:5].tolist() == pytest.approx([1 / 3, 2 / 3, 1.0, 0.5, 1.0]) and np.isnan(out[5])
    rng = np.random.default_rng(5)
    g = np.repeat(np.array(["x", "y"], dtype=object), 200)
    v = np.where(g == "x", 5.0, 0.0) + rng.normal(size=400)
    res = residualize(v, [], np.ones(400, dtype=bool), g)
    assert abs(res[g == "x"].mean() - res[g == "y"].mean()) < 1e-9
