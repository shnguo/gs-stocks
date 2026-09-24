from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research.price_strategy import (
    GRID,
    Plan,
    PlanCalibrator,
    TradeAssumptions,
    candidate_plans,
    choose_plan,
    forecast_metrics,
    ordered_quantiles,
    price_labels,
    trade_diagnostic,
)


def history():
    dates = pd.bdate_range("2024-01-01", periods=12).strftime("%Y-%m-%d").tolist()
    bars = pd.DataFrame({"instrument_id": "a", "date": dates, "open": 10.,
        "high": 11., "low": 9., "close": 10., "factor": 1., "volume": 100.,
        "amount": 1000., "upper_limit": 11., "lower_limit": 9.,
        "sequence_id": "a1", "label_sequence_id": "a1"})
    rows = pd.DataFrame({"date": [dates[0]], "date_index": [0], "instrument_id": ["a"]})
    actions = pd.DataFrame(columns=["instrument_id", "ex_date"])
    return dates, bars, rows, actions


def test_five_day_quotes_and_boundary():
    dates, bars, rows, actions = history()
    bars.loc[5, "close"] = 10.5
    result = price_labels(bars, rows, dates, actions, dates[6])
    assert result["targets"].shape == (1, 5, 4)
    assert result["targets"][0, 4, 3] == pytest.approx(np.log(1.05))
    with pytest.raises(ValueError, match="boundary"):
        price_labels(bars, rows, dates, actions, dates[5])
    # Tomorrow's prices change the label, never today's quote reference.
    bars.loc[5, "close"] = 10.8
    assert price_labels(bars, rows, dates, actions, dates[6])["reference"][0] == 10.


@pytest.mark.parametrize("barrier", ["action", "factor", "episode", "missing", "suspended"])
def test_unknown_future_preserved_not_zero(barrier):
    dates, bars, rows, actions = history()
    if barrier == "action":
        actions = pd.DataFrame({"instrument_id": ["a"], "ex_date": [dates[3]]})
    elif barrier == "factor":
        bars.loc[3:, "factor"] = 2.
    elif barrier == "episode":
        bars.loc[3:, "sequence_id"] = "a2"
    elif barrier == "missing":
        bars = bars.drop(index=3)
    else:
        bars.loc[3, "volume"] = 0
    result = price_labels(bars, rows, dates, actions, dates[6])
    assert result["valid"].tolist() == [[True, True, False, False, False]]
    assert np.isnan(result["targets"][0, 2:]).all()


def test_quantile_rearrangement_and_metrics():
    pred = np.zeros((2, 5, 4, 3))
    pred[:] = [.1, 0, -.1]
    q = ordered_quantiles(pred)
    assert (np.diff(q, axis=-1) >= 0).all()
    y = np.zeros((2, 5, 4))
    y[1, 4] = np.nan
    m = forecast_metrics(y, q)["targets"][-1]
    assert m["scored_rows"] == 1
    assert m["interval_80_coverage"] == 1
    assert m["pinball"] == pytest.approx(.02 / 3)


def test_missing_optional_trade_flag_does_not_remove_bse_labels():
    dates, bars, rows, actions = history()
    bars["source_trade_status"] = np.nan
    result = price_labels(bars, rows, dates, actions, dates[6])
    assert result["valid"].all()
    bars.loc[2, "source_trade_status"] = 0.
    result = price_labels(bars, rows, dates, actions, dates[6])
    assert result["valid"].tolist() == [[True, False, False, False, False]]


def path():
    return np.array([[9.9, 10.1, 9.8, 10.], [10., 10.1, 9.9, 10.],
                     [10., 10.1, 9.9, 10.], [10., 10.1, 9.9, 10.],
                     [10., 10.1, 9.9, 10.]])


def run(p, **kwargs):
    return trade_diagnostic(p, kwargs.pop("valid", np.ones(5, bool)),
                            Plan(10, 10.5, 9.5), TradeAssumptions(), **kwargs)


def test_buy_limit_and_expiry():
    p = path()
    p[0] = [10.2, 10.3, 10., 10.1]
    r = run(p)
    assert r["status"] == "unfilled"  # Mere low == bid is not a fill.
    p[1, 2] = 9.8
    assert run(p)["status"] == "unfilled"  # Do not carry expired buy orders.
    p[0, 2] = 9.9
    assert run(p)["buy_price"] == 10.


def test_t_plus_one_and_stop_first():
    p = path()
    p[0] = [9.9, 11., 9., 10.]
    assert run(p)["reason"] == "day5_close"
    p[1] = [10., 10.8, 9.2, 10.]
    r = run(p)
    assert r["reason"] == "stop" and r["ambiguous"]
    assert r["sell_price"] < 9.5
    p[1, 0] = 10.7
    assert run(p)["reason"] == "take_profit_at_open"


def test_gap_through_stop_and_costs():
    p = path()
    p[1] = [9., 9.1, 8.9, 9.]
    r = run(p)
    assert r["sell_price"] < 9.
    assert r["net_return"] < r["sell_price"] / r["buy_price"] - 1


def test_blocked_exit_and_unknown_outcomes():
    p = path()
    p[1:] = [9., 9., 9., 9.]
    r = run(p, upper=np.full(5, 11.), lower=np.full(5, 9.), require_limits=True)
    assert r["status"] == "unresolved" and r["filled"]
    assert r["net_return"] is None
    assert run(path(), require_limits=True)["reason"] == "price_limits_unverified"
    assert run(path(), valid=np.array([1, 1, 0, 0, 0]))["net_return"] is None


def test_grid_rounding_and_observe():
    plans = candidate_plans(10.123, TradeAssumptions())
    assert len(plans) == 12
    assert all(abs(p.buy * 100 - round(p.buy * 100)) < 1e-7 for p in plans)
    assert choose_plan([None] * len(GRID)) is None
    e = {"fill_probability": .8, "expected_net_per_order": -.01, "downside_per_order": -.02}
    assert choose_plan([e] * len(GRID)) is None


def test_conditional_calibration_distinguishes_unfilled_and_missing():
    p = np.zeros((40, 5, 4, 3))
    outcomes = [[{"net_return": .02 if i < 20 else 0., "filled": i < 20}
                 for _ in GRID] for i in range(40)]
    cal = PlanCalibrator(neighbors=40).fit(p, outcomes)
    e = cal.estimate(p[:1])[0][0]
    assert e["fill_probability"] == .5
    assert e["expected_net_if_filled"] == pytest.approx(.02)
    assert e["expected_net_per_order"] == pytest.approx(.01)
    unknown = [[{"net_return": None, "filled": None} for _ in GRID] for _ in range(40)]
    assert PlanCalibrator().fit(p, unknown).estimate(p[:1])[0][0] is None
    distant = p[:1].copy()
    distant[:, 4, 3, 1] = 1
    assert cal.estimate(distant)[0][0] is None
