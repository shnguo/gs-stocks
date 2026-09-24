import numpy as np
import pandas as pd
import pytest

from quant_research.price_stability import (
    aggregate_dates,
    date_metrics,
    doubled_costs,
    stress_selected_orders,
)
from quant_research.price_strategy import GRID, PlanCalibrator, TradeAssumptions


def test_constant_forecast_uses_all_tied_calibration_outcomes():
    q = np.zeros((100, 5, 4, 3))
    outcomes = [[{"net_return": -.1 if i < 32 else .1, "filled": True}
                 for _ in GRID] for i in range(100)]
    e = PlanCalibrator().fit(q, outcomes).estimate(q[:1])[0][0]
    assert e["neighbors"] == 100
    assert e["expected_net_if_filled"] == pytest.approx(.036)
    # Input order cannot make a constant baseline pick another outcome subset.
    r = PlanCalibrator().fit(q, outcomes[::-1]).estimate(q[:1])[0][0]
    assert e["expected_net_if_filled"] == pytest.approx(r["expected_net_if_filled"])


def test_daily_weighting_and_model_cohort_are_explicit():
    rows = pd.DataFrame({"date": ["a"] * 100 + ["b"]})
    y = np.zeros((101, 5, 4))
    y[:100] = 1
    naive = np.zeros((101, 5, 4, 3))
    kronos = naive.copy()
    kronos[1:100] = np.nan
    daily = date_metrics(rows, y, {"naive": naive, "kronos": kronos}, ["a"])
    summaries = aggregate_dates(daily)
    s = next(v for v in summaries if v["scope"] == "full" and v["model"] == "naive"
             and v["subset"] == "all_dates")
    assert s["mean_daily_pinball"] == pytest.approx(.25)
    common = daily.loc[daily.scope == "common"]
    assert common.groupby("model").rows.sum().to_dict() == {"naive": 2, "kronos": 2}
    s = next(v for v in summaries if v["scope"] == "full" and v["model"] == "kronos")
    assert s["paired_relative_loss_reduction"] is None
    assert all(v["dates"] == 1 for v in summaries if v["subset"] == "additional_price_dates")


def test_cost_stress_freezes_decisions_and_doubles_only_costs():
    a = TradeAssumptions()
    b = doubled_costs(a)
    assert b.notional == a.notional and b.tick == a.tick and b.t_plus == a.t_plus
    assert b.commission == 2 * a.commission and b.slippage_bps == 2 * a.slippage_bps
    rows = pd.DataFrame({"date": ["d"], "instrument_id": ["a"]})
    plans = rows.assign(model="lightgbm", signal="research_candidate", buy=10.,
                        take_profit=10.5, stop=9.5)
    labels = {"future": np.array([[[9.9, 10.1, 9.8, 10.]] * 5]),
              "valid": np.ones((1, 5), bool), "upper": np.full((1, 5), np.nan),
              "lower": np.full((1, 5), np.nan)}
    stressed = stress_selected_orders(plans, rows, labels, a)
    pd.testing.assert_frame_equal(stressed[plans.columns], plans)
    from quant_research.price_strategy import Plan, trade_diagnostic
    base = trade_diagnostic(labels["future"][0], labels["valid"][0], Plan(10., 10.5, 9.5), a)
    assert stressed.stress_net_return.iloc[0] < base["net_return"]
    labels["valid"][0, 0] = False
    unknown = stress_selected_orders(plans, rows, labels, a)
    assert pd.isna(unknown.stress_filled.iloc[0])
    assert pd.isna(unknown.stress_net_return.iloc[0])
    assert unknown.stress_status.iloc[0] == "unresolved"
