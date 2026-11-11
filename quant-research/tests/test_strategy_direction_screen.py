import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from compare_strategy_directions import features, fees, simulate  # noqa: E402


def sample():
    dates = pd.bdate_range("2022-01-03", periods=70).strftime("%Y-%m-%d").tolist()
    names = [f"cn.xshe.{i:06d}" for i in range(1, 13)]
    shape = (len(dates), len(names))
    arrays = {k: np.full(shape, 10.) for k in ["open", "high", "low", "close"]}
    arrays.update(volume=np.full(shape, 1e8), amount=np.full(shape, 1e9), factor=np.ones(shape),
                  source_is_st=np.zeros(shape), source_trade_status=np.ones(shape),
                  label_sequence_id=np.zeros(shape, dtype=int), sequence_id=np.zeros(shape, dtype=int), age=np.full(shape, 400))
    instruments = pd.DataFrame({"instrument_id": names, "delisted_at": [""]*len(names)})
    protocol = dict(initial_cash=200000., evaluation_start=dates[1], rebalance_sessions=20, retention_buffer_rank=12,
                    max_positions=6, position_budget_fraction=.15, adv_participation=.005,
                    minimum_history_sessions=252, minimum_adv20=30000000.)
    scenario = dict(name="base", rebalance_offset=0, delay_sessions=0, slippage_bps=10, commission_multiplier=1)
    scores = np.tile(np.arange(12, 0, -1), (len(dates), 1)).astype(float)
    return dates, names, instruments, arrays, protocol, scenario, scores


def test_costs_are_per_order_and_change_with_historical_tax_date():
    assert fees(1000., "2021-09-01", "buy") == (5., .02, 0.)
    assert fees(20000., "2021-09-01", "sell") == (10., .4, 20.)
    assert fees(20000., "2024-09-01", "sell") == (10., .2, 10.)


def test_constant_prices_lose_exactly_fees_and_slippage_and_never_borrow(tmp_path):
    dates, names, inst, arrays, cfg, scenario, scores = sample()
    result, nav = simulate(dates, names, inst, arrays, {}, scores, arrays["amount"], cfg, scenario, "test", tmp_path/"constant")
    fills = pd.read_parquet(tmp_path/"constant/fills.parquet")
    assert len(fills) == 6
    assert (fills.shares % 100 == 0).all()
    expected = cfg["initial_cash"]-fills.fees.sum()-fills.slippage.sum()
    assert abs(nav.equity.iloc[-1]-expected) < 1e-7
    assert nav.cash.min() >= 0 and nav.positions.max() <= 6 and result["net_return"] < 0


def test_split_and_unpaid_dividend_preserve_wealth_without_spending_receivable(tmp_path):
    dates, names, inst, arrays, cfg, scenario, scores = sample()
    # New shares and the cash entitlement arise before day-two orders.
    for k in ["open", "high", "low", "close"]:
        arrays[k][2:, 0] = 4.95
    actions = {2: [(0, 2., .1)]}
    scenario["dividend_delay_days"] = 10000
    _, nav = simulate(dates, names, inst, arrays, actions, scores, arrays["amount"], cfg, scenario, "test", tmp_path/"actions")
    fills = pd.read_parquet(tmp_path/"actions/fills.parquet")
    q = fills.loc[fills.instrument_id == names[0], "shares"].iloc[0]
    assert abs(nav.equity.iloc[1]-nav.equity.iloc[0]+q*.02) < 1e-7
    assert abs(nav.receivable.iloc[-1]-q*.08) < 1e-7
    assert nav.cash.iloc[-1] == nav.cash.iloc[0]


def test_blocked_sell_remains_valued_and_future_prices_do_not_change_past_scores(tmp_path):
    dates, names, inst, arrays, cfg, scenario, scores = sample()
    # Remove the original six from the ranking; a suspended holding cannot disappear.
    scores[20:, :6] = -np.inf
    arrays["source_trade_status"][21:, 0] = 0
    result, nav = simulate(dates, names, inst, arrays, {}, scores, arrays["amount"], cfg, scenario, "test", tmp_path/"blocked")
    held = pd.read_parquet(tmp_path/"blocked/holdings.parquet")
    assert names[0] in held.loc[held.date == dates[-1], "instrument_id"].tolist()
    assert result["blocked_reasons"]["no_verified_trading_bar"] > 0 and nav.positions.max() <= 6
    # Extend synthetic feature history, then perturb only its suffix.
    extended = {k: np.tile(v, (4, 1)) for k, v in arrays.items()}
    extended["close"] *= np.linspace(1, 1.5, len(extended["close"]))[:, None]
    a, _ = features(extended, cfg)
    extended["close"][230:] *= 10
    b, _ = features(extended, cfg)
    for key in a:
        np.testing.assert_array_equal(a[key][:230], b[key][:230])


def test_delay_uses_same_original_signal_and_barrier_on_exit_is_retained(tmp_path):
    dates, names, inst, arrays, cfg, scenario, scores = sample()
    scenario["delay_sessions"] = 1
    arrays["label_sequence_id"][22:, 0] = 1
    scores[20:, :6] = -np.inf
    result, _ = simulate(dates, names, inst, arrays, {}, scores, arrays["amount"], cfg, scenario, "test", tmp_path/"delay")
    fills = pd.read_parquet(tmp_path/"delay/fills.parquet")
    assert fills.date.iloc[0] == dates[2] and fills.signal_date.iloc[0] == dates[0]
    assert result["unresolved_barrier_events"] == 1
