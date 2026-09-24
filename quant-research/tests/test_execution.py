import pandas as pd
import pytest

from quant_research.backtest import simulate


def predictions(snapshot, stock="synthetic.xshg.000000", index=325):
    return pd.DataFrame([dict(date=snapshot.dates[index], instrument_id=stock, score=1.0)])


def test_new_st_exits_on_first_unblocked_open_and_stays_valued(snapshot, config):
    stock = "synthetic.xshg.000000"
    bars = snapshot.tables["bars"]
    for date in snapshot.dates[330:332]:
        mask = (bars.date == date) & (bars.instrument_id == stock)
        bars.loc[mask, "lower_limit"] = bars.loc[mask, "open"]
    result = simulate(snapshot, predictions(snapshot), 20, config["portfolio"])
    sells = result["fills"].loc[result["fills"].side == "sell"]
    assert sells.iloc[0].date == snapshot.dates[332]
    assert sells.iloc[0].reason == "risk_warning_exit"
    assert set(result["holdings"].date) >= set(snapshot.dates[330:332])
    assert sum(result["orders"].status == "blocked") >= 2


def test_st_at_execution_blocks_a_previously_normal_signal(snapshot, config):
    result = simulate(snapshot, predictions(snapshot, index=329), 5, config["portfolio"])
    assert result["fills"].empty
    assert "risk_status_at_execution" in set(result["orders"].block_reason)


def test_suspension_does_not_make_position_disappear(snapshot, config):
    stock = "synthetic.xshe.000001"
    bars = snapshot.tables["bars"]
    mask = (bars.date == snapshot.dates[331]) & (bars.instrument_id == stock)
    snapshot.tables["bars"] = bars.loc[~mask]
    result = simulate(snapshot, predictions(snapshot, stock), 5, config["portfolio"])
    row = result["holdings"].loc[result["holdings"].date == snapshot.dates[331]].iloc[0]
    assert row.shares > 0 and row.stale_mark
    assert result["summary"]["terminal_positions"] == 1


def test_t_plus_rule_blocks_early_sale(snapshot, config):
    snapshot.tables["rules"].loc[:, "t_plus"] = 6
    result = simulate(snapshot, predictions(snapshot, "synthetic.xshe.000001"), 5,
                      config["portfolio"])
    assert "t_plus_settlement" in set(result["orders"].get("block_reason", []))
    assert result["summary"]["terminal_positions"] == 1


def test_cash_fees_lots_and_volume_caps_reconcile(snapshot, config):
    result = simulate(snapshot, predictions(snapshot, "synthetic.xshe.000001"), 5,
                      config["portfolio"])
    fills = result["fills"]
    buys = fills.loc[fills.side == "buy"]
    sells = fills.loc[fills.side == "sell"]
    expected = config["portfolio"]["initial_cash"] - buys.notional.sum() + sells.notional.sum() - fills.fees.sum()
    assert result["nav"].cash.iloc[-1] == pytest.approx(expected)
    assert (buys.shares % 100 == 0).all()
    assert result["nav"].cash.min() >= 0
    assert buys.notional.max() <= config["portfolio"]["initial_cash"] * config["portfolio"]["max_weight"]


def test_split_keeps_shares_and_value_consistent(snapshot, config):
    result = simulate(snapshot, predictions(snapshot, "synthetic.xbse.000002", 333), 5,
                      config["portfolio"])
    holdings = result["holdings"].set_index("date")
    before, after = holdings.loc[snapshot.dates[334]], holdings.loc[snapshot.dates[335]]
    assert after.shares == 2 * before.shares
    assert after.shares * after.mark_price == pytest.approx(before.shares * before.mark_price,
                                                          rel=0.05)


def test_cost_and_delay_stress_actually_change_execution(snapshot, config):
    scores = predictions(snapshot, "synthetic.xshe.000001")
    base = simulate(snapshot, scores, 5, config["portfolio"])
    expensive = simulate(snapshot, scores, 5, config["portfolio"], cost_multiplier=2)
    delayed = simulate(snapshot, scores, 5, config["portfolio"], delay_days=1)
    assert expensive["summary"]["fees"] > base["summary"]["fees"]
    assert delayed["fills"].iloc[0].date == snapshot.dates[327]


def test_dividend_is_receivable_until_payment_day(snapshot, config):
    stock = "synthetic.xshe.000001"
    snapshot.tables["actions"] = pd.DataFrame([dict(instrument_id=stock,
        ex_date=snapshot.dates[328], pay_date=snapshot.dates[330], split_ratio=1.0,
        cash_per_share=1.0)])
    bars = snapshot.tables["bars"]
    selected = (bars.instrument_id == stock) & (bars.date >= snapshot.dates[328])
    bars.loc[selected, ["open", "high", "low", "close", "upper_limit", "lower_limit"]] -= 1.0
    result = simulate(snapshot, predictions(snapshot, stock), 5, config["portfolio"])
    nav = result["nav"].set_index("date")
    dividend = result["fills"].iloc[0].shares
    assert nav.loc[snapshot.dates[328], "receivables"] == dividend
    assert nav.loc[snapshot.dates[329], "receivables"] == dividend
    assert nav.loc[snapshot.dates[330], "receivables"] == 0
    assert nav.loc[snapshot.dates[330], "cash"] - nav.loc[snapshot.dates[329], "cash"] == pytest.approx(dividend)


def test_rank_order_changes_the_selected_portfolio(snapshot, config):
    config["portfolio"]["max_positions"] = 1
    stocks = snapshot.tables["instruments"].instrument_id.tolist()
    scores = pd.DataFrame({"date": snapshot.dates[320], "instrument_id": stocks,
                           "score": list(range(len(stocks)))})
    first = simulate(snapshot, scores, 5, config["portfolio"])
    scores["score"] *= -1
    second = simulate(snapshot, scores, 5, config["portfolio"])
    assert first["fills"].iloc[0].instrument_id != second["fills"].iloc[0].instrument_id
