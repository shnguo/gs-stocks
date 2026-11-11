import copy

import numpy as np
import pandas as pd
import pytest

from quant_research.earnings_ledger import (
    DECISION_COLUMNS,
    dividend_tax_rate,
    simulate_earnings_account,
    transaction_fees,
)


@pytest.fixture
def account(snapshot):
    bars = snapshot.tables["bars"]
    for name, value in {"open": 10., "close": 10., "high": 10.1, "low": 9.9,
                        "factor": 1., "upper_limit": 11., "lower_limit": 9.,
                        "volume": 1_000_000., "amount": 10_000_000.}.items():
        bars[name] = value
    snapshot.tables["actions"] = snapshot.tables["actions"].iloc[:0].copy()
    snapshot.tables["actions"]["tax_policy"] = pd.Series(dtype=str)
    industries = snapshot.tables["instruments"][["instrument_id", "industry"]].copy()
    industries["start_date"] = snapshot.dates[0]
    industries["end_date"] = ""
    industries["known_at"] = "2020-01-01T00:00:00Z"
    fees = pd.DataFrame([dict(board=b, start_date=snapshot.dates[0], end_date="",
                             commission_rate=.0005, minimum_commission=5.,
                             transfer_rate=.00001, stamp_duty=.0005, slippage_bps=0.)
                         for b in snapshot.tables["rules"].board])
    return snapshot, fees, industries


def decisions(snapshot, indices=(50,), symbols=None):
    symbols = symbols or [snapshot.tables["instruments"].instrument_id.iloc[0]]
    return pd.DataFrame([dict(date=snapshot.dates[i], instrument_id=symbol,
                              selection_order=order, decision_at=snapshot.dates[i] + "T10:00:00Z",
                              signal_atr14_raw=.2)
                         for i in indices for order, symbol in enumerate(symbols)])


def run(account, selected=None, end=80, **kwargs):
    snapshot, fees, industries = account
    return simulate_earnings_account(snapshot, selected if selected is not None else decisions(snapshot),
                                     fees, industries, snapshot.dates[50], snapshot.dates[end], **kwargs)


def test_fixed_twenty_session_exit_uses_open_and_matches_hand_calculated_cash(account):
    snapshot, _, _ = account
    symbol = snapshot.tables["instruments"].instrument_id.iloc[0]
    bars = snapshot.tables["bars"]
    bars.loc[(bars.date == snapshot.dates[71]) & (bars.instrument_id == symbol), "close"] = 1.
    result = run(account)
    assert result["fills"].date.tolist() == [snapshot.dates[51], snapshot.dates[71]]
    assert result["fills"].price.tolist() == [10., 10.]
    assert result["fills"].shares.tolist() == [3000., 3000.]
    assert result["fills"].fees.tolist() == pytest.approx([15.30, 30.30])
    assert result["nav"].cash.iloc[-1] == pytest.approx(199954.40)
    assert result["summary"]["formal_profit_qualified"] is False
    assert len(result["nav"]) == 31
    chosen = result["selections"].iloc[0]
    assert chosen.date == chosen.signal_date == snapshot.dates[50]
    assert chosen.selection_order == 0 and chosen.budget == 30_000.


def test_terminal_close_freezes_budget_without_using_unobserved_opening_quotes(account):
    snapshot = account[0]
    selected = decisions(snapshot, indices=(80,))
    original = run(account, selected, end=80)
    assert original["fills"].empty and original["nav"].nav.eq(200_000).all()
    planned = original["pending_entries"].iloc[0]
    assert planned.budget == 30_000. and planned.requested_shares == 3000.
    assert planned.entry_date == snapshot.dates[81]
    assert original["summary"]["pending_entry_orders"] == 1
    altered = copy.deepcopy(account)
    bars = altered[0].tables["bars"]
    bars.loc[bars.date.gt(snapshot.dates[80]), ["open", "close", "high", "low", "volume", "amount"]] = np.nan
    future_unknown = run(altered, selected, end=80)
    pd.testing.assert_frame_equal(future_unknown["pending_entries"], original["pending_entries"])
    extended = run(account, selected, end=81)
    pd.testing.assert_frame_equal(extended["nav"].iloc[:-1].reset_index(drop=True), original["nav"],
                                  check_exact=True, check_dtype=False)
    pd.testing.assert_frame_equal(extended["selections"], original["selections"])
    assert extended["pending_entries"].empty
    assert extended["fills"].date.tolist() == [snapshot.dates[81]]


def test_delayed_tail_reserves_both_plans_and_requires_their_calendar_sessions(account):
    snapshot = account[0]
    symbols = snapshot.tables["instruments"].instrument_id.iloc[:2]
    selected = pd.concat([decisions(snapshot, indices=(79,), symbols=[symbols.iloc[0]]),
                          decisions(snapshot, indices=(80,), symbols=[symbols.iloc[1]])], ignore_index=True)
    result = run(account, selected, end=80, delay_sessions=1)
    assert result["pending_entries"].entry_date.tolist() == [snapshot.dates[81], snapshot.dates[82]]
    assert result["pending_entries"].budget.sum() == 60_000.
    assert result["fills"].empty and result["summary"]["pending_entry_orders"] == 2
    short = copy.deepcopy(account)
    calendar = short[0].tables["calendar"]
    short[0].tables["calendar"] = calendar.loc[calendar.date.le(snapshot.dates[81])]
    with pytest.raises(ValueError, match="Calendar lacks"):
        run(short, selected, end=80, delay_sessions=1)


def test_friction_stress_doubles_broker_fees_without_doubling_statutory_taxes(account):
    base, stress = run(account), run(account, friction_multiplier=2)
    assert stress["fills"].commission.sum() == 2 * base["fills"].commission.sum()
    assert stress["fills"].stamp_duty.sum() == base["fills"].stamp_duty.sum() == 15.
    assert stress["fills"].transfer_fee.sum() == base["fills"].transfer_fee.sum() == .6
    assert stress["nav"].cash.iloc[-1] == pytest.approx(199924.40)


def test_minimum_fee_and_component_rounding_are_applied_per_actual_transaction(account):
    schedule = account[1].iloc[0]
    assert transaction_fees(1000., "buy", schedule) == {
        "commission": 5., "transfer_fee": .01, "stamp_duty": 0., "total": 5.01}
    assert transaction_fees(1000., "sell", schedule, 2)["total"] == 10.51


def test_blocked_exit_retains_position_and_retries_until_an_actual_sellable_session(account):
    snapshot, _, _ = account
    symbol = snapshot.tables["instruments"].instrument_id.iloc[0]
    bars = snapshot.tables["bars"]
    mask = bars.date.isin(snapshot.dates[71:74]) & (bars.instrument_id == symbol)
    bars.loc[mask, "lower_limit"] = 10.
    result = run(account)
    assert result["fills"].iloc[-1].date == snapshot.dates[74]
    assert set(result["holdings"].date) >= set(snapshot.dates[71:74])
    unfinished = run(account, end=73)
    assert unfinished["summary"]["terminal_positions"] == 1
    assert unfinished["nav"].nav.iloc[-1] == pytest.approx(199984.70)


def test_missing_bar_retains_the_stale_mark_instead_of_removing_the_holding(account):
    snapshot, _, _ = account
    symbol = snapshot.tables["instruments"].instrument_id.iloc[0]
    bars = snapshot.tables["bars"]
    snapshot.tables["bars"] = bars.loc[~((bars.date == snapshot.dates[71]) & (bars.instrument_id == symbol))]
    result = run(account)
    held = result["holdings"].loc[result["holdings"].date == snapshot.dates[71]].iloc[0]
    assert held.stale_mark and held.shares == 3000
    # A missing source day also leaves the next ADV20 uncertain. Do not invent a zero-volume day.
    assert result["summary"]["terminal_positions"] == 1
    assert "insufficient_completed_liquidity_history" in set(result["orders"].get("block_reason", []))


def test_no_replacement_after_a_frozen_choice_is_blocked_at_the_open(account):
    snapshot, _, industries = account
    instruments = snapshot.tables["instruments"]
    seventh = instruments.iloc[0].copy()
    seventh["instrument_id"] = "synthetic.xshg.999999"
    snapshot.tables["instruments"] = pd.concat([instruments, seventh.to_frame().T], ignore_index=True)
    bars = snapshot.tables["bars"]
    extra = bars.loc[bars.instrument_id == instruments.instrument_id.iloc[0]].copy()
    extra["instrument_id"] = seventh.instrument_id
    snapshot.tables["bars"] = pd.concat([bars, extra], ignore_index=True)
    extra_industry = industries.iloc[0].copy()
    extra_industry["instrument_id"] = seventh.instrument_id
    account = snapshot, account[1], pd.concat([industries, extra_industry.to_frame().T], ignore_index=True)
    bars = snapshot.tables["bars"]
    bars.loc[(bars.date == snapshot.dates[51]) & (bars.instrument_id == instruments.instrument_id.iloc[0]), "upper_limit"] = 10.
    selected = decisions(snapshot, symbols=snapshot.tables["instruments"].instrument_id.tolist())
    result = run(account, selected)
    buys = result["fills"].loc[result["fills"].side == "buy"]
    assert len(buys) == 5
    assert seventh.instrument_id not in set(result["orders"].instrument_id)
    assert len(result["selections"].loc[result["selections"].status == "selected_before_open"]) == 6


def test_same_instrument_is_not_overwritten_if_its_previous_exit_is_blocked(account):
    snapshot, _, _ = account
    symbol = snapshot.tables["instruments"].instrument_id.iloc[0]
    bars = snapshot.tables["bars"]
    bars.loc[(bars.date == snapshot.dates[71]) & (bars.instrument_id == symbol), "lower_limit"] = 10.
    result = run(account, decisions(snapshot, indices=(50, 70)))
    assert len(result["fills"].loc[result["fills"].side == "buy"]) == 1
    assert "existing_position_retained" in set(result["orders"].get("block_reason", []))
    assert result["holdings"].shares.max() == 3000.


def test_opening_gap_cancels_order_and_delayed_entry_keeps_twenty_complete_sessions(account):
    snapshot, _, _ = account
    symbol = snapshot.tables["instruments"].instrument_id.iloc[0]
    bars = snapshot.tables["bars"]
    mask = (bars.date == snapshot.dates[51]) & (bars.instrument_id == symbol)
    bars.loc[mask, "open"] = 10.3
    result = run(account)
    assert result["fills"].empty
    assert "opening_gap_exceeds_one_signal_atr" in set(result["orders"].block_reason)
    delayed = run(account, delay_sessions=1)
    assert delayed["fills"].date.tolist() == [snapshot.dates[52], snapshot.dates[72]]


def test_dividend_receivable_tax_reserve_and_disposal_tax_reconcile(account):
    snapshot, _, _ = account
    symbol = snapshot.tables["instruments"].instrument_id.iloc[0]
    snapshot.tables["actions"] = pd.DataFrame([dict(instrument_id=symbol,
        ex_date=snapshot.dates[55], pay_date=snapshot.dates[59], cash_per_share=1.,
        split_ratio=1., tax_policy="cn_individual_cash_2015")])
    bars = snapshot.tables["bars"]
    mask = (bars.instrument_id == symbol) & (bars.date >= snapshot.dates[55])
    bars.loc[mask, ["open", "close", "high", "low", "upper_limit", "lower_limit"]] -= 1.
    result = run(account)
    nav = result["nav"].set_index("date")
    assert nav.loc[snapshot.dates[55], "receivables"] == 3000.
    assert nav.loc[snapshot.dates[55], "dividend_tax_reserve"] == 600.
    assert nav.loc[snapshot.dates[59], "receivables"] == 0.
    assert result["fills"].iloc[-1].dividend_tax == 600.
    assert result["nav"].cash.iloc[-1] == pytest.approx(199357.43)
    assert result["nav"].nav.iloc[-1] == pytest.approx(199357.43)


def test_owed_dividend_survives_sale_and_does_not_become_spendable_cash_early(account):
    snapshot, _, _ = account
    symbol = snapshot.tables["instruments"].instrument_id.iloc[0]
    snapshot.tables["actions"] = pd.DataFrame([dict(instrument_id=symbol,
        ex_date=snapshot.dates[55], pay_date=snapshot.dates[79], cash_per_share=1.,
        split_ratio=1., tax_policy="cn_individual_cash_2015")])
    result = run(account, end=75)
    assert result["summary"]["terminal_positions"] == 0
    assert result["summary"]["terminal_receivables"] == 3000.
    assert result["nav"].nav.iloc[-1] - result["nav"].cash.iloc[-1] == 3000.


def test_cash_only_dates_are_preserved_and_training_inputs_are_rejected(account):
    snapshot, _, _ = account
    empty = pd.DataFrame(columns=sorted(DECISION_COLUMNS))
    result = run(account, empty)
    assert len(result["nav"]) == 31 and result["fills"].empty
    assert np.allclose(result["nav"].nav, 200000.)
    snapshot.manifest["declaration"]["purpose"] = "training"
    with pytest.raises(ValueError, match="Training-only"):
        run(account)


def test_natural_month_and_year_dividend_boundaries_do_not_use_a_fixed_thirty_day_shortcut():
    assert dividend_tax_rate("2026-07-10", "2026-08-10", "2026-07-20") == .20
    assert dividend_tax_rate("2026-07-10", "2026-08-11", "2026-07-20") == .10
    assert dividend_tax_rate("2025-07-10", "2026-07-10", "2025-07-20") == .10
    assert dividend_tax_rate("2025-07-10", "2026-07-11", "2025-07-20") == 0.


def test_future_or_late_decision_cannot_be_used_as_timely_input(account):
    snapshot, _, _ = account
    selected = decisions(snapshot)
    selected["decision_at"] = snapshot.dates[51] + "T01:00:00Z"
    with pytest.raises(ValueError, match="timely freezing"):
        run(account, selected)


def test_large_gains_do_not_increase_the_next_initial_capital_slot_budget(account):
    snapshot, _, _ = account
    symbol = snapshot.tables["instruments"].instrument_id.iloc[0]
    bars = snapshot.tables["bars"]
    mask = bars.instrument_id.eq(symbol) & bars.date.ge(snapshot.dates[60])
    bars.loc[mask, ["open", "high", "low", "close", "upper_limit", "lower_limit"]] *= 2
    result = run(account, decisions(snapshot, indices=(50, 72)), end=95)
    buys = result["fills"].loc[result["fills"].side.eq("buy")]
    assert buys.notional.tolist() == [30_000., 30_000.]
    assert result["nav"].nav.max() > 229_000.


def test_drawdown_pauses_new_entries_but_retains_holdings_until_their_normal_exit(account):
    snapshot, _, _ = account
    symbols = snapshot.tables["instruments"].instrument_id.tolist()
    selected = pd.concat([decisions(snapshot, symbols=symbols[:2]),
                          decisions(snapshot, indices=(56,), symbols=[symbols[3]])])
    bars = snapshot.tables["bars"]
    mask = bars.instrument_id.isin(symbols[:2]) & bars.date.ge(snapshot.dates[55])
    bars.loc[mask, ["open", "high", "low", "close", "upper_limit", "lower_limit"]] *= .5
    result = run(account, selected)
    assert result["summary"]["halted"]
    assert result["nav"].set_index("date").loc[snapshot.dates[55], "positions"] == 2
    assert set(result["fills"].loc[result["fills"].side.eq("sell"), "date"]) == {snapshot.dates[71]}
    assert "drawdown_entry_pause" in set(result["orders"].get("block_reason", []))


def test_split_doubles_shares_and_preserves_equity_without_inventing_cash(account):
    snapshot, _, _ = account
    symbol = snapshot.tables["instruments"].instrument_id.iloc[0]
    snapshot.tables["actions"] = pd.DataFrame([dict(instrument_id=symbol,
        ex_date=snapshot.dates[55], pay_date=snapshot.dates[55], cash_per_share=0.,
        split_ratio=2., tax_policy="not_taxable")])
    bars = snapshot.tables["bars"]
    mask = bars.instrument_id.eq(symbol) & bars.date.ge(snapshot.dates[55])
    bars.loc[mask, ["open", "close", "high", "low", "upper_limit", "lower_limit"]] /= 2
    result = run(account)
    assert result["fills"].shares.tolist() == [3000., 6000.]
    assert result["nav"].nav.max() <= 200_000.
    assert result["nav"].cash.iloc[-1] == pytest.approx(199954.40)


def test_future_calendar_and_missing_industry_cannot_become_account_evidence(account):
    snapshot, _, industries = account
    snapshot.tables["bars"] = snapshot.tables["bars"].loc[snapshot.tables["bars"].date.le(snapshot.dates[75])]
    with pytest.raises(ValueError, match="Future calendar"):
        run(account, end=80)
    industries.loc[industries.index[0], "industry"] = None
    with pytest.raises(ValueError, match="Unknown industry"):
        run(account, end=75)
