import copy

import numpy as np
import pandas as pd
import pytest
from test_earnings_ledger import account as account
from test_earnings_ledger import decisions, run

from quant_research.earnings_comparison import (
    PAIR_COLUMNS,
    attribute_account,
    paired_decisions,
    reconcile_paired_accounts,
)
from quant_research.earnings_ledger import DECISION_COLUMNS


def paired(account, budget=30_000.):
    snapshot = account[0]
    symbols = snapshot.tables["instruments"].instrument_id.tolist()
    day = snapshot.dates[50]
    pairs = pd.DataFrame([dict(date=day, decision_at=day + "T10:00:00Z", treated=symbols[0], control=symbols[1],
                               treated_trade_id="E1/test", control_trade_id="P_matched/E1/test",
                               selection_order=0, intended_budget=budget)])
    features = decisions(snapshot, symbols=symbols[:2])
    frozen = paired_decisions(pairs, features)
    treated = run(account, frozen["E1_matched"])
    control = run(account, frozen["P_matched_exposure"])
    return treated, control, pairs


def test_paired_orders_keep_the_exact_frozen_budget_and_full_cash_calendar(account):
    treated, control, pairs = paired(account, 15_000.)
    assert treated["fills"].iloc[0].notional == control["fills"].iloc[0].notional == 15_000.
    assert treated["nav"].cash.iloc[-1] == pytest.approx(199977.20)
    result = reconcile_paired_accounts(treated, control, pairs)
    assert result["pair_diagnostics"].same_planned_budget.all()
    assert len(result["daily_accounts"]) == 31
    assert np.allclose(result["daily_accounts"].daily_net_return_difference, 0.)
    assert result["internal_arithmetic_reconciled"]
    assert result["exposure_equality_accepted"] is False


def test_round_lot_exposure_difference_remains_visible_instead_of_being_rescaled_after_fill(account):
    snapshot = account[0]
    symbol = snapshot.tables["instruments"].instrument_id.iloc[1]
    bars = snapshot.tables["bars"]
    mask = bars.instrument_id.eq(symbol)
    bars.loc[mask, ["open", "high", "low", "close", "upper_limit", "lower_limit"]] *= 2.03
    treated, control, pairs = paired(account)
    result = reconcile_paired_accounts(treated, control, pairs)
    diagnostic = result["pair_diagnostics"].iloc[0]
    assert diagnostic.treated_filled_notional == 30_000.
    assert diagnostic.control_filled_notional == 28_420.
    assert result["daily_pair_exposure"].committed_difference.max() == 1580.
    assert result["formal_profit_qualified"] is False


def test_one_sided_entry_is_not_post_filtered_and_all_its_cash_days_remain(account):
    snapshot = account[0]
    symbol = snapshot.tables["instruments"].instrument_id.iloc[0]
    bars = snapshot.tables["bars"]
    bars.loc[bars.instrument_id.eq(symbol) & bars.date.eq(snapshot.dates[51]), "upper_limit"] = 10.
    treated, control, pairs = paired(account)
    result = reconcile_paired_accounts(treated, control, pairs)
    assert len(result["pair_diagnostics"]) == 1
    assert result["pair_diagnostics"].iloc[0].entry_status == "different_or_one_sided_entry"
    assert "control_only" in set(result["daily_pair_exposure"].holding_status)
    assert treated["summary"]["total_return"] == 0.
    assert len(result["daily_accounts"]) == 31


def test_exit_blocking_preserves_unbalanced_holding_days_and_their_returns(account):
    snapshot = account[0]
    symbol = snapshot.tables["instruments"].instrument_id.iloc[1]
    bars = snapshot.tables["bars"]
    bars.loc[bars.instrument_id.eq(symbol) & bars.date.isin(snapshot.dates[71:74]), "lower_limit"] = 10.
    treated, control, pairs = paired(account)
    result = reconcile_paired_accounts(treated, control, pairs)
    exposure = result["daily_pair_exposure"].set_index("date")
    assert set(exposure.loc[snapshot.dates[71:74], "holding_status"]) == {"control_only"}
    assert control["fills"].iloc[-1].date == snapshot.dates[74]


def test_independent_cash_fill_and_share_checks_detect_corrupted_ledger_evidence(account):
    result = run(account)
    for mode, expected in [("cash", "attribution mismatch"), ("flow", "Fill cash flow"),
                           ("shares", "holding inventory"), ("returns", "Daily account returns")]:
        changed = copy.deepcopy(result)
        if mode == "cash":
            changed["nav"].loc[changed["nav"].index[10], "cash"] += 1.
        elif mode == "flow":
            changed["fills"].loc[changed["fills"].index[0], "cash_flow"] -= 1.
        elif mode == "shares":
            changed["holdings"].loc[changed["holdings"].index[0], "shares"] += 1.
        else:
            changed["nav"].loc[changed["nav"].index[10], "daily_return"] += .01
        with pytest.raises(ValueError, match=expected):
            attribute_account(changed)


def test_split_and_unpaid_dividend_inventory_survive_sale_and_reconcile_independently(account):
    snapshot = account[0]
    symbol = snapshot.tables["instruments"].instrument_id.iloc[0]
    snapshot.tables["actions"] = pd.DataFrame([dict(instrument_id=symbol,
        ex_date=snapshot.dates[55], pay_date=snapshot.dates[79], cash_per_share=1.,
        split_ratio=2., tax_policy="cn_individual_cash_2015")])
    bars = snapshot.tables["bars"]
    mask = bars.instrument_id.eq(symbol) & bars.date.ge(snapshot.dates[55])
    bars.loc[mask, ["open", "close", "high", "low", "upper_limit", "lower_limit"]] = (
        bars.loc[mask, ["open", "close", "high", "low", "upper_limit", "lower_limit"]] - 1.) / 2
    result = run(account, end=75)
    audit = attribute_account(result)
    terminal = audit["terminal_trade_attribution"].iloc[0]
    assert result["fills"].shares.tolist() == [3000., 6000.]
    assert terminal.receivables == 3000. and terminal.market_value == 0.
    assert audit["daily_reconciliation"].reconstructed_nav.iloc[-1] == pytest.approx(result["nav"].nav.iloc[-1])
    changed = copy.deepcopy(result)
    changed["cash_events"].loc[changed["cash_events"].kind.eq("dividend_entitlement"), "shares_after"] += 100.
    with pytest.raises(ValueError, match="Corporate-action inventory"):
        attribute_account(changed)


def test_partial_exits_and_repeated_minimum_fees_reconcile_without_discarding_remainder(account):
    snapshot = account[0]
    symbol = snapshot.tables["instruments"].instrument_id.iloc[0]
    bars = snapshot.tables["bars"]
    bars.loc[bars.instrument_id.eq(symbol) & bars.date.ge(snapshot.dates[52]), "amount"] = 1_000_000.
    result = run(account)
    sells = result["fills"].loc[result["fills"].side.eq("sell")]
    assert len(sells) > 1 and sells.shares.sum() == 3000.
    assert len(result["holdings"].loc[result["holdings"].date.ge(snapshot.dates[71])]) > 0
    assert attribute_account(result)["internal_arithmetic_reconciled"]


def test_dividend_cash_receipt_cannot_be_paid_before_its_evidenced_due_date(account):
    snapshot = account[0]
    symbol = snapshot.tables["instruments"].instrument_id.iloc[0]
    snapshot.tables["actions"] = pd.DataFrame([dict(instrument_id=symbol,
        ex_date=snapshot.dates[55], pay_date=snapshot.dates[59], cash_per_share=1.,
        split_ratio=1., tax_policy="cn_individual_cash_2015")])
    bars = snapshot.tables["bars"]
    mask = bars.instrument_id.eq(symbol) & bars.date.ge(snapshot.dates[55])
    bars.loc[mask, ["open", "close", "high", "low", "upper_limit", "lower_limit"]] -= 1.
    result = run(account)
    assert attribute_account(result)["internal_arithmetic_reconciled"]
    result["cash_events"].loc[result["cash_events"].kind.eq("dividend_paid"), "date"] = snapshot.dates[56]
    with pytest.raises(ValueError, match="Dividend payment"):
        attribute_account(result)


def test_empty_paired_accounts_keep_cash_dates_and_cannot_be_promoted(account):
    empty = pd.DataFrame(columns=sorted(DECISION_COLUMNS))
    result = run(account, empty)
    pairs = pd.DataFrame(columns=sorted(PAIR_COLUMNS))
    audit = reconcile_paired_accounts(result, result, pairs)
    assert len(audit["daily_accounts"]) == 31
    assert audit["pair_diagnostics"].empty
    assert audit["formal_profit_qualified"] is False
