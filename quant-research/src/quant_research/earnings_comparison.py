"""Paired planned orders, independent account reconciliation and research attribution.

Never delete an unsuccessful member after observing fills. Reconciled means the account
arithmetic and pair identities were checked; it does not mean exposure was identical or
that sources, strategy profitability, or a formal statistical result were accepted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .earnings_ledger import DECISION_COLUMNS, cents

PAIR_COLUMNS = {"date", "decision_at", "treated", "control", "treated_trade_id", "control_trade_id",
                "selection_order", "intended_budget"}


def validate_pairs(pairs):
    if PAIR_COLUMNS - set(pairs):
        raise ValueError("Missing frozen pair identities or budgets")
    if pairs.empty:
        return
    for field in ["treated", "control", "treated_trade_id", "control_trade_id"]:
        if not pairs[field].map(lambda v: isinstance(v, str) and bool(v.strip())).all():
            raise ValueError("Frozen pair identities must be nonempty")
    clocks = pairs.decision_at.map(pd.Timestamp)
    if clocks.map(lambda v: pd.isna(v) or v.tzinfo is None).any():
        raise ValueError("Frozen paired decisions require explicit timezone clocks")
    if (pairs.treated_trade_id.duplicated().any() or pairs.control_trade_id.duplicated().any()
            or pairs.duplicated(["date", "control"]).any() or pairs.treated.eq(pairs.control).any()
            or not np.isfinite(pairs.intended_budget).all() or pairs.intended_budget.le(0).any()):
        raise ValueError("Invalid or reused frozen comparison pair")


def paired_decisions(pairs: pd.DataFrame, features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create matched-subset decisions using intended budgets, without inspecting outcomes."""
    validate_pairs(pairs)
    required = {"date", "instrument_id", "decision_at", "signal_atr14_raw"}
    if required - set(features) or features.duplicated(["date", "instrument_id"]).any():
        raise ValueError("Missing or duplicate frozen paired technical features")
    indexed = features.set_index(["date", "instrument_id"])
    arms = {"E1_matched": [], "P_matched_exposure": []}
    for row in pairs.itertuples():
        for arm, instrument, trade in [("E1_matched", row.treated, row.treated_trade_id),
                                       ("P_matched_exposure", row.control, row.control_trade_id)]:
            key = row.date, instrument
            if key not in indexed.index:
                raise ValueError("A frozen pair is missing its signal-close input")
            feature = indexed.loc[key]
            clocks = [pd.Timestamp(v) for v in [feature.decision_at, row.decision_at]]
            if any(pd.isna(v) or v.tzinfo is None for v in clocks) or clocks[0] != clocks[1]:
                raise ValueError("Frozen pair and technical information clocks differ")
            arms[arm].append({"date": row.date, "instrument_id": instrument,
                              "selection_order": row.selection_order, "decision_at": row.decision_at,
                              "signal_atr14_raw": feature.signal_atr14_raw, "trade_id": trade,
                              "requested_budget": row.intended_budget})
    columns = [*sorted(DECISION_COLUMNS), "trade_id", "requested_budget"]
    return {arm: pd.DataFrame(rows, columns=columns) for arm, rows in arms.items()}


def _frame(account, name, columns):
    frame = account[name].copy()
    if frame.empty:
        return frame.reindex(columns=columns)
    if set(columns) - set(frame):
        raise ValueError(f"Missing account evidence fields: {name}")
    return frame


def attribute_account(account: dict) -> dict:
    """Independently reconstruct each daily NAV from fills, entitlements and holdings.

    Gross dividend entitlements create assets immediately; paid amounts transfer assets
    into cash. Sale withholding is already in fill cash flow. Terminal holdings and unpaid
    dividends remain in trade P&L, including unresolved/stale marks.
    """
    required = {"date", "nav", "cash", "market_value", "receivables", "dividend_tax_reserve", "daily_return"}
    nav = account["nav"]
    if required - set(nav) or nav.empty or nav.date.duplicated().any() or nav.date.tolist() != sorted(nav.date):
        raise ValueError("Incomplete or unsorted daily account evidence")
    if not np.isfinite(nav[list(required - {"date"})].to_numpy(float)).all():
        raise ValueError("Nonfinite account valuations remain unqualified")
    initial = float(account["summary"]["initial_capital"])
    if not np.isfinite(initial) or initial <= 0:
        raise ValueError("Invalid initial account capital")
    expected_returns = nav.nav.pct_change(fill_method=None)
    expected_returns.iloc[0] = nav.nav.iloc[0] / initial - 1
    if not np.allclose(expected_returns, nav.daily_return, rtol=1e-11, atol=1e-12):
        raise ValueError("Daily account returns differ from observed NAV changes")
    fills = _frame(account, "fills", ["date", "trade_id", "side", "shares", "price", "cash_flow", "notional", "fees"])
    holdings = _frame(account, "holdings", ["date", "trade_id", "shares", "mark_price", "stale_mark",
                                          "committed_cost", "dividend_tax_reserve"])
    events = _frame(account, "cash_events", ["date", "trade_id", "kind", "amount"])
    dates = nav.date.tolist()
    for frame in [fills, holdings, events]:
        if not set(frame.date) <= set(dates):
            raise ValueError("Account attribution evidence falls outside the NAV interval")
        if not frame.trade_id.map(lambda v: isinstance(v, str) and bool(v.strip())).all():
            raise ValueError("Missing account attribution trade identity")
    for frame, fields in [(fills, ["shares", "price", "cash_flow", "notional", "fees"]),
                          (holdings, ["shares", "mark_price", "committed_cost", "dividend_tax_reserve"]),
                          (events, ["amount"])]:
        if not frame.empty and not np.isfinite(frame[fields].to_numpy(float)).all():
            raise ValueError("Nonfinite account attribution inputs")
    if not set(fills.side) <= {"buy", "sell"}:
        raise ValueError("Unknown account fill side")
    for row in fills.itertuples():
        expected = (-row.notional if row.side == "buy" else row.notional) - row.fees
        if (row.shares <= 0 or row.price <= 0 or row.fees < 0
                or abs(cents(row.shares * row.price) - row.notional) > .001
                or abs(cents(expected) - row.cash_flow) > .001):
            raise ValueError("Fill cash flow differs from shares, price or transaction fees")
    if holdings.duplicated(["date", "trade_id"]).any():
        raise ValueError("Duplicate daily holding attribution")
    if not events.empty and not set(events.kind) <= {"dividend_entitlement", "dividend_paid"}:
        raise ValueError("Unsupported non-trade cash event requires explicit attribution")
    trade_ids = sorted(set(fills.trade_id) | set(holdings.trade_id) | set(events.trade_id))
    trades = {trade: {"flow": 0., "entitled": 0., "paid": 0., "shares": 0., "receipts": []} for trade in trade_ids}
    fill_days = {day: group for day, group in fills.groupby("date")}
    event_days = {day: group for day, group in events.groupby("date")}
    holding_days = {day: group.set_index("trade_id") for day, group in holdings.groupby("date")}
    daily, checks = [], []
    cash = initial
    for record in nav.itertuples():
        for row in event_days.get(record.date, events.iloc[:0]).loc[
                lambda frame: frame.kind.eq("dividend_entitlement")].itertuples():
            fields = [getattr(row, name, np.nan) for name in ["shares_before", "shares_after", "split_ratio", "cash_per_share"]]
            if (not np.isfinite(fields).all() or row.shares_before <= 0 or row.split_ratio <= 0
                    or row.cash_per_share < 0 or abs(trades[row.trade_id]["shares"] - row.shares_before) > 1e-8
                    or abs(row.shares_after - row.shares_before * row.split_ratio) > 1e-8
                    or abs(cents(row.shares_before * row.cash_per_share) - row.amount) > .001):
                raise ValueError("Corporate-action inventory or entitlement does not reconcile")
            trades[row.trade_id]["shares"] = row.shares_after
            if row.amount:
                pay_date = getattr(row, "pay_date", "")
                if not isinstance(pay_date, str) or pay_date < record.date:
                    raise ValueError("Dividend entitlement requires an explicit valid payment date")
                trades[row.trade_id]["receipts"].append({"amount": row.amount, "pay_date": pay_date})
        for row in fill_days.get(record.date, fills.iloc[:0]).itertuples():
            inventory = trades[row.trade_id]["shares"]
            if ((row.side == "buy" and inventory > 1e-8)
                    or (row.side == "sell" and row.shares > inventory + 1e-8)):
                raise ValueError("Account share inventory does not reconcile")
            trades[row.trade_id]["shares"] += row.shares if row.side == "buy" else -row.shares
            trades[row.trade_id]["flow"] += row.cash_flow
            cash = cents(cash + row.cash_flow)
        for row in event_days.get(record.date, events.iloc[:0]).itertuples():
            if row.kind == "dividend_entitlement":
                trades[row.trade_id]["entitled"] += row.amount
            else:
                receipts = trades[row.trade_id]["receipts"]
                found = next((r for r in receipts if r["pay_date"] == getattr(row, "pay_date", None)
                              and abs(r["amount"] - row.amount) <= .001), None)
                if found is None or found["pay_date"] > record.date:
                    raise ValueError("Dividend payment lacks a due, unpaid entitlement")
                receipts.remove(found)
                trades[row.trade_id]["paid"] += row.amount
                cash = cents(cash + row.amount)
        current = holding_days.get(record.date, holdings.iloc[:0].set_index("trade_id"))
        mark, reserve, owed, pnl = 0., 0., 0., 0.
        for trade, state in trades.items():
            value, tax, committed, stale = 0., 0., 0., False
            if trade in current.index:
                held = current.loc[trade]
                value, tax = held.shares * held.mark_price, held.dividend_tax_reserve
                committed, stale = held.committed_cost, bool(held.stale_mark)
            if (abs(state["shares"] - (current.loc[trade].shares if trade in current.index else 0.)) > 1e-8
                    or value < 0 or tax < 0 or committed < 0):
                raise ValueError("Daily holding inventory differs from fills and corporate actions")
            receivable = state["entitled"] - state["paid"]
            if receivable < -.011:
                raise ValueError("Dividend payments exceed evidenced entitlements")
            profit = state["flow"] + state["entitled"] + value - tax
            daily.append({"date": record.date, "trade_id": trade, "net_pnl_cny": profit,
                          "market_value": value, "committed_cost": committed,
                          "receivables": receivable, "dividend_tax_reserve": tax, "stale_mark": stale})
            pnl += profit
            mark += value
            reserve += tax
            owed += receivable
        for actual, reconstructed, field in [(record.cash, cash, "cash"), (record.market_value, mark, "market_value"),
                (record.receivables, owed, "receivables"), (record.dividend_tax_reserve, reserve, "dividend_tax_reserve"),
                (record.nav, initial + pnl, "net_asset_value")]:
            if abs(actual - reconstructed) > .011:
                raise ValueError(f"Independent account attribution mismatch: {record.date}/{field}")
        checks.append({"date": record.date, "reconstructed_nav": initial + pnl, "reconstructed_cash": cash})
    daily = pd.DataFrame(daily, columns=["date", "trade_id", "net_pnl_cny", "market_value", "committed_cost",
                                        "receivables", "dividend_tax_reserve", "stale_mark"])
    terminal = daily.loc[daily.date.eq(dates[-1])].copy()
    # Before the first buy an account may have zero rows; its P&L and all dates still reconcile.
    return {"daily_trade_attribution": daily, "terminal_trade_attribution": terminal,
            "daily_reconciliation": pd.DataFrame(checks), "internal_arithmetic_reconciled": True,
            "formal_profit_qualified": False}


def reconcile_paired_accounts(treated: dict, control: dict, pairs: pd.DataFrame) -> dict:
    """Keep unmatched fills, blocked exits and all cash dates visible in paired diagnostics."""
    validate_pairs(pairs)
    for field in ["initial_capital", "horizon", "delay_sessions", "friction_multiplier"]:
        if treated["summary"][field] != control["summary"][field]:
            raise ValueError("Paired accounts use different capital, horizon or execution scenario")
    if treated["nav"].date.tolist() != control["nav"].date.tolist():
        raise ValueError("Paired accounts must retain exactly the same observed dates")
    t_attr, c_attr = attribute_account(treated), attribute_account(control)
    dates = treated["nav"].date.tolist()
    t_ids, c_ids = set(pairs.treated_trade_id), set(pairs.control_trade_id)
    for account, identities in [(treated, t_ids), (control, c_ids)]:
        for name in ["fills", "holdings", "cash_events"]:
            if not account[name].empty and not set(account[name].trade_id) <= identities:
                raise ValueError("Use matched-subset accounts for paired attribution; full-account extras must remain separate")
    diagnostics = []
    for row in pairs.itertuples():
        record = {key: getattr(row, key) for key in PAIR_COLUMNS}
        for label, account, identity in [("treated", treated, row.treated_trade_id), ("control", control, row.control_trade_id)]:
            selected = _frame(account, "selections", ["date", "instrument_id", "status", "trade_id", "budget", "decision_at"])
            selected = selected.loc[selected.status.eq("selected_before_open") & selected.trade_id.eq(identity)]
            bought = _frame(account, "fills", ["date", "trade_id", "side", "notional"])
            bought = bought.loc[bought.trade_id.eq(identity) & bought.side.eq("buy")]
            if len(selected) > 1 or len(bought) > 1:
                raise ValueError("A frozen pair cannot acquire multiple account entry identities")
            if not selected.empty and (selected.iloc[0].date != row.date
                    or selected.iloc[0].instrument_id != getattr(row, label)
                    or pd.Timestamp(selected.iloc[0].decision_at) != pd.Timestamp(row.decision_at)):
                raise ValueError("Account selection differs from the original frozen pair")
            record[label + "_planned_budget"] = float(selected.budget.sum())
            record[label + "_filled_notional"] = float(bought.notional.sum())
            record[label + "_entry_date"] = bought.date.iloc[0] if not bought.empty else ""
        record["same_planned_budget"] = (abs(record["treated_planned_budget"] - row.intended_budget) <= .01
                                         and abs(record["control_planned_budget"] - row.intended_budget) <= .01)
        record["entry_status"] = ("both_unfilled" if not record["treated_entry_date"] and not record["control_entry_date"]
                                   else "same_entry_session" if record["treated_entry_date"] == record["control_entry_date"]
                                   else "different_or_one_sided_entry")
        diagnostics.append(record)
    daily = []
    t_daily = t_attr["daily_trade_attribution"].set_index(["date", "trade_id"])
    c_daily = c_attr["daily_trade_attribution"].set_index(["date", "trade_id"])
    for date in dates:
        for row in pairs.itertuples():
            def value(frame, identity, field):
                key = date, identity
                return float(frame.loc[key, field]) if key in frame.index else 0.
            t_cost, c_cost = value(t_daily, row.treated_trade_id, "committed_cost"), value(c_daily, row.control_trade_id, "committed_cost")
            daily.append({"date": date, "treated_trade_id": row.treated_trade_id,
                          "control_trade_id": row.control_trade_id,
                          "treated_committed": t_cost, "control_committed": c_cost,
                          "committed_difference": t_cost - c_cost,
                          "joint_committed": min(t_cost, c_cost),
                          "holding_status": "both_held" if t_cost > 0 and c_cost > 0 else
                              "treated_only" if t_cost > 0 else "control_only" if c_cost > 0 else "both_cash",
                          "cumulative_pnl_difference_cny": value(t_daily, row.treated_trade_id, "net_pnl_cny")
                              - value(c_daily, row.control_trade_id, "net_pnl_cny")})
    comparison = treated["nav"][["date", "daily_return", "nav", "cash", "market_value"]].merge(
        control["nav"][["date", "daily_return", "nav", "cash", "market_value"]], on="date",
        suffixes=("_treated", "_control"), validate="one_to_one")
    comparison["daily_net_return_difference"] = comparison.daily_return_treated - comparison.daily_return_control
    return {"daily_accounts": comparison, "pair_diagnostics": pd.DataFrame(diagnostics),
            "daily_pair_exposure": pd.DataFrame(daily), "treated_attribution": t_attr, "control_attribution": c_attr,
            "internal_arithmetic_reconciled": True, "actual_exposure_diagnostics_available": True,
            "exposure_equality_accepted": False, "formal_profit_qualified": False,
            "qualification_note": "Planned matches remain immutable; observed fill and holding differences are retained, not post-filtered."}
