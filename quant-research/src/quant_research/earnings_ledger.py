"""Fixed-capital earnings experiment ledger; daily opens are conditional fill scenarios.

This research engine does not certify inputs or promote a strategy. Qualification belongs to
the evidence and registration runner. Holdings, unpaid dividends and unknown endpoints persist.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

from .backtest import rule_on
from .storage import Snapshot
from .universe import cutoff, risk_status

FEE_COLUMNS = {"board", "start_date", "end_date", "commission_rate", "minimum_commission",
               "transfer_rate", "stamp_duty", "slippage_bps"}
INDUSTRY_COLUMNS = {"instrument_id", "start_date", "end_date", "known_at", "industry"}
DECISION_COLUMNS = {"date", "instrument_id", "selection_order", "decision_at", "signal_atr14_raw"}
SELECTION_FIELDS = ["date", "signal_date", "instrument_id", "industry", "trade_id", "decision_at",
    "budget", "requested_budget", "selection_order", "requested_shares", "signal_close",
    "signal_atr14_raw", "entry_date", "status"]


def cents(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def dated_row(frame: pd.DataFrame, board: str, date: str) -> pd.Series:
    selected = frame.loc[(frame.board == board) & (frame.start_date <= date)
                         & ((frame.end_date == "") | (frame.end_date >= date))]
    if len(selected) != 1:
        raise ValueError(f"Missing or overlapping fee schedule: {board}/{date}")
    return selected.iloc[0]


def transaction_fees(notional: float, side: str, schedule: pd.Series,
                     multiplier: float = 1.) -> dict[str, float]:
    """Stress broker commission only; statutory transfer and stamp schedules stay actual."""
    if (side not in {"buy", "sell"} or not np.isfinite(notional) or notional < 0
            or not np.isfinite(multiplier) or multiplier <= 0):
        raise ValueError("Invalid transaction fee arguments")
    components = {"commission": cents(max(float(schedule.minimum_commission),
                          notional * float(schedule.commission_rate)) * multiplier),
                  "transfer_fee": cents(notional * float(schedule.transfer_rate)),
                  "stamp_duty": cents(notional * float(schedule.stamp_duty)) if side == "sell" else 0.}
    return {**components, "total": cents(sum(components.values()))}


def dividend_tax_rate(acquired: str, disposed: str, ex_date: str) -> float:
    """Public-market natural-person cash dividends under the post-2015-09-08 policy.

Natural month/year anniversaries include their boundary. Month-end clipping is exposed as
an engine convention; source qualification must verify applicable settlement/account rules.
"""
    if ex_date < "2015-09-08" or disposed < acquired:
        raise ValueError("Dividend tax policy outside the implemented verified date range")
    bought, sold = pd.Timestamp(acquired), pd.Timestamp(disposed)
    if sold <= bought + pd.DateOffset(months=1):
        return .20
    return .10 if sold <= bought + pd.DateOffset(years=1) else 0.


@dataclass
class Holding:
    shares: float
    bought_index: int
    bought_date: str
    last_price: float
    cost_basis: float
    entry_industry: str
    trade_id: str
    dividends: list[dict] = field(default_factory=list)


def _validate_inputs(snapshot, decisions, fees, industries, start, end):
    if snapshot.manifest["declaration"].get("purpose") == "training":
        raise ValueError("Training-only data cannot be used for an account-return simulation")
    if FEE_COLUMNS - set(fees) or INDUSTRY_COLUMNS - set(industries) or DECISION_COLUMNS - set(decisions):
        raise ValueError("Missing earnings account input columns")
    if start not in snapshot.dates or end not in snapshot.dates or start > end:
        raise ValueError("Account interval must use observed exchange sessions")
    observed = snapshot.tables["bars"].date
    if observed.empty or start < observed.min() or end > observed.max():
        raise ValueError("Future calendar sessions cannot become observed account days")
    if decisions.duplicated(["date", "instrument_id"]).any():
        raise ValueError("Duplicate account decision identity")
    if "trade_id" in decisions and (decisions.trade_id.duplicated().any()
            or not decisions.trade_id.map(lambda v: isinstance(v, str) and bool(v.strip())).all()):
        raise ValueError("Every account trade identity must be unique and nonempty")
    if (not set(decisions.date) <= set(snapshot.dates)
            or (decisions.date > end).any() or (decisions.date < start).any()):
        raise ValueError("Decision outside the observable account interval")
    instruments = snapshot.tables["instruments"].set_index("instrument_id")
    if not set(decisions.instrument_id) <= set(instruments.index):
        raise ValueError("Unknown decision instrument")
    for name in FEE_COLUMNS - {"board", "start_date", "end_date"}:
        values = pd.to_numeric(fees[name], errors="raise")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Missing or invalid fee values")
    if not industries.industry.map(lambda v: isinstance(v, str) and v.strip()
                                  and v.strip().lower() not in {"unknown", "unverified", "nan", "none"}).all():
        raise ValueError("Unknown industry must remain a blocking condition")
    if decisions.empty:
        return
    if "requested_budget" in decisions and (not np.isfinite(decisions.requested_budget).all()
            or decisions.requested_budget.le(0).any()):
        raise ValueError("Frozen paired budgets must be positive and finite")
    if (not np.isfinite(decisions.selection_order).all()
            or not np.isfinite(decisions.signal_atr14_raw).all()
            or (decisions.signal_atr14_raw <= 0).any()):
        raise ValueError("Invalid frozen selection order or ATR")
    for date, group in decisions.groupby("date"):
        if group.decision_at.nunique() != 1:
            raise ValueError("Each decision cohort must share one frozen timestamp")
        offset = snapshot.dates.index(date)
        at = pd.Timestamp(group.decision_at.iloc[0])
        if (at.tzinfo is None or offset + 1 >= len(snapshot.dates)
                or at < cutoff(date, "15:00")
                or at >= cutoff(snapshot.dates[offset + 1], "09:00")):
            raise ValueError("Account decisions require completed-close information and timely freezing")


def simulate_earnings_account(snapshot: Snapshot, decisions: pd.DataFrame, fee_schedule: pd.DataFrame,
                              industry_history: pd.DataFrame, start_date: str, end_date: str,
                              *, capital: float = 200_000., horizon: int = 20,
                              max_positions: int = 6, slot_weight: float = .15,
                              gross_weight: float = .90, industry_weight: float = .30,
                              adv_participation: float = .005, drawdown_pause: float = .10,
                              friction_multiplier: float = 1., delay_sessions: int = 0) -> dict:
    """Freeze choices at each close; reserve selected slots even when their entry is blocked.

    A position bought at session i requests exit at open i+horizon. The observed account
    continues through end_date, including blocked maturities and cash-only days.
    """
    _validate_inputs(snapshot, decisions, fee_schedule, industry_history, start_date, end_date)
    if (not all(np.isfinite(v) for v in [capital, max_positions, slot_weight, gross_weight, industry_weight,
                                       adv_participation, drawdown_pause, friction_multiplier])
            or capital <= 0 or horizon not in {5, 20} or max_positions < 1 or int(max_positions) != max_positions
            or not 0 < slot_weight <= industry_weight <= gross_weight <= 1
            or not 0 < adv_participation <= 1 or not 0 < drawdown_pause < 1
            or friction_multiplier < 1 or delay_sessions not in {0, 1}):
        raise ValueError("Invalid fixed-account protocol")
    if "requested_budget" in decisions and decisions.requested_budget.gt(capital * slot_weight + 1e-8).any():
        raise ValueError("Requested paired budget exceeds the protocol slot limit")
    dates = snapshot.dates
    index = {date: i for i, date in enumerate(dates)}
    bars = snapshot.tables["bars"].set_index(["date", "instrument_id"])
    instruments = snapshot.tables["instruments"].set_index("instrument_id")
    industries = industry_history.copy()
    industries["known_at"] = pd.to_datetime(industries.known_at, utc=True, errors="raise")
    if industries.known_at.isna().any():
        raise ValueError("Missing industry availability")
    actions = snapshot.tables["actions"]
    if not actions.empty and "tax_policy" not in actions:
        raise ValueError("Corporate actions require explicit holder-tax treatment")
    if not actions.empty and not set(actions.tax_policy) <= {"not_taxable", "cn_individual_cash_2015"}:
        raise ValueError("Unsupported corporate action tax treatment")
    prior_adv = {}
    for instrument, group in snapshot.tables["bars"].groupby("instrument_id"):
        series = group.set_index("date").amount.reindex(dates).rolling(20, min_periods=20).mean()
        prior_adv.update({(date, instrument): amount for date, amount in series.items()})

    def industry_on(instrument, date, at):
        selected = industries.loc[(industries.instrument_id == instrument) & (industries.start_date <= date)
                                  & ((industries.end_date == "") | (industries.end_date >= date))
                                  & (industries.known_at <= at)]
        if selected.empty:
            return None
        selected = selected.loc[selected.known_at == selected.known_at.max()]
        return str(selected.industry.iloc[0]) if len(selected) == 1 else None

    cash, peak = float(capital), float(capital)
    positions: dict[str, Holding] = {}
    receivables, orders, selections, fills, holdings, cash_events, nav = [], [], [], [], [], [], []
    scheduled: dict[str, list[dict]] = {}
    halted = False
    cash_flow_balance = float(capital)
    input_scope = snapshot.manifest["declaration"].get("execution_scope")
    if input_scope:
        raise ValueError("A five-day scoped snapshot cannot certify this multi-session account")
    for i in range(index[start_date], index[end_date] + 1):
        date = dates[i]
        day_adv = {}
        # Existing holders receive ex-date entitlements before any ex-date opening orders.
        for action in actions.loc[actions.ex_date == date].itertuples():
            if action.instrument_id not in positions:
                continue
            position = positions[action.instrument_id]
            original_shares = position.shares
            shares = position.shares * float(action.split_ratio)
            if abs(shares - round(shares)) > 1e-8:
                raise ValueError("Fractional corporate-action shares require explicit settlement evidence")
            amount = cents(position.shares * float(action.cash_per_share))
            position.shares = shares
            position.last_price = (position.last_price - float(action.cash_per_share)) / float(action.split_ratio)
            if position.last_price <= 0:
                raise ValueError("Invalid corporate-action reference bridge")
            if amount:
                receivables.append({"instrument_id": action.instrument_id, "amount": amount,
                                    "pay_date": action.pay_date, "trade_id": position.trade_id})
                if action.tax_policy == "cn_individual_cash_2015":
                    position.dividends.append({"gross": amount, "ex_date": date})
            cash_events.append({"date": date, "instrument_id": action.instrument_id,
                                "trade_id": position.trade_id, "kind": "dividend_entitlement", "amount": amount,
                                "shares_before": original_shares, "shares_after": shares,
                                "split_ratio": float(action.split_ratio), "cash_per_share": float(action.cash_per_share),
                                "pay_date": action.pay_date, "tax_policy": action.tax_policy})
        unpaid = []
        for item in receivables:
            if item["pay_date"] <= date:
                cash = cents(cash + item["amount"])
                cash_flow_balance = cents(cash_flow_balance + item["amount"])
                cash_events.append({"date": date, **item, "kind": "dividend_paid"})
            else:
                unpaid.append(item)
        receivables = unpaid
        states = risk_status(snapshot, date, cutoff(date, "09:25"))

        def execute(instrument, side, requested, reason, trade_id, budget=None, signal=None):
            nonlocal cash, cash_flow_balance
            order = {"date": date, "instrument_id": instrument, "side": side, "reason": reason,
                     "trade_id": trade_id, "requested_shares": requested, "status": "blocked"}
            orders.append(order)

            def block(why):
                order["block_reason"] = why
                return False

            key = date, instrument
            if key not in bars.index:
                return block("missing_bar_or_suspended")
            bar = bars.loc[key]
            if bar.volume <= 0:
                return block("no_trading_volume")
            board = str(instruments.loc[instrument, "board"])
            rule = rule_on(snapshot, board, date)
            fee = dated_row(fee_schedule, board, date)
            limit = float(bar.upper_limit if side == "buy" else bar.lower_limit)
            if not np.isfinite(limit) or limit <= 0:
                return block("unverified_price_limit")
            opening = float(bar.open)
            if (side == "buy" and opening >= limit - 1e-8) or (side == "sell" and opening <= limit + 1e-8):
                return block("opening_at_price_limit")
            if side == "buy":
                if instrument in positions:
                    return block("existing_position_retained")
                if states[instrument] != "normal":
                    return block("execution_risk_not_normal")
                if len(positions) >= max_positions:
                    return block("retained_holdings_fill_all_slots")
                # A known action between signal and entry needs a separately evidenced price bridge.
                if ((actions.instrument_id == instrument) & (actions.ex_date > signal["signal_date"])
                        & (actions.ex_date <= date)).any():
                    return block("entry_action_requires_verified_signal_price_bridge")
                if opening - signal["signal_close"] > signal["signal_atr14_raw"] + 1e-8:
                    return block("opening_gap_exceeds_one_signal_atr")
            elif i - positions[instrument].bought_index < int(rule.t_plus):
                return block("t_plus_settlement")
            price = round(opening * (1 + (1 if side == "buy" else -1)
                                    * float(fee.slippage_bps) / 10_000 * friction_multiplier), 4)
            if (side == "buy" and price > limit + 1e-8) or (side == "sell" and price < limit - 1e-8):
                return block("slippage_crosses_price_limit")
            liquidity_date = dates[i - 1] if i else ""
            if side == "buy":
                # The frozen signal's known liquidity also caps the delayed-entry scenario.
                liquidity_date = signal["signal_date"]
            adv = prior_adv.get((liquidity_date, instrument), np.nan)
            if not np.isfinite(adv):
                return block("insufficient_completed_liquidity_history")
            capacity = max(0., adv * adv_participation - day_adv.get(instrument, 0.))
            step = int(rule.buy_step if side == "buy" else rule.sell_step)
            quantity = np.floor(min(requested, capacity / price) / step) * step
            if side == "sell" and capacity / price >= positions[instrument].shares:
                quantity = min(requested, positions[instrument].shares)
            if side == "buy":
                current_industry = industry_on(instrument, date, cutoff(date, "09:25"))
                if current_industry is None or current_industry != signal["industry"]:
                    return block("entry_industry_unknown_or_changed")
                committed = sum(p.cost_basis for p in positions.values())
                held_industries = {s: industry_on(s, date, cutoff(date, "09:25")) for s in positions}
                if any(v is None for v in held_industries.values()):
                    return block("retained_holding_industry_unknown")
                industry_committed = sum(p.cost_basis for s, p in positions.items()
                                         if held_industries[s] == current_industry)
                budget = min(budget, capital * gross_weight - committed,
                             capital * industry_weight - industry_committed)
                quantity = min(quantity, np.floor(max(0., budget) / price / step) * step)
                while quantity >= int(rule.min_buy):
                    notional = cents(quantity * price)
                    cost = transaction_fees(notional, side, fee, friction_multiplier)["total"]
                    if notional + cost <= cash + 1e-8:
                        break
                    quantity -= step
                if quantity < int(rule.min_buy):
                    return block("cash_budget_lot_or_liquidity_constraint")
            elif quantity <= 0:
                return block("sell_liquidity_constraint")
            notional = cents(quantity * price)
            costs = transaction_fees(notional, side, fee, friction_multiplier)
            dividend_tax = 0.
            if side == "buy":
                positions[instrument] = Holding(float(quantity), i, date, opening, notional,
                                                signal["industry"], trade_id)
                flow = -notional - costs["total"]
            else:
                position = positions[instrument]
                fraction = float(quantity / position.shares)
                for entitlement in position.dividends:
                    taxable = entitlement["gross"] * fraction
                    dividend_tax += cents(taxable * dividend_tax_rate(position.bought_date, date,
                                                                       entitlement["ex_date"]))
                    entitlement["gross"] -= taxable
                position.cost_basis *= 1 - fraction
                position.shares -= quantity
                if position.shares <= 1e-8:
                    del positions[instrument]
                flow = notional - costs["total"] - dividend_tax
            cash = cents(cash + flow)
            cash_flow_balance = cents(cash_flow_balance + flow)
            day_adv[instrument] = day_adv.get(instrument, 0.) + notional
            order.update(status="filled" if quantity >= requested - 1e-8 else "partial",
                         filled_shares=float(quantity))
            fills.append({"date": date, "instrument_id": instrument, "side": side, "trade_id": trade_id,
                          "reason": reason, "shares": float(quantity), "price": price, "notional": notional,
                          "commission": costs["commission"], "transfer_fee": costs["transfer_fee"],
                          "stamp_duty": costs["stamp_duty"], "dividend_tax": cents(dividend_tax),
                          "fees": cents(costs["total"] + dividend_tax), "cash_flow": cents(flow),
                          "price_field": "open"})
            return True

        # Requested exits are retried each actual session. Drawdown pauses entries, not bookkeeping.
        for instrument, position in list(positions.items()):
            reason = "risk_exit" if states[instrument] in {"ST", "*ST"} else (
                "fixed_horizon_exit" if i - position.bought_index >= horizon else "")
            if reason:
                execute(instrument, "sell", position.shares, reason, position.trade_id)
        for selected in scheduled.get(date, []):
            if halted:
                orders.append({"date": date, "instrument_id": selected["instrument_id"],
                               "side": "buy", "trade_id": selected["trade_id"], "status": "blocked",
                               "block_reason": "drawdown_entry_pause", "reason": "frozen_selection"})
            else:
                execute(selected["instrument_id"], "buy", selected["requested_shares"], "frozen_selection",
                        selected["trade_id"], selected["budget"], selected)
        stale = 0
        dividend_reserve = 0.
        for instrument, position in positions.items():
            key = date, instrument
            missing = key not in bars.index
            if not missing:
                position.last_price = float(bars.loc[key, "close"])
            stale += int(missing)
            reserve = sum(cents(d["gross"] * dividend_tax_rate(position.bought_date, date, d["ex_date"]))
                          for d in position.dividends)
            dividend_reserve += reserve
            holdings.append({"date": date, "instrument_id": instrument, "trade_id": position.trade_id,
                             "shares": position.shares, "mark_price": position.last_price, "stale_mark": missing,
                             "committed_cost": position.cost_basis, "entry_industry": position.entry_industry,
                             "dividend_tax_reserve": reserve})
        marked = sum(p.shares * p.last_price for p in positions.values())
        owed = sum(p["amount"] for p in receivables)
        equity = cash + marked + owed - dividend_reserve
        peak = max(peak, equity)
        drawdown = equity / peak - 1
        halted = halted or drawdown <= -drawdown_pause
        nav.append({"date": date, "cash": cash, "market_value": marked, "receivables": owed,
                    "dividend_tax_reserve": dividend_reserve, "nav": equity, "drawdown": drawdown,
                    "positions": len(positions), "halted": halted, "stale_marks": stale})
        if cash < -.001 or abs(cash - cash_flow_balance) > .001:
            raise AssertionError("Cash flow conservation failed")

        # Decisions remain immutable. Select once at their completed close before opening outcomes.
        group = decisions.loc[decisions.date == date]
        if group.empty:
            continue
        execution = i + 1 + delay_sessions
        if execution >= len(dates):
            raise ValueError("Calendar lacks the session needed to freeze the pending entry")
        at = pd.Timestamp(group.decision_at.iloc[0])
        signal_states = risk_status(snapshot, date, at)
        expected_retained = {symbol: p for symbol, p in positions.items()
                             if execution - p.bought_index < horizon and signal_states[symbol] not in {"ST", "*ST"}}
        # Also reserve other already frozen delayed entries; never oversubscribe the six slots.
        pending = [r for day, rows in scheduled.items() if day > date for r in rows]
        pending_symbols = {r["instrument_id"] for r in pending}
        slots = max(0, max_positions - len(expected_retained) - len(pending))
        gross = sum(p.cost_basis for p in expected_retained.values()) + sum(r["budget"] for r in pending)
        industry_used = {}
        held_industries = {s: industry_on(s, date, at) for s in expected_retained}
        if any(v is None for v in held_industries.values()):
            selections.append({"date": date, "instrument_id": "", "status": "retained_holding_industry_unknown"})
            continue
        for symbol, p in expected_retained.items():
            industry = held_industries[symbol]
            industry_used[industry] = industry_used.get(industry, 0.) + p.cost_basis
        for r in pending:
            industry_used[r["industry"]] = industry_used.get(r["industry"], 0.) + r["budget"]
        chosen = []
        for row in group.sort_values(["selection_order", "instrument_id"]).itertuples():
            instrument = row.instrument_id
            if len(chosen) >= slots:
                break
            if instrument in expected_retained or instrument in pending_symbols:
                continue
            industry = industry_on(instrument, date, at)
            instrument_meta = instruments.loc[instrument]
            key = date, instrument
            if (industry is None or signal_states[instrument] != "normal" or key not in bars.index
                    or instrument_meta.listed_at > date
                    or instrument_meta.delisted_at and instrument_meta.delisted_at < date):
                selections.append({"date": date, "instrument_id": instrument, "status": "ineligible_at_freeze"})
                continue
            requested_budget = float(getattr(row, "requested_budget", capital * slot_weight))
            budget = min(requested_budget, capital * slot_weight, capital * gross_weight - gross,
                         capital * industry_weight - industry_used.get(industry, 0.))
            if budget <= 0:
                continue
            reference = float(bars.loc[key, "close"])
            rule = rule_on(snapshot, str(instrument_meta.board), date)
            quantity = np.floor(budget / reference / int(rule.buy_step)) * int(rule.buy_step)
            if quantity < int(rule.min_buy):
                continue
            trade_id = getattr(row, "trade_id", f"{date}/{instrument}")
            selection = {"date": date, "signal_date": date, "instrument_id": instrument, "industry": industry,
                         "trade_id": trade_id, "decision_at": row.decision_at, "budget": budget,
                         "requested_budget": requested_budget,
                         "selection_order": row.selection_order,
                         "requested_shares": float(quantity), "signal_close": reference,
                         "signal_atr14_raw": float(row.signal_atr14_raw), "entry_date": dates[execution]}
            chosen.append(selection)
            gross += budget
            industry_used[industry] = industry_used.get(industry, 0.) + budget
            selections.append({**selection, "status": "selected_before_open"})
        scheduled.setdefault(dates[execution], []).extend(chosen)
    nav_frame = pd.DataFrame(nav)
    nav_frame["daily_return"] = nav_frame.nav.pct_change(fill_method=None)
    nav_frame.loc[nav_frame.index[0], "daily_return"] = nav_frame.nav.iloc[0] / capital - 1
    pending_entries = [{**row, "status": "awaiting_observed_entry_session"}
                       for day, rows in sorted(scheduled.items()) if day > end_date for row in rows]
    return {"nav": nav_frame, "fills": pd.DataFrame(fills), "orders": pd.DataFrame(orders),
            "holdings": pd.DataFrame(holdings), "cash_events": pd.DataFrame(cash_events),
            "selections": pd.DataFrame(selections).reindex(columns=SELECTION_FIELDS),
            "pending_entries": pd.DataFrame(pending_entries).reindex(columns=SELECTION_FIELDS),
            "summary": {"initial_capital": capital, "total_return": float(nav_frame.nav.iloc[-1] / capital - 1),
                        "maximum_drawdown": float(nav_frame.drawdown.min()), "terminal_positions": len(positions),
                        "terminal_receivables": owed, "terminal_dividend_tax_reserve": dividend_reserve,
                        "halted": halted, "observed_sessions": len(nav_frame), "horizon": horizon,
                        "pending_entry_orders": len(pending_entries),
                        "friction_multiplier": friction_multiplier, "delay_sessions": delay_sessions,
                        "formal_profit_qualified": False,
                        "execution_assumption": "frozen_selection_next_open_fixed_horizon_open_conditional_fills",
                        "allocation": "fixed_initial_capital_slots_no_daily_reweighting",
                        "tax_policy_scope": "post_2015_public_market_individual_cash_dividends_natural_month_anniversary",
                        "qualification_note": "Input declarations, source evidence, first-event coverage and registration require separate acceptance."}}
