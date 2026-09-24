from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .storage import Snapshot
from .universe import cutoff, risk_status, universe_on


@dataclass
class Position:
    shares: float
    bought_index: int
    last_price: float


def rule_on(snapshot: Snapshot, board: str, date: str) -> pd.Series:
    rules = snapshot.tables["rules"]
    selected = rules.loc[(rules.board == board) & (rules.start_date <= date) &
                         ((rules.end_date == "") | (rules.end_date >= date))]
    if len(selected) != 1:
        raise ValueError(f"Missing or overlapping execution rule: {board} {date}")
    return selected.iloc[0]


def simulate(snapshot: Snapshot, predictions: pd.DataFrame, horizon: int, config: dict,
             cost_multiplier: float = 1.0, delay_days: int = 0) -> dict[str, pd.DataFrame | dict]:
    """Research-only cash ledger. Daily-bar open fills are conservative assumptions, not proofs."""
    if snapshot.manifest["declaration"].get("purpose") == "training":
        raise ValueError("Training-only snapshot cannot be used for portfolio simulation")
    if horizon not in {5, 20} or delay_days < 0 or cost_multiplier <= 0:
        raise ValueError("Invalid simulation parameters")
    if predictions.empty or predictions.duplicated(["date", "instrument_id"]).any():
        raise ValueError("Empty or duplicate predictions")
    if not np.isfinite(predictions.score).all():
        raise ValueError("Non-finite prediction")
    instruments = snapshot.tables["instruments"].set_index("instrument_id")
    if not set(predictions.instrument_id) <= set(instruments.index):
        raise ValueError("Prediction contains an unknown instrument")
    dates = snapshot.dates
    if not set(predictions.date) <= set(dates):
        raise ValueError("Prediction date is not in the trading calendar")
    index = {d: i for i, d in enumerate(dates)}
    scheduled = {}
    for date, group in predictions.groupby("date"):
        execution = index[date] + 1 + delay_days
        if execution < len(dates):
            scheduled[dates[execution]] = (date, group)
    if not scheduled:
        raise ValueError("No observable execution day after the signals")
    bars = snapshot.tables["bars"].set_index(["date", "instrument_id"])
    prior_adv = {}
    for instrument_id, group in snapshot.tables["bars"].groupby("instrument_id"):
        values = group.set_index("date").amount.reindex(dates).rolling(20).mean().shift(1)
        for date, value in values.items():
            prior_adv[(date, instrument_id)] = value
    actions = snapshot.tables["actions"]
    cash = float(config["initial_cash"])
    peak = cash
    halted = False
    positions: dict[str, Position] = {}
    receivables = []
    orders, fills, nav, holdings, cash_events = [], [], [], [], []
    start = index[min(scheduled)]
    # Liquidate only through scheduled maturity/risk orders. Open holdings stay on the final NAV.
    end = min(len(dates) - 1, index[max(scheduled)] + horizon)
    stale_valuation_days = 0
    for i in range(start, end + 1):
        date = dates[i]
        # Corporate-action entitlements use pre-action holdings; cash is receivable until pay day.
        for action in actions.loc[actions.ex_date == date].itertuples():
            if action.instrument_id in positions:
                position = positions[action.instrument_id]
                amount = position.shares * float(action.cash_per_share)
                old_price = position.last_price
                position.shares *= float(action.split_ratio)
                position.last_price = ((old_price - float(action.cash_per_share)) /
                                       float(action.split_ratio))
                if amount:
                    receivables.append({"instrument_id": action.instrument_id,
                                        "pay_date": action.pay_date, "amount": amount})
                cash_events.append({"date": date, "instrument_id": action.instrument_id,
                                    "kind": "action_entitlement", "amount": amount})
        unpaid = []
        for item in receivables:
            if item["pay_date"] <= date:
                cash += item["amount"]
                cash_events.append({"date": date, "instrument_id": item["instrument_id"],
                                    "kind": "dividend_paid", "amount": item["amount"]})
            else:
                unpaid.append(item)
        receivables = unpaid
        risk = risk_status(snapshot, date, cutoff(date, "09:25"))
        used_notional = defaultdict(float)

        def execute(instrument_id: str, side: str, requested: float, reason: str) -> None:
            nonlocal cash
            rule = rule_on(snapshot, str(instruments.loc[instrument_id, "board"]), date)
            order = {"date": date, "instrument_id": instrument_id, "side": side,
                     "reason": reason, "requested_shares": float(requested), "status": "blocked"}
            orders.append(order)
            key = (date, instrument_id)
            if key not in bars.index:
                order["block_reason"] = "no_bar_or_suspended"
                return
            bar = bars.loc[key]
            if bar.volume <= 0:
                order["block_reason"] = "no_trading_volume"
                return
            limit_col = "upper_limit" if side == "buy" else "lower_limit"
            limit = pd.to_numeric(bar[limit_col], errors="coerce")
            if not np.isfinite(limit) or limit <= 0:
                order["block_reason"] = "price_limit_unverified"
                return
            if ((side == "buy" and bar.open >= limit - 1e-8) or
                    (side == "sell" and bar.open <= limit + 1e-8)):
                order["block_reason"] = "open_at_price_limit"
                return
            if side == "buy" and risk[instrument_id] != "normal":
                order["block_reason"] = "risk_status_at_execution"
                return
            if side == "sell" and i - positions[instrument_id].bought_index < int(rule.t_plus):
                order["block_reason"] = "t_plus_settlement"
                return
            price = float(bar.open) * (1 + (1 if side == "buy" else -1) *
                                       float(rule.slippage_bps) / 10000 * cost_multiplier)
            if (side == "buy" and price > limit) or (side == "sell" and price < limit):
                order["block_reason"] = "slippage_crosses_limit"
                return
            adv = prior_adv.get(key, np.nan)
            if not np.isfinite(adv):
                order["block_reason"] = "insufficient_prior_liquidity_history"
                return
            capacity = max(0, adv * config["adv_participation"] - used_notional[instrument_id])
            quantity = min(requested, capacity / price)
            step = int(rule.buy_step if side == "buy" else rule.sell_step)
            quantity = float(np.floor(quantity / step) * step)
            minimum = int(rule.min_buy) if side == "buy" else step
            # Odd-lot sale is allowed only when liquidating the full residual position.
            if side == "sell" and capacity / price >= positions[instrument_id].shares:
                quantity = min(requested, positions[instrument_id].shares)
            rate = float(rule.buy_fee if side == "buy" else rule.sell_fee)
            tax_rate = float(rule.stamp_duty) if side == "sell" else 0.0

            def fee(q: float) -> float:
                return (max(float(rule.minimum_fee), q * price * rate) +
                        q * price * tax_rate) * cost_multiplier

            if side == "buy":
                while quantity >= minimum and quantity * price + fee(quantity) > cash + 1e-8:
                    quantity -= step
            if quantity < minimum and not (side == "sell" and quantity > 0 and
                                           quantity == positions[instrument_id].shares):
                order["block_reason"] = "cash_lot_or_liquidity_constraint"
                return
            cost = fee(quantity)
            if side == "buy":
                cash -= quantity * price + cost
                positions[instrument_id] = Position(quantity, i, float(bar.open))
            else:
                cash += quantity * price - cost
                positions[instrument_id].shares -= quantity
                if positions[instrument_id].shares <= 1e-8:
                    del positions[instrument_id]
            used_notional[instrument_id] += quantity * price
            order.update(status="filled" if quantity >= requested - 1e-8 else "partial",
                         filled_shares=quantity)
            fills.append({"date": date, "instrument_id": instrument_id, "side": side,
                          "shares": quantity, "price": price, "fees": cost,
                          "notional": quantity * price, "reason": reason})

        for instrument_id, position in list(positions.items()):
            reason = ("risk_warning_exit" if risk[instrument_id] in {"ST", "*ST"} else
                      "drawdown_halt_exit" if halted else
                      "holding_period_exit" if i - position.bought_index >= horizon else "")
            if reason:
                execute(instrument_id, "sell", position.shares, reason)
        if date in scheduled and not halted:
            signal_date, candidates = scheduled[date]
            universe = universe_on(snapshot, signal_date, risk_policy="exclude_st")
            universe = universe.set_index("instrument_id")
            eligible = universe.loc[universe.eligible]
            industry_weights = eligible.industry.value_counts(normalize=True).to_dict()
            # Orders are sized with the previous close, not today's future close.
            equity = cash + sum(p.shares * p.last_price for p in positions.values()) + sum(
                p["amount"] for p in receivables)
            candidates = candidates.sort_values(["score", "instrument_id"], ascending=[False, True])
            for row in candidates.itertuples():
                instrument_id = row.instrument_id
                if len(positions) >= config["max_positions"]:
                    break
                if instrument_id in positions or not universe.loc[instrument_id, "eligible"]:
                    continue
                industry = str(instruments.loc[instrument_id, "industry"])
                industry_value = sum(p.shares * p.last_price for symbol, p in positions.items()
                                     if instruments.loc[symbol, "industry"] == industry)
                headroom = equity * (industry_weights.get(industry, 0) +
                                     config["industry_deviation"]) - industry_value
                budget = min(equity * config["max_weight"], headroom, cash)
                previous_key = (dates[i - 1], instrument_id)
                if budget <= 0 or previous_key not in bars.index:
                    continue
                # No intraday resizing with hindsight: one capped order sized before the open.
                reference = float(bars.loc[previous_key, "close"])
                opening_key = (date, instrument_id)
                # An opening-auction gap may violate the max weight. Reduce at auction price.
                if opening_key in bars.index:
                    reference = max(reference, float(bars.loc[opening_key, "open"]) *
                                    (1 + float(rule_on(snapshot, str(instruments.loc[instrument_id,
                                         "board"]), date).slippage_bps) / 10000 * cost_multiplier))
                execute(instrument_id, "buy", budget / reference, "rank_selection")
        stale = 0
        for instrument_id, position in positions.items():
            key = (date, instrument_id)
            if key in bars.index:
                position.last_price = float(bars.loc[key, "close"])
            else:
                stale += 1
            holdings.append({"date": date, "instrument_id": instrument_id,
                             "shares": position.shares, "mark_price": position.last_price,
                             "stale_mark": key not in bars.index})
        stale_valuation_days += int(stale > 0)
        equity = cash + sum(p.shares * p.last_price for p in positions.values()) + sum(
            p["amount"] for p in receivables)
        peak = max(peak, equity)
        drawdown = equity / peak - 1
        if drawdown <= -config["max_drawdown"]:
            halted = True
        nav.append({"date": date, "cash": cash, "receivables": sum(p["amount"] for p in receivables),
                    "nav": equity, "drawdown": drawdown, "positions": len(positions),
                    "halted": halted, "stale_marks": stale})
        if cash < -1e-6:
            raise AssertionError("Cash ledger went negative")
    nav_frame = pd.DataFrame(nav)
    daily_returns = nav_frame.nav.pct_change(fill_method=None)
    daily_returns.iloc[0] = nav_frame.nav.iloc[0] / config["initial_cash"] - 1
    nav_frame["daily_return"] = daily_returns
    summary = {"total_return": float(nav_frame.nav.iloc[-1] / config["initial_cash"] - 1),
               "max_drawdown": float(nav_frame.drawdown.min()), "days": len(nav_frame),
               "fills": len(fills), "fees": sum(r["fees"] for r in fills),
               "stale_valuation_days": stale_valuation_days,
               "terminal_positions": len(positions), "halted": halted,
               "execution_assumption": "daily_open_with_conservative_limit_blocks",
               "industry_constraint": "static_research_industry_requires_historical_upgrade",
               "cost_multiplier": cost_multiplier, "delay_days": delay_days}
    return {"nav": nav_frame, "fills": pd.DataFrame(fills), "orders": pd.DataFrame(orders),
            "holdings": pd.DataFrame(holdings), "cash_events": pd.DataFrame(cash_events),
            "summary": summary}
