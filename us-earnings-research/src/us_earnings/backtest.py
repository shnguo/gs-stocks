"""Cash-constrained daily portfolio simulation in adjusted total-return units.

Prices are adjusted for splits, dividends and spin-offs by Alpaca. Fractional
units model reinvested distributions; these are not historical broker shares.
Stops operate in these adjusted units and are execution approximations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Position:
    ticker: str
    shares: float
    entry_date: pd.Timestamp
    entry_index: int
    entry_price: float
    invested: float
    stop: float
    signal_date: pd.Timestamp
    accession: str


def simulate(panel: dict, calendar: pd.DatetimeIndex, signals: pd.DataFrame, config: dict,
             cost_bps: float | None = None, holding_sessions: int | None = None):
    cost = (config["cost_bps_per_side"] if cost_bps is None else cost_bps) / 10000
    holding = holding_sessions or config["holding_sessions"]
    if not 0 <= cost < 0.1 or holding < 1:
        raise ValueError("Invalid cost or holding period")
    if config["max_positions"] < 1 or not 0 < config["position_fraction"] <= 1:
        raise ValueError("Invalid portfolio limits")
    groups = {date: group.sort_values(["rank", "ticker"], ascending=[False, True])
              for date, group in signals.groupby("entry_date")}
    cash, positions, history, trades, rejected = 1.0, {}, [], [], []
    active_calendar = calendar[(calendar >= config["start"]) & (calendar <= config["end"])]
    for day in active_calendar:
        index = calendar.get_loc(day)
        open_nav = cash + sum(pos.shares * panel[ticker].at[day, "open"]
                              for ticker, pos in positions.items())
        # Decisions and priority were computed at the previous session's close.
        candidates = groups.get(day)
        if candidates is not None:
            for candidate in candidates.to_dict("records"):
                ticker = candidate["ticker"]
                reason = "already_held" if ticker in positions else ""
                if not reason and len(positions) >= config["max_positions"]:
                    reason = "position_limit"
                budget = min(config["position_fraction"] * open_nav, cash)
                if not reason and budget < 1e-8:
                    reason = "insufficient_cash"
                if reason:
                    rejected.append({"date": day, "ticker": ticker, "reason": reason})
                    continue
                if pd.Timestamp(candidate["signal_date"]) >= day:
                    raise ValueError("Entry cannot use same-day or future information")
                opening = panel[ticker].at[day, "open"]
                entry = opening * (1 + cost)
                stop = opening - config["stop_atr_multiple"] * candidate["atr"]
                if stop <= 0:
                    rejected.append({"date": day, "ticker": ticker, "reason": "invalid_stop"})
                    continue
                shares = budget / entry
                cash -= budget
                positions[ticker] = Position(ticker, shares, day, index, entry, budget,
                                             stop, candidate["signal_date"], candidate["accession"])
        # Existing positions and newly entered positions can both hit today's stop.
        for ticker, position in list(positions.items()):
            bar = panel[ticker].loc[day]
            reason, fill = "", None
            if bar.low <= position.stop:
                reason = "stop"
                fill = min(bar.open, position.stop)  # An opening gap never fills at a better stop.
            elif index - position.entry_index + 1 >= holding:
                reason, fill = "time", bar.close
            elif day == active_calendar[-1]:
                reason, fill = "end_of_sample", bar.close
            if reason:
                exit_price = fill * (1 - cost)
                proceeds = position.shares * exit_price
                cash += proceeds
                trades.append({"ticker": ticker, "signal_date": position.signal_date,
                               "entry_date": position.entry_date, "exit_date": day,
                               "entry_price": position.entry_price, "exit_price": exit_price,
                               "shares": position.shares, "invested": position.invested,
                               "pnl": proceeds - position.invested,
                               "net_return": proceeds / position.invested - 1,
                               "holding_sessions": index - position.entry_index + 1,
                               "exit_reason": reason, "accession": position.accession})
                del positions[ticker]
        value = sum(pos.shares * panel[ticker].at[day, "close"]
                    for ticker, pos in positions.items())
        nav = cash + value
        if cash < -1e-9 or nav <= 0:
            raise ValueError("Cash or NAV invariant failed")
        history.append({"date": day, "nav": nav, "cash": cash,
                        "exposure": value / nav, "positions": len(positions)})
    trade_columns = ["ticker", "signal_date", "entry_date", "exit_date", "entry_price", "exit_price",
                     "shares", "invested", "pnl", "net_return", "holding_sessions", "exit_reason", "accession"]
    return (pd.DataFrame(history).set_index("date"), pd.DataFrame(trades, columns=trade_columns),
            pd.DataFrame(rejected, columns=["date", "ticker", "reason"]))


def passive(panel: dict, calendar: pd.DatetimeIndex, symbols: list[str], config: dict):
    days = calendar[(calendar >= config["start"]) & (calendar <= config["end"])]
    cost = config["cost_bps_per_side"] / 10000
    series = pd.DataFrame({s: panel[s].loc[days, "close"] /
                          (panel[s].at[days[0], "open"] * (1 + cost)) for s in symbols}).mean(axis=1)
    series.iloc[-1] *= (1 - cost)
    return pd.DataFrame({"nav": series, "cash": 0.0, "exposure": 1.0, "positions": len(symbols)})


def metrics(curve: pd.DataFrame, trades: pd.DataFrame | None = None,
            prior_nav: float = 1.0) -> dict:
    returns = curve.nav.pct_change(fill_method=None)
    returns.iloc[0] = curve.nav.iloc[0] / prior_nav - 1
    total = curve.nav.iloc[-1] / prior_nav - 1
    volatility = returns.std(ddof=1) * np.sqrt(252)
    peak = np.maximum.accumulate(np.r_[prior_nav, curve.nav.to_numpy()])[1:]
    drawdown = curve.nav.to_numpy() / peak - 1
    result = {"total_return": float(total), "cagr": float((1 + total) ** (252 / len(curve)) - 1),
              "annualized_volatility": float(volatility),
              "sharpe_zero_cash": float(returns.mean() * 252 / volatility) if volatility > 0 else None,
              "max_drawdown": float(drawdown.min()), "average_exposure": float(curve.exposure.mean()),
              "sessions": len(curve)}
    if trades is not None:
        result.update(trades=len(trades), win_rate=float((trades.net_return > 0).mean()) if len(trades) else None,
                      mean_trade_return=float(trades.net_return.mean()) if len(trades) else None,
                      censored_trades=int((trades.exit_reason == "end_of_sample").sum()))
    return result


def event_returns(panel: dict, calendar: pd.DatetimeIndex, signals: pd.DataFrame,
                  config: dict, holding: int = 20) -> pd.DataFrame:
    """Equal event observations; not a funded portfolio. No stops; identical horizons."""
    cost = config["cost_bps_per_side"] / 10000
    rows = []
    for event in signals.loc[signals.earnings].to_dict("records"):
        start = calendar.get_loc(event["entry_date"])
        end = start + holding - 1
        if end >= len(calendar) or calendar[end] > pd.Timestamp(config["end"]):
            continue
        day = calendar[end]
        stock = panel[event["ticker"]].at[day, "close"] * (1 - cost) / (
            panel[event["ticker"]].at[event["entry_date"], "open"] * (1 + cost)) - 1
        market = panel[config["benchmark"]].at[day, "close"] * (1 - cost) / (
            panel[config["benchmark"]].at[event["entry_date"], "open"] * (1 + cost)) - 1
        rows.append({**event, "exit_date": day, "stock_return": stock,
                     "benchmark_return": market, "excess_return": stock - market,
                     "signal_month": event["signal_date"].strftime("%Y-%m")})
    return pd.DataFrame(rows)


def cluster_interval(events: pd.DataFrame, seed: int, repetitions: int = 2000) -> dict:
    if events.empty:
        return {"events": 0, "months": 0, "mean_excess": None, "lower_95": None, "upper_95": None}
    groups = [group.excess_return.to_numpy() for _, group in events.groupby("signal_month")]
    if len(groups) < 5:
        return {"events": len(events), "months": len(groups),
                "mean_excess": float(events.excess_return.mean()),
                "lower_95": None, "upper_95": None,
                "interval_status": "unavailable_fewer_than_five_signal_months"}
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repetitions):
        picked = rng.integers(0, len(groups), size=len(groups))
        samples.append(np.concatenate([groups[i] for i in picked]).mean())
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {"events": len(events), "months": len(groups), "mean_excess": float(events.excess_return.mean()),
            "lower_95": float(lower), "upper_95": float(upper), "interval_status": "month_cluster_bootstrap"}
