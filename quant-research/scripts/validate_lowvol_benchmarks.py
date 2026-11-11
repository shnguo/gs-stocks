"""Frozen low-volatility research comparison; read-only with respect to source data."""

from __future__ import annotations

import json
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/lowvol-benchmark-20261013-v1"
PRIOR = ROOT / "artifacts/strategy-screen-20261013-v1"


def metric(nav):
    v = np.r_[200000.0, nav.equity.to_numpy()]
    r = v[1:] / v[:-1] - 1
    d = -(v / np.maximum.accumulate(v) - 1).min()
    c = (v[-1] / v[0]) ** (252 / (len(v) - 1)) - 1
    annual = {
        y: float(np.prod(1 + g.ret) - 1)
        for y, g in pd.DataFrame({"year": nav.date.str[:4], "ret": r}).groupby("year")
    }
    return {
        "net_return": float(v[-1] / v[0] - 1),
        "profit_cny": float(v[-1] - v[0]),
        "cagr": float(c),
        "max_drawdown": float(d),
        "calmar": float(c / d) if d else None,
        "mean_exposure": float((nav.market_value / nav.equity).mean()),
        "annual_returns": annual,
    }


def load_etfs():
    # Dates are effective exchange ex-right dates, not the preceding record dates.
    splits = {
        "510500": [("2022-08-29", 1.14539)],
        "512800": [("2025-07-07", 2.0)],
        "512890": [("2021-10-25", 2.0)],
        "510300": [],
    }
    prices, actions = {}, {}
    for code in ["510300", "510500", "512800", "512890"]:
        raw = json.loads((RUN / f"raw/{code}-bars0.txt").read_text())["data"]["klines"]
        frame = (
            pd.DataFrame(
                [s.split(",") for s in raw],
                columns=[
                    "date",
                    "open",
                    "close",
                    "high",
                    "low",
                    "volume",
                    "amount",
                    "amp",
                    "pct",
                    "change",
                    "turnover",
                ],
            )
            .set_index("date")
            .astype(float)
        )
        assert frame.index.is_unique and frame.index.max() == "2025-08-06"
        prices[code] = frame
        events = {day: {"split": ratio, "dividend": 0.0} for day, ratio in splits[code]}
        for row in json.loads((RUN / f"raw/{code}-action-table.json").read_text()):
            if len(row) == 5 and "2021-09-01" <= row[2] <= "2025-08-06":
                amount = float(row[3].replace("每10份派现金", "").replace("元", "")) / 10
                events[row[2]] = {"split": 1.0, "dividend": amount, "actual_pay_date": row[4]}
        actions[code] = events
    (RUN / "normalized-etf-actions.json").write_text(
        json.dumps(actions, ensure_ascii=False, indent=2) + "\n"
    )
    return prices, actions


def etf_account(prices, actions, dates, weights, scenario, rebalance, cash_delay=60):
    """Invest integer fund shares using frozen weights and prior-close account size."""
    cash = 200000.0
    shares = {c: 0 for c in weights}
    receivables = []
    last = {c: float(prices[c].loc[prices[c].index < dates[0], "close"].iloc[-1]) for c in weights}
    offset = {"phase5": 5, "phase10": 10, "phase15": 15}.get(scenario, 0)
    delay = int(scenario == "delay1")
    slip = 0.002 if scenario == "double_cost" else 0.001
    comm = 0.001 if scenario == "double_cost" else 0.0005
    first = offset + delay
    due = set(range(first, len(dates), 20)) if rebalance else {first}
    nav = []
    fills = []
    events = []
    held = []
    history = {-1: 200000.0}
    orders = []
    remaining_budget = {c: 200000.0 * w for c, w in weights.items()}
    for i, day in enumerate(dates):
        for c in weights:
            action = actions[c].get(day)
            if action:
                entitled = shares[c] * action["dividend"]
                if entitled:
                    payable = (
                        action.get("actual_pay_date")
                        if cash_delay is None
                        else str(date.fromisoformat(day) + timedelta(days=cash_delay))
                    )
                    receivables.append((payable, entitled, c))
                    events.append(
                        {
                            "date": day,
                            "code": c,
                            "kind": "dividend_entitlement",
                            "amount": entitled,
                            "due": payable,
                        }
                    )
                old = shares[c]
                shares[c] = math.floor(shares[c] * action["split"] + 1e-8)
                last[c] = (last[c] - action["dividend"]) / action["split"]
                if old and action["split"] != 1:
                    events.append(
                        {
                            "date": day,
                            "code": c,
                            "kind": "split",
                            "old_shares": old,
                            "new_shares": shares[c],
                            "amount": 0.0,
                        }
                    )
        remain = []
        for payable, amount, c in receivables:
            if payable <= day:
                cash += amount
                events.append({"date": day, "code": c, "kind": "dividend_paid", "amount": amount})
            else:
                remain.append((payable, amount, c))
        receivables = remain
        if i in due or (
            not rebalance and i >= first and any(v > 0 for v in remaining_budget.values())
        ):
            signal = i - 1 - delay if rebalance else first - 1 - delay
            budget_eq = history.get(signal, 200000.0)
            targets = {}
            for c, w in weights.items():
                if day not in prices[c].index:
                    continue
                opening = float(prices[c].loc[day, "open"])
                targets[c] = (
                    math.floor(budget_eq * w / opening / 100) * 100
                    if rebalance
                    else shares[c] + math.floor(remaining_budget[c] / opening / 100) * 100
                )
            # Sell before buying. A per-order fee is charged, including minimum 5 yuan.
            for side in ["sell", "buy"]:
                for c, target in targets.items():
                    want = target - shares[c]
                    if (side == "buy" and want <= 0) or (side == "sell" and want >= 0):
                        continue
                    opening = float(prices[c].loc[day, "open"])
                    price = opening * (1 + slip if side == "buy" else 1 - slip)
                    reference = last[c]
                    upper = math.floor(reference * 1.1 * 1000 + 0.5) / 1000
                    lower = math.floor(reference * 0.9 * 1000 + 0.5) / 1000
                    prior = prices[c].loc[prices[c].index < day].tail(20)
                    capacity = float(prior.amount.mean()) * 0.005
                    cap = math.floor(capacity / price / 100) * 100
                    q = min(abs(want), cap)
                    order = {"date": day, "code": c, "side": side, "status": "blocked"}
                    orders.append(order)
                    if (
                        (side == "buy" and (opening >= upper or price > upper))
                        or (side == "sell" and (opening <= lower or price < lower))
                        or prices[c].loc[day, "volume"] <= 0
                    ):
                        continue
                    if side == "buy":
                        q = math.floor(q / 100) * 100
                        max_budget = (
                            min(cash, max(0.0, budget_eq * weights[c] - shares[c] * price))
                            if rebalance
                            else min(cash, remaining_budget[c])
                        )
                        while q > 0 and q * price + max(5.0, q * price * comm) > max_budget + 1e-8:
                            q -= 100
                    else:
                        q = math.floor(q / 100) * 100
                    if q <= 0:
                        continue
                    value = q * price
                    fee = max(5.0, value * comm)
                    cash += value - fee if side == "sell" else -value - fee
                    shares[c] += q if side == "buy" else -q
                    if not rebalance and side == "buy":
                        remaining_budget[c] = max(0.0, remaining_budget[c] - value - fee)
                        if remaining_budget[c] < 100 * price + max(5.0, 100 * price * comm):
                            remaining_budget[c] = 0.0
                    assert cash >= -1e-7 and shares[c] >= 0
                    fills.append(
                        {
                            "date": day,
                            "code": c,
                            "side": side,
                            "shares": q,
                            "price": price,
                            "reference_open": opening,
                            "notional": value,
                            "commission": fee,
                            "slippage": q * abs(price - opening),
                            "signal_date": dates[signal] if signal >= 0 else "2021-08-31",
                        }
                    )
                    order.update(status="filled", shares=q)
        for c in weights:
            if day in prices[c].index:
                last[c] = float(prices[c].loc[day, "close"])
            held.append(
                {
                    "date": day,
                    "code": c,
                    "shares": shares[c],
                    "mark": last[c],
                    "value": shares[c] * last[c],
                }
            )
        mv = sum(shares[c] * last[c] for c in weights)
        rec = sum(a for _, a, _ in receivables)
        equity = cash + mv + rec
        history[i] = equity
        nav.append(
            {"date": day, "equity": equity, "cash": cash, "market_value": mv, "receivable": rec}
        )
    nf = pd.DataFrame(nav)
    f = pd.DataFrame(fills)
    result = metric(nf)
    result.update(
        scenario=scenario,
        trades=len(f),
        commission=float(f.commission.sum()),
        slippage=float(f.slippage.sum()),
        minimum_fee_fills=int((f.commission <= 5.0000001).sum()),
        final_equity=float(nf.equity.iloc[-1]),
    )
    return result, {
        "nav": nf,
        "fills": f,
        "holdings": pd.DataFrame(held),
        "cash_events": pd.DataFrame(events),
        "orders": pd.DataFrame(orders),
    }


def attribution(bank_ids, scenario):
    folder = PRIOR / "results" / scenario / "lowvol60"
    fills = pd.read_parquet(folder / "fills.parquet")
    hold = pd.read_parquet(folder / "holdings.parquet")
    events = pd.read_parquet(folder / "cash_events.parquet")
    nav = pd.read_parquet(folder / "nav.parquet")
    ids = sorted(set(hold.instrument_id))
    days = nav.date.tolist()
    market = (
        hold.pivot(index="date", columns="instrument_id", values="value")
        .reindex(index=days, columns=ids)
        .fillna(0.0)
    )
    movement = market.diff().fillna(market.iloc[0])
    flows = fills.assign(
        flow=np.where(fills.side.eq("sell"), fills.notional, -fills.notional) - fills.fees
    )
    cf = (
        flows.pivot_table(index="date", columns="instrument_id", values="flow", aggfunc="sum")
        .reindex(index=days, columns=ids)
        .fillna(0.0)
    )
    dv = (
        events.loc[events.kind.eq("dividend_entitlement")]
        .pivot_table(index="date", columns="instrument_id", values="amount", aggfunc="sum")
        .reindex(index=days, columns=ids)
        .fillna(0.0)
    )
    pnl = movement + cf + dv
    total = float(pnl.sum().sum())
    expected = float(nav.equity.iloc[-1] - 200000.0)
    assert abs(total - expected) < 1e-6, (total, expected)
    bank_cols = [c for c in ids if c in bank_ids]
    daily = pd.DataFrame(
        {
            "date": days,
            "bank_pnl": pnl[bank_cols].sum(axis=1).values,
            "nonbank_pnl": pnl.drop(columns=bank_cols).sum(axis=1).values,
            "bank_value": market[bank_cols].sum(axis=1).values,
            "stock_value": market.sum(axis=1).values,
        }
    )
    daily["year"] = daily.date.str[:4]
    annual = (
        daily.groupby("year")[["bank_pnl", "nonbank_pnl"]].sum().reset_index().to_dict("records")
    )
    bank = float(daily.bank_pnl.sum())
    nonbank = float(daily.nonbank_pnl.sum())
    security = pnl.sum().rename("net_pnl").to_frame()
    security["bank"] = security.index.isin(bank_ids)
    return (
        {
            "scenario": scenario,
            "bank_net_pnl": bank,
            "nonbank_net_pnl": nonbank,
            "total_net_pnl": total,
            "bank_share_of_net_profit": bank / total,
            "bank_share_of_stock_value_days": float(
                daily.bank_value.sum() / daily.stock_value.sum()
            ),
            "annual_pnl": annual,
        },
        daily,
        security,
    )


def main():
    prices, actions = load_etfs()
    dates = pd.read_parquet(PRIOR / "results/base/lowvol60/nav.parquet").date.tolist()
    scenarios = ["base", "double_cost", "delay1", "phase5", "phase10", "phase15"]
    output = RUN / "etfs"
    output.mkdir()
    summaries = []
    for sc in scenarios:
        for code in prices:
            for rule, w, rebalance in [
                ("hold90", 0.9, False),
                ("hold100", 1.0, False),
                ("rebalance90", 0.9, True),
            ]:
                result, frames = etf_account(prices, actions, dates, {code: w}, sc, rebalance)
                result.update(code=code, rule=rule)
                summaries.append(result)
                folder = output / sc / code / rule
                folder.mkdir(parents=True)
                for k, frame in frames.items():
                    frame.to_parquet(folder / (k + ".parquet"), index=False)
        result, frames = etf_account(
            prices, actions, dates, {"510300": 0.45, "512800": 0.45}, sc, True
        )
        result.update(code="bank45_broad45", rule="rebalance90")
        summaries.append(result)
        folder = output / sc / "bank45_broad45" / "rebalance90"
        folder.mkdir(parents=True)
        for k, frame in frames.items():
            frame.to_parquet(folder / (k + ".parquet"), index=False)
    (RUN / "etf-summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n"
    )
    pd.DataFrame([{k: v for k, v in x.items() if k != "annual_returns"} for x in summaries]).to_csv(
        RUN / "etf-summary.csv", index=False
    )
    banks = set(pd.read_csv(RUN / "bank-identities.csv").instrument_id)
    parts = []
    for sc in scenarios:
        result, daily, security = attribution(banks, sc)
        parts.append(result)
        daily.to_csv(RUN / f"attribution-daily-{sc}.csv", index=False)
        security.to_csv(RUN / f"attribution-security-{sc}.csv")
    (RUN / "attribution.json").write_text(json.dumps(parts, ensure_ascii=False, indent=2) + "\n")
    print(
        pd.DataFrame(summaries)
        .query("scenario=='base' and rule=='hold90'")[
            ["code", "net_return", "cagr", "max_drawdown", "mean_exposure"]
        ]
        .to_string(index=False)
    )
    print(json.dumps(parts[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
