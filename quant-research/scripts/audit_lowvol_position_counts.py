"""Independent source/ledger recomputation for position-count research accounts."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from quant_research.storage import file_hash, utc_now


def audit(root):
    root = Path(root)
    cfg = json.loads((root / "protocol.json").read_text())
    freeze = json.loads((root / "implementation-freeze.json").read_text())
    for name, value in freeze["files"].items():
        if file_hash(Path(name)) != value or file_hash(root / "code" / Path(name).name) != value:
            raise ValueError("Implementation changed after freeze")
    result = json.loads((root / "results/comparison.json").read_text())
    source = Path(cfg["source"])
    names = set()
    for summary in result["summaries"]:
        path = root / "results" / summary["scenario"] / summary["strategy"]
        names.update(pd.read_parquet(path / "holdings.parquet").instrument_id)
    filt = (
        (ds.field("date") >= cfg["load_start"])
        & (ds.field("date") <= cfg["evaluation_end"])
        & ds.field("instrument_id").isin(sorted(names))
    )
    quotes = (
        ds.dataset(source / "bars.parquet")
        .to_table(
            columns=[
                "date",
                "instrument_id",
                "open",
                "close",
                "source_trade_status",
                "source_is_st",
            ],
            filter=filt,
        )
        .to_pandas()
        .set_index(["date", "instrument_id"])
    )
    if quotes.index.get_level_values(0).max() >= cfg["sealed_start"]:
        raise ValueError("Sealed data was read")
    action_frame = pd.read_parquet(source / "actions.parquet")
    action_frame = action_frame.loc[
        action_frame.instrument_id.isin(names)
        & action_frame.ex_date.between(cfg["evaluation_start"], cfg["evaluation_end"])
    ]
    actions = defaultdict(list)
    for row in action_frame.itertuples():
        actions[row.ex_date].append(row)
    records = []
    scenarios = {s["name"]: s for s in cfg["scenarios"]}
    for summary in result["summaries"]:
        path = root / "results" / summary["scenario"] / summary["strategy"]
        scenario = scenarios[summary["scenario"]]
        nav, fills, held = [
            pd.read_parquet(path / (name + ".parquet")) for name in ["nav", "fills", "holdings"]
        ]
        fills_by_day = {k: list(v.itertuples()) for k, v in fills.groupby("date", sort=False)}
        held_by_day = {k: v.set_index("instrument_id") for k, v in held.groupby("date", sort=False)}
        quantities, marks = {}, {}
        cash = cfg["initial_cash"]
        receivables = []
        max_cash_error = max_nav_error = 0.0
        recomputed = []
        for day in nav.itertuples():
            for action in actions[day.date]:
                symbol = action.instrument_id
                if symbol not in quantities:
                    continue
                old = quantities[symbol]
                if action.cash_per_share:
                    payable = str(
                        date.fromisoformat(day.date)
                        + timedelta(days=scenario.get("dividend_delay_days", 60))
                    )
                    receivables.append((payable, old * action.cash_per_share * 0.8))
                quantities[symbol] = math.floor(old * action.split_ratio + 1e-8)
                marks[symbol] = (marks[symbol] - action.cash_per_share) / action.split_ratio
            cash += sum(amount for payable, amount in receivables if payable <= day.date)
            receivables = [
                (payable, amount) for payable, amount in receivables if payable > day.date
            ]
            for fill in fills_by_day.get(day.date, []):
                q = quotes.loc[(day.date, fill.instrument_id)]
                if q.source_trade_status != 1 or q.source_is_st not in (0.0, 1.0):
                    raise ValueError("Filled an unknown or suspended source state")
                if fill.side == "buy" and (q.source_is_st != 0 or fill.shares % 100 != 0):
                    raise ValueError("Buy violates risk state or lot size")
                sign = 1 if fill.side == "buy" else -1
                execution = q.open * (1 + sign * scenario["slippage_bps"] / 10000)
                if abs(execution - fill.price) > 1e-9:
                    raise ValueError("Fill price differs from original open and slippage")
                amount = fill.shares * execution
                fee = max(5.0, amount * 0.0005 * scenario["commission_multiplier"])
                fee += amount * (0.00002 if day.date < "2022-04-29" else 0.00001)
                if fill.side == "sell":
                    fee += amount * (0.001 if day.date < "2023-08-28" else 0.0005)
                if abs(fee - fill.fees) > 1e-7:
                    raise ValueError("Per-order costs differ")
                if fill.signal_date and fill.signal_date >= fill.date:
                    raise ValueError("Signal is not before execution")
                cash += -sign * amount - fee
                quantities[fill.instrument_id] = (
                    quantities.get(fill.instrument_id, 0) + sign * fill.shares
                )
                marks[fill.instrument_id] = q.open
                if quantities[fill.instrument_id] <= 1e-8:
                    del quantities[fill.instrument_id]
                    del marks[fill.instrument_id]
            actual = held_by_day.get(day.date, pd.DataFrame())
            if set(quantities) != set(actual.index):
                raise ValueError("Holdings disappeared or were added without a fill/action")
            for symbol, quantity in quantities.items():
                if (day.date, symbol) in quotes.index:
                    mark = quotes.loc[(day.date, symbol), "close"]
                    if np.isfinite(mark) and mark > 0:
                        marks[symbol] = float(mark)
                if (
                    abs(quantity - actual.loc[symbol, "shares"]) > 1e-7
                    or abs(marks[symbol] - actual.loc[symbol, "mark"]) > 1e-7
                ):
                    raise ValueError("Holding differs from fills, source action or source close")
            total = (
                cash
                + sum(quantities[s] * marks[s] for s in quantities)
                + sum(v for _, v in receivables)
            )
            max_cash_error = max(max_cash_error, abs(cash - day.cash))
            max_nav_error = max(max_nav_error, abs(total - day.equity))
            recomputed.append(total)
            if cash < -1e-7 or len(quantities) > summary["position_count"]:
                raise ValueError("Borrowing or excess position count")
        values = np.r_[cfg["initial_cash"], recomputed]
        dd = -np.min(values / np.maximum.accumulate(values) - 1)
        net = values[-1] / cfg["initial_cash"] - 1
        if (
            max_nav_error > 1e-6
            or max_cash_error > 1e-6
            or abs(net - summary["net_return"]) > 1e-10
            or abs(dd - summary["max_drawdown"]) > 1e-10
        ):
            raise ValueError("Independent net return, drawdown or cash differs")
        records.append(
            dict(
                strategy=summary["strategy"],
                scenario=summary["scenario"],
                fills=len(fills),
                nav_rows=len(nav),
                max_cash_error=max_cash_error,
                max_nav_error=max_nav_error,
                net_return=float(net),
                max_drawdown=float(dd),
            )
        )
    source_hashes = {str(p): file_hash(p) for p in root.rglob("*") if p.is_file()}
    source_hashes[str(Path(__file__))] = file_hash(Path(__file__))
    return dict(
        audited_at=utc_now(),
        accounts=len(records),
        records=records,
        original_quote_rows=len(quotes),
        sealed_price_rows_read=0,
        shares_cash_actions_and_nav_independently_replayed=True,
        daily_quotes_and_assumed_execution_not_live_fills=True,
        formal_profit_qualified=False,
        source_hashes=source_hashes,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Audit output must be new")
    value = audit(args.root)
    with args.output.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                k: value[k]
                for k in [
                    "accounts",
                    "original_quote_rows",
                    "shares_cash_actions_and_nav_independently_replayed",
                ]
            }
        )
    )
