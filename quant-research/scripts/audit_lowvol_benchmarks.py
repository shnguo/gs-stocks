"""Independent ETF cash/share audit using retained source prices; no simulator import."""

import hashlib
import json
import math
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
run = root / "artifacts/lowvol-benchmark-20261013-v1"
actions = json.loads((run / "normalized-etf-actions.json").read_text())
prices = {}
for c in actions:
    rows = json.loads((run / f"raw/{c}-bars0.txt").read_text())["data"]["klines"]
    prices[c] = {r.split(",")[0]: [float(x) for x in r.split(",")[1:]] for r in rows}
records = []
for folder in sorted((run / "etfs").glob("*/*/*")):
    scenario, code, rule = folder.relative_to(run / "etfs").parts
    nav = pd.read_parquet(folder / "nav.parquet")
    fills = pd.read_parquet(folder / "fills.parquet")
    hold = pd.read_parquet(folder / "holdings.parquet")
    codes = hold.code.unique()
    by_day = {d: f for d, f in fills.groupby("date")}
    held = {d: g.set_index("code") for d, g in hold.groupby("date")}
    cash = 200000.0
    quantities = defaultdict(float)
    marks = {c: prices[c]["2021-08-31"][1] for c in codes}
    receivables = []
    max_error = 0.0
    wealth = []
    for day in nav.itertuples():
        for c in codes:
            action = actions[c].get(day.date)
            if action:
                receivables.append(
                    (
                        str(date.fromisoformat(day.date) + timedelta(days=60)),
                        quantities[c] * action["dividend"],
                    )
                )
                quantities[c] = math.floor(quantities[c] * action["split"] + 1e-8)
                marks[c] = (marks[c] - action["dividend"]) / action["split"]
        cash += sum(v for pay, v in receivables if pay <= day.date)
        receivables = [(pay, v) for pay, v in receivables if pay > day.date]
        for f in by_day.get(day.date, pd.DataFrame()).itertuples():
            sign = 1 if f.side == "buy" else -1
            opening = prices[f.code][f.date][0]
            price = opening * (1 + sign * (0.002 if scenario == "double_cost" else 0.001))
            fee = max(5, f.shares * price * (0.001 if scenario == "double_cost" else 0.0005))
            assert abs(price - f.price) < 1e-9 and abs(fee - f.commission) < 1e-8
            assert f.shares % 100 == 0 and f.signal_date < f.date
            quantities[f.code] += sign * f.shares
            cash -= sign * f.shares * price + fee
            assert quantities[f.code] >= 0 and cash >= -1e-7
        for c in codes:
            if day.date in prices[c]:
                marks[c] = prices[c][day.date][1]
            assert quantities[c] == held[day.date].loc[c, "shares"]
            assert abs(marks[c] - held[day.date].loc[c, "mark"]) < 1e-9
        eq = cash + sum(quantities[c] * marks[c] for c in codes) + sum(v for _, v in receivables)
        error = max(abs(cash - day.cash), abs(eq - day.equity))
        max_error = max(max_error, error)
        wealth.append(eq)
    assert max_error < 1e-6
    v = np.r_[200000, wealth]
    dd = float(-(v / np.maximum.accumulate(v) - 1).min())
    net = float(v[-1] / 200000 - 1)
    records.append(
        {
            "scenario": scenario,
            "code": code,
            "rule": rule,
            "max_cash_nav_error": max_error,
            "net_return": net,
            "max_drawdown": dd,
            "fills": len(fills),
        }
    )
summary = pd.read_csv(run / "etf-summary.csv").set_index(["scenario", "code", "rule"])
for row in records:
    old = summary.loc[(row["scenario"], row["code"], row["rule"])]
    assert (
        abs(row["net_return"] - old.net_return) < 1e-10
        and abs(row["max_drawdown"] - old.max_drawdown) < 1e-10
    )
paths = (
    [
        run / "protocol.json",
        run / "normalized-etf-actions.json",
        run / "capture-manifest.json",
        run / "etf-summary.csv",
        Path(__file__),
        root / "scripts/validate_lowvol_benchmarks.py",
    ]
    + list((run / "raw").glob("*-bars0.txt"))
    + list((run / "etfs").rglob("*.parquet"))
)
result = {
    "audited_at": datetime.now(timezone.utc).isoformat(),
    "accounts": len(records),
    "records": records,
    "max_cash_nav_error": max(x["max_cash_nav_error"] for x in records),
    "source_hashes": {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
    },
    "source_prices_crosschecked_exactly": True,
    "sealed_return_rows_used": 0,
}
(run / "etf-independent-audit.json").write_text(json.dumps(result, indent=2) + "\n")
print({k: v for k, v in result.items() if k not in ["records", "source_hashes"]})
