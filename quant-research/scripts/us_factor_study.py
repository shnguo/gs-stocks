"""Monthly momentum / reversal / value study on point-in-time US index members.

Usage: uv run python scripts/us_factor_study.py configs/us-factor-v1.json

Universe on each month end: S&P 500, Nasdaq-100 or rebuilt Russell 1000 member with
a traded bar that day. Entry at the next session's open, exit at the open after the
next month end. Value inputs come only from SEC filings filed by the signal date.
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.factors import (
    PHYSICS_FACTORS,
    composite,
    correlations,
    evaluate,
    forward_returns,
    momentum,
    performance,
    physics_factors,
    physics_momentum,
    price_panel,
    reversal,
)
from quant_research.storage import file_hash, write_json
from quant_research.us_fundamentals import fundamentals_asof, value_yields
from quant_research.us_sources import gap_segments

BASE = Path(__file__).resolve().parents[1]
UNIVERSES = {"union": lambda m: m.in_sp500 | m.in_ndx | m.in_r1000, "sp500": lambda m: m.in_sp500,
             "ndx": lambda m: m.in_ndx, "r1000": lambda m: m.in_r1000}
VALUE = ["earnings_yield", "book_to_price", "sales_yield", "cashflow_yield"]
FACTORS = ["momentum_12_1", "reversal_1m", "value", *VALUE, "small_size", "value_momentum", *PHYSICS_FACTORS]


def company_values(task):
    """All (date, entity) value inputs for one CIK."""
    cik, facts, requests, prices, (lo_turnover, hi_turnover) = task
    rows = []
    for date, entity in requests:
        fund = fundamentals_asof(facts, date)
        series = prices.get(entity)
        cap = np.nan
        if fund["shares"] and series is not None and date in series.index:
            # Market cap = filed shares x unadjusted price on the share-count date,
            # carried to the signal date by the adjusted price change (split-safe).
            before = series.loc[:fund["shares_date"]]
            anchor = before.iloc[-1] if len(before) else None
            if anchor is not None and (pd.Timestamp(fund["shares_date"]) - pd.Timestamp(before.index[-1])).days <= 10:
                cap = fund["shares"] * anchor.raw_close * series.loc[date, "close"] / anchor.close
            elif (pd.Timestamp(date) - pd.Timestamp(fund["shares_date"])).days <= 120:
                cap = fund["shares"] * series.loc[date, "raw_close"]
        if np.isfinite(cap) and series is not None:
            # Reject caps out of line with trading: mismatched share units (ADRs, share classes).
            # Volume is split-adjusted, so pair it with the adjusted close.
            recent = series.loc[:date].tail(20)
            adv = float((recent.close * recent.volume).mean())
            if not lo_turnover * cap <= adv <= hi_turnover * cap:
                cap = np.nan
        rows.append({"date": date, "entity": entity, "cik": cik, "market_cap": cap,
                     "shares_date": fund["shares_date"], **value_yields(fund, cap)})
    return rows


def value_table(members, bars, facts_path, workers, turnover_bounds):
    facts = pd.read_parquet(facts_path)
    wanted = members.dropna(subset=["cik"]).astype({"cik": int})
    facts = facts.loc[facts.cik.isin(set(wanted.cik))]
    by_cik = dict(tuple(facts.groupby("cik")))
    prices = {e: g.set_index("date")[["close", "raw_close", "volume"]].sort_index()
              for e, g in bars.loc[bars.entity.isin(set(wanted.entity))].groupby("entity")}
    tasks = []
    for cik, group in wanted.groupby("cik"):
        if cik in by_cik:
            ents = set(group.entity)
            tasks.append((cik, by_cik[cik], list(zip(group.date, group.entity)),
                          {e: prices[e] for e in ents if e in prices}, turnover_bounds))
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, part in enumerate(pool.map(company_values, tasks, chunksize=8)):
            rows += part
            if (i + 1) % 500 == 0:
                print(f"  value inputs {i + 1}/{len(tasks)} companies", flush=True)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    cfg = json.loads(parser.parse_args().config.read_text())
    source = BASE / cfg["universe_output"]
    output = BASE / cfg["study_output"]
    if output.exists():
        raise FileExistsError(f"Factor studies use new output paths: {output}")
    started = time.time()
    manifest = json.loads((source / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if file_hash(source / name) != expected:
            raise ValueError(f"Universe file changed: {name}")
    raw_bars = pd.read_parquet(source / "bars.parquet")
    members = pd.read_parquet(source / "membership.parquet")
    for flag in ["in_sp500", "in_ndx", "in_r1000"]:
        members[flag] = members[flag].astype("boolean").fillna(False).astype(bool)

    values = value_table(members, raw_bars, BASE / cfg["sec"]["facts"], cfg.get("workers", os.cpu_count()),
                         tuple(cfg["turnover_bounds"]))
    print(f"value inputs {len(values)} rows, {time.time() - started:.0f}s", flush=True)

    bars = raw_bars.rename(columns={"entity": "instrument_id"}).assign(
        factor=1.0, source_trade_status=1.0, source_is_st=0.0)
    calendar = pd.read_parquet(source / "calendar.parquet").date.tolist()
    segment = bars.groupby("instrument_id", group_keys=False).date.apply(
        lambda d: gap_segments(d, calendar, cfg["max_gap_sessions"]))
    bars["sequence_id"] = bars.instrument_id + "#" + segment.astype(str)
    panel = price_panel(bars)
    ends = sorted(members.date.unique())
    print(f"panel {len(panel.dates)} sessions x {len(panel.ids)} entities", flush=True)

    m, r, ph = cfg["momentum"], cfg["reversal"], cfg["physics"]
    cohorts = {name: [] for name in UNIVERSES}
    coverage = []
    for date, following in zip(ends[:-1], ends[1:]):
        t = panel.index(date)
        if panel.index(following) + 1 >= len(panel.dates):
            break
        month = members.loc[members.date == date].set_index("entity").reindex(panel.ids)
        flags = month[["in_sp500", "in_ndx", "in_r1000"]].astype("boolean").fillna(False).astype(bool)
        v = values.loc[values.date == date].set_index("entity").reindex(panel.ids)
        returns, reason = forward_returns(panel, date, following)
        base = {"momentum_12_1": momentum(panel, date, m["lookback"], m["skip"], m["min_traded"]),
                "reversal_1m": reversal(panel, date, r["lookback"], r["min_traded"]),
                "small_size": -np.log(v.market_cap.to_numpy(float)),
                **{k: v[k].to_numpy(float) for k in VALUE}}
        phys = physics_momentum(panel, date, m["lookback"], m["skip"], m["min_traded"], ph["jump_sigma"])
        row = {"date": date}
        for name, rule in UNIVERSES.items():
            mask = rule(flags).to_numpy() & panel.traded[t]
            c = {"date": date, "ids": np.asarray(panel.ids), "returns": np.where(mask, returns, np.nan),
                 **{k: np.where(mask, x, np.nan) for k, x in base.items()}}
            c["value"] = composite([c[k] for k in VALUE], mask, cfg["value_min_components"])
            c["value_momentum"] = composite([c["value"], c["momentum_12_1"]], mask, 2)
            c.update(physics_factors(phys, mask, c["momentum_12_1"], c["reversal_1m"]))
            cohorts[name].append(c)
            row[f"{name}_members"] = int(rule(flags).sum())
            row[f"{name}_with_return"] = int(np.isfinite(c["returns"]).sum())
            row[f"{name}_with_momentum"] = int(np.isfinite(c["momentum_12_1"]).sum())
            row[f"{name}_with_value"] = int(np.isfinite(c["value"]).sum())
        row["entry_not_tradable"] = int(((reason == 1) & np.isfinite(base["momentum_12_1"])).sum())
        coverage.append(row)

    results, daily = {}, []
    for name, items in cohorts.items():
        results[name] = {"factors": {}, "correlations": correlations(
            items, ["value", "momentum_12_1", "reversal_1m", "small_size", *PHYSICS_FACTORS])}
        for factor in FACTORS:
            out = evaluate(items, factor, cfg["quantiles"], cfg["cost_one_way"],
                           cfg["bootstrap_block_months"], cfg["bootstrap_repetitions"])
            daily += [{"universe": name, "factor": factor, **row} for row in out.pop("daily", [])]
            results[name]["factors"][factor] = out
        print(f"evaluated {name}, {time.time() - started:.0f}s", flush=True)

    output.mkdir(parents=True)
    frame = pd.DataFrame(daily)
    frame.to_csv(output / "daily.csv", index=False)
    pd.DataFrame(coverage).to_csv(output / "coverage.csv", index=False)
    values.to_parquet(output / "value-inputs.parquet", index=False)
    universe_ew = frame.loc[(frame.universe == "union") & (frame.factor == "momentum_12_1")]
    write_json(output / "results.json", {
        "version": cfg["version"], "config": cfg,
        "universe_manifest_sha256": file_hash(source / "manifest.json"),
        "facts_sha256": file_hash(BASE / cfg["sec"]["facts"]),
        "limitations": cfg["limitations"],
        "universe_equal_weight": performance(universe_ew.universe_return.to_numpy()),
        "universes": results,
        "elapsed_seconds": round(time.time() - started, 1)})
    print(f"wrote {output} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
