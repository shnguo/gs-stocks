"""Backtest fundamental turnaround baskets (困境反转) on the US index union.

Usage: uv run python scripts/turnaround_fundamental.py configs/turnaround-fundamental-v1.json
"""
import argparse
import importlib.util
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.baskets import BasketBook
from quant_research.factors import composite, evaluate, forward_returns, month_end_dates, percentile
from quant_research.storage import file_hash, write_json
from quant_research.us_fundamentals import REVENUE, instant_asof, ttm_asof

BASE = Path(__file__).resolve().parents[1]
FIELDS = ["net_income", "operating_cash", "revenue", "gross_profit", "assets", "assets_current",
          "liabilities_current", "lt_debt"]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, BASE / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def company_inputs(task):
    """Point-in-time fundamentals for one CIK at each requested date."""
    cik, facts, dates = task
    rows = []
    for date in dates:
        revenue = ttm_asof(facts, REVENUE, date)
        gross = ttm_asof(facts, ["gross_profit"], date)
        if gross is None and revenue is not None:
            cost = ttm_asof(facts, ["cost_of_revenue", "cost_of_goods"], date)
            gross = revenue - cost if cost is not None else None
        rows.append({"cik": cik, "date": date,
                     "net_income": ttm_asof(facts, ["net_income"], date),
                     "operating_cash": ttm_asof(facts, ["operating_cash", "operating_cash_cont"], date),
                     "revenue": revenue, "gross_profit": gross,
                     "assets": instant_asof(facts, ["assets"], date),
                     "assets_current": instant_asof(facts, ["assets_current"], date),
                     "liabilities_current": instant_asof(facts, ["liabilities_current"], date),
                     "lt_debt": instant_asof(facts, ["lt_debt", "lt_debt_total"], date)})
    return rows


def fundamentals_table(members, month_ends, facts_paths, lag, workers):
    """Inputs at every member date and at the month end `lag` months earlier."""
    facts = pd.concat([pd.read_parquet(BASE / p) for p in facts_paths], ignore_index=True)
    wanted = members.dropna(subset=["cik"]).astype({"cik": int})
    facts = facts.loc[facts.cik.isin(set(wanted.cik))]
    by_cik = dict(tuple(facts.groupby("cik")))
    position = {d: i for i, d in enumerate(month_ends)}
    tasks = []
    for cik, group in wanted.groupby("cik"):
        if cik not in by_cik:
            continue
        dates = set(group.date)
        dates |= {month_ends[position[d] - lag] for d in group.date if d in position and position[d] >= lag}
        tasks.append((cik, by_cik[cik], sorted(dates)))
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, part in enumerate(pool.map(company_inputs, tasks, chunksize=8)):
            rows += part
            if (i + 1) % 500 == 0:
                print(f"  fundamentals {i + 1}/{len(tasks)} companies", flush=True)
    return pd.DataFrame(rows)


def ratio(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.isfinite(a) & np.isfinite(b) & (b > 0), a / b, np.nan)


def checklist(now, prev, share_change, issuance):
    """Piotroski signals (1 pass, 0 fail, nan unavailable), plus roa and revenue growth."""
    roa, roa_prev = ratio(now["net_income"], now["assets"]), ratio(prev["net_income"], prev["assets"])
    lev, lev_prev = ratio(now["lt_debt"], now["assets"]), ratio(prev["lt_debt"], prev["assets"])
    cur, cur_prev = (ratio(now["assets_current"], now["liabilities_current"]),
                     ratio(prev["assets_current"], prev["liabilities_current"]))
    gm, gm_prev = ratio(now["gross_profit"], now["revenue"]), ratio(prev["gross_profit"], prev["revenue"])
    turn, turn_prev = ratio(now["revenue"], now["assets"]), ratio(prev["revenue"], prev["assets"])
    cfo = now["operating_cash"]

    def signal(cond, *inputs):
        ok = np.all([np.isfinite(x) for x in inputs], axis=0)
        return np.where(ok, cond.astype(float), np.nan)

    with np.errstate(invalid="ignore"):
        signals = [signal(roa > 0, roa), signal(cfo > 0, cfo), signal(roa > roa_prev, roa, roa_prev),
                   signal(cfo > now["net_income"], cfo, now["net_income"]),
                   signal(lev < lev_prev, lev, lev_prev), signal(cur > cur_prev, cur, cur_prev),
                   signal(share_change <= issuance, share_change), signal(gm > gm_prev, gm, gm_prev),
                   signal(turn > turn_prev, turn, turn_prev)]
        growth = ratio(now["revenue"], prev["revenue"]) - 1
    stack = np.column_stack(signals)
    return {"roa": roa, "delta_roa": roa - roa_prev, "revenue_growth": growth,
            "signals_available": np.isfinite(stack).sum(axis=1).astype(float),
            "signals_passed": np.nansum(stack, axis=1)}


def drawdown(panel, t, days, min_share):
    window = slice(max(0, t - days + 1), t + 1)
    same = np.where(panel.sequence[window] == panel.sequence[t], panel.close[window], np.nan)
    count = np.isfinite(same).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        high = np.nanmax(same, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(count >= min_share * days, panel.close[t] / high - 1, np.nan)


def month_inputs(panel, t, date, ids, cik_of, fund, values, prev_date, cfg):
    """Arrays aligned to panel ids: drawdown, checklist fields, book_to_price."""
    cik = np.array([cik_of.get(e, -1) for e in ids])
    now = {f: fund.get(date, {}).get(f, pd.Series(dtype=float)).reindex(cik).to_numpy(float) for f in FIELDS}
    prev = {f: fund.get(prev_date, {}).get(f, pd.Series(dtype=float)).reindex(cik).to_numpy(float)
            for f in FIELDS} if prev_date else {f: np.full(len(ids), np.nan) for f in FIELDS}
    v_now = values.get(date, pd.DataFrame(columns=["market_cap", "book_to_price"])).reindex(ids)
    v_prev = values.get(prev_date, pd.DataFrame(columns=["market_cap"])).reindex(ids) if prev_date else None
    share_change = np.full(len(ids), np.nan)
    if v_prev is not None:
        tp = panel.index(prev_date)
        with np.errstate(divide="ignore", invalid="ignore"):
            share_change = ((v_now.market_cap.to_numpy(float) / panel.close[t])
                            / (v_prev.market_cap.to_numpy(float) / panel.close[tp]) - 1)
    out = checklist(now, prev, share_change, cfg["issuance_threshold"])
    out.update({"drawdown": drawdown(panel, t, cfg["drawdown_days"], cfg["drawdown_min_share"]),
                "book_to_price": v_now.book_to_price.to_numpy(float),
                "net_income": now["net_income"], "net_income_prev": prev["net_income"],
                "operating_cash": now["operating_cash"]})
    with np.errstate(invalid="ignore", divide="ignore"):
        out["fscore"] = np.where(out["signals_available"] >= cfg["fscore_min_signals"],
                                 out["signals_passed"] / out["signals_available"], np.nan)
    return out


def turnaround_picks(x, mask, cfg):
    dd_ok = mask & np.isfinite(x["drawdown"])
    beaten = dd_ok & (percentile(x["drawdown"], dd_ok) <= cfg["beaten_share"])
    bp_ok = mask & np.isfinite(x["book_to_price"])
    cheap = bp_ok & (percentile(x["book_to_price"], bp_ok) > 1 - cfg["cheap_share"])
    with np.errstate(invalid="ignore"):
        improving = (x["delta_roa"] > 0) & (x["revenue_growth"] > 0)
        was_loss = x["net_income_prev"] < 0
        high_f, low_f = x["fscore"] >= cfg["fscore_high"], x["fscore"] <= cfg["fscore_low"]
        picks = {"beaten_all": beaten,
                 "profit_turn": mask & was_loss & (x["net_income"] > 0),
                 "loss_improving": mask & was_loss & (x["delta_roa"] > 0),
                 "beaten_improving": beaten & improving,
                 "beaten_improving_cash": beaten & improving & (x["operating_cash"] > 0),
                 "beaten_high_f": beaten & high_f, "beaten_low_f": beaten & low_f,
                 "cheap_high_f": cheap & high_f, "cheap_low_f": cheap & low_f}
    return picks, beaten


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    cfg = json.loads(parser.parse_args().config.read_text())
    output = BASE / cfg["output"]
    if output.exists():
        raise FileExistsError(f"Use a new output path: {output}")
    started = time.time()
    us = json.loads((BASE / cfg["us_config"]).read_text())
    panel, months, source_hash = load_script("physics_robustness").us_months(us)
    source = BASE / us["universe_output"]
    members = pd.read_parquet(source / "membership.parquet")
    month_ends = month_end_dates(panel.dates)
    member_dates = {d for d, *_ in months}
    fund_rows = fundamentals_table(members.loc[members.date.isin(member_dates)], month_ends, cfg["facts"],
                                   cfg["lag_months"], cfg["workers"])
    print(f"fundamentals: {len(fund_rows)} rows in {time.time() - started:.0f}s", flush=True)
    fund = {d: g.set_index("cik") for d, g in fund_rows.groupby("date")}
    values = pd.read_parquet(BASE / cfg["value_inputs"])
    values = {d: g.drop_duplicates("entity").set_index("entity") for d, g in values.groupby("date")}
    position = {d: i for i, d in enumerate(month_ends)}
    ids = np.asarray(panel.ids)

    book = BasketBook(cfg["baskets"], us["cost_one_way"])
    cohorts = []
    long_h = max(cfg["horizon_months"])
    for i, (date, following, mask, _) in enumerate(months):
        t = panel.index(date)
        p = position.get(date)
        prev_date = month_ends[p - cfg["lag_months"]] if p is not None and p >= cfg["lag_months"] else None
        month = members.loc[members.date == date].dropna(subset=["cik"]).drop_duplicates("entity")
        cik_of = dict(zip(month.entity, month.cik.astype(int)))
        x = month_inputs(panel, t, date, ids, cik_of, fund, values, prev_date, cfg)
        returns, _ = forward_returns(panel, date, following)
        returns = np.where(mask, returns, np.nan)
        long_returns = None
        if i + long_h - 1 < len(months):
            long_returns, _ = forward_returns(panel, date, months[i + long_h - 1][1])
            long_returns = np.where(mask, long_returns, np.nan)
        keep = lambda v: np.where(mask, v, np.nan)  # noqa: E731
        cohorts.append({"date": date, "ids": ids, "returns": returns, "delta_roa": keep(x["delta_roa"]),
                        "revenue_growth": keep(x["revenue_growth"]), "fscore": keep(x["fscore"]),
                        "turnaround_score": composite([-x["drawdown"], x["delta_roa"], x["revenue_growth"]], mask, 3)})
        valid = mask & np.isfinite(returns)
        if valid.sum() < 100:
            continue
        picks, _ = turnaround_picks(x, mask, cfg)
        book.add(date, ids, returns, valid, picks, long_returns)

    scores = {}
    for name in cfg["scores"]:
        out = evaluate(cohorts, name, us["quantiles"], us["cost_one_way"], us["bootstrap_block_months"],
                       us["bootstrap_repetitions"])
        out.pop("daily")
        scores[name] = {k: out[k] for k in ["periods", "rank_ic_mean", "rank_ic_t", "rank_ic_bootstrap",
                                            "quantile_mean_monthly", "long_short_net", "top_excess_net"]}
    output.mkdir(parents=True)
    book.frame().to_csv(output / "baskets.csv", index=False)
    fund_rows.to_parquet(output / "fundamental-inputs.parquet", index=False)
    write_json(output / "results.json", {
        "version": cfg["version"], "config": cfg, "source_manifest_sha256": source_hash,
        "facts_sha256": {p: file_hash(BASE / p) for p in cfg["facts"]},
        "baskets": book.summary(cfg["baskets"], us["bootstrap_block_months"], us["bootstrap_repetitions"]),
        "scores": scores, "elapsed_seconds": round(time.time() - started, 1)})
    print(f"wrote {output} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
