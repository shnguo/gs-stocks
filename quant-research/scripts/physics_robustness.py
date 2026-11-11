"""Robustness checks for the drift (jump-days-removed) momentum factor, US and A-shares.

Usage: uv run python scripts/physics_robustness.py configs/physics-robustness-v1.json

The checks are fixed in the config before running; every result is reported.
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.evaluation import block_bootstrap
from quant_research.factors import (
    evaluate,
    forward_returns,
    group_percentile,
    momentum,
    month_end_dates,
    physics_momentum,
    price_panel,
    residualize,
)
from quant_research.storage import Snapshot, file_hash, write_json
from quant_research.us_sources import gap_segments

BASE = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, BASE / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def us_months(cfg, industry_path=None):
    """(panel, [(date, following, mask, sector)]) for the US index union. Sector is the
    current Fama-French 12 industry from SEC SIC codes (not point-in-time), or None."""
    source = BASE / cfg["universe_output"]
    bars = pd.read_parquet(source / "bars.parquet").rename(columns={"entity": "instrument_id"})
    bars = bars.assign(factor=1.0, source_trade_status=1.0, source_is_st=0.0)
    calendar = pd.read_parquet(source / "calendar.parquet").date.tolist()
    segment = bars.groupby("instrument_id", group_keys=False).date.apply(
        lambda d: gap_segments(d, calendar, cfg["max_gap_sessions"]))
    bars["sequence_id"] = bars.instrument_id + "#" + segment.astype(str)
    panel = price_panel(bars)
    members = pd.read_parquet(source / "membership.parquet")
    for flag in ["in_sp500", "in_ndx", "in_r1000"]:
        members[flag] = members[flag].astype("boolean").fillna(False).astype(bool)
    members = members.loc[members.in_sp500 | members.in_ndx | members.in_r1000]
    sector_of = None
    if industry_path:
        industry = pd.read_parquet(BASE / industry_path)
        sector_of = industry.dropna(subset=["ff12"]).set_index("cik").ff12
    ends = sorted(members.date.unique())
    months = []
    for date, following in zip(ends[:-1], ends[1:]):
        if panel.index(following) + 1 >= len(panel.dates):
            break
        month = members.loc[members.date == date].drop_duplicates("entity").set_index("entity")
        listed = panel.ids.isin(set(month.index))
        sector = None
        if sector_of is not None:
            sector = month.cik.map(sector_of).reindex(panel.ids).to_numpy(dtype=object)
        months.append((date, following, listed & panel.traded[panel.index(date)], sector))
    return panel, months, file_hash(source / "manifest.json")


def cn_months(cfg):
    study = load_script("factor_study")
    snapshot = Snapshot(BASE / cfg["snapshot"])
    instruments = snapshot.tables["instruments"]
    panel = price_panel(snapshot.tables["bars"])
    ends = [d for d in month_end_dates(panel.dates) if panel.index(d) >= cfg["momentum"]["lookback"]]
    months = []
    for date, following in study.signal_pairs(ends, panel, cfg["sealed_from"]):
        mask = study.eligible(panel, date, instruments, min_listing_days=cfg["min_listing_days"],
                              exclude_exchanges=cfg["exclude_exchanges"])
        months.append((date, following, mask, None))
    return panel, months, file_hash(BASE / cfg["snapshot"] / "manifest.json")


def cohorts_for(panel, months, market_cfg, rcfg):
    m, s = market_cfg["momentum"], rcfg["short_window"]
    cohorts = []
    for date, following, mask, sector in months:
        returns, _ = forward_returns(panel, date, following)
        keep = lambda x: np.where(mask, x, np.nan)  # noqa: E731
        mom = keep(momentum(panel, date, m["lookback"], m["skip"], m["min_traded"]))
        c = {"date": date, "ids": np.asarray(panel.ids), "returns": keep(returns), "momentum_12_1": mom}
        for sigma in rcfg["jump_sigmas"]:
            phys = physics_momentum(panel, date, m["lookback"], m["skip"], m["min_traded"], sigma)
            c[f"drift_{sigma}"] = keep(phys["drift"])
            c[f"impulse_{sigma}"] = keep(phys["impulse"])
            if sigma == rcfg["base_sigma"]:
                base = phys
        vol, size, drift = keep(base["volatility"]), keep(base["traded_value"]), c[f"drift_{rcfg['base_sigma']}"]
        c["low_volatility"] = -vol
        c["drift_ex_vol_size"] = residualize(drift, [vol, size], mask)
        c["drift_ex_momentum"] = residualize(drift, [mom], mask)
        c["drift_ex_all"] = residualize(drift, [mom, vol, size], mask)
        c["size"] = size
        if sector is not None:
            c["drift_in_sector"] = group_percentile(drift, sector, mask)
            c["momentum_in_sector"] = group_percentile(mom, sector, mask)
            c["drift_ex_sector"] = residualize(drift, [], mask, sector)
            c["drift_ex_sector_all"] = residualize(drift, [mom, vol, size], mask, sector)
            c["sector_known"] = int((mask & pd.notna(sector)).sum())
        short = physics_momentum(panel, date, s["lookback"], s["skip"], s["min_traded"], rcfg["base_sigma"])
        c["momentum_6_1"] = keep(np.expm1(short["total"]))
        c["drift_6_1"] = keep(short["drift"])
        cohorts.append(c)
    return cohorts


def subset(cohorts, keep_fn):
    """Same cohorts with returns hidden outside the kept stocks."""
    out = []
    for c in cohorts:
        keep = keep_fn(c)
        out.append({**c, "returns": np.where(keep, c["returns"], np.nan)})
    return out


def summary(cohorts, factor, cfg):
    out = evaluate(cohorts, factor, cfg["quantiles"], cfg["cost_one_way"],
                   cfg["bootstrap_block_months"], cfg["bootstrap_repetitions"])
    daily = pd.DataFrame(out.pop("daily"))
    keep = {k: out[k] for k in ["periods", "mean_stocks", "rank_ic_mean", "rank_ic_t"]}
    keep["rank_ic_ci"] = out["rank_ic_bootstrap"].get("confidence_interval")
    keep["long_short_net_annual"] = out["long_short_net"]["annual_return"]
    keep["top_excess_net_annual"] = out["top_excess_net"]["annual_return"]
    return keep, daily.set_index("date").rank_ic


def gap(ic_a, ic_b, cfg):
    diff = (ic_a - ic_b).dropna().to_numpy()
    b = block_bootstrap(diff, block_days=cfg["bootstrap_block_months"], repetitions=cfg["bootstrap_repetitions"])
    return {"mean": float(diff.mean()), "t": float(diff.mean() / diff.std() * np.sqrt(len(diff))),
            "ci": b.get("confidence_interval"), "share_positive": float((diff > 0).mean())}


def run_market(cohorts, market_cfg, rcfg):
    res = {}
    base, ic_mom = summary(cohorts, "momentum_12_1", market_cfg)
    res["momentum_12_1"] = base
    for sigma in rcfg["jump_sigmas"]:
        d, ic_d = summary(cohorts, f"drift_{sigma}", market_cfg)
        i, _ = summary(cohorts, f"impulse_{sigma}", market_cfg)
        res[f"sigma_{sigma}"] = {"drift": d, "impulse": i, "drift_minus_momentum": gap(ic_d, ic_mom, market_cfg)}
    for name in ["low_volatility", "drift_ex_vol_size", "drift_ex_momentum", "drift_ex_all"]:
        res[name], _ = summary(cohorts, name, market_cfg)
    if "drift_in_sector" in cohorts[-1]:
        res["sector_coverage"] = float(np.mean([c["sector_known"] / max(1, np.isfinite(c["returns"]).sum())
                                                for c in cohorts]))
        _, ic_base = summary(cohorts, f"drift_{rcfg['base_sigma']}", market_cfg)
        for name in ["drift_in_sector", "momentum_in_sector", "drift_ex_sector", "drift_ex_sector_all"]:
            s_x, ic_x = summary(cohorts, name, market_cfg)
            res[name] = {**s_x, "minus_momentum": gap(ic_x, ic_mom, market_cfg)}
        _, ic_ms = summary(cohorts, "momentum_in_sector", market_cfg)
        _, ic_ds = summary(cohorts, "drift_in_sector", market_cfg)
        res["drift_in_sector"]["minus_momentum_in_sector"] = gap(ic_ds, ic_ms, market_cfg)
    drift = f"drift_{rcfg['base_sigma']}"
    half = len(cohorts) // 2
    for label, part in [("first_half", cohorts[:half]), ("second_half", cohorts[half:])]:
        s_d, ic_d = summary(part, drift, market_cfg)
        s_m, ic_m = summary(part, "momentum_12_1", market_cfg)
        s_i, _ = summary(part, f"impulse_{rcfg['base_sigma']}", market_cfg)
        res[label] = {"first": part[0]["date"], "last": part[-1]["date"], "drift": s_d, "momentum": s_m,
                      "impulse": s_i, "drift_minus_momentum": gap(ic_d, ic_m, market_cfg)}
    for label, large in [("large_half", True), ("small_half", False)]:
        def keep(c, large=large):
            above = c["size"] > np.nanmedian(c["size"])
            return np.isfinite(c["size"]) & (above if large else ~above)
        part = subset(cohorts, keep)
        s_d, ic_d = summary(part, drift, market_cfg)
        s_m, ic_m = summary(part, "momentum_12_1", market_cfg)
        res[label] = {"drift": s_d, "momentum": s_m, "drift_minus_momentum": gap(ic_d, ic_m, market_cfg)}
    s_d, ic_d = summary(cohorts, "drift_6_1", market_cfg)
    s_m, ic_m = summary(cohorts, "momentum_6_1", market_cfg)
    res["window_6_1"] = {"drift": s_d, "momentum": s_m, "drift_minus_momentum": gap(ic_d, ic_m, market_cfg)}
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    rcfg = json.loads(parser.parse_args().config.read_text())
    output = BASE / rcfg["output"]
    if output.exists():
        raise FileExistsError(f"Use a new output path: {output}")
    started = time.time()
    results, lineage = {}, {}
    for market, key in [("us", "us_config"), ("cn", "cn_config")]:
        market_cfg = json.loads((BASE / rcfg[key]).read_text())
        if market == "us":
            panel, months, source_hash = us_months(market_cfg, rcfg.get("us_industry"))
        else:
            panel, months, source_hash = cn_months(market_cfg)
        cohorts = cohorts_for(panel, months, market_cfg, rcfg)
        results[market] = run_market(cohorts, market_cfg, rcfg)
        lineage[market] = {"config": rcfg[key], "source_manifest_sha256": source_hash, "months": len(cohorts)}
        print(f"{market}: {len(cohorts)} months, {time.time() - started:.0f}s", flush=True)
    output.mkdir(parents=True)
    write_json(output / "results.json", {"version": rcfg["version"], "config": rcfg, "lineage": lineage,
                                         "results": results, "elapsed_seconds": round(time.time() - started, 1)})
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
