"""Monthly momentum / reversal / value factor study.

Usage: uv run python scripts/factor_study.py configs/factor-study-v1.json

Study A: month-end momentum and reversal over the full price history.
Study B: value, momentum, reversal, small size and value+momentum on the last
valuation capture of each month, raw and industry/size neutralized.
No forward return may end on or after the sealed date.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.factors import (
    PHYSICS_FACTORS,
    VALUE_COMPONENTS,
    composite,
    correlations,
    eligible,
    evaluate,
    forward_returns,
    industry_on,
    last_per_month,
    load_industries,
    load_valuation,
    momentum,
    month_end_dates,
    neutralize,
    physics_factors,
    physics_momentum,
    price_panel,
    reversal,
    value_factors,
)
from quant_research.industry_context import read_snapshot
from quant_research.storage import Snapshot, file_hash, write_json

BASE = Path(__file__).resolve().parents[1]


def base_cohort(panel, instruments, date, following, cfg):
    mask = eligible(panel, date, instruments, min_listing_days=cfg["min_listing_days"],
                    exclude_exchanges=cfg["exclude_exchanges"])
    returns, reason = forward_returns(panel, date, following)
    m, r = cfg["momentum"], cfg["reversal"]
    cohort = {
        "date": date, "ids": np.asarray(panel.ids), "mask": mask,
        "returns": np.where(mask, returns, np.nan),
        "momentum_12_1": np.where(mask, momentum(panel, date, m["lookback"], m["skip"], m["min_traded"]), np.nan),
        "reversal_1m": np.where(mask, reversal(panel, date, r["lookback"], r["min_traded"]), np.nan),
    }
    if "physics" in cfg:
        phys = physics_momentum(panel, date, m["lookback"], m["skip"], m["min_traded"],
                                cfg["physics"]["jump_sigma"])
        cohort.update(physics_factors(phys, mask, cohort["momentum_12_1"], cohort["reversal_1m"]))
    coverage = {"date": date, "exit_session": panel.dates[panel.index(following) + 1],
                "eligible": int(mask.sum()),
                "with_return": int(np.isfinite(cohort["returns"]).sum()),
                "entry_not_tradable": int((mask & (reason == 1)).sum()),
                "entry_limit_up": int((mask & (reason == 2)).sum()),
                "no_exit_price": int((mask & (reason == 3)).sum()),
                "sequence_break": int((mask & (reason == 4)).sum()),
                "with_momentum": int(np.isfinite(cohort["momentum_12_1"]).sum())}
    return cohort, coverage


def signal_pairs(signals, panel, sealed_from, max_gap_days=45):
    """Consecutive signal pairs; a pair spanning a data gap longer than max_gap_days is skipped."""
    pairs = []
    for date, following in zip(signals[:-1], signals[1:]):
        exit_index = panel.index(following) + 1
        if exit_index >= len(panel.dates) or panel.dates[exit_index] >= sealed_from:
            break
        if (pd.Timestamp(following) - pd.Timestamp(date)).days <= max_gap_days:
            pairs.append((date, following))
    return pairs


def run_study(cohorts, factors, cfg):
    results, daily = {}, []
    for name in factors:
        out = evaluate(cohorts, name, cfg["quantiles"], cfg["cost_one_way"],
                       cfg["bootstrap_block_months"], cfg["bootstrap_repetitions"])
        rows = out.pop("daily")
        daily += [{"factor": name, **row} for row in rows]
        results[name] = out
    return results, pd.DataFrame(daily)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    cfg = json.loads(parser.parse_args().config.read_text())
    output = BASE / cfg["output"]
    if output.exists():
        raise FileExistsError(f"Factor studies use new output paths: {output}")
    started = time.time()
    snapshot = Snapshot(BASE / cfg["snapshot"])
    instruments = snapshot.tables["instruments"]
    panel = price_panel(snapshot.tables["bars"])
    print(f"panel {len(panel.dates)} sessions x {len(panel.ids)} stocks, {time.time() - started:.0f}s", flush=True)
    sealed = cfg["sealed_from"]

    # Study A: month-end, full history.
    ends = [d for d in month_end_dates(panel.dates) if panel.index(d) >= cfg["momentum"]["lookback"]]
    cohorts_a, coverage_a = [], []
    for date, following in signal_pairs(ends, panel, sealed):
        cohort, coverage = base_cohort(panel, instruments, date, following, cfg)
        cohorts_a.append(cohort)
        coverage_a.append(coverage)
    factors_a = ["momentum_12_1", "reversal_1m", *(PHYSICS_FACTORS if "physics" in cfg else [])]
    results_a, daily_a = run_study(cohorts_a, factors_a, cfg)
    print(f"study A {len(cohorts_a)} cohorts, {time.time() - started:.0f}s", flush=True)

    # Study B: last valuation capture per month.
    valuation, valuation_lineage, valuation_missing = load_valuation(BASE / cfg["valuation_source"], sealed)
    print(f"valuation dates available {len(valuation_lineage)}, pruned {len(valuation_missing)}", flush=True)
    snapshots = load_industries([BASE / p for p in cfg["industry_sources"]])
    sessions = set(panel.dates)
    signals = [d for d in last_per_month(valuation.date.unique()) if d in sessions]
    cohorts_b, coverage_b = [], []
    for date, following in signal_pairs(signals, panel, sealed):
        cohort, coverage = base_cohort(panel, instruments, date, following, cfg)
        mask = cohort["mask"]
        v = value_factors(panel, date, valuation)
        for name in VALUE_COMPONENTS:
            cohort[name] = np.where(mask, v[name].to_numpy(), np.nan)
        size = np.where(mask, v.log_float_cap.to_numpy(), np.nan)
        cohort["small_size"] = -size
        cohort["value"] = composite([cohort[n] for n in VALUE_COMPONENTS], mask, cfg["value_min_components"])
        cohort["value_momentum"] = composite([cohort["value"], cohort["momentum_12_1"]], mask, 2)
        industry, industry_date = industry_on(date, snapshots, panel.ids,
                                              cfg["industry_max_age_days"], read_snapshot)
        for name in ["momentum_12_1", "reversal_1m", "value", "value_momentum"]:
            cohort[name + "_neutral"] = neutralize(cohort[name], industry, size)
        coverage.update(valuation_matched=int((mask & v.valuation_matched.to_numpy()).sum()),
                        with_value=int(np.isfinite(cohort["value"]).sum()),
                        industry_snapshot=industry_date,
                        with_industry=int((mask & pd.notna(industry)).sum()),
                        with_neutral_value=int(np.isfinite(cohort["value_neutral"]).sum()))
        cohorts_b.append(cohort)
        coverage_b.append(coverage)
    factors_b = ["value", *VALUE_COMPONENTS, "momentum_12_1", "reversal_1m", "small_size",
                 "value_momentum", "value_neutral", "momentum_12_1_neutral",
                 "reversal_1m_neutral", "value_momentum_neutral"]
    results_b, daily_b = run_study(cohorts_b, factors_b, cfg)
    print(f"study B {len(cohorts_b)} cohorts, {time.time() - started:.0f}s", flush=True)

    output.mkdir(parents=True)
    daily_a.to_csv(output / "daily-study-a.csv", index=False)
    daily_b.to_csv(output / "daily-study-b.csv", index=False)
    pd.DataFrame(coverage_a).to_csv(output / "coverage-study-a.csv", index=False)
    pd.DataFrame(coverage_b).to_csv(output / "coverage-study-b.csv", index=False)
    write_json(output / "results.json", {
        "version": cfg["version"], "config": cfg,
        "lineage": {"snapshot_manifest_sha256": file_hash(BASE / cfg["snapshot"] / "manifest.json"),
                    "snapshot_dataset_id": snapshot.dataset_id,
                    "valuation_captures": len(valuation_lineage),
                    "valuation_dates_pruned": valuation_missing,
                    "industry_snapshots_available": len(snapshots)},
        "limitations": cfg["limitations"],
        "study_a": {"description": "Month-end signals, full price history", "factors": results_a,
                    "correlations": correlations(cohorts_a, factors_a)},
        "study_b": {"description": "Last valuation capture per month",
                    "status": ("insufficient_history" if len(cohorts_b) < cfg["min_study_months"]
                               else "estimated"),
                    "cohorts": len(cohorts_b), "factors": results_b,
                    "correlations": correlations(cohorts_b, ["value", "momentum_12_1", "reversal_1m",
                                                             "small_size"])},
        "elapsed_seconds": round(time.time() - started, 1),
    })
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
