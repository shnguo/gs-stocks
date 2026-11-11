"""Read-only evidence audit for the 2026-10-12 opportunity decision.

This script audits saved research; it does not train, collect data, or simulate trades.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def audit(root: Path) -> dict:
    artifacts = root / "artifacts"
    store = artifacts / "daily-token-live-v2"
    research = read_json(store / "research-latest.json")
    current = read_json(store / "current.json")
    input_root = Path(current["source"])
    declaration = read_json(input_root / "snapshot/manifest.json")["declaration"]
    refresh = read_json(input_root / "refresh-summary.json")

    sources = [
        store / "research-latest.json",
        store / "current.json",
        input_root / "snapshot/manifest.json",
        input_root / "refresh-summary.json",
        artifacts / "retrained-early-stage-20260922-v1/daily.csv",
        artifacts / "retrained-early-stage-20260922-v1/summary.csv",
        artifacts / "retrained-early-stage-20260922-v1/decision.json",
        artifacts / "token-three-ideas-20260917-v1/completion-audit/conclusions.md",
        artifacts / "token-three-ideas-20260917-v1/completion-audit/verification.json",
        root / "docs/plan-risk-decision-20260914.md",
        root / "src/quant_research/event_sources.py",
        root / "src/quant_research/event_radar.py",
        root / "configs/research-review-v1.json",
        root / "scripts/retrained_early_stage_compare.py",
        Path(__file__).resolve(),
    ]

    aggregates: dict = defaultdict(lambda: {
        "signal_dates": 0, "known": 0, "absolute_error_sum": 0.0,
        "baseline_known": 0, "baseline_absolute_error_sum": 0.0,
        "direction_known": 0, "direction_hits": 0,
    })
    publications = []
    for report in research["analysis_reports"]:
        for arm in report["arms"]:
            scope = "prospective" if arm["publication"]["prospective"] else "retrospective"
            group = aggregates[(scope, arm["arm"])]
            group["signal_dates"] += 1
            quality = arm["quality"]
            for key in group:
                if key != "signal_dates":
                    group[key] += quality[key]
            publications.append({
                "signal_date": report["signal_date"], "arm": arm["arm"],
                "prospective": arm["publication"]["prospective"],
                "minutes_late": arm["publication"].get("minutes_late"),
            })
    price_results = []
    for (scope, arm), values in sorted(aggregates.items()):
        known, baseline_known = values["known"], values["baseline_known"]
        row = {
            "scope": scope, "arm": arm, "signal_dates": values["signal_dates"],
            "known_stock_forecast_day_rows": known,
            "model_mape_pct": 100 * values["absolute_error_sum"] / known if known else None,
            "baseline_mape_pct": (
                100 * values["baseline_absolute_error_sum"] / baseline_known
                if baseline_known else None
            ),
            "direction_agreement_pct": (
                100 * values["direction_hits"] / values["direction_known"]
                if values["direction_known"] else None
            ),
        }
        overview = next(x for x in research["analysis_overview"]
                        if x["scope"] == scope and x["arm"] == arm)
        assert overview["known"] == known
        if known:
            assert math.isclose(row["model_mape_pct"],
                                100 * overview["mean_absolute_percentage_error"],
                                abs_tol=1e-10)
            assert math.isclose(row["baseline_mape_pct"],
                                100 * overview["baseline_mean_absolute_percentage_error"],
                                abs_tol=1e-10)
        price_results.append(row)

    daily_path = artifacts / "retrained-early-stage-20260922-v1/daily.csv"
    with daily_path.open() as handle:
        daily = list(csv.DictReader(handle))
    by_arm: dict = defaultdict(dict)
    for row in daily:
        key = (row["fold"], row["date"])
        assert key not in by_arm[row["arm"]], "duplicate arm/fold/date"
        by_arm[row["arm"]][key] = row
    controls = by_arm["control"]
    with (artifacts / "retrained-early-stage-20260922-v1/summary.csv").open() as handle:
        summaries = {row["arm"]: row for row in csv.DictReader(handle)}
    comparisons = []
    original_decisions = {
        row["arm"]: row
        for row in read_json(artifacts / "retrained-early-stage-20260922-v1/decision.json")["arms"]
    }
    for arm, keyed in sorted(by_arm.items()):
        assert set(keyed) == set(controls), "unpaired dates"
        dates = sorted(keyed)
        returns = [float(keyed[k]["execution_return_pct"]) for k in dates]
        delta = [value - float(controls[k]["execution_return_pct"])
                 for k, value in zip(dates, returns)]
        assert math.isclose(mean(returns), float(summaries[arm]["execution_return_pct"]),
                            abs_tol=1e-10)
        win_fraction = mean(value > 1e-10 for value in delta)
        if arm in original_decisions:
            assert math.isclose(mean(delta),
                                original_decisions[arm]["execution_improvement_pp"],
                                abs_tol=1e-10)
            assert math.isclose(win_fraction,
                                original_decisions[arm]["execution_date_win_fraction"],
                                abs_tol=1e-10)
        comparisons.append({
            "arm": arm, "date_count": len(dates),
            "start": min(k[1] for k in dates), "end": max(k[1] for k in dates),
            "mean_open_to_forecast_selected_day_high_scenario_pct": mean(returns),
            "paired_mean_change_pp": mean(delta),
            "improved_date_fraction": win_fraction,
            "improvement_tolerance_pp": 1e-10,
            "mean_adverse_excursion_pct": mean(
                float(keyed[k]["adverse_excursion_pct"]) for k in dates
            ),
            "mean_overextended_fraction": mean(
                float(keyed[k]["overextended_fraction"]) for k in dates
            ),
        })

    # This is bounded discovery of conventional event artifact locations, not proof
    # that a database does not exist elsewhere or in a remote archive.
    event_candidates = []
    for parent in [root / "data", artifacts]:
        event_candidates += [str(p.relative_to(root)) for p in parent.glob("*event*")]
    event_candidates += [str(p.relative_to(root)) for p in (root / "data").glob("*.sqlite*")]
    source_statuses = []
    for source in research.get("source_observations", {}).get("sources", []):
        if source.get("api") in {"stock_st", "stk_limit"}:
            source_statuses.append({
                key: source.get(key)
                for key in ["name", "api", "status", "failure_reason", "observed_at"]
            })
    source_receipts = [{
        "file": str(path.relative_to(root)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    } for path in sources]

    return {
        "schema_version": 1,
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "task_date_client": "2026-10-12",
        "scope": "saved_local_research_only",
        "capital_cny_user_supplied": 200000,
        "latest_saved_signal_date": current["signal_date"],
        "price_observed_through": research["observed_through"],
        "summary_generated_at": research["generated_at"],
        "saved_prospective_mature_dates": research["prospective_mature_dates"],
        "publications": publications,
        "price_results_recomputed": price_results,
        "early_stage_results_recomputed": comparisons,
        "input_status": {
            "formal_ready": refresh["formal_ready"],
            "current_members": refresh["current_members"],
            "feature_complete": refresh["feature_complete"],
            "declaration": {key: declaration.get(key) for key in [
                "purpose", "vintage", "synthetic", "st_history_verified", "actions_verified",
                "rules_verified", "historical_universe_verified", "industry_history_verified",
            ]},
        },
        "saved_execution_source_rejections": source_statuses,
        "event_candidates_in_conventional_locations": sorted(set(event_candidates)),
        "full_universe_prior_audit": read_json(sources[8]),
        "limitations": [
            "Saved daily reports are stale and retrospective; no new market data were fetched.",
            "Stock/forecast-day rows and overlapping signal windows are not independent trials.",
            "The early-stage CSV field execution_return_pct uses a daily high on a selected day; it is a scenario, not an executed account return.",
            "Full-universe conclusions were read with their prior audit receipt; 290 million paths were not replayed in this audit.",
            "An event earnings-surprise strategy has not been tested by this audit.",
        ],
        "sources": source_receipts,
        "checks_passed": [
            "Recent MAPE recomputed from error sums and row counts matches saved overview.",
            "Early-stage arm/fold/date keys are unique and paired with the control.",
            "Early-stage scenario means recomputed from daily rows match saved summary.",
            "Paired improvement means and win fractions match saved decisions with the original 1e-10 pp tie tolerance.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root.resolve())
    if args.output.exists():
        raise FileExistsError("Use a new audit output path to preserve prior evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "checks": result["checks_passed"],
                      "saved_prospective_mature_dates": result["saved_prospective_mature_dates"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
