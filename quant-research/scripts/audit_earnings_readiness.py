"""Audit current source evidence and frozen forecast reviews without training or trading."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from quant_research.daily_loop import read, verify
from quant_research.storage import file_hash, utc_now


def source_status(workspace):
    rows = []
    for path in sorted(Path(workspace).glob("*/capture-progress.json")):
        for job in read(path)["jobs"]:
            result = {key: job.get(key) for key in ["name", "api", "status", "records", "observed_at"]}
            raw = path.parent / "raw" / job["name"]
            manifest_path = raw / "manifest.json"
            if manifest_path.exists():
                manifest = read(manifest_path)
                if file_hash(raw / "response.json") != manifest["raw_sha256"]:
                    raise ValueError("Archived provider bytes changed")
                result["provider_code"] = manifest["provider_code"]
                result["raw_sha256"] = manifest["raw_sha256"]
                if job["status"] == "captured":
                    norm = path.parent / "normalized" / job["name"]
                    normalized = verify(norm)
                    if normalized["capture_manifest_sha256"] != file_hash(manifest_path):
                        raise ValueError("Normalized capture has changed provenance")
                elif manifest["provider_code"] != 0:
                    message = str(read(raw / "response.json").get("msg", ""))
                    result["failure_reason"] = (
                        "rate_limited" if any(x in message for x in ["频率", "每分钟", "每小时", "每秒"])
                        else "permission_denied" if "权限" in message else "source_rejected"
                    )
            rows.append(result)
    return rows


def audit(workspace, store):
    workspace, store = Path(workspace), Path(store)
    source = workspace / "market-v1/input"
    snapshot = verify(source / "snapshot")
    summary = read(store / "research-latest.json")
    groups = defaultdict(lambda: {"known": 0, "absolute_error_sum": 0.,
                                  "baseline_known": 0, "baseline_absolute_error_sum": 0.,
                                  "direction_known": 0, "direction_hits": 0})
    for report in summary["analysis_reports"]:
        if report["outcome_manifest_sha256"] != file_hash(source / "snapshot/manifest.json"):
            raise ValueError("Review outcomes do not use the audited refreshed snapshot")
        for arm in report["arms"]:
            scope = "prospective" if arm["publication"]["prospective"] else "retrospective"
            row = groups[(scope, arm["arm"])]
            for key in row:
                row[key] += arm["quality"][key]
    measurements = []
    for (scope, arm), row in sorted(groups.items()):
        result = {
            "scope": scope, "arm": arm, "known_stock_forecast_day_rows": row["known"],
            "model_mape_pct": 100 * row["absolute_error_sum"] / row["known"] if row["known"] else None,
            "baseline_mape_pct": (100 * row["baseline_absolute_error_sum"] / row["baseline_known"]
                                  if row["baseline_known"] else None),
            "direction_agreement_pct": (100 * row["direction_hits"] / row["direction_known"]
                                        if row["direction_known"] else None),
        }
        overview = next(r for r in summary["analysis_overview"] if r["scope"] == scope and r["arm"] == arm)
        if row["known"]:
            assert math.isclose(result["model_mape_pct"], 100 * overview["mean_absolute_percentage_error"], abs_tol=1e-10)
        measurements.append(result)
    return {
        "schema_version": 1, "audited_at": utc_now(), "capital_cny": 200000,
        "user_commission_rate": 0.0005, "minimum_commission_confirmed": False,
        "price_observed_through": summary["observed_through"], "freshness": summary["freshness"],
        "prospective_mature_dates": summary["prospective_mature_dates"],
        "forecast_measurements_recomputed": measurements,
        "source_observations": source_status(workspace),
        "execution_declaration": snapshot["declaration"],
        "sources": [{"file": str(path), "sha256": file_hash(path)} for path in [
            store / "research-latest.json", source / "snapshot/manifest.json",
            source / "refresh-summary.json", Path(__file__).resolve(),
        ]],
        "conclusions": [
            "Existing frozen forecasts can now be compared with completed observed prices; their publication timeliness is unchanged.",
            "The refreshed retrospective MAPE comparison does not establish a price-prediction advantage over the unchanged-close baseline.",
            "Missing provider permissions and adjustment-factor limits are data-access findings, not tests of earnings-strategy profitability.",
            "No new event-strategy return or prospective performance conclusion has been established.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.workspace, args.store)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "observed_through": result["price_observed_through"],
                      "prospective_mature_dates": result["prospective_mature_dates"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
