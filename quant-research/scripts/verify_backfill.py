"""Independent, read-only validation of a completed Rust campaign status report."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(campaign: Path, report_path: Path, output: Path) -> None:
    if output.exists() or output.with_suffix(".csv").exists():
        raise FileExistsError("Verification outputs must use new paths")
    manifest = json.loads((campaign / "manifest.json").read_text())
    assert sha(campaign / "plan.json") == manifest["plan_sha256"]
    plan = json.loads((campaign / "plan.json").read_text())
    report = json.loads(report_path.read_text())
    assert report["universe_id"] == plan["universe_id"] == manifest["universe_id"]
    members = {r["instrument_id"]: r for r in plan["members"]}
    assert len(members) == len(plan["members"]) == plan["counts"]["members"]
    assert set(members) == {r["instrument_id"] for r in report["members"]}
    assert all(r["retained_in_expected_pool"] for r in report["members"])
    captures = {sha(p): p for p in campaign.glob("captures/*/manifest.json")}
    rows = Counter()
    verified = 0
    for job in plan["jobs"]:
        state = report["jobs"][job["job_id"]]
        if state["status"] not in {"captured_raw_verified", "empty_requires_review"}:
            continue
        path = captures[state["manifest_sha256"]]
        capture = json.loads(path.read_text())
        identity = job["identity"]
        assert capture["provider"] == identity["provider"]
        if capture["provider"] == "baostock":
            assert capture["query"] == identity["query"]
            assert sha(path.parent / "records.json") == capture["records_sha256"]
            for raw in capture["raw_files"]:
                assert sha(path.parent / raw["file"]) == raw["sha256"]
            data = json.loads((path.parent / "records.json").read_text())
            assert data["pagination_complete"] is True
            count = len(data["rows"])
        else:
            assert capture["params"] == identity["query"]["params"]
            assert capture["api"] == identity["query"]["api"]
            assert capture["provider_code"] == 0 and capture["http_status"] == 200
            assert sha(path.parent / "response.json") == capture["raw_sha256"]
            raw = json.loads((path.parent / "response.json").read_text())
            count = len(raw["data"]["items"])
        assert count == state["rows"]
        rows[state["api"]] += count
        verified += 1
    assert dict(rows) == report["captured_rows"]
    preserved_reviews = 0
    for path in campaign.glob("retry-reviews/*/*.json"):
        review = json.loads(path.read_text())
        previous = campaign / "captures" / review["previous_attempt_key"] / "manifest.json"
        if review["previous_manifest_sha256"]:
            assert sha(previous) == review["previous_manifest_sha256"]
        preserved_reviews += 1
    imported_states = 0
    for path in campaign.glob("imports/*.json"):
        imported = json.loads(path.read_text())
        if imported.get("source_manifest_sha256"):
            source = Path(imported["source_capture"]) / "manifest.json"
            assert sha(source) == imported["source_manifest_sha256"]
            target = campaign / "captures" / imported["job_id"] / "manifest.json"
            assert sha(target) == sha(source)
            for original in source.parent.iterdir():
                if original.is_file():
                    assert sha(target.parent / original.name) == sha(original)
        imported_states += 1
    ledger = []
    for row in report["members"]:
        member = members[row["instrument_id"]]
        entry = {key: member[key] for key in
                 ["instrument_id", "provider_code", "exchange", "in_selection_pool"]}
        entry["status"] = row["status"]
        entry["retained_in_expected_pool"] = True
        for index, kind in enumerate(["daily", "factor"]):
            state = (report["jobs"][row["job_ids"][index]] if row["job_ids"]
                     else {"status": "outside_requested_history"})
            entry[f"{kind}_status"] = state["status"]
            entry[f"{kind}_rows"] = state.get("rows", 0)
        ledger.append(entry)
    summary = {"report": str(report_path.resolve()), "report_sha256": sha(report_path),
               "universe_id": plan["universe_id"], "members": len(members),
               "planned_jobs": len(plan["jobs"]), "verified_capture_manifests": verified,
               "job_counts": report["job_counts"], "member_counts": report["member_counts"],
               "captured_rows": dict(rows), "preserved_retry_reviews": preserved_reviews,
               "imported_capture_states_verified": imported_states,
               "all_expected_members_retained": True, "data_ready": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    with output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ledger[0]))
        writer.writeheader()
        writer.writerows(ledger)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    verify(args.campaign, args.report, args.output)
