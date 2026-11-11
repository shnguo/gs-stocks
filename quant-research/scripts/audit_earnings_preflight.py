"""Read-only preflight of the current unregistered earnings research inputs.

This is a source-gap screen, not a qualification authority or a trading runner.
Hash validity, source declarations and same-agent reference checks never certify
historical availability, semantic accuracy or strategy profitability.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from audit_earnings_numeric import audit as audit_numeric
from audit_earnings_readiness import source_status

from quant_research.daily_loop import digest as structured_digest
from quant_research.daily_loop import verify
from quant_research.earnings_validation import normal_holding_capacity
from quant_research.storage import TABLES, file_hash


def read(path):
    return json.loads(Path(path).read_text())


def automatic_numeric_observation(path, candidates, packet):
    """Inspect an explicitly unqualified Rust arithmetic receipt, never grant acceptance."""
    value = read(path)
    if (value.get("source_candidates_sha256") != file_hash(candidates)
            or value.get("source_review_packet_sha256") != file_hash(packet)
            or value.get("candidate_source_replay_equal") is not True
            or value.get("human_review_required") is not False):
        raise ValueError("Automatic numeric receipt source binding changed")
    for key in ["independent_extraction_path_verified", "automatic_financial_acceptance",
                "financial_fields_verified", "historical_pit_verified",
                "first_numeric_version_verified", "formal_profit_qualified", "formal_return_experiment_run"]:
        if value.get(key) is not False:
            raise ValueError("Arithmetic-only receipt cannot promote qualification")
    expected = read(candidates)["records"]
    rows = value.get("records", [])
    identities = ["announcement_id", "instrument_id", "pdf_sha256", "text_sha256", "available_at", "availability_basis"]
    if len(rows) != len(expected):
        raise ValueError("Automatic numeric receipt coverage differs")
    reasons = Counter()
    for row, original in zip(rows, expected, strict=True):
        if any(row.get(k) != original.get(k) for k in identities):
            raise ValueError("Automatic numeric receipt identity changed")
        check = row.get("check", {})
        if check.get("status") not in {"pass", "skip", "reject"} or not check.get("reason"):
            raise ValueError("Invalid automatic numeric screening result")
        if any(check.get(k) is not False for k in ["automatic_financial_acceptance",
                "financial_fields_verified", "historical_pit_verified", "first_numeric_version_verified",
                "formal_profit_qualified"]):
            raise ValueError("Per-document arithmetic receipt cannot promote qualification")
        reasons[check["status"] + ":" + check["reason"]] += 1
    if dict(reasons) != value.get("status_reason_counts"):
        raise ValueError("Automatic numeric receipt counts differ")
    return {"documents": len(rows), "status_reason_counts": dict(reasons),
            "independent_extraction_path_verified": False, "automatic_financial_acceptance": False,
            "human_review_required": False}


def archived_failure(capture):
    """Recompute failure findings from actual archived pages, without normalizing them."""
    capture = Path(capture).resolve()
    manifest_path = capture / "manifest.json"
    manifest = read(manifest_path)
    if (manifest.get("status") != "failed_source_or_validation"
            or manifest.get("historical_pit_verified") is not False
            or manifest.get("complete_announcement_coverage_verified") is not False):
        raise ValueError("Expected isolated unqualified failed capture")
    sources = {str(manifest_path): file_hash(manifest_path)}
    totals, rows, keys = [], [], []
    for name, digest in sorted(manifest["files"].items()):
        path = (capture / name).resolve()
        if not path.is_relative_to(capture) or not path.is_file() or file_hash(path) != digest:
            raise ValueError("Failed capture source bytes or containment changed")
        sources[str(path)] = digest
        if name.startswith("index-") and name.endswith(".json"):
            raw = read(path)
            totals.append(raw["totalAnnouncement"])
            entries = raw.get("announcements")
            if not isinstance(entries, list):
                entries = []
            rows.append(len(entries))
            keys.extend((r["secCode"], r["announcementId"]) for r in entries)
    return {"capture": capture.name, "request": manifest["request"],
            "actual_observed_at": manifest["finished_at"], "page_totals": totals,
            "page_rows": rows, "total_changed": len(set(totals)) > 1,
            "duplicate_listing_rows": len(keys) - len(set(keys)),
            "usable_as_complete_index": False, "source_hashes": sources,
            "formal_profit_qualified": False}


def verify_saved_audit(path, workspace):
    value = read(path)
    for name, digest in value["source_hashes"].items():
        target = Path(name)
        if not target.is_absolute():
            target = workspace / target
        target = target.resolve()
        if not target.is_relative_to(workspace.resolve()):
            raise ValueError("Saved audit source path escaped the research workspace")
        if file_hash(target) != digest:
            raise ValueError("Saved audit no longer matches its actual input sources")
    return value


def audit(config_path, workspace):
    workspace = Path(workspace).resolve()
    config_path = Path(config_path).resolve()
    config = read(config_path)
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported earnings preflight specification")
    def member(value):
        result = (workspace / value).resolve()
        if not result.is_relative_to(workspace):
            raise ValueError("Research preflight inputs must remain inside the workspace")
        return result
    protocol_path, snapshot_path = member(config["protocol"]), member(config["snapshot"])
    protocol = read(protocol_path)
    snapshot = verify(snapshot_path)
    if set(snapshot["files"]) != {name + ".parquet" for name in TABLES}:
        raise ValueError("Current market snapshot does not bind every required table")
    if snapshot.get("schema_version") == 1 and structured_digest({k: snapshot[k] for k in [
            "schema_version", "declaration", "files"]}) != snapshot["dataset_id"]:
        raise ValueError("Canonical snapshot identity changed")
    declaration = snapshot["declaration"]
    args = {k: member(v) for k, v in config["numeric"].items()}
    numeric = audit_numeric(args["capture"], args["packet"], args["candidates"], args["reference"])
    registry_audit_path = member(config["registry_audit"])
    registry_audit = verify_saved_audit(registry_audit_path, workspace)
    registry_view_path = member(config["registry_view"])
    if str(registry_view_path.relative_to(workspace)) not in registry_audit["source_hashes"]:
        raise ValueError("Requested version view is not covered by the saved source audit")
    view = read(registry_view_path)
    for value in [view, registry_audit]:
        if any(value.get(k) is not False for k in ["historical_pit_verified",
                "first_numeric_version_verified", "formal_profit_qualified"]):
            raise ValueError("Unregistered observed versions cannot upgrade source qualifications")
    failures = [archived_failure(member(v)) for v in config["failed_captures"]]
    observations = source_status(member(config["capture_workspace"]))
    execution = source_status(member(config["execution_capture_workspace"]))
    bars = pd.read_parquet(snapshot_path / "bars.parquet", columns=["date", "instrument_id"])
    instruments = pd.read_parquet(snapshot_path / "instruments.parquet", columns=["instrument_id"])
    counts = bars.groupby("date").instrument_id.nunique()
    components = []
    def gap(key, status, observed, needed):
        components.append({"id": key, "status": status, "observed": observed,
                           "required_next_evidence": needed})
    gap("protocol_registration", "not_registered", protocol["status"],
        "Resolve draft statistical conventions and freeze signed protocol/code/environment before prospective start.")
    gap("runner", "not_implemented", protocol["existing_runner_compatible"],
        "Implement registered daily four-arm runner with immutable decisions and a single formal look.")
    gap("input_purpose", "not_accepted", declaration.get("purpose"),
        "Use accepted full-experiment execution inputs; training refresh cannot enter the account experiment.")
    gap("canonical_snapshot_identity", "not_accepted" if snapshot.get("schema_version") != 1 else "integrity_only",
        snapshot.get("schema_version"),
        "Daily refresh and canonical Snapshot formats differ; accepted canonical identity and derivation are required.")
    gap("calendar_acceptance", "not_accepted", declaration.get("calendar_basis"),
        "Bind reviewed exchange calendars covering every development and execution session, not only current holidays.")
    walk = protocol["validation"]["historical_walk_forward"]
    first, last = pd.Timestamp(counts.index.min()), pd.Timestamp(counts.index.max())
    earliest_test_end = first + pd.DateOffset(years=walk["train_years"] + walk["validation_years"],
                                            months=walk["test_months"])
    gap("selected_snapshot_history", "insufficient_span" if earliest_test_end > last else "needs_fold_and_purge_audit",
        {"first_date": first.strftime("%Y-%m-%d"), "last_date": last.strftime("%Y-%m-%d"),
         "train_years": walk["train_years"], "validation_years": walk["validation_years"], "test_months": walk["test_months"]},
        "Provide sufficiently long accepted history and actual-label-end purge/20-session embargo; other system history is outside this selected input audit.")
    for key, flag in [("historical_universe", "historical_universe_verified"),
                      ("industry", "industry_history_verified"), ("risk", "st_history_verified"),
                      ("actions", "actions_verified"), ("rules_and_fees", "rules_verified")]:
        gap(key, "not_accepted", declaration.get(flag),
            "Provide effective-dated source records, coverage, actual availability and independent semantic acceptance; a true declaration alone is insufficient.")
    automatic = protocol.get("data_acceptance", {}).get("mode") == "fully_automatic_fail_closed"
    automatic_path = member(config["automatic_numeric_screen"]) if automatic and config.get("automatic_numeric_screen") else None
    automatic_observation = (automatic_numeric_observation(automatic_path, args["candidates"], args["packet"])
                             if automatic_path else None)
    gap("original_numeric_acceptance", "not_accepted", {
        "same_agent_documents_inspected": numeric["reference_documents_inspected"],
        "independent_100_document_acceptance": numeric["independent_100_document_acceptance"],
        "human_review_required": not automatic, "automatic_screen": automatic_observation},
        ("Implement and verify distinct extraction paths, source-bound semantic agreement and arithmetic checks; automatically exclude missing, ambiguous or conflicting events. Human review is not required."
         if automatic else "Complete independent human acceptance of at least 100 originals, including unsupported and adverse cases."))
    gap("first_numeric_versions", "not_accepted", view["first_numeric_version_verified"],
        "Accept company/report identities and complete predecessor evidence without replacing negative or unknown first versions.")
    gap("complete_event_coverage", "not_accepted", {
        "failed_archived_windows": len(failures), "stable_index_windows_do_not_certify_market_coverage": True},
        "Resolve Jan-Feb source pagination failures and verify complete event coverage including controls.")
    gap("historical_availability", "not_accepted", view["historical_pit_verified"],
        "Provide publication-archive proof for historical testing; actual late local observation remains late.")
    gap("limits_suspensions_and_settlement", "missing_acceptance", {
        "provider_failures": [r for r in execution if r["api"] in {"stock_st", "stk_limit"}]},
        "Acquire usable ST/normal-risk, suspension, daily limit and settlement evidence for the entire tested universe.")
    gap("point_in_time_float_cap", "not_provided", None,
        "Supply company/session point-in-time float market capitalization for frozen caliper matching.")
    gap("frozen_model_ranking", "not_provided", None,
        "Bind existing five-day model ranks and model/input hashes to each timely decision date for M.")
    gap("actual_broker_minimum_fee", "not_accepted", {
        "commission": protocol["costs"]["broker_commission_rate"],
        "minimum_placeholder": protocol["costs"]["minimum_commission_cny_placeholder"],
        "minimum_commission_cny": protocol["costs"].get("minimum_commission_cny"),
        "minimum_confirmed": protocol["costs"]["minimum_commission_user_confirmed"],
        "broker_name": protocol["costs"].get("broker_name"),
        "commission_includes_regulatory_fees": protocol["costs"].get("commission_includes_regulatory_fees")},
        ("Minimum CNY5 is confirmed; identify broker schedule and commission fee inclusion to avoid missing or double-counting charges."
         if protocol["costs"]["minimum_commission_user_confirmed"] else
         "Obtain actual minimum broker charge and what the commission includes before net-profit acceptance."))
    capacity = normal_holding_capacity(protocol["validation"]["single_formal_look_sessions"],
        protocol["portfolio"]["maximum_positions"], protocol["execution"]["primary_holding_sessions"],
        protocol["validation"]["minimum_mature_unique_company_report_events"])
    gap("normal_holding_sample_design", "design_conflict" if not capacity["event_count_feasible_without_early_exits"] else "needs_signal_feasibility",
        capacity, "Verify signal opportunity and at least 60 mature entry dates under the selected horizon; a capacity bound does not guarantee either event or date coverage.")
    paths = [config_path, protocol_path, snapshot_path / "manifest.json", registry_audit_path, registry_view_path,
             Path(__file__), *[args[k] for k in ["packet", "candidates", "reference"]]]
    if automatic_path:
        paths.append(automatic_path)
    sources = {str(p): file_hash(p) for p in paths}
    sources.update({str(snapshot_path / name): digest for name, digest in snapshot["files"].items()})
    sources.update({str((workspace / name).resolve()): digest
                    for name, digest in registry_audit["source_hashes"].items()})
    capture_manifest = read(args["capture"] / "manifest.json")
    sources[str(args["capture"] / "manifest.json")] = file_hash(args["capture"] / "manifest.json")
    for name, digest in capture_manifest["files"].items():
        sources[str(args["capture"] / name)] = digest
    for document in read(args["packet"])["documents"]:
        sources[str(args["packet"].parent / "text" / (document["announcement_id"] + ".txt"))] = document["text_sha256"]
    for source_root in [member(config["capture_workspace"]), member(config["execution_capture_workspace"])]:
        for progress in source_root.glob("*/capture-progress.json"):
            sources[str(progress)] = file_hash(progress)
            for job in read(progress)["jobs"]:
                for kind in ["raw", "normalized"]:
                    folder = progress.parent / kind / job["name"]
                    manifest_path = folder / "manifest.json"
                    if not manifest_path.exists():
                        continue
                    sources[str(manifest_path)] = file_hash(manifest_path)
                    manifest = read(manifest_path)
                    names = manifest.get("files", {}) if kind == "normalized" else ["response.json"]
                    for name in names:
                        path = (folder / name).resolve()
                        if not path.is_relative_to(workspace):
                            raise ValueError("Observation lineage escaped the research workspace")
                        sources[str(path)] = file_hash(path)
    for f in failures:
        sources.update(f["source_hashes"])
    return {"schema_version": 1, "audited_at": datetime.now(timezone.utc).isoformat(),
        "status": "blocked_current_inputs_not_qualified_for_real_return_experiment",
        "formal_profit_qualified": False, "formal_return_experiment_run": False,
        "prospective_start_registered": False, "formal_runner_implemented": False,
        "capital_cny": protocol["capital_cny"], "commission_rate": protocol["costs"]["broker_commission_rate"],
        "protocol_version": protocol["version"], "human_review_required": not automatic,
        "components": components, "component_status_counts": dict(Counter(c["status"] for c in components)),
        "market_observed": {"first_date": str(counts.index.min()), "last_date": str(counts.index.max()),
            "price_rows": len(bars), "instruments_in_snapshot": len(instruments),
            "minimum_distinct_symbols_per_observed_date": int(counts.min()),
            "maximum_distinct_symbols_per_observed_date": int(counts.max()),
            "coverage_interpretation": "Observed rows only; listing and suspension gaps prevent expected-universe completeness certification."},
        "numeric_source_recomputed": {k: numeric[k] for k in ["pdfs", "extraction_statuses", "reference_documents_inspected",
            "independent_100_document_acceptance", "qualified_historical_signals"]},
        "version_source_audit": {"registered_batches": registry_audit["registered_batches"],
            "source_listing_observations": view["source_listing_observations"],
            "numeric_observation_versions": view["numeric_observation_versions"], "qualification": "not_accepted"},
        "source_observations": observations, "execution_source_observations": execution,
        "failed_index_windows": failures, "normal_holding_capacity": capacity, "source_hashes": sources,
        "limits": ["Research preflight inventories actual gaps; it never grants formal source or profit qualification.",
            "Hash agreement and producer declarations do not establish semantic or historical acceptance.",
            "No ledger, forecast, training, trade, schedule or source mutation is launched by this audit.",
            "Current source format gaps must be resolved without backfilling old observation clocks."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preflight output must be new")
    result = audit(args.config, args.workspace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "status": result["status"],
        "components": len(result["components"]), "formal_profit_qualified": False}))


if __name__ == "__main__":
    main()
