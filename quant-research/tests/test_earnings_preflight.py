import json
import shutil
from pathlib import Path

import pytest
from audit_earnings_preflight import (
    archived_failure,
    audit,
    automatic_numeric_observation,
    verify_saved_audit,
)

from quant_research.daily_loop import digest
from quant_research.storage import file_hash


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def failed_capture(root):
    root.mkdir()
    for page, total in enumerate([131, 132], 1):
        write(root / f"index-{page:03}.json", {"totalAnnouncement": total, "announcements": [
            {"secCode": "000001", "announcementId": str(page)}]})
    write(root / "manifest.json", {"status": "failed_source_or_validation", "historical_pit_verified": False,
        "complete_announcement_coverage_verified": False, "finished_at": "2026-10-12T14:21:21Z",
        "request": {"start": "2026-01-01", "end": "2026-01-15"},
        "files": {p.name: file_hash(p) for p in root.glob("index-*.json")}})
    return root


def test_failed_index_is_not_reinterpreted_as_complete_or_historical(tmp_path):
    result = archived_failure(failed_capture(tmp_path / "failed"))
    assert result["page_totals"] == [131, 132] and result["total_changed"]
    assert result["page_rows"] == [1, 1]
    assert not result["usable_as_complete_index"] and not result["formal_profit_qualified"]


def test_failed_raw_tampering_and_outside_source_path_are_rejected(tmp_path):
    root = failed_capture(tmp_path / "failed")
    (root / "index-001.json").write_text("{}")
    with pytest.raises(ValueError, match="source bytes"):
        archived_failure(root)
    write(tmp_path / "outside.json", {})
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["files"] = {"../outside.json": file_hash(tmp_path / "outside.json")}
    write(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="containment"):
        archived_failure(root)


def test_saved_audit_rejects_changed_source_and_scope_escape(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write(workspace / "source.json", {})
    write(workspace / "audit.json", {"source_hashes": {"source.json": file_hash(workspace / "source.json")}})
    verify_saved_audit(workspace / "audit.json", workspace)
    write(workspace / "source.json", {"changed": True})
    with pytest.raises(ValueError, match="actual input sources"):
        verify_saved_audit(workspace / "audit.json", workspace)
    write(tmp_path / "outside.json", {})
    write(workspace / "audit.json", {"source_hashes": {"../outside.json": file_hash(tmp_path / "outside.json")}})
    with pytest.raises(ValueError, match="escaped"):
        verify_saved_audit(workspace / "audit.json", workspace)


@pytest.fixture
def inputs(tmp_path, base_snapshot, monkeypatch):
    workspace = tmp_path
    shutil.copytree(base_snapshot.path, workspace / "snapshot")
    protocol_path = Path(__file__).parents[1] / "configs/earnings-drift-validation-20261012-v1.json"
    shutil.copyfile(protocol_path, workspace / "protocol.json")
    view = {"historical_pit_verified": False, "first_numeric_version_verified": False,
            "formal_profit_qualified": False, "source_listing_observations": 0, "numeric_observation_versions": 0}
    write(workspace / "view.json", view)
    write(workspace / "version-audit.json", {**view, "registered_batches": 0,
        "source_hashes": {"view.json": file_hash(workspace / "view.json")}})
    write(workspace / "capture/manifest.json", {"files": {}})
    write(workspace / "packet.json", {"documents": []})
    write(workspace / "candidates.json", {})
    write(workspace / "reference.json", {})
    (workspace / "market-captures").mkdir()
    (workspace / "execution-captures").mkdir()
    failure = failed_capture(workspace / "failed")
    config = {"schema_version": 1, "protocol": "protocol.json", "snapshot": "snapshot",
        "numeric": {"capture": "capture", "packet": "packet.json", "candidates": "candidates.json", "reference": "reference.json"},
        "registry_audit": "version-audit.json", "registry_view": "view.json",
        "capture_workspace": "market-captures", "execution_capture_workspace": "execution-captures",
        "failed_captures": [failure.name]}
    write(workspace / "config.json", config)
    # Original PDF/numeric semantics are independently exercised in test_earnings_review.
    # This fixture isolates control flow without treating artificial bytes as real PDFs.
    monkeypatch.setattr("audit_earnings_preflight.audit_numeric", lambda *args: {
        "pdfs": 100, "extraction_statuses": {}, "reference_documents_inspected": 100,
        "independent_100_document_acceptance": False, "qualified_historical_signals": 0})
    return workspace, workspace / "config.json"


def test_source_declarations_and_same_agent_reference_do_not_qualify_real_experiment(inputs):
    workspace, config = inputs
    manifest_path = workspace / "snapshot/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for flag in ["st_history_verified", "actions_verified", "rules_verified", "historical_universe_verified", "industry_history_verified"]:
        manifest["declaration"][flag] = True
    manifest["declaration"].update(purpose="full_experiment", vintage="verified_pit", synthetic=False)
    manifest["dataset_id"] = digest({k: manifest[k] for k in ["schema_version", "declaration", "files"]})
    write(manifest_path, manifest)
    result = audit(config, workspace)
    components = {r["id"]: r for r in result["components"]}
    assert components["risk"]["observed"] is True and components["risk"]["status"] == "not_accepted"
    assert components["original_numeric_acceptance"]["observed"]["same_agent_documents_inspected"] == 100
    assert components["normal_holding_sample_design"]["status"] == "design_conflict"
    assert components["selected_snapshot_history"]["status"] == "insufficient_span"


def test_v2_horizon_removes_only_the_capacity_conflict_and_retains_source_blocks(inputs):
    workspace, config = inputs
    v2 = Path(__file__).parents[1] / "configs/earnings-drift-validation-20261012-v2.json"
    shutil.copyfile(v2, workspace / "protocol.json")
    result = audit(config, workspace)
    components = {r["id"]: r for r in result["components"]}
    capacity = components["normal_holding_sample_design"]
    assert capacity["status"] == "needs_signal_feasibility"
    assert capacity["observed"]["observed_sessions"] == 504
    assert capacity["observed"]["no_early_exit_mature_trade_upper_bound"] == 150
    assert components["protocol_registration"]["status"] == "not_registered"
    assert not result["formal_profit_qualified"]
    assert components["calendar_acceptance"]["status"] == "not_accepted"
    assert not result["formal_return_experiment_run"] and not result["formal_profit_qualified"]
    assert not result["prospective_start_registered"]


def test_v3_user_confirmed_fee_and_automation_remove_human_requirement_only(inputs):
    workspace, config = inputs
    v3 = Path(__file__).parents[1] / "configs/earnings-drift-validation-20261012-v3.json"
    shutil.copyfile(v3, workspace / "protocol.json")
    result = audit(config, workspace)
    components = {r["id"]: r for r in result["components"]}
    assert result["human_review_required"] is False
    assert components["original_numeric_acceptance"]["observed"]["human_review_required"] is False
    assert "Human review is not required" in components["original_numeric_acceptance"]["required_next_evidence"]
    assert components["actual_broker_minimum_fee"]["observed"]["minimum_confirmed"] is True
    assert components["actual_broker_minimum_fee"]["observed"]["minimum_commission_cny"] == 5
    assert components["actual_broker_minimum_fee"]["status"] == "not_accepted"
    assert components["normal_holding_sample_design"]["observed"]["observed_sessions"] == 504
    assert not result["formal_profit_qualified"]


def test_arithmetic_receipt_requires_source_identity_counts_and_false_qualification(tmp_path):
    row = dict(announcement_id="1", instrument_id="cn.xshe.000001", pdf_sha256="p",
               text_sha256="t", available_at="2026-10-13T02:00:00Z", availability_basis="local_observation")
    write(tmp_path / "candidates.json", {"records": [row]})
    write(tmp_path / "packet.json", {})
    check_flags = {k: False for k in ["automatic_financial_acceptance", "financial_fields_verified",
                  "historical_pit_verified", "first_numeric_version_verified", "formal_profit_qualified"]}
    value = {**check_flags, "independent_extraction_path_verified": False,
             "formal_return_experiment_run": False, "human_review_required": False,
             "candidate_source_replay_equal": True,
             "source_candidates_sha256": file_hash(tmp_path / "candidates.json"),
             "source_review_packet_sha256": file_hash(tmp_path / "packet.json"),
             "records": [{**row, "check": {**check_flags, "status": "pass", "reason": "consistent"}}],
             "status_reason_counts": {"pass:consistent": 1}}
    path = tmp_path / "auto.json"
    write(path, value)
    assert automatic_numeric_observation(path, tmp_path / "candidates.json", tmp_path / "packet.json")["documents"] == 1
    value["automatic_financial_acceptance"] = True
    write(path, value)
    with pytest.raises(ValueError, match="cannot promote"):
        automatic_numeric_observation(path, tmp_path / "candidates.json", tmp_path / "packet.json")
    value["automatic_financial_acceptance"] = False
    value["records"][0]["pdf_sha256"] = "changed"
    write(path, value)
    with pytest.raises(ValueError, match="identity changed"):
        automatic_numeric_observation(path, tmp_path / "candidates.json", tmp_path / "packet.json")


def test_legacy_daily_manifest_format_is_reported_without_qualification_upgrade(inputs):
    workspace, config = inputs
    path = workspace / "snapshot/manifest.json"
    value = json.loads(path.read_text())
    del value["schema_version"]
    write(path, value)
    result = audit(config, workspace)
    component = next(r for r in result["components"] if r["id"] == "canonical_snapshot_identity")
    assert component["status"] == "not_accepted" and component["observed"] is None
    assert result["market_observed"]["price_rows"] == 2400


def test_rehashed_snapshot_declaration_with_unchanged_identity_is_rejected(inputs):
    workspace, config = inputs
    path = workspace / "snapshot/manifest.json"
    value = json.loads(path.read_text())
    value["declaration"]["source"] = "changed"
    write(path, value)
    with pytest.raises(ValueError, match="Canonical snapshot identity changed"):
        audit(config, workspace)


def test_unqualified_version_view_cannot_be_silently_promoted(inputs):
    workspace, config = inputs
    path = workspace / "view.json"
    value = json.loads(path.read_text())
    value["first_numeric_version_verified"] = True
    write(path, value)
    saved = json.loads((workspace / "version-audit.json").read_text())
    saved["source_hashes"]["view.json"] = file_hash(path)
    write(workspace / "version-audit.json", saved)
    with pytest.raises(ValueError, match="cannot upgrade"):
        audit(config, workspace)
