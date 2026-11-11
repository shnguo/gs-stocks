import shutil

import pandas as pd
import pytest
from test_research_review import market, protocol

from quant_research.daily_loop import read
from quant_research.execution_snapshot import (
    COMPONENTS,
    ExecutionEvidence,
    build_execution_snapshot,
)
from quant_research.publication import launch_plan
from quant_research.research_review import freeze_decision, refresh_research
from quant_research.storage import TABLES, Snapshot, file_hash, write_json


def evidence_fixture(tmp_path, snapshot):
    root = tmp_path / "evidence"
    root.mkdir()
    for name in TABLES:
        shutil.copyfile(snapshot.path / (name + ".parquet"), root / (name + ".parquet"))
    # A synthetic archive and its normalizer are deliberately labeled engineering-only.
    (root / "raw.txt").write_text("deterministic market fixture; not real evidence")
    (root / "publication.txt").write_text("synthetic historical publication assertion")
    (root / "normalizer.py").write_text("# synthetic canonical fixture producer\n")
    rows = [{"component": component, "instrument_id": "*", "start_date": snapshot.dates[0],
             "end_date": snapshot.dates[-1], "known_at": "2023-01-01T00:00:00Z",
             "source_id": "fixture", "status": "verified_pit"} for component in COMPONENTS]
    pd.DataFrame(rows).to_csv(root / "coverage.csv", index=False)
    spec = {"schema_version": 1, "synthetic": True, "sources": {"fixture": {
        "path": "raw.txt", "sha256": file_hash(root / "raw.txt"),
        "source_url": "synthetic:fixture", "observed_at": "2023-06-01T00:00:00Z",
        "available_at": "2023-01-01T00:00:00Z", "availability_basis": "verified_publication_archive",
        "publication_proof_path": "publication.txt", "publication_proof_sha256": file_hash(root / "publication.txt")}},
        "tables": {name: {"path": name + ".parquet", "sha256": file_hash(root / (name + ".parquet")),
                          "source_ids": ["fixture"], "normalizer_path": "normalizer.py",
                          "normalizer_sha256": file_hash(root / "normalizer.py")} for name in TABLES},
        "coverage": {"path": "coverage.csv", "sha256": file_hash(root / "coverage.csv")}}
    write_json(root / "execution-evidence.json", spec)
    return root


def change_coverage(root, change):
    frame = pd.read_csv(root / "coverage.csv")
    change(frame)
    frame.to_csv(root / "coverage.csv", index=False)
    spec = read(root / "execution-evidence.json")
    spec["coverage"]["sha256"] = file_hash(root / "coverage.csv")
    write_json(root / "execution-evidence.json", spec)


def test_no_source_evidence_never_upgrades_even_declared_snapshot(tmp_path):
    snapshot, _, run = market(tmp_path)
    bundle = freeze_decision(run, protocol())
    result = build_execution_snapshot(bundle, snapshot)
    report = read(result / "gap-report.json")
    assert report["status"] == "blocked" and not (result / "snapshot").exists()
    assert len(report["candidates"]) == 6
    assert all(not row["ready"] for row in report["candidates"])
    assert all(check["missing"] == 6 for check in report["global_checks"])
    assert file_hash(snapshot.path / "manifest.json") == file_hash(run.parents[2] / "source/snapshot/manifest.json")


def test_complete_evidence_creates_immutable_snapshot_and_cost_review(tmp_path):
    snapshot, store, run = market(tmp_path)
    evidence = evidence_fixture(tmp_path, snapshot)
    cfg = protocol()
    cfg["allow_synthetic_for_tests"] = True
    bundle = freeze_decision(run, cfg)
    result = build_execution_snapshot(bundle, snapshot, evidence)
    report = read(result / "gap-report.json")
    assert report["snapshot_generated"] and report["synthetic"]
    imported = Snapshot(result / "snapshot")
    assert imported.manifest["declaration"]["vintage"] == "synthetic"
    assert "forward_execution_snapshot_not_full_historical_experiment" in imported.formal_blockers()
    assert ExecutionEvidence(result / "evidence").identity == ExecutionEvidence(evidence).identity
    assert build_execution_snapshot(bundle, snapshot, evidence) == result
    refreshed = refresh_research(store, snapshot.path.parent, cfg, execution_evidence=evidence)
    public = read(refreshed["summary"])
    assert public["latest"]["execution_data"]["snapshot_generated"]
    assert public["latest"]["arms"][0]["execution"]["status"] == "simulated"
    assert len(public["latest"]["arms"][0]["execution"]["scenarios"]) == 3
    assert public["prospective_simulated_dates"] == 0
    # Re-use the registered version without re-import; no source bundle edit is tolerated.
    assert refresh_research(store, snapshot.path.parent, cfg)
    (evidence / "raw.txt").write_text("changed bytes")
    with pytest.raises(ValueError, match="hash mismatch"):
        refresh_research(store, snapshot.path.parent, cfg)


def test_late_claims_are_stock_gaps_and_cannot_generate_snapshot(tmp_path):
    snapshot, _, run = market(tmp_path)
    evidence = evidence_fixture(tmp_path, snapshot)
    change_coverage(evidence, lambda frame: frame.loc.__setitem__((frame.component.eq("industry"), "known_at"), "2023-06-01T00:00:00Z"))
    result = build_execution_snapshot(freeze_decision(run, protocol()), snapshot, evidence)
    report = read(result / "gap-report.json")
    assert not report["snapshot_generated"]
    assert any(gap["reason"] == "available_too_late" for row in report["candidates"] for gap in row["gaps"])


def test_local_observation_cannot_claim_an_earlier_publication(tmp_path):
    snapshot, _, _ = market(tmp_path)
    root = evidence_fixture(tmp_path, snapshot)
    spec = read(root / "execution-evidence.json")
    spec["sources"]["fixture"]["availability_basis"] = "local_observation"
    write_json(root / "execution-evidence.json", spec)
    with pytest.raises(ValueError, match="backdated"):
        ExecutionEvidence(root)


def test_real_complete_prices_cannot_be_claimed_before_their_close(tmp_path):
    snapshot, _, _ = market(tmp_path)
    root = evidence_fixture(tmp_path, snapshot)
    spec = read(root / "execution-evidence.json")
    spec["synthetic"] = False
    spec["sources"]["fixture"]["source_url"] = "https://example.invalid/test-only"
    write_json(root / "execution-evidence.json", spec)
    with pytest.raises(ValueError, match="precedes its final close"):
        ExecutionEvidence(root)


@pytest.mark.parametrize("target", ["raw.txt", "publication.txt", "normalizer.py", "coverage.csv", "bars.parquet"])
def test_every_evidence_dependency_is_hash_bound(tmp_path, target):
    snapshot, _, _ = market(tmp_path)
    root = evidence_fixture(tmp_path, snapshot)
    with (root / target).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        ExecutionEvidence(root)


def test_missing_no_action_coverage_does_not_mean_no_actions(tmp_path):
    snapshot, _, run = market(tmp_path)
    root = evidence_fixture(tmp_path, snapshot)
    frame = pd.read_csv(root / "coverage.csv")
    frame = frame[~frame.component.eq("actions")]
    frame.to_csv(root / "coverage.csv", index=False)
    spec = read(root / "execution-evidence.json")
    spec["coverage"]["sha256"] = file_hash(root / "coverage.csv")
    write_json(root / "execution-evidence.json", spec)
    result = build_execution_snapshot(freeze_decision(run, protocol()), snapshot, root)
    assert not read(result / "gap-report.json")["snapshot_generated"]
    assert all("actions" in row["missing_components"] for row in read(result / "gap-report.json")["candidates"])


def test_frozen_history_cannot_be_replaced_by_later_prices(tmp_path):
    snapshot, _, run = market(tmp_path)
    bundle = freeze_decision(run, protocol())
    root = evidence_fixture(tmp_path, snapshot)
    bars = pd.read_parquet(root / "bars.parquet")
    bars.loc[bars.date.eq(run.name), "amount"] *= 2
    bars.to_parquet(root / "bars.parquet", index=False)
    spec = read(root / "execution-evidence.json")
    spec["tables"]["bars"]["sha256"] = file_hash(root / "bars.parquet")
    write_json(root / "execution-evidence.json", spec)
    report = read(build_execution_snapshot(bundle, snapshot, root) / "gap-report.json")
    assert not report["snapshot_generated"]
    assert any(gap["reason"] == "history_changed" for row in report["candidates"] for gap in row["gaps"])


def test_launch_budget_excludes_reused_runs_and_reports_bottleneck(tmp_path):
    calendar = pd.DataFrame({"date": pd.bdate_range("2026-09-21", periods=8).strftime("%Y-%m-%d"), "is_open": True})
    unmeasured = launch_plan(calendar, "2026-09-21", tmp_path, "2026-09-21T12:00:00Z")
    assert unmeasured["timing_basis"] == "unmeasured" and unmeasured["recommended_latest_start"] is None
    for i, status in enumerate(("completed", "already_completed")):
        write_json(tmp_path / "operations" / f"{i}.json", {"started_at": f"2026-09-{18+i}T10:00:00Z",
            "status": status, "seconds": 3600 if i == 0 else 99999,
            "stages": [{"stage": "train_seed_17", "status": "completed", "seconds": 2500}]})
    plan = launch_plan(calendar, "2026-09-21", tmp_path, "2026-09-22T00:30:00Z")
    assert plan["status"] == "insufficient_measured_budget"
    assert plan["p90_cycle_seconds"] == 3600 and plan["measured_cycles"] == 1
    assert plan["bottlenecks"][0]["stage"] == "train_seed_17"
    assert launch_plan(calendar, "2026-09-21", tmp_path, "2026-09-22T01:15:00Z")["status"] == "missed_deadline"


def test_publication_freezes_decision_before_outcome_review(tmp_path, monkeypatch):
    import daily_token_cycle as operator

    from quant_research.publication import OperationJournal
    snapshot, store, run = market(tmp_path)
    called = []
    real = operator.freeze_decision
    def record(path, cfg):
        called.append("freeze")
        return real(path, cfg)
    monkeypatch.setattr(operator, "freeze_decision", record)
    bundle = operator.freeze_published(run, protocol(), OperationJournal(store))
    assert called == ["freeze"]
    assert read(store / "decision-publications" / (run.name + ".json"))["manifest_sha256"] == file_hash(bundle / "manifest.json")


def test_existing_frozen_cohort_survives_code_revision(tmp_path, monkeypatch):
    import quant_research.research_review as module
    snapshot, store, run = market(tmp_path)
    refresh_research(store, snapshot.path.parent, protocol())
    ref = read(store / "decision-publications" / (run.name + ".json"))
    def reject(*args):
        raise AssertionError("Attempted to recreate a frozen cohort")
    monkeypatch.setattr(module, "frozen_features", reject)
    assert str(module.freeze_decision(run, protocol()).resolve()) == ref["bundle"]


def test_nested_snapshot_manifest_is_covered_by_import_identity(tmp_path):
    from quant_research.daily_loop import verify
    snapshot, _, run = market(tmp_path)
    bundle = freeze_decision(run, protocol())
    root = evidence_fixture(tmp_path, snapshot)
    result = build_execution_snapshot(bundle, snapshot, root)
    assert "snapshot/manifest.json" in read(result / "manifest.json")["files"]
    with (result / "snapshot/manifest.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify(result)


def test_partial_future_session_is_pending_even_with_declared_source_rows(tmp_path, monkeypatch):
    import quant_research.execution_snapshot as module
    snapshot, _, run = market(tmp_path)
    bundle = freeze_decision(run, protocol())
    root = evidence_fixture(tmp_path, snapshot)
    monkeypatch.setattr(module, "utc_now", lambda: snapshot.dates[-1] + "T06:00:00Z")
    spec = read(root / "execution-evidence.json")
    spec["sources"]["fixture"]["observed_at"] = snapshot.dates[-1] + "T05:00:00Z"
    write_json(root / "execution-evidence.json", spec)
    report = read(build_execution_snapshot(bundle, snapshot, root) / "gap-report.json")
    assert not report["snapshot_generated"]
    assert any(gap["reason"] == "outcome_pending" for row in report["candidates"] for gap in row["gaps"])
