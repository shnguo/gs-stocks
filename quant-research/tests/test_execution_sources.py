from pathlib import Path

import pandas as pd
import pytest

from collect_execution_sources import collect, run_job
from quant_research.daily_loop import read, verify, write_manifest
from quant_research.execution_snapshot import ExecutionEvidence, cutoff
from quant_research.execution_sources import export_observations, public_observations, register_observations
from quant_research.fixtures import create_fixture
from quant_research.storage import file_hash, freeze, utc_now, write_json


def archive(workspace, name, api, params, records, observed=None, provider_code=0, msg=""):
    observed = observed or utc_now()
    raw, normalized = workspace / "raw" / name, workspace / "normalized" / name
    write_json(raw / "response.json", {"code": provider_code, "msg": msg, "data": {"fields": [], "items": []}})
    manifest = {"provider": "tushare", "api": api, "params": params, "http_status": 200,
                "provider_code": provider_code, "rows": len(records), "fetched_at": observed,
                "available_at": observed, "availability_basis": "local_observation",
                "raw_sha256": file_hash(raw / "response.json")}
    write_json(raw / "manifest.json", manifest)
    write_json(normalized / "normalized.json", {"api": api, "records": records,
               "coverage": {}, "blockers": [], "complete_history_verified": False})
    write_manifest(normalized)
    norm = read(normalized / "manifest.json")
    norm.update(capture_manifest_sha256=file_hash(raw / "manifest.json"), raw_sha256=manifest["raw_sha256"])
    write_json(normalized / "manifest.json", norm)
    return {"name": name, "api": api, "params": params, "status": "captured" if provider_code == 0 else "provider_rejected",
            "rows": len(records), "observed_at": observed, "raw_sha256": manifest["raw_sha256"]}


def fixture(tmp_path):
    snapshot = create_fixture(tmp_path / "raw", tmp_path / "source/snapshot", stocks=6, days=80)
    ids = {row.instrument_id: f"cn.{row.exchange.lower()}.{ {'XSHG': 600000, 'XSHE': 1, 'XBSE': 920000}[row.exchange] + index:06d}"
           for index, row in enumerate(snapshot.tables["instruments"].itertuples())}
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    for name, frame in snapshot.tables.items():
        frame = frame.copy()
        if "instrument_id" in frame:
            frame.instrument_id = frame.instrument_id.map(ids)
        frame.to_parquet(canonical / (name + ".parquet"), index=False)
    snapshot = freeze(canonical, tmp_path / "mapped/snapshot", snapshot.manifest["declaration"])
    workspace, repo = tmp_path / "captures", tmp_path / "rust"
    for path in (repo / "crates/lakehouse-sources/src/tushare_research.rs", repo / "src/research_sources.rs"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("// deterministic Rust producer fixture\n")
    master = snapshot.tables["instruments"].iloc[0]
    symbol = master.instrument_id
    venue = {"XSHG": "SH", "XSHE": "SZ", "XBSE": "BJ"}[master.exchange]
    record = {"provider_code": symbol.split(".")[-1] + "." + venue, "exchange": master.exchange,
              "industry": "观察行业", "listed_at": master.listed_at, "source_code_kind": "exchange_symbol", "listing_status": "L"}
    contract = {"date": pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d"), "symbols": [symbol]}
    return snapshot, workspace, repo, record, contract


def test_current_industry_has_actual_time_and_no_historical_universe_proof(tmp_path):
    snapshot, workspace, repo, record, contract = fixture(tmp_path)
    job = archive(workspace, "current-master", "stock_basic", {"list_status": "L"}, [record])
    bundle = export_observations(workspace, snapshot.path.parent, repo, contract, [job])
    evidence = ExecutionEvidence(bundle)
    symbol, today = contract["symbols"][0], contract["date"]
    assert evidence.tables["instruments"].set_index("instrument_id").loc[symbol, "industry"] == "观察行业"
    assert evidence.check("industry", symbol, today, pd.Timestamp(job["observed_at"])) is None
    assert evidence.check("industry", symbol, today, cutoff(today, "00:00")) == "available_too_late"
    assert evidence.check("industry", symbol, snapshot.dates[-1], cutoff(snapshot.dates[-1])) == "evidence_missing"
    assert evidence.check("universe", symbol, today, pd.Timestamp(job["observed_at"])) == "evidence_missing"
    assert not read(bundle / "observation-receipt.json")["formal_execution_ready"]
    assert evidence.spec["synthetic"] and read(bundle / "observation-receipt.json")["synthetic"]
    assert export_observations(workspace, snapshot.path.parent, repo, contract, [job]) == bundle


def test_listing_incarnation_mismatch_cannot_receive_current_industry_or_risk_proof(tmp_path):
    snapshot, workspace, repo, record, contract = fixture(tmp_path)
    record["listed_at"] = "2021-01-01"
    jobs = [archive(workspace, "current-master", "stock_basic", {"list_status": "L"}, [record]),
            archive(workspace, "st-list", "stock_st", {"trade_date": contract["date"].replace("-", "")},
                    [{**record, "date": contract["date"], "status": "*ST"}])]
    bundle = export_observations(workspace, snapshot.path.parent, repo, contract, jobs)
    evidence = ExecutionEvidence(bundle)
    assert not evidence.coverage
    assert "risk_events" not in evidence.tables


def test_empty_st_source_never_certifies_normal_or_exchange_coverage(tmp_path):
    snapshot, workspace, repo, record, contract = fixture(tmp_path)
    jobs = [archive(workspace, "current-master", "stock_basic", {"list_status": "L"}, [record]),
            archive(workspace, "st-list", "stock_st", {"trade_date": contract["date"].replace("-", "")}, [])]
    bundle = export_observations(workspace, snapshot.path.parent, repo, contract, jobs)
    evidence = ExecutionEvidence(bundle)
    assert "risk_events" not in evidence.tables and "risk_coverage" not in evidence.tables
    assert evidence.check("risk", contract["symbols"][0], contract["date"], pd.Timestamp.now(tz="UTC")) == "evidence_missing"


def test_limit_observations_do_not_create_missing_ohlc_bars_or_action_entitlements(tmp_path):
    snapshot, workspace, repo, record, contract = fixture(tmp_path)
    future_day = contract["date"]
    limits = {**record, "date": future_day, "upper_limit": 12., "lower_limit": 8., "limits_present": True}
    dividend = {**record, "ex_date": future_day, "cash_div": 1., "holder_terms_verified": False,
                "complete_action_coverage": False}
    jobs = [archive(workspace, "limit", "stk_limit", {"ts_code": record["provider_code"], "trade_date": future_day.replace("-", "")}, [limits]),
            archive(workspace, "dividend", "dividend", {"ts_code": record["provider_code"]}, [dividend])]
    bundle = export_observations(workspace, snapshot.path.parent, repo, contract, jobs)
    evidence = ExecutionEvidence(bundle)
    assert "bars" not in evidence.tables and "actions" not in evidence.tables
    assert read(bundle / "limit-observations.json")[0]["upper_limit"] == 12.
    assert not read(bundle / "dividend-observations.json")[0]["holder_terms_verified"]
    assert not evidence.coverage


def test_existing_bar_limits_are_bound_to_observation_and_never_certify_prices(tmp_path):
    snapshot, workspace, repo, record, contract = fixture(tmp_path)
    day, symbol = snapshot.dates[-1], contract["symbols"][0]
    price = snapshot.tables["bars"].query("instrument_id == @symbol and date == @day").iloc[0]
    row = {**record, "date": day, "upper_limit": float(price.close) * 1.1,
           "lower_limit": float(price.close) * .9, "limits_present": True}
    job = archive(workspace, "limit", "stk_limit", {"ts_code": record["provider_code"], "trade_date": day.replace("-", "")}, [row])
    bundle = export_observations(workspace, snapshot.path.parent, repo, contract, [job])
    evidence = ExecutionEvidence(bundle)
    bars = evidence.tables["bars"]
    assert len(bars) == len(snapshot.tables["bars"])
    pd.testing.assert_series_equal(bars.close, snapshot.tables["bars"].close, check_names=False)
    assert evidence.check("limits", symbol, day, pd.Timestamp(job["observed_at"])) is None
    assert evidence.check("limits", symbol, day, cutoff(day, "09:25")) == "available_too_late"
    assert evidence.check("bars", symbol, day, pd.Timestamp(job["observed_at"])) == "evidence_missing"


def test_nested_source_envelopes_are_hash_protected_and_public_receipt_has_no_paths(tmp_path):
    snapshot, workspace, repo, record, contract = fixture(tmp_path)
    job = archive(workspace, "current-master", "stock_basic", {"list_status": "L"}, [record])
    bundle = export_observations(workspace, snapshot.path.parent, repo, contract, [job])
    register_observations(tmp_path / "store", bundle)
    public = public_observations(tmp_path / "store")
    assert str(tmp_path) not in str(public) and "params" not in public["sources"][0]
    assert "raw/current-master/manifest.json" in read(bundle / "manifest.json")["files"]
    write_json(bundle / "raw/current-master/manifest.json", {"fetched_at": "2000-01-01T00:00:00Z"})
    with pytest.raises(ValueError, match="hash mismatch"):
        public_observations(tmp_path / "store")


def test_local_capture_backdating_or_raw_tampering_is_rejected(tmp_path):
    snapshot, workspace, repo, record, contract = fixture(tmp_path)
    job = archive(workspace, "current-master", "stock_basic", {"list_status": "L"}, [record])
    path = workspace / "raw/current-master/response.json"
    path.write_text("changed")
    with pytest.raises(ValueError, match="hash changed"):
        export_observations(workspace, snapshot.path.parent, repo, contract, [job])


def test_local_observation_cannot_be_backdated_even_with_consistent_hashes(tmp_path):
    snapshot, workspace, repo, record, contract = fixture(tmp_path)
    job = archive(workspace, "current-master", "stock_basic", {"list_status": "L"}, [record])
    path = workspace / "raw/current-master/manifest.json"
    manifest = read(path)
    manifest["available_at"] = "2020-01-01T00:00:00Z"
    write_json(path, manifest)
    norm_path = workspace / "normalized/current-master/manifest.json"
    norm = read(norm_path)
    norm["capture_manifest_sha256"] = file_hash(path)
    write_json(norm_path, norm)
    with pytest.raises(ValueError, match="availability must equal"):
        export_observations(workspace, snapshot.path.parent, repo, contract, [job])


def test_failed_api_is_not_retried_and_other_api_remains_available(tmp_path, monkeypatch):
    workspace = tmp_path / "captures"
    jobs = [{"name": "a", "api": "stk_limit", "params": {"ts_code": "600000.SH"}},
            {"name": "b", "api": "stk_limit", "params": {"ts_code": "600001.SH"}},
            {"name": "c", "api": "dividend", "params": {"ts_code": "600001.SH"}}]
    called = []
    def capture(job, *args):
        called.append(job["name"])
        return {**job, "status": "provider_rejected" if job["name"] == "a" else "captured"}
    monkeypatch.setattr("collect_execution_sources.run_job", capture)
    monkeypatch.setattr("collect_execution_sources.time.sleep", lambda _: None)
    rows = collect(jobs, workspace, Path("not-used"), tmp_path, tmp_path, 8, 3)
    assert called == ["a", "c"]
    assert rows[1]["status"] == "skipped_api_failure"
    called.clear()
    collect(jobs, workspace, Path("not-used"), tmp_path, tmp_path, 8, 3)
    assert not called


def test_resume_validates_archived_bytes_and_classifies_quota_without_request(tmp_path):
    workspace = tmp_path / "captures"
    job = archive(workspace, "failure", "dividend", {"ts_code": "600000.SH"}, [],
                  provider_code=40203, msg="频率超限(1次/分钟)")
    result = run_job(job, workspace, Path("must-not-run"), tmp_path, tmp_path)
    assert result["failure_reason"] == "rate_limited"
    assert "频率超限" not in str(result)
    write_json(workspace / "raw/failure/response.json", {"code": 0})
    with pytest.raises(ValueError, match="Archived source"):
        run_job(job, workspace, Path("must-not-run"), tmp_path, tmp_path)


def test_manual_batch_limits_new_dividend_requests(tmp_path, monkeypatch):
    jobs = [{"name": str(index), "api": "dividend", "params": {"ts_code": "600000.SH"}} for index in range(3)]
    called = []
    def capture(job, *args):
        called.append(job["name"])
        return {**job, "status": "captured"}
    monkeypatch.setattr("collect_execution_sources.run_job", capture)
    monkeypatch.setattr("collect_execution_sources.time.sleep", lambda _: None)
    rows = collect(jobs, tmp_path, Path("not-used"), tmp_path, tmp_path, 8, 3)
    assert called == ["0"]
    assert [row["status"] for row in rows] == ["captured", "not_requested_budget", "not_requested_budget"]


def test_non_json_provider_rejection_is_retained_without_crashing_error_classification(tmp_path):
    workspace = tmp_path / "captures"
    job = archive(workspace, "failure", "stock_st", {"trade_date": "20260930"}, [], provider_code=None)
    path = workspace / "raw/failure/response.json"
    path.write_text("<html>gateway unavailable</html>")
    envelope = workspace / "raw/failure/manifest.json"
    manifest = read(envelope)
    manifest.update(http_status=502, raw_sha256=file_hash(path))
    write_json(envelope, manifest)
    result = run_job(job, workspace, Path("must-not-run"), tmp_path, tmp_path)
    assert result["status"] == "provider_rejected" and result["failure_reason"] == "source_rejected"


def test_failed_normalization_is_not_retried_and_missing_archive_is_not_refetched(tmp_path):
    import shutil
    workspace = tmp_path / "captures"
    job = archive(workspace, "bad", "stock_basic", {"list_status": "L"}, [])
    job["status"] = "normalization_failed"
    shutil.rmtree(workspace / "normalized/bad")
    result = run_job(job, workspace, Path("must-not-run"), tmp_path, tmp_path, archive_only=True)
    assert result["status"] == "normalization_failed"
    shutil.rmtree(workspace / "raw/bad")
    with pytest.raises(ValueError, match="Saved source capture is missing"):
        run_job(job, workspace, Path("must-not-run"), tmp_path, tmp_path, archive_only=True)
