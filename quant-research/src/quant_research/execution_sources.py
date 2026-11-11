"""Research adapters for archived Rust observations, with conservative coverage.

Provider parsing stays in Rust. Current observations do not establish historical
listing identity, normal-risk coverage, action entitlements or trading rules.
"""
from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

from .daily_loop import digest, read, verify, write_manifest
from .execution_snapshot import ExecutionEvidence, aware
from .storage import file_hash, read_table, utc_now, write_json

SOURCE_URLS = {
    "stock_basic": "https://tushare.pro/document/1?doc_id=25",
    "stock_st": "https://tushare.pro/document/2?doc_id=397",
    "stk_limit": "https://tushare.pro/document/2?doc_id=183",
    "dividend": "https://tushare.pro/document/2?doc_id=103",
}
API_LABELS = {"stock_basic": "当前股票与行业", "stock_st": "ST风险状态",
              "stk_limit": "逐股涨跌停价格", "dividend": "分红观察记录"}
STATUS_LABELS = {"captured": "已采集", "provider_rejected": "来源拒绝请求",
                 "transport_failed": "连接失败", "normalization_failed": "来源校验未通过",
                 "skipped_api_failure": "同接口失败后停止", "not_requested_budget": "等待后续手动采集"}
FAILURE_LABELS = {"permission_denied": "缺少接口访问权限", "rate_limited": "接口频率限制",
                  "source_rejected": "来源拒绝请求"}
COVERAGE_COLUMNS = ["component", "instrument_id", "start_date", "end_date", "known_at", "source_id", "status"]


def instrument_id(record):
    """Reuse the Rust-validated venue/code, without merging listing incarnations."""
    code = record["provider_code"].split(".")[0]
    return "cn." + record["exchange"].lower() + "." + code


def load_capture(workspace, job):
    raw = workspace / "raw" / job["name"]
    normalized = workspace / "normalized" / job["name"]
    manifest = read(raw / "manifest.json")
    if (manifest.get("provider") != "tushare" or manifest["api"] != job["api"] or
            manifest["params"] != job["params"] or manifest["provider_code"] != 0 or
            manifest["http_status"] != 200 or file_hash(raw / "response.json") != manifest["raw_sha256"]):
        raise ValueError("Execution observation capture identity/status/hash changed")
    verify(normalized)
    norm = read(normalized / "manifest.json")
    if norm["capture_manifest_sha256"] != file_hash(raw / "manifest.json") or norm["raw_sha256"] != manifest["raw_sha256"]:
        raise ValueError("Execution observation normalized lineage changed")
    observed = aware(manifest["fetched_at"])
    if (manifest.get("availability_basis") != "local_observation" or
            aware(manifest["available_at"]) != observed or observed > aware(utc_now())):
        raise ValueError("Local observation availability must equal its actual capture time")
    data = read(normalized / "normalized.json")
    if data["api"] != job["api"]:
        raise ValueError("Rust observation scope changed")
    return manifest, data


def export_observations(workspace, source, rust_repo, contract, jobs):
    workspace, source, rust_repo = map(Path, (workspace, source, rust_repo))
    snapshot_manifest = verify(source / "snapshot")
    synthetic = snapshot_manifest["declaration"]["synthetic"]
    captures = {}
    for job in jobs:
        if job["status"] == "captured":
            captures[job["name"]] = load_capture(workspace, job)
    producer_files = {"execution_sources.py": Path(__file__),
                      "tushare_research.rs": rust_repo / "crates/lakehouse-sources/src/tushare_research.rs",
                      "research_sources.rs": rust_repo / "src/research_sources.rs"}
    archived_files = {}
    for job in jobs:
        for kind in ("raw", "normalized"):
            directory = workspace / kind / job["name"]
            for filename in ("response.json", "normalized.json", "manifest.json"):
                path = directory / filename
                if path.exists():
                    archived_files[f"{kind}/{job['name']}/{filename}"] = path
    identity = {"contract": contract, "jobs": jobs, "source_snapshot_sha256": file_hash(source / "snapshot/manifest.json"),
                "capture_files": {key: file_hash(path) for key, path in archived_files.items()},
                "producer_files": {key: file_hash(path) for key, path in producer_files.items()}}
    destination = workspace / "exports" / digest(identity)[:16]
    if destination.exists():
        verify(destination)
        ExecutionEvidence(destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".observations-", dir=destination.parent))
    try:
        for relative, path in archived_files.items():
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        with zipfile.ZipFile(staging / "producer.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, path in producer_files.items():
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                archive.writestr(info, path.read_bytes())
        sources = {}
        for name, (manifest, _) in captures.items():
            sources[name] = {"path": f"raw/{name}/response.json", "sha256": manifest["raw_sha256"],
                             "source_url": SOURCE_URLS[manifest["api"]], "observed_at": manifest["fetched_at"],
                             "available_at": manifest["available_at"], "availability_basis": "local_observation"}
        claims, tables = [], {}

        def claim(component, symbol, day, name, status="verified_pit"):
            claims.append(dict(component=component, instrument_id=symbol, start_date=day, end_date=day,
                               known_at=sources[name]["available_at"], source_id=name, status=status))

        def table(name, frame, source_ids):
            frame.to_parquet(staging / (name + ".parquet"), index=False)
            tables[name] = {"path": name + ".parquet", "sha256": file_hash(staging / (name + ".parquet")),
                            "source_ids": source_ids, "normalizer_path": "producer.zip",
                            "normalizer_sha256": file_hash(staging / "producer.zip")}

        master = read_table(source / "snapshot", "instruments").copy()
        master_index = master.set_index("instrument_id")
        current_identities = set()
        if "current-master" in captures:
            manifest, data = captures["current-master"]
            day = aware(manifest["fetched_at"]).tz_convert("Asia/Shanghai").strftime("%Y-%m-%d")
            for record in data["records"]:
                symbol, industry = instrument_id(record), record.get("industry")
                if symbol not in master_index.index:
                    continue
                frozen = master_index.loc[symbol]
                if (str(frozen.listed_at) != record["listed_at"] or str(frozen.delisted_at) not in {"", "None"} or
                        record.get("source_code_kind") != "exchange_symbol" or record["listing_status"] != "L"):
                    continue
                current_identities.add(symbol)
                if industry and str(industry).strip() not in {"unknown", "未知"}:
                    master.loc[master.instrument_id.eq(symbol), "industry"] = industry
                    claim("industry", symbol, day, "current-master")
            table("instruments", master, ["current-master"])

        # Explicit ST observations only. Absence from a current list does not
        # certify a historical instrument or the entire exchange as normal.
        if "st-list" in captures:
            manifest, data = captures["st-list"]
            events = read_table(source / "snapshot", "risk_events").copy()
            additions = []
            for record in data["records"]:
                symbol = instrument_id(record)
                master_known = aware(captures["current-master"][0]["fetched_at"]) if "current-master" in captures else None
                if (symbol in current_identities and record["status"] in {"ST", "*ST"} and
                        master_known is not None and aware(manifest["fetched_at"]) >= master_known):
                    additions.append(dict(instrument_id=symbol, start_date=record["date"], end_date=record["date"],
                                          known_at=manifest["fetched_at"], status=record["status"]))
                    claim("risk", symbol, record["date"], "st-list")
            if additions:
                events = pd.concat([events, pd.DataFrame(additions)], ignore_index=True)
                events.known_at = pd.to_datetime(events.known_at, utc=True)
                table("risk_events", events, ["current-master", "st-list"])

        limits, dividends = [], []
        for name, (manifest, data) in captures.items():
            for record in data["records"]:
                if manifest["api"] == "stk_limit":
                    limits.append({**record, "instrument_id": instrument_id(record), "source_id": name,
                                   "observed_at": manifest["fetched_at"]})
                elif manifest["api"] == "dividend":
                    dividends.append({**record, "instrument_id": instrument_id(record), "source_id": name,
                                      "observed_at": manifest["fetched_at"]})
        write_json(staging / "limit-observations.json", limits)
        write_json(staging / "dividend-observations.json", dividends)
        if limits:
            bars = read_table(source / "snapshot", "bars").copy()
            matched = []
            for record in limits:
                mask = bars.instrument_id.eq(record["instrument_id"]) & bars.date.eq(record["date"])
                if mask.any() and record["limits_present"]:
                    bars.loc[mask, ["upper_limit", "lower_limit"]] = [record["upper_limit"], record["lower_limit"]]
                    matched.append(record["source_id"])
                    claim("limits", record["instrument_id"], record["date"], record["source_id"])
            if matched:
                table("bars", bars, sorted(set(matched)))

        pd.DataFrame(claims, columns=COVERAGE_COLUMNS).to_csv(staging / "coverage.csv", index=False)
        write_json(staging / "execution-evidence.json", {"schema_version": 1, "synthetic": synthetic,
                   "sources": sources, "tables": tables,
                   "coverage": {"path": "coverage.csv", "sha256": file_hash(staging / "coverage.csv")}})
        records = [{**{key: value for key, value in job.items() if key not in {"params", "exit_code"}},
                    "symbol": job["params"].get("ts_code"), "source_url": SOURCE_URLS[job["api"]],
                    "label": API_LABELS[job["api"]], "status_label": STATUS_LABELS[job["status"]],
                    "failure_label": FAILURE_LABELS.get(job.get("failure_reason", ""))}
                   for job in jobs]
        # This receipt is public. Paths, credentials and raw provider error
        # messages stay in the private, immutable archive.
        receipt = {"schema_version": 1, "date": contract["date"], "generated_at": utc_now(),
                   "availability_basis": "local_observation", "synthetic": synthetic,
                   "planned_symbols": len(contract["symbols"]), "sources": records,
                   "captured_requests": sum(row["status"] == "captured" for row in jobs),
                   "failed_requests": sum(row["status"] in {"provider_rejected", "transport_failed", "normalization_failed"} for row in jobs),
                   "unrequested_requests": sum(row["status"] in {"skipped_api_failure", "not_requested_budget"} for row in jobs),
                   "limit_records": len(limits), "dividend_records": len(dividends),
                   "formal_execution_ready": False,
                   "remaining_requirements": ["historical_universe_identity", "normal_risk_coverage",
                       "complete_action_entitlements", "official_rules_and_fees", "complete_price_factor_coverage"],
                   "policy": "Current captures retain their actual availability. They never repair late historical publications or prove complete action coverage."}
        write_json(staging / "observation-receipt.json", receipt)
        write_json(staging / "lineage.json", identity)
        ExecutionEvidence(staging)
        write_manifest(staging)
        manifest = read(staging / "manifest.json")
        # The generic helper excludes manifest basenames. These nested source
        # envelopes also carry scope and timestamps and must be hash protected.
        manifest["files"].update({relative: file_hash(staging / relative) for relative in archived_files
                                  if relative.endswith("/manifest.json")})
        write_json(staging / "manifest.json", manifest)
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination


def register_observations(store, bundle):
    store, bundle = Path(store), Path(bundle).resolve()
    verify(bundle)
    write_json(store / "execution-observations-publication.json", {
        "bundle": str(bundle), "manifest_sha256": file_hash(bundle / "manifest.json")})


def public_observations(store):
    pointer = Path(store) / "execution-observations-publication.json"
    if not pointer.exists():
        return None
    reference = read(pointer)
    bundle = Path(reference["bundle"])
    verify(bundle)
    if file_hash(bundle / "manifest.json") != reference["manifest_sha256"]:
        raise ValueError("Registered execution observations changed")
    return {**read(bundle / "observation-receipt.json"), "manifest_sha256": reference["manifest_sha256"]}
