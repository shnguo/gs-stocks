#!/usr/bin/env python3
"""Manually archive bounded execution observations through the Rust source CLI."""
from __future__ import annotations

import argparse
import fcntl
import subprocess
import time
from pathlib import Path

import pandas as pd

from quant_research.daily_loop import read, verify
from quant_research.execution_sources import export_observations, register_observations
from quant_research.storage import file_hash, utc_now, write_json


def requests_for(decision, date, max_symbols):
    verify(decision)
    spec = read(decision / "decision.json")
    frame = pd.read_csv(decision / "cohort.csv").sort_values(["rank", "arm"])
    frame = frame.groupby("arm", sort=False).head(spec["protocol"]["top_n"])
    symbols = frame.instrument_id.drop_duplicates().tolist()[:max_symbols]
    jobs = [dict(name="current-master", api="stock_basic", params={"list_status": "L"}),
            dict(name="st-list", api="stock_st", params={"trade_date": date.replace("-", "")})]
    for symbol in symbols:
        parts = symbol.split(".")
        if len(parts) != 3 or parts[0] != "cn" or parts[1] not in {"xshg", "xshe", "xbse"} or not (len(parts[2]) == 6 and parts[2].isdigit()):
            raise ValueError("Candidate has no unambiguous provider code")
        code = parts[2] + "." + {"xshg": "SH", "xshe": "SZ", "xbse": "BJ"}[parts[1]]
        for api in ("stk_limit", "dividend"):
            params = {"ts_code": code}
            if api == "stk_limit":
                params["trade_date"] = date.replace("-", "")
            jobs.append(dict(name=api + "-" + code, api=api, params=params))
    return jobs, symbols


def run_job(job, workspace, executable, repo, env_file, archive_only=False):
    """Never retry failed captures or accept a mismatched archived envelope."""
    raw = workspace / "raw" / job["name"]
    normalized = workspace / "normalized" / job["name"]
    request = workspace / "requests" / (job["name"] + ".json")
    if not raw.exists():
        if archive_only:
            raise ValueError("Saved source capture is missing; restore its archive")
        write_json(request, job["params"])
        result = subprocess.run([str(executable), "research-source-capture", job["api"],
                                 str(request), str(raw), str(env_file)], cwd=repo,
                                capture_output=True, timeout=45)
        if not (raw / "manifest.json").exists():
            return {**job, "status": "transport_failed", "exit_code": result.returncode}
    manifest = read(raw / "manifest.json")
    if (manifest.get("provider") != "tushare" or manifest.get("api") != job["api"] or
            manifest.get("params") != job["params"] or
            file_hash(raw / "response.json") != manifest.get("raw_sha256")):
        raise ValueError("Archived source request or raw bytes changed")
    base = {**job, "http_status": manifest["http_status"], "provider_code": manifest["provider_code"],
            "raw_rows": manifest["rows"], "observed_at": manifest["fetched_at"],
            "raw_sha256": manifest["raw_sha256"]}
    if manifest["http_status"] != 200 or manifest["provider_code"] != 0:
        # Only a safe error category is exposed, never the provider message.
        try:
            response = read(raw / "response.json")
            message = str(response.get("msg", "")) if isinstance(response, dict) else ""
        except ValueError:
            message = ""
        category = "rate_limited" if "频率超限" in message else "permission_denied" if "访问权限" in message else "source_rejected"
        return {**base, "status": "provider_rejected", "failure_reason": category}
    if archive_only and job.get("status") == "normalization_failed":
        return {**base, "status": "normalization_failed"}
    if not normalized.exists():
        if archive_only:
            raise ValueError("Saved normalized source is missing; restore its archive")
        result = subprocess.run([str(executable), "research-source-normalize", str(raw), str(normalized)],
                                cwd=repo, capture_output=True, timeout=45)
        if result.returncode:
            return {**base, "status": "normalization_failed", "exit_code": result.returncode}
    verify(normalized)
    norm = read(normalized / "manifest.json")
    if (norm["capture_manifest_sha256"] != file_hash(raw / "manifest.json") or
            norm["raw_sha256"] != manifest["raw_sha256"]):
        raise ValueError("Normalized source has changed lineage")
    data = read(normalized / "normalized.json")
    return {**base, "status": "captured", "rows": len(data["records"]), "blockers": data["blockers"]}


def collect(jobs, workspace, executable, repo, env_file, max_requests, interval, max_dividends=1):
    receipt_path = workspace / "capture-progress.json"
    previous = read(receipt_path) if receipt_path.exists() else {"jobs": []}
    saved = {row["name"]: row for row in previous["jobs"]}
    blocked = {row["api"] for row in saved.values() if row["status"] in
               {"provider_rejected", "transport_failed", "normalization_failed"}}
    dividend_times = [pd.Timestamp(row["observed_at"]) for row in saved.values()
                      if row["api"] == "dividend" and row.get("observed_at")]
    dividend_waiting = bool(dividend_times and max(dividend_times) > pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=61))
    count, dividends, result = 0, 0, []
    for job in jobs:
        old = saved.get(job["name"])
        if old and old["status"] not in {"not_requested_budget", "skipped_api_failure"}:
            # Verify successful/failed archived bytes on resume, without another request.
            row = run_job({**job, "status": old["status"]}, workspace, executable, repo, env_file,
                          archive_only=True) if old.get("raw_sha256") else old
        elif job["api"] in blocked:
            row = {**job, "status": "skipped_api_failure"}
        elif count >= max_requests or (job["api"] == "dividend" and (dividends >= max_dividends or dividend_waiting)):
            row = {**job, "status": "not_requested_budget"}
        else:
            if count:
                time.sleep(interval)
            try:
                row = run_job(job, workspace, executable, repo, env_file)
            except subprocess.TimeoutExpired:
                row = {**job, "status": "transport_failed"}
            count += 1
            if job["api"] == "dividend":
                dividends += 1
            if row["status"] != "captured":
                blocked.add(job["api"])
        result.append(row)
        write_json(receipt_path, {"updated_at": utc_now(), "jobs": result +
                   [saved[item["name"]] for item in jobs[len(result):] if item["name"] in saved]})
        if row["status"] not in {"skipped_api_failure", "not_requested_budget"}:
            print(f"{job['api']} {job['params'].get('ts_code', 'market')}: {row['status']}", flush=True)
    print(f"Planned {len(jobs)}; new attempts {count}; unrequested " +
          str(sum(row["status"] in {"skipped_api_failure", "not_requested_budget"} for row in result)), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--workspace", type=Path, required=True, help="New archive or exact resume directory")
    p.add_argument("--executable", type=Path, help="Rust CLI; defaults to the locally built debug binary")
    p.add_argument("--date", default=pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d"))
    p.add_argument("--max-symbols", type=int, default=40)
    p.add_argument("--max-requests", type=int, default=82, help="New requests this invocation; failures are never retried")
    p.add_argument("--interval", type=float, default=3, help="Seconds between new requests, at least 2")
    args = p.parse_args()
    if not (1 <= args.max_symbols <= 40 and 1 <= args.max_requests <= 82 and 2 <= args.interval <= 60):
        p.error("Require 1–40 stocks, 1–82 requests, and 2–60 seconds between requests")
    day = pd.Timestamp(args.date)
    if day.strftime("%Y-%m-%d") != args.date or day.date() > pd.Timestamp.now(tz="Asia/Shanghai").date():
        p.error("Capture date must be an explicit completed or current calendar date")
    cfg = read(args.config)
    store, repo = Path(cfg["store"]), Path(cfg["rust_repo"])
    current = read(store / "current.json")
    decision = Path(read(store / "decision-publications" / (current["signal_date"] + ".json"))["bundle"])
    jobs, symbols = requests_for(decision, args.date, args.max_symbols)
    executable = (args.executable or repo / "target/debug/mootdx-cf-rs").resolve()
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    with (workspace / ".capture.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        contract = {"schema_version": 1, "date": args.date, "decision_sha256": file_hash(decision / "manifest.json"),
                    "jobs": jobs, "symbols": symbols, "executable_sha256": file_hash(executable)}
        contract_path = workspace / "capture-contract.json"
        if contract_path.exists() and read(contract_path) != contract:
            raise ValueError("Resume contract changed; use a new workspace")
        if not contract_path.exists():
            write_json(contract_path, contract)
        rows = collect(jobs, workspace, executable, repo, Path(cfg["env_file"]), args.max_requests, args.interval)
        bundle = export_observations(workspace, Path(current["source"]), repo, contract, rows)
        register_observations(store, bundle)
        print("Observation bundle:", bundle, flush=True)


if __name__ == "__main__":
    main()
