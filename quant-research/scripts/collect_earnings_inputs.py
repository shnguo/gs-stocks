"""Bounded research orchestration of Rust-owned market and earnings captures.

Failed attempts are archived and never silently retried. This does not train,
trade, or upgrade source observations to historical point-in-time evidence.
"""
from __future__ import annotations

import argparse
import fcntl
import subprocess
import time
from pathlib import Path

import pandas as pd

from quant_research.daily_input import normalized, refresh
from quant_research.daily_loop import read, verify
from quant_research.storage import file_hash, utc_now, write_json


def capture_job(job, workspace, executable, repo, env_file):
    raw = workspace / "raw" / job["name"]
    target = workspace / "normalized" / job["name"]
    request = workspace / "requests" / (job["name"] + ".json")
    manifest_path = raw / "manifest.json"
    if not manifest_path.exists():
        write_json(request, job["params"])
        result = subprocess.run(
            [str(executable), "research-source-capture", job["api"], str(request),
             str(raw), str(env_file)],
            cwd=repo, capture_output=True, timeout=40,
        )
        if not manifest_path.exists():
            return {**job, "status": "transport_failed", "exit_code": result.returncode}
    manifest = read(manifest_path)
    if (manifest.get("provider") != "tushare" or manifest["api"] != job["api"]
            or manifest["params"] != job["params"]
            or file_hash(raw / "response.json") != manifest["raw_sha256"]):
        raise ValueError("Archived capture scope or raw hash changed")
    result = {**job, "observed_at": manifest["fetched_at"],
              "raw_sha256": manifest["raw_sha256"], "provider_code": manifest["provider_code"],
              "http_status": manifest["http_status"], "source_rows": manifest["rows"]}
    if manifest["http_status"] != 200 or manifest["provider_code"] != 0:
        message = str(read(raw / "response.json").get("msg", ""))
        category = ("rate_limited" if any(word in message for word in
                    ["频率", "每分钟", "每小时", "每秒"]) else
                    "permission_denied" if "权限" in message else "source_rejected")
        return {**result, "status": "provider_rejected", "failure_reason": category}
    if not target.exists():
        child = subprocess.run(
            [str(executable), "research-source-normalize", str(raw), str(target)],
            cwd=repo, capture_output=True, timeout=40,
        )
        if child.returncode:
            # Raw error strings can include provider data; retain only the category.
            return {**result, "status": "normalization_failed"}
    norm = verify(target)
    if (norm["raw_sha256"] != manifest["raw_sha256"]
            or norm["capture_manifest_sha256"] != file_hash(manifest_path)):
        raise ValueError("Normalized capture lineage changed")
    data = read(target / "normalized.json")
    return {**result, "status": "captured", "records": len(data["records"]),
            "blockers": data["blockers"]}


def collect_jobs(jobs, workspace, executable, repo, env_file, max_requests, interval):
    progress = workspace / "capture-progress.json"
    saved = {row["name"]: row for row in read(progress)["jobs"]} if progress.exists() else {}
    terminal_failures = {"provider_rejected", "transport_failed", "normalization_failed"}
    blocked = {row["api"] for row in saved.values() if row["status"] in terminal_failures}
    new_attempts = 0
    results = []
    for job in jobs:
        old = saved.get(job["name"])
        if old and old["status"] == "captured":
            row = capture_job(job, workspace, executable, repo, env_file)
        elif old and old["status"] in terminal_failures:
            row = old
        elif job["api"] in blocked:
            row = {**job, "status": "skipped_api_failure"}
        elif new_attempts >= max_requests:
            row = {**job, "status": "not_requested_budget"}
        else:
            if new_attempts:
                time.sleep(interval)
            try:
                row = capture_job(job, workspace, executable, repo, env_file)
            except subprocess.TimeoutExpired:
                row = {**job, "status": "transport_failed"}
            new_attempts += 1
        if row["status"] in terminal_failures:
            blocked.add(job["api"])
        results.append(row)
        write_json(progress, {"updated_at": utc_now(), "new_attempts_this_invocation": new_attempts,
                              "jobs": results + [saved[j["name"]] for j in jobs[len(results):]
                                                 if j["name"] in saved]})
        print(f"{job['name']}: {row['status']}", flush=True)
    return results


def market_jobs(config, through):
    store = Path(config["store"])
    current = read(store / "current.json")
    base = Path(current["source"])
    panel = read(base / "panel-h5/manifest.json")
    anchor = panel["price_data_through"]
    evidence = read(config["calendar_evidence"])
    cache = store / "calendar" / (evidence["valid_from"] + "_" + evidence["valid_through"])
    calendar, _ = normalized(cache / "normalized/calendar")
    if not read(cache / "normalized/calendar/normalized.json")["coverage"]["calendar_range_complete"]:
        raise ValueError("Cached calendar is incomplete")
    closed = set()
    for start, end in evidence.get("closed_ranges", []):
        closed.update(pd.date_range(start, end).strftime("%Y-%m-%d"))
    expected = pd.to_datetime(calendar.date).dt.weekday.lt(5) & ~calendar.date.isin(closed)
    if not expected.equals(calendar.is_open.astype(bool)):
        raise ValueError("Provider calendar differs from reviewed exchange notices")
    opened = sorted(calendar.loc[calendar.is_open, "date"])
    if through not in opened or through <= anchor:
        raise ValueError("Market refresh must extend the source to a known completed session")
    now = pd.Timestamp.now(tz="Asia/Shanghai")
    if pd.Timestamp(through + "T15:00:00", tz="Asia/Shanghai") > now:
        raise ValueError("Requested close has not completed")
    sessions = [day for day in opened if anchor <= day <= through]
    jobs = [
        {"name": "calendar", "api": "trade_cal", "params": {
            "exchange": "SSE", "start_date": evidence["valid_from"].replace("-", ""),
            "end_date": evidence["valid_through"].replace("-", "")}},
        {"name": "current-master", "api": "stock_basic", "params": {"list_status": "L"}},
    ]
    for day in sessions:
        stamp = day.replace("-", "")
        jobs.extend([
            {"name": f"daily-{stamp}-reference", "api": "daily",
             "params": {"trade_date": stamp, "limit": 6000}},
            {"name": f"factor-{stamp}", "api": "adj_factor",
             "params": {"trade_date": stamp, "limit": 6000}},
        ])
    next_session = opened[opened.index(through) + 1]
    return jobs, {"base": str(base), "price_through": through, "sessions": sessions,
                  "next_session": next_session, "calendar_evidence": config["calendar_evidence"]}


def market_input_ready(rows):
    """Existing conservative refresh uses references; optional new factors are separate evidence."""
    required = [row for row in rows if row["api"] in {"trade_cal", "stock_basic", "daily"}]
    return bool(required) and all(row["status"] == "captured" for row in required)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/daily-token-v2.json"))
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--phase", choices=["market", "forecasts"], required=True)
    parser.add_argument("--through", help="Last completed market session")
    parser.add_argument("--period", help="One explicit quarter end, YYYYMMDD")
    parser.add_argument("--pilot-code", default="600519.SH")
    parser.add_argument("--max-requests", type=int, default=20)
    parser.add_argument("--interval", type=float, default=2.5)
    parser.add_argument("--build-input", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_requests <= 24 or not 2 <= args.interval <= 60:
        parser.error("Require 1–24 requests and at least 2 seconds between new requests")
    config = read(args.config)
    repo = Path(config["rust_repo"])
    executable = repo / "target/debug/mootdx-cf-rs"
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    if args.phase == "market":
        if not args.through:
            parser.error("Market phase needs --through")
        jobs, context = market_jobs(config, args.through)
    else:
        if not args.period:
            parser.error("Forecast phase needs --period")
        jobs = [
            {"name": "forecast-pilot-" + args.pilot_code, "api": "forecast",
             "params": {"ts_code": args.pilot_code, "period": args.period}},
            {"name": "forecast-market-" + args.period, "api": "forecast_vip",
             "params": {"period": args.period}},
        ]
        context = {"period": args.period, "historical_pit_verified": False}
    with (workspace / ".capture.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        contract = {"schema_version": 1, "phase": args.phase, "context": context,
                    "jobs": jobs, "executable_sha256": file_hash(executable)}
        contract_path = workspace / "capture-contract.json"
        if contract_path.exists() and read(contract_path) != contract:
            raise ValueError("Resume contract changed; do not overwrite or retry archived failures")
        if not contract_path.exists():
            write_json(contract_path, contract)
        rows = collect_jobs(jobs, workspace, executable, repo, Path(config["env_file"]),
                            args.max_requests, args.interval)
        complete = all(row["status"] == "captured" for row in rows)
        input_ready = args.phase == "market" and market_input_ready(rows)
        if args.build_input and input_ready:
            target = workspace / "input"
            if target.exists():
                verify(target / "snapshot")
                verify(target / "panel-h5")
            else:
                refresh(context["base"], workspace / "normalized", target,
                        context["next_session"], context["calendar_evidence"])
            print("Research input:", target, flush=True)
        write_json(workspace / "source-status.json", {
            "updated_at": utc_now(), "phase": args.phase, "all_requests_successful": complete,
            "conservative_research_input_ready": input_ready,
            "captured_requests": sum(row["status"] == "captured" for row in rows),
            "failed_requests": sum(row["status"] in {
                "provider_rejected", "transport_failed", "normalization_failed"} for row in rows),
            "formal_ready": False,
        })


if __name__ == "__main__":
    main()
