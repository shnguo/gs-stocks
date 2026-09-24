"""Bounded, read-only provider and D1 audit. Never print or persist a request token."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env-file", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--market-date", default="20260904")
    p.add_argument("--include-catalog", action="store_true",
                   help="stock_basic may be limited to one request per hour")
    p.add_argument("--name-start", default="20260801")
    p.add_argument("--name-end", default="20260831")
    p.add_argument("--collector", type=Path, default=Path("/Users/guo/github/mootdx-cf/workers/collector"))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    values = {}
    for line in args.env_file.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, val = line.split("=", 1)
            values[key.strip()] = val.strip().strip(chr(34)).strip(chr(39))
    token = values.get("TUSHARE_TOKEN") or values.get("TUSHARE_API_TOKEN")
    if not token:
        raise SystemExit("Tushare token missing from the specified environment file")
    calls = [
        ("stock_st", {"trade_date": args.market_date}, "ts_code,name,trade_date,type,type_name"),
        ("st", {"ts_code": "600228.SH"}, "ts_code,name,pub_date,imp_date,st_type"),
        ("namechange", {"start_date": args.name_start, "end_date": args.name_end},
         "ts_code,name,start_date,end_date,ann_date,change_reason"),
        ("daily", {"trade_date": args.market_date}, "ts_code,trade_date,open,high,low,close,vol,amount"),
        ("adj_factor", {"trade_date": args.market_date}, "ts_code,trade_date,adj_factor"),
    ]
    if args.include_catalog:
        calls.append(("stock_basic", {"list_status": "D"},
                      "ts_code,name,exchange,market,list_date,delist_date"))
    report = {"observed_at": datetime.now(timezone.utc).isoformat(), "provider": [], "d1": []}
    for api, params, fields in calls:
        request_id = hashlib.sha256(json.dumps([api, params, fields], sort_keys=True).encode()).hexdigest()
        cached = args.output / f"{api}-{request_id}.json"
        if cached.exists():
            record = json.loads(cached.read_text())
        else:
            body = json.dumps({"api_name": api, "params": params, "fields": fields, "token": token})
            req = urllib.request.Request("https://api.tushare.pro", data=body.encode(),
                                         headers={"Content-Type": "application/json"})
            record = {"api": api, "params": params, "observed_at": datetime.now(timezone.utc).isoformat()}
            try:
                with urllib.request.urlopen(req, timeout=20) as response:
                    raw = response.read(16 * 1024 * 1024 + 1)
                if len(raw) > 16 * 1024 * 1024:
                    raise ValueError("response exceeds audit byte budget")
                if token.encode() in raw:
                    raise ValueError("provider echoed a secret; response was not retained")
                raw_path = args.output / ("raw-" + hashlib.sha256(raw).hexdigest() + ".json")
                raw_path.write_bytes(raw)
                obj = json.loads(raw)
                data = obj.get("data") or {}
                record.update(code=obj.get("code"), message=obj.get("msg"),
                              rows=len(data.get("items") or []), fields=data.get("fields"),
                              raw_file=raw_path.name, raw_sha256=hashlib.sha256(raw).hexdigest())
            except Exception as error:
                record.update(error_type=type(error).__name__, error=str(error).replace(token, "[redacted]")[:300])
            cached.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
            time.sleep(1)
        report["provider"].append(record)
        print(json.dumps({k: record.get(k) for k in ["api", "code", "rows", "error_type"]}), flush=True)
    queries = {
        "current_universes": "SELECT snapshot_id,as_of_date,member_count FROM ashare_universe_snapshots ORDER BY as_of_date DESC,created_at DESC LIMIT 5",
        "history": "SELECT bootstrap_id,requested_start,requested_end,status,expected_windows,manifest_verified FROM historical_bootstraps ORDER BY updated_at DESC LIMIT 12",
        "listing_inventory": "SELECT exchange_id,COUNT(DISTINCT instrument_id) AS instruments,MIN(valid_from) AS earliest_interval,MAX(valid_to) AS latest_closed_interval FROM reference_listing_revisions GROUP BY exchange_id",
        "calendars": "SELECT exchange_id,MIN(trading_date) AS first_date,MAX(trading_date) AS last_date,COUNT(DISTINCT trading_date) AS dates FROM reference_trading_day_revisions GROUP BY exchange_id",
        "daily_coverage": "SELECT business_date,status FROM ashare_market_day_coverage ORDER BY business_date DESC LIMIT 12",
        "instrument_name_history": "SELECT MIN(fetched_at) AS first_observed,MAX(fetched_at) AS last_observed,COUNT(*) AS revisions,COUNT(DISTINCT instrument_id) AS instruments,SUM(CASE WHEN UPPER(json_extract(record_json,'$.name')) LIKE 'ST%' OR UPPER(json_extract(record_json,'$.name')) LIKE '*ST%' OR UPPER(json_extract(record_json,'$.name')) LIKE 'SST%' OR UPPER(json_extract(record_json,'$.name')) LIKE 'S*ST%' THEN 1 ELSE 0 END) AS risk_name_revisions,SUM(CASE WHEN json_extract(record_json,'$.published_at') IS NULL THEN 1 ELSE 0 END) AS without_published_at FROM reference_instrument_revisions",
    }
    for name, sql in queries.items():
        command = [str(args.collector / "node_modules/.bin/wrangler"), "d1", "execute",
                   "mootdx-cf-control-preview-deployment-01", "--remote", "--config",
                   "wrangler.preview.jsonc", "--command", sql, "--json"]
        try:
            result = subprocess.run(command, cwd=args.collector, capture_output=True, text=True, timeout=40,
                                    env={**os.environ, "WRANGLER_SEND_METRICS": "false"})
            raw = result.stdout.replace(token, "[redacted]")
            parsed = json.loads(raw)
            record = {"query": name, "sql": sql, "exit_code": result.returncode, "result": parsed}
        except Exception as error:
            record = {"query": name, "error_type": type(error).__name__}
        report["d1"].append(record)
        print(json.dumps({"query": name, "exit_code": record.get("exit_code"), "error_type": record.get("error_type")}), flush=True)
    (args.output / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
