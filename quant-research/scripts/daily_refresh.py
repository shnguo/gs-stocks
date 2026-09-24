#!/usr/bin/env python3
"""Fetch through the Rust data service, refresh research inputs, emit a daily strategy."""

import argparse
import subprocess
from pathlib import Path

import pandas as pd

from quant_research.daily_input import normalized, refresh
from quant_research.daily_loop import generate, read, verify
from quant_research.storage import utc_now, write_json


def capture(repo, env_file, workspace, api, name, params):
    request = workspace / "requests" / f"{name}.json"
    destination = workspace / "raw" / name
    executable = repo / "target/release/mootdx-cf-rs"
    if not destination.exists():
        write_json(request, params)
        subprocess.run(
            [
                str(executable),
                "research-source-capture",
                api,
                str(request),
                str(destination),
                str(env_file),
            ],
            cwd=repo,
            check=True,
        )
    manifest = read(destination / "manifest.json")
    if manifest["provider_code"] != 0 or manifest["params"] != params:
        raise ValueError(
            "Failed or mismatched archived capture; retained for review, no quota bypass"
        )
    target = workspace / "normalized" / name
    if not target.exists():
        subprocess.run(
            [str(executable), "research-source-normalize", str(destination), str(target)],
            cwd=repo,
            check=True,
        )
    verify(target)
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in [
        "base",
        "workspace",
        "rust-repo",
        "env-file",
        "package",
        "store",
        "calendar-evidence",
    ]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--date", required=True)
    p.add_argument("--mode", choices=["prospective", "intraday_research"], default="prospective")
    p.add_argument(
        "--skip-fetch", action="store_true", help="Use already archived successful captures"
    )
    p.add_argument("--input-version", default="input")
    args = p.parse_args()
    for name in [
        "base",
        "workspace",
        "rust_repo",
        "env_file",
        "package",
        "store",
        "calendar_evidence",
    ]:
        setattr(args, name, getattr(args, name).resolve())
    if not args.skip_fetch:
        evidence = read(args.calendar_evidence)
        calendar_path = capture(
            args.rust_repo,
            args.env_file,
            args.workspace,
            "trade_cal",
            "calendar",
            {
                "exchange": "SSE",
                "start_date": evidence["valid_from"].replace("-", ""),
                "end_date": evidence["valid_through"].replace("-", ""),
            },
        )
        calendar, _ = normalized(calendar_path)
        capture(
            args.rust_repo,
            args.env_file,
            args.workspace,
            "stock_basic",
            "current-master",
            {"list_status": "L"},
        )
        anchor = (
            read(args.base / "panel-h5/manifest.json")["price_data_through"]
            if "price_data_through" in read(args.base / "panel-h5/manifest.json")
            else read(args.base / "panel-h5/manifest.json")["dates"][-1]
        )
        for date in calendar.loc[
            calendar.is_open & calendar.date.ge(anchor) & calendar.date.lt(args.date), "date"
        ]:
            d = date.replace("-", "")
            capture(
                args.rust_repo,
                args.env_file,
                args.workspace,
                "daily",
                f"daily-{d}-reference",
                {"trade_date": d, "limit": 6000},
            )
    source = args.workspace / args.input_version
    if source.exists():
        verify(source / "snapshot")
        verify(source / "panel-h5")
        if read(source / "refresh-summary.json")["horizon_dates"][0] != args.date:
            raise ValueError("Saved input belongs to another trade date")
    else:
        refresh(args.base, args.workspace / "normalized", source, args.date, args.calendar_evidence)
    cutoff = pd.Timestamp(utc_now()).tz_convert("Asia/Shanghai").isoformat()
    run = generate(source, args.package, args.store, args.date, cutoff, args.mode)
    receipt = {
        "trade_date": args.date,
        "mode": args.mode,
        "input": str(source),
        "run": str(run),
        "status": read(run / "run.json")["status"],
        "generated_at": utc_now(),
    }
    write_json(args.workspace / f"delivery-{run.name}.json", receipt)
    print(run / "report.md")
    if receipt["status"] == "blocked":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
