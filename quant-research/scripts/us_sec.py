"""Fetch SEC inputs for the US factor study: ticker <-> CIK history, companyfacts and
industry (SIC) codes for every company in the universe.

Usage:
  SEC_USER_AGENT="name contact@example.com" uv run python scripts/us_sec.py configs/us-factor-v2.json

SEC requires a contact in the User-Agent. It is read from SEC_USER_AGENT and never
written to disk. Existing outputs are kept; delete one to rebuild it.
"""
import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

from quant_research.us_fundamentals import (
    company_industry,
    parse_companyfacts,
    ticker_snapshot,
    wayback_snapshots,
)
from quant_research.us_sources import Http

BASE = Path(__file__).resolve().parents[1]
COMPANYFACTS = "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"


def download(url: str, path: Path, proxy: str | None, user_agent: str) -> None:
    proxies = {"http": proxy, "https": proxy} if proxy else None
    part = path.with_suffix(path.suffix + ".part")
    with requests.get(url, headers={"User-Agent": user_agent}, proxies=proxies, stream=True, timeout=120) as r:
        r.raise_for_status()
        with part.open("wb") as out:
            for block in r.iter_content(1 << 20):
                out.write(block)
    part.rename(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    cfg = json.loads(parser.parse_args().config.read_text())
    sec = {k: BASE / v for k, v in cfg["sec"].items()}
    started = time.time()

    if not sec["ticker_history"].exists():
        # Archived copies come from the Wayback Machine, not sec.gov, so no contact is needed.
        http = Http(BASE / cfg["cache"], cfg["proxy"], cfg["user_agent"], cfg["min_interval_seconds"])
        stamps = wayback_snapshots(http, cfg["start"], cfg["end"])
        history = pd.concat([ticker_snapshot(http, s) for s in stamps], ignore_index=True)
        sec["ticker_history"].parent.mkdir(parents=True, exist_ok=True)
        history.to_parquet(sec["ticker_history"], index=False)
        print(f"ticker history: {len(stamps)} snapshots, {len(history)} rows", flush=True)

    def sec_agent():
        agent = os.environ.get("SEC_USER_AGENT")
        if not agent:
            raise SystemExit("Set SEC_USER_AGENT to a name and contact email to download from sec.gov")
        return agent

    if not sec["companyfacts_zip"].exists():
        agent = sec_agent()
        sec["companyfacts_zip"].parent.mkdir(parents=True, exist_ok=True)
        download(COMPANYFACTS, sec["companyfacts_zip"], cfg["proxy"], agent)
        print(f"downloaded companyfacts, {time.time() - started:.0f}s", flush=True)

    if not sec["facts"].exists():
        facts = parse_companyfacts(sec["companyfacts_zip"])
        facts.to_parquet(sec["facts"], index=False)
        print(f"facts: {len(facts)} rows, {facts.cik.nunique()} companies", flush=True)
    if "industry" in sec and not sec["industry"].exists():
        # SEC allows 10 requests a second; responses are not cached (large filing lists).
        agent = sec_agent()
        http = Http(BASE / cfg["cache"], cfg["proxy"], cfg["user_agent"], 0.15)
        members = pd.read_parquet(BASE / cfg["universe_output"] / "membership.parquet")
        ciks = sorted(members.cik.dropna().astype(int).unique())
        rows = []
        for i, cik in enumerate(ciks):
            try:
                rows.append(company_industry(http, cik, agent))
            except requests.HTTPError as error:
                rows.append({"cik": int(cik), "error": str(error)})
            if (i + 1) % 250 == 0:
                print(f"  industry {i + 1}/{len(ciks)}, {time.time() - started:.0f}s", flush=True)
        pd.DataFrame(rows).to_parquet(sec["industry"], index=False)
        print(f"industry: {len(rows)} companies", flush=True)
    print(f"done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
