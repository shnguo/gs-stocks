"""Rebuild an approximate Russell 1000 by its ranking rule: once a year, take the 1,000
largest US companies by market cap on the last trading day of May; membership runs
from the July month end through the next June month end.

Usage:
  uv run python scripts/us_russell.py configs/us-factor-v1.json --credentials-file /path/.env

Market cap = cover-page shares outstanding (latest 10-K/10-Q filed by the rank date)
x that day's unadjusted close. ETFs, commodity trusts, partnerships, and stocks under $1
are excluded. Approximation only: no banding, float or country rules.
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.storage import file_hash, utc_now, write_json
from quant_research.us_fundamentals import SHARE_FIELDS, mapping_on
from quant_research.us_sources import (
    Http,
    alpaca_assets,
    alpaca_bars,
    alpaca_credentials,
    alpaca_symbol,
)

BASE = Path(__file__).resolve().parents[1]
EXCLUDED_NAME = re.compile(
    r"\b(?:ETF|ETN|ISHARES|SPDR|PROSHARES|GRAYSCALE|INVESCO DB|TEUCRIUM|BITCOIN|ETHER|"
    r"GOLD TRUST|SILVER TRUST|UNITED STATES OIL|UNITED STATES NATURAL GAS)\b|\bL\.?P\.?$|ROYALTY TRUST",
    re.IGNORECASE)
DEPOSITARY = re.compile(r"depositary|\bADR\b|\bADS\b", re.IGNORECASE)
NON_COMMON = re.compile(r"[-.](?:P|PR|W|WS|WT|U|UN|R|RT)[A-Z]?$")


def shares_on(facts: pd.DataFrame, date: str, max_age_days: int) -> pd.DataFrame:
    """Per CIK: the first share field (in SHARE_FIELDS order) with a recent filing."""
    found = []
    for field in SHARE_FIELDS:
        table = field_shares_on(facts.loc[facts.field == field], date, max_age_days)
        found.append(table.loc[~table.cik.isin(pd.concat([f.cik for f in found]) if found else [])]
                     .assign(share_field=field))
    return pd.concat(found, ignore_index=True)


def field_shares_on(shares: pd.DataFrame, date: str, max_age_days: int) -> pd.DataFrame:
    """Per CIK: shares from the latest filing on or before date, summed across classes."""
    rows = shares.loc[shares.filed <= date]
    latest = rows.loc[rows.filed == rows.groupby("cik").filed.transform("max")]
    latest = latest.loc[latest.end == latest.groupby("cik").end.transform("max")]
    total = (latest.drop_duplicates(["cik", "val", "accn"]).groupby("cik")
             .agg(shares=("val", "sum"), shares_date=("end", "first"), filed=("filed", "first")))
    age = (pd.Timestamp(date) - pd.to_datetime(total.shares_date)).dt.days
    return total.loc[(age <= max_age_days) & (total.shares > 0)].reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--credentials-file", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    r = cfg["russell"]
    out = BASE / r["output"]
    if out.exists():
        raise FileExistsError(f"Use a new output path: {out}")
    started = time.time()
    http = Http(BASE / cfg["cache"], cfg["proxy"], cfg["user_agent"], cfg["min_interval_seconds"])
    creds = alpaca_credentials(args.credentials_file)
    history = pd.read_parquet(BASE / cfg["sec"]["ticker_history"])
    facts = pd.read_parquet(BASE / cfg["sec"]["facts"], columns=["cik", "field", "end", "val", "filed", "accn"])
    shares = facts.loc[facts.field.isin(SHARE_FIELDS)]
    assets = alpaca_assets(http, creds)
    depositary = set(assets.loc[assets.name.str.contains(DEPOSITARY), "symbol"])
    calendar = pd.read_parquet(BASE / r["calendar"]).date.tolist()
    s = pd.Series(calendar)
    month_end = s.groupby(s.str[:7]).max()
    rank_dates = [month_end[f"{y}-05"] for y in range(int(cfg["start"][:4]), int(cfg["end"][:4]) + 1)
                  if f"{y}-05" in month_end.index]

    ranked = []
    for date in rank_dates:
        mapping = mapping_on(history, date)
        mapping = mapping.loc[~mapping.title.str.contains(EXCLUDED_NAME)]
        held = shares_on(shares, date, r["max_shares_age_days"])
        cands = mapping.merge(held, on="cik")
        cands["symbol"] = cands.ticker.map(alpaca_symbol)
        cands = cands.loc[~cands.symbol.isin(depositary)]
        cands = cands.loc[~cands.symbol.str.contains(NON_COMMON) | (cands.groupby("cik").symbol.transform("size") == 1)]
        bars = alpaca_bars(http, creds, sorted(set(cands.symbol)), date, date, date,
                           chunk=r["chunk"], adjustment="raw", retry_missing=False)
        day = bars.loc[bars.date == date, ["symbol", "close", "volume"]]
        cands = cands.merge(day, on="symbol")
        cands["dollar_volume"] = cands.close * cands.volume
        # Primary class = most traded ticker; other classes kept if reasonably traded.
        top = cands.groupby("cik").dollar_volume.transform("max")
        cands["primary"] = cands.dollar_volume == top
        cands = cands.loc[cands.dollar_volume >= r["secondary_volume_share"] * top]
        primary = cands.loc[cands.primary].drop_duplicates("cik")
        primary = primary.assign(market_cap=primary.shares * primary.close)
        # A market cap far out of line with trading (e.g. ordinary shares x ADR price) is an error.
        primary = primary.loc[(primary.close >= r["min_price"])
                              & (primary.dollar_volume >= r["min_turnover"] * primary.market_cap)]
        primary = primary.sort_values("market_cap", ascending=False)
        primary["rank"] = np.arange(1, len(primary) + 1)
        keep = primary.loc[primary["rank"] <= r["size"], ["cik", "rank", "market_cap"]]
        members = cands.merge(keep, on="cik")
        members["rank_date"] = date
        ranked.append(members[["rank_date", "cik", "ticker", "symbol", "title", "primary", "rank",
                               "market_cap", "shares", "share_field", "shares_date", "close",
                               "dollar_volume"]])
        print(f"{date}: {len(cands.cik.unique())} priced companies, cutoff market cap "
              f"{keep.market_cap.min() / 1e9:.2f}bn, {len(members)} tickers, "
              f"{http.fetched} requests, {time.time() - started:.0f}s", flush=True)

    members = pd.concat(ranked, ignore_index=True)
    out.mkdir(parents=True)
    members.to_parquet(out / "members.parquet", index=False)
    write_json(out / "manifest.json", {
        "version": cfg["version"], "created_at": utc_now(), "rule": r, "rank_dates": rank_dates,
        "ticker_history_sha256": file_hash(BASE / cfg["sec"]["ticker_history"]),
        "facts_sha256": file_hash(BASE / cfg["sec"]["facts"]),
        "files": {p.name: file_hash(p) for p in out.iterdir() if p.is_file()},
        "elapsed_seconds": round(time.time() - started, 1)})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
