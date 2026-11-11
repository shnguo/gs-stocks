"""Collect point-in-time S&P 500 / Nasdaq-100 / rebuilt Russell 1000 membership and Alpaca daily bars.

Usage:
  uv run python scripts/us_collect.py configs/us-factor-v1.json --credentials-file /path/.env

Each membership run (ticker, first month, last month) is priced with Alpaca symbol
resolution as of its last month, so renamed tickers keep their full history. Runs
that resolve to the same company (identical bar series) are merged into one entity.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd

from quant_research.storage import file_hash, utc_now, write_json
from quant_research.us_fundamentals import mapping_on
from quant_research.us_sources import (
    Http,
    alpaca_bars,
    alpaca_credentials,
    alpaca_symbol,
    membership_runs,
    ndx_changes,
    ndx_current,
    ndx_membership,
    sp500_membership,
    wiki_tables,
)

BASE = Path(__file__).resolve().parents[1]


def month_ends(dates: list[str]) -> list[str]:
    s = pd.Series(sorted(dates))
    return s.groupby(s.str[:7]).max().tolist()


def cik_for(history: pd.DataFrame, ticker: str, date: str):
    """CIK holding the ticker in the SEC ticker map nearest before date (None if unknown)."""
    match = mapping_on(history, date)
    found = match.loc[match.ticker == ticker.replace(".", "-")]
    if found.empty:
        found = match.loc[match.ticker == ticker]
    return int(found.cik.iloc[0]) if len(found) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--credentials-file", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    out = BASE / cfg["universe_output"]
    if out.exists():
        raise FileExistsError(f"Collections use new output paths: {out}")
    started = time.time()
    http = Http(BASE / cfg["cache"], cfg["proxy"], cfg["user_agent"], cfg["min_interval_seconds"])
    creds = alpaca_credentials(args.credentials_file)

    # Trading calendar from the benchmark's bars.
    bench = alpaca_bars(http, creds, [cfg["benchmark"]], None, cfg["start"], cfg["end"])
    sessions = sorted(bench.date.unique())
    # Complete months only; the month containing the end date is still open.
    ends = [d for d in month_ends(sessions) if d < cfg["end"][:7] + "-01"]
    print(f"{len(sessions)} sessions, {len(ends)} month ends", flush=True)

    sp = sp500_membership(http, cfg["sp500_commit"])
    sp_dates = sp.date.tolist()
    sp_monthly = {}
    for d in ends:
        k = sum(x <= d for x in sp_dates) - 1
        if k < 0:
            raise ValueError("S&P 500 history starts after the study")
        sp_monthly[d] = set(sp.tickers.iloc[k])
    current_tables, _ = wiki_tables(http, cfg["ndx_current_page"], cfg["ndx_current_revision"])
    history_tables, _ = wiki_tables(http, cfg["ndx_history_page"], cfg["ndx_history_revision"])
    ndx_monthly = ndx_membership(ndx_current(current_tables), ndx_changes(history_tables),
                                 cfg["ndx_as_of"], ends)

    union = {d: sp_monthly[d] | ndx_monthly[d] for d in ends}
    runs = membership_runs(union)
    runs["asof"] = runs["last"]
    runs["source"] = "sp500_ndx"
    history = pd.read_parquet(BASE / cfg["sec"]["ticker_history"])
    runs["cik"] = [cik_for(history, t, d) for t, d in zip(runs.ticker, runs["last"])]

    # Russell 1000 rebuild: members ranked at the end of May, held July through June.
    russell = pd.read_parquet(BASE / cfg["russell"]["output"] / "members.parquet")
    extra = []
    for (rank_date, cik, ticker), _ in russell.groupby(["rank_date", "cik", "ticker"]):
        year = int(rank_date[:4])
        months = [m for m in ends if f"{year}-07-01" <= m < f"{year + 1}-07-01"]
        if months:
            extra.append(dict(ticker=ticker, first=months[0], last=months[-1], asof=rank_date,
                              source="r1000", cik=int(cik)))
    runs = pd.concat([runs, pd.DataFrame(extra)], ignore_index=True)
    runs["symbol"] = runs.ticker.map(alpaca_symbol)
    runs["key"] = runs.symbol + "@" + runs["asof"]
    print(f"{len(runs)} membership runs, {runs['asof'].nunique()} resolution dates", flush=True)

    frames, raw_frames = [], []
    groups = list(runs.groupby("asof"))
    for i, (asof, group) in enumerate(groups):
        symbols = sorted(set(group.symbol))
        bars = alpaca_bars(http, creds, symbols, asof, cfg["start"], cfg["end"])
        bars["key"] = bars.symbol + "@" + asof
        frames.append(bars)
        raw = alpaca_bars(http, creds, symbols, asof, cfg["start"], cfg["end"], adjustment="raw")
        raw_frames.append(raw.assign(key=raw.symbol + "@" + asof)[["key", "date", "close"]])
        if (i + 1) % 20 == 0 or i + 1 == len(groups):
            print(f"  priced {i + 1}/{len(groups)} dates, {http.fetched} requests, "
                  f"{time.time() - started:.0f}s", flush=True)
    bars = pd.concat(frames, ignore_index=True)
    raw = pd.concat(raw_frames, ignore_index=True).rename(columns={"close": "raw_close"})

    # Merge keys that resolve to the same company: identical (date, close) series.
    signature = {k: hashlib.sha256(g.sort_values("date")[["date", "close"]].to_csv(index=False).encode())
                 .hexdigest() for k, g in bars.groupby("key")}
    canonical = {}
    for key, sig in sorted(signature.items(), key=lambda kv: kv[0].split("@")[1], reverse=True):
        canonical.setdefault(sig, key)
    runs["entity"] = runs.key.map(lambda k: canonical.get(signature.get(k)))
    bars["entity"] = bars.key.map(lambda k: canonical[signature[k]])
    bars = bars.loc[bars.key == bars.entity]
    bars = bars.merge(raw, on=["key", "date"], how="left").drop(columns=["key"])

    rows = []
    for d in ends:
        for r in runs.loc[(runs["first"] <= d) & (runs["last"] >= d)].itertuples():
            sp_ndx = r.source == "sp500_ndx"
            rows.append(dict(date=d, ticker=r.ticker, entity=r.entity, cik=r.cik,
                             in_sp500=sp_ndx and r.ticker in sp_monthly[d],
                             in_ndx=sp_ndx and r.ticker in ndx_monthly[d],
                             in_r1000=not sp_ndx))
    members = pd.DataFrame(rows)
    priced = members.dropna(subset=["entity"])
    # One row per entity and month; flags combine when two tickers map to one company.
    priced = priced.groupby(["date", "entity"], as_index=False).agg(
        ticker=("ticker", "first"), cik=("cik", "first"), in_sp500=("in_sp500", "max"),
        in_ndx=("in_ndx", "max"), in_r1000=("in_r1000", "max"))

    out.mkdir(parents=True)
    bars.to_parquet(out / "bars.parquet", index=False)
    members.to_parquet(out / "membership-raw.parquet", index=False)
    priced.to_parquet(out / "membership.parquet", index=False)
    pd.Series(sessions, name="date").to_frame().to_parquet(out / "calendar.parquet", index=False)
    monthly = members.groupby("date").agg(
        members=("ticker", "size"), sp500=("in_sp500", "sum"), ndx=("in_ndx", "sum"),
        r1000=("in_r1000", "sum"),
        without_bars=("entity", lambda s: int(s.isna().sum()))).reset_index()
    priced_counts = priced.groupby("date").size().rename("priced_entities").reset_index()
    monthly = monthly.merge(priced_counts, on="date", how="left")
    monthly.to_csv(out / "coverage.csv", index=False)
    unpriced = runs.loc[runs.entity.isna(), ["ticker", "first", "last", "source", "cik"]]
    unpriced.to_csv(out / "unpriced-runs.csv", index=False)
    write_json(out / "manifest.json", {
        "version": cfg["version"], "created_at": utc_now(), "config": cfg,
        "sources": {"sp500": f"github.com/{cfg['sp500_commit']}",
                    "ndx_current_revision": cfg["ndx_current_revision"],
                    "ndx_history_revision": cfg["ndx_history_revision"],
                    "alpaca": "v2/stocks/bars daily, adjustment=all plus raw close, feed=sip",
                    "russell_manifest_sha256": file_hash(BASE / cfg["russell"]["output"] / "manifest.json"),
                    "ticker_history_sha256": file_hash(BASE / cfg["sec"]["ticker_history"])},
        "resolution": bars.groupby("resolution").entity.nunique().to_dict(),
        "counts": {"sessions": len(sessions), "month_ends": len(ends), "runs": len(runs),
                   "unpriced_runs": int(runs.entity.isna().sum()),
                   "entities": int(bars.entity.nunique()), "bar_rows": len(bars)},
        "files": {p.name: file_hash(p) for p in out.iterdir() if p.is_file()},
        "elapsed_seconds": round(time.time() - started, 1),
    })
    print(json.dumps({"runs": len(runs), "unpriced": int(runs.entity.isna().sum()),
                      "entities": int(bars.entity.nunique()), "bars": len(bars)}), flush=True)


if __name__ == "__main__":
    main()
