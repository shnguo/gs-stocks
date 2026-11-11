"""US equity sources for factor research: index membership and Alpaca daily bars.

All requests go through a configurable proxy and a write-once disk cache. Credentials
travel only in request headers and never enter cached metadata or outputs.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ALPACA_BARS = "https://data.alpaca.markets/v2/stocks/bars"
VALID_SYMBOL = re.compile(r"^[A-Z][A-Z0-9]*(\.[A-Z0-9]+)?$")
WIKI_API = "https://en.wikipedia.org/w/api.php"
SP500_REPO = "fja05680/sp500"
SP500_FILE = "S&P 500 Historical Components & Changes (Updated).csv"


class Http:
    """Cached GET client. The cache key is the URL; headers (credentials) are excluded."""

    def __init__(self, cache: Path, proxy: str | None, user_agent: str, min_interval: float = 0.0):
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        if proxy:
            self.session.proxies = {"http": proxy, "https": proxy}
        self.user_agent = user_agent
        self.min_interval = min_interval
        self.last = 0.0
        self.fetched = 0

    def get(self, url: str, headers: dict | None = None, refresh: bool = False, store: bool = True) -> bytes:
        key = hashlib.sha256(url.encode()).hexdigest()
        path = self.cache / f"{key}.bin"
        if path.exists() and not refresh:
            return path.read_bytes()
        merged = {"User-Agent": self.user_agent, **(headers or {})}
        for attempt in range(5):
            time.sleep(max(0.0, self.min_interval - (time.monotonic() - self.last)))
            self.last = time.monotonic()
            try:
                response = self.session.get(url, headers=merged, timeout=90)
            except requests.RequestException:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
                continue
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(2 ** (attempt + 1))
                continue
            response.raise_for_status()
            body = response.content
            self.fetched += 1
            if not store:
                return body
            path.write_bytes(body)
            path.with_suffix(".json").write_text(json.dumps({
                "url": url, "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}, indent=2))
            return body
        raise RuntimeError(f"Retries exhausted: {url}")

    def json(self, url: str, headers: dict | None = None, refresh: bool = False, store: bool = True):
        return json.loads(self.get(url, headers, refresh, store))


def alpaca_credentials(env_file: Path) -> dict:
    values = {}
    for line in Path(env_file).read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            if key.strip() in ("ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY"):
                values[key.strip()] = value.strip().strip("\"'")
    if len(values) != 2:
        raise RuntimeError("Alpaca credentials file lacks ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY")
    return {"APCA-API-KEY-ID": values["ALPACA_API_KEY_ID"],
            "APCA-API-SECRET-KEY": values["ALPACA_API_SECRET_KEY"]}


# ---------- membership ----------

def sp500_membership(http: Http, commit: str) -> pd.DataFrame:
    """Point-in-time S&P 500 lists (date, tickers) from a pinned dataset commit."""
    url = (f"https://raw.githubusercontent.com/{SP500_REPO}/{commit}/"
           + urllib.parse.quote(SP500_FILE))
    frame = pd.read_csv(io.BytesIO(http.get(url)))
    if list(frame.columns) != ["date", "tickers"]:
        raise ValueError(f"Unexpected S&P 500 columns: {list(frame.columns)}")
    frame["date"] = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d")
    frame["tickers"] = frame.tickers.str.split(",").map(lambda xs: sorted({x.strip() for x in xs if x.strip()}))
    if frame.date.duplicated().any():
        raise ValueError("Duplicate S&P 500 dates")
    return frame.sort_values("date").reset_index(drop=True)


def wiki_tables(http: Http, page: str, revision: int) -> tuple[list[pd.DataFrame], int]:
    url = WIKI_API + "?" + urllib.parse.urlencode(
        {"action": "parse", "oldid": revision, "prop": "text|revid", "format": "json",
         "formatversion": 2})
    data = http.json(url)["parse"]
    if data["revid"] != revision or data["title"].replace(" ", "_") != page:
        raise ValueError(f"Unexpected Wikipedia revision for {page}")
    return pd.read_html(io.StringIO(data["text"])), revision


def wiki_latest_revision(http: Http, page: str) -> int:
    url = WIKI_API + "?" + urllib.parse.urlencode(
        {"action": "query", "prop": "revisions", "titles": page, "rvprop": "ids|timestamp",
         "format": "json", "formatversion": 2})
    return int(http.json(url, refresh=True)["query"]["pages"][0]["revisions"][0]["revid"])


def _clean_ticker(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).split("[")[0].strip().upper()
    return text or None


def ndx_current(tables: list[pd.DataFrame]) -> set[str]:
    for table in tables:
        columns = [str(c[-1] if isinstance(c, tuple) else c) for c in table.columns]
        if "Ticker" in columns and 90 <= len(table) <= 110:
            tickers = {_clean_ticker(v) for v in table.iloc[:, columns.index("Ticker")]}
            return {t for t in tickers if t}
    raise ValueError("Current Nasdaq-100 component table not found")


def ndx_changes(tables: list[pd.DataFrame]) -> pd.DataFrame:
    table = tables[0]
    expected = [("Date", "Date"), ("Added", "Ticker"), ("Added", "Security"),
                ("Removed", "Ticker"), ("Removed", "Security"), ("Reason", "Reason")]
    if list(table.columns) != expected:
        raise ValueError("Unexpected Nasdaq-100 change table")
    out = pd.DataFrame({
        "date": pd.to_datetime(table[("Date", "Date")].astype(str).str.split("[").str[0].str.strip(),
                               format="%B %d, %Y").dt.strftime("%Y-%m-%d"),
        "added": table[("Added", "Ticker")].map(_clean_ticker),
        "removed": table[("Removed", "Ticker")].map(_clean_ticker),
    })
    return out.sort_values("date", kind="stable").reset_index(drop=True)


def ndx_membership(current: set[str], changes: pd.DataFrame, as_of: str,
                   dates: list[str]) -> dict[str, set[str]]:
    """Walk changes backwards from the current list. Membership on date d includes
    every change dated on or before d (changes take effect at their listed date)."""
    state = set(current)
    pending = changes.loc[changes.date <= as_of].sort_values("date", ascending=False)
    result, rows = {}, list(pending.itertuples())
    k = 0
    for date in sorted(dates, reverse=True):
        while k < len(rows) and rows[k].date > date:
            if rows[k].added:
                state.discard(rows[k].added)
            if rows[k].removed:
                state.add(rows[k].removed)
            k += 1
        result[date] = set(state)
    return result


def membership_runs(monthly: dict[str, set[str]]) -> pd.DataFrame:
    """Contiguous (ticker, first, last) runs; the last month resolves the Alpaca symbol."""
    dates = sorted(monthly)
    runs, open_runs = [], {}
    for date in dates:
        for ticker in monthly[date]:
            if ticker in open_runs:
                open_runs[ticker][2] = date
            else:
                open_runs[ticker] = [ticker, date, date]
        for ticker in [t for t in open_runs if t not in monthly[date]]:
            runs.append(open_runs.pop(ticker))
    runs += list(open_runs.values())
    return pd.DataFrame(runs, columns=["ticker", "first", "last"]).sort_values(["ticker", "first"])


# ---------- prices ----------

def alpaca_assets(http: Http, credentials: dict) -> pd.DataFrame:
    """Alpaca's active and inactive US equity list (symbol, name, exchange, status)."""
    rows = []
    for status in ["active", "inactive"]:
        url = f"https://paper-api.alpaca.markets/v2/assets?status={status}&asset_class=us_equity"
        rows += http.json(url, credentials)
    return pd.DataFrame(rows)[["symbol", "name", "exchange", "status"]]


def alpaca_symbol(ticker: str) -> str:
    """Index lists write class shares as BRK-B / BF.B; Alpaca uses BRK.B."""
    return ticker.replace("-", ".").replace("/", ".")


def alpaca_bars(http: Http, credentials: dict, symbols: list[str], asof: str | None,
                start: str, end: str, chunk: int = 50, adjustment: str = "all",
                retry_missing: bool = True) -> pd.DataFrame:
    """Daily split- and dividend-adjusted SIP bars. With asof, each symbol is resolved as of
    that date and its full history (across later renames) is returned under the request key."""
    def fetch(batch: list[str], resolve: str | None, label: str) -> list[tuple]:
        params = {"symbols": ",".join(batch), "timeframe": "1Day",
                  "start": start + "T00:00:00Z", "end": end + "T23:59:59Z",
                  "adjustment": adjustment, "feed": "sip", "limit": 10000, "sort": "asc"}
        if resolve:
            params["asof"] = resolve
        out, tokens = [], set()
        while True:
            data = http.json(ALPACA_BARS + "?" + urllib.parse.urlencode(params), credentials)
            for symbol, bars in (data.get("bars") or {}).items():
                out += [(symbol, b["t"][:10], b["o"], b["h"], b["l"], b["c"], b["v"],
                         b.get("vw"), b.get("n"), label) for b in bars]
            token = data.get("next_page_token")
            if not token:
                return out
            if token in tokens:
                raise ValueError("Repeated Alpaca page token")
            tokens.add(token)
            params["page_token"] = token

    # Alpaca rejects a whole request if any symbol has characters like ^ or /.
    symbols = [x for x in symbols if VALID_SYMBOL.match(x)]
    rows = []
    for i in range(0, len(symbols), chunk):
        batch = symbols[i:i + chunk]
        found = fetch(batch, asof, "batch")
        rows += found
        # A multi-symbol response returns one key when two symbols resolve to the same
        # history (e.g. DOW and DD); retry those alone, then with today's mapping.
        missing = sorted(set(batch) - {r[0] for r in found}) if retry_missing else []
        for symbol in missing:
            single = fetch([symbol], asof, "single")
            if not single and asof:
                single = fetch([symbol], None, "today_mapping")
            rows += single
    frame = pd.DataFrame(rows, columns=["symbol", "date", "open", "high", "low", "close",
                                        "volume", "vwap", "trades", "resolution"])
    if frame.duplicated(["symbol", "date"]).any():
        raise ValueError("Duplicate Alpaca bars")
    return frame


def gap_segments(dates: pd.Series, calendar: list[str], max_gap_sessions: int = 20) -> pd.Series:
    """Segment number per bar; a new segment starts after more than max_gap_sessions
    missing sessions, so a spliced symbol history never forms one return series."""
    position = pd.Index(calendar).get_indexer(dates)
    if (position < 0).any():
        raise ValueError("Bar date outside the calendar")
    order = pd.Series(position, index=dates.index).sort_values()
    breaks = order.diff().fillna(1).gt(max_gap_sessions + 1).cumsum()
    return breaks.reindex(dates.index)
