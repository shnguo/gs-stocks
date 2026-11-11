"""Read-only, cached source collection. Authentication never enters provenance files."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# SEC inline-XBRL filings are HTML with XML declarations. Preserve HTML parsing.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


class UnclassifiedExhibit(ValueError):
    """A document was retrieved but was not classified as an earnings release."""


class SourceCache:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.last_sec = 0.0

    def get(self, url: str, headers: dict | None = None) -> bytes:
        key = hashlib.sha256(url.encode()).hexdigest()
        path = self.root / (key + ".bin")
        if path.exists():
            return path.read_bytes()
        safe_headers = {"User-Agent": os.getenv(
            "SEC_USER_AGENT", "us-earnings-research/0.1 personal financial research"
        )}
        safe_headers.update(headers or {})
        for attempt in range(3):
            if "sec.gov/" in url:
                time.sleep(max(0.0, 0.35 - (time.monotonic() - self.last_sec)))
                self.last_sec = time.monotonic()
            try:
                req = urllib.request.Request(url, headers=safe_headers)
                with urllib.request.urlopen(req, timeout=30) as response:
                    body = response.read()
                path.write_bytes(body)
                metadata = {
                    "url": url, "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                    "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body),
                    "cache_file": path.name,
                }
                path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
                return body
            except urllib.error.HTTPError as error:
                if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                    raise
                time.sleep(2 ** (attempt + 1))
        raise RuntimeError("Source retries exhausted")

    def json(self, url: str, headers: dict | None = None):
        return json.loads(self.get(url, headers))


def load_alpaca_credentials(env_file: Path | None = None) -> dict:
    values = dict(os.environ)
    if env_file is not None:
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                if key.strip() in ("ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY"):
                    values.setdefault(key.strip(), value.strip().strip("\"").strip("'"))
    try:
        return {"APCA-API-KEY-ID": values["ALPACA_API_KEY_ID"],
                "APCA-API-SECRET-KEY": values["ALPACA_API_SECRET_KEY"]}
    except KeyError as error:
        raise RuntimeError("Set Alpaca market-data environment variables or --credentials-file") from error


def collect_prices(config: dict, cache: SourceCache, credentials: dict, output: Path) -> pd.DataFrame:
    symbols = sorted({c["ticker"] for c in config["companies"]} | {config["benchmark"]})
    params = {"symbols": ",".join(symbols), "timeframe": "1Day",
              "start": config["price_start"] + "T00:00:00Z",
              "end": config["end"] + "T23:59:59Z", "adjustment": "all",
              "feed": "sip", "limit": 10000, "sort": "asc", "asof": config["end"]}
    rows = []
    page_tokens = set()
    while True:
        url = "https://data.alpaca.markets/v2/stocks/bars?" + urllib.parse.urlencode(params)
        data = cache.json(url, credentials)
        for ticker, bars in data.get("bars", {}).items():
            for bar in bars:
                rows.append({"ticker": ticker, "date": bar["t"][:10],
                             **{name: bar[key] for name, key in
                                [("open", "o"), ("high", "h"), ("low", "l"),
                                 ("close", "c"), ("volume", "v")]}})
        token = data.get("next_page_token")
        print(f"Price page: {len(rows)} cumulative rows", flush=True)
        if not token:
            break
        if token in page_tokens:
            raise ValueError("Repeated Alpaca pagination token")
        page_tokens.add(token)
        params["page_token"] = token
    prices = pd.DataFrame(rows).sort_values(["ticker", "date"])
    if set(prices.ticker) != set(symbols):
        raise ValueError(f"Missing requested symbols: {set(symbols) - set(prices.ticker)}")
    if prices.duplicated(["ticker", "date"]).any():
        raise ValueError("Duplicate price keys")
    output.mkdir(parents=True, exist_ok=True)
    prices.to_csv(output / "prices.csv", index=False)
    return prices


def filing_catalog(company: dict, config: dict, cache: SourceCache) -> pd.DataFrame:
    prefix = "https://data.sec.gov/submissions/"
    data = cache.json(prefix + f"CIK{company['cik']:010d}.json")
    chunks = [pd.DataFrame(data["filings"]["recent"])]
    for older in data["filings"].get("files", []):
        if older["filingTo"] >= config["start"] and older["filingFrom"] <= config["end"]:
            chunks.append(pd.DataFrame(cache.json(prefix + older["name"])))
    catalog = pd.concat([chunk.dropna(axis=1, how="all") for chunk in chunks if not chunk.empty],
                        ignore_index=True).drop_duplicates("accessionNumber")
    selected = catalog.loc[(catalog.form == "8-K") &
                           catalog["items"].str.contains(r"(?:^|,)2\.02(?:,|$)", na=False) &
                           catalog.filingDate.between(config["start"], config["end"])].copy()
    selected["ticker"] = company["ticker"]
    selected["cik"] = company["cik"]
    return selected.sort_values("filingDate")


def html_text(body: bytes) -> str:
    soup = BeautifulSoup(body, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return " ".join(soup.get_text(" ", strip=True).split())


def earnings_exhibit(filing: dict, cache: SourceCache) -> tuple[str, str, str]:
    accession = filing["accessionNumber"].replace("-", "")
    base = f"https://www.sec.gov/Archives/edgar/data/{filing['cik']}/{accession}/"
    primary = cache.get(base + filing["primaryDocument"])
    soup = BeautifulSoup(primary, "lxml")
    candidates = []
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        label = anchor.get_text(" ", strip=True).lower()
        filename = urllib.parse.urlparse(href).path.split("/")[-1].lower()
        if not filename.endswith((".htm", ".html")) or filename == filing["primaryDocument"].lower():
            continue
        score = 0
        if "99" in label or "99" in filename:
            score += 3
        if "99.1" in label or "99d1" in filename or "99_1" in filename or "991" in filename:
            score += 2
        if "earnings" in label or "press release" in label:
            score += 4
        if score:
            url = urllib.parse.urldefrag(urllib.parse.urljoin(base, href))[0]
            if urllib.parse.urlparse(url).hostname == "www.sec.gov":
                candidates.append((score, url))
    candidates = sorted(set(candidates), reverse=True)
    errors = []
    for _, url in candidates[:3]:
        try:
            body = cache.get(url)
            text = html_text(body)
            lower = text.lower()
            if (re.search(r"quarter.{0,100}(?:results|earnings)|(?:results|earnings).{0,100}quarter|"
                          r"quarter ended|fiscal year ended", lower)
                    and any(s in lower for s in ["revenue", "sales", "net income", "net earnings",
                                                 "earnings per share"])
                    and len(text) > 1500):
                return text, url, hashlib.sha256(body).hexdigest()
            errors.append("exhibit_not_classified_as_earnings:" + url)
        except (urllib.error.URLError, TimeoutError) as error:
            errors.append(type(error).__name__ + ":" + str(error))
    message = "No validated earnings exhibit; " + "; ".join(errors)[:500]
    if errors and all(error.startswith("exhibit_not_classified_as_earnings:") for error in errors):
        raise UnclassifiedExhibit(message)
    raise ValueError(message)


def collect_filings(config: dict, cache: SourceCache, output: Path) -> pd.DataFrame:
    output.mkdir(parents=True, exist_ok=True)
    text_dir = output / "texts"
    text_dir.mkdir(exist_ok=True)
    rows = []
    for company in config["companies"]:
        catalog = filing_catalog(company, config, cache)
        print(f"{company['ticker']}: {len(catalog)} Item 2.02 filings", flush=True)
        for filing in catalog.to_dict("records"):
            row = {k: filing[k] for k in ["ticker", "cik", "accessionNumber", "filingDate",
                                         "acceptanceDateTime", "reportDate", "primaryDocument"]}
            text_path = text_dir / (filing["accessionNumber"] + ".txt")
            meta_path = text_path.with_suffix(".json")
            try:
                if text_path.exists() and meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    text = text_path.read_text()
                    url, digest = meta["source_url"], meta["source_sha256"]
                else:
                    text, url, digest = earnings_exhibit(filing, cache)
                    text_path.write_text(text)
                    meta_path.write_text(json.dumps({"source_url": url, "source_sha256": digest}))
                row.update(status="ok", source_url=url, source_sha256=digest,
                           text_file=str(text_path.relative_to(output)), text_chars=len(text), error="")
            except (urllib.error.URLError, TimeoutError, ValueError) as error:
                status = "unclassified_exhibit" if isinstance(error, UnclassifiedExhibit) else "missing"
                row.update(status=status, source_url="", source_sha256="", text_file="",
                           text_chars=0, error=f"{type(error).__name__}: {error}"[:600])
            rows.append(row)
        pd.DataFrame(rows).to_csv(output / "filings.csv", index=False)
        print(f"  retrieved {sum(r['status'] == 'ok' for r in rows if r['ticker'] == company['ticker'])}",
              flush=True)
    return pd.DataFrame(rows)
