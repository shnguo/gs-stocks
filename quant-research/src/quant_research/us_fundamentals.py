"""SEC-based identity and fundamentals for US factor research.

Ticker <-> CIK history comes from archived snapshots of SEC's company_tickers.json.
Fundamentals come from the SEC companyfacts bulk archive. Every value is used only
after its filing date, so factors reflect what was public at the time.
"""
from __future__ import annotations

import json
import urllib.parse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .us_sources import Http

TICKER_FILE = "sec.gov/files/company_tickers.json"
DOMESTIC_FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A", "10-KT", "10-QT", "10-KT/A", "10-QT/A"}
CONCEPTS = {
    ("dei", "EntityCommonStockSharesOutstanding"): "shares",
    ("us-gaap", "CommonStockSharesOutstanding"): "shares_balance",
    ("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic"): "shares_basic",
    ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding"): "shares_diluted",
    ("us-gaap", "NetIncomeLoss"): "net_income",
    ("us-gaap", "StockholdersEquity"): "equity",
    ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"): "equity_total",
    ("us-gaap", "Revenues"): "revenue",
    ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"): "revenue_contract",
    ("us-gaap", "RevenueFromContractWithCustomerIncludingAssessedTax"): "revenue_contract_tax",
    ("us-gaap", "SalesRevenueNet"): "revenue_sales",
    ("us-gaap", "NetCashProvidedByUsedInOperatingActivities"): "operating_cash",
    ("us-gaap", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"): "operating_cash_cont",
}
# Extra balance-sheet and margin inputs for the turnaround checklist (Piotroski-style).
CHECKLIST_CONCEPTS = {
    ("us-gaap", "Assets"): "assets",
    ("us-gaap", "AssetsCurrent"): "assets_current",
    ("us-gaap", "LiabilitiesCurrent"): "liabilities_current",
    ("us-gaap", "LongTermDebtNoncurrent"): "lt_debt",
    ("us-gaap", "LongTermDebt"): "lt_debt_total",
    ("us-gaap", "GrossProfit"): "gross_profit",
    ("us-gaap", "CostOfRevenue"): "cost_of_revenue",
    ("us-gaap", "CostOfGoodsAndServicesSold"): "cost_of_goods",
}
REVENUE = ["revenue", "revenue_contract", "revenue_contract_tax", "revenue_sales"]
# Multi-class companies tag cover-page shares by class, which the bulk archive omits;
# fall back to balance-sheet, then weighted-average share counts.
SHARE_FIELDS = ["shares", "shares_balance", "shares_basic", "shares_diluted"]


def wayback_snapshots(http: Http, start: str, end: str) -> list[str]:
    """One successful snapshot timestamp per month of SEC's ticker file."""
    url = "https://web.archive.org/cdx/search/cdx?" + urllib.parse.urlencode({
        "url": TICKER_FILE, "output": "json", "fl": "timestamp", "filter": "statuscode:200",
        "collapse": "timestamp:6", "from": start.replace("-", ""), "to": end.replace("-", "")})
    rows = http.json(url)[1:]
    return sorted(r[0] for r in rows)


def ticker_snapshot(http: Http, timestamp: str) -> pd.DataFrame:
    url = f"https://web.archive.org/web/{timestamp}id_/https://www.{TICKER_FILE}"
    data = json.loads(http.get(url))
    frame = pd.DataFrame(list(data.values()))
    if not {"cik_str", "ticker", "title"} <= set(frame.columns):
        raise ValueError(f"Unexpected ticker snapshot {timestamp}")
    date = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}"
    return pd.DataFrame({"snapshot": date, "cik": frame.cik_str.astype(int),
                         "ticker": frame.ticker.str.upper().str.strip(), "title": frame.title})


def mapping_on(history: pd.DataFrame, date: str) -> pd.DataFrame:
    """Ticker map from the latest snapshot on or before date (earliest if none)."""
    snapshots = sorted(history.snapshot.unique())
    before = [s for s in snapshots if s <= date]
    chosen = before[-1] if before else snapshots[0]
    return history.loc[history.snapshot == chosen]


def parse_companyfacts(zip_path: Path, concepts: dict | None = None,
                       ciks: set[int] | None = None) -> pd.DataFrame:
    """Selected concepts from every company in the bulk archive (or only `ciks`),
    domestic forms only."""
    concepts = CONCEPTS if concepts is None else concepts
    rows = []
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            if not name.endswith(".json"):
                continue
            if ciks is not None and name[3:13].isdigit() and int(name[3:13]) not in ciks:
                continue
            data = json.loads(archive.read(name))
            cik = int(data.get("cik") or name[3:13])
            facts = data.get("facts", {})
            for (taxonomy, concept), field in concepts.items():
                units = facts.get(taxonomy, {}).get(concept, {}).get("units", {})
                for unit, items in units.items():
                    if unit not in ("USD", "shares"):
                        continue
                    for f in items:
                        if f.get("form") not in DOMESTIC_FORMS:
                            continue
                        rows.append((cik, field, f.get("start"), f["end"], float(f["val"]),
                                     f["filed"], f["form"], f.get("accn")))
    frame = pd.DataFrame(rows, columns=["cik", "field", "start", "end", "val", "filed", "form", "accn"])
    return frame.drop_duplicates()


# ---------- point-in-time values ----------

def shares_asof(facts: pd.DataFrame, date: str,
                max_age_days: int = 400) -> tuple[float, str] | tuple[None, None]:
    """Cover-page shares from the latest filing on or before date, summed across classes.
    Falls back through SHARE_FIELDS. Returns (shares, as-of date)."""
    for field in SHARE_FIELDS:
        rows = facts.loc[(facts.field == field) & (facts.filed <= date)]
        if rows.empty:
            continue
        latest = rows.loc[rows.filed == rows.filed.max()]
        latest = latest.loc[latest.end == latest.end.max()]
        if (pd.Timestamp(date) - pd.Timestamp(latest.end.iloc[0])).days > max_age_days:
            continue
        total = latest.drop_duplicates(["val", "accn"]).val.sum()
        if total > 0:
            return float(total), str(latest.end.iloc[0])
    return None, None


def instant_asof(facts: pd.DataFrame, fields: list[str], date: str, max_age_days: int = 550):
    for field in fields:
        rows = facts.loc[(facts.field == field) & (facts.filed <= date) & facts.start.isna()]
        if rows.empty:
            continue
        end = rows.end.max()
        if (pd.Timestamp(date) - pd.Timestamp(end)).days > max_age_days:
            continue
        pick = rows.loc[rows.end == end]
        return float(pick.loc[pick.filed == pick.filed.max()].val.iloc[0])
    return None


def ttm_asof(facts: pd.DataFrame, fields: list[str], date: str, max_age_days: int = 550):
    """Trailing twelve months from what was filed by date: latest annual value, rolled
    forward by the latest later year-to-date value minus the prior-year equivalent."""
    for field in fields:
        rows = facts.loc[(facts.field == field) & (facts.filed <= date) & facts.start.notna()]
        if rows.empty:
            continue
        rows = rows.assign(days=(pd.to_datetime(rows.end) - pd.to_datetime(rows.start)).dt.days)
        # Latest filed value per period (restatements replace earlier values).
        rows = rows.sort_values("filed").drop_duplicates(["start", "end"], keep="last")
        annual = rows.loc[rows.days.between(350, 380)]
        if annual.empty:
            continue
        a = annual.loc[annual.end == annual.end.max()].iloc[0]
        if (pd.Timestamp(date) - pd.Timestamp(a.end)).days > max_age_days:
            continue
        fiscal_start = (pd.Timestamp(a.end) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        later = rows.loc[(rows.end > a.end) & rows.days.between(80, 290) & (rows.start > fiscal_start)]
        if later.empty:
            return float(a.val)
        q = later.loc[later.end == later.end.max()]
        q = q.loc[q.days == q.days.max()].iloc[0]
        target_end = pd.Timestamp(q.end) - pd.DateOffset(years=1)
        prior = rows.loc[rows.days.between(q.days - 20, q.days + 20)]
        prior = prior.loc[(pd.to_datetime(prior.end) - target_end).abs().dt.days <= 20]
        if prior.empty:
            return float(a.val)
        return float(a.val + q.val - prior.iloc[0].val)
    return None


def fundamentals_asof(facts: pd.DataFrame, date: str) -> dict:
    shares, shares_date = shares_asof(facts, date)
    return {"shares": shares, "shares_date": shares_date,
            "net_income": ttm_asof(facts, ["net_income"], date),
            "equity": instant_asof(facts, ["equity", "equity_total"], date),
            "revenue": ttm_asof(facts, REVENUE, date),
            "operating_cash": ttm_asof(facts, ["operating_cash", "operating_cash_cont"], date)}


def value_yields(fund: dict, market_cap: float) -> dict:
    out = {}
    for name, field in [("earnings_yield", "net_income"), ("book_to_price", "equity"),
                        ("sales_yield", "revenue"), ("cashflow_yield", "operating_cash")]:
        value = fund.get(field)
        out[name] = value / market_cap if value is not None and market_cap and market_cap > 0 else np.nan
    return out


# ---------- industry ----------

# Fama-French 12 industries by 4-digit SIC, from Ken French's Siccodes12.txt.
# Codes outside every range fall into "Other".
FF12 = {
    "NoDur": [(100, 999), (2000, 2399), (2700, 2749), (2770, 2799), (3100, 3199), (3940, 3989)],
    "Durbl": [(2500, 2519), (2590, 2599), (3630, 3659), (3710, 3711), (3714, 3714), (3716, 3716),
              (3750, 3751), (3792, 3792), (3900, 3939), (3990, 3999)],
    "Manuf": [(2520, 2589), (2600, 2699), (2750, 2769), (3000, 3099), (3200, 3569), (3580, 3629),
              (3700, 3709), (3712, 3713), (3715, 3715), (3717, 3749), (3752, 3791), (3793, 3799),
              (3830, 3839), (3860, 3899)],
    "Enrgy": [(1200, 1399), (2900, 2999)],
    "Chems": [(2800, 2829), (2840, 2899)],
    "BusEq": [(3570, 3579), (3660, 3692), (3694, 3699), (3810, 3829), (7370, 7379)],
    "Telcm": [(4800, 4899)],
    "Utils": [(4900, 4949)],
    "Shops": [(5000, 5999), (7200, 7299), (7600, 7699)],
    "Hlth": [(2830, 2839), (3693, 3693), (3840, 3859), (8000, 8099)],
    "Money": [(6000, 6999)],
}
FF12_NAMES = {"NoDur": "消费品（非耐用）", "Durbl": "耐用消费品", "Manuf": "制造业", "Enrgy": "能源",
              "Chems": "化工", "BusEq": "科技设备与软件", "Telcm": "电信传媒", "Utils": "公用事业",
              "Shops": "零售批发", "Hlth": "医疗", "Money": "金融", "Other": "其他"}


def ff12(sic) -> str | None:
    """Fama-French 12 industry for a SIC code; None when the code is missing."""
    try:
        code = int(sic)
    except (TypeError, ValueError):
        return None
    for name, ranges in FF12.items():
        if any(lo <= code <= hi for lo, hi in ranges):
            return name
    return "Other"


def company_industry(http: Http, cik: int, user_agent: str) -> dict:
    """Current SIC code and description from SEC's submissions record (not point-in-time)."""
    url = f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json"
    data = http.json(url, {"User-Agent": user_agent}, store=False)
    sic = data.get("sic") or None
    return {"cik": int(cik), "sec_name": data.get("name"), "sic": int(sic) if sic else None,
            "sic_description": data.get("sicDescription") or None, "ff12": ff12(sic)}
