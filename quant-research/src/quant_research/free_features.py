"""Research-only daily feature alignment and independent provider replay.

Provider values are retrospective vintages. No publication-time certification,
current-master backfill, cross-date fill, or synthetic replacement of missing rows.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .storage import file_hash

RAW_FIELDS = dict(
    close="CLOSE_PRICE",
    total_cap="TOTAL_MARKET_CAP",
    float_cap="NOTLIMITED_MARKETCAP_A",
    total_shares="TOTAL_SHARES",
    float_shares="FREE_SHARES_A",
    pe_ttm="PE_TTM",
    pb="PB_MRQ",
    ps_ttm="PS_TTM",
    pcf_ttm="PCF_OCF_TTM",
)
FEATURES = [
    "turnover_pct",
    "log_total_cap",
    "log_float_cap",
    "log_total_shares",
    "log_float_shares",
    "float_share_ratio",
    "earnings_yield",
    "book_to_price",
    "sales_yield",
    "cashflow_yield",
    "source_available",
]


def replay(directory):
    directory = Path(directory)
    meta = json.loads((directory / "manifest.json").read_text())
    if meta["status"] != "completed" or meta["provider"] != "eastmoney":
        raise ValueError("Incomplete daily feature capture")
    day = meta["date"]
    if day >= "2025-08-07":
        raise ValueError("Sealed research date")
    for name, expected in meta["files"].items():
        if Path(name).name != name or file_hash(directory / name) != expected:
            raise ValueError("Changed or unsafe capture evidence")
    query = json.loads((directory / "query.json").read_text())
    if query["date"] != day or query["api"] != "RPT_VALUEANALYSIS_DET":
        raise ValueError("Changed query")
    rows, totals, page = [], None, 1
    while totals is None or page <= totals[0]:
        selected = json.loads((directory / f"page-{page:02d}.selected.json").read_text())["stem"]
        if selected not in [f"page-{page:02d}-attempt-{a}" for a in [1, 2, 3]]:
            raise ValueError("Unsafe selected page")
        raw = json.loads((directory / f"{selected}.body").read_text())
        receipt = json.loads((directory / f"{selected}.http.json").read_text())
        if receipt["status"] != 200 or not raw["success"]:
            raise ValueError("Failed provider page")
        result = raw["result"]
        current = (result["pages"], result["count"])
        if not 0 < current[0] <= 20 or current[0] != (current[1] + 4999) // 5000:
            raise ValueError("Invalid pagination")
        if totals is not None and totals != current:
            raise ValueError("Changed pagination")
        totals = current
        expected = 5000 if page < totals[0] else totals[1] - 5000 * (totals[0] - 1)
        if len(result["data"]) != expected:
            raise ValueError("Truncated page")
        for source in result["data"]:
            code, exchange = source["SECUCODE"].split(".")
            if code != source["SECURITY_CODE"] or source["TRADE_DATE"] != day + " 00:00:00":
                raise ValueError("Wrong security or date")
            record = dict(
                instrument_id="cn."
                + {"SH": "xshg", "SZ": "xshe", "BJ": "xbse"}[exchange]
                + "."
                + code,
                date=day,
            )
            for key, field in RAW_FIELDS.items():
                value = source[field]
                value = None if value in [None, "", "-"] else float(value)
                if value is not None and not np.isfinite(value):
                    raise ValueError("Nonfinite source value")
                if (
                    key in ["close", "total_cap", "float_cap", "total_shares", "float_shares"]
                    and value is not None
                    and value <= 0
                ):
                    value = None
                record[key] = value
            rows.append(record)
        page += 1
    stored = json.loads((directory / "records.json").read_text())
    if rows != stored or len(rows) != totals[1] or len(rows) != meta["rows"]:
        raise ValueError("Independent replay differs from Rust")
    frame = pd.DataFrame(rows)
    if frame.duplicated(["instrument_id", "date"]).any():
        raise ValueError("Duplicate source key")
    return frame


def align(rows, bars, source):
    """One exact signal-date join; retain every original cohort row and order."""
    keys = ["instrument_id", "date"]
    if rows.duplicated(keys).any() or bars.duplicated(keys).any() or source.duplicated(keys).any():
        raise ValueError("Duplicate alignment key")
    if (rows.date >= "2025-08-07").any():
        raise ValueError("Sealed research date")
    requested = pd.MultiIndex.from_frame(rows[keys])
    b = bars.set_index(keys).reindex(requested)
    s = source.set_index(keys).reindex(requested)
    close = pd.to_numeric(s.close, errors="raise").to_numpy(float)
    reference = b.close.to_numpy(float)
    available = (
        np.isfinite(close) & np.isfinite(reference) & (np.abs(close - reference) <= 0.010001)
    )
    present = np.isfinite(close)
    data = {}
    for k in RAW_FIELDS:
        value = pd.to_numeric(s[k], errors="raise").to_numpy(float)
        if np.isinf(value).any():
            raise ValueError("Infinite feature")
        data[k] = np.where(available, value, np.nan)
    out = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        # All share counts and original snapshot volume use shares, not lots.
        out["turnover_pct"] = b.volume.to_numpy(float) / data["float_shares"] * 100
        for field in ["total_cap", "float_cap", "total_shares", "float_shares"]:
            out["log_" + field] = np.log(np.where(data[field] > 0, data[field], np.nan))
        out["float_share_ratio"] = data["float_shares"] / data["total_shares"]
        for feature, field in [
            ("earnings_yield", "pe_ttm"),
            ("book_to_price", "pb"),
            ("sales_yield", "ps_ttm"),
            ("cashflow_yield", "pcf_ttm"),
        ]:
            out[feature] = np.where(np.abs(data[field]) > 1e-8, 1 / data[field], np.nan)
    out["source_available"] = available.astype(float)
    context = np.column_stack([out[k] for k in FEATURES]).astype(np.float32)
    if np.isinf(context).any():
        raise ValueError("Infinite transformed feature")
    reason = np.where(
        ~present, "source_missing", np.where(available, "available", "price_mismatch")
    )
    coverage = dict(
        rows=len(rows),
        source_available=int(available.sum()),
        source_missing=int((~present).sum()),
        price_mismatch=int((present & ~available).sum()),
        missing={k: int(np.isnan(context[:, i]).sum()) for i, k in enumerate(FEATURES)},
    )
    return context, coverage, reason
