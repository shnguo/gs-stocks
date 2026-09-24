"""Full-market membership is independent of personal lists and successfully downloaded bars."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .coverage import load_coverage
from .storage import Snapshot, file_hash


def load_pool(path: str | Path) -> tuple[dict, dict]:
    path = Path(path)
    manifest = json.loads(path.read_text())
    if (manifest.get("scope") != "all_a_shares"
            or manifest.get("membership_source") not in {"ashare_universe_snapshots", "tushare_stock_basic"}
            or manifest.get("not_a_personal_watchlist") is not True):
        raise ValueError("Research membership must come from the full market catalog")
    required = {"universe.json", "selection-members.csv", "historical-candidates.csv",
                "acquisition-plan.json"}
    if set(manifest["files"]) != required:
        raise ValueError("Incomplete full-market catalog manifest")
    for name, expected in manifest["files"].items():
        if file_hash(path.parent / name) != expected:
            raise ValueError("Full-market catalog hash mismatch")
    pool = json.loads((path.parent / "universe.json").read_text())
    if (manifest["universe_id"] != file_hash(path.parent / "universe.json")
            or pool["scope"] != "all_a_shares"
            or pool["membership_source"] != manifest["membership_source"]):
        raise ValueError("Full-market catalog identity mismatch")
    selected = pool["selection_members"]
    if (len(selected) != pool["source_member_count"]
            or len(selected) != manifest["counts"]["selection_members"]
            or len({r["instrument_id"] for r in selected}) != len(selected)
            or {r["exchange"] for r in selected} != {"XSHG", "XSHE", "XBSE"}
            or not all(r["in_selection_pool"] for r in selected)):
        raise ValueError("Full-market catalog count, exchange, or member mismatch")
    return manifest, pool


def pool_coverage(snapshot: Snapshot, catalog_path: str | Path,
                  coverage_evidence: dict | None = None) -> dict:
    manifest, pool = load_pool(catalog_path)
    exceptions = load_coverage(snapshot, coverage_evidence)
    first = snapshot.dates[0] if snapshot.dates else ""
    last = snapshot.dates[-1] if snapshot.dates else ""
    actual = snapshot.tables["instruments"].copy()
    # Keep the full symbol, including any legacy prefix; remove only the listing-instance suffix.
    actual["catalog_instrument_id"] = actual.instrument_id.str.split("@", n=1).str[0]
    duplicate_identities = sorted(actual.loc[actual.catalog_instrument_id.duplicated(False),
                                             "catalog_instrument_id"].unique())
    mapping = actual.set_index("catalog_instrument_id").instrument_id.to_dict()
    bars = snapshot.tables["bars"]
    available = {key: set(group.date) for key, group in bars.groupby("instrument_id")}
    calendar = snapshot.tables["calendar"]
    open_dates = {e: sorted(group.loc[group.is_open.astype(bool), "date"])
                  for e, group in calendar.groupby("exchange")}
    rows, expected_ids = [], set()
    current_present = 0
    for member in pool["historical_candidates"]:
        key = member["instrument_id"]
        start = max(first, member["listed_at"])
        end = min(last, member["delisted_at"] or last)
        in_period = bool(first and start <= end)
        actual_id = mapping.get(key)
        if member["in_selection_pool"] and actual_id:
            current_present += 1
        if in_period:
            expected_ids.add(key)
        expected_dates = [date for date in open_dates.get(member["exchange"], [])
                          if start <= date <= end] if in_period else []
        absent_dates = [date for date in expected_dates
                        if date not in available.get(actual_id, set())]
        explained_days = sum((key, date) in exceptions for date in absent_dates)
        missing_days = len(absent_dates) - explained_days
        if not in_period:
            status = "outside_snapshot_period"
        elif actual_id is None:
            status = "missing_instrument"
        elif not expected_dates:
            status = "missing_exchange_calendar"
        elif not available.get(actual_id) and missing_days:
            status = "missing_bars"
        elif missing_days:
            status = "incomplete_calendar_day_coverage"
        else:
            status = "rows_present_identity_and_quality_still_require_validation"
        rows.append({"instrument_id": key, "provider_code": member["provider_code"],
                     "exchange": member["exchange"], "in_selection_pool": member["in_selection_pool"],
                     "required_in_snapshot_period": in_period, "status": status,
                     "expected_market_days": len(expected_dates), "missing_market_days": missing_days,
                     "observed_market_days": len(expected_dates) - len(absent_dates),
                     "verified_nonprice_or_inactive_days": explained_days,
                     "retained_in_expected_pool": True})
    missing = [r["instrument_id"] for r in rows if r["status"] == "missing_instrument"]
    gaps = [r["instrument_id"] for r in rows if r["required_in_snapshot_period"] and r["status"] !=
            "rows_present_identity_and_quality_still_require_validation"]
    known_ids = {r["instrument_id"] for r in pool["historical_candidates"]}
    unexpected = sorted(set(mapping) - known_ids)
    return {"universe_id": manifest["universe_id"], "scope": "all_a_shares",
            "membership_source": pool["membership_source"], "catalog_as_of": pool["as_of"],
            "selection_pool_members": len(pool["selection_members"]),
            "selection_members_in_snapshot": current_present,
            "expected_historical_members": len(expected_ids), "snapshot_instruments": len(actual),
            "missing_instruments": missing, "instruments_with_data_gaps": gaps,
            "unexpected_instruments": unexpected, "duplicate_listing_identities": duplicate_identities,
            "full_membership_present": not missing and not unexpected and not duplicate_identities,
            "full_calendar_day_coverage": bool(expected_ids) and not gaps and not unexpected and
                not duplicate_identities,
            "historical_identity_verified": pool["historical_identity_verified"],
            "data_ready": False, "coverage": rows}


def require_full_pool(snapshot: Snapshot, config: dict) -> dict:
    path = config.get("universe_catalog")
    if not path:
        raise ValueError("Full-market catalog is required; a supplied stock subset cannot define the pool")
    result = pool_coverage(snapshot, path, config.get("coverage_evidence"))
    if not result["full_calendar_day_coverage"]:
        raise ValueError(
            "Full-market coverage is incomplete: "
            f"expected {result['expected_historical_members']} historical members, "
            f"missing {len(result['missing_instruments'])}, "
            f"data gaps {len(result['instruments_with_data_gaps'])}. "
            "Run pool-audit for the complete gap ledger; small samples are test fixtures only.")
    return result


def coverage_frame(result: dict) -> pd.DataFrame:
    return pd.DataFrame(result["coverage"])
