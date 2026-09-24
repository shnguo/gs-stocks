"""Mature compact early-stage matching cohorts during the manual daily cycle."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.daily_loop import digest, read, verify, write_manifest
from quant_research.daily_token import panel
from quant_research.storage import file_hash, utc_now, write_json


def mature(cohort, bars, horizon_dates, cost):
    """Attach realized outcomes without changing the previously frozen cohort."""
    groups = {name: value.set_index("date") for name, value in bars.groupby("instrument_id")}
    records = []
    required = ["open", "high", "low", "close", "volume", "amount", "factor"]
    for row in cohort.itertuples(index=False):
        group = groups.get(row.instrument_id)
        if group is None:
            continue
        dates = [row.date, *horizon_dates]
        block = group.reindex(dates)
        values = block[required].to_numpy(float)
        known = (np.isfinite(values).all() and (values[:, 4:] > 0).all() and
                 (values[:, 1] >= values[:, :4].max(axis=1)).all() and
                 (values[:, 2] <= values[:, :4].min(axis=1)).all())
        if "source_trade_status" in block and block.source_trade_status.eq(0).any():
            known = False
        result = row._asdict()
        result["label_known"] = bool(known)
        for name in ["actual_execution_return", "actual_extrema_scenario",
                     "actual_ideal_execution_return", "actual_adverse"]:
            result[name] = np.nan
        if known:
            factor = values[:, 6]
            adjusted_open = values[:, 0]*factor
            adjusted_high = values[:, 1]*factor
            adjusted_low = values[:, 2]*factor
            entry = adjusted_open[1]
            sell = dates.index(row.sell_reference_date)
            result["actual_execution_return"] = adjusted_high[sell]/entry-1-cost
            result["actual_extrema_scenario"] = adjusted_high[sell]/adjusted_low[1]-1-cost
            result["actual_ideal_execution_return"] = adjusted_high[2:].max()/entry-1-cost
            result["actual_adverse"] = adjusted_low[1:sell+1].min()/entry-1
        records.append(result)
    return pd.DataFrame(records)


def review(store, source):
    store, source = Path(store), Path(source)
    meta, bars, _ = panel(source)
    observed = meta["price_data_through"]
    source_hash = file_hash(source/"snapshot/manifest.json")
    reports = []
    pointers = store/"early-stage-publications"
    for pointer_path in sorted(pointers.glob("*.json")) if pointers.exists() else []:
        pointer = read(pointer_path)
        publication = Path(pointer["report"])
        cohort_path = publication/"matching-cohort.csv"
        if not cohort_path.exists():
            reports.append({"signal_date": pointer["signal_date"],
                            "status": "legacy_without_matching_cohort"})
            continue
        verify(publication)
        delivery = read(publication/"delivery.json")
        horizon = delivery["horizon_dates"]
        if horizon[-1] > observed:
            reports.append({"signal_date": delivery["signal_date"], "status": "pending",
                            "mature_after": horizon[-1]})
            continue
        target = store/"early-stage-scorecards"/delivery["signal_date"]/digest(
            [source_hash, file_hash(publication/"manifest.json")])[:16]
        if (target/"manifest.json").exists():
            verify(target)
            reports.append(read(target/"scorecard.json"))
            continue
        cohort = pd.read_csv(cohort_path)
        frame = mature(cohort, bars, horizon, delivery["cost_scenario"])
        target.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target/"rows.csv", index=False)
        report = {
            "signal_date": delivery["signal_date"],
            "status": "mature",
            "observed_through": observed,
            "cohort_rows": len(cohort),
            "known_rows": int(frame.label_known.sum()) if len(frame) else 0,
            "unknown_rows": len(cohort)-int(frame.label_known.sum()) if len(frame) else len(cohort),
            "source_manifest_sha256": source_hash,
            "publication_manifest_sha256": file_hash(publication/"manifest.json"),
            "created_at": utc_now(),
        }
        write_json(target/"scorecard.json", report)
        write_manifest(target)
        reports.append(report)
    write_json(store/"early-stage-evaluation-latest.json", {
        "at": utc_now(),
        "observed_through": observed,
        "scorecards": reports,
        "policy": "Cohorts are frozen at publication; outcomes are attached only after T+4 matures.",
    })
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    source = args.source or Path(read(args.store/"current.json")["source"])
    print(review(args.store, source))
