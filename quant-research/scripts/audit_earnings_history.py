"""Read-only selected-source profiling and pre-holdout development calendar preparation."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from quant_research.config import digest
from quant_research.daily_loop import read, verify
from quant_research.earnings_walkforward import historical_development_folds
from quant_research.storage import TABLES, file_hash, utc_now

NUMERIC = ["open", "high", "low", "close", "volume", "amount", "factor"]
MAIN_BOARDS = {"shanghai_main", "shenzhen_main"}


def bound_snapshot(root):
    root = Path(root).resolve()
    manifest = verify(root / "snapshot")
    if set(manifest["files"]) != {name + ".parquet" for name in TABLES}:
        raise ValueError(f"Source snapshot must hash all {len(TABLES)} tables")
    if "schema_version" in manifest:
        if digest({k: manifest[k] for k in ["schema_version", "declaration", "files"]}) != manifest["dataset_id"]:
            raise ValueError("Canonical source identity mismatch")
    else:
        lineage = read(root / "lineage.json")
        if (digest(lineage) != manifest["dataset_id"]
                or file_hash(root / "code/daily_input.py") != lineage["implementation_sha256"]):
            raise ValueError("Refresh derivation identity mismatch")
    return manifest


def profile_prices(root, sealed_start, sealed_end, history, batch_size=65536):
    """Read only permitted price rows; hash checks may inspect all file bytes separately."""
    snapshot = Path(root) / "snapshot"
    meta = pd.read_parquet(snapshot / "instruments.parquet")
    if meta.instrument_id.duplicated().any():
        raise ValueError("Duplicate instrument identities")
    boards = meta.set_index("instrument_id").board.to_dict()
    instrument_index = {symbol: i for i, symbol in enumerate(meta.instrument_id)}
    dataset = ds.dataset(snapshot / "bars.parquet", format="parquet")
    columns = ["instrument_id", "date", *NUMERIC]
    columns += [name for name in ["factor_evidence", "sequence_id"] if name in dataset.schema.names]
    permitted = ds.field("date") < sealed_start if history else ((ds.field("date") < sealed_start) | (ds.field("date") > sealed_end))
    rows, missing_rows, invalid_rows = 0, 0, 0
    nonfinite = Counter({name: 0 for name in NUMERIC})
    date_counts, missing_dates, board_counts, missing_boards, reasons = Counter(), Counter(), Counter(), Counter(), Counter()
    affected, sequences, key_parts = set(), defaultdict(set), []
    for batch in dataset.scanner(columns=columns, filter=permitted, batch_size=batch_size).to_batches():
        frame = batch.to_pandas()
        if frame.empty:
            continue
        if not frame.instrument_id.isin(instrument_index).all():
            raise ValueError("Price row has unknown instrument identity")
        values = frame[NUMERIC].to_numpy(dtype=float)
        missing = ~np.isfinite(values)
        bad = missing.any(axis=1)
        nonfinite.update(dict(zip(NUMERIC, missing.sum(axis=0).tolist())))
        board = frame.instrument_id.map(boards)
        factor_missing = missing[:, NUMERIC.index("factor")]
        rows += len(frame)
        missing_rows += int(bad.sum())
        invalid_rows += int(((values[:, :4] <= 0).any(axis=1) | (values[:, 6] <= 0)
            | (values[:, 4:6] < 0).any(axis=1) | (values[:, 1] < np.maximum(values[:, 0], values[:, 3]))
            | (values[:, 2] > np.minimum(values[:, 0], values[:, 3])) | (values[:, 1] < values[:, 2])).sum())
        date_counts.update(frame.date.value_counts().to_dict())
        board_counts.update(board.value_counts().to_dict())
        missing_dates.update(frame.loc[factor_missing, "date"].value_counts().to_dict())
        missing_boards.update(board.loc[factor_missing].value_counts().to_dict())
        affected.update(frame.loc[factor_missing, "instrument_id"])
        if "factor_evidence" in frame:
            reasons.update(frame.loc[factor_missing, "factor_evidence"].fillna("no_reason_in_source").value_counts().to_dict())
        if "sequence_id" in frame:
            for symbol, group in frame.groupby("instrument_id", sort=False):
                sequences[symbol].update(group.sequence_id.dropna().astype(str).unique())
        ids = frame.instrument_id.map(instrument_index).to_numpy(np.uint64)
        ordinal = pd.to_datetime(frame.date, format="%Y-%m-%d", errors="raise").to_numpy("datetime64[D]").astype(np.int64).astype(np.uint64)
        key_parts.append((ids << np.uint64(32)) | ordinal)
    keys = np.concatenate(key_parts) if key_parts else np.array([], dtype=np.uint64)
    duplicate_excess = len(keys) - len(np.unique(keys))
    missing_factor = nonfinite["factor"]
    return dict(file_rows=pq.ParquetFile(snapshot / "bars.parquet").metadata.num_rows,
        inspected_rows=rows, inspected_first_date=min(date_counts, default=None), inspected_last_date=max(date_counts, default=None),
        observed_dates=sorted(date_counts), observed_sessions=len(date_counts),
        nonfinite_by_column=dict(nonfinite), rows_with_nonfinite_required_field=missing_rows,
        nonfinite_required_row_share=missing_rows / rows if rows else None,
        nonfinite_factor_rows=missing_factor, nonfinite_factor_share=missing_factor / rows if rows else None,
        duplicate_business_key_excess=duplicate_excess, invalid_price_volume_rows=invalid_rows,
        rows_by_board=dict(board_counts), missing_factor_rows_by_board=dict(missing_boards),
        missing_factor_rows_by_date=dict(sorted(missing_dates.items())), missing_factor_rows_by_reason=dict(reasons),
        missing_factor_instruments=len(affected), instruments_with_multiple_sequence_ids=sum(len(x) > 1 for x in sequences.values()),
        identified_main_board_rows=sum(board_counts[b] for b in MAIN_BOARDS),
        identified_main_board_missing_factor_rows=sum(missing_boards[b] for b in MAIN_BOARDS),
        sealed_price_rows_read=0, returns_or_targets_read=False, formal_profit_qualified=False)


def observed_calendar(calendar, profile):
    dates = sorted(calendar.loc[calendar.is_open.astype(bool), "date"].unique())
    actual = profile["observed_dates"]
    expected = [d for d in dates if profile["inspected_first_date"] <= d <= profile["inspected_last_date"]]
    if not actual or actual != expected:
        raise ValueError("Observed price dates must cover the full selected calendar; do not compress missing sessions")
    return expected


def audit(config, output):
    config, output = Path(config).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Historical source audit destination must be new")
    project = Path(__file__).resolve().parents[1]
    cfg = read(config)
    paths = {name: (project / cfg[name]).resolve() for name in ["protocol", "history", "history_alias", "refresh", "holdout_policy"]}
    if any(output.is_relative_to(paths[name]) for name in ["history", "history_alias", "refresh"]):
        raise ValueError("Audit output cannot modify its sources")
    protocol, policy = read(paths["protocol"]), read(paths["holdout_policy"])
    if protocol["validation"]["historical_walk_forward"] != dict(train_years=3, validation_years=1, test_months=3):
        raise ValueError("Historical implementation requires the original 3+1-year/quarter protocol")
    start, end = policy["sealed_holdout_start"], policy["sealed_holdout_end"]
    if start > end or policy["holdout_included"] is not False:
        raise ValueError("Existing sealed holdout boundaries must remain explicit")
    history, alias, refresh = [bound_snapshot(paths[name]) for name in ["history", "history_alias", "refresh"]]
    acceptance_path = paths["history"] / "development-acceptance.json"
    acceptance = read(acceptance_path)
    if (file_hash(paths["holdout_policy"]) != acceptance["evidence"]["schedule.json"]
            or file_hash(paths["history"] / "snapshot/manifest.json") != acceptance["evidence"]["snapshot/manifest.json"]):
        raise ValueError("Historical source selection or sealed policy differs from its bound evidence")
    output.mkdir(parents=True)
    receipt = dict(schema_version=1, started_at=utc_now(), status="profiling_selected_sources",
        sealed_holdout_start=start, sealed_holdout_end=end, formal_profit_qualified=False)
    try:
        long_profile = profile_prices(paths["history"], start, end, True)
        current_profile = profile_prices(paths["refresh"], start, end, False)
        calendar = pd.read_parquet(paths["history"] / "snapshot/calendar.parquet")
        dates = observed_calendar(calendar, long_profile)
        folds = historical_development_folds(dates, start)
        if any(d >= start for d in long_profile["observed_dates"]):
            raise ValueError("Sealed historical price rows were read")
        if any(start <= d <= end for d in current_profile["observed_dates"]):
            raise ValueError("Sealed refresh price rows were read")
        (output / "historical-profile.json").write_text(json.dumps(long_profile, indent=2, allow_nan=False) + "\n")
        (output / "refresh-profile.json").write_text(json.dumps(current_profile, indent=2, allow_nan=False) + "\n")
        (output / "development-folds.json").write_text(json.dumps(dict(folds=folds,
            formal_profit_qualified=False, historical_return_experiment_run=False,
            sample_endpoint_partitioning_applied=False,
            convention="20 full sessions strictly between actual fitted label end and the next partition; requires registration"), indent=2) + "\n")
        receipt.update(status="selected_history_and_refresh_profiled_no_return_experiment", completed_at=utc_now(),
            history_and_alias_identical_table_bytes=history["files"] == alias["files"],
            unique_long_history_table_sets=1 if history["files"] == alias["files"] else 2,
            historical_dataset_id=history["dataset_id"], refresh_dataset_id=refresh["dataset_id"],
            historical_declaration=history["declaration"], refresh_declaration=refresh["declaration"],
            historical_inspected_rows=long_profile["inspected_rows"], refresh_inspected_rows=current_profile["inspected_rows"],
            whole_development_quarters=len(folds), first_fold=folds[0]["fold_id"] if folds else None,
            last_fold=folds[-1]["fold_id"] if folds else None, sealed_price_rows_read=0, outcomes_read=False,
            historical_calendar_price_dates_exactly_reconciled=True,
            required_market_tables=len(TABLES),
            historical_return_experiment_run=False, formal_daily_runner_implemented=False,
            source_hashes={str(p): file_hash(p) for p in [config, paths["protocol"], paths["holdout_policy"], acceptance_path,
                Path(__file__), project / "src/quant_research/earnings_walkforward.py",
                *[paths[name] / "snapshot" / f for name in ["history", "history_alias", "refresh"]
                  for f in ["manifest.json", *history["files"]]]]},
            limits=["History and current refresh remain reconstructed training sources, not accepted account inputs.",
                "Metadata calendar preparation is not a fitted model, an event cohort, a historical account comparison or formal registration.",
                "Existing gross-holder label acceptance does not establish executable earnings account returns.",
                "The alias is counted once when table hashes agree; full source bytes are hashed but sealed price rows and outcomes are not queried."])
        receipt["output_hashes"] = {p.name: file_hash(p) for p in output.iterdir() if p.is_file()}
    except Exception as exc:
        receipt.update(status="failed_source_profile_attempt", error_type=type(exc).__name__, reason=str(exc))
        (output / "audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
        raise
    (output / "audit.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.config, args.output)
    print(json.dumps({"status": result["status"], "development_quarters": result["whole_development_quarters"],
        "sealed_price_rows_read": 0, "formal_profit_qualified": False}))


if __name__ == "__main__":
    main()
