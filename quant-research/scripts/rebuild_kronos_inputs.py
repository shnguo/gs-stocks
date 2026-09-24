"""Rebuild a frozen Kronos OHLCVA matrix and require its historical SHA256."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.storage import file_hash, utc_now, write_json

BASE = Path(__file__).resolve().parents[1]


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def rebuild(training_root: Path, target_manifest_path: Path, output: Path) -> Path:
    training_root = training_root.resolve()
    target_manifest_path = target_manifest_path.resolve()
    output = output.resolve()
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    panel_manifest_path = training_root / "panel-h5/manifest.json"
    snapshot_manifest_path = training_root / "snapshot/manifest.json"
    bars_path = training_root / "snapshot/bars.parquet"
    mask_path = training_root / "panel-h5/values.npy"
    panel = read(panel_manifest_path)
    snapshot = read(snapshot_manifest_path)
    target = read(target_manifest_path)
    if not (panel["dataset_id"] == snapshot["dataset_id"] == target["dataset_id"]):
        raise ValueError("Dataset identity differs")
    if panel["instruments"] != target["instruments"]:
        raise ValueError("Instrument axes differ")
    if target["dates"] != panel["dates"][: len(target["dates"])]:
        raise ValueError("Target dates are not a prefix of the frozen panel")
    if file_hash(bars_path) != snapshot["files"]["bars.parquet"]:
        raise ValueError("Frozen bars hash differs")
    if file_hash(mask_path) != panel["files"]["values.npy"]:
        raise ValueError("Frozen suspension mask hash differs")

    dates = target["dates"]
    fields = target["fields"]
    required = ["open", "high", "low", "close", "volume", "amount", "factor"]
    if fields != required:
        raise ValueError(f"Unexpected frozen fields: {fields}")
    frame = pd.read_parquet(
        bars_path,
        columns=["instrument_id", "date", *fields, "sequence_id"],
        filters=[("date", "<=", dates[-1])],
    )
    groups = frame.groupby("instrument_id", sort=False).indices
    mask_values = np.load(mask_path, mmap_mode="r")
    mask_index = panel["feature_names"].index("unpriced_suspension")
    output.mkdir(parents=True)
    values = np.lib.format.open_memmap(
        output / "values.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(panel["instruments"]), len(dates), len(fields)),
    )
    values[:] = np.nan
    for index, symbol in enumerate(panel["instruments"]):
        raw = frame.iloc[groups.get(symbol, [])].set_index("date").reindex(dates)
        carry = (mask_values[index, : len(dates), mask_index] == 1) & raw.close.isna()
        sequence = raw.sequence_id.ffill().astype("string").fillna("")
        episodes = sequence.ne(sequence.shift()).cumsum()
        close = raw.close.groupby(episodes).ffill()
        factor = raw.factor.groupby(episodes).ffill()
        for field in fields[:4]:
            raw.loc[carry, field] = close.loc[carry]
        raw.loc[carry, "factor"] = factor.loc[carry]
        raw.loc[carry, ["volume", "amount"]] = 0.0
        values[index] = raw[fields].to_numpy(np.float32)
        if index % 500 == 0:
            write_json(
                output / "progress.json",
                {
                    "status": "rebuilding",
                    "completed": index,
                    "total": len(panel["instruments"]),
                    "updated_at": utc_now(),
                },
            )
    values.flush()
    del values
    actual = file_hash(output / "values.npy")
    if actual != target["values_sha256"]:
        raise ValueError(
            f"Rebuilt values differ from frozen target: expected {target['values_sha256']}, got {actual}"
        )
    shutil.copy2(target_manifest_path, output / "manifest.json")
    shutil.copy2(Path(__file__), output / Path(__file__).name)
    sources = {
        str(path): file_hash(path)
        for path in [
            panel_manifest_path,
            snapshot_manifest_path,
            bars_path,
            mask_path,
            target_manifest_path,
            Path(__file__),
        ]
    }
    write_json(
        output / "source.json",
        {
            "at": utc_now(),
            "method": "deterministic rebuild from the frozen bars and suspension mask",
            "sources": sources,
            "expected_values_sha256": target["values_sha256"],
            "actual_values_sha256": actual,
        },
    )
    write_json(
        output / "verification.json",
        {
            "passed": True,
            "values_sha256_matches_frozen_target": True,
            "instruments": len(panel["instruments"]),
            "dates": len(dates),
            "fields": fields,
        },
    )
    files = {
        path.name: file_hash(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name not in {"completed.json", "progress.json"}
    }
    write_json(
        output / "completed.json",
        {"at": utc_now(), "passed": True, "files": files},
    )
    write_json(
        output / "progress.json",
        {"status": "completed", "values_sha256": actual, "updated_at": utc_now()},
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-root",
        type=Path,
        default=BASE / "artifacts/full-market-training-20260910-v1",
    )
    parser.add_argument(
        "--target-manifest",
        type=Path,
        default=BASE / "artifacts/kronos-inputs-20260914-v3/manifest.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(rebuild(args.training_root, args.target_manifest, args.output))


if __name__ == "__main__":
    main()
