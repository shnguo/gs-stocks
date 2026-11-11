"""Structural research adapter: preserve daily-refresh bytes and qualification limits."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from .config import digest
from .daily_loop import read, verify
from .storage import TABLES, Snapshot, file_hash, utc_now


def adapt_daily_training_input(source: Path, output: Path) -> dict:
    """Make daily-refresh tables loadable by Snapshot, without allowing account returns.

    Production provider parsing remains Rust-owned. This only copies already archived
    research Parquet tables and their declarations; it never invents values, risk
    coverage, board identity, entitlements, earlier clocks or accepted sources.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Adapter output must be new; failed attempts are not retried")
    if output.is_relative_to(source):
        raise ValueError("Adapter output cannot modify the source tree")
    original = verify(source / "snapshot")
    if original.get("schema_version") is not None:
        raise ValueError("Use the canonical Snapshot loader for an already canonical source")
    if set(original["files"]) != {name + ".parquet" for name in TABLES}:
        raise ValueError(f"Daily input must bind all {len(TABLES)} research tables")
    declaration = original["declaration"]
    if declaration.get("purpose") != "training":
        raise ValueError("This structural adapter accepts training-only daily refreshes")
    for key in ["st_history_verified", "actions_verified", "rules_verified",
                "historical_universe_verified", "industry_history_verified", "synthetic"]:
        if type(declaration.get(key)) is not bool:
            raise ValueError("Research source declarations require explicit booleans")
    lineage_path = source / "lineage.json"
    lineage = read(lineage_path)
    if digest(lineage) != original["dataset_id"]:
        raise ValueError("Daily dataset identity differs from original derivation")
    archived_code = source / "code/daily_input.py"
    if file_hash(archived_code) != lineage["implementation_sha256"]:
        raise ValueError("Archived daily-refresh implementation differs from lineage")
    observed = pd.Timestamp(original["created_at"])
    if pd.isna(observed) or observed.tzinfo is None:
        raise ValueError("Original observation clock requires an explicit timezone")
    capture_sources = {}
    for record in lineage["captures"]:
        capture = Path(record["path"])
        manifest = verify(capture)
        at = pd.Timestamp(record["fetched_at"])
        if (file_hash(capture / "manifest.json") != record["manifest_sha256"]
                or manifest["raw_sha256"] != record["raw_sha256"]
                or manifest["fetched_at"] != record["fetched_at"]
                or pd.isna(at) or at.tzinfo is None or at > observed):
            raise ValueError("Normalized capture binding or actual observation time changed")
        capture_sources[str(capture / "manifest.json")] = record["manifest_sha256"]
        capture_sources.update({str(capture / name): expected for name, expected in manifest["files"].items()})
    output.mkdir(parents=True)
    origin = output / "origin"
    origin.mkdir()
    for path in [source / "snapshot/manifest.json", lineage_path, archived_code]:
        target = origin / ("daily-snapshot-manifest.json" if path.name == "manifest.json" else path.name)
        shutil.copyfile(path, target)
    for name in original["files"]:
        shutil.copyfile(source / "snapshot" / name, output / name)
    new_declaration = {**declaration, "daily_source_dataset_id": original["dataset_id"],
        "daily_source_manifest_sha256": file_hash(source / "snapshot/manifest.json"),
        "daily_lineage_sha256": file_hash(lineage_path),
        "structural_adapter_source_sha256": file_hash(Path(__file__))}
    identity = {"schema_version": 1, "declaration": new_declaration, "files": original["files"]}
    manifest = {**identity, "dataset_id": digest(identity), "created_at": utc_now(),
                "source_created_at": original["created_at"]}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    receipt = {"schema_version": 1, "adapted_at": utc_now(), "status": "validating_canonical_training_structure",
        "source_dataset_id": original["dataset_id"], "adapted_dataset_id": manifest["dataset_id"],
        "source_created_at_preserved": original["created_at"], "purpose": "training",
        "table_bytes_unchanged": None, "source_declarations_unchanged": True,
        "formal_profit_qualified": False, "account_simulation_allowed": False,
        "capture_source_hashes": capture_sources,
        "source_hashes": {str(p): file_hash(p) for p in [source / "snapshot/manifest.json", lineage_path, archived_code]},
        "limits": ["Canonical structure is not semantic, execution or historical acceptance.",
                   "Training purpose and every original qualification/availability field remain unchanged."]}
    try:
        for name, expected in original["files"].items():
            if file_hash(output / name) != expected:
                raise ValueError("Copied research table bytes changed")
        receipt["table_bytes_unchanged"] = True
        Snapshot(output)
        receipt["status"] = "canonical_training_structure_verified_account_returns_blocked"
    except Exception as exc:
        receipt["status"] = "failed_canonical_structure_validation"
        receipt["error_type"] = type(exc).__name__
        receipt["reason"] = str(exc)
        (output / "adapter-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        raise
    receipt["output_hashes"] = {str(p.relative_to(output)): file_hash(p)
                                for p in sorted(output.rglob("*")) if p.is_file()}
    (output / "adapter-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    return receipt
