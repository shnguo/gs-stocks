"""Read hash-bound coverage exceptions without turning missing data into prices."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .storage import Snapshot, file_hash

SUSPENSIONS = {"source_verified_suspension", "announcement_verified_suspension"}
INACTIVE = {"outside_verified_code_window", "official_delisting_date_exclusive",
            "verified_outside_history", "reviewed_post_final_trade"}


def load_coverage(snapshot: Snapshot, descriptor: dict | None) -> dict[tuple[str, str], str]:
    if descriptor is None:
        return {}
    manifest_path = Path(descriptor["export_manifest"])
    expected = snapshot.manifest["declaration"].get("canonical_export_manifest_sha256")
    if not expected or file_hash(manifest_path) != expected:
        raise ValueError("Coverage source is not bound to this snapshot")
    manifest = json.loads(manifest_path.read_text())
    source = manifest["sources"]["gap_ledger"]
    path = Path(descriptor.get("gap_ledger", source["path"]))
    if file_hash(path) != source["sha256"]:
        raise ValueError("Coverage evidence hash mismatch")
    result = {}
    for line in path.open():
        row = json.loads(line)
        key = row["instrument_id"], row["date"]
        kind = row["classification"]
        if key in result or kind not in SUSPENSIONS | INACTIVE:
            raise ValueError("Duplicate or unresolved coverage exception")
        if row["resolution"]["classification"] != kind:
            raise ValueError("Coverage resolution mismatch")
        result[key] = kind
    observations = manifest_path.parent / "observation-exclusions.csv"
    expected = manifest["files"][observations.name]["sha256"]
    if file_hash(observations) != expected:
        raise ValueError("Observation exclusion evidence hash mismatch")
    for row in csv.DictReader(observations.open()):
        key = row["instrument_id"], row["date"]
        kind = row["reason"]
        if key in result or kind not in INACTIVE:
            raise ValueError("Unresolved or duplicate source observation exclusion")
        result[key] = kind
    return result
