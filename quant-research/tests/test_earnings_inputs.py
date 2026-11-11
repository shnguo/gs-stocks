import json

import pandas as pd
import pytest

from quant_research.config import digest
from quant_research.earnings_inputs import adapt_daily_training_input
from quant_research.earnings_ledger import simulate_earnings_account
from quant_research.fixtures import create_fixture
from quant_research.storage import TABLES, Snapshot, file_hash


def legacy_input(tmp_path):
    snapshot = create_fixture(tmp_path / "tables", tmp_path / "source/snapshot", stocks=3, days=40)
    source = snapshot.path.parent
    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "normalized.json").write_text("{}\n")
    info = dict(fetched_at="2023-01-01T00:00:00Z", raw_sha256="a" * 64,
                files={"normalized.json": file_hash(capture / "normalized.json")})
    (capture / "manifest.json").write_text(json.dumps(info))
    (source / "code").mkdir()
    archived = source / "code/daily_input.py"
    archived.write_text("# artificial derivation\n")
    lineage = dict(implementation_sha256=file_hash(archived), captures=[dict(path=str(capture),
        fetched_at=info["fetched_at"], raw_sha256=info["raw_sha256"], manifest_sha256=file_hash(capture / "manifest.json"))])
    (source / "lineage.json").write_text(json.dumps(lineage))
    manifest = snapshot.manifest.copy()
    manifest.pop("schema_version")
    manifest["declaration"] = {**manifest["declaration"], "purpose": "training"}
    manifest["created_at"] = "2023-04-01T00:00:00Z"
    manifest["dataset_id"] = digest(lineage)
    (source / "snapshot/manifest.json").write_text(json.dumps(manifest))
    return source, manifest


def test_adapter_preserves_bytes_declarations_clocks_and_blocks_returns(tmp_path):
    source, original = legacy_input(tmp_path)
    output = tmp_path / "adapted"
    receipt = adapt_daily_training_input(source, output)
    loaded = Snapshot(output)
    assert receipt["table_bytes_unchanged"] and receipt["source_declarations_unchanged"]
    assert loaded.manifest["source_created_at"] == original["created_at"]
    assert pd.Timestamp(loaded.manifest["created_at"]) > pd.Timestamp(original["created_at"])
    assert all(loaded.manifest["declaration"][key] == value for key, value in original["declaration"].items())
    assert all(file_hash(output / (name + ".parquet")) == original["files"][name + ".parquet"] for name in TABLES)
    with pytest.raises(ValueError, match="Training-only"):
        simulate_earnings_account(loaded, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), loaded.dates[0], loaded.dates[-1])
    with pytest.raises(FileExistsError):
        adapt_daily_training_input(source, output)


@pytest.mark.parametrize("change", ["table", "lineage", "capture", "purpose"])
def test_adapter_rejects_changed_source_before_output_creation(tmp_path, change):
    source, original = legacy_input(tmp_path)
    if change == "table":
        with (source / "snapshot/bars.parquet").open("ab") as f:
            f.write(b"tampered")
    elif change == "lineage":
        (source / "lineage.json").write_text("{}")
    elif change == "capture":
        (tmp_path / "capture/normalized.json").write_text("changed")
    else:
        original["declaration"]["purpose"] = "full_experiment"
        (source / "snapshot/manifest.json").write_text(json.dumps(original))
    with pytest.raises(ValueError):
        adapt_daily_training_input(source, tmp_path / "adapted")
    assert not (tmp_path / "adapted").exists()


def test_adapter_preserves_invalid_table_failure_without_filling_missing_factor(tmp_path):
    source, original = legacy_input(tmp_path)
    path = source / "snapshot/bars.parquet"
    bars = pd.read_parquet(path)
    bars.loc[0, "factor"] = float("nan")
    bars.to_parquet(path, index=False)
    original["files"]["bars.parquet"] = file_hash(path)
    (source / "snapshot/manifest.json").write_text(json.dumps(original))
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="non-finite"):
        adapt_daily_training_input(source, output)
    receipt = json.loads((output / "adapter-receipt.json").read_text())
    assert receipt["status"] == "failed_canonical_structure_validation"
    assert receipt["table_bytes_unchanged"] and "non-finite" in receipt["reason"]
    assert file_hash(output / "bars.parquet") == file_hash(path)
    assert pd.read_parquet(output / "bars.parquet").factor.isna().sum() == 1
    with pytest.raises(FileExistsError):
        adapt_daily_training_input(source, output)
    assert not receipt["account_simulation_allowed"]
