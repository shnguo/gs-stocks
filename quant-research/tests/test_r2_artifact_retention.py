import importlib.util
import json
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "archive_research_artifact", BASE / "scripts/archive_research_artifact.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def source(path: Path, size: int, value: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value.encode() * size)
    return MODULE.SourceFile(
        path, path.relative_to(BASE), path.stat().st_size, MODULE.file_hash(path)
    )


def test_package_plan_is_deterministic_and_keeps_oversize_file_singleton(tmp_path):
    root = BASE / "artifacts" / "retention-test-fixture"
    files = [
        source(root / f"{index}.bin", size, str(index))
        for index, size in enumerate([4, 4, 12, 3], 1)
    ]
    packages = MODULE.plan_packages(files, 10)
    assert [[item.size for item in package] for package in packages] == [[4, 4], [12], [3]]
    for path in reversed([item.path for item in files]):
        path.unlink()
    root.rmdir()


def test_collect_requires_completion_and_rejects_symlinks(tmp_path):
    config = json.loads((BASE / "configs/r2-artifact-retention-v1.json").read_text())
    root = BASE / "artifacts" / "retention-test-collection"
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.bin").write_bytes(b"data")
    with pytest.raises(ValueError, match="No completed"):
        MODULE.collect_files([root], config)
    (root / "completed.json").write_text(
        json.dumps({"files": {"data.bin": MODULE.file_hash(root / "data.bin")}})
    )
    target = root / "target.bin"
    target.write_bytes(b"target")
    link = root / "link.bin"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symbolic link"):
        MODULE.collect_files([root], config)
    link.unlink()
    for path in root.iterdir():
        path.unlink()
    root.rmdir()


def test_collect_rejects_incomplete_manifest():
    config = json.loads((BASE / "configs/r2-artifact-retention-v1.json").read_text())
    root = BASE / "artifacts" / "retention-test-incomplete"
    root.mkdir(parents=True, exist_ok=True)
    (root / "completed.json").write_text(json.dumps({"files": {"missing.bin": "0" * 64}}))
    with pytest.raises(ValueError, match="missing or changed"):
        MODULE.collect_files([root], config)
    (root / "completed.json").unlink()
    root.rmdir()


def test_stage_records_exact_hashes(tmp_path):
    root = BASE / "artifacts" / "retention-test-stage-source"
    item = source(root / "completed.json", 1, "x")
    stage = tmp_path / "stage"
    manifest = {"files": MODULE.inventory([item])}
    MODULE.stage_package(stage, [item], manifest)
    staged = stage / item.relative
    assert MODULE.file_hash(staged) == item.sha256
    assert json.loads((stage / "artifacts/_retention/package.json").read_text()) == manifest
    item.path.unlink()
    root.rmdir()
