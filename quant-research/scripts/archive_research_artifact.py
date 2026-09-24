"""Archive completed research artifacts to immutable, restore-verified R2 packages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SourceFile:
    path: Path
    relative: Path
    size: int
    sha256: str


def _inside_base(path: Path) -> Path:
    resolved = path.resolve()
    try:
        return resolved.relative_to(BASE.resolve())
    except ValueError as error:
        raise ValueError(f"Archive source must be inside {BASE}: {resolved}") from error


def collect_files(paths: list[Path], config: dict, output: Path | None = None) -> list[SourceFile]:
    excluded_names = set(config["excluded_names"])
    excluded_suffixes = tuple(config["excluded_suffixes"])
    output_resolved = output.resolve() if output else None
    selected: dict[Path, Path] = {}
    completion_roots: list[tuple[Path, Path]] = []
    for source in paths:
        source = source.resolve()
        relative_source = _inside_base(source)
        if not source.exists():
            raise FileNotFoundError(source)
        candidates = [source] if source.is_file() else sorted(source.rglob("*"))
        if source.is_dir():
            markers = [
                source / marker
                for marker in config["completion_markers"]
                if (source / marker).is_file()
            ]
            completion_roots.extend((source, marker) for marker in markers)
        for candidate in candidates:
            if output_resolved and (
                candidate == output_resolved or output_resolved in candidate.parents
            ):
                continue
            if candidate.is_symlink():
                raise ValueError(f"Archive source contains a symbolic link: {candidate}")
            if candidate.is_dir():
                continue
            if not candidate.is_file():
                raise ValueError(f"Archive source contains a special file: {candidate}")
            relative = _inside_base(candidate)
            if excluded_names.intersection(relative.parts) or candidate.name.endswith(
                excluded_suffixes
            ):
                continue
            selected[relative] = candidate
        if source.is_file() and source.name in config["completion_markers"]:
            completion_roots.append((source.parent, source))
        if not candidates and relative_source:
            raise ValueError(f"Archive source is empty: {source}")
    if not selected:
        raise ValueError("No archive files selected")
    if not completion_roots:
        raise ValueError("No completed artifact manifest was selected")
    files = []
    for relative, path in sorted(selected.items()):
        size = path.stat().st_size
        if size > config["single_file_max_bytes"]:
            raise ValueError(f"File exceeds retention single-file limit: {relative} ({size} bytes)")
        files.append(SourceFile(path, relative, size, file_hash(path)))
    hashes = {item.path.resolve(): item.sha256 for item in files}
    for root, marker in completion_roots:
        completed = read_json(marker)
        expected_files = completed.get("files")
        if not isinstance(expected_files, dict) or not expected_files:
            raise ValueError(f"Completion manifest has no file inventory: {marker}")
        for relative, expected in expected_files.items():
            expected_hash = expected.get("sha256") if isinstance(expected, dict) else expected
            path = (root / relative).resolve()
            if hashes.get(path) != expected_hash:
                raise ValueError(f"Completion manifest file is missing or changed: {path}")
    return files


def plan_packages(files: list[SourceFile], maximum: int) -> list[list[SourceFile]]:
    if maximum <= 0:
        raise ValueError("Package maximum must be positive")
    packages: list[list[SourceFile]] = []
    current: list[SourceFile] = []
    current_bytes = 0
    for item in files:
        if current and current_bytes + item.size > maximum:
            packages.append(current)
            current = []
            current_bytes = 0
        current.append(item)
        current_bytes += item.size
        if item.size > maximum or current_bytes >= maximum:
            packages.append(current)
            current = []
            current_bytes = 0
    if current:
        packages.append(current)
    return packages


def inventory(files: list[SourceFile]) -> dict:
    return {str(item.relative): {"bytes": item.size, "sha256": item.sha256} for item in files}


def stage_package(stage: Path, files: list[SourceFile], package_manifest: dict) -> None:
    if stage.exists():
        recorded = stage / "artifacts" / "_retention" / "package.json"
        if not recorded.is_file() or read_json(recorded) != package_manifest:
            raise ValueError(f"Existing package stage differs: {stage}")
        for item in files:
            destination = stage / item.relative
            if not destination.is_file() or file_hash(destination) != item.sha256:
                raise ValueError(f"Existing staged file differs: {destination}")
        return
    for item in files:
        destination = stage / item.relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(item.path, destination)
        except OSError:
            shutil.copy2(item.path, destination)
    write_json(stage / "artifacts" / "_retention" / "package.json", package_manifest)


def next_restore_path(package_root: Path) -> Path:
    attempt = 1
    while True:
        candidate = package_root / f"restored-attempt-{attempt:02d}"
        if not candidate.exists():
            return candidate
        attempt += 1


def archive_package(
    cli: Path,
    cloud_config: Path,
    package_root: Path,
    stage: Path,
) -> dict:
    receipt = package_root / "receipt.json"
    if receipt.exists():
        saved = read_json(receipt)
        if not saved.get("download_and_restore_verified"):
            raise ValueError(f"Unverified existing receipt: {receipt}")
        return saved
    restored = next_restore_path(package_root)
    command = [
        str(cli),
        "research-cloud",
        "archive-sharded",
        str(cloud_config),
        str(stage),
        str(restored),
        str(receipt),
    ]
    subprocess.run(command, cwd=cli.parent.parent.parent, check=True)
    saved = read_json(receipt)
    if not saved.get("download_and_restore_verified"):
        raise ValueError(f"Archive did not pass restore verification: {receipt}")
    return saved


def archive(
    config_path: Path,
    archive_id: str,
    sources: list[Path],
    output: Path,
    cli: Path,
    cloud_config: Path,
) -> Path:
    config_path = config_path.resolve()
    output = output.resolve()
    cli = cli.resolve()
    cloud_config = cloud_config.resolve()
    sources = [
        (BASE / path).resolve() if not path.is_absolute() else path.resolve() for path in sources
    ]
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,79}", archive_id):
        raise ValueError("Archive id must be a lowercase hyphenated identifier")
    config = read_json(config_path)
    completed = output / "retention.json"
    if completed.exists():
        saved = read_json(completed)
        if saved.get("passed") is not True:
            raise ValueError("Existing retention result is incomplete")
        for relative, expected in saved["files"].items():
            path = BASE / relative
            if not path.is_file() or file_hash(path) != expected["sha256"]:
                raise ValueError(f"Retained source changed: {relative}")
        return completed
    output.mkdir(parents=True, exist_ok=True)
    files = collect_files(sources, config, output)
    packages = plan_packages(files, config["package_max_bytes"])
    plan = {
        "version": config["version"],
        "archive_id": archive_id,
        "created_at": utc_now(),
        "package_max_bytes": config["package_max_bytes"],
        "files": inventory(files),
        "packages": [
            {
                "index": index,
                "bytes": sum(item.size for item in package),
                "files": [str(item.relative) for item in package],
            }
            for index, package in enumerate(packages, 1)
        ],
    }
    plan_path = output / "plan.json"
    if plan_path.exists():
        prior = read_json(plan_path)
        stable_keys = ("version", "archive_id", "package_max_bytes", "files", "packages")
        if any(prior[key] != plan[key] for key in stable_keys):
            raise ValueError("Retention inputs changed since the saved plan")
        plan["created_at"] = prior["created_at"]
    else:
        write_json(plan_path, plan)

    receipts = []
    for index, package in enumerate(packages, 1):
        package_root = output / f"package-{index:04d}"
        package_root.mkdir(parents=True, exist_ok=True)
        package_manifest = {
            "version": config["version"],
            "archive_id": archive_id,
            "package": index,
            "package_count": len(packages),
            "files": inventory(package),
        }
        manifest_path = package_root / "package.json"
        if manifest_path.exists() and read_json(manifest_path) != package_manifest:
            raise ValueError(f"Package plan changed: {manifest_path}")
        write_json(manifest_path, package_manifest)
        stage = package_root / "stage"
        stage_package(stage, package, package_manifest)
        receipt = archive_package(cli, cloud_config, package_root, stage)
        receipts.append({"package": index, "receipt": receipt})

    catalog = {
        "version": config["version"],
        "archive_id": archive_id,
        "created_at": utc_now(),
        "source_plan_sha256": file_hash(plan_path),
        "files": inventory(files),
        "packages": receipts,
        "all_packages_restore_verified": all(
            item["receipt"].get("download_and_restore_verified") for item in receipts
        ),
    }
    catalog_root = output / "catalog"
    catalog_root.mkdir(parents=True, exist_ok=True)
    catalog_source = catalog_root / "catalog-source"
    catalog_file = catalog_source / "artifacts" / archive_id / "retention-catalog.json"
    if catalog_source.exists():
        prior = read_json(catalog_file)
        if prior != catalog:
            raise ValueError("Existing R2 catalog stage differs")
    else:
        write_json(catalog_file, catalog)
    catalog_receipt = archive_package(cli, cloud_config, catalog_root, catalog_source)
    result = {
        **catalog,
        "completed_at": utc_now(),
        "catalog_receipt": catalog_receipt,
        "passed": catalog["all_packages_restore_verified"]
        and catalog_receipt.get("download_and_restore_verified") is True,
        "policy": config["policy"],
    }
    if not result["passed"]:
        raise ValueError("R2 retention verification failed")
    write_json(completed, result)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=BASE / "configs/r2-artifact-retention-v1.json"
    )
    parser.add_argument("--archive-id", required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--archive-cli",
        type=Path,
        default=Path("/Users/guo/github/mootdx-cf/target/debug/mootdx-cf-rs"),
    )
    parser.add_argument(
        "--cloud-config",
        type=Path,
        default=Path("/Users/guo/github/mootdx-cf/workers/research/operator.env"),
    )
    args = parser.parse_args()
    print(
        archive(
            args.config,
            args.archive_id,
            args.source,
            args.output,
            args.archive_cli,
            args.cloud_config,
        )
    )


if __name__ == "__main__":
    main()
