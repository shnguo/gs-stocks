"""Freeze and resume bounded Rust-acquired research features, independently replayed."""

import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from quant_research.free_features import replay
from quant_research.storage import file_hash, utc_now, write_json


def read(p):
    return json.loads(p.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--rust-repo", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    prior = a.prior.resolve()
    repo = a.rust_repo.resolve()
    folds = [1, 7, 15]
    if not out.exists():
        out.mkdir()
        (out / "captures").mkdir()
        (out / "daily").mkdir()
        (out / "code").mkdir()
        dates = sorted(
            {
                d
                for f in folds
                for ds in read(prior / f"fold-{f:02d}/config.json")["dates"].values()
                for d in ds
            }
        )
        if max(dates) >= "2025-08-07":
            raise ValueError("Sealed dates")
        write_json(
            out / "protocol.json",
            dict(
                folds=folds,
                dates=dates,
                prior=str(prior),
                parent_status_sha256=file_hash(prior / "run-status.json"),
                concurrency=2,
                arms=["daily_information"],
                sealed_holdout_start="2025-08-07",
                promotion=False,
                source_publication_verified=False,
            ),
        )
        shutil.copy2(repo / "target/debug/mootdx-cf-rs", out / "collector")
        for name in [
            "src/research_daily_features.rs",
            "crates/lakehouse-sources/src/eastmoney_daily_features.rs",
            "Cargo.lock",
        ]:
            dst = out / "code" / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo / name, dst)
        shutil.copy2(Path(__file__), out / "code/free_feature_ingest.py")
        shutil.copy2(
            Path(__file__).resolve().parents[1] / "src/quant_research/free_features.py",
            out / "code/free_features.py",
        )
        write_json(
            out / "frozen-manifest.json",
            {str(f.relative_to(out)): file_hash(f) for f in out.rglob("*") if f.is_file()},
        )
    protocol = read(out / "protocol.json")
    if file_hash(prior / "run-status.json") != protocol["parent_status_sha256"]:
        raise ValueError("Parent changed")
    for n, h in read(out / "frozen-manifest.json").items():
        if file_hash(out / n) != h:
            raise ValueError("Frozen acquisition changed")
    state_path = out / "run-status.json"
    if state_path.exists() and read(state_path)["status"] == "completed":
        for name, expected in read(state_path)["files"].items():
            if file_hash(out / name) != expected:
                raise ValueError("Completed acquisition changed")
        print("Already complete and verified; preserving its identity", flush=True)
        return
    lock = out / "active.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)

    def capture(day):
        receipt = out / "daily" / f"{day}.json"
        if receipt.exists():
            m = read(receipt)
            if file_hash(out / "daily" / f"{day}.parquet") != m["parquet_sha256"]:
                raise ValueError("Changed partition")
            directory = out / m["capture"]
            if file_hash(directory / "manifest.json") != m["capture_sha256"]:
                raise ValueError("Changed receipt")
            replay(directory)
            return m
        attempt = 1
        while (out / "captures" / f"{day}-attempt-{attempt}").exists():
            attempt += 1
        directory = out / "captures" / f"{day}-attempt-{attempt}"
        log = out / "captures" / f"{day}-attempt-{attempt}.log"
        with log.open("x") as handle:
            subprocess.run(
                [str(out / "collector"), "research-capture-daily-features", day, str(directory)],
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
            subprocess.run(
                [str(out / "collector"), "research-replay-daily-features", str(directory)],
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
        df = replay(directory)
        df.to_parquet(out / "daily" / f"{day}.parquet", index=False)
        m = dict(
            date=day,
            rows=len(df),
            capture=str(directory.relative_to(out)),
            capture_sha256=file_hash(directory / "manifest.json"),
            parquet_sha256=file_hash(out / "daily" / f"{day}.parquet"),
        )
        write_json(receipt, m)
        return m

    try:
        write_json(
            out / "run-status.json", dict(status="running", pid=os.getpid(), started_at=utc_now())
        )
        results = []
        # Batches bound work after any failure and never flood the endpoint.
        with ThreadPoolExecutor(max_workers=2) as pool:
            for start in range(0, len(protocol["dates"]), 2):
                results.extend(pool.map(capture, protocol["dates"][start : start + 2]))
                write_json(
                    out / "progress.json",
                    dict(
                        completed=len(results),
                        total=len(protocol["dates"]),
                        rows=sum(r["rows"] for r in results),
                        updated_at=utc_now(),
                    ),
                )
                print("Captured", len(results), "/", len(protocol["dates"]), flush=True)
        write_json(out / "index.json", {m["date"]: m for m in results})
        write_json(
            out / "verification.json",
            dict(
                passed=True,
                dates=len(results),
                rows=sum(m["rows"] for m in results),
                raw_replay="Rust and independent Python",
                original_publication_verified=False,
            ),
        )
        write_json(
            out / "run-status.json",
            dict(
                status="completed",
                finished_at=utc_now(),
                files={
                    str(f.relative_to(out)): file_hash(f)
                    for f in out.rglob("*")
                    if f.is_file()
                    and f.name not in ["active.lock", "run-status.json", "progress.json"]
                    and "__pycache__" not in f.parts
                },
            ),
        )
    except BaseException as e:
        write_json(
            out / "run-status.json", dict(status="failed", error=repr(e), updated_at=utc_now())
        )
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
