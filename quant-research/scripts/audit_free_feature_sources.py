"""Compare the acquired primary source to previously frozen Sina reference archives."""

import argparse
from pathlib import Path

import pandas as pd
from price_pilot import verify_files
from verify_plan_asof_fit import read

from quant_research.storage import file_hash, utc_now, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--probe", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    source = a.source.resolve()
    probe = a.probe.resolve()
    out = a.output.resolve()
    assert read(source / "run-status.json")["status"] == "completed"
    verify_files(probe, read(probe / "evidence-manifest.json"))
    frames = []
    for path in sorted(probe.glob("sina_bars-*.decoded.json")):
        symbol = path.name.removeprefix("sina_bars-").removesuffix(".decoded.json")
        df = pd.DataFrame(read(path))
        df["date"] = df.date.str[:10]
        df["instrument_id"] = (
            "cn." + {"sh": "xshg", "sz": "xshe", "bj": "xbse"}[symbol[:2]] + "." + symbol[2:]
        )
        frames.append(df[["instrument_id", "date", "close", "volume"]])
    sina = pd.concat(frames, ignore_index=True)
    assert sina.date.max() < "2025-08-07"
    ids = set(sina.instrument_id)
    records = []
    for day, receipt in read(source / "index.json").items():
        path = source / "daily" / f"{day}.parquet"
        assert file_hash(path) == receipt["parquet_sha256"]
        df = pd.read_parquet(path)
        records.append(df[df.instrument_id.isin(ids)])
    primary = pd.concat(records, ignore_index=True)
    matched = primary.merge(
        sina,
        on=["instrument_id", "date"],
        suffixes=("_primary", "_sina"),
        how="left",
        validate="one_to_one",
    )
    diff = (matched.close_primary - matched.close_sina).abs()
    out.mkdir()
    matched.to_parquet(out / "sina-price-crosscheck.parquet", index=False)
    write_json(
        out / "verification.json",
        dict(
            passed=True,
            verified_at=utc_now(),
            source=str(source),
            source_status_sha256=file_hash(source / "run-status.json"),
            probe=str(probe),
            probe_manifest_sha256=file_hash(probe / "evidence-manifest.json"),
            primary_rows=len(primary),
            overlap=int(diff.notna().sum()),
            within_one_cent=int((diff <= 0.010001).sum()),
            max_difference=float(diff.max()),
            unmatched_rows=int(diff.isna().sum()),
            scope="Auxiliary historical Sina price check only; no automatic fallback/overwrite and no original-publication certification.",
            files={
                "sina-price-crosscheck.parquet": file_hash(out / "sina-price-crosscheck.parquet")
            },
        ),
    )
    print(read(out / "verification.json"))


if __name__ == "__main__":
    main()
