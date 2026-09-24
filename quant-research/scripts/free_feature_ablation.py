"""Single-arm information ablation on three frozen development windows."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from plan_value_run import compact_targets, decision_metrics
from price_pilot import load_arrays, verify_files

from quant_research.free_features import FEATURES, align
from quant_research.plan_value import (
    apply_calibration,
    calibration_bins,
    compare_heads,
    fit_calibration,
    fit_heads,
    head_metrics,
    predict_heads,
)
from quant_research.storage import file_hash, utc_now, write_json


def read(p):
    return json.loads(p.read_text())


def freeze(args):
    out = args.output.resolve()
    source = args.source.resolve()
    prior = args.prior.resolve()
    for root in [source, prior]:
        state = read(root / "run-status.json")
        if state["status"] != "completed" or not read(root / "verification.json")["passed"]:
            raise ValueError("Unverified experiment/source")
        verify_files(root, state["files"])
    folds = read(source / "protocol.json")["folds"]
    dates = set(read(source / "index.json"))
    required = {
        d
        for f in folds
        for ds in read(prior / f"fold-{f:02d}/config.json")["dates"].values()
        for d in ds
    }
    if dates != required:
        raise ValueError("Source dates differ")
    out.mkdir()
    protocol = dict(
        fold_indices=folds,
        arms=["daily_information"],
        added_features=FEATURES,
        primary="daily_information_raw_vs_baseline27_raw",
        selection="Predeclared fold 1,7,15; no tuning on results",
        missingness="exact same-date join; missing/quote mismatch become NaN with availability flag; original cohort unchanged",
        min_available_fraction=0.90,
        original_publication_verified=False,
        promotion=False,
        executable=False,
    )
    write_json(out / "protocol.json", protocol)
    write_json(
        out / "experiment.json",
        dict(
            prior=str(prior),
            source=str(source),
            parent_status_sha256=file_hash(prior / "run-status.json"),
            source_status_sha256=file_hash(source / "run-status.json"),
            training_protocol={**read(prior / "protocol.json"), "fold_indices": folds},
            created_at=utc_now(),
            executable=False,
        ),
    )
    base = Path(__file__).resolve().parents[1]
    shutil.copytree(base / "src", out / "code/src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(
        base / "scripts", out / "code/scripts", ignore=shutil.ignore_patterns("__pycache__")
    )
    shutil.copy2(base / "uv.lock", out / "code/uv.lock")
    write_json(
        out / "frozen-manifest.json",
        {str(f.relative_to(out)): file_hash(f) for f in out.rglob("*") if f.is_file()},
    )


def context_for(rows, source, snapshot):
    index = read(source / "index.json")
    tables = []
    for day in sorted(rows.date.unique()):
        m = index[day]
        path = source / "daily" / f"{day}.parquet"
        if file_hash(path) != m["parquet_sha256"]:
            raise ValueError("Feature source changed")
        tables.append(pd.read_parquet(path))
    records = pd.concat(tables, ignore_index=True)
    bars = pd.read_parquet(
        snapshot / "bars.parquet",
        columns=["instrument_id", "date", "close", "volume"],
        filters=[("date", "in", sorted(rows.date.unique()))],
    )
    return align(rows, bars, records)


def fit(args):
    out = args.output.resolve()
    exp = read(out / "experiment.json")
    protocol = read(out / "protocol.json")
    prior = Path(exp["prior"]) / f"fold-{args.fold:02d}"
    source = Path(exp["source"])
    snapshot = Path(read(prior / "config.json")["root"]) / "snapshot"
    dest = out / f"fold-{args.fold:02d}"
    dest.mkdir()
    write_json(dest / "features.json", dict(base_count=27, added=FEATURES))

    def load(part):
        data = load_arrays(prior / f"{part}.npz")
        rows = pd.read_parquet(prior / f"{part}-rows.parquet")
        if rows.label_end.max() >= "2025-08-07":
            raise ValueError("Sealed labels")
        context, coverage, reasons = context_for(rows, source, snapshot)
        if coverage["source_available"] / len(rows) < protocol["min_available_fraction"]:
            raise ValueError(f"Coverage below frozen threshold: {coverage}")
        np.save(dest / f"{part}-context.npy", context)
        write_json(dest / f"{part}-coverage.json", coverage)
        rows[["instrument_id", "date"]].assign(feature_status=reasons).to_parquet(
            dest / f"{part}-source-status.parquet", index=False
        )
        data["x"] = np.column_stack([data["x"], context])
        return data, rows

    train, tr = load("train")
    selection, sr = load("selection")
    metadata = fit_heads(
        train["x"],
        tr.date,
        compact_targets(train),
        selection["x"],
        sr.date,
        compact_targets(selection),
        dest / "plan-models",
        exp["training_protocol"],
    )
    write_json(dest / "models.json", metadata)
    del train, selection
    cal, cr = load("calibration")
    calibration = fit_calibration(
        predict_heads(cal["x"], dest / "plan-models", metadata), compact_targets(cal), cr.date
    )
    write_json(dest / "calibration.json", calibration)
    del cal
    evaluation, rows = load("evaluation")
    raw = predict_heads(evaluation["x"], dest / "plan-models", metadata)
    forecasts = dict(
        daily_information_raw=raw, daily_information=apply_calibration(raw, calibration)
    )
    for model, pred in forecasts.items():
        np.savez_compressed(dest / f"{model}-evaluation.npz", **pred)
        metrics, chosen = decision_metrics(rows, pred, evaluation, f"fold-{args.fold:02d}", model)
        metrics.to_csv(dest / f"{model}-decision-metrics.csv", index=False)
        chosen.to_parquet(dest / f"{model}-chosen.parquet", index=False)
    head_metrics(rows, compact_targets(evaluation), forecasts, f"fold-{args.fold:02d}").to_csv(
        dest / "head-metrics.csv", index=False
    )
    calibration_bins(rows, compact_targets(evaluation), forecasts, f"fold-{args.fold:02d}").to_csv(
        dest / "calibration-bins.csv", index=False
    )
    write_json(
        dest / "completed.json",
        dict(
            status="completed",
            files={str(f.relative_to(dest)): file_hash(f) for f in dest.rglob("*") if f.is_file()},
        ),
    )


def run(args):
    out = args.output.resolve()
    exp = read(out / "experiment.json")
    protocol = read(out / "protocol.json")
    prior = Path(exp["prior"])
    verify_files(out, read(out / "frozen-manifest.json"))
    for root, key in [
        (prior, "parent_status_sha256"),
        (Path(exp["source"]), "source_status_sha256"),
    ]:
        if file_hash(root / "run-status.json") != exp[key]:
            raise ValueError("Parent changed")
        verify_files(root, read(root / "run-status.json")["files"])
    if (out / "run-status.json").exists():
        raise FileExistsError("Preserve previous run")
    write_json(
        out / "run-status.json", dict(status="running", pid=os.getpid(), started_at=utc_now())
    )
    try:
        for i, fold in enumerate(protocol["fold_indices"]):
            write_json(
                out / "progress.json",
                dict(
                    completed=i,
                    total=len(protocol["fold_indices"]),
                    fold=fold,
                    updated_at=utc_now(),
                ),
            )
            print("Starting information fold", fold, flush=True)
            with (out / f"fold-{fold:02d}.log").open("x") as log:
                subprocess.run(
                    [
                        sys.executable,
                        str(out / "code/scripts/free_feature_ablation.py"),
                        "fit",
                        "--output",
                        str(out),
                        "--fold",
                        str(fold),
                    ],
                    env={**os.environ, "PYTHONPATH": str(out / "code/src")},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            verify_files(
                out / f"fold-{fold:02d}", read(out / f"fold-{fold:02d}/completed.json")["files"]
            )
        folds = [f"fold-{f:02d}" for f in protocol["fold_indices"]]
        baseline = pd.read_csv(prior / "head-metrics.csv")
        baseline = baseline[
            baseline.window.isin(folds) & baseline.model.isin(["learned_raw", "learned"])
        ].replace({"model": {"learned_raw": "baseline27_raw", "learned": "baseline27"}})
        all_metrics = pd.concat(
            [baseline, *[pd.read_csv(out / f / "head-metrics.csv") for f in folds]],
            ignore_index=True,
        )
        all_metrics.to_csv(out / "head-metrics.csv", index=False)
        decisions = pd.concat(
            [
                pd.read_csv(out / f / f"{n}-decision-metrics.csv")
                for f in folds
                for n in ["daily_information_raw", "daily_information"]
            ],
            ignore_index=True,
        )
        decisions.to_csv(out / "decision-metrics.csv", index=False)
        comparisons = {}
        for suffix in ["_raw", ""]:
            candidate = "daily_information" + suffix
            reference = "baseline27" + suffix
            paired = all_metrics[all_metrics.model.isin([candidate, reference])].replace(
                {"model": {candidate: "learned", reference: "empirical"}}
            )
            comparisons[candidate + "_vs_" + reference] = compare_heads(paired)
        write_json(
            out / "assessment.json",
            dict(
                comparisons=comparisons,
                primary=protocol["primary"],
                quality_promotion=False,
                executable=False,
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
                    and f.name not in ["run-status.json", "progress.json"]
                    and "__pycache__" not in f.parts
                },
            ),
        )
        write_json(
            out / "progress.json",
            dict(completed=len(folds), total=len(folds), status="completed", updated_at=utc_now()),
        )
    except BaseException as e:
        write_json(
            out / "run-status.json", dict(status="failed", error=repr(e), updated_at=utc_now())
        )
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["freeze", "run", "fit"])
    p.add_argument("--prior", type=Path)
    p.add_argument("--source", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fold", type=int)
    a = p.parse_args()
    globals()[a.stage](a)
