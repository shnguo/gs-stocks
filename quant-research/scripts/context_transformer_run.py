"""Temporal Transformer plus frozen signal-day information; matched existing baselines."""

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

from quant_research.free_features import FEATURES
from quant_research.plan_value import (
    apply_calibration,
    calibration_bins,
    compare_heads,
    fit_calibration,
    head_metrics,
)
from quant_research.storage import file_hash, utc_now, write_json


def read(path):
    return json.loads(path.read_text())


def freeze(args):
    prior, feature, baseline, out = [
        p.resolve() for p in [args.prior, args.features, args.baseline, args.output]
    ]
    for path in [prior, feature, baseline]:
        state = read(path / "run-status.json")
        if state["status"] != "completed" or not read(path / "verification.json")["passed"]:
            raise ValueError("Unverified parent")
        verify_files(path, state["files"])
    fe, be = read(feature / "experiment.json"), read(baseline / "experiment.json")
    if Path(fe["prior"]) != prior or Path(be["prior"]) != prior:
        raise ValueError("Different cohorts")
    folds = read(feature / "protocol.json")["fold_indices"]
    if not set(folds).issubset(be["fold_indices"]):
        raise ValueError("Missing baseline folds")
    config = read(baseline / "protocol.json")
    config.update(
        protocol_id="context-plan-transformer-v1",
        context_features=len(FEATURES),
        purpose="Three predeclared frozen information cohorts; same temporal backbone, optimization and targets as original Transformer",
        input="60 historical sessions x 27 features plus current signal-day 11 features; train-only finite normalization and 11 missing masks",
        context_architecture="22 input values/masks -> width -> GELU -> width, additive fusion with last temporal token",
        promotion=False,
    )
    panel = Path(be["panel"])
    verify_files(panel, read(panel / "manifest.json")["files"])
    out.mkdir()
    write_json(out / "protocol.json", config)
    write_json(
        out / "experiment.json",
        dict(
            prior=str(prior),
            features=str(feature),
            baseline=str(baseline),
            parent_status_sha256=file_hash(prior / "run-status.json"),
            feature_status_sha256=file_hash(feature / "run-status.json"),
            baseline_status_sha256=file_hash(baseline / "run-status.json"),
            fold_indices=folds,
            panel=str(panel),
            panel_manifest_sha256=file_hash(panel / "manifest.json"),
            executable=False,
            created_at=utc_now(),
        ),
    )
    write_json(
        out / "context-manifest.json",
        {
            f"fold-{fold:02d}/{part}-context.npy": file_hash(
                feature / f"fold-{fold:02d}/{part}-context.npy"
            )
            for fold in folds
            for part in ["train", "selection", "calibration", "evaluation"]
        },
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


def fit(args):
    from quant_research.plan_transformer import predict, train

    out = args.output.resolve()
    exp, config = read(out / "experiment.json"), read(out / "protocol.json")
    verify_files(out, read(out / "frozen-manifest.json"))
    if args.fold not in exp["fold_indices"]:
        raise ValueError("Unregistered fold")
    feature = Path(exp["features"])
    prior = Path(exp["prior"]) / f"fold-{args.fold:02d}"
    panel = Path(exp["panel"])
    assert file_hash(panel / "manifest.json") == exp["panel_manifest_sha256"]
    values = np.load(panel / "values.npy", mmap_mode="r")
    dest = out / f"fold-{args.fold:02d}"
    inference_only = getattr(args, "inference_only", False)
    if inference_only:
        verify_files(dest, read(dest / "trained.json")["files"])
    else:
        dest.mkdir()
    rows = {
        p: pd.read_parquet(prior / f"{p}-rows.parquet")
        for p in ["train", "selection", "calibration", "evaluation"]
    }
    context = {}
    for part, frame in rows.items():
        assert frame.label_end.max() < config["sealed_holdout_start"]
        key = f"fold-{args.fold:02d}/{part}-context.npy"
        if file_hash(feature / key) != read(out / "context-manifest.json")[key]:
            raise ValueError("Changed financial context")
        ctx = np.load(feature / key)
        if ctx.shape != (len(frame), config["context_features"]):
            raise ValueError("Context shape differs")
        status = pd.read_parquet(feature / f"fold-{args.fold:02d}/{part}-source-status.parquet")
        pd.testing.assert_frame_equal(
            frame[["instrument_id", "date"]], status[["instrument_id", "date"]]
        )
        context[part] = ctx
    if inference_only:
        import torch

        from quant_research.plan_transformer import PlanTransformer

        torch.set_num_threads(4)
        saved = load_arrays(dest / "scalers.npz")
        model = PlanTransformer(
            saved["target_center"],
            saved["target_scale"],
            saved["prior"],
            config["width"],
            config["context_features"],
            config.get("dropout", 0.1),
        ).to(config["device"])
        model.load_state_dict(
            torch.load(dest / "transformer.pt", map_location="cpu", weights_only=True)["state_dict"]
        )
        mean, scale = saved["mean"], saved["scale"]
    else:
        tr, selection = load_arrays(prior / "train.npz"), load_arrays(prior / "selection.npz")
        model, mean, scale = train(
            values,
            rows["train"],
            tr,
            compact_targets(tr),
            rows["selection"],
            selection,
            compact_targets(selection),
            dest,
            config,
            train_context=context["train"],
            select_context=context["selection"],
        )
        del tr, selection
        if getattr(args, "train_only", False):
            write_json(
                dest / "trained.json",
                dict(
                    status="trained_selection_only",
                    files={
                        str(f.relative_to(dest)): file_hash(f)
                        for f in dest.rglob("*")
                        if f.is_file()
                    },
                ),
            )
            return
    with np.load(dest / "scalers.npz") as scalers:
        cm, cs = scalers["context_mean"], scalers["context_scale"]
    cal = load_arrays(prior / "calibration.npz")
    raw_cal, price_cal = predict(
        model,
        values,
        rows["calibration"],
        mean,
        scale,
        context=context["calibration"],
        context_mean=cm,
        context_scale=cs,
    )
    np.savez_compressed(dest / "raw-calibration.npz", **raw_cal)
    np.save(dest / "price-calibration.npy", price_cal)
    calibration = fit_calibration(raw_cal, compact_targets(cal), rows["calibration"].date)
    write_json(dest / "calibration.json", calibration)
    del cal
    data = load_arrays(prior / "evaluation.npz")
    raw, price = predict(
        model,
        values,
        rows["evaluation"],
        mean,
        scale,
        context=context["evaluation"],
        context_mean=cm,
        context_scale=cs,
    )
    np.save(dest / "price-evaluation.npy", price)
    forecasts = dict(
        context_transformer_raw=raw, context_transformer=apply_calibration(raw, calibration)
    )
    window = f"fold-{args.fold:02d}"
    for name, pred in forecasts.items():
        np.savez_compressed(dest / f"{name}-evaluation.npz", **pred)
        metrics, chosen = decision_metrics(rows["evaluation"], pred, data, window, name)
        metrics.to_csv(dest / f"{name}-decision-metrics.csv", index=False)
        chosen.to_parquet(dest / f"{name}-chosen.parquet", index=False)
    head_metrics(rows["evaluation"], compact_targets(data), forecasts, window).to_csv(
        dest / "head-metrics.csv", index=False
    )
    calibration_bins(rows["evaluation"], compact_targets(data), forecasts, window).to_csv(
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
    e = read(out / "experiment.json")
    verify_files(out, read(out / "frozen-manifest.json"))
    for key, hashkey in [
        ("prior", "parent_status_sha256"),
        ("features", "feature_status_sha256"),
        ("baseline", "baseline_status_sha256"),
    ]:
        root = Path(e[key])
        assert file_hash(root / "run-status.json") == e[hashkey]
        verify_files(root, read(root / "run-status.json")["files"])
    if (out / "run-status.json").exists():
        raise FileExistsError("Keep prior attempts")
    write_json(
        out / "run-status.json", dict(status="running", pid=os.getpid(), started_at=utc_now())
    )
    try:
        for index, fold in enumerate(e["fold_indices"]):
            write_json(
                out / "progress.json",
                dict(
                    status="training",
                    completed_fits=index,
                    total_fits=len(e["fold_indices"]),
                    fold=fold,
                    updated_at=utc_now(),
                ),
            )
            print("Starting context Transformer", fold, flush=True)
            with (out / f"fold-{fold:02d}.log").open("x") as log:
                subprocess.run(
                    [
                        sys.executable,
                        str(out / "code/scripts/context_transformer_run.py"),
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
            print("Completed context Transformer", fold, flush=True)
        finalize(out)
    except BaseException as exc:
        write_json(
            out / "run-status.json", dict(status="failed", error=repr(exc), updated_at=utc_now())
        )
        raise


def finalize(out):
    e = read(out / "experiment.json")
    windows = [f"fold-{f:02d}" for f in e["fold_indices"]]
    neural_reference = pd.read_csv(Path(e["baseline"]) / "head-metrics.csv")
    neural_reference = neural_reference[
        neural_reference.model.isin(["transformer", "transformer_raw"])
    ]
    frames = [pd.read_csv(Path(e["features"]) / "head-metrics.csv"), neural_reference]
    frames += [pd.read_csv(out / w / "head-metrics.csv") for w in windows]
    metrics = pd.concat(frames, ignore_index=True)
    metrics = metrics[metrics.window.isin(windows)]
    if metrics.duplicated(["window", "date", "model", "head"]).any():
        raise ValueError("Duplicate comparison metrics")
    metrics.to_csv(out / "head-metrics.csv", index=False)
    decisions = pd.concat(
        [
            pd.read_csv(out / w / f"{n}-decision-metrics.csv")
            for w in windows
            for n in ["context_transformer_raw", "context_transformer"]
        ],
        ignore_index=True,
    )
    decisions.to_csv(out / "decision-metrics.csv", index=False)
    comparisons = {}
    for suffix in ["_raw", ""]:
        candidate = "context_transformer" + suffix
        for base in ["transformer", "daily_information", "baseline27"]:
            reference = base + suffix
            pair = metrics[metrics.model.isin([candidate, reference])].replace(
                {"model": {candidate: "learned", reference: "empirical"}}
            )
            comparisons[candidate + "_vs_" + reference] = compare_heads(pair)
    write_json(
        out / "assessment.json",
        dict(
            comparisons=comparisons,
            primary="context_transformer_raw_vs_transformer_raw",
            quality_promotion=False,
            executable=False,
            interpretation="Matched pipeline comparison. Auxiliary context branch changes parameter count; not a pure architecture causal test. Retrospective feature publication/missingness limitations remain.",
        ),
    )
    write_json(
        out / "progress.json",
        dict(
            status="completed",
            completed_fits=len(windows),
            total_fits=len(windows),
            updated_at=utc_now(),
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
                if f.is_file() and f.name != "run-status.json" and "__pycache__" not in f.parts
            },
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["freeze", "fit", "run"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prior", type=Path)
    p.add_argument("--features", type=Path)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--fold", type=int)
    a = p.parse_args()
    globals()[a.stage](a)
