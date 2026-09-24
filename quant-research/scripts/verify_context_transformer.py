"""Isolate Torch reload from statistical/decision verification in subprocesses."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from price_pilot import load_arrays, verify_files

from quant_research.storage import file_hash, utc_now, write_json


def read(p):
    return json.loads(p.read_text())


def torch_stage(root):
    import torch

    from quant_research.plan_transformer import PlanTransformer, predict
    from quant_research.plan_value import HEADS

    e, cfg = read(root / "experiment.json"), read(root / "protocol.json")
    values = np.load(Path(e["panel"]) / "values.npy", mmap_mode="r")
    counts = dict(context_scaler_values=0, restored_plan_values=0, restored_price_values=0)
    torch.set_num_threads(4)
    for fold in e["fold_indices"]:
        dest = root / f"fold-{fold:02d}"
        parent = Path(e["prior"]) / f"fold-{fold:02d}"
        feature = Path(e["features"]) / f"fold-{fold:02d}"
        scalers = load_arrays(dest / "scalers.npz")
        original = load_arrays(Path(e["baseline"]) / f"fold-{fold:02d}/scalers.npz")
        for key in ["mean", "scale", "target_center", "target_scale", "prior"]:
            np.testing.assert_array_equal(scalers[key], original[key])
        train = np.load(feature / "train-context.npy").astype(float)
        means = []
        stds = []
        numbers = []
        for col in train.T:
            valid = col[np.isfinite(col)]
            numbers.append(len(valid))
            means.append(valid.mean() if len(valid) else 0.0)
            stds.append(max(valid.std() if len(valid) else 1.0, 1e-4))
        np.testing.assert_allclose(
            scalers["context_mean"], np.array(means, dtype=np.float32), rtol=2e-6, atol=1e-7
        )
        np.testing.assert_allclose(
            scalers["context_scale"], np.array(stds, dtype=np.float32), rtol=2e-6, atol=1e-7
        )
        np.testing.assert_array_equal(scalers["context_count"], numbers)
        counts["context_scaler_values"] += len(means) * 3
        saved = torch.load(dest / "transformer.pt", map_location="cpu", weights_only=True)
        model = PlanTransformer(
            scalers["target_center"],
            scalers["target_scale"],
            scalers["prior"],
            cfg["width"],
            cfg["context_features"],
        )
        model.load_state_dict(saved["state_dict"], strict=True)
        shape = read(dest / "training-shape.json")
        assert shape["device"] == "mps" and shape["parameter_device"] == "mps:0"
        assert shape["parameters"] == sum(p.numel() for p in model.parameters())
        log = read(dest / "training-log.json")
        result = read(dest / "training-result.json")
        best = float("inf")
        selected = 0
        for item in log:
            if item["selection_net_mse"] < best - cfg["min_delta"]:
                best = item["selection_net_mse"]
                selected = item["epoch"]
        assert selected == saved["selected_epoch"] == result["selected_epoch"]
        assert (
            result["epochs_completed"] == len(log)
            and cfg["min_epochs"] <= len(log) <= cfg["max_epochs"]
        )
        np.testing.assert_allclose(best, saved["selection_net_mse"], atol=1e-15, rtol=0)
        for part in ["calibration", "evaluation"]:
            rows = pd.read_parquet(parent / f"{part}-rows.parquet")
            assert rows.label_end.max() < cfg["sealed_holdout_start"]
            context = np.load(feature / f"{part}-context.npy")
            sample = np.unique(np.linspace(0, len(rows) - 1, 48, dtype=int))
            raw = load_arrays(
                dest
                / (
                    "raw-calibration.npz"
                    if part == "calibration"
                    else "context_transformer_raw-evaluation.npz"
                )
            )
            restored, prices = predict(
                model,
                values,
                rows.iloc[sample],
                scalers["mean"],
                scalers["scale"],
                context=context[sample],
                context_mean=scalers["context_mean"],
                context_scale=scalers["context_scale"],
            )
            for head in HEADS:
                np.testing.assert_allclose(restored[head], raw[head][sample], rtol=5e-4, atol=5e-6)
                counts["restored_plan_values"] += restored[head].size
            np.testing.assert_allclose(
                prices, np.load(dest / f"price-{part}.npy")[sample], rtol=5e-4, atol=5e-6
            )
            counts["restored_price_values"] += prices.size
        print("CPU checkpoint and scaler verified", fold, flush=True)
    write_json(
        root / "verification-torch.json", dict(passed=True, counts=counts, verified_at=utc_now())
    )


def metric_stage(root):
    from dataclasses import replace

    from verify_category_ablation import verify_metrics
    from verify_industry_decisions import verify_decisions
    from verify_plan_prequential import (
        calibrator_check,
        verify_comparisons,
        verify_decision_assembly,
    )

    from quant_research.plan_targets import plan_targets
    from quant_research.plan_value import head_targets
    from quant_research.price_strategy import TradeAssumptions

    e = read(root / "experiment.json")
    counts = dict(calibrators=0, head_metrics=0, chosen_rows=0, cost_dates=0)
    reports = []
    for fold in e["fold_indices"]:
        dest = root / f"fold-{fold:02d}"
        parent = Path(e["prior"]) / f"fold-{fold:02d}"
        data = load_arrays(parent / "evaluation.npz")
        cal = load_arrays(parent / "calibration.npz")
        rows = pd.read_parquet(parent / "evaluation-rows.parquet")
        cal_rows = pd.read_parquet(parent / "calibration-rows.parquet")
        raw = load_arrays(dest / "context_transformer_raw-evaluation.npz")
        corrected = load_arrays(dest / "context_transformer-evaluation.npz")
        cal_raw = load_arrays(dest / "raw-calibration.npz")
        counts["calibrators"] += calibrator_check(
            cal_raw,
            head_targets(plan_targets(cal)),
            cal_rows.date.to_numpy(),
            read(dest / "calibration.json"),
            raw,
            corrected,
        )
        base = plan_targets(data)
        a = TradeAssumptions()
        stress = plan_targets(
            data,
            replace(
                a,
                commission=a.commission * 2,
                minimum_fee=a.minimum_fee * 2,
                sell_tax=a.sell_tax * 2,
                slippage_bps=a.slippage_bps * 2,
            ),
        )
        forecasts = dict(context_transformer_raw=raw, context_transformer=corrected)
        counts["head_metrics"] += verify_metrics(
            rows, head_targets(base), forecasts, pd.read_csv(dest / "head-metrics.csv")
        )
        for name, pred in forecasts.items():
            daily = pd.read_csv(dest / f"{name}-decision-metrics.csv")
            reports.append(daily)
            n, dates = verify_decisions(
                rows, pred, base, stress, pd.read_parquet(dest / f"{name}-chosen.parquet"), daily
            )
            counts["chosen_rows"] += n
            counts["cost_dates"] += dates
        print("Calibration, losses and cost choices verified", fold, flush=True)
    verify_decision_assembly(reports, pd.read_csv(root / "decision-metrics.csv"))
    verify_comparisons(pd.read_csv(root / "head-metrics.csv"), read(root / "assessment.json"))
    write_json(
        root / "verification-metrics.json", dict(passed=True, counts=counts, verified_at=utc_now())
    )


def main(args):
    root = args.output.resolve()
    if args.stage == "torch":
        return torch_stage(root)
    if args.stage == "metrics":
        return metric_stage(root)
    if (root / "verification.json").exists():
        raise FileExistsError("Preserve previous verification")
    state = read(root / "run-status.json")
    e = read(root / "experiment.json")
    assert state["status"] == "completed"
    verify_files(root, state["files"])
    hashes = len(state["files"])
    for key, hashkey in [
        ("prior", "parent_status_sha256"),
        ("features", "feature_status_sha256"),
        ("baseline", "baseline_status_sha256"),
    ]:
        path = Path(e[key])
        assert file_hash(path / "run-status.json") == e[hashkey]
        files = read(path / "run-status.json")["files"]
        verify_files(path, files)
        hashes += len(files)
    panel = Path(e["panel"])
    assert file_hash(panel / "manifest.json") == e["panel_manifest_sha256"]
    verify_files(panel, read(panel / "manifest.json")["files"])
    for name, h in read(root / "context-manifest.json").items():
        assert file_hash(Path(e["features"]) / name) == h
    for stage in ["torch", "metrics"]:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--output",
                str(root),
                "--stage",
                stage,
            ],
            env=os.environ.copy(),
            check=True,
        )
    verify_files(root, state["files"])
    write_json(
        root / "verification.json",
        dict(
            passed=True,
            verified_at=utc_now(),
            hashes=hashes,
            torch=read(root / "verification-torch.json"),
            metrics=read(root / "verification-metrics.json"),
            scope="Same cohorts and feature hashes; baseline-compatible temporal/target scalers; train-only context scaler; CPU checkpoint samples; all calibrators, daily losses, choices/cost stress and paired block statistics. Retrospective sources and three windows do not certify profitability.",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stage", choices=["all", "torch", "metrics"], default="all")
    main(p.parse_args())
