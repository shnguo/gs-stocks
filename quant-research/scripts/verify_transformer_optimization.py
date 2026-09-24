"""Independent selection lock, checkpoint diagnostics and matched-grid audit."""

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


def balanced_mse(pred, target, dates):
    means = []
    for plan in range(12):
        daily = []
        for date in np.unique(dates):
            ids = (dates == date) & np.isfinite(target[:, plan])
            if ids.any():
                daily.append(np.mean((pred[ids, plan] - target[ids, plan]) ** 2))
        means.append(np.mean(daily))
    return float(np.mean(means))


def torch_audit(root):
    import torch
    from plan_value_run import compact_targets

    from quant_research.plan_transformer import PlanTransformer, predict
    from quant_research.plan_value import HEADS, head_targets

    torch.set_num_threads(4)
    trials = sorted((root / "trials").iterdir())
    records = []
    for trial in trials:
        cfg, e = read(trial / "protocol.json"), read(trial / "experiment.json")
        fold = f"fold-{e['fold_indices'][0]:02d}"
        dest, source = trial / fold, Path(e["prior"]) / fold
        verify_files(dest, read(dest / "trained.json")["files"])
        scalers = load_arrays(dest / "scalers.npz")
        train_rows = pd.read_parquet(source / "train-rows.parquet")
        rows = pd.read_parquet(source / "selection-rows.parquet")
        target = head_targets(compact_targets(load_arrays(source / "selection.npz")))[
            HEADS[2]
        ].astype(float)
        train_target = head_targets(compact_targets(load_arrays(source / "train.npz")))[
            HEADS[2]
        ].astype(float)
        values = np.load(Path(e["panel"]) / "values.npy", mmap_mode="r")
        context = np.load(Path(e["features"]) / fold / "selection-context.npy")
        train_context = np.load(Path(e["features"]) / fold / "train-context.npy")
        initial, log = read(dest / "initial-diagnostics.json"), read(dest / "training-log.json")
        ids = np.array(initial["probe_ids"], int)
        np.testing.assert_array_equal(
            ids,
            np.unique(np.linspace(0, len(train_rows) - 1, min(1024, len(train_rows)), dtype=int)),
        )
        probe_weights = np.zeros((len(ids), 12))
        for plan in range(12):
            good = np.isfinite(train_target[:, plan])
            dates, counts = np.unique(train_rows.date.to_numpy()[good], return_counts=True)
            w = {d: len(train_rows) / len(dates) / n for d, n in zip(dates, counts)}
            probe_weights[:, plan] = [
                w[d] if good[i] else 0 for i, d in zip(ids, train_rows.date.to_numpy()[ids])
            ]
        result = read(dest / "training-result.json")
        assert cfg["min_epochs"] <= len(log) <= cfg["max_epochs"]
        best, best_epoch, bad = float("inf"), 0, 0
        for row in log:
            if row["selection_net_mse"] < best - cfg["min_delta"]:
                best, best_epoch, bad = row["selection_net_mse"], row["epoch"], 0
            else:
                bad += 1
            expected = (
                np.mean(list(row["training_head_losses"].values()))
                + cfg["price_auxiliary_weight"] * row["training_price_pinball"] / 0.05
            )
            np.testing.assert_allclose(row["training_total_objective"], expected, rtol=2e-6)
            np.testing.assert_allclose(
                row["selection_net_mse"], np.mean(row["selection_plan_net_mse"]), rtol=2e-6
            )
        assert result["selected_epoch"] == best_epoch
        assert result["budget_limited"] == (len(log) == cfg["max_epochs"] and bad < cfg["patience"])
        constant = np.broadcast_to(scalers["prior"][:, 2], target.shape)
        np.testing.assert_allclose(
            balanced_mse(constant, target, rows.date.to_numpy()),
            initial["selection_constant_prior_net_mse"],
            rtol=2e-6,
        )
        checks = {}
        for name, saved, expected in [
            ("initial", None, initial),
            ("best", "transformer.pt", log[best_epoch - 1]),
            ("last", "last-checkpoint.pt", log[-1]),
        ]:
            torch.manual_seed(cfg["seed"])
            model = PlanTransformer(
                scalers["target_center"],
                scalers["target_scale"],
                scalers["prior"],
                cfg["width"],
                cfg["context_features"],
                cfg["dropout"],
            )
            if saved:
                state = torch.load(dest / saved, map_location="cpu", weights_only=False)
                model.load_state_dict(state["state_dict"], strict=True)
                if name == "last":
                    assert (
                        state["optimizer"]["param_groups"][0]["weight_decay"] == cfg["weight_decay"]
                    )
            pred, _ = predict(
                model,
                values,
                rows,
                scalers["mean"],
                scalers["scale"],
                context=context,
                context_mean=scalers["context_mean"],
                context_scale=scalers["context_scale"],
            )
            mse = balanced_mse(pred[HEADS[2]], target, rows.date.to_numpy())
            np.testing.assert_allclose(mse, expected["selection_net_mse"], rtol=1e-5, atol=1e-9)
            probe, _ = predict(
                model,
                values,
                train_rows.iloc[ids],
                scalers["mean"],
                scalers["scale"],
                context=train_context[ids],
                context_mean=scalers["context_mean"],
                context_scale=scalers["context_scale"],
            )
            delta = np.where(np.isfinite(train_target[ids]), probe[HEADS[2]] - train_target[ids], 0)
            probe_mse = float(np.mean(delta**2 * probe_weights))
            np.testing.assert_allclose(
                probe_mse, expected["training_probe_net_mse"], rtol=1e-5, atol=1e-9
            )
            checks[name] = dict(selection_mse=mse, training_probe_mse=probe_mse)
        records.append(
            dict(
                trial=trial.name,
                checks=checks,
                best_epoch=best_epoch,
                epochs=len(log),
                budget_limited=result["budget_limited"],
            )
        )
        print("Checkpoint diagnostics verified", trial.name, flush=True)
    write_json(
        root / "verification-diagnostics.json", dict(passed=True, trials=records, at=utc_now())
    )


def main(args):
    root = args.output.resolve()
    if args.stage == "torch":
        return torch_audit(root)
    if (root / "verification.json").exists():
        raise FileExistsError("Preserve verification")
    p, lock, state = (
        read(root / "protocol.json"),
        read(root / "selection-lock.json"),
        read(root / "run-status.json"),
    )
    assert state["status"] == "completed"
    verify_files(root, state["files"])
    verify_files(root, read(root / "frozen-manifest.json"))
    verify_files(root, lock["screen_files"])
    scores = {}
    for variant in p["variants"]:
        trial = root / "trials" / f"{variant}-s17-f07"
        scores[variant] = read(trial / "fold-07/training-result.json")["selection_net_mse"]
    assert scores == lock["selection_scores"]
    assert lock["chosen"] == sorted((v, k) for k, v in scores.items() if k != "reference")[0][1]
    assert lock["no_evaluation_forecasts_present"]
    expected = {(v, s, 7) for v in ["reference", lock["chosen"]] for s in [17, 29, 43]} | {
        (lock["chosen"], 17, f) for f in [1, 15]
    }
    completed = set()
    for trial in (root / "trials").iterdir():
        e = read(trial / "experiment.json")
        cfg = read(trial / "protocol.json")
        for key, value in p["variants"][e["optimization_variant"]].items():
            assert cfg[key] == value
        if (trial / "verification.json").exists():
            assert read(trial / "verification.json")["passed"]
            assert read(trial / "run-status.json")["finished_at"] > lock["selected_at"]
            completed.add((e["optimization_variant"], e["optimization_seed"], e["fold_indices"][0]))
            verify_files(trial, read(trial / "run-status.json")["files"])
        else:
            assert not list(trial.glob("fold-*/*evaluation.npz"))
    assert completed == expected
    from verify_plan_prequential import verify_comparisons

    data = pd.read_csv(root / "head-metrics.csv")
    pairs = []
    for suffix in ["_raw", ""]:
        frame = data[data.seed.eq(17) & data.model.eq("context_transformer" + suffix)].copy()
        frame["model"] = frame.variant.map(
            {"reference": "reference" + suffix, lock["chosen"]: "challenger" + suffix}
        )
        pairs.append(frame)
    verify_comparisons(pd.concat(pairs), read(root / "assessment.json"))
    for row in read(root / "seed-comparison.json"):
        frame = data[
            data.window.eq("fold-07")
            & data.seed.eq(row["seed"])
            & data.model.eq(row["model"])
            & data["head"].eq(row["head"])
        ]
        mean = frame.groupby("variant")[row["metric"]].mean()
        np.testing.assert_allclose(row["reference"], mean["reference"], rtol=1e-12)
        np.testing.assert_allclose(row["challenger"], mean[lock["chosen"]], rtol=1e-12)
        np.testing.assert_allclose(
            row["error_change_pct"],
            (mean[lock["chosen"]] / mean["reference"] - 1) * 100,
            rtol=1e-10,
            atol=1e-10,
        )
    parent = Path(p["parent"])
    assert file_hash(parent / "run-status.json") == p["parent_status_sha256"]
    verify_files(parent, read(parent / "run-status.json")["files"])
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--output", str(root), "--stage", "torch"],
        check=True,
        env=os.environ.copy(),
    )
    verify_files(root, state["files"])
    write_json(
        root / "verification.json",
        dict(
            passed=True,
            verified_at=utc_now(),
            hashes=len(state["files"]),
            evaluated_trials=len(expected),
            diagnostic_trials=len(list((root / "trials").iterdir())),
            scope="All per-trial calibrations, predictions and cost choices; selection-only configuration lock; CPU initial/best/last MSE; training-probe, optimizer and matched seed comparisons. Partial grid and reused development dates.",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["all", "torch"], default="all")
    main(parser.parse_args())
