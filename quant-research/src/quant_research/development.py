"""Evidence-bound reconstructed-history training, separate from formal evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import digest
from .features import Panel, Standardizer
from .models import (
    date_weights,
    predict_transformer,
    rank_ic,
    save_transformer,
    tabular_values,
    train_transformer,
)
from .splits import Fold
from .storage import file_hash, utc_now, write_json
from .tree_runner import isolated_tree


def read_panel(root: Path, acceptance: dict, horizon: int) -> Panel:
    path = root / f"panel-h{horizon}"
    if file_hash(path / "manifest.json") != acceptance["panels"][str(horizon)]:
        raise ValueError("Unaccepted panel manifest")
    m = json.loads((path / "manifest.json").read_text())
    if m["dataset_id"] != acceptance["dataset_id"] or m["horizon"] != horizon or not m["completed"]:
        raise ValueError("Panel identity mismatch")
    for name, expected in m["files"].items():
        if Path(name).name != name or file_hash(path / name) != expected:
            raise ValueError("Panel artifact hash mismatch")
    return Panel(np.load(path / "values.npy", mmap_mode="r", allow_pickle=False), m["dates"],
                 m["instruments"], m["feature_names"], pd.read_parquet(path / "samples.parquet"),
                 pd.DataFrame(), m["lookback"], horizon, m["dataset_id"], "include")


def accepted_schedule(root: Path, acceptance: dict, fold_index: int) -> Fold:
    if (acceptance.get("mode") != "reconstructed_history_development_training"
            or acceptance.get("passed") is not True or acceptance.get("formal_ready") is not False
            or acceptance.get("universe_scope") != "all_a_shares"):
        raise ValueError("Development acceptance is missing or invalid")
    for name, sha in acceptance["evidence"].items():
        if file_hash(root / name) != sha:
            raise ValueError("Development acceptance evidence changed")
    schedule = json.loads((root / "schedule.json").read_text())
    fold = Fold(**schedule["development"][fold_index])
    if fold.purpose != "development" or fold.test_end >= schedule["sealed_holdout_start"]:
        raise ValueError("Final holdout cannot be inspected")
    return fold


def run_development(root: Path, config: dict, horizon: int, seed: int,
                    output: Path, fold_index: int = 0) -> None:
    if output.exists():
        raise FileExistsError("Use a new immutable run directory")
    if (horizon not in config["horizons"] or seed not in config["seeds"]
            or config["training_risk_policy"] != "include"):
        raise ValueError("Run is outside the registered full-universe configuration")
    acceptance_path = root / "development-acceptance.json"
    acceptance = json.loads(acceptance_path.read_text())
    fold = accepted_schedule(root, acceptance, fold_index)
    panel = read_panel(root, acceptance, horizon)
    if len(panel.instruments) != acceptance["candidate_instruments"]:
        raise ValueError("Full instrument identities were not retained")
    splits = fold.select(panel.samples)
    holdout_start = json.loads((root / "schedule.json").read_text())["sealed_holdout_start"]
    if (splits["test"].label_end >= holdout_start).any():
        raise ValueError("Test labels would inspect the final holdout; purge this fold first")
    identity = {"dataset_id": panel.dataset_id, "acceptance_sha256": file_hash(acceptance_path),
                "config": config, "horizon": horizon, "seed": seed, "fold": fold.to_dict(),
                "mode": acceptance["mode"], "universe_scope": "all_a_shares", "formal_ready": False,
                "software_sha256": digest({p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")})}
    registry = output.parent / "trial-registry"
    registry.mkdir(parents=True, exist_ok=True)
    key = digest({"dataset_id": panel.dataset_id, "model": config["model"], "horizon": horizon})
    trial = registry / f"h{horizon}-{key}.json"
    if not trial.exists():
        if len(list(registry.glob(f"h{horizon}-*.json"))) >= config["max_trials_per_horizon"]:
            raise ValueError("Trial budget exhausted")
        with trial.open("x") as f:
            json.dump(identity, f)
    output.mkdir()
    status = {"run_id": digest(identity), "identity": identity, "status": "preparing",
              "created_at": utc_now(), "partition_rows": {k: len(v) for k, v in splits.items()},
              "limitations": acceptance["limitations"], "test_label_coverage":
              splits["test"].label_status.value_counts().to_dict()}
    write_json(output / "run.json", status)
    try:
        scaler = Standardizer.fit(panel, splits["train"])
        write_json(output / "scaler.json", {"mean": scaler.mean.tolist(), "scale": scaler.scale.tolist(),
                                            "fit_partition": "train_only"})
        test = splits["test"]
        status.update(status="training_baseline", updated_at=utc_now())
        write_json(output / "run.json", status)
        scores = isolated_tree(tabular_values(panel, splits["train"], scaler),
            splits["train"].target.to_numpy(), tabular_values(panel, splits["validation"], scaler),
            splits["validation"].target.to_numpy(), tabular_values(panel, test, scaler), seed,
            output / "lightgbm.txt", date_weights(splits["train"]), date_weights(splits["validation"]),
            splits["validation"].date.to_numpy(dtype="U10"))
        momentum = panel.values[test.stock_index.to_numpy(dtype=int), test.date_index.to_numpy(dtype=int),
                                panel.feature_names.index("return_20")]
        metrics = {"lightgbm": rank_ic(test, scores), "momentum": rank_ic(test, momentum)}
        predictions = test.copy()
        predictions["lightgbm_score"] = scores
        predictions["momentum_score"] = momentum
        predictions.to_parquet(output / "baseline-predictions.parquet", index=False)
        write_json(output / "baseline-metrics.json", metrics)
        status.update(status="training_transformer", updated_at=utc_now())
        write_json(output / "run.json", status)
        model, log = train_transformer(panel, splits["train"], splits["validation"], scaler,
                                      config["model"], seed, progress_dir=output / "checkpoint")
        save_transformer(output / "transformer.pt", model, panel, scaler, config["model"], seed)
        write_json(output / "training-log.json", log)
        predictions["transformer_score"] = predict_transformer(model, panel, test, scaler,
                                                                 config["model"]["batch_size"])
        predictions.to_parquet(output / "predictions.parquet", index=False)
        metrics["transformer"] = rank_ic(test, predictions.transformer_score.to_numpy())
        write_json(output / "metrics.json", metrics)
        status.update(status="completed", updated_at=utc_now(), formal_ready=False)
        write_json(output / "run.json", status)
        write_json(output / "checkpoint/progress.json", {"status": "completed", "updated_at": utc_now(),
                                                          "epochs": len(log), "formal_ready": False})
    except BaseException as error:
        status.update(status="failed", error=str(error)[:500], updated_at=utc_now())
        write_json(output / "run.json", status)
        raise
