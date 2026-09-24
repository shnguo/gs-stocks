"""Prepare, fine-tune, and compare Kronos on the accepted development experiment."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from quant_research.config import digest
from quant_research.development import accepted_schedule, read_panel
from quant_research.kronos_ranker import (
    FIELDS,
    batch_inputs,
    compare_predictions,
    load_pretrained,
    timestamps,
)
from quant_research.models import date_weights, rank_ic, seed_everything
from quant_research.storage import file_hash, utc_now, write_json


def emit(output, stage, **fields):
    event = {"stage": stage, "updated_at": utc_now(), **fields}
    write_json(output / "progress.json", event)
    with (output / "events.jsonl").open("a") as f:
        f.write(json.dumps(event) + "\n")
    print(json.dumps(event), flush=True)


def context(root, horizon, fold_index):
    acceptance = json.loads((root / "development-acceptance.json").read_text())
    fold = accepted_schedule(root, acceptance, fold_index)
    panel = read_panel(root, acceptance, horizon)
    parts = fold.select(panel.samples)
    holdout = json.loads((root / "schedule.json").read_text())["sealed_holdout_start"]
    if (parts["test"].label_end >= holdout).any():
        raise ValueError("Development labels cross the sealed holdout")
    return acceptance, fold, panel, parts


def register_trials(root, bundle, config):
    registered = json.loads((root / "registered-training-config.json").read_text())
    if (config["horizons"] != registered["horizons"] or config["seeds"] != registered["seeds"]
            or config["lookback"] != registered["lookback"]
            or config["adaptation"] != "pretrained_backbone_plus_supervised_rank_head_frozen_tokenizer"
            or not 0 < config["micro_batch_size"] <= config["batch_size"]
            or min(config["max_epochs"], config["patience"], config["log_every_steps"],
                   config["checkpoint_every_steps"]) <= 0):
        raise ValueError("Kronos configuration is outside the registered comparison")
    registry = root / "runs/trial-registry"
    registry.mkdir(exist_ok=True)
    for horizon in config["horizons"]:
        identity = {"model": "kronos_ranker", "config": config, "horizon": horizon,
                    "pretrained_provenance_sha256": file_hash(bundle / "pretrained-provenance.json")}
        path = registry / f"h{horizon}-{digest(identity)}.json"
        if not path.exists():
            if len(list(registry.glob(f"h{horizon}-*.json"))) >= registered["max_trials_per_horizon"]:
                raise ValueError("Registered per-horizon trial budget exhausted")
            with path.open("x") as f:
                json.dump(identity, f, indent=2)


def prepare(root, bundle, fold_index):
    output = bundle / "inputs"
    if output.exists():
        raise FileExistsError("Prepared inputs are immutable; use a new bundle")
    acceptance, fold, panel, parts = context(root, 5, fold_index)
    manifest = json.loads((root / "snapshot/manifest.json").read_text())
    bars_path = root / "snapshot/bars.parquet"
    if file_hash(bars_path) != manifest["files"]["bars.parquet"]:
        raise ValueError("Raw bars do not match the accepted snapshot")
    if manifest["dataset_id"] != panel.dataset_id:
        raise ValueError("Snapshot/panel dataset mismatch")
    output.mkdir()
    # Predicate excludes all raw bars after this development test window.
    dates = [day for day in panel.dates if day <= fold.test_end]
    frame = pd.read_parquet(bars_path, columns=["instrument_id", "date", *FIELDS,
                                               "sequence_id"],
                            filters=[("date", "<=", fold.test_end)])
    groups = frame.groupby("instrument_id", sort=False).indices
    values = np.lib.format.open_memmap(output / "values.npy", mode="w+", dtype=np.float32,
                                      shape=(len(panel.instruments), len(dates), len(FIELDS)))
    values[:] = np.nan
    mask_index = panel.feature_names.index("unpriced_suspension")
    for index, symbol in enumerate(panel.instruments):
        raw = frame.iloc[groups.get(symbol, [])].set_index("date").reindex(dates)
        # Only the accepted, evidenced suspension mask permits a past close carry.
        carry_mask = (panel.values[index, :len(dates), mask_index] == 1) & raw.close.isna()
        sequence = raw.sequence_id.ffill().fillna("")
        episodes = sequence.ne(sequence.shift()).cumsum()
        close = raw.close.groupby(episodes).ffill()
        factor = raw.factor.groupby(episodes).ffill()
        for field in FIELDS[:4]:
            raw.loc[carry_mask, field] = close.loc[carry_mask]
        raw.loc[carry_mask, "factor"] = factor.loc[carry_mask]
        raw.loc[carry_mask, ["volume", "amount"]] = 0.0
        values[index] = raw[FIELDS].to_numpy(dtype=np.float32)
        if (index + 1) % 500 == 0:
            emit(output, "preparing_inputs", instruments=index + 1,
                 total_instruments=len(panel.instruments))
    values.flush()
    # All candidate windows must exist: never silently shrink the baseline cohort.
    for name, rows in parts.items():
        for start in range(0, len(rows), 8192):
            coords = rows.iloc[start:start + 8192][["stock_index", "date_index"]].to_numpy(int)
            batch_inputs(values, timestamps(dates), coords, panel.lookback, "cpu")
        emit(output, "verified_partition", partition=name, rows=len(rows))
    write_json(output / "manifest.json", {
        "dataset_id": panel.dataset_id, "acceptance_sha256": file_hash(root / "development-acceptance.json"),
        "fold": fold.to_dict(), "fold_index": fold_index, "dates": dates,
        "instruments": panel.instruments, "lookback": panel.lookback,
        "fields": FIELDS, "values_sha256": file_hash(output / "values.npy"),
        "raw_bars_sha256": manifest["files"]["bars.parquet"],
        "mask_panel_manifest_sha256": acceptance["panels"]["5"],
        "adjustment": "OHLC times historical factor / signal-date factor; raw share/RMB turnover",
        "suspensions": "Only accepted unpriced_suspension mask; past close within price episode",
        "completed": True, "formal_ready": False})
    emit(output, "completed", formal_ready=False)


def read_inputs(root, bundle, panel, fold):
    m = json.loads((bundle / "inputs/manifest.json").read_text())
    if (not m["completed"] or m["dataset_id"] != panel.dataset_id
            or m["instruments"] != panel.instruments or m["lookback"] != panel.lookback
            or m["dates"] != [d for d in panel.dates if d <= fold.test_end]
            or m["fold"] != fold.to_dict()
            or m["acceptance_sha256"] != file_hash(root / "development-acceptance.json")
            or m["values_sha256"] != file_hash(bundle / "inputs/values.npy")):
        raise ValueError("Prepared inputs differ from the accepted experiment")
    return np.load(bundle / "inputs/values.npy", mmap_mode="r"), timestamps(m["dates"])


def predict(model, values, stamps, rows, lookback, batch_size, output, stage):
    model.eval()
    result = np.empty(len(rows), dtype=np.float32)
    coords = rows[["stock_index", "date_index"]].to_numpy(int)
    device = next(model.parameters()).device
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            x, stamp = batch_inputs(values, stamps, coords[start:start + batch_size], lookback, device)
            result[start:start + batch_size] = model(x, stamp).cpu().numpy()
            if start % (batch_size * 100) == 0:
                emit(output, stage, scored_rows=min(start + batch_size, len(rows)), total_rows=len(rows))
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite Kronos predictions")
    return result


def checkpoint(path, model, optimizer, **state):
    temporary = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), **state}, temporary)
    temporary.replace(path)


def train(root, bundle, config, horizon, seed):
    register_trials(root, bundle, config)
    output = bundle / "runs" / f"fold{config['fold_index']}-h{horizon}-seed{seed}"
    if output.exists():
        raise FileExistsError("Use a new immutable Kronos run directory")
    acceptance, fold, panel, parts = context(root, horizon, config["fold_index"])
    baseline = root / "runs" / output.name
    baseline_status = json.loads((baseline / "run.json").read_text())
    baseline_identity = baseline_status["identity"]
    if (baseline_status["status"] != "completed"
            or baseline_identity["dataset_id"] != panel.dataset_id
            or baseline_identity["fold"] != fold.to_dict()
            or baseline_identity["seed"] != seed or baseline_identity["horizon"] != horizon
            or baseline_identity["acceptance_sha256"] != file_hash(root / "development-acceptance.json")):
        raise ValueError("Matching LightGBM/Transformer run must complete first")
    values, stamps = read_inputs(root, bundle, panel, fold)
    output.mkdir(parents=True)
    identity = {"dataset_id": panel.dataset_id, "fold": fold.to_dict(), "horizon": horizon,
                "seed": seed, "config": config, "acceptance_sha256": file_hash(root / "development-acceptance.json"),
                "inputs_manifest_sha256": file_hash(bundle / "inputs/manifest.json"),
                "pretrained_provenance_sha256": file_hash(bundle / "pretrained-provenance.json"),
                "baseline_run_id": baseline_status["run_id"],
                "baseline_predictions_sha256": file_hash(baseline / "predictions.parquet"),
                "training_objective": "same date-weighted Huber cross-sectional holder-return rank",
                "code_manifest_sha256": file_hash(bundle / "code-manifest.json"),
                "environment_lock_sha256": file_hash(bundle / "environment.lock.txt"),
                "formal_ready": False, "pretraining_test_overlap": "unverified"}
    write_json(output / "run.json", {"identity": identity, "run_id": digest(identity),
                                      "status": "training", "limitations": acceptance["limitations"]})
    try:
        seed_everything(seed)
        model = load_pretrained(bundle, config["device"])
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"],
                                      weight_decay=config["weight_decay"])
        train_rows = parts["train"]
        coords = train_rows[["stock_index", "date_index"]].to_numpy(int)
        targets = train_rows.target.to_numpy(dtype=np.float32)
        weights = date_weights(train_rows)
        size = config["batch_size"]
        micro = config["micro_batch_size"]
        best, bad, log, steps = -float("inf"), 0, [], 0
        last_progress = time.monotonic()
        for epoch in range(1, config["max_epochs"] + 1):
            model.train()
            order = np.random.default_rng(seed + epoch).permutation(len(train_rows))
            total_loss = 0.0
            for start in range(0, len(order), size):
                indices = order[start:start + size]
                optimizer.zero_grad(set_to_none=True)
                step_loss = 0.0
                for offset in range(0, len(indices), micro):
                    ids = indices[offset:offset + micro]
                    x, stamp = batch_inputs(values, stamps, coords[ids], panel.lookback, config["device"])
                    target = torch.from_numpy(targets[ids]).to(config["device"])
                    weight = torch.from_numpy(weights[ids]).to(config["device"])
                    loss = (torch.nn.functional.huber_loss(model(x, stamp), target,
                            delta=0.1, reduction="none") * weight).sum() / len(indices)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite Kronos loss")
                    loss.backward()
                    step_loss += float(loss.detach())
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                steps += 1
                total_loss += step_loss * len(indices)
                if (start == 0 or steps % config["log_every_steps"] == 0
                        or time.monotonic() - last_progress >= 15):
                    emit(output, "training", epoch=epoch, optimizer_steps=steps, loss=step_loss,
                         rows_in_epoch=start + len(indices), train_rows=len(order), horizon=horizon, seed=seed)
                    last_progress = time.monotonic()
                if steps % config["checkpoint_every_steps"] == 0:
                    checkpoint(output / "latest.pt", model, optimizer, identity=identity,
                               epoch=epoch, rows_in_epoch=start + len(indices), optimizer_steps=steps)
            score = predict(model, values, stamps, parts["validation"], panel.lookback,
                            micro, output, "validation")
            ic = rank_ic(parts["validation"], score)["mean_rank_ic"]
            if ic is None:
                raise ValueError("Validation Rank IC is undefined")
            log.append({"epoch": epoch, "train_loss": total_loss / len(order), "validation_rank_ic": ic})
            write_json(output / "training-log.json", log)
            checkpoint(output / "latest.pt", model, optimizer, identity=identity,
                       epoch=epoch, rows_in_epoch=len(order), optimizer_steps=steps)
            if ic > best:
                best, bad = ic, 0
                checkpoint(output / "best.pt", model, optimizer, identity=identity,
                           epoch=epoch, validation_rank_ic=ic)
            else:
                bad += 1
            emit(output, "epoch_completed", **log[-1], best_validation_rank_ic=best)
            if bad >= config["patience"]:
                break
        # Only now access test scores. Selection used validation exclusively.
        model.load_state_dict(torch.load(output / "best.pt", map_location="cpu", weights_only=True)["model"])
        predictions = parts["test"].copy()
        predictions["kronos_score"] = predict(model, values, stamps, predictions,
                                               panel.lookback, micro, output, "test_scoring")
        predictions.to_parquet(output / "kronos-predictions.parquet", index=False)
        if file_hash(baseline / "predictions.parquet") != identity["baseline_predictions_sha256"]:
            raise ValueError("Baseline predictions changed during fine-tuning")
        merged, metrics = compare_predictions(pd.read_parquet(baseline / "predictions.parquet"), predictions)
        merged.to_parquet(output / "comparison-predictions.parquet", index=False)
        write_json(output / "comparison.json", metrics)
        status = json.loads((output / "run.json").read_text())
        status.update(status="completed", updated_at=utc_now())
        write_json(output / "run.json", status)
        emit(output, "completed", formal_ready=False)
    except BaseException as error:
        status = json.loads((output / "run.json").read_text())
        status.update(status="failed", error=str(error)[:500], updated_at=utc_now())
        write_json(output / "run.json", status)
        emit(output, "failed", error=str(error)[:500])
        raise


def queue(root, bundle, config):
    register_trials(root, bundle, config)
    queue_path = bundle / "queue.json"
    # An exclusive lock prevents accidental second GPU consumers.
    with (bundle / "queue.lock").open("x") as f:
        f.write(str(os.getpid()))
    try:
        source = json.loads((root / "training-queue.json").read_text())
        print(json.dumps({"status": "waiting_for_baseline_queue", "waiting_for_pid": source["pid"],
                          "jobs": [[h, s] for h in config["horizons"] for s in config["seeds"]],
                          "updated_at": utc_now()}), flush=True)
        while True:
            current = json.loads((root / "training-queue-progress.json").read_text())
            if current["status"] == "completed":
                break
            if current["status"] == "failed":
                raise RuntimeError("Existing baseline queue failed; inspect it before continuing")
            os.kill(source["pid"], 0)
            write_json(queue_path, {"status": "waiting_for_baseline_queue", "pid": os.getpid(),
                                    "waiting_for_pid": source["pid"], "updated_at": utc_now()})
            time.sleep(30)
        # Frozen copies prevent another task's source edits changing a later job.
        for horizon in config["horizons"]:
            for seed in config["seeds"]:
                for name, sha in json.loads((bundle / "code-manifest.json").read_text()).items():
                    if file_hash(bundle / name) != sha:
                        raise ValueError("Frozen experiment source changed")
                write_json(queue_path, {"status": "running", "pid": os.getpid(),
                                        "horizon": horizon, "seed": seed, "updated_at": utc_now()})
                train(root, bundle, config, horizon, seed)
                summary = []
                for path in sorted((bundle / "runs").glob("*/comparison.json")):
                    result = json.loads(path.read_text())
                    run = json.loads((path.parent / "run.json").read_text())["identity"]
                    for name in ["lightgbm", "transformer", "kronos"]:
                        summary.append({"horizon": run["horizon"], "seed": run["seed"],
                                        "model": name, "mean_rank_ic": result[name]["mean_rank_ic"],
                                        "daily_rank_ic_std": result[name]["rank_ic_std"],
                                        "scored_rows": result["scored_rows"],
                                        "labelled_rows": result["labelled_rows"], "formal_ready": False})
                write_json(bundle / "comparison-summary.json", {
                    "results": summary, "pretraining_test_overlap": "unverified",
                    "formal_ready": False})
        write_json(queue_path, {"status": "completed", "updated_at": utc_now(), "formal_ready": False})
    except BaseException as error:
        write_json(queue_path, {"status": "failed", "error": str(error)[:500], "updated_at": utc_now()})
        raise
    finally:
        (bundle / "queue.lock").unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "train", "queue"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--horizon", type=int, choices=[5, 20], default=5)
    parser.add_argument("--seed", type=int, choices=[17, 29, 43], default=17)
    args = parser.parse_args()
    root, bundle = args.root.resolve(), args.bundle.resolve()
    config = json.loads((bundle / "config.json").read_text())
    if args.action == "prepare":
        prepare(root, bundle, config["fold_index"])
    elif args.action == "train":
        train(root, bundle, config, args.horizon, args.seed)
    else:
        queue(root, bundle, config)


if __name__ == "__main__":
    main()
