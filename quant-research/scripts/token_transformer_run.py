"""Prepare, train, sample and verify a small Kronos-token decoder, without plan labels."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from quant_research.kronos_ranker import timestamps
from quant_research.storage import file_hash, utc_now, write_json
from quant_research.token_transformer import (
    FIELDS,
    TokenConfig,
    TokenTransformer,
    checkpoint_payload,
    decode_paths,
    generate_tokens,
    load_tokenizer,
    normalize_training,
    restore_model,
    token_loss,
    valid_bars,
)


def read(path):
    return json.loads(path.read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def verify_files(root, files):
    for name, expected in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or file_hash(path) != expected:
            raise ValueError(f"Artifact hash mismatch: {name}")


def hashes(root, paths):
    return {str(p.relative_to(root)): file_hash(p) for p in paths}


def validate_protocol(config):
    model = TokenConfig(**config["model"])
    if (not 2 <= config["lookback"] <= model.max_context
            or config["horizon"] != 5 or config["lookback"] + config["horizon"] > model.max_context
            or not 1 <= config["min_epochs"] <= config["max_epochs"]
            or config["patience"] < 1 or config["batch_size"] < 1
            or config["path_samples"] < 8 or config["rows_per_date"] < 0
            or config["research_start"] <= "2024-06-30"
            or config["sealed_holdout_start"] > "2025-08-07"):
        raise ValueError("Invalid token research protocol or temporal boundary")
    for key in ["learning_rate", "clip", "temperature"]:
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if not 0 < config["top_p"] <= 1 or config["top_k"] < 0:
        raise ValueError("Invalid sampling settings")
    return model


def select_rows(rows, config, part):
    if rows.duplicated(["instrument_id", "date"]).any():
        raise ValueError("Duplicate source rows")
    rows = rows.loc[rows.date.ge(config["research_start"])].copy()
    dates = sorted(rows.date.unique())
    count = config["max_dates"].get(part, 0)
    if count < 0:
        raise ValueError("Negative date budget")
    if count and count < len(dates):
        dates = [dates[i] for i in np.linspace(0, len(dates)-1, count, dtype=int)]
    rows = rows.loc[rows.date.isin(dates)].copy()
    rows["source_row"] = rows.index
    # Choose identities before reading any future targets or input availability.
    rows["sample_key"] = [hashlib.sha256(f'{config["seed"]}:{s}'.encode()).hexdigest()
                          for s in rows.instrument_id]
    rows = rows.sort_values(["date", "sample_key"])
    if config["rows_per_date"]:
        rows = rows.groupby("date", sort=False).head(config["rows_per_date"])
    return rows.drop(columns="sample_key").reset_index(drop=True)


def prepare(args):
    out, prior, inputs, bundle = [p.resolve() for p in [args.output, args.prior, args.inputs, args.bundle]]
    config = read(args.config)
    model_config = validate_protocol(config)
    status, input_manifest = read(prior / "run-status.json"), read(inputs / "manifest.json")
    if status["status"] != "completed" or not read(prior / "verification.json")["passed"]:
        raise ValueError("Unverified parent price dataset")
    fold = prior / f'fold-{config["fold"]:02d}'
    source_config = read(fold / "config.json")
    if (source_config["dataset_id"] != input_manifest["dataset_id"]
            or input_manifest["fields"] != [*FIELDS, "factor"]
            or not input_manifest["completed"]):
        raise ValueError("Raw input dataset/fields mismatch")
    verify_files(inputs, {"values.npy": input_manifest["values_sha256"]})
    sources = [fold / "config.json"] + [fold / f"{p}{suffix}" for p in
               ["train", "selection", "evaluation"] for suffix in ["-rows.parquet", ".npz"]]
    for path in sources:
        key = str(path.relative_to(prior))
        verify_files(prior, {key: status["files"][key]})
    rows = {p: select_rows(pd.read_parquet(fold / f"{p}-rows.parquet"), config, p)
            for p in ["train", "selection", "evaluation"]}
    dates = input_manifest["dates"]
    if dates != sorted(set(dates)):
        raise ValueError("Invalid raw calendar")
    date_index = {d: i for i, d in enumerate(dates)}
    instruments = {s: i for i, s in enumerate(input_manifest["instruments"])}
    for part, frame in rows.items():
        if frame.empty or frame.label_end.max() >= config["sealed_holdout_start"]:
            raise ValueError(f"Empty partition or sealed horizon: {part}")
        for row in frame.itertuples():
            t = date_index.get(row.date, -1)
            if (t < config["lookback"] - 1 or t + config["horizon"] >= len(dates)
                    or dates[t + config["horizon"]] != row.label_end
                    or instruments.get(row.instrument_id) != row.stock_index
                    or t != row.date_index):
                raise ValueError("Source identity/calendar/history alignment differs")
    if (rows["train"].label_end.max() >= rows["selection"].date.min()
            or rows["selection"].label_end.max() >= rows["evaluation"].date.min()):
        raise ValueError("Partitions overlap through future target horizon")
    tokenizer = load_tokenizer(bundle, config["device"])
    if (tokenizer.s1_bits != model_config.s1_bits or tokenizer.s2_bits != model_config.s2_bits):
        raise ValueError("Tokenizer and decoder vocabularies differ")
    out.mkdir()
    write_json(out / "protocol.json", config)
    write_json(out / "experiment.json", {
        "created_at": utc_now(), "prior": str(prior), "inputs": str(inputs), "bundle": str(bundle),
        "dataset_id": input_manifest["dataset_id"], "source_files": hashes(prior, sources),
        "parent_verification_sha256": file_hash(prior / "verification.json"),
        "input_manifest_sha256": file_hash(inputs / "manifest.json"),
        "tokenizer_provenance_sha256": file_hash(bundle / "pretrained-provenance.json"),
        "backbone_initialization": "random; no pretrained predictor weights",
        "tokenizer": "frozen official Kronos tokenizer; exact pretraining dates unverified",
        "cohort": "Inherited price-data stock/date rows; deterministic optional input-only subsample",
        "executable": False, "scope": config["scope"],
    })
    base = Path(__file__).resolve().parents[1]
    shutil.copytree(base / "src", out / "code/src", ignore=shutil.ignore_patterns("__pycache__"))
    (out / "code/scripts").mkdir()
    shutil.copy2(Path(__file__), out / "code/scripts/token_transformer_run.py")
    shutil.copy2(base / "uv.lock", out / "code/uv.lock")
    # The tokenizer runs in the existing isolated Kronos runtime, not the base lock.
    write_json(out / "runtime.json", {"torch": str(torch.__version__), "numpy": np.__version__,
        "device": config["device"], "python": sys.executable, "python_version": sys.version,
        "tokenizer_dependencies": {k: importlib.metadata.version(k)
                                   for k in ["safetensors", "einops", "huggingface_hub"]}})
    values = np.load(inputs / "values.npy", mmap_mode="r")
    stamps = timestamps(dates).astype(np.int64)
    length, horizon, batch = config["lookback"], config["horizon"], config["batch_size"]
    coverage = {}
    for part, frame in rows.items():
        targets = arrays(fold / f"{part}.npz")
        chunks, accepted, ledger = [], [], []
        for start in range(0, len(frame), batch):
            selected = frame.iloc[start:start+batch]
            s, t = selected[["stock_index", "date_index"]].to_numpy(int).T
            history_times = t[:, None] + np.arange(-length+1, 1)
            raw = np.array(values[s[:, None], history_times])
            available = (valid_bars(raw[..., :6]).all(1)
                         & np.isfinite(raw[..., 6]).all(1) & (raw[..., 6] > 0).all(1))
            ledger.append(selected.assign(input_available=available))
            if not available.any():
                continue
            selected, raw, t = selected.loc[available], raw[available], t[available]
            ids = selected.source_row.to_numpy(int)
            future = np.concatenate([targets["future"][ids], targets["turnover"][ids]], -1)
            normalized, valid, mean, scale = normalize_training(raw, future, targets["valid"][ids], config["clip"])
            with torch.inference_mode():
                a, b = tokenizer.encode(torch.from_numpy(normalized).to(config["device"]), half=True)
            times = t[:, None] + np.arange(-length+1, horizon+1)
            chunks.append(dict(s1=a.cpu().numpy(), s2=b.cpu().numpy(), stamps=stamps[times],
                               valid=valid, mean=mean, scale=scale, future=future,
                               last=raw[:, -1, :6], normalized=normalized))
            accepted.append(selected)
            if start % (batch * 50) == 0:
                write_json(out / "progress.json", dict(stage="encoding", partition=part,
                    processed=min(start+batch, len(frame)), total=len(frame), updated_at=utc_now()))
                print("Encoding", part, min(start+batch, len(frame)), "/", len(frame), flush=True)
        if not chunks:
            raise ValueError(f"No usable histories: {part}")
        combined = {key: np.concatenate([c[key] for c in chunks]) for key in chunks[0]}
        if part != "evaluation" and not combined["valid"].any():
            raise ValueError(f"No known targets: {part}")
        np.savez_compressed(out / f"{part}.npz", **combined)
        pd.concat(accepted).to_parquet(out / f"{part}-rows.parquet", index=False)
        pd.concat(ledger).to_parquet(out / f"{part}-input-coverage.parquet", index=False)
        coverage[part] = {"selected": len(frame), "input_available": len(combined["s1"]),
                          "known_target_days": int(combined["valid"].sum()),
                          "known_target_rows": int(combined["valid"].any(1).sum())}
        print(part, coverage[part], flush=True)
    write_json(out / "coverage.json", coverage)
    if config.get("path_identities_per_exchange"):
        count = config["path_identities_per_exchange"]
        last_train = rows["train"].loc[rows["train"].date.eq(rows["train"].date.max())]
        identities = []
        for exchange in ["xshg", "xshe", "xbse"]:
            symbols = [s for s in last_train.instrument_id.unique() if s.split('.')[1] == exchange]
            identities.extend(sorted(symbols, key=lambda s: hashlib.sha256(
                f'{config["seed"]}:{s}'.encode()).hexdigest())[:count])
        write_json(out / "path-identities.json", identities)
        evaluation_rows = pd.read_parquet(out / "evaluation-rows.parquet")
        ids = np.flatnonzero(evaluation_rows.instrument_id.isin(identities))
        if not len(ids):
            raise ValueError("No preselected generation identities in evaluation")
        np.save(out / "path-evaluation-indices.npy", ids)
    write_json(out / "prepared.json", {"status": "prepared", "files": hashes(
        out, sorted(p for p in out.rglob("*") if p.is_file() and p.name != "progress.json"))})


def prepared(out):
    verify_files(out, read(out / "prepared.json")["files"])
    config, exp = read(out / "protocol.json"), read(out / "experiment.json")
    validate_protocol(config)
    if file_hash(Path(exp["bundle"]) / "pretrained-provenance.json") != exp["tokenizer_provenance_sha256"]:
        raise ValueError("Tokenizer provenance changed")
    return config, exp


def tensors(data, ids, device):
    return [torch.as_tensor(data[k][ids], device=device) for k in ["s1", "s2", "stamps", "valid"]]


@torch.inference_mode()
def score(model, data, rows, config):
    model.eval()
    records = []
    for start in range(0, len(rows), config["batch_size"]):
        ids = np.arange(start, min(start+config["batch_size"], len(rows)))
        known = data["valid"][ids].any(1)
        ids = ids[known]
        if not len(ids):
            continue
        _, per_row = token_loss(model, *tensors(data, ids, config["device"]), config["lookback"])
        records.append(pd.DataFrame({"date": rows.iloc[ids].date.to_numpy(),
                                     "ce": per_row.mean(1).cpu().numpy()}))
    if not records:
        raise ValueError("Selection has no known targets")
    return float(pd.concat(records).groupby("date").ce.mean().mean())


def train(args):
    out = args.output.resolve()
    config, _ = prepared(out)
    dest = out / "training"
    dest.mkdir()
    torch.set_num_threads(4)
    torch.manual_seed(config["seed"])
    rng = np.random.default_rng(config["seed"])
    model = TokenTransformer(TokenConfig(**config["model"])).to(config["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    data, selection = arrays(out / "train.npz"), arrays(out / "selection.npz")
    rows, sr = [pd.read_parquet(out / f"{p}-rows.parquet") for p in ["train", "selection"]]
    known_ids = np.flatnonzero(data["valid"].any(1))
    counts = rows.iloc[known_ids].date.value_counts()
    # Date-weighted unbiased stochastic objective; normalized over the full training set.
    weights = rows.date.map(1 / counts).fillna(0).to_numpy(np.float32)
    weights *= len(known_ids) / weights.sum()
    best, bad, log = float("inf"), 0, []
    started = time.monotonic()
    initial = score(model, selection, sr, config)
    write_json(dest / "progress.json", dict(stage="training", initial_selection_ce=initial,
        training_rows=len(known_ids), training_dates=len(counts), device=config["device"],
        parameters=sum(p.numel() for p in model.parameters()), updated_at=utc_now()))
    print("Training", len(known_ids), "rows; initial selection CE", initial, flush=True)
    for epoch in range(1, config["max_epochs"]+1):
        model.train()
        order, total, count = rng.permutation(known_ids), 0.0, 0
        for start in range(0, len(order), config["batch_size"]):
            ids = order[start:start+config["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            _, per_row = token_loss(model, *tensors(data, ids, config["device"]), config["lookback"])
            loss = (per_row.mean(1) * torch.as_tensor(weights[ids], device=config["device"])).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            nnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(nnorm):
                raise ValueError("Nonfinite gradients")
            optimizer.step()
            total += float(loss.detach()) * len(ids)
            count += len(ids)
            if start % (config["batch_size"] * 50) == 0:
                progress = dict(epoch=epoch, processed=count, total=len(order),
                    training_ce=total/count, elapsed_seconds=time.monotonic()-started,
                    updated_at=utc_now())
                write_json(dest / "progress.json", progress)
                print(progress, flush=True)
        value = score(model, selection, sr, config)
        entry = dict(epoch=epoch, training_ce=total/count, selection_ce=value,
                     elapsed_seconds=time.monotonic()-started)
        log.append(entry)
        payload = checkpoint_payload(model, epoch=epoch, selection_ce=value,
                                     prepared_sha256=file_hash(out / "prepared.json"))
        torch.save(payload, dest / "last.pt")
        if value < best - config["min_delta"]:
            best, bad = value, 0
            torch.save(payload, dest / "best.pt")
        else:
            bad += 1
        write_json(dest / "history.json", log)
        print(entry, flush=True)
        if epoch >= config["min_epochs"] and bad >= config["patience"]:
            break
    model, selected = restore_model(dest / "best.pt")
    model.eval()
    with torch.inference_mode():
        a, b, stamps, _ = tensors(selection, np.arange(min(2, len(sr))), "cpu")
        logits = model(a[:, :-1], b[:, :-1], stamps[:, :-1], next_s1=a[:, 1:])
    np.savez_compressed(dest / "reload-reference.npz", coarse=logits[0].numpy(), fine=logits[1].numpy())
    write_json(dest / "summary.json", {
        "initial_selection_ce": initial, "best_selection_ce": best,
        "selected_epoch": selected["epoch"], "epochs": len(log),
        "parameters": sum(p.numel() for p in model.parameters()),
        "training_rows": len(known_ids), "training_dates": len(counts),
        "selection_rows": len(sr), "selection_dates": int(sr.date.nunique()),
        "stopping": "early_stopping" if bad >= config["patience"] else "budget_limited",
        "scope": config["scope"], "executable": False,
    })
    write_json(dest / "completed.json", {"status": "trained", "files": hashes(
        dest, sorted(p for p in dest.iterdir() if p.is_file()))})


def forecast_metrics(data, rows, paths, valid_paths):
    metrics = []
    for day in range(paths.shape[2]):
        for field, name in enumerate(FIELDS):
            records = []
            for i in range(len(paths)):
                if not data["valid"][i, day] or valid_paths[i].sum() < 8:
                    continue
                samples = paths[i, valid_paths[i], day, field]
                actual, scale = data["future"][i, day, field], data["scale"][i, 0, field]
                sorted_samples = np.sort(samples)
                n = len(samples)
                spread = np.sum((2*np.arange(1, n+1)-n-1) * sorted_samples) / n**2
                q = np.quantile(samples, [0.1, 0.5, 0.9])
                records.append(dict(date=rows.iloc[i].date,
                    normalized_mae=abs(q[1]-actual)/scale,
                    normalized_crps=(np.mean(abs(samples-actual))-spread)/scale,
                    persistence_normalized_mae=abs(data["last"][i, field]-actual)/scale,
                    coverage80=float(q[0] <= actual <= q[2])))
            means = pd.DataFrame(records).groupby("date").mean().mean().to_dict() if records else {}
            metrics.append(dict(day=day+1, field=name, known_rows=int(data["valid"][:, day].sum()),
                                scored_rows=len(records), **means))
    return metrics


def predict(args):
    out = args.output.resolve()
    config, exp = prepared(out)
    trained = out / "training"
    verify_files(trained, read(trained / "completed.json")["files"])
    dest = out / "forecast"
    dest.mkdir()
    model, saved = restore_model(trained / "best.pt", config["device"])
    if saved["prepared_sha256"] != file_hash(out / "prepared.json"):
        raise ValueError("Checkpoint belongs to another dataset")
    tokenizer = load_tokenizer(Path(exp["bundle"]), config["device"])
    data = arrays(out / "evaluation.npz")
    rows = pd.read_parquet(out / "evaluation-rows.parquet")
    all_rows = len(rows)
    full_token_ce = score(model, data, rows, config)
    if (out / "path-evaluation-indices.npy").exists():
        indices = np.load(out / "path-evaluation-indices.npy")
        data = {k: v[indices] for k, v in data.items()}
        rows = rows.iloc[indices].reset_index(drop=True)
    rows.to_parquet(dest / "rows.parquet", index=False)
    history, results = config["lookback"], []
    # One input per call gives each stock/date an independent, batch-stable seed.
    for i, row in enumerate(rows.itertuples()):
        seed = int(hashlib.sha256(f'{config["seed"]}:{row.instrument_id}:{row.date}'.encode()).hexdigest()[:8], 16)
        pairs = generate_tokens(model,
            torch.as_tensor(data["s1"][i:i+1, :history]),
            torch.as_tensor(data["s2"][i:i+1, :history]),
            torch.as_tensor(data["stamps"][i:i+1, :history]),
            torch.as_tensor(data["stamps"][i:i+1, history:]),
            samples=config["path_samples"], seed=seed, temperature=config["temperature"],
            top_p=config["top_p"], top_k=config["top_k"])
        paths, valid = decode_paths(tokenizer, pairs, data["mean"][i:i+1],
                                    data["scale"][i:i+1], config["horizon"], model.config.max_context)
        result = dict(paths=paths, valid_paths=valid, s1=pairs[0].cpu().numpy(),
                      s2=pairs[1].cpu().numpy(), seed=np.array([seed], dtype=np.int64))
        np.savez_compressed(dest / f"paths-{i:06d}.npz", **result)
        results.append(result)
        if (i+1) % 16 == 0:
            print("Generated", i+1, "/", len(rows), flush=True)
            write_json(dest / "progress.json", dict(generated=i+1, total=len(rows),
                updated_at=utc_now()))
    combined = {k: np.concatenate([r[k] for r in results]) for k in results[0]}
    np.savez_compressed(dest / "paths.npz", **combined)
    pd.DataFrame(forecast_metrics(data, rows, combined["paths"], combined["valid_paths"])).to_csv(
        dest / "metrics.csv", index=False)
    write_json(dest / "summary.json", {
        "rows": len(rows), "path_shape": list(combined["paths"].shape), "fields": FIELDS,
        "full_evaluation_rows": all_rows, "full_evaluation_token_ce": full_token_ce,
        "path_sampling_scope": "preselected historical identities" if len(rows) != all_rows else "all evaluation rows",
        "valid_paths": int(combined["valid_paths"].sum()),
        "generated_paths": int(combined["valid_paths"].size),
        "rows_with_eight_valid_paths": int((combined["valid_paths"].sum(1) >= 8).sum()),
        "scoring": "Date-equal forecast errors conditional on valid paths; unknown targets retained",
        "scope": config["scope"], "executable": False,
    })
    write_json(dest / "completed.json", {"status": "forecasted", "files": hashes(
        dest, sorted(p for p in dest.iterdir() if p.is_file()))})


def verify(args):
    out = args.output.resolve()
    config, exp = prepared(out)
    for part in ["training", "forecast"]:
        verify_files(out / part, read(out / part / "completed.json")["files"])
    model, _ = restore_model(out / "training/best.pt")
    model.eval()
    selection = arrays(out / "selection.npz")
    reference = arrays(out / "training/reload-reference.npz")
    with torch.inference_mode():
        a, b, stamps, _ = tensors(selection, np.arange(len(reference["coarse"])), "cpu")
        logits = model(a[:, :-1], b[:, :-1], stamps[:, :-1], next_s1=a[:, 1:])
    for name, actual in zip(["coarse", "fine"], logits):
        np.testing.assert_allclose(actual.numpy(), reference[name], atol=1e-6, rtol=1e-5)
    tokenizer = load_tokenizer(Path(exp["bundle"]), "cpu")
    data, result = arrays(out / "evaluation.npz"), arrays(out / "forecast/paths.npz")
    rows = pd.read_parquet(out / "evaluation-rows.parquet")
    if (out / "path-evaluation-indices.npy").exists():
        indices = np.load(out / "path-evaluation-indices.npy")
        data = {k: v[indices] for k, v in data.items()}
        rows = rows.iloc[indices].reset_index(drop=True)
    pd.testing.assert_frame_equal(rows, pd.read_parquet(out / "forecast/rows.parquet"))
    h = config["lookback"]
    with torch.inference_mode():
        full = tokenizer.encode(torch.from_numpy(data["normalized"][:2]), half=True)
        prefix = tokenizer.encode(torch.from_numpy(data["normalized"][:2, :h]), half=True)
    for complete, past in zip(full, prefix):
        torch.testing.assert_close(complete[:, :h], past, atol=0, rtol=0)
    pairs = [torch.from_numpy(result[k][:1]) for k in ["s1", "s2"]]
    decoded, valid = decode_paths(tokenizer, pairs, data["mean"][:1], data["scale"][:1],
                                  config["horizon"], model.config.max_context)
    np.testing.assert_allclose(decoded, result["paths"][:1], atol=0.05, rtol=1e-4)
    np.testing.assert_array_equal(valid, result["valid_paths"][:1])
    # Reproduce sampling on the originally selected device: CPU and MPS logits
    # need not lead to identical categorical draws at distribution boundaries.
    model.to(config["device"])
    sampled = generate_tokens(model, torch.from_numpy(data["s1"][:1, :h]),
        torch.from_numpy(data["s2"][:1, :h]), torch.from_numpy(data["stamps"][:1, :h]),
        torch.from_numpy(data["stamps"][:1, h:]), samples=config["path_samples"],
        seed=int(result["seed"][0]), temperature=config["temperature"], top_p=config["top_p"],
        top_k=config["top_k"])
    for name, actual in zip(["s1", "s2"], sampled):
        np.testing.assert_array_equal(actual.cpu().numpy(), result[name][:1])
    np.testing.assert_array_equal(valid_bars(result["paths"]).all(-1), result["valid_paths"])
    recalculated = pd.DataFrame(forecast_metrics(data, rows, result["paths"], result["valid_paths"]))
    pd.testing.assert_frame_equal(recalculated, pd.read_csv(out / "forecast/metrics.csv"),
                                  check_dtype=False, atol=1e-9, rtol=1e-7)
    write_json(out / "verification.json", {"passed": True, "checked_at": utc_now(),
        "cpu_reload_logit_values": int(sum(x.numel() for x in logits)),
        "tokenizer_prefix_causality": True, "sampled_path_reproduction": True,
        "metrics_recomputed": len(recalculated), "scope": config["scope"],
        "executable": False})
    print("Verified", out, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "train", "predict", "verify"])
    parser.add_argument("--output", required=True, type=Path)
    for name in ["prior", "inputs", "bundle", "config"]:
        parser.add_argument("--"+name, type=Path)
    args = parser.parse_args()
    if args.stage == "prepare" and any(getattr(args, k) is None for k in ["prior", "inputs", "bundle", "config"]):
        parser.error("prepare requires --prior, --inputs, --bundle and --config")
    torch.set_num_threads(4)
    {"prepare": prepare, "train": train, "predict": predict, "verify": verify}[args.stage](args)


if __name__ == "__main__":
    main()
