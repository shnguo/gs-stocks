"""Paired capacity evaluation on frozen rows; token CE and sampled OHLCVA paths."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from token_transformer_run import arrays, forecast_metrics, prepared, read, tensors, verify_files

from quant_research.storage import file_hash, utc_now, write_json
from quant_research.token_transformer import restore_model, token_loss


@torch.inference_mode()
def per_row_ce(model, data, config):
    model.eval()
    result = np.full((len(data["s1"]), 2), np.nan)
    for start in range(0, len(result), config["batch_size"]):
        ids = np.arange(start, min(start+config["batch_size"], len(result)))
        ids = ids[data["valid"][ids].any(1)]
        if not len(ids):
            continue
        _, parts = token_loss(model, *tensors(data, ids, config["device"]), config["lookback"])
        result[ids] = parts.cpu().numpy()
    return result


def frequency_baseline(train, dates, evaluation, lookback, bits):
    """Training-only token frequencies, date/row/horizon equal, Laplace smoothing."""
    dates = pd.Series(np.asarray(dates))
    valid = train["valid"]
    counts = dates[valid.any(1)].value_counts()
    weight = dates.map(1/counts).fillna(0).to_numpy()[:, None]
    weight = weight / valid.sum(1).clip(1)[:, None] * valid
    weight *= valid.sum() / weight.sum()
    out = np.full((len(evaluation["s1"]), 2), np.nan)
    for i, key in enumerate(["s1", "s2"]):
        freq = np.bincount(train[key][:, lookback:].ravel(), weights=weight.ravel(),
                           minlength=2**bits[i]) + 1.0
        probability = freq / freq.sum()
        losses = -np.log(probability[evaluation[key][:, lookback:]])
        known = evaluation["valid"].any(1)
        out[known, i] = (losses * evaluation["valid"]).sum(1)[known] / evaluation["valid"].sum(1)[known]
    return out


def paired_summary(values):
    values = np.asarray(values, float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("No finite paired dates")
    rng = np.random.default_rng(17)
    boot = []
    for _ in range(2000):
        starts = rng.integers(0, len(values), size=(len(values)+1)//2)
        ids = np.column_stack([starts, (starts+1) % len(values)]).ravel()[:len(values)]
        boot.append(values[ids].mean())
    return dict(mean_difference=float(values.mean()),
                two_date_block_95=np.quantile(boot, [0.025, 0.975]).tolist(), dates=len(values))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ["large", "small", "output"]:
        p.add_argument("--"+k, type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    roots = {"decoder_3720k": a.large.resolve(), "decoder_282k": a.small.resolve()}
    configs = {}
    for name, root in roots.items():
        configs[name], _ = prepared(root)
        if not read(root / "verification.json")["passed"]:
            raise ValueError("Unverified model run")
        for part in ["training", "forecast"]:
            verify_files(root / part, read(root / part / "completed.json")["files"])
    for name in ["train.npz", "selection.npz", "evaluation.npz", "path-evaluation-indices.npy",
                 "train-rows.parquet", "selection-rows.parquet", "evaluation-rows.parquet"]:
        if file_hash(a.large/name) != file_hash(a.small/name):
            raise ValueError(f"Different matched data: {name}")
    out.mkdir()
    torch.set_num_threads(4)
    data = arrays(a.large / "evaluation.npz")
    rows = pd.read_parquet(a.large / "evaluation-rows.parquet")
    ce = {}
    for name, root in roots.items():
        model, _ = restore_model(root / "training/best.pt", configs[name]["device"])
        ce[name] = per_row_ce(model, data, configs[name])
        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    tr = arrays(a.large / "train.npz")
    cfg = configs["decoder_3720k"]
    tr_dates = pd.read_parquet(a.large / "train-rows.parquet").date.to_numpy()
    ce["training_token_frequency"] = frequency_baseline(tr, tr_dates, data, cfg["lookback"],
        [cfg["model"]["s1_bits"], cfg["model"]["s2_bits"]])
    daily = []
    for name, values in ce.items():
        frame = rows[["date", "instrument_id"]].copy()
        frame["coarse_ce"], frame["fine_ce"] = values[:, 0], values[:, 1]
        frame["ce"] = values.mean(1)
        frame.to_parquet(out / f"{name}-token-ce.parquet", index=False)
        means = frame.groupby("date")[["coarse_ce", "fine_ce", "ce"]].mean().reset_index()
        means["model"] = name
        daily.append(means)
    daily = pd.concat(daily, ignore_index=True)
    daily.to_csv(out / "daily-token-ce.csv", index=False)
    pivot = daily.pivot(index="date", columns="model", values="ce")
    primary = paired_summary(pivot.decoder_3720k - pivot.decoder_282k)
    indices = np.load(a.large / "path-evaluation-indices.npy")
    data = {k: v[indices] for k, v in data.items()}
    rows = rows.iloc[indices].reset_index(drop=True)
    paths = {name: arrays(root / "forecast/paths.npz") for name, root in roots.items()}
    common = np.logical_and.reduce([(d["valid_paths"].sum(1) >= 8) for d in paths.values()])
    records = []
    for scope in ["own_available", "common_available"]:
        for name, value in paths.items():
            for day in sorted(rows.date.unique()):
                ix = np.flatnonzero(rows.date.eq(day))
                subset = {k: v[ix].copy() for k, v in data.items()}
                if scope == "common_available":
                    subset["valid"] &= common[ix, None]
                for record in forecast_metrics(subset, rows.iloc[ix].reset_index(drop=True),
                                               value["paths"][ix], value["valid_paths"][ix]):
                    records.append(dict(model=name, scope=scope, date=day, **record))
    metrics = pd.DataFrame(records)
    metrics.to_csv(out / "daily-path-metrics.csv", index=False)
    path_summary = metrics.groupby(["scope", "model", "day", "field"])[
        ["normalized_mae", "normalized_crps", "persistence_normalized_mae", "coverage80", "scored_rows"]
    ].mean().reset_index()
    path_summary.to_csv(out / "path-summary.csv", index=False)
    close = metrics.loc[(metrics.scope == "common_available") & (metrics.field == "close") & (metrics.day == 5)]
    close_pivot = close.pivot(index="date", columns="model", values="normalized_crps").dropna()
    secondary = paired_summary(close_pivot.decoder_3720k - close_pivot.decoder_282k) if len(close_pivot) else None
    summary = dict(created_at=utc_now(), full_token_rows=len(ce["decoder_3720k"]),
        full_token_dates=int(pivot.index.nunique()), path_rows=len(rows), common_path_rows=int(common.sum()),
        token_ce=daily.groupby("model").ce.mean().to_dict(),
        primary_large_minus_small_token_ce=primary,
        secondary_day5_close_crps_large_minus_small=secondary,
        training={name: read(root / "training/summary.json") for name, root in roots.items()},
        path_coverage={name: read(root / "forecast/summary.json") for name, root in roots.items()},
        limitations=["One previously researched development window and one seed",
                     "43 training signal dates; stock rows are not independent market periods",
                     "Tokenizer pretraining timeline not independently verified",
                     "Sampled paths use identities fixed from the last training date",
                     "Common-valid-path metrics are conditional and retain coverage counts",
                     "No final holdout accessed and no profitability claim"], executable=False)
    write_json(out / "summary.json", summary)
    write_json(out / "sources.json", {str(root / "verification.json"): file_hash(root / "verification.json")
                                      for root in roots.values()})
    write_json(out / "completed.json", {"files": {p.name: file_hash(p) for p in out.iterdir() if p.is_file()},
                                         "status": "completed"})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
