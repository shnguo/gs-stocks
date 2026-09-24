from __future__ import annotations

import copy
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .features import Panel, Standardizer
from .storage import utc_now, write_json


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.use_deterministic_algorithms(True)


class TemporalRanker(nn.Module):
    """One shared per-stock encoder; every input token is at or before the signal cutoff."""
    def __init__(self, features: int, lookback: int, config: dict):
        super().__init__()
        width = config["hidden_size"]
        self.input = nn.Linear(features, width)
        self.position = nn.Parameter(torch.empty(1, lookback, width))
        nn.init.normal_(self.position, std=0.02)
        # Build independent layers: TransformerEncoder cloning otherwise repeats initialization.
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(width, config["heads"], config["feedforward_size"],
                                       config["dropout"], batch_first=True, norm_first=True,
                                       activation="gelu") for _ in range(config["layers"])
        ])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        x = self.input(values) + self.position[:, :values.shape[1]]
        for layer in self.layers:
            x = layer(x)
        return self.output(x[:, -1]).squeeze(-1)


class Windows(Dataset):
    def __init__(self, panel: Panel, samples: pd.DataFrame, scaler: Standardizer):
        self.panel = panel
        self.samples = samples.reset_index(drop=True)
        self.scaler = scaler
        self.indices = samples[["stock_index", "date_index"]].to_numpy(dtype=np.int64)
        self.targets = samples.target.to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        s, t = self.indices[index]
        values = self.panel.values[s, t - self.panel.lookback + 1:t + 1]
        x = torch.from_numpy(self.scaler.transform(values))
        return x, torch.tensor(self.targets[index], dtype=torch.float32)


def rank_ic(rows: pd.DataFrame, scores: np.ndarray) -> dict:
    frame = rows[["date", "forward_return"]].copy()
    frame["score"] = scores
    daily = []
    for date, group in frame.groupby("date", sort=True):
        group = group.dropna()
        if len(group) >= 3 and group.score.nunique() > 1 and group.forward_return.nunique() > 1:
            daily.append({"date": date, "rank_ic": float(group.score.rank().corr(
                group.forward_return.rank())), "stocks": len(group)})
    values = np.array([r["rank_ic"] for r in daily])
    return {"mean_rank_ic": float(values.mean()) if len(values) else None,
            "rank_ic_std": float(values.std(ddof=1)) if len(values) > 1 else None,
            "observations": len(values), "daily": daily}


def predict_transformer(model: TemporalRanker, panel: Panel, rows: pd.DataFrame,
                        scaler: Standardizer, batch_size: int) -> np.ndarray:
    model.eval()
    result = []
    with torch.no_grad():
        for x, _ in DataLoader(Windows(panel, rows, scaler), batch_size=batch_size):
            result.append(model(x.to(next(model.parameters()).device)).cpu().numpy())
    return np.concatenate(result) if result else np.empty(0)


def train_transformer(panel: Panel, train: pd.DataFrame, validation: pd.DataFrame,
                      scaler: Standardizer, config: dict, seed: int,
                      progress_dir: Path | None = None) -> tuple[TemporalRanker, list]:
    seed_everything(seed)
    device = torch.device(config.get("device", "cpu"))
    model = TemporalRanker(len(panel.feature_names), panel.lookback, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    loss_fn = nn.HuberLoss(delta=0.1, reduction="none")
    generator = torch.Generator().manual_seed(seed)
    windows = Windows(panel, train, scaler)
    weights = date_weights(train)
    # Draw each row uniformly and weight it by inverse date size. Expected
    # contribution is equal per signal date, regardless of the listed count.
    loader = DataLoader(torch.utils.data.TensorDataset(torch.arange(len(train))),
                        batch_size=config["batch_size"], shuffle=True, generator=generator)
    started = time.monotonic()
    steps = 0

    def progress(epoch, batch, loss):
        if progress_dir is None:
            return
        progress_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = progress_dir / ".latest.pt"
        torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "dataset_id": panel.dataset_id, "horizon": panel.horizon,
                    "feature_names": panel.feature_names, "lookback": panel.lookback,
                    "config": config, "seed": seed, "epoch": epoch, "batch": batch,
                    "steps": steps, "training_rows": len(train),
                    "scaler_mean": scaler.mean, "scaler_scale": scaler.scale,
                    "selection_status": "in_progress_not_validation_selected"}, checkpoint)
        checkpoint.replace(progress_dir / "latest.pt")
        write_json(progress_dir / "progress.json", {
            "status": "training", "dataset_id": panel.dataset_id, "horizon": panel.horizon,
            "epoch": epoch, "batch": batch, "batches_per_epoch": len(loader),
            "optimizer_steps": steps, "loss": loss, "train_rows": len(train),
            "validation_rows": len(validation), "device": str(device),
            "date_weighting": "equal_signal_date", "elapsed_seconds": time.monotonic()-started,
            "updated_at": utc_now()})
    best_score, best_state, bad_epochs, log = -float("inf"), None, 0, []
    for epoch in range(config["max_epochs"]):
        model.train()
        total, count = 0.0, 0
        for batch, (indices,) in enumerate(loader, 1):
            ids = indices.numpy()
            coords = windows.indices[ids]
            offsets = np.arange(-panel.lookback + 1, 1)
            raw = panel.values[coords[:, 0, None], coords[:, 1, None] + offsets]
            x = torch.from_numpy(scaler.transform(raw)).to(device)
            y = torch.from_numpy(windows.targets[ids]).to(device)
            weight = torch.from_numpy(weights[ids]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = (loss_fn(model(x), y) * weight).mean()
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(x)
            count += len(x)
            steps += 1
            if batch == 1 or batch % 100 == 0:
                progress(epoch + 1, batch, float(loss.detach()))
        prediction = predict_transformer(model, panel, validation, scaler, config["batch_size"])
        ic = rank_ic(validation, prediction)["mean_rank_ic"]
        if ic is None:
            raise ValueError("Validation cross sections cannot support Rank IC")
        log.append({"epoch": epoch + 1, "train_loss": total / count, "validation_rank_ic": ic})
        if ic > best_score:
            best_score, best_state, bad_epochs = ic, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
        if progress_dir is not None:
            progress(epoch + 1, len(loader), total / count)
            write_json(progress_dir / "training-log.json", log)
            torch.save({"state_dict": best_state, "best_validation_rank_ic": best_score,
                        "dataset_id": panel.dataset_id, "config": config, "seed": seed},
                       progress_dir / "best-validation.pt")
        if bad_epochs >= config["patience"]:
            break
    if best_state is None:
        raise ValueError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, log


def date_weights(rows: pd.DataFrame) -> np.ndarray:
    if rows.empty:
        raise ValueError("Cannot weight empty training rows")
    counts = rows.groupby("date").date.transform("size").to_numpy(dtype=float)
    return (len(rows) / rows.date.nunique() / counts).astype(np.float32)


def tabular_values(panel: Panel, rows: pd.DataFrame, scaler: Standardizer,
                   flatten: bool = False) -> np.ndarray:
    if flatten:
        # Explicit equal-information control; guard memory rather than silently subsampling.
        bytes_needed = len(rows) * panel.lookback * len(panel.feature_names) * 4
        if bytes_needed > 2 * 1024**3:
            raise ValueError("Flattened baseline exceeds 2 GiB; use a partitioned training runner")
        return scaler.transform(panel.windows(rows)).reshape(len(rows), -1)
    return scaler.transform(panel.values[rows.stock_index.to_numpy(dtype=int),
                                        rows.date_index.to_numpy(dtype=int)])


def save_transformer(path: Path, model: TemporalRanker, panel: Panel,
                     scaler: Standardizer, config: dict, seed: int) -> None:
    torch.save({"state_dict": model.state_dict(), "dataset_id": panel.dataset_id,
                "training_risk_policy": panel.risk_policy,
                "feature_names": panel.feature_names, "lookback": panel.lookback,
                "horizon": panel.horizon, "config": config, "seed": seed,
                "mean": scaler.mean.tolist(), "scale": scaler.scale.tolist()}, path)
