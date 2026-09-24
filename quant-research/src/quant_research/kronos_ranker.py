"""Supervised Kronos adaptation; the pretrained tokenizer stays frozen."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .storage import file_hash

FIELDS = ["open", "high", "low", "close", "volume", "amount", "factor"]


def normalize_windows(raw: np.ndarray, clip: float = 5.0) -> np.ndarray:
    """Normalize each past-only window using the upstream population-std convention.

    OHLC is adjusted to the signal-date factor anchor. Turnover stays in the
    recorded share/RMB units. No future prices, statistics, or factors are used.
    """
    if raw.ndim != 3 or raw.shape[-1] != 7 or not np.isfinite(raw).all():
        raise ValueError("Kronos requires complete past OHLCV/amount/factor windows")
    if (raw[..., 6] <= 0).any() or (raw[..., :4] <= 0).any():
        raise ValueError("Nonpositive price or adjustment factor")
    if (raw[..., 4:6] < 0).any():
        raise ValueError("Negative turnover")
    x = raw[..., :6].astype(np.float32, copy=True)
    x[..., :4] *= raw[..., 6:7] / raw[:, -1:, 6:7]
    x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-5)
    return np.clip(x, -clip, clip).astype(np.float32)


def timestamps(dates: list[str]) -> np.ndarray:
    dates = pd.DatetimeIndex(dates)
    return np.column_stack([np.zeros(len(dates)), np.zeros(len(dates)),
                            dates.weekday, dates.day, dates.month]).astype(np.float32)


def batch_inputs(values, stamps, coordinates, lookback, device):
    s, t = coordinates.T
    if (t < lookback - 1).any() or (t >= values.shape[1]).any():
        raise ValueError("Window crosses the prepared development boundary")
    times = t[:, None] + np.arange(-lookback + 1, 1)
    raw = values[s[:, None], times]
    x = torch.from_numpy(normalize_windows(raw)).to(device)
    stamp = torch.from_numpy(stamps[times].copy()).to(device)
    return x, stamp


class KronosRanker(nn.Module):
    def __init__(self, tokenizer, backbone):
        super().__init__()
        self.tokenizer = tokenizer.requires_grad_(False).eval()
        self.backbone = backbone
        # Native vocabulary/dependency heads do not participate in ranking.
        backbone.head.requires_grad_(False)
        backbone.dep_layer.requires_grad_(False)
        self.rank_head = nn.Sequential(nn.LayerNorm(backbone.d_model),
                                       nn.Linear(backbone.d_model, 1))

    def train(self, mode=True):
        super().train(mode)
        self.tokenizer.eval()
        return self

    def forward(self, x, stamp):
        with torch.no_grad():
            s1, s2 = self.tokenizer.encode(x, half=True)
        backbone = self.backbone
        hidden = backbone.embedding([s1, s2]) + backbone.time_emb(stamp)
        hidden = backbone.token_drop(hidden)
        for layer in backbone.transformer:
            hidden = layer(hidden)
        return self.rank_head(backbone.norm(hidden)[:, -1]).squeeze(-1)


def load_pretrained(bundle: Path, device: str = "cpu") -> KronosRanker:
    from safetensors.torch import load_model

    provenance = json.loads((bundle / "pretrained-provenance.json").read_text())
    for name, expected in provenance["files"].items():
        path = (bundle / name).resolve()
        if not path.is_relative_to(bundle.resolve()) or file_hash(path) != expected:
            raise ValueError("Pretrained source/checkpoint hash mismatch")
    upstream = (bundle / "upstream").resolve()
    existing = sys.modules.get("model")
    if existing and Path(existing.__file__).resolve().parent != upstream / "model":
        raise ValueError("Another package named model is already imported")
    sys.path.insert(0, str(upstream))
    api = importlib.import_module("model")
    tokenizer = api.KronosTokenizer(**json.loads((bundle / "tokenizer/config.json").read_text()))
    backbone = api.Kronos(**json.loads((bundle / "model/config.json").read_text()))
    load_model(tokenizer, str(bundle / "tokenizer/model.safetensors"), strict=True)
    load_model(backbone, str(bundle / "model/model.safetensors"), strict=True)
    return KronosRanker(tokenizer, backbone).to(device)


def compare_predictions(baseline: pd.DataFrame, kronos: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Reject missing/extra/duplicate samples instead of silently inner-joining."""
    from .evaluation import paired_rank_ic_difference
    from .models import rank_ic

    keys = ["date", "instrument_id"]
    for frame in [baseline, kronos]:
        if frame.duplicated(keys).any():
            raise ValueError("Duplicate comparison sample")
    merged = baseline.merge(kronos[keys + ["forward_return", "kronos_score"]],
                            on=keys, how="outer", validate="one_to_one",
                            indicator=True, suffixes=("", "_kronos"))
    if not merged._merge.eq("both").all():
        raise ValueError("Comparison sample sets differ")
    if not np.allclose(merged.forward_return, merged.forward_return_kronos,
                       equal_nan=True, rtol=0, atol=0):
        raise ValueError("Comparison labels differ")
    metrics = {}
    for name in ["lightgbm", "transformer", "kronos"]:
        scores = merged[f"{name}_score"].to_numpy()
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite comparison score")
        metrics[name] = rank_ic(merged, scores)
    metrics["paired_kronos_minus_baselines"] = {
        name: paired_rank_ic_difference(metrics["kronos"], metrics[name])
        for name in ["lightgbm", "transformer"]}
    metrics["scored_rows"] = len(merged)
    metrics["labelled_rows"] = int(merged.forward_return.notna().sum())
    metrics["formal_ready"] = False
    metrics["pretraining_test_overlap"] = "unverified; retrospective diagnostic only"
    return merged.drop(columns=["_merge", "forward_return_kronos"]), metrics
