"""Native, frozen Kronos forecasts with individual paths retained."""
from __future__ import annotations

import importlib

import numpy as np
import torch

from .kronos_ranker import load_pretrained, timestamps


def path_quantiles(paths: np.ndarray, reference: np.ndarray, min_paths: int = 8):
    if paths.ndim != 4 or paths.shape[2:] != (5, 6) or len(reference) != len(paths):
        raise ValueError("Expected [signal, sample, day, OHLCVA]")
    good = np.isfinite(paths).all(axis=(2, 3))
    good &= (paths[..., :4] > 0).all(axis=(2, 3))
    good &= (paths[..., 4:] >= 0).all(axis=(2, 3))
    good &= (paths[..., 1] >= paths[..., :4].max(axis=-1)).all(axis=2)
    good &= (paths[..., 2] <= paths[..., :4].min(axis=-1)).all(axis=2)
    q = np.full((len(paths), 5, 4, 3), np.nan)
    for i, mask in enumerate(good):
        if mask.sum() >= min_paths:
            ratios = np.log(paths[i, mask, :, :4] / reference[i])
            q[i] = np.quantile(ratios, [.1, .5, .9], axis=0).transpose(1, 2, 0)
    return q, good


def native_paths(bundle, raw, past_dates, future_dates, *, samples=16, seed=17, device="cpu"):
    """Use official inference with sample_count=1 for each repeated input.

    The official averaging axis then has size one, so independent paths survive.
    Normalization/denormalization sees historical data only. No fine-tuning.
    """
    if (raw.ndim != 3 or raw.shape[-1] != 7 or not np.isfinite(raw).all()
            or (raw[..., :4] <= 0).any() or (raw[..., 6] <= 0).any()
            or (raw[..., 4:6] < 0).any() or samples < 8):
        raise ValueError("Invalid past windows or insufficient path samples")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = load_pretrained(bundle, device).eval().requires_grad_(False)
    inference = importlib.import_module("model.kronos").auto_regressive_inference
    output = []
    for i in range(len(raw)):
        x = raw[i, :, :6].astype(np.float32, copy=True)
        x[:, :4] *= raw[i, :, 6:7] / raw[i, -1, 6]
        mean, std = x.mean(axis=0), x.std(axis=0) + 1e-5
        normalized = (x - mean) / std
        inp = np.repeat(normalized[None], samples, axis=0)
        xp = np.repeat(timestamps(past_dates[i])[None], samples, axis=0)
        yp = np.repeat(timestamps(future_dates[i])[None], samples, axis=0)
        with torch.inference_mode():
            result = inference(model.tokenizer, model.backbone,
                torch.from_numpy(inp).to(device), torch.from_numpy(xp).to(device),
                torch.from_numpy(yp).to(device), max_context=512, pred_len=5,
                T=1., top_k=0, top_p=.9, sample_count=1, verbose=False)
        output.append(result[:, -5:] * std + mean)
        print(f"Kronos native prediction {i + 1}/{len(raw)}; {samples} paths retained", flush=True)
    return np.stack(output)
