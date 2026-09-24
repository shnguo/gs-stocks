"""Versioned prediction interface; daily storage does not depend on model architecture."""

from pathlib import Path
from typing import Protocol

import numpy as np

from .price_strategy import FIELDS, ordered_quantiles


class ForecastAdapter(Protocol):
    def predict(self, features: np.ndarray, history: np.ndarray | None = None) -> np.ndarray:
        """Return [instrument, 5 sessions, OHLC, q10/q50/q90] log quote changes."""
        ...


class LightGBMAdapter:
    def __init__(self, root: Path):
        import lightgbm as lgb

        self.models = {
            (d, f, j): lgb.Booster(model_file=str(root / "weights" / f"d{d + 1}-{field}-q{q}.txt"))
            for d in range(5)
            for f, field in enumerate(FIELDS)
            for j, q in enumerate([0.1, 0.5, 0.9])
        }

    def predict(self, features, history=None):
        result = np.empty((len(features), 5, 4, 3))
        for (d, f, j), model in self.models.items():
            result[:, d, f, j] = model.predict(features, num_threads=4)
        return ordered_quantiles(result)


class EmpiricalAdapter:
    def __init__(self, root: Path):
        self.quantiles = np.load(root / "naive.npy")

    def predict(self, features, history=None):
        return np.broadcast_to(self.quantiles, (len(features), 5, 4, 3)).copy()


ADAPTERS = {"lightgbm": LightGBMAdapter, "naive": EmpiricalAdapter}


def adapter(name, root):
    if name not in ADAPTERS:
        raise ValueError(f"Model adapter not installed: {name}; no silent fallback")
    return ADAPTERS[name](root)


def validate_forecast(result, rows):
    if result.shape != (rows, 5, 4, 3) or not np.isfinite(result).all():
        raise ValueError("Invalid forecast axes or nonfinite model output")
    if (np.diff(result, axis=-1) < 0).any():
        raise ValueError("Crossed prediction quantiles")
