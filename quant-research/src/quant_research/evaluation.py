from __future__ import annotations

import numpy as np
import pandas as pd


def block_bootstrap(values: np.ndarray, block_days: int = 20,
                    repetitions: int = 1000, seed: int = 17) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2 * block_days:
        return {"status": "insufficient_history", "observations": len(values),
                "block_days": block_days, "confidence_interval": None}
    rng = np.random.default_rng(seed)
    means = []
    blocks = int(np.ceil(len(values) / block_days))
    for _ in range(repetitions):
        starts = rng.integers(0, len(values) - block_days + 1, size=blocks)
        sample = np.concatenate([values[s:s + block_days] for s in starts])[:len(values)]
        means.append(sample.mean())
    return {"status": "estimated", "observations": len(values), "block_days": block_days,
            "confidence_interval": np.quantile(means, [0.025, 0.975]).tolist(),
            "mean": float(values.mean()), "repetitions": repetitions}


def paired_rank_ic_difference(first: dict, second: dict) -> dict:
    a = pd.DataFrame(first["daily"])
    b = pd.DataFrame(second["daily"])
    if a.empty or b.empty:
        return {"status": "insufficient_history"}
    merged = a.merge(b, on="date", suffixes=("_a", "_b"))
    return block_bootstrap((merged.rank_ic_a - merged.rank_ic_b).to_numpy())

