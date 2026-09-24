from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def isolated_tree(train_x: np.ndarray, train_y: np.ndarray, validation_x: np.ndarray,
                  validation_y: np.ndarray, test_x: np.ndarray, seed: int,
                  model_path: Path, train_weight: np.ndarray | None = None,
                  validation_weight: np.ndarray | None = None,
                  validation_dates: np.ndarray | None = None) -> np.ndarray:
    # exec starts a clean interpreter; fork would inherit initialized OpenMP thread pools.
    with tempfile.TemporaryDirectory(prefix=".tree-worker-", dir=model_path.parent) as directory:
        root = Path(directory)
        for name, array in [("train_x", train_x), ("train_y", train_y),
                            ("validation_x", validation_x), ("validation_y", validation_y),
                            ("test_x", test_x)]:
            np.save(root / f"{name}.npy", array, allow_pickle=False)
        for name, array in [("train_weight", train_weight), ("validation_weight", validation_weight),
                            ("validation_dates", validation_dates)]:
            if array is not None:
                np.save(root / f"{name}.npy", array, allow_pickle=False)
        subprocess.run([sys.executable, "-m", "quant_research.tree_worker", str(root),
                        "--seed", str(seed)], check=True, timeout=1800)
        shutil.copyfile(root / "model.txt", model_path)
        shutil.copyfile(root / "runtime.json", model_path.with_suffix(".runtime.json"))
        return np.load(root / "prediction.npy", allow_pickle=False)
