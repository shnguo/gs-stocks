"""A fresh process owns LightGBM's OpenMP runtime; PyTorch must never be imported here."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    root = args.directory
    def load(name: str) -> np.ndarray:
        return np.load(root / f"{name}.npy", allow_pickle=False)
    def weight(name):
        return load(name) if (root / f"{name}.npy").exists() else None
    train = lgb.Dataset(load("train_x"), label=load("train_y"), weight=weight("train_weight"))
    validation = lgb.Dataset(load("validation_x"), label=load("validation_y"), weight=weight("validation_weight"), reference=train)
    metric = None
    if (root / "validation_dates.npy").exists():
        codes, _ = pd.factorize(load("validation_dates"))
        truth = pd.Series(load("validation_y")).groupby(codes).rank().to_numpy()
        counts = np.bincount(codes)
        centered_y = truth - (np.bincount(codes, weights=truth) / counts)[codes]
        y_var = np.bincount(codes, weights=centered_y ** 2)
        def metric(prediction, dataset):
            rank = pd.Series(prediction).groupby(codes).rank().to_numpy()
            x = rank - (np.bincount(codes, weights=rank) / counts)[codes]
            denom = np.sqrt(np.bincount(codes, weights=x ** 2) * y_var)
            valid = (counts >= 3) & (denom > 0)
            if not valid.any():
                raise ValueError("Validation dates cannot support Rank IC")
            ic = np.bincount(codes, weights=x * centered_y)[valid] / denom[valid]
            return "mean_daily_rank_ic", float(ic.mean()), True
    model = lgb.train({"objective": "huber", "alpha": 0.9, "learning_rate": 0.03,
                       "num_leaves": 15, "min_data_in_leaf": 40, "lambda_l2": 1.0,
                       "seed": args.seed, "num_threads": 4, "deterministic": True,
                       "force_col_wise": True, "verbosity": -1,
                       "metric": "None" if metric else "huber"}, train, num_boost_round=500,
                      valid_sets=[validation], feval=metric, callbacks=[lgb.early_stopping(30, verbose=False)])
    model.save_model(str(root / "model.txt"))
    np.save(root / "prediction.npy", model.predict(load("test_x")))
    if "torch" in sys.modules:
        raise RuntimeError("Unexpected PyTorch import in isolated tree worker")
    (root / "runtime.json").write_text(json.dumps({"torch_loaded": False,
                                                 "lightgbm_version": lgb.__version__,
                                                 "selection_metric": "mean_daily_rank_ic" if metric else "huber",
                                                 "best_iteration": model.best_iteration}))


if __name__ == "__main__":
    main()
