"""Backtest three chart shapes x four periods on the US index union.

Usage: uv run python scripts/shapes_check.py configs/shapes-check-v1.json
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.evaluation import block_bootstrap
from quant_research.factors import (
    forward_returns,
    performance,
    physics_momentum,
    rank_ic,
)
from quant_research.shapes import shape_scores
from quant_research.storage import write_json

BASE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    cfg = json.loads(parser.parse_args().config.read_text())
    output = BASE / cfg["output"]
    if output.exists():
        raise FileExistsError(f"Use a new output path: {output}")
    started = time.time()
    spec = importlib.util.spec_from_file_location("physics_robustness", BASE / "scripts/physics_robustness.py")
    robustness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(robustness)
    us = json.loads((BASE / cfg["us_config"]).read_text())
    panel, months, source_hash = robustness.us_months(us)
    m = us["momentum"]
    cost = us["cost_one_way"]

    rows, previous = [], {}
    for date, following, mask, _ in months:
        returns, _ = forward_returns(panel, date, following)
        returns = np.where(mask, returns, np.nan)
        base = np.where(mask, physics_momentum(panel, date, m["lookback"], m["skip"], m["min_traded"],
                                               cfg["jump_sigma"])["drift"], np.nan)
        shapes = shape_scores(panel, date, mask, base, cfg)
        valid = np.isfinite(returns)
        if valid.sum() < 100:
            continue
        universe = float(np.mean(returns[valid]))
        row = {"date": date, "universe": universe}
        for key, (score, basket) in shapes.items():
            pick = basket & valid
            ids = set(np.asarray(panel.ids)[pick])
            turnover = 1 - len(ids & previous[key]) / len(ids) if key in previous and ids else 1.0
            previous[key] = ids
            gross = float(np.mean(returns[pick])) - universe if pick.sum() >= 5 else np.nan
            row.update({f"{key}|ic": rank_ic(score, returns), f"{key}|n": int(pick.sum()),
                        f"{key}|turnover": turnover, f"{key}|excess_net": gross - 2 * cost * turnover})
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame["year"] = frame.date.str[:4]
    results = {}
    for key in sorted({c.split("|")[0] for c in frame.columns if "|" in c}):
        x = frame[f"{key}|excess_net"].dropna()
        ic = frame[f"{key}|ic"].dropna()
        boot = block_bootstrap(x.to_numpy(), block_days=us["bootstrap_block_months"],
                               repetitions=us["bootstrap_repetitions"])
        by_year = frame.dropna(subset=[f"{key}|excess_net"]).groupby("year")[f"{key}|excess_net"].sum()
        results[key] = {
            "months": len(x), "first": frame.loc[x.index[0], "date"] if len(x) else None,
            "mean_stocks": float(frame.loc[x.index, f"{key}|n"].mean()),
            "mean_turnover": float(frame.loc[x.index, f"{key}|turnover"].iloc[1:].mean()),
            "rank_ic_mean": float(ic.mean()), "rank_ic_t": float(ic.mean() / ic.std() * np.sqrt(len(ic))),
            "excess_net": performance(x.to_numpy()), "monthly_ci": boot.get("confidence_interval"),
            "losing_years": int((by_year < 0).sum()), "years": int(len(by_year)),
            "by_year": by_year.round(4).to_dict()}
    output.mkdir(parents=True)
    frame.to_csv(output / "monthly.csv", index=False)
    write_json(output / "results.json", {"version": cfg["version"], "config": cfg, "source_manifest_sha256": source_hash,
                                         "results": results, "elapsed_seconds": round(time.time() - started, 1)})
    print(f"wrote {output} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
