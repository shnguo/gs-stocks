"""Backtest 'fell for months, turning up over the last 3 months' on the US index union.

Usage: uv run python scripts/turnaround_check.py configs/turnaround-check-v1.json
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
    composite,
    evaluate,
    forward_returns,
    percentile,
    performance,
    physics_momentum,
)
from quant_research.storage import write_json

BASE = Path(__file__).resolve().parents[1]


def turnaround_inputs(panel, date, cfg):
    p, r = cfg["prior_window"], cfg["recent_window"]
    prior = physics_momentum(panel, date, p["lookback"], p["skip"], p["min_traded"], cfg["jump_sigma"])["drift"]
    recent = physics_momentum(panel, date, r["lookback"], r["skip"], r["min_traded"], cfg["jump_sigma"])["drift"]
    return prior, recent


def baskets(prior, recent, score, mask, cut):
    ok = mask & np.isfinite(prior) & np.isfinite(recent)
    turned = ok & (prior < 0) & (recent > 0)
    p_rank, r_rank = percentile(prior, ok), percentile(recent, ok)
    deep = ok & (p_rank <= cut) & (r_rank >= 1 - cut)
    top = np.zeros(len(mask), dtype=bool)
    candidates = np.where(turned & np.isfinite(score))[0]
    top[candidates[np.argsort(-score[candidates])[:50]]] = True
    return {"turned_up": turned, "deep_turn": deep, "turned_up_top50": top}


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

    cohorts, rows, previous = [], [], {}
    for date, following, mask, _ in months:
        returns, _ = forward_returns(panel, date, following)
        keep = lambda x: np.where(mask, x, np.nan)  # noqa: E731
        prior, recent = (keep(x) for x in turnaround_inputs(panel, date, cfg))
        score = composite([-prior, recent], mask, 2)
        c = {"date": date, "ids": np.asarray(panel.ids), "returns": keep(returns),
             "turnaround_score": score, "recent_drift_3m": recent, "prior_fall_9m": -prior}
        cohorts.append(c)
        valid = np.isfinite(c["returns"]) & np.isfinite(prior) & np.isfinite(recent)
        if valid.sum() < 100:
            continue
        universe = float(np.mean(c["returns"][valid]))
        row = {"date": date, "universe": universe}
        for name, pick in baskets(prior, recent, score, mask, cfg["relative_cut"]).items():
            pick = pick & valid
            ids = set(np.asarray(panel.ids)[pick])
            turnover = 1 - len(ids & previous[name]) / len(ids) if name in previous and ids else 1.0
            previous[name] = ids
            gross = float(np.mean(c["returns"][pick])) - universe if pick.sum() >= 5 else np.nan
            row.update({f"{name}_n": int(pick.sum()), f"{name}_excess": gross,
                        f"{name}_excess_net": gross - 2 * us["cost_one_way"] * turnover, f"{name}_turnover": turnover})
        rows.append(row)

    results = {"scores": {}, "baskets": {}}
    for name in cfg["scores"]:
        out = evaluate(cohorts, name, us["quantiles"], us["cost_one_way"], us["bootstrap_block_months"],
                       us["bootstrap_repetitions"])
        out.pop("daily")
        results["scores"][name] = {k: out[k] for k in ["periods", "rank_ic_mean", "rank_ic_t", "rank_ic_bootstrap",
                                                       "quantile_mean_monthly", "top_turnover_mean",
                                                       "long_short_net", "top_excess_net"]}
    frame = pd.DataFrame(rows)
    frame["year"] = frame.date.str[:4]
    for name in cfg["baskets"]:
        x = frame[f"{name}_excess_net"].dropna()
        boot = block_bootstrap(x.to_numpy(), block_days=us["bootstrap_block_months"],
                               repetitions=us["bootstrap_repetitions"])
        results["baskets"][name] = {
            "description": cfg["baskets"][name], "months": len(x),
            "mean_stocks": float(frame[f"{name}_n"].mean()), "mean_turnover": float(frame[f"{name}_turnover"].mean()),
            "excess_net": performance(x.to_numpy()), "monthly_ci": boot.get("confidence_interval"),
            "by_year": frame.groupby("year")[f"{name}_excess_net"].sum().round(4).to_dict()}
    output.mkdir(parents=True)
    frame.to_csv(output / "baskets.csv", index=False)
    write_json(output / "results.json", {"version": cfg["version"], "config": cfg, "source_manifest_sha256": source_hash,
                                         "results": results, "elapsed_seconds": round(time.time() - started, 1)})
    print(f"wrote {output} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
