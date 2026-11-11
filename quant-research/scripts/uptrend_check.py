"""Does requiring a current uptrend improve the drift screen? US index union backtest.

Usage: uv run python scripts/uptrend_check.py configs/uptrend-check-v1.json
"""
import argparse
import importlib.util
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.evaluation import block_bootstrap
from quant_research.factors import (
    evaluate,
    forward_returns,
    percentile,
    physics_momentum,
    quantile_labels,
)
from quant_research.storage import write_json

BASE = Path(__file__).resolve().parents[1]


def moving_average(panel, t, days, min_share):
    window = panel.close[max(0, t - days + 1):t + 1]
    seq = panel.sequence[max(0, t - days + 1):t + 1]
    same = np.where(seq == panel.sequence[t], window, np.nan)
    count = np.isfinite(same).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(same, axis=0)
    return np.where(count >= min_share * days, mean, np.nan)


def gated(score, gate, mask):
    """Stocks passing the gate rank ahead of all others; drift orders within each side."""
    pct = percentile(score, mask)
    return np.where(np.isfinite(gate), pct + np.where(gate > 0, 1.0, 0.0), np.nan)


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
    m, f, a = us["momentum"], cfg["full_window"], cfg["averages"]

    cohorts, diagnostic = [], []
    for date, following, mask, _ in months:
        t = panel.index(date)
        returns, _ = forward_returns(panel, date, following)
        keep = lambda x: np.where(mask, x, np.nan)  # noqa: E731
        d121 = keep(physics_momentum(panel, date, m["lookback"], m["skip"], m["min_traded"], cfg["jump_sigma"])["drift"])
        d120 = keep(physics_momentum(panel, date, f["lookback"], f["skip"], f["min_traded"], cfg["jump_sigma"])["drift"])
        last = panel.close[t]
        above50 = np.where(np.isfinite(ma := moving_average(panel, t, a["short"], a["min_share"])),
                           (last > ma).astype(float), np.nan)
        above200 = np.where(np.isfinite(ma := moving_average(panel, t, a["long"], a["min_share"])),
                            (last > ma).astype(float), np.nan)
        c = {"date": date, "ids": np.asarray(panel.ids), "returns": keep(returns),
             "drift_12_1": d121, "drift_12_0": d120,
             "drift_12_1_above_ma50": gated(d121, keep(above50), mask),
             "drift_12_1_above_ma200": gated(d121, keep(above200), mask),
             "drift_12_0_above_ma50": gated(d120, keep(above50), mask)}
        cohorts.append(c)
        labels = quantile_labels(d121, c["returns"], 5)
        if labels.any():
            universe = np.nanmean(c["returns"][labels > 0])
            top = labels == 5
            row = {"date": date, "universe": universe, "top_share_above_ma50": float(np.nanmean(above50[top]))}
            for name, side in [("top_above_ma50", 1.0), ("top_below_ma50", 0.0)]:
                pick = top & (above50 == side)
                row[name] = float(np.mean(c["returns"][pick]) - universe) if pick.sum() >= 5 else np.nan
            diagnostic.append(row)

    results = {}
    for name in cfg["variants"]:
        out = evaluate(cohorts, name, us["quantiles"], us["cost_one_way"], us["bootstrap_block_months"],
                       us["bootstrap_repetitions"])
        daily = pd.DataFrame(out.pop("daily"))
        excess = daily.top_excess_net.to_numpy()
        boot = block_bootstrap(excess, block_days=us["bootstrap_block_months"], repetitions=us["bootstrap_repetitions"])
        daily["year"] = daily.date.str[:4]
        results[name] = {
            "description": cfg["variants"][name], "periods": out["periods"], "rank_ic_mean": out["rank_ic_mean"],
            "rank_ic_t": out["rank_ic_t"], "top_turnover": out["top_turnover_mean"],
            "top_excess_net": out["top_excess_net"], "top_excess_net_monthly_ci": boot.get("confidence_interval"),
            "long_short_net": out["long_short_net"],
            "top_excess_net_by_year": (daily.groupby("year").top_excess_net.sum()).round(4).to_dict()}
    diag = pd.DataFrame(diagnostic)
    summary = {"months": len(diag), "top_share_above_ma50_mean": float(diag.top_share_above_ma50.mean())}
    for name in ["top_above_ma50", "top_below_ma50"]:
        x = diag[name].dropna().to_numpy()
        b = block_bootstrap(x, block_days=us["bootstrap_block_months"], repetitions=us["bootstrap_repetitions"])
        summary[name] = {"months": len(x), "mean_monthly_excess": float(x.mean()),
                         "annualized": float(x.mean() * 12), "ci_monthly": b.get("confidence_interval")}
    output.mkdir(parents=True)
    diag.to_csv(output / "diagnostic.csv", index=False)
    write_json(output / "results.json", {"version": cfg["version"], "config": cfg, "source_manifest_sha256": source_hash,
                                         "variants": results, "diagnostic": summary,
                                         "elapsed_seconds": round(time.time() - started, 1)})
    print(f"wrote {output} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
