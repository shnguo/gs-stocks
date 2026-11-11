"""Backtest rules that cut the US screen to a 10 or 20 stock shortlist.

Usage: uv run python scripts/shortlist_check.py configs/shortlist-check-v1.json

Every rule picks N names a month from a group the pullback backtest already covers (the top
fifth by drift, or its mild-pullback part) and is compared with that parent group and with
random N-name picks from it. Rules, sizes and the decision rule are in the config and were
fixed before running. Research output, not a trade list.
"""
import argparse
import importlib.util
import json
import time
from functools import cache
from pathlib import Path

import numpy as np

from quant_research.baskets import MIN_STOCKS, BasketBook, top_n
from quant_research.evaluation import block_bootstrap
from quant_research.factors import forward_returns, physics_momentum, residualize
from quant_research.storage import write_json

BASE = Path(__file__).resolve().parents[1]
MILD, TREND = "depth_0_10", "trend_all"
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]


@cache
def load_script(name):
    spec = importlib.util.spec_from_file_location(name, BASE / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def basket_names(cfg):
    """{basket: (parent group, description)} for every rule and size."""
    names = {f"top_drift_{n}": (TREND, cfg["rules"]["top_drift"]) for n in [*cfg["sizes"], 50]}
    for rule in ["mild_drift", "mild_controlled", "mild_sector", "mild_lowvol"]:
        names.update({f"{rule}_{n}": (MILD, cfg["rules"][rule]) for n in cfg["sizes"]})
    return names


def shortlist_picks(drift, controlled, volatility, sector, groups, pool, cfg):
    """Every shortlist for one signal date. groups: the pullback baskets of that date."""
    mild = groups[MILD]
    picks = {f"top_drift_{n}": top_n(drift, pool, n) for n in [*cfg["sizes"], 50]}
    for n in cfg["sizes"]:
        cap = max(1, round(cfg["sector_cap_share"] * n))
        picks[f"mild_drift_{n}"] = top_n(drift, mild, n)
        picks[f"mild_controlled_{n}"] = top_n(controlled, mild, n)
        picks[f"mild_sector_{n}"] = top_n(drift, mild, n, sector, cap)
        picks[f"mild_lowvol_{n}"] = top_n(-volatility, mild, n)
    return picks


class RandomPaths:
    """Independent paths that draw n names from a group every month, booked like BasketBook."""

    def __init__(self, n, paths, cost_one_way, seed):
        self.n, self.paths, self.cost = n, paths, cost_one_way
        self.rng = np.random.default_rng(seed)
        self.previous = None
        self.net, self.turnover = [], []

    def add(self, group, returns, valid, universe):
        members = np.flatnonzero(group)
        chosen = np.zeros((self.paths, len(returns)), dtype=bool)
        if len(members):
            draw = np.argsort(self.rng.random((self.paths, len(members))), axis=1)[:, :self.n]
            np.put_along_axis(chosen, members[draw], True, axis=1)
        chosen &= valid
        count = chosen.sum(axis=1)
        turnover = np.ones(self.paths)
        if self.previous is not None:
            kept = (chosen & self.previous).sum(axis=1)
            turnover = np.where(count > 0, 1 - kept / np.maximum(count, 1), 1.0)
        self.previous = chosen
        mean = np.where(chosen, returns, 0.0).sum(axis=1) / np.maximum(count, 1)
        gross = np.where(count >= MIN_STOCKS, mean - universe, np.nan)
        self.net.append(gross - 2 * self.cost * turnover)
        self.turnover.append(np.where(count >= MIN_STOCKS, turnover, np.nan))

    def summary(self):
        net = np.array(self.net)                       # months x paths
        annual = np.nanmean(net, axis=0) * 12
        return {"paths": self.paths, "months": int(np.isfinite(net).all(axis=1).sum()),
                "annual_mean_net": dict(zip(map(str, QUANTILES), np.quantile(annual, QUANTILES).tolist())),
                "annual_volatility_median": float(np.median(np.nanstd(net, axis=0, ddof=1) * np.sqrt(12))),
                "worst_month_median": float(np.median(np.nanmin(net, axis=0))),
                "mean_turnover": float(np.nanmean(self.turnover))}, annual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    cfg = json.loads(parser.parse_args().config.read_text())
    output = BASE / cfg["output"]
    if output.exists():
        raise FileExistsError(f"Use a new output path: {output}")
    started = time.time()
    us = json.loads((BASE / cfg["us_config"]).read_text())
    pcfg = json.loads((BASE / cfg["pullback_config"]).read_text())
    pullback = load_script("pullback_check")
    panel, months, source_hash = load_script("physics_robustness").us_months(us, cfg["industry"])
    m = us["momentum"]
    names = basket_names(cfg)
    descriptions = {**cfg["groups"], **{k: v[1] for k, v in names.items()}}
    book = BasketBook(descriptions, us["cost_one_way"])
    random = {n: RandomPaths(n, cfg["random_paths"], us["cost_one_way"], cfg["random_seed"] + n)
              for n in cfg["sizes"]}
    for date, following, mask, sector in months:
        t = panel.index(date)
        returns, _ = forward_returns(panel, date, following)
        phys = physics_momentum(panel, date, m["lookback"], m["skip"], m["min_traded"], pcfg["jump_sigma"])
        drift = np.where(mask, phys["drift"], np.nan)
        pool = mask & np.isfinite(drift)
        valid = pool & np.isfinite(returns)
        if valid.sum() < 100:
            continue
        groups, _ = pullback.pullback_picks(panel, t, drift, mask, pcfg)
        controlled = residualize(phys["drift"], [np.expm1(phys["total"]), phys["volatility"],
                                                 phys["traded_value"]], pool)
        picks = shortlist_picks(drift, controlled, phys["volatility"], sector, groups, pool, cfg)
        picks.update({TREND: groups[TREND], MILD: groups[MILD]})
        book.add(date, np.asarray(panel.ids), returns, valid, picks)
        universe = float(np.mean(returns[valid]))
        for paths in random.values():
            paths.add(groups[MILD], np.where(valid, returns, 0.0), valid, universe)

    frame = book.frame()
    block, reps = us["bootstrap_block_months"], us["bootstrap_repetitions"]
    results = book.summary(descriptions, block, reps)
    random_out, random_annual = {}, {}
    for n, paths in random.items():
        random_out[str(n)], random_annual[n] = paths.summary()
    for name, (parent, _) in names.items():
        own, base = frame[f"{name}_excess_net"], frame[f"{parent}_excess_net"]
        diff = (own - base).dropna()
        boot = block_bootstrap(diff.to_numpy(), block_days=block, repetitions=reps)
        res = results[name]
        res.update({"parent": parent, "worst_month": float(own.min()),
                    "vs_parent": {"annual_mean": float(diff.mean() * 12),
                                  "monthly_ci": boot.get("confidence_interval"),
                                  "share_of_months_better": float((diff > 0).mean())}})
        size = int(name.rsplit("_", 1)[1])
        if parent == MILD and size in random_annual:
            res["random_percentile"] = float((random_annual[size] < res["annual_mean_net"]).mean())
        ci = res["monthly_ci"]
        res["supported"] = bool(ci and ci[0] > 0 and res["annual_mean_net"] >= results[parent]["annual_mean_net"])
    for group in [TREND, MILD]:
        results[group]["worst_month"] = float(frame[f"{group}_excess_net"].min())

    output.mkdir(parents=True)
    frame.to_csv(output / "baskets.csv", index=False)
    write_json(output / "results.json", {
        "version": cfg["version"], "config": cfg, "source_manifest_sha256": source_hash,
        "baskets": results, "random": random_out, "elapsed_seconds": round(time.time() - started, 1)})
    print(f"wrote {output} in {time.time() - started:.0f}s")
    print(f"{'basket':22}{'n':>5}{'net/yr':>8}{'95% interval/yr':>20}{'vol':>7}{'worst':>8}{'lose':>6}"
          f"{'turn':>6}{'vs parent':>10}{'rand%':>7}  ok")
    for name, r in results.items():
        ci = r["monthly_ci"] or [np.nan, np.nan]
        vs = r.get("vs_parent", {}).get("annual_mean")
        print(f"{name:22}{r['median_stocks']:5.0f}{r['annual_mean_net'] * 100:7.1f}%"
              f"{f'[{ci[0] * 1200:+.1f}, {ci[1] * 1200:+.1f}]':>20}"
              f"{r['excess_net']['annual_volatility'] * 100:6.1f}%{r['worst_month'] * 100:7.1f}%"
              f"{r['losing_years']:>3}/{r['years']:<2}{r['mean_turnover'] * 100:5.0f}%"
              f"{'' if vs is None else f'{vs * 100:+.1f}':>10}"
              f"{'' if 'random_percentile' not in r else f'{r['random_percentile'] * 100:.0f}':>7}"
              f"  {'yes' if r.get('supported') else ''}")
    for n, r in random_out.items():
        q = r["annual_mean_net"]
        print(f"random {n} from {MILD}: annual net 5/50/95% = {q['0.05'] * 100:+.1f} / {q['0.5'] * 100:+.1f} / "
              f"{q['0.95'] * 100:+.1f}, vol {r['annual_volatility_median'] * 100:.1f}%, "
              f"worst month {r['worst_month_median'] * 100:.1f}%, turnover {r['mean_turnover'] * 100:.0f}%")


if __name__ == "__main__":
    main()
