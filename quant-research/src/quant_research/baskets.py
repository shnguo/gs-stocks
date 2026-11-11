"""Monthly rule-based baskets: equal weight, excess over the equal-weight pool, turnover cost."""
import numpy as np
import pandas as pd

from quant_research.evaluation import block_bootstrap
from quant_research.factors import performance

MIN_STOCKS = 5


class BasketBook:
    """Collects one row per month for a fixed set of baskets."""

    def __init__(self, names, cost_one_way):
        self.names, self.cost = list(names), cost_one_way
        self.previous: dict[str, set] = {}
        self.rows: list[dict] = []

    def add(self, date, ids, returns, valid, picks, long_returns=None):
        """picks: {name: bool array}. returns: one-month forward return. long_returns: optional
        longer-horizon forward return (gross diagnostic)."""
        universe = float(np.mean(returns[valid]))
        row = {"date": date, "pool": int(valid.sum()), "universe": universe}
        long_ok = long_returns is not None and np.isfinite(long_returns).sum() > 100
        if long_ok:
            long_valid = valid & np.isfinite(long_returns)
            long_universe = float(np.mean(long_returns[long_valid]))
        for name in self.names:
            pick = picks[name] & valid
            n = int(pick.sum())
            chosen = set(ids[pick])
            turnover = (1 - len(chosen & self.previous[name]) / len(chosen)) if name in self.previous and chosen else 1.0
            self.previous[name] = chosen
            gross = float(np.mean(returns[pick])) - universe if n >= MIN_STOCKS else np.nan
            row.update({f"{name}_n": n, f"{name}_excess": gross,
                        f"{name}_excess_net": gross - 2 * self.cost * turnover, f"{name}_turnover": turnover})
            if long_ok:
                lp = pick & long_valid
                row[f"{name}_excess_long"] = (float(np.mean(long_returns[lp])) - long_universe
                                              if lp.sum() >= MIN_STOCKS else np.nan)
        self.rows.append(row)

    def frame(self):
        frame = pd.DataFrame(self.rows)
        frame["year"] = frame.date.str[:4]
        return frame

    def summary(self, descriptions, block, repetitions):
        frame = self.frame()
        out = {}
        for name in self.names:
            x = frame[f"{name}_excess_net"].dropna()
            boot = block_bootstrap(x.to_numpy(), block_days=block, repetitions=repetitions)
            by_year = frame.groupby("year")[f"{name}_excess_net"].sum().round(4)
            res = {"description": descriptions.get(name, ""), "months": len(x),
                   "mean_stocks": float(frame[f"{name}_n"].mean()),
                   "median_stocks": float(frame[f"{name}_n"].median()),
                   "mean_turnover": float(frame.loc[frame[f"{name}_n"] >= MIN_STOCKS, f"{name}_turnover"].mean()),
                   "excess_net": performance(x.to_numpy()), "annual_mean_net": float(x.mean() * 12),
                   "monthly_ci": boot.get("confidence_interval"),
                   "losing_years": int((by_year < 0).sum()), "years": int(by_year.notna().sum()),
                   "by_year": by_year.to_dict()}
            if f"{name}_excess_long" in frame:
                y = frame[f"{name}_excess_long"].dropna()
                b = block_bootstrap(y.to_numpy(), block_days=block, repetitions=repetitions)
                res["long_horizon_gross"] = {"months": len(y), "mean": float(y.mean()) if len(y) else None,
                                             "ci": b.get("confidence_interval")}
            out[name] = res
        return out


def top_n(score, mask, n, groups=None, cap=None):
    """Boolean pick of the n highest finite scores within mask. With groups and cap, walks down
    the ranking and skips a stock once its group already holds cap names (a missing group is
    one shared bucket), so the pick can end with fewer than n names."""
    score = np.asarray(score, dtype=float)
    order = np.flatnonzero(mask & np.isfinite(score))
    order = order[np.argsort(-score[order], kind="stable")]
    pick = np.zeros(len(score), dtype=bool)
    if groups is None or cap is None:
        pick[order[:n]] = True
        return pick
    held: dict = {}
    for i in order:
        if pick.sum() >= n:
            break
        group = groups[i] if pd.notna(groups[i]) else "unknown"
        if held.get(group, 0) < cap:
            pick[i] = True
            held[group] = held.get(group, 0) + 1
    return pick
