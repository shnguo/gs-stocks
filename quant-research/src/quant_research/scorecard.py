"""Forward scorecard: what a frozen screen run did after its signal date.

Same convention as the backtest (factors.forward_returns): signal at the close, entry at the
next session's open, exit at the open `holding` sessions later, and a stock that stops
trading is valued at its last traded close. A run is scored once, when its exit open exists.
Until then it only has an interim mark at the latest close, which never enters the record.
"""
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .factors import PricePanel, forward_returns, rank_ic
from .storage import file_hash

NEW_YORK = ZoneInfo("America/New_York")
MARKET_OPEN = time(9, 30)


def last_complete_day(now: datetime | None = None) -> str:
    """Latest New York date whose regular session is over (from 17:00 local)."""
    now = (now or datetime.now(NEW_YORK)).astimezone(NEW_YORK)
    return (now.date() if now.hour >= 17 else now.date() - timedelta(days=1)).isoformat()


def verified_files(folder: Path, files: dict[str, str]) -> None:
    """Raise if any file no longer matches the SHA-256 recorded when it was frozen."""
    for name, digest in files.items():
        if file_hash(folder / name) != digest:
            raise ValueError(f"File changed after it was frozen: {folder / name}")


def timing(created_at: str, entry_date: str) -> str:
    """'prospective' if the run was written before its entry session opened, else 'late'."""
    opened = datetime.combine(datetime.fromisoformat(entry_date).date(), MARKET_OPEN, NEW_YORK)
    return "prospective" if datetime.fromisoformat(created_at) < opened else "late"


def holding_returns(panel: PricePanel, as_of: str, holding: int) -> tuple[np.ndarray, dict]:
    """Per-stock return from the open after `as_of`, and where the holding period stands.

    mature: exit at the open `holding` sessions after entry (the backtest's own function).
    interim: marked at the last traded close so far. waiting: the entry session has not traded.
    """
    t, last = panel.index(as_of), len(panel.dates) - 1
    e, z = t + 1, t + 1 + holding
    if e > last:
        return np.full(len(panel.ids), np.nan), {
            "status": "waiting", "entry_date": None, "exit_date": None, "sessions_held": 0}
    if z <= last:
        values, _ = forward_returns(panel, as_of, panel.dates[z - 1])
        return values, {"status": "mature", "entry_date": panel.dates[e], "exit_date": panel.dates[z],
                        "sessions_held": holding}
    window = panel.traded[e:]
    row = e + window.shape[0] - 1 - np.argmax(window[::-1], axis=0)
    cols = np.arange(len(panel.ids))
    one_price = (panel.open[e] == panel.high[e]) & (panel.high[e] == panel.low[e])
    ok = (panel.traded[e] & ~(one_price & (panel.open[e] > panel.close[t]))
          & (panel.sequence[row, cols] == panel.sequence[e]))
    with np.errstate(divide="ignore", invalid="ignore"):
        values = panel.close[row, cols] / panel.open[e] - 1
    return np.where(ok, values, np.nan), {
        "status": "interim", "entry_date": panel.dates[e], "exit_date": None, "sessions_held": last - t}


def basket_members(ranking: pd.DataFrame, spec: dict) -> np.ndarray | None:
    """Boolean membership from a ranking column; None when that run has no such column."""
    if spec["column"] not in ranking:
        return None
    values = ranking[spec["column"]]
    if values.dtype == object:
        values = values.map({"True": 1.0, "False": 0.0, True: 1.0, False: 0.0})
    values = values.to_numpy(dtype=float)
    return values <= spec["max"] if "max" in spec else values == float(spec["equals"])


def number(value) -> float | None:
    return round(float(value), 6) if value is not None and np.isfinite(value) else None


def basket_outcome(returns: np.ndarray, members: np.ndarray, universe: float, benchmark: float,
                   round_trip_cost: float, min_stocks: int) -> dict:
    """Equal-weight basket: gross, net of a full round trip, and excess over both yardsticks."""
    held = returns[members & np.isfinite(returns)]
    out = {"listed": int(members.sum()), "stocks": int(len(held)), "gross": None, "net": None,
           "excess_universe": None, "excess_benchmark": None}
    if len(held) < min_stocks:
        return out
    gross = float(held.mean())
    net = gross - round_trip_cost
    return {**out, "gross": number(gross), "net": number(net),
            "excess_universe": number(net - universe), "excess_benchmark": number(net - benchmark)}


def score_run(ranking: pd.DataFrame, panel: PricePanel, benchmark_id: str, as_of: str,
              cfg: dict) -> tuple[dict, pd.DataFrame]:
    """Outcome of one run and the per-stock returns behind it.

    `panel` holds bars from `as_of` onward keyed by ranking symbol, plus the benchmark.
    """
    values, state = holding_returns(panel, as_of, cfg["holding_sessions"])
    position = panel.ids.get_indexer(ranking.symbol)
    returns = np.where(position >= 0, values[position], np.nan)
    t = panel.index(as_of)
    close = np.where(position >= 0, panel.close[t][position], np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        moved = np.abs(close / ranking.last_close.to_numpy(float) - 1) > cfg["price_check_tolerance"]
    outcome = {
        "as_of": as_of, **state, "sessions_left": None, "scored": int(len(ranking)),
        "priced": int(np.isfinite(returns).sum()), "price_mismatches": int(moved.sum()),
        "universe_return": None, "benchmark_return": None, "rank_ic": None, "quintiles": None, "baskets": {},
    }
    stocks = pd.DataFrame({"symbol": ranking.symbol, "rank": ranking["rank"],
                           "drift_quintile": ranking.drift_quintile, "return": returns})
    if state["status"] == "waiting":
        outcome["sessions_left"] = cfg["holding_sessions"] + 1
        return outcome, stocks
    exit_index = t + 1 + cfg["holding_sessions"]
    outcome["sessions_left"] = max(0, exit_index - (len(panel.dates) - 1))
    universe = float(np.nanmean(returns))
    benchmark = float(values[panel.ids.get_loc(benchmark_id)])
    if not np.isfinite(benchmark):
        raise ValueError(f"No benchmark return for {as_of}")
    quintile = ranking.drift_quintile.to_numpy(float)
    outcome.update({
        "universe_return": number(universe), "benchmark_return": number(benchmark),
        "rank_ic": number(rank_ic(ranking.drift_percentile.to_numpy(float), returns)),
        "quintiles": [number(np.nanmean(returns[quintile == q])) if np.isfinite(returns[quintile == q]).any()
                      else None for q in range(1, 6)],
    })
    for spec in cfg["baskets"]:
        members = basket_members(ranking, spec)
        if members is not None:
            outcome["baskets"][spec["key"]] = basket_outcome(
                returns, members, universe, benchmark, 2 * cfg["cost_one_way"], cfg["min_stocks"])
    return outcome, stocks


def independent(runs: list[dict]) -> list[dict]:
    """Non-overlapping runs: the earliest, then each next one entered on or after the last exit."""
    chosen = []
    for run in sorted(runs, key=lambda r: r["as_of"]):
        if not chosen or run["entry_date"] >= chosen[-1]["exit_date"]:
            chosen.append(run)
    return chosen


def summarize(runs: list[dict], basket_keys: list[str], min_independent: int) -> dict:
    """Record over mature runs, split by timing; overlapping and independent runs side by side."""
    def stats(group: list[dict], key: str) -> dict:
        rows = [r["baskets"][key] for r in group if r["baskets"].get(key, {}).get("net") is not None]
        if not rows:
            return {"runs": 0, "mean_net": None, "mean_excess_universe": None,
                    "mean_excess_benchmark": None, "share_beating_universe": None}
        excess = np.array([r["excess_universe"] for r in rows])
        return {"runs": len(rows), "mean_net": number(np.mean([r["net"] for r in rows])),
                "mean_excess_universe": number(excess.mean()),
                "mean_excess_benchmark": number(np.mean([r["excess_benchmark"] for r in rows])),
                "share_beating_universe": number((excess > 0).mean())}

    out = {}
    for label in ["prospective", "late"]:
        group = [r for r in runs if r["status"] == "mature" and r["timing"] == label]
        alone = independent(group)
        ics = [r["rank_ic"] for r in alone if r["rank_ic"] is not None]
        out[label] = {
            "runs": len(group), "independent_runs": len(alone),
            "enough": label == "prospective" and len(alone) >= min_independent,
            "mean_rank_ic_independent": number(np.mean(ics)) if ics else None,
            "baskets": {key: {"all": stats(group, key), "independent": stats(alone, key)} for key in basket_keys},
        }
    return out
