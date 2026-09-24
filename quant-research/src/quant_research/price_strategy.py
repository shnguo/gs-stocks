"""Five-session price forecasts and explicitly hypothetical daily-bar trade diagnostics.

This module does not certify execution or replace the portfolio cash ledger.
Every order expires after day 1; every position reaches its time exit on day 5.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product

import numpy as np
import pandas as pd

FIELDS = ("open", "high", "low", "close")
QUANTILES = np.array([.1, .5, .9])
HORIZON = 5


def price_labels(bars: pd.DataFrame, rows: pd.DataFrame, dates: list[str],
                 actions: pd.DataFrame, boundary: str) -> dict[str, np.ndarray]:
    """Raw quote log ratios, with cumulative missing/action/episode barriers.

    Cohort selection is the caller's past-only decision. Do not drop rows with
    unknown outcomes. Do not smooth corporate actions into executable prices.
    """
    if rows.duplicated(["instrument_id", "date"]).any():
        raise ValueError("Duplicate signal")
    coordinates = rows.date_index.to_numpy(dtype=int)
    if (coordinates < 0).any() or (coordinates + HORIZON >= len(dates)).any():
        raise ValueError("Incomplete five-session calendar")
    if any(dates[t + HORIZON] >= boundary for t in coordinates):
        raise ValueError("Labels cross partition/holdout boundary")
    if any(dates[t] != d for t, d in zip(coordinates, rows.date)):
        raise ValueError("Signal/calendar mismatch")
    bars = bars.set_index(["instrument_id", "date"], verify_integrity=True)
    keys = pd.MultiIndex.from_tuples(
        [(s, dates[t + h]) for s, t in zip(rows.instrument_id, coordinates)
         for h in range(HORIZON + 1)], names=bars.index.names)
    frame = bars.reindex(keys)
    n = len(rows)
    prices = frame[list(FIELDS)].to_numpy(float).reshape(n, HORIZON + 1, 4)
    factors = frame.factor.to_numpy(float).reshape(n, HORIZON + 1)
    turnover = frame[["volume", "amount"]].to_numpy(float).reshape(n, HORIZON + 1, 2)
    status = frame.get("source_trade_status", pd.Series(1., index=frame.index))
    status = status.to_numpy(float).reshape(n, HORIZON + 1)
    good = (np.isfinite(prices).all(axis=2) & (prices > 0).all(axis=2)
            & np.isfinite(factors) & (factors > 0)
            & np.isfinite(turnover).all(axis=2) & (turnover > 0).all(axis=2)
            # Providers such as the BSE source omit this optional flag. Positive
            # observed turnover establishes a priced session; explicit 0 is a halt.
            & (status != 0))
    good &= ((prices[:, :, 1] >= prices.max(axis=2))
             & (prices[:, :, 2] <= prices.min(axis=2)))
    good &= np.isclose(factors, factors[:, :1], rtol=1e-8, atol=0)
    for col in ("sequence_id", "label_sequence_id"):
        if col in frame:
            seq = frame[col].fillna("").to_numpy().reshape(n, HORIZON + 1)
            good &= (seq == seq[:, :1]) & (seq != "")
    events = set(zip(actions.instrument_id, actions.ex_date))
    for i, (s, t) in enumerate(zip(rows.instrument_id, coordinates)):
        for h in range(1, HORIZON + 1):
            if (s, dates[t + h]) in events:
                good[i, h] = False
    valid = np.logical_and.accumulate(good, axis=1)[:, 1:]
    reference = prices[:, 0, 3]
    if not np.isfinite(reference).all() or (reference <= 0).any():
        raise ValueError("Signal quote unavailable; retain in input coverage before labelling")
    with np.errstate(divide="ignore", invalid="ignore"):
        targets = np.log(prices[:, 1:] / reference[:, None, None])
    targets[~valid] = np.nan
    return {"targets": targets, "reference": reference, "future": prices[:, 1:],
            "valid": valid, "turnover": turnover[:, 1:],
            "upper": pd.to_numeric(frame.upper_limit, errors="coerce").to_numpy(float).reshape(n, 6)[:, 1:],
            "lower": pd.to_numeric(frame.lower_limit, errors="coerce").to_numpy(float).reshape(n, 6)[:, 1:]}


def ordered_quantiles(predictions: np.ndarray) -> np.ndarray:
    """Monotone rearrangement of independently fitted marginal quantiles."""
    if predictions.ndim != 4 or predictions.shape[1:] != (5, 4, 3):
        raise ValueError("Expected [signal, day, OHLC, quantile]")
    result = np.sort(predictions, axis=-1)
    # Quantiles of H/L must respect those of O/C; this is not a joint path.
    result[:, :, 1] = np.maximum(result[:, :, 1], result[:, :, [0, 3]].max(axis=2))
    result[:, :, 2] = np.minimum(result[:, :, 2], result[:, :, [0, 3]].min(axis=2))
    return result


def forecast_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict:
    if predictions.shape != (*targets.shape, 3) or targets.shape[1:] != (5, 4):
        raise ValueError("Forecast/target shape mismatch")
    result = []
    for h, field in product(range(5), range(4)):
        y, p = targets[:, h, field], predictions[:, h, field]
        good = np.isfinite(y) & np.isfinite(p).all(axis=1)
        e = y[good, None] - p[good]
        loss = np.maximum(QUANTILES * e, (QUANTILES - 1) * e)
        result.append({"day": h + 1, "field": FIELDS[field],
            "cohort_rows": len(y), "scored_rows": int(good.sum()),
            "pinball": float(loss.mean()) if good.any() else None,
            "median_log_mae": float(abs(e[:, 1]).mean()) if good.any() else None,
            "interval_80_coverage": float(((e[:, 0] >= 0) & (e[:, 2] <= 0)).mean())
                if good.any() else None,
            "quantile_frequencies": [(float((e[:, j] <= 0).mean()) if good.any() else None)
                                     for j in range(3)]})
    return {"targets": result, "formal_ready": False}


@dataclass(frozen=True)
class TradeAssumptions:
    notional: float = 10000.
    commission: float = .0003
    minimum_fee: float = 5.
    sell_tax: float = .0005
    slippage_bps: float = 5.
    tick: float = .01
    t_plus: int = 1
    # This is a per-order diagnostic, no portfolio capital/lot/capacity claims.

    def __post_init__(self):
        values = np.array(list(asdict(self).values()), float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Invalid cost/execution assumptions")
        if self.notional <= 0 or self.tick <= 0 or self.t_plus < 1:
            raise ValueError("Invalid notional, tick or settlement")


@dataclass(frozen=True)
class Plan:
    buy: float
    take_profit: float
    stop: float

    def __post_init__(self):
        if not np.isfinite([self.buy, self.take_profit, self.stop]).all():
            raise ValueError("Non-finite plan")
        if not 0 < self.stop < self.buy < self.take_profit:
            raise ValueError("Invalid plan price ordering")


GRID = tuple(product((-.02, -.01, 0.), (.03, .06), (.02, .04)))


def candidate_plans(reference: float, assumptions: TradeAssumptions) -> list[Plan]:
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError("Invalid reference")
    tick = assumptions.tick
    plans = []
    for offset, up, down in GRID:
        buy = np.floor(reference * (1 + offset) / tick + 1e-9) * tick
        take = np.ceil(buy * (1 + up) / tick - 1e-9) * tick
        stop = np.floor(buy * (1 - down) / tick + 1e-9) * tick
        plans.append(Plan(round(buy, 8), round(take, 8), round(stop, 8)))
    return plans


def trade_diagnostic(prices: np.ndarray, valid: np.ndarray, plan: Plan,
                     assumptions: TradeAssumptions, upper=None, lower=None,
                     require_limits: bool = False) -> dict:
    """Conservative OHLC-touch scenario; never a proof of actual order execution.

    Strict penetration required for intraday limits. Buy day cannot sell.
    Stop wins an ambiguous same-day stop/target touch. A triggered but blocked
    stop stays active on following days. Unresolved positions are not zero P&L.
    """
    if prices.shape != (5, 4) or np.asarray(valid).shape != (5,):
        raise ValueError("Expected five OHLC bars and validity mask")
    up = np.full(5, np.nan) if upper is None else np.asarray(upper, float)
    lo = np.full(5, np.nan) if lower is None else np.asarray(lower, float)
    result = {"status": "unresolved", "filled": None, "net_return": None,
              "buy_price": None, "sell_price": None, "exit_day": None,
              "ambiguous": False, "limits_verified": bool(require_limits)}
    if not valid[0]:
        return {**result, "reason": "entry_data_or_action_barrier"}
    if require_limits and (not np.isfinite(up).all() or not np.isfinite(lo).all()
                           or (up <= lo).any() or (lo <= 0).any()):
        return {**result, "reason": "price_limits_unverified"}
    o, h, low, c = prices[0]
    slip = assumptions.slippage_bps / 10000
    if np.isfinite(up[0]) and (o >= up[0] or plan.buy > up[0]):
        return {**result, "status": "unfilled", "filled": False, "net_return": 0.,
                "reason": "entry_upper_limit"}
    if o <= plan.buy and o * (1 + slip) <= plan.buy:
        buy = o * (1 + slip)
    elif low < plan.buy - assumptions.tick / 2:
        buy = plan.buy
    else:
        return {**result, "status": "unfilled", "filled": False, "net_return": 0.,
                "reason": "entry_expired"}
    result.update(filled=True, buy_price=float(buy))
    stopped = False
    for d in range(1, 5):
        if not valid[d]:
            return {**result, "reason": "holding_data_or_action_barrier"}
        if d < assumptions.t_plus:
            continue
        o, h, low, c = prices[d]
        previously_stopped = stopped
        stopped |= low <= plan.stop
        sell, reason = None, ""
        if np.isfinite(lo[d]) and o <= lo[d]:
            # Conservative: do not presume recovery/queue execution later today.
            continue
        if stopped:
            sell = min(o, plan.stop) * (1 - slip)
            reason = "stop"
            result["ambiguous"] = bool(result["ambiguous"] or
                                       (h > plan.take_profit and o < plan.take_profit))
            # If open already above target, its auction exit precedes later low.
            if not previously_stopped and o * (1 - slip) >= plan.take_profit:
                sell, reason = plan.take_profit, "take_profit_at_open"
                result["ambiguous"] = False
        elif o * (1 - slip) >= plan.take_profit or h > plan.take_profit + assumptions.tick / 2:
            sell, reason = plan.take_profit, "take_profit"
        elif d == 4:
            sell, reason = c * (1 - slip), "day5_close"
        if sell is None or (np.isfinite(lo[d]) and sell <= lo[d]):
            continue
        quantity = assumptions.notional / buy
        entry_fee = max(assumptions.minimum_fee, assumptions.notional * assumptions.commission)
        proceeds = quantity * sell
        exit_fee = max(assumptions.minimum_fee, proceeds * assumptions.commission)
        net = (proceeds - exit_fee - proceeds * assumptions.sell_tax
               - assumptions.notional - entry_fee) / (assumptions.notional + entry_fee)
        return {**result, "status": "closed", "reason": reason, "sell_price": float(sell),
                "net_return": float(net), "exit_day": d + 1}
    return {**result, "reason": "exit_blocked_at_horizon"}


def plan_features(predictions: np.ndarray) -> np.ndarray:
    """Comparable forecast summaries; marginal quantiles are not sampled paths."""
    return np.column_stack([predictions[:, 0, 2, 1], predictions[:, 0, 0, 1],
                            predictions[:, 4, 3, 1],
                            predictions[:, 4, 3, 2] - predictions[:, 4, 3, 0]])


class PlanCalibrator:
    """Local empirical conditional outcomes from a separate calibration period.

    One neighborhood per candidate plan: do not confuse fill probability with
    unconditional upside. These small-sample estimates require later validation.
    """
    def __init__(self, neighbors=32, min_neighbors=20):
        if not 1 <= min_neighbors <= neighbors:
            raise ValueError("Invalid calibration neighborhood")
        self.neighbors, self.min_neighbors = neighbors, min_neighbors

    def fit(self, predictions, outcomes):
        from sklearn.neighbors import NearestNeighbors
        self.x = plan_features(predictions)
        self.center = np.nanmedian(self.x, axis=0)
        self.scale = np.maximum(np.nanstd(self.x, axis=0), 1e-4)
        self.models = []
        for j in range(len(GRID)):
            ys = [r[j] for r in outcomes]
            mask = np.array([r["net_return"] is not None for r in ys])
            mask &= np.isfinite(self.x).all(axis=1)
            ids = np.flatnonzero(mask)
            if len(ids) < self.min_neighbors:
                self.models.append(None)
                continue
            model = NearestNeighbors(n_neighbors=min(self.neighbors, len(ids)))
            model.fit((self.x[ids] - self.center) / self.scale)
            self.models.append((model, ids, ys))
        return self

    def estimate(self, predictions):
        x = plan_features(predictions)
        estimates = [[None for _ in GRID] for _ in x]
        good = np.flatnonzero(np.isfinite(x).all(axis=1))
        if not len(good):
            return estimates
        # Constant baselines share forecasts across the whole cohort. Query each
        # distinct point once instead of repeating a degenerate tree search.
        unique, inverse = np.unique(x[good], axis=0, return_inverse=True)
        for j, item in enumerate(self.models):
            if item is None:
                continue
            model, ids, ys = item
            # A constant forecast cannot discriminate among historical examples.
            # Use all its calibration outcomes, not an arbitrary first k ties.
            if np.all(self.x[ids] == self.x[ids[0]]):
                estimate = conditional_estimate([ys[k] for k in ids])
                for i in good:
                    if np.linalg.norm((x[i] - self.x[ids[0]]) / self.scale) <= 4:
                        estimates[i][j] = estimate
                continue
            distances, indices = model.kneighbors((unique - self.center) / self.scale)
            cached = []
            for ds, neighbors in zip(distances, indices):
                # A forecast far outside calibration support must abstain.
                if ds[-1] > 4:
                    cached.append(None)
                    continue
                rows = [ys[k] for k in ids[neighbors]]
                cached.append(conditional_estimate(rows))
            for i, k in zip(good, inverse):
                estimates[i][j] = cached[k]
        return estimates


def conditional_estimate(rows):
    r = np.array([v["net_return"] for v in rows], float)
    filled = np.array([v["filled"] for v in rows], bool)
    return {"neighbors": len(rows), "fill_probability": float(filled.mean()),
        "expected_net_per_order": float(r.mean()),
        "expected_net_if_filled": float(r[filled].mean()) if filled.any() else None,
        "loss_probability_if_filled": float((r[filled] < 0).mean()) if filled.any() else None,
        "downside_per_order": float(np.minimum(r, 0).mean()),
        "probability_basis": "calibration_neighborhood_hypothetical_fills"}


def choose_plan(estimates, min_fill=.2, min_expected=.001, downside_penalty=.5):
    candidates = []
    for j, e in enumerate(estimates):
        if e is None or e["fill_probability"] < min_fill:
            continue
        utility = e["expected_net_per_order"] + downside_penalty * e["downside_per_order"]
        if utility >= min_expected:
            candidates.append((utility, -j, j))
    return max(candidates)[2] if candidates else None
