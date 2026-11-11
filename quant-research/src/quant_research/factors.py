"""Monthly cross-sectional factor research: momentum, short-term reversal and value.

Research-only. Valuation and industry inputs are retrospective provider vintages,
not point-in-time publications, so value results can be optimistic. Prices use
backward-adjusted bars (close * factor); a return never spans two bar sequences.

Timing: factors are measured at the signal-date close. Positions enter at the next
session's open and exit at the open of the session after the next signal date.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import block_bootstrap
from .storage import file_hash


@dataclass
class PricePanel:
    dates: list[str]
    ids: pd.Index
    close: np.ndarray      # adjusted close, NaN without a bar
    open: np.ndarray       # adjusted open
    high: np.ndarray       # adjusted high
    low: np.ndarray        # adjusted low
    raw_close: np.ndarray  # unadjusted close, for joining provider valuations
    traded: np.ndarray     # bar exists with positive volume and no halt flag
    st: np.ndarray         # 1 = ST, 0 = normal, NaN = unknown
    sequence: np.ndarray   # integer bar-sequence code, -1 without a bar
    dollar_volume: np.ndarray | None = None  # traded value: amount, else close x volume

    def index(self, date: str) -> int:
        return self.dates.index(date)


def price_panel(bars: pd.DataFrame) -> PricePanel:
    dates = sorted(bars.date.unique())
    ids = pd.Index(sorted(bars.instrument_id.unique()))
    t = pd.Index(dates).get_indexer(bars.date)
    i = ids.get_indexer(bars.instrument_id)
    shape = (len(dates), len(ids))

    def wide(values, fill=np.nan, dtype=float):
        out = np.full(shape, fill, dtype=dtype)
        out[t, i] = values
        return out

    factor = bars.factor.to_numpy(float)
    status = bars.source_trade_status.to_numpy(float) if "source_trade_status" in bars else np.ones(len(bars))
    traded = (bars.volume.to_numpy(float) > 0) & (status != 0)
    st = bars.source_is_st.to_numpy(float) if "source_is_st" in bars else np.zeros(len(bars))
    seq = bars.sequence_id if "sequence_id" in bars else bars.instrument_id
    codes = pd.factorize(seq)[0]
    return PricePanel(
        dates=dates, ids=ids,
        close=wide(bars.close.to_numpy(float) * factor),
        open=wide(bars.open.to_numpy(float) * factor),
        high=wide(bars.high.to_numpy(float) * factor),
        low=wide(bars.low.to_numpy(float) * factor),
        raw_close=wide(bars.close.to_numpy(float)),
        traded=wide(traded, False, bool),
        st=wide(st),
        sequence=wide(codes, -1, np.int64),
        dollar_volume=wide(bars.amount.to_numpy(float) if "amount" in bars
                           else bars.close.to_numpy(float) * bars.volume.to_numpy(float)),
    )


def month_end_dates(dates: list[str]) -> list[str]:
    frame = pd.Series(dates)
    return frame.groupby(frame.str[:7]).max().tolist()


def last_per_month(dates) -> list[str]:
    return month_end_dates(sorted(dates))


def eligible(panel: PricePanel, date: str, instruments: pd.DataFrame, *,
             min_listing_days: int, exclude_exchanges: list[str]) -> np.ndarray:
    """Tradable, non-ST, seasoned stocks on the signal date. Unknown ST status is excluded."""
    t = panel.index(date)
    meta = instruments.set_index("instrument_id").reindex(panel.ids)
    listed = pd.to_datetime(meta.listed_at.replace("", None), errors="coerce")
    seasoned = (pd.Timestamp(date) - listed).dt.days.to_numpy() >= min_listing_days
    excluded = meta.exchange.isin(exclude_exchanges).to_numpy()
    return panel.traded[t] & (panel.st[t] == 0) & seasoned & ~excluded


def _past_return(panel: PricePanel, t: int, start: int, end: int, min_traded: int) -> np.ndarray:
    """Adjusted close return from t-start to t-end within one bar sequence."""
    if t - start < 0:
        return np.full(len(panel.ids), np.nan)
    a, b = panel.close[t - start], panel.close[t - end]
    same = (panel.sequence[t - start] == panel.sequence[t - end]) & (panel.sequence[t - end] >= 0)
    count = panel.traded[t - start:t - end + 1].sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = b / a - 1
    return np.where(same & (count >= min_traded) & np.isfinite(value), value, np.nan)


def momentum(panel: PricePanel, date: str, lookback: int = 252, skip: int = 21,
             min_traded: int = 200) -> np.ndarray:
    """12-1 month momentum: return from t-lookback to t-skip. Higher = stronger trend."""
    return _past_return(panel, panel.index(date), lookback, skip, min_traded)


def physics_momentum(panel: PricePanel, date: str, lookback: int = 252, skip: int = 21,
                     min_traded: int = 200, jump_sigma: float = 3.0) -> dict[str, np.ndarray]:
    """Momentum read as a ball rolling on the price track, over the 12-1 window.

    total    log of the 12-1 return (same endpoints as `momentum`)
    impulse  sum of daily log returns on jump days, |r - median| > jump_sigma x robust sigma
             (the pushes); drift = total - impulse (the coasting part)
    smooth   12-1 return x (1 - ID), ID = sign(return) x (share of down days - share of up
             days): a steady move counts up to double, a move made of a few jumps counts less
    mass     money needed to move the price: mean traded value / mean |daily return|
    volatility, traded_value  daily log-return std and log mean traded value (controls)
    jump_days  number of jump days in the window
    """
    t = panel.index(date)
    n = len(panel.ids)
    nan = np.full(n, np.nan)
    if t - lookback < 0:
        return {k: nan.copy() for k in ["total", "impulse", "drift", "smooth", "mass", "volatility",
                                        "traded_value", "jump_days"]}
    window = slice(t - lookback, t - skip + 1)
    close, seq = panel.close[window], panel.sequence[window]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(close[1:] / close[:-1])
    r = np.where((seq[1:] == seq[:-1]) & (seq[1:] >= 0) & np.isfinite(r), r, np.nan)
    valid = np.isfinite(r)
    count = valid.sum(axis=0)
    ret = _past_return(panel, t, lookback, skip, min_traded)
    ok = np.isfinite(ret) & (count >= min_traded)

    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        total = np.log1p(ret)
        median = np.nanmedian(r, axis=0)
        sigma = 1.4826 * np.nanmedian(np.abs(r - median), axis=0)
        sigma = np.where(sigma > 0, sigma, np.nanstd(r, axis=0))  # MAD is 0 on discrete paths
        jump = valid & (np.abs(r - median) > jump_sigma * sigma)
        impulse = np.where(jump, r, 0.0).sum(axis=0)
        up = (r > 0).sum(axis=0) / count
        down = (r < 0).sum(axis=0) / count
        discreteness = np.sign(ret) * (down - up)
        traded_value = np.where(valid, panel.dollar_volume[window][1:], np.nan) \
            if panel.dollar_volume is not None else np.full(r.shape, np.nan)
        mass = np.nanmean(traded_value, axis=0) / np.nanmean(np.abs(r), axis=0)
        volatility = np.nanstd(r, axis=0)
        level = np.log(np.nanmean(traded_value, axis=0))
    out = {"total": total, "impulse": impulse, "drift": total - impulse,
           "smooth": ret * (1 - discreteness), "mass": np.where(mass > 0, mass, np.nan),
           "volatility": volatility, "traded_value": level,
           "jump_days": jump.sum(axis=0).astype(float)}
    return {k: np.where(ok & np.isfinite(v), v, np.nan) for k, v in out.items()}


def reversal(panel: PricePanel, date: str, lookback: int = 21, min_traded: int = 15) -> np.ndarray:
    """Negative last-month return, oriented so that higher = expected to do better."""
    return -_past_return(panel, panel.index(date), lookback, 0, min_traded)


def forward_returns(panel: PricePanel, date: str, exit_signal: str) -> tuple[np.ndarray, np.ndarray]:
    """Next-open entry to the open after the next signal. Returns (returns, reason codes).

    Reasons: 0 ok, 1 entry not tradable, 2 entry one-price limit-up, 3 no exit price,
    4 sequence break. A stock that stops trading before exit is valued at its last
    traded close in the same sequence (delisting/suspension), not dropped.
    """
    t, x = panel.index(date), panel.index(exit_signal)
    if t + 1 >= len(panel.dates) or x + 1 >= len(panel.dates) or x <= t:
        raise ValueError("Signal or exit has no observable next session")
    e, z = t + 1, x + 1
    entry = panel.open[e]
    reason = np.zeros(len(panel.ids), dtype=np.int8)
    reason[~panel.traded[e]] = 1
    one_price = (panel.open[e] == panel.high[e]) & (panel.high[e] == panel.low[e])
    reason[(reason == 0) & one_price & (panel.open[e] > panel.close[t])] = 2
    exit_price = np.where(panel.traded[z], panel.open[z], np.nan)
    exit_seq = np.where(panel.traded[z], panel.sequence[z], -1)
    missing = ~np.isfinite(exit_price)
    if missing.any():
        window = panel.traded[e:z + 1][:, missing]
        last = window.shape[0] - 1 - np.argmax(window[::-1], axis=0)
        has = window.any(axis=0)
        cols = np.flatnonzero(missing)
        rows = e + last
        exit_price[cols[has]] = panel.close[rows[has], cols[has]]
        exit_seq[cols[has]] = panel.sequence[rows[has], cols[has]]
    reason[(reason == 0) & ~np.isfinite(exit_price)] = 3
    reason[(reason == 0) & (exit_seq != panel.sequence[e])] = 4
    with np.errstate(divide="ignore", invalid="ignore"):
        value = exit_price / entry - 1
    return np.where(reason == 0, value, np.nan), reason


def load_valuation(source_root: Path, sealed_from: str) -> tuple[pd.DataFrame, dict, list[str]]:
    """Eastmoney daily valuation records, hash-checked against the capture index.

    Captures whose records were pruned are skipped and returned as missing dates;
    a present but changed file is an error.
    """
    source_root = Path(source_root)
    index = json.loads((source_root / "index.json").read_text())
    frames, lineage, missing = [], {}, []
    for day, item in sorted(index.items()):
        if day >= sealed_from:
            continue
        capture = source_root / item["capture"]
        manifest_path = capture / "manifest.json"
        records = capture / "records.json"
        if not records.exists():
            missing.append(day)
            continue
        if file_hash(manifest_path) != item["capture_sha256"]:
            raise ValueError(f"Valuation capture changed: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if manifest["status"] != "completed" or file_hash(records) != manifest["files"]["records.json"]:
            raise ValueError(f"Valuation records changed: {records}")
        frame = pd.DataFrame(json.loads(records.read_text()))
        if len(frame) != item["rows"] or not frame.date.eq(day).all():
            raise ValueError("Valuation row/date mismatch")
        frames.append(frame)
        lineage[str(manifest_path)] = item["capture_sha256"]
    if not frames:
        raise ValueError("No valuation records available")
    frame = pd.concat(frames, ignore_index=True)
    if frame.duplicated(["instrument_id", "date"]).any():
        raise ValueError("Duplicate valuation key")
    return frame, lineage, missing


VALUE_COMPONENTS = {"earnings_yield": "pe_ttm", "book_to_price": "pb",
                    "sales_yield": "ps_ttm", "cashflow_yield": "pcf_ttm"}


def value_factors(panel: PricePanel, date: str, valuation: pd.DataFrame) -> pd.DataFrame:
    """Yield-style value components (1 / multiple) and log float cap, aligned to panel ids.

    A provider row is used only if its close matches the bar close within 0.01.
    """
    t = panel.index(date)
    rows = valuation.loc[valuation.date == date].set_index("instrument_id").reindex(panel.ids)
    close = pd.to_numeric(rows.close, errors="coerce").to_numpy(float)
    matched = np.isfinite(close) & (np.abs(close - panel.raw_close[t]) <= 0.010001)
    out = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for name, field in VALUE_COMPONENTS.items():
            multiple = pd.to_numeric(rows[field], errors="coerce").to_numpy(float)
            out[name] = np.where(matched & (np.abs(multiple) > 1e-8), 1 / multiple, np.nan)
        cap = pd.to_numeric(rows.float_cap, errors="coerce").to_numpy(float)
        out["log_float_cap"] = np.where(matched & (cap > 0), np.log(cap), np.nan)
    out["valuation_matched"] = matched
    return pd.DataFrame(out, index=panel.ids)


def percentile(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Cross-sectional percentile rank in (0, 1] among masked finite values."""
    out = np.full(len(values), np.nan)
    use = mask & np.isfinite(values)
    if use.any():
        out[use] = pd.Series(values[use]).rank(pct=True).to_numpy()
    return out


def composite(columns: list[np.ndarray], mask: np.ndarray, min_components: int) -> np.ndarray:
    ranks = np.column_stack([percentile(c, mask) for c in columns])
    count = np.isfinite(ranks).sum(axis=1)
    total = np.where(np.isfinite(ranks), ranks, 0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count >= min_components, total / count, np.nan)


def neutralize(values: np.ndarray, industry: np.ndarray, size: np.ndarray) -> np.ndarray:
    """Residual of factor percentile on industry dummies and size percentile."""
    use = np.isfinite(values) & np.isfinite(size) & pd.notna(industry) & (industry != "")
    out = np.full(len(values), np.nan)
    if use.sum() < 50:
        return out
    y = percentile(values, use)[use]
    dummies = pd.get_dummies(pd.Series(industry[use]), dtype=float).to_numpy()
    x = np.column_stack([dummies, percentile(size, use)[use]])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    out[use] = y - x @ beta
    return out


def load_industries(roots: list[Path]) -> dict[str, Path]:
    """Completed baostock industry snapshots by requested date; pruned snapshots are skipped."""
    found: dict[str, Path] = {}
    for root in roots:
        for manifest in sorted(Path(root).rglob("manifest.json")):
            meta = json.loads(manifest.read_text())
            query = meta.get("query", {})
            complete = all((manifest.parent / spec["file"]).exists() for spec in meta.get("raw_files", []))
            if (meta.get("status") == "completed" and query.get("api") == "stock_industry"
                    and complete and (manifest.parent / "records.json").exists()):
                found.setdefault(query["date"], manifest.parent)
    return found


def industry_on(date: str, snapshots: dict[str, Path], ids: pd.Index, max_age_days: int,
                reader) -> tuple[np.ndarray, str | None]:
    """Latest industry snapshot not after date and within max_age_days; else all missing."""
    candidates = [d for d in snapshots if d <= date]
    if not candidates:
        return np.full(len(ids), None, dtype=object), None
    chosen = max(candidates)
    if (pd.Timestamp(date) - pd.Timestamp(chosen)).days > max_age_days:
        return np.full(len(ids), None, dtype=object), None
    frame = reader(snapshots[chosen], chosen)
    mapped = frame.set_index("instrument_id").industry.reindex(ids)
    return mapped.where(mapped.notna() & mapped.ne(""), None).to_numpy(dtype=object), chosen


# ---------- evaluation ----------

def rank_ic(factor: np.ndarray, returns: np.ndarray) -> float:
    use = np.isfinite(factor) & np.isfinite(returns)
    if use.sum() < 30:
        return np.nan
    return float(pd.Series(factor[use]).rank().corr(pd.Series(returns[use]).rank()))


def quantile_labels(factor: np.ndarray, returns: np.ndarray, quantiles: int) -> np.ndarray:
    """1..quantiles by factor rank among stocks with both values; 0 elsewhere."""
    labels = np.zeros(len(factor), dtype=np.int8)
    use = np.isfinite(factor) & np.isfinite(returns)
    if use.sum() < quantiles * 10:
        return labels
    ranks = pd.Series(factor[use]).rank(method="first", pct=True).to_numpy()
    labels[use] = np.clip(np.ceil(ranks * quantiles), 1, quantiles).astype(np.int8)
    return labels


def evaluate(cohorts: list[dict], factor: str, quantiles: int, cost_one_way: float,
             block: int, repetitions: int) -> dict:
    """cohorts: [{date, ids, factor values..., returns}] in date order."""
    daily, previous = [], {}
    for c in cohorts:
        values, returns = c[factor], c["returns"]
        labels = quantile_labels(values, returns, quantiles)
        if not labels.any():
            continue
        row = {"date": c["date"], "rank_ic": rank_ic(values, returns),
               "stocks": int((labels > 0).sum()),
               "universe_return": float(np.nanmean(returns[labels > 0]))}
        for q in range(1, quantiles + 1):
            members = labels == q
            row[f"q{q}"] = float(returns[members].mean())
            current = set(c["ids"][members])
            prior = previous.get(q)
            row[f"turnover_q{q}"] = (1 - len(current & prior) / len(current)) if prior else 1.0
            previous[q] = current
        daily.append(row)
    frame = pd.DataFrame(daily)
    if frame.empty:
        return {"status": "no_cohorts", "daily": []}
    top, bottom = f"q{quantiles}", "q1"
    frame["long_short"] = frame[top] - frame[bottom]
    frame["long_short_net"] = frame.long_short - 2 * cost_one_way * (
        frame[f"turnover_{top}"] + frame[f"turnover_{bottom}"])
    frame["top_excess"] = frame[top] - frame.universe_return
    frame["top_excess_net"] = frame.top_excess - 2 * cost_one_way * frame[f"turnover_{top}"]
    ic = frame.rank_ic.dropna()
    summary = {
        "status": "estimated", "periods": int(len(frame)),
        "first_date": frame.date.iloc[0], "last_date": frame.date.iloc[-1],
        "mean_stocks": float(frame.stocks.mean()),
        "rank_ic_mean": float(ic.mean()), "rank_ic_std": float(ic.std()),
        "rank_ic_ir_annual": float(ic.mean() / ic.std() * np.sqrt(12)) if ic.std() > 0 else None,
        "rank_ic_t": float(ic.mean() / ic.std() * np.sqrt(len(ic))) if ic.std() > 0 else None,
        "rank_ic_positive_share": float((ic > 0).mean()),
        "rank_ic_bootstrap": block_bootstrap(ic.to_numpy(), block_days=block, repetitions=repetitions),
        "quantile_mean_monthly": {f"q{q}": float(frame[f"q{q}"].mean()) for q in range(1, quantiles + 1)},
        "top_turnover_mean": float(frame[f"turnover_{top}"].iloc[1:].mean()) if len(frame) > 1 else None,
    }
    for name in ["long_short", "long_short_net", "top_excess", "top_excess_net"]:
        summary[name] = performance(frame[name].to_numpy())
    summary["long_short"]["bootstrap"] = block_bootstrap(
        frame.long_short.to_numpy(), block_days=block, repetitions=repetitions)
    return {**summary, "daily": frame.to_dict("records")}


def performance(monthly: np.ndarray) -> dict:
    r = np.asarray(monthly, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return {"periods": int(len(r))}
    nav = np.cumprod(1 + r)
    drawdown = nav / np.maximum.accumulate(nav) - 1
    vol = r.std(ddof=1) * np.sqrt(12)
    return {"periods": int(len(r)), "mean_monthly": float(r.mean()),
            "annual_return": float(nav[-1] ** (12 / len(r)) - 1),
            "annual_volatility": float(vol),
            "sharpe": float(r.mean() * 12 / vol) if vol > 0 else None,
            "max_drawdown": float(drawdown.min()), "hit_rate": float((r > 0).mean())}


def correlations(cohorts: list[dict], factors: list[str]) -> dict:
    """Mean cross-sectional Spearman correlation between factor pairs."""
    out = {}
    for a_i, a in enumerate(factors):
        for b in factors[a_i + 1:]:
            values = [rank_ic(c[a], c[b]) for c in cohorts]
            values = [v for v in values if np.isfinite(v)]
            out[f"{a}|{b}"] = float(np.mean(values)) if values else None
    return out


PHYSICS_FACTORS = ["momentum_drift", "momentum_impulse", "momentum_smooth", "momentum_mass",
                   "momentum_friction"]


def residualize(values: np.ndarray, controls: list[np.ndarray], mask: np.ndarray,
                groups: np.ndarray | None = None) -> np.ndarray:
    """Residual of the factor percentile on control percentiles (with intercept) and, when
    given, group dummies such as industry, within mask."""
    use = mask & np.isfinite(values)
    if controls:
        use &= np.all([np.isfinite(c) for c in controls], axis=0)
    if groups is not None:
        use &= pd.notna(groups)
    out = np.full(len(values), np.nan)
    if use.sum() < 50:
        return out
    y = percentile(values, use)[use]
    parts = [np.ones(use.sum()), *[percentile(c, use)[use] for c in controls]]
    if groups is not None:
        parts += list(pd.get_dummies(pd.Series(groups[use]), drop_first=True, dtype=float).to_numpy().T)
    x = np.column_stack(parts)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    out[use] = y - x @ beta
    return out


def group_percentile(values: np.ndarray, groups: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Percentile rank in (0, 1] within each group (e.g. industry), among masked finite values."""
    use = mask & np.isfinite(values) & pd.notna(groups)
    out = np.full(len(values), np.nan)
    if use.any():
        frame = pd.DataFrame({"v": values[use], "g": groups[use]})
        out[use] = frame.groupby("g").v.rank(pct=True).to_numpy()
    return out


def physics_factors(phys: dict[str, np.ndarray], mask: np.ndarray, momentum_values: np.ndarray,
                    reversal_values: np.ndarray) -> dict[str, np.ndarray]:
    """The pre-registered ball-on-a-track variants, restricted to mask.

    drift     coasting part of the 12-1 move (jump days removed)
    impulse   jump-day part only (diagnostic: the pushes)
    smooth    12-1 return scaled up for steady paths, down for jumpy ones
    mass      p = m x v: 12-1 log return x cross-sectional percentile of mass
    friction  12-1 momentum and 1-month reversal, equal-weight percentile composite
    """
    keep = lambda x: np.where(mask, x, np.nan)  # noqa: E731
    return {"momentum_drift": keep(phys["drift"]),
            "momentum_impulse": keep(phys["impulse"]),
            "momentum_smooth": keep(phys["smooth"]),
            "momentum_mass": keep(phys["total"] * percentile(phys["mass"], mask)),
            "momentum_friction": composite([keep(momentum_values), keep(reversal_values)], mask, 2)}
