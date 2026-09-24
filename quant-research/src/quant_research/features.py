from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .labels import wealth_state
from .storage import Snapshot


def stock_features(bars: pd.DataFrame) -> pd.DataFrame:
    """All operations are backward looking on the exchange calendar, never compressed bars."""
    if "sequence_id" in bars:
        # Reindexing introduces calendar gaps. Forward fill only the episode
        # identity, never prices or factors; missing observations stay missing.
        sequence = bars.sequence_id.ffill().fillna("")
        episodes = sequence.ne(sequence.shift()).cumsum()
        return pd.concat([_stock_features(group) for _, group in bars.groupby(episodes, sort=False)])
    return _stock_features(bars)


def _stock_features(bars: pd.DataFrame) -> pd.DataFrame:
    adjusted = bars.close * bars.factor
    ret = adjusted.pct_change(fill_method=None)
    values = {}
    for window in [1, 2, 3, 5, 10, 20, 60]:
        values[f"return_{window}"] = adjusted / adjusted.shift(window) - 1
    for window in [5, 10, 20, 60]:
        values[f"volatility_{window}"] = ret.rolling(window).std()
        values[f"price_mean_{window}"] = adjusted / adjusted.rolling(window).mean() - 1
    values["range"] = (bars.high - bars.low) / bars.close
    values["body"] = bars.close / bars.open - 1
    values["gap"] = bars.open * bars.factor / adjusted.shift(1) - 1
    values["close_position"] = ((bars.close - bars.low) / (bars.high - bars.low)
                                .replace(0, np.nan)).fillna(0.5).where(bars.close.notna())
    values["log_volume"] = np.log1p(bars.volume)
    values["log_amount"] = np.log1p(bars.amount)
    for window in [5, 20]:
        values[f"volume_mean_{window}"] = bars.volume / bars.volume.rolling(window).mean() - 1
        values[f"amount_mean_{window}"] = bars.amount / bars.amount.rolling(window).mean() - 1
    return pd.DataFrame(values, index=bars.index).replace([np.inf, -np.inf], np.nan)


@dataclass
class Panel:
    values: np.ndarray
    dates: list[str]
    instruments: list[str]
    feature_names: list[str]
    samples: pd.DataFrame
    exclusions: pd.DataFrame
    lookback: int
    horizon: int
    dataset_id: str
    risk_policy: str = "exclude_st"

    def windows(self, rows: pd.DataFrame) -> np.ndarray:
        # Materialize one mini-batch, not the entire all-market sequence tensor.
        return np.stack([self.values[int(s), int(t) - self.lookback + 1:int(t) + 1]
                         for s, t in zip(rows.stock_index, rows.date_index)])


def build_panel(snapshot: Snapshot, horizon: int, lookback: int = 60,
                signal_time: str = "20:00", *, risk_policy: str = "exclude_st",
                coverage_evidence: dict | None = None, masked: bool = False,
                progress=None) -> Panel:
    if horizon not in {5, 20} or lookback < 2:
        raise ValueError("Unsupported target horizon/lookback")
    from .coverage import INACTIVE, SUSPENSIONS, load_coverage
    from .universe import universe_on
    dates = snapshot.dates
    instruments = sorted(snapshot.tables["instruments"].instrument_id)
    date_index = pd.Index(dates, name="date")
    day_values = np.asarray(dates, dtype=object)
    n = len(dates)
    meta = snapshot.tables["instruments"].set_index("instrument_id")
    exceptions = load_coverage(snapshot, coverage_evidence)
    exception_days = {}
    for (symbol, day), kind in exceptions.items():
        exception_days.setdefault(symbol, {})[day] = kind
    risk_names = np.asarray(["normal", "ST", "*ST", "unknown"], dtype=object)
    reason_names = np.asarray(["eligible", "not_yet_listed", "delisted",
                               "risk_warning", "risk_status_unknown"], dtype=object)
    risk_codes = {v: i for i, v in enumerate(risk_names)}
    reason_codes = {v: i for i, v in enumerate(reason_names)}
    risks = np.empty((len(instruments), n), dtype=np.uint8)
    reasons = np.empty_like(risks)
    for t, day in enumerate(dates):
        state = universe_on(snapshot, day, signal_time, risk_policy=risk_policy)
        state = state.set_index("instrument_id").loc[instruments]
        risks[:, t] = state.risk_status.map(risk_codes).to_numpy(dtype=np.uint8)
        reasons[:, t] = state.reason.map(reason_codes).to_numpy(dtype=np.uint8)
    bars = snapshot.tables["bars"]
    grouped = bars.groupby("instrument_id", sort=False).indices
    action_groups = {k: v for k, v in snapshot.tables["actions"].groupby("instrument_id")}
    samples, exclusions, arrays = [], [], []
    for stock_index, symbol in enumerate(instruments):
        rows = bars.iloc[grouped.get(symbol, [])]
        frame = rows.set_index("date").reindex(date_index)
        ex = exception_days.get(symbol, {})
        halt = date_index.isin([d for d, k in ex.items() if k in SUSPENSIONS])
        inactive = date_index.isin([d for d, k in ex.items() if k in INACTIVE])
        feature_frame = frame.copy()
        if masked:
            # Research imputation uses only the last observed close within an
            # episode, solely on independently evidenced suspension dates.
            # Original bars and executable endpoint prices remain untouched.
            seq = frame.sequence_id.ffill() if "sequence_id" in frame else pd.Series(symbol, index=date_index)
            episode = seq.fillna("").ne(seq.fillna("").shift()).cumsum()
            for _, group in feature_frame.groupby(episode, sort=False):
                mask = group.index.isin(date_index[halt]) & group.close.isna().to_numpy()
                if not mask.any():
                    continue
                carry = group.close.ffill()
                carry_factor = group.factor.ffill()
                for col in ["open", "high", "low", "close"]:
                    feature_frame.loc[group.index[mask], col] = carry.loc[group.index[mask]]
                feature_frame.loc[group.index[mask], "factor"] = carry_factor.loc[group.index[mask]]
                feature_frame.loc[group.index[mask], ["volume", "amount"]] = 0.
            for col in ["sequence_id", "label_sequence_id"]:
                if col in feature_frame:
                    feature_frame.loc[halt, col] = frame[col].ffill().loc[halt]
            features = stock_features(feature_frame)
            # A genuinely all-suspended rolling window has zero turnover; its
            # turnover ratio is neutral and explicitly masked, never infinite.
            for length in [5, 20]:
                for field in ["volume", "amount"]:
                    zero = feature_frame[field].rolling(length).sum().eq(0)
                    features.loc[zero, f"{field}_mean_{length}"] = 0.
            features["suspended"] = (halt | frame.get("source_trade_status", pd.Series(1, index=date_index)).eq(0)).astype(float)
            features["unpriced_suspension"] = (halt & frame.close.isna()).astype(float)
        else:
            features = stock_features(frame)
        names = list(features.columns)
        array = features.to_numpy(dtype=np.float32)
        arrays.append(array)
        complete = pd.Series(np.isfinite(array).all(axis=1)).rolling(lookback).sum().eq(lookback).to_numpy()
        why = reason_names[reasons[stock_index]].copy()
        why[(why == "eligible") & inactive] = "verified_inactive_code_or_listing"
        why[(why == "eligible") & ~complete] = "feature_history_incomplete"
        eligible = why == "eligible"
        risk = risk_names[risks[stock_index]]
        exclusions.append(pd.DataFrame({"date": day_values[~eligible], "instrument_id": symbol,
                                        "reason": why[~eligible], "risk_status": risk[~eligible]}))
        t = np.flatnonzero(eligible)
        entry = t + 1
        end = t + horizon + 1
        mature = end < n
        ee = np.minimum(entry, n - 1)
        xx = np.minimum(end, n - 1)
        opening = frame.open.to_numpy(dtype=float)
        factor = frame.factor.to_numpy(dtype=float)
        endpoint = np.full(n, "available", dtype=object)
        status = frame.get("source_trade_status", pd.Series(1, index=date_index)).to_numpy()
        suspended = ((status == 0) | (frame.volume.to_numpy(dtype=float) <= 0)
                     | (frame.amount.to_numpy(dtype=float) <= 0))
        endpoint[suspended] = "suspended_or_no_turnover"
        endpoint[~np.isfinite(opening * factor) | (opening * factor <= 0)] = "missing_price"
        endpoint[halt] = "suspended_or_no_turnover"
        delisted = str(meta.loc[symbol, "delisted_at"])
        if delisted:
            endpoint[day_values > delisted] = "after_delisting_without_settlement"
        valid = mature & (endpoint[ee] == "available") & (endpoint[xx] == "available")
        crosses = np.zeros(len(t), dtype=bool)
        sequence_frame = feature_frame if masked else frame
        for col in ["sequence_id", "label_sequence_id"]:
            if col in sequence_frame:
                codes, _ = pd.factorize(sequence_frame[col], sort=False)
                changes = np.r_[0, np.cumsum(codes[1:] != codes[:-1])]
                crosses |= (changes[xx] != changes[ee]) | (codes[ee] < 0)
        label_status = np.full(len(t), "not_matured", dtype=object)
        label_status[mature] = "missing_execution_endpoint"
        label_status[valid & crosses] = "unverified_corporate_action_boundary"
        valid &= ~crosses
        label_status[valid] = "available"
        shares, cash = wealth_state(date_index, action_groups.get(symbol, snapshot.tables["actions"].iloc[:0]))
        returns = np.full(len(t), np.nan)
        en, out = ee[valid], xx[valid]
        returns[valid] = ((opening[out] * shares[out] + cash[out] - cash[en]) /
                          shares[en] / opening[en] - 1)
        samples.append(pd.DataFrame({"date": day_values[t], "instrument_id": symbol,
            "stock_index": stock_index, "date_index": t, "entry_date": np.where(entry < n, day_values[ee], ""),
            "label_end": np.where(mature, day_values[xx], ""), "forward_return": returns,
            "label_status": label_status, "risk_status": risk[t], "trading_eligible": risk[t] == "normal",
            "entry_status": np.where(mature, endpoint[ee], "not_matured"),
            "exit_status": np.where(mature, endpoint[xx], "not_matured"),
            "source_is_st": frame.get("source_is_st", pd.Series(np.nan, index=date_index)).to_numpy()[t]}))
        if progress is not None and ((stock_index + 1) % 100 == 0 or stock_index + 1 == len(instruments)):
            progress(stock_index + 1, len(instruments))
    sample_frame = pd.concat(samples, ignore_index=True)
    grouped_returns = sample_frame.groupby("date").forward_return
    counts = grouped_returns.transform("count")
    sample_frame["target"] = ((grouped_returns.rank(method="average") - 1) /
                              (counts - 1).where(counts > 1) - .5)
    # Fixed signal-date cohort benchmark is undefined if any cohort terminal
    # value is unresolved. Preserve that fact instead of dropping those stocks.
    cohort_complete = grouped_returns.transform("size").eq(counts)
    sample_frame["benchmark_return"] = grouped_returns.transform("mean").where(cohort_complete)
    sample_frame["excess_return"] = sample_frame.forward_return - sample_frame.benchmark_return
    sample_frame["benchmark_status"] = np.where(cohort_complete, "available", "unresolved_cohort_terminal")
    return Panel(np.stack(arrays), dates, instruments, names, sample_frame,
                 pd.concat(exclusions, ignore_index=True), lookback, horizon, snapshot.dataset_id, risk_policy)

@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, panel: Panel, training: pd.DataFrame) -> "Standardizer":
        if training.empty:
            raise ValueError("No training samples after purging")
        # Unique training tokens only: validation and test windows cannot affect statistics.
        mask = np.zeros(panel.values.shape[:2], dtype=bool)
        for s, t in zip(training.stock_index, training.date_index):
            mask[int(s), int(t) - panel.lookback + 1:int(t) + 1] = True
        values = panel.values[mask].astype(np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite training features")
        return cls(values.mean(axis=0), np.maximum(values.std(axis=0), 1e-6))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.clip((values - self.mean) / self.scale, -5, 5).astype(np.float32)
