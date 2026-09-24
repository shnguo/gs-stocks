from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .event_store import SCORE_RANGES

POSITIVE_SCORE_FIELDS = (
    "certainty",
    "transmission",
    "expectation_gap",
    "flow_impact",
    "price_confirmation",
)
PENALTY_FIELDS = ("crowding_penalty", "gap_penalty", "liquidity_penalty")
EMPTY_RADAR_COLUMNS = [
    "event_id", "event_type", "scope", "instrument_id", "title", "scheduled_date",
    "actual_date", "available_at", "status", "event_score", "score_complete",
    "radar_state", "confirmation_passed", "reward_risk", "blockers",
]


def event_score(event: Mapping[str, object]) -> dict[str, object]:
    missing = [field for field in SCORE_RANGES if event.get(field) in {None, ""}]
    if missing:
        return {"event_score": None, "score_complete": False,
                "missing_score_fields": missing}
    positive = sum(float(event[field]) for field in POSITIVE_SCORE_FIELDS)
    penalties = sum(float(event[field]) for field in PENALTY_FIELDS)
    return {"event_score": float(np.clip(positive - penalties, 0, 100)),
            "score_complete": True, "missing_score_fields": []}


def _required_bar_columns(frame: pd.DataFrame) -> None:
    required = {"instrument_id", "date", "open", "high", "low", "close", "volume", "factor"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"bars missing columns: {sorted(missing)}")


def completed_technical_snapshot(bars: pd.DataFrame, completed_through: str) -> pd.DataFrame:
    """Calculate signals from completed adjusted daily bars only."""
    date.fromisoformat(completed_through)
    _required_bar_columns(bars)
    frame = bars.copy()
    frame["date"] = frame["date"].astype(str)
    frame = frame.loc[frame.date <= completed_through]
    if frame.duplicated(["instrument_id", "date"]).any():
        raise ValueError("duplicate instrument/date bars")
    numeric = ["open", "high", "low", "close", "volume", "factor"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame[numeric].to_numpy()).all():
        raise ValueError("bars contain non-finite values")
    if (frame[["open", "high", "low", "close", "factor"]] <= 0).any().any():
        raise ValueError("bars contain non-positive prices or factors")
    if (frame.volume < 0).any():
        raise ValueError("bars contain negative volume")

    rows: list[dict[str, object]] = []
    for instrument_id, group in frame.groupby("instrument_id", sort=True):
        group = group.sort_values("date").tail(40).copy()
        row: dict[str, object] = {
            "instrument_id": instrument_id,
            "last_completed_date": group.date.iloc[-1] if len(group) else "",
            "sufficient_history": len(group) >= 21,
        }
        if len(group) < 21:
            rows.append(row)
            continue
        factor = group.factor.to_numpy(dtype=float)
        close = group.close.to_numpy(dtype=float) * factor
        high = group.high.to_numpy(dtype=float) * factor
        low = group.low.to_numpy(dtype=float) * factor
        volume = group.volume.to_numpy(dtype=float)
        previous_close = np.r_[np.nan, close[:-1]]
        true_range = np.maximum.reduce([
            high - low,
            np.abs(high - previous_close),
            np.abs(low - previous_close),
        ])
        atr14 = float(np.nanmean(true_range[-14:]))
        ma5 = float(close[-5:].mean())
        previous_ma5 = float(close[-6:-1].mean())
        ma10 = float(close[-10:].mean())
        ma20 = float(close[-20:].mean())
        latest = float(close[-1])
        prior_volume = float(volume[-21:-1].mean())
        volume_ratio = float(volume[-1] / prior_volume) if prior_volume > 0 else None
        return10 = float(latest / close[-11] - 1)
        tight_base = bool(high[-5:].max() / low[-5:].min() - 1 <= 0.04)
        extension_flags = {
            "extension_atr": bool(atr14 > 0 and latest - ma5 > 0.8 * atr14),
            "extension_ma10": bool(latest / ma10 - 1 > 0.04),
            "extension_ma20": bool(latest / ma20 - 1 > 0.06),
            "extension_ten_session": bool(return10 > 0.10 and not tight_base),
        }
        row.update({
            "adjusted_close": latest,
            "ma5": ma5,
            "ma10": ma10,
            "ma20": ma20,
            "ma5_rising": bool(ma5 > previous_ma5),
            "above_ma5": bool(latest >= ma5),
            "atr14": atr14,
            "volume_ratio_20": volume_ratio,
            "return_10": return10,
            "tight_base_5": tight_base,
            **extension_flags,
            "extension_count": sum(extension_flags.values()),
            "overextended": sum(extension_flags.values()) >= 2,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _reward_risk(event: Mapping[str, object], close: float | None) -> float | None:
    if close is None or not np.isfinite(close):
        return None
    target = event.get("target_price")
    failure = event.get("failure_price")
    if target in {None, ""} or failure in {None, ""}:
        return None
    upside = float(target) - close
    downside = close - float(failure)
    if upside <= 0 or downside <= 0:
        return 0.0
    return upside / downside


def classify_event(event: Mapping[str, object], technical: Mapping[str, object] | None,
                   as_of_date: str, *, minimum_score: float = 50.0,
                   high_priority_score: float = 65.0) -> dict[str, object]:
    date.fromisoformat(as_of_date)
    scored = event_score(event)
    blockers: list[str] = []
    status = str(event["status"])
    if status == "cancelled":
        return {**scored, "radar_state": "avoid", "priority": "none",
                "confirmation_passed": False, "reward_risk": None,
                "blockers": ["event_cancelled"]}
    if status == "postponed":
        blockers.append("event_postponed")
    if status == "scheduled" and str(event["scheduled_date"]) < as_of_date:
        blockers.append("scheduled_event_overdue")
    if not scored["score_complete"]:
        blockers.append("event_score_incomplete")
    elif float(scored["event_score"]) < minimum_score:
        blockers.append("event_score_below_minimum")

    technical = dict(technical or {})
    price_data_stale = False
    if event.get("scope") == "instrument":
        if not technical or not technical.get("sufficient_history", False):
            blockers.append("insufficient_completed_price_history")
        else:
            last_completed = date.fromisoformat(str(technical["last_completed_date"]))
            price_data_stale = (date.fromisoformat(as_of_date) - last_completed).days > 5
            if price_data_stale:
                blockers.append("price_data_stale")
            if not technical.get("above_ma5", False):
                blockers.append("below_formal_ma5")
            if not technical.get("ma5_rising", False):
                blockers.append("formal_ma5_not_rising")
            if technical.get("overextended", False):
                blockers.append("price_overextended")

    close = technical.get("adjusted_close")
    reward_risk = _reward_risk(event, float(close) if close is not None else None)
    if event.get("scope") == "instrument":
        if reward_risk is None:
            blockers.append("reward_risk_not_defined")
        elif reward_risk < 2:
            blockers.append("reward_risk_below_two")

    completed = status == "completed"
    source_event_date = str(event.get("actual_date") or event["scheduled_date"])
    available_date = pd.Timestamp(str(event["available_at"])).tz_convert(
        "Asia/Shanghai"
    ).date().isoformat()
    event_date = max(source_event_date, available_date)
    post_event_close = bool(
        completed and technical.get("last_completed_date", "") > event_date
    )
    if completed and not post_event_close:
        blockers.append("first_post_event_close_pending")
    volume_confirmed = bool((technical.get("volume_ratio_20") or 0) >= 1)
    confirmation_passed = bool(
        completed
        and post_event_close
        and technical.get("above_ma5", False)
        and technical.get("ma5_rising", False)
        and not technical.get("overextended", True)
        and volume_confirmed
        and reward_risk is not None
        and reward_risk >= 2
        and not price_data_stale
    )

    fatal = {"event_cancelled", "event_score_below_minimum", "reward_risk_below_two"}
    if fatal.intersection(blockers):
        state = "avoid"
    elif (completed and not confirmation_passed) or "scheduled_event_overdue" in blockers:
        state = "wait_confirmation"
    else:
        state = "observe"
    score = scored["event_score"]
    priority = ("high" if score is not None and score >= high_priority_score and not blockers
                else "medium" if score is not None and score >= minimum_score
                else "low")
    return {**scored, "radar_state": state, "priority": priority,
            "confirmation_passed": confirmation_passed, "reward_risk": reward_risk,
            "blockers": sorted(set(blockers))}


def build_radar(events: Iterable[Mapping[str, object]], bars: pd.DataFrame,
                as_of_date: str, *, completed_through: str | None = None,
                past_days: int = 5, future_days: int = 60) -> pd.DataFrame:
    as_of = date.fromisoformat(as_of_date)
    start = (as_of - timedelta(days=past_days)).isoformat()
    end = (as_of + timedelta(days=future_days)).isoformat()
    completed_through = completed_through or as_of_date
    technical = completed_technical_snapshot(bars, completed_through)
    by_instrument = ({row.instrument_id: row._asdict() for row in technical.itertuples(index=False)}
                     if not technical.empty else {})
    rows = []
    for event in events:
        if not start <= str(event["scheduled_date"]) <= end:
            continue
        tech = by_instrument.get(str(event.get("instrument_id", "")))
        decision = classify_event(event, tech, as_of_date)
        rows.append({**event, **(tech or {}), **decision})
    if not rows:
        return pd.DataFrame(columns=EMPTY_RADAR_COLUMNS)
    result = pd.DataFrame(rows)
    result["priority_rank"] = result.priority.map({"high": 0, "medium": 1, "low": 2,
                                                   "none": 3}).fillna(4)
    result = result.sort_values(
        ["scheduled_date", "priority_rank", "event_score", "event_id"],
        ascending=[True, True, False, True], na_position="last",
    ).drop(columns="priority_rank").reset_index(drop=True)
    return result
