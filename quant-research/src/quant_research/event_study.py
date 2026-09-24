from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable, Mapping
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

SHANGHAI = ZoneInfo("Asia/Shanghai")


def _local_timestamp(value: str) -> pd.Timestamp:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("event availability timestamps must be timezone-aware")
    return pd.Timestamp(parsed.astimezone(SHANGHAI))


def _prepare_bars(bars: pd.DataFrame) -> pd.DataFrame:
    required = {"instrument_id", "date", "open", "factor"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"bars missing columns: {sorted(missing)}")
    frame = bars.copy()
    if frame.duplicated(["instrument_id", "date"]).any():
        raise ValueError("duplicate instrument/date bars")
    frame["adjusted_open"] = (pd.to_numeric(frame.open, errors="raise") *
                              pd.to_numeric(frame.factor, errors="raise"))
    if not np.isfinite(frame.adjusted_open).all() or (frame.adjusted_open <= 0).any():
        raise ValueError("invalid adjusted open prices")
    return frame.sort_values(["instrument_id", "date"])


def event_study(events: Iterable[Mapping[str, object]], bars: pd.DataFrame, *,
                mode: str, horizon_sessions: int = 5) -> tuple[pd.DataFrame, dict[str, object]]:
    if mode not in {"pre", "post"}:
        raise ValueError("mode must be pre or post")
    if horizon_sessions < 1:
        raise ValueError("horizon_sessions must be positive")
    frame = _prepare_bars(bars)
    groups = {key: value.reset_index(drop=True)
              for key, value in frame.groupby("instrument_id", sort=False)}
    results = []
    exclusions: dict[str, int] = {}

    def exclude(reason: str) -> None:
        exclusions[reason] = exclusions.get(reason, 0) + 1

    for event in events:
        instrument_id = str(event.get("instrument_id", ""))
        if event.get("scope") != "instrument" or instrument_id not in groups:
            exclude("not_an_instrument_or_missing_bars")
            continue
        if event.get("status") == "cancelled":
            exclude("cancelled")
            continue
        group = groups[instrument_id]
        dates = group.date.astype(str).tolist()
        scheduled = str(event["scheduled_date"])
        if scheduled not in dates:
            future = [index for index, value in enumerate(dates) if value >= scheduled]
            if not future:
                exclude("event_date_outside_bars")
                continue
            event_index = future[0]
        else:
            event_index = dates.index(scheduled)
        available = _local_timestamp(str(event["available_at"]))

        if mode == "pre":
            entry_index = event_index - horizon_sessions
            exit_index = event_index
            if entry_index < 0:
                exclude("insufficient_pre_event_history")
                continue
            entry_cutoff = pd.Timestamp(dates[entry_index], tz=SHANGHAI) + timedelta(
                hours=9, minutes=25
            )
            if available > entry_cutoff:
                exclude("event_not_known_before_entry")
                continue
        else:
            available_date = available.date().isoformat()
            same_day_before_open = available.hour < 9 or (
                available.hour == 9 and available.minute <= 25
            )
            candidates = [index for index, value in enumerate(dates)
                          if value > available_date or (value == available_date and
                                                        same_day_before_open)]
            if not candidates:
                exclude("no_tradable_session_after_publication")
                continue
            entry_index = candidates[0]
            exit_index = entry_index + horizon_sessions
            if exit_index >= len(group):
                exclude("outcome_not_mature")
                continue

        entry = float(group.iloc[entry_index].adjusted_open)
        exit_price = float(group.iloc[exit_index].adjusted_open)
        results.append({
            "event_id": event["event_id"],
            "event_type": event["event_type"],
            "instrument_id": instrument_id,
            "scheduled_date": scheduled,
            "entry_date": dates[entry_index],
            "exit_date": dates[exit_index],
            "entry_adjusted_open": entry,
            "exit_adjusted_open": exit_price,
            "gross_return": exit_price / entry - 1,
            "mode": mode,
            "horizon_sessions": horizon_sessions,
        })
    result = pd.DataFrame(results)
    summary = {
        "mode": mode,
        "horizon_sessions": horizon_sessions,
        "eligible_events": len(result),
        "excluded_events": sum(exclusions.values()),
        "exclusions": exclusions,
        "mean_gross_return": float(result.gross_return.mean()) if len(result) else None,
        "median_gross_return": float(result.gross_return.median()) if len(result) else None,
        "positive_rate": float((result.gross_return > 0).mean()) if len(result) else None,
        "execution_assumption": "adjusted_open_to_adjusted_open_without_costs",
    }
    return result, summary
