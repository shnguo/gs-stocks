"""Publication deadlines and durable per-attempt stage measurements."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import pandas as pd

from .storage import utc_now, write_json


def launch_plan(calendar, signal_date, store, now, after="16:30", margin_minutes=15):
    """An offline launch budget from measured new cycles; no invented historical timings."""
    import json
    import math

    now = pd.Timestamp(now)
    if now.tzinfo is None or margin_minutes < 0:
        raise ValueError("Launch plan requires a timezone and nonnegative margin")
    dates = sorted(calendar.loc[calendar.is_open.astype(bool), "date"].unique())
    if signal_date not in dates:
        raise ValueError("Launch signal must be a verified trading session")
    horizons = dates[dates.index(signal_date) + 1:dates.index(signal_date) + 6]
    if len(horizons) != 5:
        raise ValueError("Launch calendar needs five future sessions")
    deadline = pd.Timestamp(horizons[0] + "T09:15:00", tz="Asia/Shanghai")
    earliest = pd.Timestamp(signal_date + "T" + after + ":00", tz="Asia/Shanghai")
    attempts = [json.loads(path.read_text()) for path in Path(store).glob("operations/*.json")]
    attempts.sort(key=lambda row: row["started_at"])
    new_runs = [run for run in attempts if run["status"] == "completed" and
                any(stage["stage"].startswith("train_seed_") for stage in run["stages"])]
    stages = {}
    for run in new_runs[-20:]:
        for row in run["stages"]:
            if row["status"] == "completed" and "seconds" in row:
                stages.setdefault(row["stage"], []).append(row["seconds"])
    def p90(values):
        return sorted(values)[max(0, math.ceil(len(values) * .9) - 1)] if values else None
    durations = [run["seconds"] for run in new_runs[-20:] if "seconds" in run]
    estimate = p90(durations)
    latest = deadline - pd.Timedelta(float(estimate + margin_minutes * 60), unit="s") if estimate is not None else None
    status = ("waiting_for_close" if now < earliest else "missed_deadline" if now >= deadline else
              "insufficient_measured_budget" if latest is not None and now >= latest else "can_start")
    return {"signal_date": signal_date, "horizon_dates": horizons, "checked_at": now.isoformat(),
            "earliest_start": earliest.isoformat(), "deadline": deadline.isoformat(), "status": status,
            "seconds_until_deadline": max(0, (deadline - now).total_seconds()),
            "measured_cycles": len(durations), "p90_cycle_seconds": estimate,
            "publication_margin_minutes": margin_minutes,
            "recommended_latest_start": latest.isoformat() if latest is not None else None,
            "bottlenecks": sorted([{"stage": name, "p90_seconds": p90(values), "samples": len(values)}
                                   for name, values in stages.items()],
                                  key=lambda item: -item["p90_seconds"]),
            "timing_basis": "measured_new_cycles" if durations else "unmeasured"}


def publication_status(signal_date, horizon_dates, published_at):
    if not horizon_dates or sorted(set(horizon_dates)) != list(horizon_dates):
        raise ValueError("Forecast sessions must be nonempty, unique and ordered")
    if signal_date >= horizon_dates[0]:
        raise ValueError("Forecast sessions must follow the completed signal close")
    published = pd.Timestamp(published_at)
    if published.tzinfo is None:
        raise ValueError("Publication time must include its timezone")
    deadline = pd.Timestamp(horizon_dates[0] + "T09:15:00", tz="Asia/Shanghai")
    if published < pd.Timestamp(signal_date + "T15:00:00", tz="Asia/Shanghai"):
        raise ValueError("Publication precedes the completed signal close")
    return {
        "signal_date": signal_date,
        "published_at": published.isoformat(),
        "deadline": deadline.isoformat(),
        "prospective": bool(published < deadline),
        "minutes_late": max(0.0, (published - deadline).total_seconds() / 60),
    }


class OperationJournal:
    """Sidecar measurements never change a frozen model run or its manifest."""

    def __init__(self, store):
        self.path = Path(store) / "operations" / (uuid4().hex + ".json")
        self.value = {"started_at": utc_now(), "status": "running", "stages": []}
        self.start = perf_counter()
        self.flush()

    def flush(self):
        write_json(self.path, self.value)

    def set_signal(self, signal):
        self.value["signal_date"] = signal
        self.flush()

    @contextmanager
    def stage(self, name):
        row = {"stage": name, "started_at": utc_now(), "status": "running"}
        self.value["stages"].append(row)
        self.flush()
        start = perf_counter()
        try:
            yield
        except BaseException as error:
            row.update(status="failed", error_type=type(error).__name__)
            raise
        else:
            row["status"] = "completed"
        finally:
            row.update(completed_at=utc_now(), seconds=perf_counter() - start)
            self.flush()

    def finish(self, status):
        self.value.update(status=status, completed_at=utc_now(), seconds=perf_counter() - self.start)
        self.flush()
