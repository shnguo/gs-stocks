from __future__ import annotations

import pandas as pd

from .storage import Snapshot


def cutoff(date: str, time: str = "20:00") -> pd.Timestamp:
    return pd.Timestamp(f"{date} {time}", tz="Asia/Shanghai").tz_convert("UTC")


def risk_status(snapshot: Snapshot, date: str, at: pd.Timestamp) -> dict[str, str]:
    instruments = snapshot.tables["instruments"]
    coverage = snapshot.tables["risk_coverage"]
    complete = set(coverage.loc[
        (coverage.date == date) & coverage.complete.astype(bool) & (coverage.known_at <= at),
        "exchange"])
    result = {r.instrument_id: "normal" if r.exchange in complete else "unknown"
              for r in instruments.itertuples()}
    events = snapshot.tables["risk_events"]
    selected = events.loc[(events.start_date <= date) &
                          ((events.end_date == "") | (events.end_date >= date)) &
                          (events.known_at <= at)].sort_values("known_at")
    # Conflicting latest revisions cannot silently become normal.
    for instrument_id, group in selected.groupby("instrument_id", sort=False):
        latest = group.loc[group.known_at == group.known_at.max(), "status"].unique()
        result[instrument_id] = latest[0] if len(latest) == 1 else "unknown"
    return result


def universe_on(snapshot: Snapshot, date: str, signal_time: str = "20:00", *,
                risk_policy: str = "exclude_st") -> pd.DataFrame:
    if risk_policy not in {"include", "exclude_st"}:
        raise ValueError("Invalid universe risk policy")
    rows = snapshot.tables["instruments"].copy()
    states = risk_status(snapshot, date, cutoff(date, signal_time))
    rows["date"] = date
    rows["risk_status"] = rows.instrument_id.map(states)
    rows["reason"] = "eligible"
    if risk_policy == "exclude_st":
        rows.loc[rows.risk_status != "normal", "reason"] = rows.risk_status.map(
            {"ST": "risk_warning", "*ST": "risk_warning", "unknown": "risk_status_unknown"})
    rows.loc[rows.listed_at > date, "reason"] = "not_yet_listed"
    rows.loc[(rows.delisted_at != "") & (rows.delisted_at < date), "reason"] = "delisted"
    rows["eligible"] = rows.reason == "eligible"
    return rows


def audit_snapshot(snapshot: Snapshot, signal_time: str = "20:00", min_years: int = 10, *,
                   training_risk_policy: str = "exclude_st") -> dict:
    daily = []
    bars = snapshot.tables["bars"]
    present = set(zip(bars.instrument_id, bars.date))
    for date in snapshot.dates:
        frame = universe_on(snapshot, date, signal_time, risk_policy=training_risk_policy)
        for exchange, group in frame.groupby("exchange"):
            counts = group.reason.value_counts().to_dict()
            missing = sum((r.instrument_id, date) not in present
                          for r in group.loc[group.eligible].itertuples())
            daily.append({"date": date, "exchange": exchange,
                          "counts": {k: int(v) for k, v in counts.items()},
                          "eligible_risk_status_counts": {
                              k: int(v) for k, v in group.loc[group.eligible].risk_status
                              .value_counts().items()},
                          "eligible_without_bar": missing})
    year_counts = (bars.assign(year=bars.date.str[:4])
                   .merge(snapshot.tables["instruments"][["instrument_id", "exchange", "board"]],
                          on="instrument_id")
                   .groupby(["year", "exchange", "board"]).agg(
                       rows=("date", "size"), instruments=("instrument_id", "nunique"))
                   .reset_index().to_dict("records"))
    training_blockers = snapshot.formal_blockers(
        min_years, require_st_history=training_risk_policy == "exclude_st",
        require_portfolio_data=False)
    backtest_blockers = snapshot.formal_blockers(min_years)
    unknown = any(d["counts"].get("risk_status_unknown", 0) or
                  d["eligible_risk_status_counts"].get("unknown", 0) for d in daily)
    if unknown:
        backtest_blockers.append("risk_status_unknown_in_expected_universe")
        if training_risk_policy == "exclude_st":
            training_blockers.append("risk_status_unknown_in_expected_universe")
    blockers = sorted(set(training_blockers + backtest_blockers))
    return {"dataset_id": snapshot.dataset_id, "declaration": snapshot.manifest["declaration"],
            "training_risk_policy": training_risk_policy, "trading_risk_policy": "exclude_st",
            "first_date": snapshot.dates[0] if snapshot.dates else None,
            "last_date": snapshot.dates[-1] if snapshot.dates else None,
            "instruments": len(snapshot.tables["instruments"]), "bar_rows": len(bars),
            "year_exchange_board": year_counts, "daily": daily,
            "training_data_blockers": sorted(set(training_blockers)),
            "training_data_ready": not training_blockers,
            "backtest_data_blockers": sorted(set(backtest_blockers)),
            "backtest_data_ready": not backtest_blockers,
            "formal_blockers": blockers, "formal_ready": not blockers}
