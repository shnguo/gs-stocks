"""Historical development partitions for earnings research; no fitting or return claims."""
from __future__ import annotations

import bisect

import pandas as pd


def checked_dates(dates):
    if not dates or dates != sorted(set(dates)):
        raise ValueError("Observed exchange dates must be sorted and unique")
    if not all(isinstance(d, str) and pd.Timestamp(d).date().isoformat() == d for d in dates):
        raise ValueError("Exchange dates require ISO calendar dates")


def historical_development_folds(dates, sealed_start):
    """Three calendar years, one year, next whole quarter; preserve existing holdout.

    The 21-session observation tail is the minimum for next-open entry and a normal
    20-session exit. Blocked exits can still extend beyond it and remain unknown.
    Label-end purging and a 20-session gap are applied separately to actual samples.
    """
    checked_dates(dates)
    if pd.Timestamp(sealed_start).date().isoformat() != sealed_start:
        raise ValueError("Sealed boundary must be an explicit ISO date")
    allowed = [d for d in dates if d < sealed_start]
    if not allowed:
        return []
    earliest = pd.Timestamp(allowed[0]) + pd.DateOffset(years=4)
    start = earliest.to_period("Q").start_time
    if start < earliest:
        start += pd.DateOffset(months=3)
    folds = []
    while start.strftime("%Y-%m-%d") < sealed_start:
        next_start = start + pd.DateOffset(months=3)
        end = (next_start + pd.DateOffset(days=-1)).strftime("%Y-%m-%d")
        observed = [d for d in allowed if start.strftime("%Y-%m-%d") <= d <= end]
        last = bisect.bisect_right(allowed, end) - 1
        if not observed or last + 21 >= len(allowed) or end >= sealed_start:
            break
        validation = start - pd.DateOffset(years=1)
        training = validation - pd.DateOffset(years=3)
        folds.append(dict(fold_id=str(start.to_period("Q")), train_start=training.strftime("%Y-%m-%d"),
            validation_start=validation.strftime("%Y-%m-%d"), test_start=start.strftime("%Y-%m-%d"),
            test_end=end, normal_exit_observed_through=allowed[last + 21], embargo_sessions=20,
            evidence_role="retrospective_development_only", formal_profit_qualified=False))
        start = next_start
    return folds


def partition_sample_metadata(samples, fold, dates, sealed_start):
    """Retain all evaluation rows; purge fits using actual label ends, never fixed guesses.

    This operates only on row identity and clocks, never outcome magnitudes. Unknown
    evaluation endpoints remain visible; fitting cannot consume them. A conservative
    draft convention requires 20 full observed sessions strictly between the final
    fitted label endpoint and the next partition. It needs protocol registration.
    """
    checked_dates(dates)
    required = {"sample_id", "date", "feature_end", "label_end", "label_status"}
    if set(samples) != required:
        raise ValueError("Supply metadata columns only; outcome values must stay outside endpoint partitioning")
    if samples.sample_id.duplicated().any():
        raise ValueError("Sample clocks require unique identities and all endpoint fields")
    if not fold["train_start"] < fold["validation_start"] < fold["test_start"] <= fold["test_end"] < sealed_start:
        raise ValueError("Partitions must be chronological and precede the sealed holdout")
    if fold["embargo_sessions"] != 20:
        raise ValueError("Original research protocol requires a 20-session minimum gap")
    index = {d: i for i, d in enumerate(dates)}
    work = samples.copy()
    if (not work.sample_id.map(lambda v: isinstance(v, str) and bool(v.strip())).all()
            or not work.date.isin(dates).all() or not work.feature_end.isin(dates).all()
            or work.feature_end.gt(work.date).any()):
        raise ValueError("Sample identity or causal feature clock is invalid")
    if work.date.ge(sealed_start).any():
        raise ValueError("Do not supply sealed sample dates to development partitioning")
    available = work.label_status.eq("available")
    if (not work.loc[available, "label_end"].isin(dates).all()
            or work.loc[available, "label_end"].lt(work.loc[available, "date"]).any()):
        raise ValueError("Available labels require actual observed, chronological endpoints")
    work["partition"] = "outside_fold"
    work["partition_reason"] = "outside_fold"
    for name, begin, stop in [("train", fold["train_start"], fold["validation_start"]),
                              ("validation", fold["validation_start"], fold["test_start"]),
                              ("test", fold["test_start"], None)]:
        selected = work.date.ge(begin) & (work.date.lt(stop) if stop else work.date.le(fold["test_end"]))
        work.loc[selected, "partition"] = name
        if name == "test":
            work.loc[selected, "partition_reason"] = "evaluation_retained_endpoint_status_disclosed"
            blocked = selected & available & work.label_end.ge(sealed_start)
            work.loc[blocked, "partition_reason"] = "evaluation_endpoint_enters_sealed_interval_do_not_read_outcome"
            continue
        boundary = bisect.bisect_left(dates, stop)
        endpoint_indices = work.label_end.map(index)
        reason = pd.Series("fit_eligible", index=work.index)
        reason.loc[~available] = "label_unavailable_not_fit"
        reason.loc[available & work.label_end.ge(stop)] = "label_crosses_next_partition"
        reason.loc[available & work.label_end.lt(stop) & endpoint_indices.ge(boundary - 20)] = "fewer_than_20_clear_sessions_after_label_end"
        work.loc[selected, "partition_reason"] = reason.loc[selected]
    return dict(train=work.loc[work.partition.eq("train") & work.partition_reason.eq("fit_eligible")].copy(),
        validation=work.loc[work.partition.eq("validation") & work.partition_reason.eq("fit_eligible")].copy(),
        test=work.loc[work.partition.eq("test")].copy(),
        audit=work, formal_profit_qualified=False, historical_return_experiment_run=False,
        convention="20 full exchange sessions strictly between fitted label end and the next partition")
