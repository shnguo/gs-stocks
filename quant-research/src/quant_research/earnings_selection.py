"""Outcome-blind research selectors for the fixed earnings protocol.

Evidence declarations are necessary inputs, not evidence acceptance. This module never
certifies a profit result. Daily technical inputs end at the completed signal close;
first-event identity is resolved before positivity or technical filters are considered.
"""
from __future__ import annotations

import re
from datetime import date as calendar_date

import numpy as np
import pandas as pd

from .event_radar import completed_technical_snapshot
from .storage import Snapshot
from .universe import cutoff, universe_on

MAIN_BOARDS = {"shanghai_main", "shenzhen_main"}
EVENT_COLUMNS = {"announcement_id", "instrument_id", "fiscal_period", "available_at",
                 "document_kind", "profit_metric", "net_profit_lower_cny", "net_profit_upper_cny",
                 "prior_same_period_profit_cny", "disclosed_yoy_lower", "disclosed_yoy_upper",
                 "financial_fields_verified", "first_numeric_version_verified", "historical_pit_verified",
                 "availability_basis", "pdf_sha256"}
MATCH_FIELDS = ("past20_return", "volatility20", "adv20", "float_market_cap")
DECISION_OUTPUT = ["date", "instrument_id", "selection_order", "decision_at", "signal_atr14_raw",
                   "trade_id", "arm", "announcement_id", "fiscal_period", "source_pdf_sha256"]


def aware(value) -> pd.Timestamp:
    at = pd.Timestamp(value)
    if pd.isna(at) or at.tzinfo is None:
        raise ValueError("Information clocks must contain an explicit timezone")
    return at.tz_convert("UTC")


def signal_clock(dates, date, decision_at):
    if date not in dates:
        raise ValueError("Signal date is not an observed exchange session")
    at = aware(decision_at)
    offset = dates.index(date)
    if (offset + 1 >= len(dates) or at < cutoff(date, "15:00")
            or at >= cutoff(dates[offset + 1], "09:00")):
        raise ValueError("Selection requires completed-close data and a timely freeze")
    return at


def event_frame(events: pd.DataFrame) -> pd.DataFrame:
    if EVENT_COLUMNS - set(events):
        raise ValueError("Missing canonical earnings-event columns")
    frame = events.copy()
    if frame.announcement_id.duplicated().any():
        raise ValueError("Duplicate announcement identity must be reconciled before selection")
    for key in ("financial_fields_verified", "first_numeric_version_verified", "historical_pit_verified"):
        if not frame[key].map(lambda v: isinstance(v, (bool, np.bool_))).all():
            raise ValueError("Event qualification flags must be actual booleans")
    frame["available_at"] = frame.available_at.map(aware)
    if not frame.availability_basis.isin({"local_observation", "verified_historical_publication"}).all():
        raise ValueError("Unsupported event availability basis")
    if not frame.pdf_sha256.map(lambda v: isinstance(v, str) and bool(re.fullmatch(r"[0-9a-f]{64}", v))).all():
        raise ValueError("Each canonical event requires a bound original document hash")
    # Do not accept source publication claims as an alternative to the availability clock.
    return frame


def first_event_audit(events: pd.DataFrame, dates: list[str], *, historical=False) -> pd.DataFrame:
    """Select the first numeric forecast identity before testing any financial gate.

    A later positive revision cannot replace the first negative or unverified document.
    Equal earliest clocks are ambiguous; tie-breaking IDs cannot establish chronology.
    """
    if dates != sorted(set(dates)) or not dates:
        raise ValueError("Exchange sessions must be sorted and unique")
    frame = event_frame(events)
    rows = []
    for row in frame.itertuples():
        if row.document_kind != "earnings_forecast":
            rows.append({"announcement_id": row.announcement_id, "instrument_id": row.instrument_id,
                         "fiscal_period": row.fiscal_period, "reason": "non_primary_document_kind",
                         "eligible_signal_date": "", "eligible": False})
    primary = frame.loc[frame.document_kind.eq("earnings_forecast")]
    for _, group in primary.groupby(["instrument_id", "fiscal_period"], dropna=False, sort=True):
        group = group.sort_values(["available_at", "announcement_id"])
        first = group.available_at.min()
        ambiguous = group.available_at.eq(first).sum() != 1
        for row in group.itertuples():
            reason = "eligible"
            local_day = row.available_at.tz_convert("Asia/Shanghai").date().isoformat()
            signal = next((d for d in dates if d > local_day), "")
            if row.available_at != first:
                reason = "later_document_for_same_company_report"
            elif ambiguous:
                reason = "ambiguous_first_document_clock"
            elif not isinstance(row.fiscal_period, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", row.fiscal_period):
                reason = "missing_or_invalid_fiscal_period"
            elif not _valid_period(row.fiscal_period):
                reason = "missing_or_invalid_fiscal_period"
            elif not row.first_numeric_version_verified:
                reason = "first_numeric_version_unverified"
            elif not row.financial_fields_verified:
                reason = "financial_fields_unverified"
            elif historical and (not row.historical_pit_verified
                                  or row.availability_basis != "verified_historical_publication"):
                reason = "historical_availability_unverified"
            elif row.profit_metric != "attributable_to_parent_same_reporting_period":
                reason = "profit_metric_or_reporting_basis_unverified"
            else:
                fields = [row.net_profit_lower_cny, row.net_profit_upper_cny,
                          row.prior_same_period_profit_cny, row.disclosed_yoy_lower, row.disclosed_yoy_upper]
                if not all(isinstance(v, (int, float, np.number)) and not isinstance(v, (bool, np.bool_))
                           and np.isfinite(v) for v in fields):
                    reason = "missing_numeric_primary_gate"
                elif (row.net_profit_lower_cny > row.net_profit_upper_cny
                      or row.disclosed_yoy_lower > row.disclosed_yoy_upper):
                    reason = "reversed_verified_numeric_bounds"
                elif row.prior_same_period_profit_cny <= 0:
                    reason = "prior_loss_or_zero_separate_exploratory_stratum"
                elif row.net_profit_lower_cny <= 0 or row.disclosed_yoy_lower < .30:
                    reason = "positive_growth_lower_bound_gate_failed"
                elif local_day < dates[0]:
                    reason = "availability_precedes_observed_calendar"
                elif not signal:
                    reason = "first_full_post_availability_session_not_observed"
            rows.append({"announcement_id": row.announcement_id, "instrument_id": row.instrument_id,
                         "fiscal_period": row.fiscal_period, "reason": reason,
                         "eligible_signal_date": signal, "eligible": reason == "eligible"})
    return pd.DataFrame(rows, columns=["announcement_id", "instrument_id", "fiscal_period", "reason",
                                       "eligible_signal_date", "eligible"])


def _valid_period(value):
    try:
        calendar_date.fromisoformat(value)
        return True
    except ValueError:
        return False


def latest_pit_value(frame, instrument, date, at, value):
    selected = frame.loc[(frame.instrument_id == instrument) & (frame.start_date <= date)
                         & ((frame.end_date == "") | (frame.end_date >= date))
                         & (frame.known_at <= at)]
    if selected.empty:
        return None
    selected = selected.loc[selected.known_at == selected.known_at.max()]
    return selected[value].iloc[0] if len(selected) == 1 else None


def frozen_selection_features(snapshot: Snapshot, date: str, decision_at, industries: pd.DataFrame,
                              capitalisations: pd.DataFrame) -> pd.DataFrame:
    """One causal cross-section, with PIT industry and free-float value supplied separately.

    Corporate-action and source-vintage acceptance remain the evidence runner's job.
    Gaps are rejected before calling the existing radar, which otherwise compresses bars.
    """
    at = signal_clock(snapshot.dates, date, decision_at)
    histories = []
    for frame, value in [(industries, "industry"), (capitalisations, "float_market_cap")]:
        if {"instrument_id", "start_date", "end_date", "known_at", value} - set(frame):
            raise ValueError("Missing point-in-time matching covariates")
        copy = frame.copy()
        copy["known_at"] = copy.known_at.map(aware)
        histories.append(copy)
    industries, capitalisations = histories
    universe = universe_on(snapshot, date, availability_at=at)
    days = snapshot.dates[max(0, snapshot.dates.index(date) - 20):snapshot.dates.index(date) + 1]
    bars = snapshot.tables["bars"]
    bars = bars.loc[bars.date.isin(days)]
    groups = {symbol: group.set_index("date").reindex(days)
              for symbol, group in bars.groupby("instrument_id")}
    rows = []
    for meta in universe.itertuples():
        symbol = meta.instrument_id
        row = {"date": date, "instrument_id": symbol, "decision_at": at.isoformat(),
               "board": meta.board, "exchange": meta.exchange, "common_eligible": False,
               "confirmation_eligible": False, "feature_reason": "eligible",
               "signal_atr14_raw": np.nan, "past20_return": np.nan}
        rows.append(row)
        if meta.board not in MAIN_BOARDS or meta.exchange not in {"XSHG", "XSHE"}:
            row["feature_reason"] = "outside_verified_main_board_scope"
            continue
        if not meta.eligible:
            row["feature_reason"] = meta.reason
            continue
        listed = sum(meta.listed_at <= d <= date for d in snapshot.dates)
        if listed < 120:
            row["feature_reason"] = "fewer_than_120_observed_listed_sessions"
            continue
        history = groups.get(symbol)
        columns = ["open", "high", "low", "close", "volume", "amount", "factor"]
        if (history is None or len(history) != 21
                or not np.isfinite(history[columns].to_numpy(float)).all()
                or (history[["open", "high", "low", "close", "factor"]] <= 0).any().any()
                or (history[["volume", "amount"]] < 0).any().any()):
            row["feature_reason"] = "missing_consecutive_completed_price_history"
            continue
        if "sequence_id" in history and (history.sequence_id.isna().any() or history.sequence_id.nunique() != 1):
            row["feature_reason"] = "unverified_corporate_action_or_identity_boundary"
            continue
        industry = latest_pit_value(industries, symbol, date, at, "industry")
        if (not isinstance(industry, str) or not industry.strip()
                or industry.strip().lower() in {"unknown", "unverified", "nan", "none"}):
            row["feature_reason"] = "point_in_time_industry_unknown"
            continue
        market_cap = latest_pit_value(capitalisations, symbol, date, at, "float_market_cap")
        technical = completed_technical_snapshot(history.reset_index(), date).iloc[0]
        adjusted = history.close * history.factor
        adv = float(history.amount.iloc[-20:].mean())
        row.update(industry=industry, float_market_cap=market_cap, adv20=adv,
                   past20_return=float(adjusted.iloc[-1] / adjusted.iloc[0] - 1),
                   volatility20=float(adjusted.pct_change(fill_method=None).iloc[1:].std()),
                   signal_atr14_raw=float(technical.atr14 / history.factor.iloc[-1]),
                   ma5=float(technical.ma5), adjusted_close=float(technical.adjusted_close),
                   ma5_rising=bool(technical.ma5_rising), volume_ratio_20=technical.volume_ratio_20,
                   overextended=bool(technical.overextended))
        if adv < 30_000_000:
            row["feature_reason"] = "adv20_below_30_million_cny"
        elif not technical.above_ma5:
            row["feature_reason"] = "below_formal_ma5"
        elif not np.isfinite(row["signal_atr14_raw"]) or row["signal_atr14_raw"] <= 0:
            row["feature_reason"] = "atr_not_positive"
        else:
            row["common_eligible"] = True
            row["confirmation_eligible"] = bool(technical.ma5_rising and not technical.overextended
                                                and technical.volume_ratio_20 is not None
                                                and technical.volume_ratio_20 >= 1)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    # One rank population for every arm, fixed before any event or model filtering.
    covariates = result.reindex(columns=MATCH_FIELDS).apply(pd.to_numeric, errors="coerce")
    complete = result.common_eligible & np.isfinite(covariates).all(axis=1) & covariates.float_market_cap.gt(0)
    result["matching_covariates_complete"] = complete
    for field in MATCH_FIELDS:
        result[field + "_rank"] = covariates.loc[complete, field].rank(method="average", pct=True)
    return result


def select_earnings_arms(features: pd.DataFrame, events: pd.DataFrame, dates: list[str],
                         *, model_ranks: pd.DataFrame | None = None, historical=False) -> dict:
    """Return ranked candidate lists. The account ledger alone allocates actual slots."""
    required = {"date", "instrument_id", "decision_at", "common_eligible", "confirmation_eligible",
                "signal_atr14_raw", "past20_return"}
    if required - set(features) or features.duplicated(["date", "instrument_id"]).any():
        raise ValueError("Incomplete or duplicate frozen feature identity")
    for date, group in features.groupby("date"):
        if group.decision_at.nunique() != 1:
            raise ValueError("Feature cross-section must share one frozen clock")
        signal_clock(dates, date, group.decision_at.iloc[0])
    canonical = event_frame(events)
    audit = first_event_audit(events, dates, historical=historical)
    primary = audit.loc[audit.eligible].merge(canonical, on=["announcement_id", "instrument_id", "fiscal_period"],
                                            validate="one_to_one")
    primary = primary.rename(columns={"eligible_signal_date": "date"})
    event_features = features.merge(primary, on=["date", "instrument_id"], validate="one_to_many")
    event_features = event_features.loc[event_features.available_at <= event_features.decision_at.map(aware)]
    # One entry per company/report. Multiple reports on one date are resolved by the registered rank.
    event_features = event_features.sort_values(["date", "available_at", "disclosed_yoy_lower", "instrument_id"],
                                                ascending=[True, True, False, True])
    event_features = event_features.drop_duplicates(["date", "instrument_id"], keep="first")
    frames = {"P": features.loc[features.confirmation_eligible].sort_values(
                  ["date", "past20_return", "instrument_id"], ascending=[True, False, True]),
              "E0": event_features.loc[event_features.common_eligible],
              "E1": event_features.loc[event_features.confirmation_eligible]}
    model_exclusions = []
    frames["M"] = features.iloc[:0]
    if model_ranks is not None:
        if {"date", "instrument_id", "model_rank", "available_at", "model_manifest_sha256"} - set(model_ranks):
            raise ValueError("Missing frozen model ranking provenance")
        if model_ranks.duplicated(["date", "instrument_id"]).any():
            raise ValueError("Duplicate model ranking identity")
        model = model_ranks.copy()
        model["model_available_at"] = model.pop("available_at").map(aware)
        merged = features.merge(model, on=["date", "instrument_id"], validate="one_to_one")
        valid = (merged.model_available_at <= merged.decision_at.map(aware)) & np.isfinite(merged.model_rank)
        valid &= merged.model_manifest_sha256.map(lambda v: isinstance(v, str) and bool(re.fullmatch(r"[0-9a-f]{64}", v)))
        model_exclusions = merged.loc[~valid, ["date", "instrument_id"]].to_dict("records")
        frames["M"] = merged.loc[valid & merged.confirmation_eligible].sort_values(
            ["date", "model_rank", "instrument_id"])
    arms = {}
    for name, frame in frames.items():
        frame = frame.copy()
        frame["selection_order"] = frame.groupby("date").cumcount()
        frame["arm"] = name
        frame["trade_id"] = [f"{name}/{d}/{s}" for d, s in zip(frame.date, frame.instrument_id)]
        if "pdf_sha256" in frame:
            frame["source_pdf_sha256"] = frame.pdf_sha256
        arms[name] = frame.reindex(columns=DECISION_OUTPUT)
    return {"arms": arms, "event_audit": audit, "model_exclusions": model_exclusions,
            "formal_profit_qualified": False}


def match_frozen_event_selections(selections: pd.DataFrame, features: pd.DataFrame,
                                  events: pd.DataFrame, coverage: pd.DataFrame,
                                  dates: list[str]) -> dict:
    """Match intended E1 slots before opening outcomes, without looking at future events.

    Coverage is required over the preceding 20 sessions plus signal date. Any known
    earnings document in that window excludes a control. This is a draft implementation
    choice that must be registered before evaluation. Actual fill/holding exposure still
    needs independent paired-account reconciliation; intended weights do not prove it.
    """
    required = {"date", "instrument_id", "status", "budget", "decision_at", "trade_id", "selection_order"}
    if required - set(selections):
        raise ValueError("Missing frozen E1 selection fields")
    if {"exchange", "start_date", "end_date", "known_at", "complete", "source_sha256"} - set(coverage):
        raise ValueError("Missing independently auditable event coverage declarations")
    if features.duplicated(["date", "instrument_id"]).any():
        raise ValueError("Duplicate matching feature identity")
    cover = coverage.copy()
    cover["known_at"] = cover.known_at.map(aware)
    if not cover.complete.map(lambda v: isinstance(v, (bool, np.bool_))).all():
        raise ValueError("Coverage completeness must be an explicit boolean")
    canonical = event_frame(events)
    pairs, missed = [], []
    total_weight, matched_weight = 0., 0.
    selected = selections.loc[selections.status.eq("selected_before_open")]
    if not selected.empty and (not np.isfinite(selected.budget).all() or selected.budget.le(0).any()
                               or selected.duplicated(["date", "instrument_id"]).any()):
        raise ValueError("Frozen matched selections require unique identities and positive finite budgets")
    for date, group in selected.groupby("date", sort=True):
        if group.decision_at.nunique() != 1:
            raise ValueError("Matched cohort must share one frozen timestamp")
        at = signal_clock(dates, date, group.decision_at.iloc[0])
        section = features.loc[features.date.eq(date)].set_index("instrument_id")
        if not section.empty and not section.decision_at.map(aware).eq(at).all():
            raise ValueError("Matching features belong to a different information freeze")
        start = dates[max(0, dates.index(date) - 20)]
        recent = canonical.loc[(canonical.available_at <= at)
                               & (canonical.available_at.map(lambda t: t.tz_convert("Asia/Shanghai").date().isoformat()) >= start)]
        excluded = set(recent.instrument_id) | set(group.instrument_id)
        used = set()
        for row in group.sort_values(["selection_order", "instrument_id", "trade_id"]).itertuples():
            total_weight += row.budget
            reason = "no_control_within_all_fixed_calipers"
            if row.instrument_id not in section.index or not section.loc[row.instrument_id].matching_covariates_complete:
                reason = "treated_matching_covariates_incomplete"
                missed.append({"date": date, "trade_id": row.trade_id, "reason": reason, "budget": row.budget})
                continue
            treated = section.loc[row.instrument_id]
            verified_cover = cover.loc[(cover.exchange == treated.exchange) & cover.complete
                                      & (cover.start_date <= start) & (cover.end_date >= date)
                                      & (cover.known_at <= at)]
            verified_cover = verified_cover.loc[verified_cover.source_sha256.map(
                lambda v: isinstance(v, str) and bool(re.fullmatch(r"[0-9a-f]{64}", v)))]
            if verified_cover.empty or dates.index(date) < 20:
                reason = "event_coverage_incomplete_or_not_known_at_freeze"
                missed.append({"date": date, "trade_id": row.trade_id, "reason": reason, "budget": row.budget})
                continue
            choices = section.loc[section.confirmation_eligible & section.matching_covariates_complete
                                  & section.board.eq(treated.board) & section.industry.eq(treated.industry)
                                  & ~section.index.isin(excluded | used)].copy()
            distance = pd.Series(0., index=choices.index)
            legal = pd.Series(True, index=choices.index)
            for field in MATCH_FIELDS:
                gap = (choices[field + "_rank"] - treated[field + "_rank"]).abs()
                legal &= gap.le(.10 + 1e-12)
                distance += (gap / .10) ** 2
            choices["match_distance"] = distance
            choices = choices.loc[legal].reset_index().sort_values(["match_distance", "instrument_id"])
            if choices.empty:
                missed.append({"date": date, "trade_id": row.trade_id, "reason": reason, "budget": row.budget})
                continue
            chosen = choices.iloc[0]
            used.add(chosen.instrument_id)
            matched_weight += row.budget
            pairs.append({"date": date, "decision_at": at.isoformat(), "treated": row.instrument_id,
                          "control": chosen.instrument_id, "treated_trade_id": row.trade_id,
                          "control_trade_id": "P_matched/" + row.trade_id, "selection_order": row.selection_order,
                          "intended_budget": row.budget, "match_distance": chosen.match_distance,
                          **{field + "_rank_gap": abs(chosen[field + "_rank"] - treated[field + "_rank"])
                             for field in MATCH_FIELDS}})
    pair_columns = ["date", "decision_at", "treated", "control", "treated_trade_id", "control_trade_id",
                    "selection_order", "intended_budget", "match_distance", *[field + "_rank_gap" for field in MATCH_FIELDS]]
    return {"pairs": pd.DataFrame(pairs, columns=pair_columns), "unmatched": pd.DataFrame(missed),
            "intended_budget_coverage": matched_weight / total_weight if total_weight else None,
            "actual_account_exposure_verified": False, "formal_profit_qualified": False,
            "no_event_definition": "no_earnings_document_known_by_freeze_in_previous_20_sessions_plus_signal"}
