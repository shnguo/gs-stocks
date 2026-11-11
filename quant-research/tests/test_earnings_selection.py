import numpy as np
import pandas as pd
import pytest

from quant_research.earnings_selection import (
    EVENT_COLUMNS,
    MATCH_FIELDS,
    first_event_audit,
    frozen_selection_features,
    match_frozen_event_selections,
    select_earnings_arms,
)


def event(symbol="a", identifier="1", at="2023-07-27T10:00:00Z", **kwargs):
    return dict(announcement_id=identifier, instrument_id=symbol, fiscal_period="2023-06-30",
                available_at=at, document_kind="earnings_forecast",
                profit_metric="attributable_to_parent_same_reporting_period",
                net_profit_lower_cny=140., net_profit_upper_cny=160., prior_same_period_profit_cny=100.,
                disclosed_yoy_lower=.40, disclosed_yoy_upper=.60, financial_fields_verified=True,
                first_numeric_version_verified=True, historical_pit_verified=False,
                availability_basis="local_observation", pdf_sha256="a" * 64) | kwargs


def events(*rows):
    return pd.DataFrame(rows, columns=sorted(EVENT_COLUMNS))


def test_local_clock_sets_exact_first_full_session_and_never_uses_claimed_publication(snapshot):
    at = snapshot.dates[150] + "T23:30:00Z"  # Already the next Shanghai calendar date.
    original = events(event(at=at))
    original["source_publication_date_claim"] = snapshot.dates[140]
    audit = first_event_audit(original, snapshot.dates)
    local_day = pd.Timestamp(at).tz_convert("Asia/Shanghai").date().isoformat()
    expected = next(d for d in snapshot.dates if d > local_day)
    assert audit.iloc[0].eligible_signal_date == expected
    assert audit.iloc[0].eligible
    historical = first_event_audit(original, snapshot.dates, historical=True)
    assert historical.iloc[0].reason == "historical_availability_unverified"


def test_first_negative_document_cannot_be_replaced_by_later_positive_revision(snapshot):
    first = event(at=snapshot.dates[148] + "T10:00:00Z", disclosed_yoy_lower=-.5,
                  disclosed_yoy_upper=-.3)
    later = event(identifier="2", at=snapshot.dates[149] + "T10:00:00Z")
    audit = first_event_audit(events(first, later), snapshot.dates).set_index("announcement_id")
    assert not audit.eligible.any()
    assert audit.loc["1", "reason"] == "positive_growth_lower_bound_gate_failed"
    assert audit.loc["2", "reason"] == "later_document_for_same_company_report"


def test_ambiguous_first_clocks_and_unverified_numbers_remain_blocked(snapshot):
    at = snapshot.dates[148] + "T10:00:00Z"
    tied = first_event_audit(events(event(at=at), event(identifier="2", at=at)), snapshot.dates)
    assert set(tied.reason) == {"ambiguous_first_document_clock"}
    unverified = first_event_audit(events(event(at=at, financial_fields_verified=False)), snapshot.dates)
    assert unverified.iloc[0].reason == "financial_fields_unverified"
    with pytest.raises(ValueError, match="actual booleans"):
        first_event_audit(events(event(at=at, financial_fields_verified="true")), snapshot.dates)
    missing = first_event_audit(events(event(at=at, disclosed_yoy_lower=np.nan)), snapshot.dates)
    assert missing.iloc[0].reason == "missing_numeric_primary_gate"


@pytest.fixture
def feature_input(snapshot):
    meta = snapshot.tables["instruments"]
    meta["board"] = meta.exchange.map({"XSHG": "shanghai_main", "XSHE": "shenzhen_main", "XBSE": "beijing"})
    bars = snapshot.tables["bars"]
    indices = bars.date.map({d: i for i, d in enumerate(snapshot.dates)})
    for col in ("open", "close"):
        bars[col] = 10. + indices * .001
    bars["high"] = bars.close + .10
    bars["low"] = bars.close - .10
    bars["factor"] = 1.
    bars["volume"] = 4_000_000.
    bars["amount"] = bars.close * bars.volume
    industries = meta[["instrument_id", "industry"]].copy()
    industries["start_date"], industries["end_date"], industries["known_at"] = snapshot.dates[0], "", "2020-01-01T00:00:00Z"
    caps = industries.drop(columns="industry").copy()
    caps["float_market_cap"] = 1_000_000_000.
    return snapshot, industries, caps


def feature_rows(feature_input, indices=(150,)):
    snapshot, industries, caps = feature_input
    return pd.concat([frozen_selection_features(snapshot, snapshot.dates[i],
                     snapshot.dates[i] + "T10:00:00Z", industries, caps) for i in indices], ignore_index=True)


def test_frozen_features_ignore_future_prices_and_reject_gaps_or_late_industry(feature_input):
    snapshot, industries, _ = feature_input
    original = feature_rows(feature_input)
    symbol = original.loc[original.common_eligible, "instrument_id"].iloc[0]
    snapshot.tables["bars"].loc[snapshot.tables["bars"].date > snapshot.dates[150], "close"] = 1_000_000.
    pd.testing.assert_frame_equal(original, feature_rows(feature_input))
    mask = ((snapshot.tables["bars"].date == snapshot.dates[145])
            & (snapshot.tables["bars"].instrument_id == symbol))
    snapshot.tables["bars"] = snapshot.tables["bars"].loc[~mask]
    changed = feature_rows(feature_input).set_index("instrument_id")
    assert changed.loc[symbol, "feature_reason"] == "missing_consecutive_completed_price_history"
    other = original.loc[original.common_eligible & original.instrument_id.ne(symbol), "instrument_id"].iloc[0]
    industries.loc[industries.instrument_id == other, "known_at"] = snapshot.dates[151] + "T00:00:00Z"
    assert feature_rows(feature_input).set_index("instrument_id").loc[other, "feature_reason"] == "point_in_time_industry_unknown"


def test_confirmation_cannot_be_rescued_on_a_later_better_day_and_model_must_be_timely(feature_input):
    snapshot, _, _ = feature_input
    rows = feature_rows(feature_input, (150, 151))
    symbol = rows.loc[rows.common_eligible, "instrument_id"].iloc[0]
    rows.loc[(rows.instrument_id == symbol) & rows.date.eq(snapshot.dates[150]), "confirmation_eligible"] = False
    source = events(event(symbol, at=snapshot.dates[149] + "T10:00:00Z"))
    model = pd.DataFrame([dict(date=snapshot.dates[150], instrument_id=symbol, model_rank=1.,
                               available_at=snapshot.dates[151] + "T02:00:00Z", model_manifest_sha256="b" * 64)])
    result = select_earnings_arms(rows, source, snapshot.dates, model_ranks=model)
    assert result["arms"]["E0"].date.tolist() == [snapshot.dates[150]]
    assert result["arms"]["E1"].empty
    assert result["arms"]["M"].empty
    assert len(result["model_exclusions"]) == 1
    assert result["formal_profit_qualified"] is False


def test_empty_event_input_and_all_excluded_features_are_valid_cash_candidates(feature_input):
    snapshot, industries, caps = feature_input
    industries["known_at"] = snapshot.dates[151] + "T00:00:00Z"
    rows = frozen_selection_features(snapshot, snapshot.dates[150], snapshot.dates[150] + "T10:00:00Z", industries, caps)
    result = select_earnings_arms(rows, events(), snapshot.dates)
    assert all(frame.empty for frame in result["arms"].values())


def matching_inputs(snapshot):
    day = snapshot.dates[150]
    features = pd.DataFrame([dict(date=day, instrument_id=s, decision_at=day + "T10:00:00Z",
                                  board="shanghai_main", exchange="XSHG", industry="industry1",
                                  common_eligible=True, confirmation_eligible=True,
                                  matching_covariates_complete=True,
                                  **{f + "_rank": rank for f in MATCH_FIELDS})
                             for s, rank in [("a", .50), ("b", .52), ("c", .55)]])
    selections = pd.DataFrame([dict(date=day, instrument_id="a", decision_at=day + "T10:00:00Z",
                                   status="selected_before_open", budget=30_000., trade_id="E1/a", selection_order=0)])
    coverage = pd.DataFrame([dict(exchange="XSHG", start_date=snapshot.dates[100], end_date=day,
                                 known_at=day + "T09:00:00Z", complete=True, source_sha256="c" * 64)])
    return selections, features, coverage


def test_matching_uses_exact_industry_and_fixed_calipers_without_future_event_exclusion(snapshot):
    selections, features, coverage = matching_inputs(snapshot)
    future = events(event("b", at=snapshot.dates[151] + "T10:00:00Z"))
    result = match_frozen_event_selections(selections, features, future, coverage, snapshot.dates)
    assert result["pairs"].control.tolist() == ["b"]
    assert result["intended_budget_coverage"] == 1.
    assert result["actual_account_exposure_verified"] is False
    known = events(event("b", at=snapshot.dates[149] + "T10:00:00Z"))
    assert match_frozen_event_selections(selections, features, known, coverage, snapshot.dates)["pairs"].control.tolist() == ["c"]
    features.loc[features.instrument_id.eq("c"), "industry"] = "different_industry"
    unmatched = match_frozen_event_selections(selections, features, known, coverage, snapshot.dates)
    assert unmatched["intended_budget_coverage"] == 0.


def test_matching_refuses_late_coverage_and_reuses_no_control_in_the_same_cohort(snapshot):
    selections, features, coverage = matching_inputs(snapshot)
    coverage["known_at"] = snapshot.dates[151] + "T00:00:00Z"
    result = match_frozen_event_selections(selections, features, events(), coverage, snapshot.dates)
    assert result["pairs"].empty
    assert result["unmatched"].iloc[0].reason == "event_coverage_incomplete_or_not_known_at_freeze"
    coverage["known_at"] = snapshot.dates[150] + "T09:00:00Z"
    second = selections.iloc[0].copy()
    second["instrument_id"], second["trade_id"] = "c", "E1/c"
    second["selection_order"] = 1
    selections = pd.concat([selections, second.to_frame().T], ignore_index=True)
    selections["budget"] = selections.budget.astype(float)
    result = match_frozen_event_selections(selections, features, events(), coverage, snapshot.dates)
    assert len(result["pairs"]) == 1
    assert result["pairs"].control.tolist() == ["b"]
    assert result["intended_budget_coverage"] == .5
