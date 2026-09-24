import pandas as pd
from token_early_stage_matched import attach_outcomes, match, segment, source_availability


def sample():
    return pd.DataFrame({
        "local_row": [1, 2, 3, 4],
        "date": ["2025-01-01"]*4,
        "market_segment": ["MAIN"]*4,
        "instrument_id": ["a", "b", "c", "d"],
        "predicted_rank": [.80, .60, .79, .61],
        "volatility_rank": [.40, .70, .41, .69],
        "liquidity_rank": [.50, .30, .49, .31],
        "early_stage_rank": [.95, .90, .20, .10],
        "early_stage_score": [.8, .7, 0., 0.],
        "right_side": [True, True, False, False],
        "label_known": [True]*4,
        "actual_execution_return": [.1, .2, -.1, -.2],
        "actual_extrema_scenario": [.2, .3, 0., -.1],
        "actual_ideal_execution_return": [.3, .4, .1, 0.],
        "actual_adverse": [-.02, -.03, -.08, -.09],
    })


def test_segment_mapping():
    assert segment("cn.xbse.920001") == "XBSE"
    assert segment("cn.xshg.688001") == "STAR"
    assert segment("cn.xshe.300001") == "CHINEXT"
    assert segment("cn.xshg.600001") == "MAIN"


def test_matching_is_deterministic_and_outcome_blind():
    frame = sample()
    spec = dict(treated_quantile=.8, control_quantile=.5,
        predicted_rank_caliper=.03, volatility_rank_caliper=.05,
        liquidity_rank_caliper=.05)
    covariates = frame.drop(columns=["label_known", "actual_execution_return",
        "actual_extrema_scenario", "actual_ideal_execution_return", "actual_adverse"])
    first = match(covariates, spec)
    changed = frame.copy()
    changed["actual_execution_return"] *= -10
    second = match(changed[covariates.columns], spec)
    pd.testing.assert_frame_equal(first, second)
    assert first[["treated_row", "control_row"]].values.tolist() == [[1, 3], [2, 4]]
    outcomes = attach_outcomes(first, frame)
    assert outcomes.pair_known.all()
    assert outcomes.difference_actual_execution_return.tolist() == [.2, .4]


def test_empty_match_retains_a_stable_schema():
    frame = sample()
    spec = dict(treated_quantile=.99, control_quantile=.01,
        predicted_rank_caliper=.001, volatility_rank_caliper=.001,
        liquidity_rank_caliper=.001)
    found = match(frame.drop(columns=["label_known", "actual_execution_return",
        "actual_extrema_scenario", "actual_ideal_execution_return", "actual_adverse"]), spec)
    assert found.empty
    assert "treated_row" in found


def test_source_preflight_reports_missing_inputs():
    cfg = {
        "source": "does-not-exist/stage",
        "forecast_source": "does-not-exist/forecast",
        "price_input": "does-not-exist/prices",
        "source_mode": "paired",
    }
    result = source_availability(cfg)
    assert not result["passed"]
    assert result["missing_files"] == 5
