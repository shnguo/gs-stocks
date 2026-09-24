import numpy as np
import pandas as pd

from quant_research.early_stage import (
    history_features,
    market_segment,
    matching_history_controls,
    path_execution_statistics,
    rerank,
    select_start_stage,
)


def history(close, volume=None):
    close = np.asarray(close, float)
    volume = np.ones(len(close))*100 if volume is None else np.asarray(volume, float)
    raw = np.zeros((1, len(close), 7), float)
    raw[0, :, 0] = close*.995
    raw[0, :, 1] = close*1.01
    raw[0, :, 2] = close*.99
    raw[0, :, 3] = close
    raw[0, :, 4] = volume
    raw[0, :, 5] = close*volume
    raw[0, :, 6] = 1.
    return raw


def test_early_stage_rewards_confirmed_unextended_trend():
    early = history_features(history(np.r_[np.linspace(10, 10.2, 50), np.linspace(10.2, 10.7, 10)]))
    extended = history_features(history(np.r_[np.ones(50)*10, np.linspace(10, 15, 10)]))
    falling = history_features(history(np.linspace(15, 10, 60)))
    assert early.right_side.iloc[0]
    assert early.early_stage_score.iloc[0] > extended.early_stage_score.iloc[0]
    assert extended.overextended.iloc[0]
    assert extended.extension_flags.iloc[0] >= 2
    assert not falling.right_side.iloc[0]
    assert falling.early_stage_score.iloc[0] == 0


def test_history_features_are_price_scale_invariant():
    raw = history(np.linspace(10, 11, 60), np.linspace(100, 140, 60))
    scaled = raw.copy()
    scaled[..., :4] *= 10
    scaled[..., 6] /= 10
    np.testing.assert_allclose(history_features(raw).select_dtypes("number"),
                               history_features(scaled).select_dtypes("number"), atol=1e-12)


def test_matching_controls_are_causal_and_scale_invariant():
    raw = history(np.linspace(10, 12, 60), np.linspace(100, 150, 60))
    scaled = raw.copy()
    scaled[..., :4] *= 5
    scaled[..., 6] /= 5
    found = matching_history_controls(raw)
    changed = matching_history_controls(scaled)
    np.testing.assert_allclose(found, changed, atol=1e-12)
    assert found.historical_volatility_20.iloc[0] >= 0
    assert np.isfinite(found.log_amount_20.iloc[0])


def test_market_segment_mapping():
    assert market_segment("cn.xbse.920001") == "XBSE"
    assert market_segment("cn.xshg.688001") == "STAR"
    assert market_segment("cn.xshe.300001") == "CHINEXT"
    assert market_segment("cn.xshe.000001") == "MAIN"


def test_execution_statistics_use_first_open_and_selected_high():
    paths = np.ones((2, 8, 5, 6), float)
    paths[..., 0] = 10
    paths[..., 1] = 11
    paths[..., 2] = 9
    paths[..., 3] = 10
    paths[..., 4] = 100
    paths[..., 5] = 1000
    paths[0, :, 2, 1] = 12
    paths[1, :, 4, 1] = 13
    result = path_execution_statistics([paths], np.array([2, 4]), cost=.0025, minimum_paths=4)
    np.testing.assert_allclose(result.predicted_execution_return, [.1975, .2975])
    np.testing.assert_allclose(result.predicted_downside, [-.1, -.1])
    assert result.execution_forecast_eligible.all()


def test_reranking_is_forecast_only_and_deterministic():
    frame = pd.DataFrame({
        "instrument_id": ["a", "b", "c"],
        "predicted": [.2, .15, .1],
        "predicted_execution_return": [.1, .14, .08],
        "predicted_downside": [-.2, -.05, -.02],
        "early_stage_score": [0., .9, .8],
        "extension_risk": [1., 0., 0.],
    })
    weights = dict(overheat_penalty=.35, early_return=.7, early_quality=.3,
        combined_return=.1, combined_execution=.35, combined_early=.35, combined_downside=.2)
    ranked = rerank(frame, weights)
    assert ranked[ranked.arm.eq("control")].iloc[0].instrument_id == "a"
    assert ranked[ranked.arm.eq("overheat_penalty")].iloc[0].instrument_id == "b"
    assert ranked[ranked.arm.eq("early_stage")].iloc[0].instrument_id == "b"
    assert ranked[ranked.arm.eq("combined")].iloc[0].instrument_id == "b"


def test_start_stage_filters_before_ranking_by_forecast_return():
    frame = pd.DataFrame({
        "instrument_id": ["extended", "early-high", "early-low", "falling"],
        "reference_price_net_return": [.30, .08, .05, .20],
        "return_5": [.15, .04, .03, .02],
        "return_10": [.25, .05, .03, .04],
        "return_20": [.30, .06, .04, .05],
        "ma20_distance": [.12, .03, .02, .02],
        "volume_ratio_20": [3.2, 1.3, 1.1, 1.2],
        "pullback_5": [-.01, -.03, -.02, -.03],
        "early_stage_score": [.1, .7, .8, .7],
        "extension_flags": [3, 0, 0, 0],
        "right_side": [True, True, True, False],
    })
    criteria = dict(min_return_5=0., max_return_5=.10,
        min_return_10=0., max_return_10=.08, min_return_20=-.02, max_return_20=.12,
        min_ma20_distance=0., max_ma20_distance=.05, min_volume_ratio_20=.9,
        max_volume_ratio_20=2.5, min_pullback_5=-.06, max_pullback_5=.005,
        min_early_stage_score=.5, max_extension_flags=0)
    selected = select_start_stage(frame, criteria)
    assert selected.instrument_id.tolist() == ["early-high", "early-low"]
    assert selected.start_stage_rank.tolist() == [1, 2]
