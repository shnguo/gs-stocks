import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from plan_risk_decision import choose, outcomes, paired_metrics

from quant_research.plan_value import HEADS, choose_research_plan


def forecasts():
    return {h: np.full((2, 12), v) for h, v in zip(HEADS, [.5, .95, .01, .1, .005])}


def test_downside_changes_choice_without_changing_expected_return():
    pred = forecasts()
    pred[HEADS[2]][:, 0] = .02
    assert choose(pred, risk=True).tolist() == [0, 0]
    original = choose(pred)
    pred[HEADS[4]][0, 0] = .08
    pred[HEADS[4]][1, :] = .08
    np.testing.assert_array_equal(choose(pred), original)
    np.testing.assert_array_equal(choose(pred), choose_research_plan(pred))
    assert choose(pred, risk=True).tolist() == [1, -1]


def test_invalid_forecast_is_not_an_abstention():
    pred = forecasts()
    pred[HEADS[4]][0, 0] = np.nan
    with pytest.raises(ValueError):
        choose(pred, risk=True)


def test_no_action_unknown_and_ambiguous_are_distinct():
    chosen = np.array([-1, 0, 1, 2])
    base = dict(scenario_order_net_return=np.full((4, 12), np.nan),
        ambiguous=np.zeros((4, 12), bool), filled=np.full((4, 12), np.nan))
    base['scenario_order_net_return'][2, 1] = 0
    base['filled'][2, 1] = 0
    base['scenario_order_net_return'][3, 2] = -.02
    base['filled'][3, 2] = 1
    base['ambiguous'][3, 2] = True
    actual = outcomes(chosen, base, base)
    np.testing.assert_allclose(actual['primary'], [0, np.nan, 0, np.nan], equal_nan=True)
    np.testing.assert_allclose(actual['stress'], [0, np.nan, 0, -.02], equal_nan=True)
    assert actual['selected'].tolist() == [False, True, True, True]
    rows = pd.DataFrame(dict(date=['2025-01-01']*4))
    other = dict(primary=np.array([.01, .02, np.nan, -.01]), stress=np.array([.01, .02, np.nan, -.01]))
    paired = paired_metrics(rows, actual, other, 'fold-01', 'candidate_vs_baseline')
    p = paired[paired.scenario.eq('primary')].iloc[0]
    assert p.common_known == 1 and p.candidate_only_known == 1 and p.baseline_only_known == 2
    assert p.delta == -.01
