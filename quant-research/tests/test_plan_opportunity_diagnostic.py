import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from plan_opportunity_diagnostic import evaluate_groups, prediction_groups

from quant_research.plan_value import HEADS


def test_prediction_groups_preserve_ties_under_row_permutation():
    rows = pd.DataFrame(dict(date=['2025-01-01']*7, instrument_id=list('gfedcba')))
    pred = {h: np.full((7, 12), v) for h, v in zip(HEADS, [.5, .95, .02, .4, .01])}
    chosen, groups = prediction_groups(rows, pred, [.1, 1.])
    order = [3, 6, 1, 5, 0, 4, 2]
    moved = rows.iloc[order].reset_index(drop=True)
    _, other = prediction_groups(moved, {h: v[order] for h, v in pred.items()}, [.1, 1.])
    assert (chosen == 0).all()
    for left, right in zip(groups, other):
        assert rows.instrument_id.iloc[left[4]].tolist() == moved.instrument_id.iloc[right[4]].tolist()
    bins = [g[4] for g in groups if g[1] == 'conditional_mean' and g[2] == 'quintile_best_first']
    assert sorted(np.concatenate(bins)) == list(range(7))


def test_no_eligible_predictions_keep_empty_date():
    rows = pd.DataFrame(dict(date=['2025-01-01'], instrument_id=['a']))
    pred = {h: np.full((1, 12), np.nan) for h in HEADS}
    chosen, groups = prediction_groups(rows, pred, [.1, 1.])
    assert chosen.tolist() == [-1]
    assert len(groups) == 14 and all(len(g[4]) == 0 for g in groups)


def test_unknown_return_is_not_zero_or_conditional_fill():
    rows = pd.DataFrame(dict(date=['2025-01-01']*2, instrument_id=['a', 'b']))
    pred = {h: np.full((2, 12), v) for h, v in zip(HEADS, [.5, .95, .02, .4, .01])}
    base = {k: np.full((2, 12), np.nan) for k in ['filled', 'conditional_net_return',
        'scenario_order_net_return', 'conditional_loss', 'conditional_downside']}
    base['filled'][0] = 0
    base['scenario_order_net_return'][0] = 0
    base['ambiguous'] = np.zeros((2, 12), bool)
    chosen, groups = prediction_groups(rows, pred, [1.])
    result = evaluate_groups(rows, pred, base, base, chosen, groups, 1, 'test').iloc[0]
    assert result.selected == 2 and result.return_known == 1 and result.return_unknown == 1
    assert result.known_order_scenario_mean == 0 and result.conditional_known == 0
    assert np.isnan(result.observed_conditional_mean)
