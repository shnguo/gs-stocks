import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from verify_industry_decisions import verify_decisions

from quant_research.plan_value import HEADS


def test_unknown_selected_return_and_empty_date_are_preserved():
    rows = pd.DataFrame(dict(date=['2025-01-02', '2025-01-03'], instrument_id=['a', 'b']))
    pred = {h: np.full((2, 12), v) for h, v in zip(HEADS, [.5, .95, .02, .4, .01])}
    pred[HEADS[2]][1] = -1
    recorded = rows.copy()
    recorded['plan_index'] = [0, -1]
    for head in HEADS:
        recorded[head] = [pred[head][0, 0], np.nan]
    recorded['executable'] = False
    base = {k: np.full((2, 12), np.nan) for k in ['filled', 'scenario_order_net_return']}
    base['ambiguous'] = np.zeros((2, 12), bool)
    daily = pd.DataFrame(dict(date=rows.date, candidate_stock_rows=[1, 1], selected=[1, 0],
        fill_unknown=[1, 0], filled=[0, 0], return_unknown=[1, 0], ambiguous=[0, 0],
        known_selected_scenario_mean=[np.nan, np.nan], stress_known_selected_scenario_mean=[np.nan, np.nan],
        stress_return_unknown=[1, 0]))
    assert verify_decisions(rows, pred, base, base, recorded, daily) == (2, 2)
    daily.loc[0, 'known_selected_scenario_mean'] = 0
    with pytest.raises(AssertionError):
        verify_decisions(rows, pred, base, base, recorded, daily)
