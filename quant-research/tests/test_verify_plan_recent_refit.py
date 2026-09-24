import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from verify_plan_recent_refit import policy_check


def test_policy_verifier_preserves_unknown_against_cash():
    rows = pd.DataFrame(dict(date=['2024-01-01']*2))
    base = dict(scenario_order_net_return=np.full((2, 12), np.nan),
        filled=np.full((2, 12), np.nan), ambiguous=np.zeros((2, 12), bool))
    choices = dict(fixed=np.full(2, 4), cash=np.full(2, -1))
    for model in ['recent_raw', 'rolling_raw']:
        choices[model+'/original'] = np.zeros(2, int)
        choices[model+'/risk'] = np.full(2, -1)
    daily, paired = policy_check(rows, choices, base, base, 'fold-01')
    daily, paired = pd.DataFrame(daily), pd.DataFrame(paired)
    selected = daily[daily.policy.eq('recent_raw/original')].iloc[0]
    assert selected.primary_known == 0 and selected.primary_unknown == 2 and np.isnan(selected.primary_mean)
    skipped = daily[daily.policy.eq('recent_raw/risk')].iloc[0]
    assert skipped.no_action == 2 and skipped.primary_known == 2 and skipped.primary_mean == 0
    pair = paired[paired.comparison.eq('recent_raw/original_vs_cash') & paired.scenario.eq('primary')].iloc[0]
    assert pair.common_known == 0 and pair.baseline_only_known == 2 and np.isnan(pair.delta)
