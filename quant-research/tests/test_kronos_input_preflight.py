import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from plan_kronos_run import preflight_rows


def test_preflight_rejects_missing_dates_before_inference():
    dates = pd.bdate_range('2025-01-01', periods=90).strftime('%Y-%m-%d').tolist()
    row = pd.DataFrame([dict(date=dates[70], label_end=dates[75])])
    protocol = {'sealed_holdout_start': '2025-08-07'}
    preflight_rows(row, {'dates': dates}, protocol)
    with pytest.raises(ValueError, match='Incomplete'):
        preflight_rows(row, {'dates': dates[:70]}, protocol)
    with pytest.raises(ValueError, match='Incomplete'):
        preflight_rows(row, {'dates': dates[:75]}, protocol)
    with pytest.raises(ValueError, match='differ'):
        preflight_rows(row.assign(label_end=dates[76]), {'dates': dates}, protocol)
    with pytest.raises(ValueError, match='sealed'):
        preflight_rows(row, {'dates': dates}, {'sealed_holdout_start': dates[75]})
    with pytest.raises(ValueError, match='Incomplete'):
        preflight_rows(row.assign(date=dates[58], label_end=dates[63]), {'dates': dates}, protocol)
