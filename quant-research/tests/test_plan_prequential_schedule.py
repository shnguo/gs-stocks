import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from plan_prequential_schedule import make_schedule


def test_asof_dependencies_mature_and_future_calendar_is_irrelevant():
    days = pd.bdate_range('2020-01-01',periods=900).strftime('%Y-%m-%d').tolist()
    evaluations = [days[i] for i in range(600,640,5)]
    result = make_schedule(days,evaluations,days[0],days[800])
    assert len(result['fits']) == 20 and len(result['predictions']) == 8
    assert result == make_schedule(days[:700],evaluations,days[0],days[800])
    for asof,fit in result['fits'].items():
        assert len(fit['train']) == 96 and len(fit['selection']) == 4
        assert fit['train_label_end'] < fit['selection'][0]
        assert fit['selection_label_end'] == asof
        assert fit['training_label_age_sessions'] == 21
    for prediction in result['predictions']:
        assert len(prediction['calibration_prediction_dates']) == 12
        assert max(prediction['calibration_label_ends']) == prediction['date']
    with pytest.raises(ValueError,match='sealed'):
        make_schedule(days,evaluations,days[0],days[640])
    with pytest.raises(ValueError,match='Insufficient'):
        make_schedule(days,[days[30]],days[0],days[800])
