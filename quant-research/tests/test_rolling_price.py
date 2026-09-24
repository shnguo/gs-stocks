import copy

import pandas as pd
import pytest

from quant_research.rolling_price import build_date_plans, validate_date_plan


def fixture():
    dates = pd.bdate_range('2019-01-01', '2025-08-06').strftime('%Y-%m-%d').tolist()
    fold = dict(train_start='2019-03-01', validation_start='2024-03-01',
                test_start='2025-03-01', test_end='2025-05-31', purpose='development')
    return dates, fold, '2025-08-07'


def test_rolling_density_contrast_and_label_separation():
    dates, fold, sealed = fixture()
    plans = build_date_plans(dates, fold, sealed)
    assert len(plans['recent96']['train']) == 96
    assert len(plans['recent24']['train']) == 24
    assert plans['recent96']['train'][0] == plans['recent24']['train'][0]
    assert plans['recent96']['train'][-1] == plans['recent24']['train'][-1]
    for part in ['selection', 'calibration', 'evaluation']:
        assert plans['recent24'][part] == plans['recent96'][part]
    assert plans['legacy24']['evaluation'] == plans['recent24']['evaluation']
    assert plans['recent24']['train'][-1] > plans['legacy24']['train'][-1]
    for plan in plans.values():
        bounds = validate_date_plan(plan, dates, fold, sealed)
        for part, ds in plan.items():
            assert dates[dates.index(ds[-1])+5] < bounds[part]


@pytest.mark.parametrize('defect', ['purge', 'overlap', 'sealed', 'unsorted', 'outside'])
def test_date_plan_rejects_leakage_and_overlapping_test_labels(defect):
    dates, fold, sealed = fixture()
    plan = copy.deepcopy(build_date_plans(dates, fold, sealed)['recent24'])
    if defect == 'purge':
        plan['train'][-1] = dates[dates.index(plan['selection'][0])-5]
    elif defect == 'overlap':
        plan['evaluation'][1] = dates[dates.index(plan['evaluation'][0])+1]
    elif defect == 'sealed':
        sealed = '2025-04-01'
    elif defect == 'unsorted':
        plan['train'].reverse()
    else:
        plan['evaluation'][0] = '2025-02-28'
    with pytest.raises(ValueError):
        validate_date_plan(plan, dates, fold, sealed)
