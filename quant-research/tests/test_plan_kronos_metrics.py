import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.plan_value import HEADS


def test_common_forecast_mask_keeps_missing_coverage_visible():
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location('plan_kronos_metrics_test', scripts/'plan_kronos_run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    net = np.zeros((2, 12))
    labels = dict(filled=np.ones_like(net), conditional_net_return=net,
        conditional_loss=net, conditional_downside=net)
    labels['conditional_net_return'][0, 1] = np.nan
    models = {m: {h: np.full((2, 12), .1) for h in HEADS} for m in ['lightgbm', 'transformer', 'kronos']}
    models['kronos'][HEADS[2]][0, 0] = np.nan
    models['transformer'][HEADS[2]][0, 2] = np.nan
    rows = pd.DataFrame(dict(date=['a', 'b']))
    metrics = module.matched_metrics(rows, labels, models, 13, 'raw')
    mean_a = metrics.loc[metrics.date.eq('a') & metrics['head'].eq(HEADS[2])]
    assert (mean_a.cohort_plan_rows == 12).all()
    assert (mean_a.label_known == 11).all()
    assert (mean_a.common_available == 10).all()
    assert (mean_a.common_known == 9).all()
    np.testing.assert_allclose(mean_a.mse, .01)
    assert mean_a.model_available.tolist() == [12, 11, 11]
