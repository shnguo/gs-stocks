import subprocess
import sys
from pathlib import Path

import numpy as np

from quant_research.plan_value import (
    HEADS,
    apply_calibration,
    choose_research_plan,
    date_weights,
    fit_calibration,
    fit_heads,
    head_targets,
    predict_heads,
)


def test_observed_date_weights_do_not_downweight_dates_with_missing_labels():
    weights = date_weights(['a']*100+['b']*3, np.array([True]*100+[False, False, True]))
    assert np.isclose(weights[:100].sum(), 1)
    assert weights[-1] == 1


def test_no_imputation_of_unknown_return_into_conditional_mean():
    outcomes = {'filled': np.array([[np.nan, 0, 1, 1]]),
                'conditional_net_return': np.array([[np.nan, np.nan, np.nan, .02]]),
                'conditional_loss': np.array([[np.nan, np.nan, np.nan, 0]]),
                'conditional_downside': np.array([[np.nan, np.nan, np.nan, 0]])}
    heads = head_targets(outcomes)
    np.testing.assert_equal(heads[HEADS[1]], [[np.nan, np.nan, 0, 1]])
    np.testing.assert_equal(heads[HEADS[2]], outcomes['conditional_net_return'])


def test_calibration_uses_independent_dates_and_keeps_input_predictions_unchanged():
    prediction = {h: np.full((101, 1), .5 if 'probability' in h else 0.) for h in HEADS}
    actual = np.ones((101, 1))*.01
    actual[-1] = .05
    outcomes = {'filled': np.ones_like(actual), 'conditional_net_return': actual,
                'conditional_loss': np.zeros_like(actual), 'conditional_downside': np.zeros_like(actual)}
    calibration = fit_calibration(prediction, outcomes, ['a']*100+['b'])
    result = apply_calibration(prediction, calibration)
    np.testing.assert_allclose(result[HEADS[2]], .03)
    np.testing.assert_allclose(prediction[HEADS[2]], 0)
    assert calibration[HEADS[2]][0]['known_dates'] == 2
    assert np.all(result[HEADS[0]] == 1)


def test_selection_can_hold_and_does_not_treat_high_unresolved_score_as_mean_profit():
    predictions = {HEADS[0]: np.array([[.8, .9], [.8, .9]]),
                   HEADS[1]: np.array([[.95, .1], [.95, .95]]),
                   HEADS[2]: np.array([[.01, .1], [-.01, -.02]])}
    np.testing.assert_equal(choose_research_plan(predictions), [0, -1])


def _booster_roundtrip(tmp_path):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(700, 3))
    net = (.02*np.sign(x[:, 0]))[:, None]
    outcomes = dict(filled=np.ones_like(net), conditional_net_return=net,
                    conditional_loss=(net < 0).astype(float), conditional_downside=np.maximum(-net, 0))
    protocol = dict(iterations=5, seed=17, tree_parameters=dict(num_leaves=3,
        min_child_samples=10, learning_rate=.1, reg_lambda=0., n_jobs=1, early_stopping_rounds=2))
    a = {k: v[:500] for k, v in outcomes.items()}
    b = {k: v[500:] for k, v in outcomes.items()}
    metadata = fit_heads(x[:500], np.repeat(['a', 'b'], 250), a, x[500:],
        np.repeat(['c', 'd'], 100), b, tmp_path/'models', protocol)
    prediction = predict_heads(x[500:], tmp_path/'models', metadata)
    assert set(prediction) == set(HEADS)
    assert np.all(prediction[HEADS[0]] == 1)
    assert prediction[HEADS[2]][x[500:, 0] > 0].mean() > prediction[HEADS[2]][x[500:, 0] < 0].mean()
    assert np.isfinite(prediction[HEADS[3]]).all()


def test_real_boosters_roundtrip_for_mean_and_probability_heads(tmp_path):
    # The full suite imports Torch's OpenMP runtime. Match production's separate
    # tree process instead of mixing the two runtimes during booster reload.
    code = """import runpy, sys
from pathlib import Path
runpy.run_path(sys.argv[1])['_booster_roundtrip'](Path(sys.argv[2]))
"""
    subprocess.run([sys.executable, '-c', code, str(Path(__file__).resolve()), str(tmp_path)],
                   check=True, capture_output=True, text=True, timeout=45)
