import numpy as np
import pandas as pd
import pytest
import torch

from quant_research.price_calibration import apply_intervals, fit_intervals
from quant_research.price_transformer import PriceTransformer, masked_pinball, windows


def test_empirical_calibration_weights_dates_and_preserves_median():
    q = np.zeros((101, 5, 4, 3))
    y = np.zeros((101, 5, 4))
    y[-1] = 1
    fit = fit_intervals(q, y, ['a'] * 100 + ['b'], min_rows=1, min_dates=2)
    # The one-stock date has equal weight to the 100-stock date.
    np.testing.assert_array_equal(fit['offsets'], np.ones((5, 4)))
    adjusted = apply_intervals(q, fit['offsets'])
    np.testing.assert_array_equal(adjusted[..., 1], q[..., 1])
    assert (adjusted[..., 0] <= q[..., 0]).all()
    assert (adjusted[..., 2] >= q[..., 2]).all()
    assert fit['coverage_guarantee'] is False


def test_calibration_missing_evidence_abstains_and_cannot_shrink():
    q = np.zeros((120, 5, 4, 3))
    q[..., 0], q[..., 2] = -1, 1
    y = np.zeros((120, 5, 4))
    y[:, 4, 3] = np.nan
    fit = fit_intervals(q, y, np.repeat(np.arange(6), 20))
    assert fit['offsets'][0, 0] == 0
    assert np.isnan(apply_intervals(q, fit['offsets'])[:, 4, 3]).all()
    with pytest.raises(ValueError):
        apply_intervals(q, np.full((5, 4), -1.))


def test_price_transformer_order_and_masked_gradient():
    torch.set_num_threads(2)
    model = PriceTransformer()
    prediction = model(torch.randn(3, 60, 27))
    assert prediction.shape == (3, 5, 4, 3)
    assert torch.all(prediction[..., 0] <= prediction[..., 1])
    assert torch.all(prediction[..., 1] <= prediction[..., 2])
    prediction.retain_grad()
    target = torch.zeros(3, 5, 4)
    target[0] = float('nan')
    target[1, 4] = float('nan')
    loss, known = masked_pinball(prediction, target)
    assert known.tolist() == [False, True, True]
    loss.mean().backward()
    assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad[0]) == 0
    assert torch.count_nonzero(prediction.grad[1, 4]) == 0
    assert torch.count_nonzero(prediction.grad[2]) > 0


def test_window_cannot_read_future_features():
    raw = np.arange(2 * 100 * 27, dtype=np.float32).reshape(2, 100, 27)
    rows = pd.DataFrame({'stock_index': [0, 1], 'date_index': [65, 70]})
    mean, scale = np.zeros(27), np.full(27, 10000)
    before = windows(raw, rows, np.arange(2), mean, scale)
    raw[0, 66:] = np.nan
    raw[1, 71:] = np.nan
    np.testing.assert_array_equal(windows(raw, rows, np.arange(2), mean, scale), before)
    raw[0, 65] = np.nan
    with pytest.raises(ValueError):
        windows(raw, rows, np.arange(2), mean, scale)


def test_followup_report_keeps_decisions_when_evaluation_future_changes(tmp_path, monkeypatch):
    import importlib.util
    import json
    from dataclasses import asdict
    from pathlib import Path

    from quant_research.price_strategy import TradeAssumptions

    scripts = Path(__file__).parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('price_next_test', scripts / 'price_next.py')
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    p, out = tmp_path / 'prior', tmp_path / 'out'
    p.mkdir()
    out.mkdir()
    (p / 'panel-h5').mkdir()
    dates = pd.bdate_range('2024-01-01', periods=30).strftime('%Y-%m-%d').tolist()
    (p / 'panel-h5/manifest.json').write_text(json.dumps({'dates': dates}))
    (p / 'config.json').write_text(json.dumps({'root': str(p), 'assumptions': asdict(TradeAssumptions())}))
    path = np.array([[9.9, 10.2, 9.8, 10.], [10., 10.9, 9.9, 10.7],
                     [10.7, 10.9, 10.6, 10.8], [10.8, 11., 10.7, 10.9],
                     [10.9, 11.1, 10.8, 11.]])
    for part, size in [('calibration', 240), ('evaluation', 3)]:
        np.savez(p / f'{part}.npz', future=np.tile(path, (size, 1, 1)),
                 valid=np.ones((size, 5), bool), reference=np.full(size, 10.),
                 targets=np.tile(np.log(path / 10), (size, 1, 1)),
                 upper=np.full((size, 5), np.nan), lower=np.full((size, 5), np.nan))
        ds = np.repeat(dates[:12], 20) if part == 'calibration' else [dates[20]] * size
        pd.DataFrame({'date': ds, 'instrument_id': [f's{i}' for i in range(size)],
                      'date_index': [20] * size, 'label_end': [dates[25]] * size}).to_parquet(p / f'{part}-rows.parquet')
        for model in ['naive', 'lightgbm', 'kronos', 'transformer']:
            np.save((out if model == 'transformer' else p) / f'{model}-{part}.npy',
                    np.zeros((size, 5, 4, 3)))
    cfg = {'prior': str(p), 'interval_calibration_dates': dates[:6], 'plan_calibration_dates': dates[6:12]}
    script.report(out, cfg)
    before = pd.read_csv(out / 'trade-plans.csv')
    assert before.signal.eq('research_candidate').all()
    assert not before.executable.any()
    data = script.load_arrays(p / 'evaluation.npz')
    data['future'][:, 1:] = [8., 8.1, 7.9, 8.]
    data['targets'] = np.log(data['future'] / 10.)
    np.savez(p / 'evaluation.npz', **data)
    script.report(out, cfg)
    after = pd.read_csv(out / 'trade-plans.csv')
    decisions = ['model', 'date', 'instrument_id', 'signal', 'buy', 'take_profit', 'stop',
                 'buy_valid_until', 'time_exit_date', 'candidate_index', 'expected_net_per_order']
    pd.testing.assert_frame_equal(before[decisions], after[decisions])
    assert not np.allclose(before.net_return, after.net_return)
