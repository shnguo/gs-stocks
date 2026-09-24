"""The exported plan must not change when evaluation outcomes change."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from quant_research.price_strategy import TradeAssumptions


def load_script():
    spec = importlib.util.spec_from_file_location(
        "price_pilot", Path(__file__).parents[1] / "scripts/price_pilot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_decisions_do_not_peek_at_evaluation_labels(tmp_path):
    from dataclasses import asdict
    script = load_script()
    dates = pd.bdate_range("2024-01-01", periods=7).strftime("%Y-%m-%d").tolist()
    (tmp_path / "panel-h5").mkdir()
    (tmp_path / "panel-h5/manifest.json").write_text(json.dumps({"dates": dates}))
    (tmp_path / "config.json").write_text(json.dumps({"root": str(tmp_path),
        "assumptions": asdict(TradeAssumptions())}))
    path = np.array([[9.9, 10.2, 9.8, 10.], [10., 10.9, 9.9, 10.7],
                     [10.7, 10.9, 10.6, 10.8], [10.8, 11., 10.7, 10.9],
                     [10.9, 11.1, 10.8, 11.]])
    for part, size in [("calibration", 40), ("evaluation", 3)]:
        np.savez_compressed(tmp_path / f"{part}.npz", future=np.tile(path, (size, 1, 1)),
            valid=np.ones((size, 5), bool), reference=np.full(size, 10.),
            upper=np.full((size, 5), np.nan), lower=np.full((size, 5), np.nan),
            targets=np.tile(np.log(path / 10.), (size, 1, 1)))
        for model in ["naive", "lightgbm", "kronos"]:
            np.save(tmp_path / f"{model}-{part}.npy", np.zeros((size, 5, 4, 3)))
    pd.DataFrame({"date": [dates[0]] * 3, "instrument_id": ["a", "b", "c"],
        "date_index": [0] * 3, "label_end": [dates[5]] * 3,
        "risk_status": ["unknown"] * 3}).to_parquet(tmp_path / "evaluation-rows.parquet")
    script.report(SimpleNamespace(output=tmp_path))
    first = pd.read_csv(tmp_path / "trade-plans.csv")
    data = script.load_arrays(tmp_path / "evaluation.npz")
    data["future"][:, 1:] = [8., 8.1, 7.9, 8.]
    data["targets"][:] = -.2
    np.savez_compressed(tmp_path / "evaluation.npz", **data)
    script.report(SimpleNamespace(output=tmp_path))
    second = pd.read_csv(tmp_path / "trade-plans.csv")
    plan_columns = ["signal", "candidate_index", "buy", "take_profit", "stop", "fill_probability"]
    pd.testing.assert_frame_equal(first[plan_columns], second[plan_columns])
    assert not np.allclose(first.net_return, second.net_return)
    assert not second.executable.any()
    assert second.buy_valid_until.eq(dates[1]).all()
    assert second.time_exit_date.eq(dates[5]).all()


def test_date_selection_purges_true_fifth_day():
    script = load_script()
    dates = pd.bdate_range("2024-01-01", periods=20).strftime("%Y-%m-%d").tolist()
    chosen = script.pick_dates(dates, dates[0], dates[15], 20, dates[10])
    assert chosen == dates[:5]
