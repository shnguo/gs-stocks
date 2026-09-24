import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from quant_research.kronos_ranker import (
    KronosRanker,
    batch_inputs,
    compare_predictions,
    normalize_windows,
    timestamps,
)


def test_windows_ignore_future_prices_and_factor_anchor():
    values = np.ones((1, 6, 7), dtype=np.float32)
    values[0, :, :4] = np.arange(1, 7)[:, None]
    dates = timestamps(pd.date_range("2025-01-01", periods=6).astype(str).tolist())
    before, stamps = batch_inputs(values, dates, np.array([[0, 3]]), 3, "cpu")
    values[:, 4:] = 1e9
    after, _ = batch_inputs(values, dates, np.array([[0, 3]]), 3, "cpu")
    torch.testing.assert_close(before, after)
    assert stamps[0, -1, 3] == 4
    with pytest.raises(ValueError, match="boundary"):
        batch_inputs(values, dates, np.array([[0, 1]]), 3, "cpu")


def test_adjusted_price_continuity_and_missing_data_rejection():
    raw = np.ones((1, 4, 7), dtype=np.float32)
    raw[0, :, :4] = np.array([100, 102, 51, 52])[:, None]
    raw[0, :, 6] = [1, 1, 2, 2]
    expected = np.array([50, 51, 51, 52], dtype=np.float32)
    expected = (expected - expected.mean()) / (expected.std() + 1e-5)
    np.testing.assert_allclose(normalize_windows(raw)[0, :, 0], expected)
    raw[0, 1, 0] = np.nan
    with pytest.raises(ValueError, match="complete"):
        normalize_windows(raw)


def frames():
    baseline = pd.DataFrame({"date": ["2025-01-01"] * 4,
        "instrument_id": ["a", "b", "c", "d"], "forward_return": [0.1, 0.2, 0.3, np.nan],
        "lightgbm_score": [1, 2, 3, 4], "transformer_score": [3, 2, 1, 0]})
    kronos = baseline[["date", "instrument_id", "forward_return"]].copy()
    kronos["kronos_score"] = [1, 2, 3, 4]
    return baseline, kronos


def test_comparison_aligns_by_identity_and_keeps_unlabelled_scores():
    baseline, kronos = frames()
    merged, metrics = compare_predictions(baseline, kronos.iloc[::-1])
    assert len(merged) == 4
    assert metrics["labelled_rows"] == 3
    assert metrics["kronos"]["mean_rank_ic"] == pytest.approx(1.0)
    assert metrics["formal_ready"] is False


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "label", "nonfinite"])
def test_comparison_rejects_unfair_cohorts(mutation):
    baseline, kronos = frames()
    if mutation == "missing":
        kronos = kronos.iloc[:-1]
    elif mutation == "duplicate":
        kronos = pd.concat([kronos, kronos.iloc[:1]])
    elif mutation == "label":
        kronos.loc[0, "forward_return"] = 7
    else:
        kronos.loc[0, "kronos_score"] = np.nan
    with pytest.raises(ValueError):
        compare_predictions(baseline, kronos)


class Tokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def encode(self, x, half):
        assert half and not self.training and not torch.is_grad_enabled()
        ids = (x[..., 0] * self.weight).long() % 4
        return ids, ids


class Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.table = nn.Embedding(4, 8)

    def forward(self, tokens):
        return self.table(tokens[0]) + self.table(tokens[1])


def test_supervised_gradient_updates_backbone_with_frozen_tokenizer():
    backbone = nn.Module()
    backbone.d_model = 8
    backbone.head = nn.Linear(8, 4)
    backbone.dep_layer = nn.Linear(8, 8)
    backbone.embedding = Embedding()
    backbone.time_emb = nn.Linear(5, 8)
    backbone.token_drop = nn.Identity()
    backbone.transformer = nn.ModuleList([nn.Linear(8, 8)])
    backbone.norm = nn.LayerNorm(8)
    model = KronosRanker(Tokenizer(), backbone).train()
    model(torch.ones(2, 3, 6), torch.zeros(2, 3, 5)).sum().backward()
    assert model.tokenizer.weight.grad is None
    assert backbone.head.weight.grad is None
    assert backbone.transformer[0].weight.grad.abs().sum() > 0
    assert model.rank_head[1].weight.grad.abs().sum() > 0


def test_training_runner_writes_matched_comparison_after_validation(tmp_path, monkeypatch):
    from quant_research.storage import file_hash

    spec = importlib.util.spec_from_file_location("kronos_runner", Path(__file__).parents[1]
                                               / "scripts/kronos_comparison.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    root, bundle = tmp_path / "baseline", tmp_path / "kronos"
    baseline = root / "runs/fold14-h5-seed17"
    baseline.mkdir(parents=True)
    (bundle / "inputs").mkdir(parents=True)
    (bundle / "runs").mkdir()
    for path in [root / "development-acceptance.json", bundle / "inputs/manifest.json",
                 bundle / "pretrained-provenance.json", bundle / "code-manifest.json",
                 bundle / "environment.lock.txt"]:
        path.write_text("{}")
    dates = pd.date_range("2025-01-01", periods=6).astype(str).tolist()
    values = np.ones((4, 6, 7), dtype=np.float32)
    for stock in range(4):
        values[stock, :, :4] = (np.arange(1, 7) ** (1 + stock / 3))[:, None]
    parts = {}
    for name, index in [("train", 2), ("validation", 3), ("test", 4)]:
        parts[name] = pd.DataFrame({"date": [dates[index]] * 4,
            "instrument_id": list("abcd"), "stock_index": range(4), "date_index": index,
            "target": [-0.5, -0.2, 0.2, 0.5], "forward_return": [0.1, 0.2, 0.3, 0.4]})
    predictions = parts["test"].copy()
    predictions["lightgbm_score"] = range(4)
    predictions["transformer_score"] = range(4)
    predictions.to_parquet(baseline / "predictions.parquet", index=False)
    (baseline / "run.json").write_text(json.dumps({"status": "completed", "run_id": "baseline",
        "identity": {"dataset_id": "fixture", "fold": {}, "horizon": 5, "seed": 17,
                     "acceptance_sha256": file_hash(root / "development-acceptance.json")}}))
    panel = SimpleNamespace(dataset_id="fixture", lookback=3)
    monkeypatch.setattr(runner, "context", lambda *a: ({"limitations": ["fixture"]},
                        SimpleNamespace(to_dict=lambda: {}), panel, parts))
    monkeypatch.setattr(runner, "read_inputs", lambda *a: (values, timestamps(dates)))
    monkeypatch.setattr(runner, "register_trials", lambda *a: None)

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

        def forward(self, x, stamp):
            return self.weight * x[:, -1, 0]

    monkeypatch.setattr(runner, "load_pretrained", lambda *a: Toy())
    config = {"fold_index": 14, "device": "cpu", "learning_rate": 1e-5,
              "weight_decay": 0.01, "batch_size": 4, "micro_batch_size": 2,
              "max_epochs": 1, "log_every_steps": 1, "checkpoint_every_steps": 1, "patience": 1}
    runner.train(root, bundle, config, 5, 17)
    output = bundle / "runs/fold14-h5-seed17"
    assert json.loads((output / "run.json").read_text())["status"] == "completed"
    assert json.loads((output / "comparison.json").read_text())["scored_rows"] == 4
    assert (output / "best.pt").exists()
    events = [json.loads(line)["stage"] for line in (output / "events.jsonl").read_text().splitlines()]
    assert events.index("validation") < events.index("test_scoring") < events.index("completed")
