import copy

import pandas as pd
import pytest

from quant_research.backtest import simulate
from quant_research.storage import freeze
from quant_research.universe import audit_snapshot


def training_snapshot(snapshot, tmp_path):
    source = tmp_path / "training-source"
    source.mkdir()
    for name, frame in snapshot.tables.items():
        if name in {"rules", "risk_events", "risk_coverage", "actions"}:
            frame = frame.iloc[:0]
        frame.to_parquet(source / f"{name}.parquet", index=False)
    declaration = copy.deepcopy(snapshot.manifest["declaration"])
    declaration["purpose"] = "training"
    return freeze(source, tmp_path / "training-snapshot", declaration)


def test_missing_execution_contract_is_only_allowed_for_training(snapshot, tmp_path, config):
    training = training_snapshot(snapshot, tmp_path)
    assert training.tables["rules"].empty
    with pytest.raises(ValueError, match="Missing board execution rules"):
        declaration = {**training.manifest["declaration"], "purpose": "full_experiment"}
        freeze(training.path, tmp_path / "bad-snapshot", declaration)
    with pytest.raises(ValueError, match="Training-only snapshot"):
        simulate(training, pd.DataFrame(), 5, config["portfolio"])
    audit = audit_snapshot(training, training_risk_policy="include")
    assert not {"rules_verified_missing", "industry_history_verified_missing",
                "st_history_verified_missing"}.intersection(audit["training_data_blockers"])
    assert {"actions_verified_missing", "historical_universe_verified_missing",
            "historical_information_vintage_unverified"} <= set(audit["training_data_blockers"])
    assert "training_snapshot_has_no_execution_contract" in audit["backtest_data_blockers"]


def test_training_only_runs_models_without_calling_execution(snapshot, tmp_path, config, monkeypatch):
    from quant_research import experiment
    from quant_research.report import render_report
    from quant_research.splits import Fold

    training = training_snapshot(snapshot, tmp_path)
    config["model"]["max_epochs"] = 1
    config["model"]["hidden_size"] = 16
    config["model"]["feedforward_size"] = 32
    config["model"]["layers"] = 1
    dates = training.dates
    fold = Fold(dates[120], dates[150], dates[180], dates[185])

    def reject_execution(*args, **kwargs):
        raise AssertionError("Training-only experiment called portfolio execution")

    monkeypatch.setattr(experiment, "simulate", reject_execution)
    output = tmp_path / "run"
    summary = experiment.run_experiment(training, config, fold, 5, 17, output,
                                        engineering=True, training_only=True,
                                        allow_partial_universe=True)
    assert summary["training_only"]
    assert not summary["portfolios"] and not summary["stress"]
    assert summary["partition_risk_status_counts"]["train"]["unknown"] > 0
    assert set(summary["metrics"]) == {"momentum", "lightgbm", "transformer", "ensemble"}
    assert (output / "transformer.pt").exists()
    assert (output / "predictions-transformer.parquet").exists()
    assert not list(output.glob("*-nav.parquet"))
    report = render_report(output).read_text()
    assert "未运行组合模拟" in report
    assert "模拟净值" not in report
    with pytest.raises(ValueError, match="requires --training-only"):
        experiment.run_experiment(training, config, fold, 5, 17, tmp_path / "bad-run",
                                  engineering=True, allow_partial_universe=True)
