from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_research.backtest import simulate
from quant_research.config import load_config
from quant_research.features import build_panel
from quant_research.universe import audit_snapshot, universe_on


@pytest.mark.parametrize("status", ["normal", "ST", "*ST", "unknown"])
def test_inclusive_training_preserves_status_and_trading_restrictions(snapshot, config, status):
    symbol = "synthetic.xshg.000000"
    date = snapshot.dates[330]
    snapshot.tables["risk_events"].loc[:, "status"] = status
    panel = build_panel(snapshot, 5, risk_policy=config["training_risk_policy"])
    sample = panel.samples.loc[(panel.samples.date == date) &
                              (panel.samples.instrument_id == symbol)].iloc[0]
    assert sample.risk_status == status
    assert sample.label_status == "available"
    assert bool(sample.trading_eligible) == (status == "normal")
    assert panel.values.shape[-1] == 25
    assert "risk_status" not in panel.feature_names
    predictions = pd.DataFrame([dict(date=date, instrument_id=symbol, score=1.0)])
    result = simulate(snapshot, predictions, 5, config["portfolio"])
    assert result["fills"].empty == (status != "normal")


def test_missing_st_source_does_not_change_inclusive_features_targets_or_sample_ids(snapshot):
    before = build_panel(snapshot, 20, risk_policy="include")
    snapshot.tables["risk_events"] = snapshot.tables["risk_events"].iloc[:0]
    snapshot.tables["risk_coverage"] = snapshot.tables["risk_coverage"].iloc[:0]
    after = build_panel(snapshot, 20, risk_policy="include")
    assert set(after.samples.risk_status) == {"unknown"}
    assert not after.samples.trading_eligible.any()
    assert not after.samples.empty
    pd.testing.assert_frame_equal(
        before.samples.drop(columns=["risk_status", "trading_eligible"]),
        after.samples.drop(columns=["risk_status", "trading_eligible"]))
    np.testing.assert_array_equal(before.values, after.values)
    assert build_panel(snapshot, 20, risk_policy="exclude_st").samples.empty


def test_inclusive_training_keeps_listing_boundaries(snapshot):
    symbols = snapshot.tables["instruments"].instrument_id.tolist()
    date = snapshot.dates[200]
    snapshot.tables["instruments"].loc[0, "listed_at"] = snapshot.dates[201]
    snapshot.tables["instruments"].loc[1, "delisted_at"] = snapshot.dates[199]
    rows = universe_on(snapshot, date, risk_policy="include").set_index("instrument_id")
    assert rows.loc[symbols[0], "reason"] == "not_yet_listed"
    assert rows.loc[symbols[1], "reason"] == "delisted"


def test_st_waiver_is_specific_to_training_and_does_not_certify_data(snapshot):
    snapshot.manifest["declaration"]["st_history_verified"] = False
    snapshot.tables["risk_coverage"].loc[:, "complete"] = False
    audit = audit_snapshot(snapshot, training_risk_policy="include")
    st_blockers = {"st_history_verified_missing", "risk_status_unknown_in_expected_universe"}
    assert not st_blockers.intersection(audit["training_data_blockers"])
    assert st_blockers <= set(audit["backtest_data_blockers"])
    assert not audit["training_data_ready"]
    assert "synthetic_data_is_engineering_validation_only" in audit["training_data_blockers"]
    assert audit["declaration"]["st_history_verified"] is False
    strict = audit_snapshot(snapshot, training_risk_policy="exclude_st")
    assert st_blockers <= set(strict["training_data_blockers"])


def test_configuration_preserves_old_runs_and_rejects_policy_typos(tmp_path, config):
    assert config["training_risk_policy"] == "include"
    source = (Path(__file__).parents[1] / "configs/research.toml").read_text()
    path = tmp_path / "config.toml"
    path.write_text(source.replace('training_risk_policy = "include"\n', ""))
    assert load_config(path)["training_risk_policy"] == "exclude_st"
    path.write_text(source.replace('training_risk_policy = "include"',
                                   'training_risk_policy = "incldue"'))
    with pytest.raises(ValueError, match="Invalid training risk policy"):
        load_config(path)
