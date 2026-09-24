import numpy as np
import pandas as pd
import pytest

from quant_research.features import Standardizer, build_panel, stock_features
from quant_research.splits import Fold, rolling_folds
from quant_research.storage import Snapshot, validate_tables
from quant_research.universe import audit_snapshot, cutoff, risk_status, universe_on


def test_future_st_does_not_delete_today_sample(snapshot):
    panel = build_panel(snapshot, 20)
    symbol = "synthetic.xshg.000000"
    before, effective = snapshot.dates[329:331]
    assert universe_on(snapshot, before).set_index("instrument_id").loc[symbol, "eligible"]
    assert not universe_on(snapshot, effective).set_index("instrument_id").loc[symbol, "eligible"]
    sample = panel.samples.loc[(panel.samples.date == before) &
                              (panel.samples.instrument_id == symbol)].iloc[0]
    assert sample.label_status == "available"
    assert sample.label_end == snapshot.dates[350]


def test_status_unknown_is_not_treated_as_normal(snapshot):
    date = snapshot.dates[200]
    snapshot.tables["risk_coverage"].loc[:, "complete"] = False
    assert set(universe_on(snapshot, date).reason) == {"risk_status_unknown"}


def test_effective_date_and_knowledge_timestamp_are_both_required(snapshot):
    date = snapshot.dates[330]
    symbol = "synthetic.xshg.000000"
    snapshot.tables["risk_events"].loc[:, "known_at"] = cutoff(date, "10:00")
    assert risk_status(snapshot, date, cutoff(date, "09:25"))[symbol] == "normal"
    assert risk_status(snapshot, date, cutoff(date, "20:00"))[symbol] == "*ST"


def test_delisted_security_retains_pre_delisting_history(snapshot):
    symbol = "synthetic.xshe.000001"
    snapshot.tables["instruments"].loc[snapshot.tables["instruments"].instrument_id == symbol,
                                       "delisted_at"] = snapshot.dates[200]
    assert universe_on(snapshot, snapshot.dates[199]).set_index("instrument_id").loc[symbol, "eligible"]
    assert not universe_on(snapshot, snapshot.dates[201]).set_index("instrument_id").loc[symbol, "eligible"]


def test_features_do_not_change_when_future_prices_change(snapshot):
    rows = snapshot.tables["bars"]
    bars = rows.loc[rows.instrument_id == rows.instrument_id.iloc[0]].set_index("date")
    before = stock_features(bars)
    bars.loc[bars.index[250:], ["open", "high", "low", "close", "factor", "volume", "amount"]] *= 4
    after = stock_features(bars)
    pd.testing.assert_frame_equal(before.iloc[:250], after.iloc[:250])


def test_missing_market_day_is_not_compressed_into_sequence(snapshot):
    symbol = "synthetic.xshe.000001"
    date = snapshot.dates[200]
    bars = snapshot.tables["bars"]
    snapshot.tables["bars"] = bars.loc[~((bars.date == date) & (bars.instrument_id == symbol))]
    panel = build_panel(snapshot, 5)
    rows = panel.samples.loc[(panel.samples.instrument_id == symbol) &
                             (panel.samples.date >= date) &
                             (panel.samples.date <= snapshot.dates[259])]
    assert rows.empty


@pytest.mark.parametrize("horizon", [5, 20])
def test_labels_use_next_open_and_actual_horizon_maturity(snapshot, horizon):
    panel = build_panel(snapshot, horizon)
    row = panel.samples.loc[panel.samples.date == snapshot.dates[200]].iloc[0]
    bars = snapshot.tables["bars"].set_index(["instrument_id", "date"])
    entry = bars.loc[(row.instrument_id, snapshot.dates[201])]
    exit_bar = bars.loc[(row.instrument_id, snapshot.dates[201 + horizon])]
    assert row.forward_return == pytest.approx(exit_bar.open * exit_bar.factor /
                                               (entry.open * entry.factor) - 1)
    assert row.label_end == snapshot.dates[201 + horizon]


def test_date_partitions_purge_overlapping_targets_and_train_scaler(snapshot):
    panel = build_panel(snapshot, 20)
    fold = Fold(snapshot.dates[120], snapshot.dates[240], snapshot.dates[310], snapshot.dates[375])
    parts = fold.select(panel.samples)
    assert parts["train"].label_end.max() < fold.validation_start
    assert parts["validation"].label_end.max() < fold.test_start
    scaler = Standardizer.fit(panel, parts["train"])
    panel.values[:, 240:] = 1e6
    second = Standardizer.fit(panel, parts["train"])
    np.testing.assert_array_equal(scaler.mean, second.mean)
    np.testing.assert_array_equal(scaler.scale, second.scale)


@pytest.mark.parametrize("issue", ["missing", "suspended", "delisted"])
def test_unavailable_exit_is_retained_for_scoring_without_fabricated_return(snapshot, issue):
    symbol = "synthetic.xshe.000001"
    signal, exit_date = snapshot.dates[200], snapshot.dates[206]
    bars = snapshot.tables["bars"]
    selected = (bars.instrument_id == symbol) & (bars.date == exit_date)
    if issue == "missing":
        snapshot.tables["bars"] = bars.loc[~selected]
    elif issue == "suspended":
        bars.loc[selected, ["volume", "amount"]] = 0
        bars["source_trade_status"] = 1
        bars.loc[selected, "source_trade_status"] = 0
    else:
        snapshot.tables["instruments"].loc[
            snapshot.tables["instruments"].instrument_id == symbol, "delisted_at"] = snapshot.dates[203]
    panel = build_panel(snapshot, 5, risk_policy="include")
    row = panel.samples.loc[(panel.samples.date == signal) &
                            (panel.samples.instrument_id == symbol)].iloc[0]
    assert row.entry_status == "available"
    assert row.label_status == "missing_execution_endpoint"
    assert np.isnan(row.forward_return) and np.isnan(row.target)
    expected = {"missing": "missing_price", "suspended": "suspended_or_no_turnover",
                "delisted": "after_delisting_without_settlement"}
    assert row.exit_status == expected[issue]
    parts = Fold(snapshot.dates[120], snapshot.dates[160], signal, signal).select(panel.samples)
    assert symbol in set(parts["test"].instrument_id)


def test_future_suspension_does_not_change_past_features_or_sample_selection(snapshot):
    before = build_panel(snapshot, 5, risk_policy="include")
    bars = snapshot.tables["bars"]
    bars.loc[bars.date == snapshot.dates[206], ["volume", "amount"]] = 0
    after = build_panel(snapshot, 5, risk_policy="include")
    np.testing.assert_array_equal(before.values[:, :206], after.values[:, :206])
    assert set(before.samples.loc[before.samples.date == snapshot.dates[200], "instrument_id"]) == set(
        after.samples.loc[after.samples.date == snapshot.dates[200], "instrument_id"])


def test_last_year_holdout_is_never_in_development_folds():
    dates = pd.bdate_range("2012-01-02", "2026-09-04").strftime("%Y-%m-%d").tolist()
    schedule = rolling_folds(dates)
    assert schedule["development"]
    assert all(fold["test_end"] < schedule["sealed_holdout_start"]
               for fold in schedule["development"])
    assert not schedule["holdout_included"]


def test_synthetic_data_cannot_pass_formal_gate(snapshot):
    audit = audit_snapshot(snapshot)
    assert not audit["formal_ready"]
    assert "synthetic_data_is_engineering_validation_only" in audit["formal_blockers"]


def test_snapshot_detects_tampered_raw_table(snapshot, tmp_path):
    import shutil
    path = tmp_path / "tampered"
    shutil.copytree(snapshot.path, path)
    with (path / "bars.parquet").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        Snapshot(path)


def test_duplicate_bars_fail_instead_of_picking_latest(snapshot):
    bars = snapshot.tables["bars"]
    snapshot.tables["bars"] = pd.concat([bars, bars.iloc[:1]])
    with pytest.raises(ValueError, match="duplicate business key"):
        validate_tables(snapshot.tables)


def test_unverified_action_boundary_resets_features_without_changing_past(snapshot):
    bars = snapshot.tables["bars"]
    rows = bars.loc[bars.instrument_id == "synthetic.xshe.000001"].set_index("date").copy()
    rows["sequence_id"] = "before"
    before = stock_features(rows)
    boundary = snapshot.dates[250]
    rows.loc[rows.index >= boundary, "sequence_id"] = "after"
    rows.loc[rows.index >= boundary, "factor"] *= 10
    after = stock_features(rows)
    pd.testing.assert_frame_equal(before.loc[before.index < boundary],
                                  after.loc[after.index < boundary])
    assert np.isnan(after.loc[boundary, "return_1"])
    assert np.isnan(after.loc[boundary, "return_60"])
    assert np.isnan(after.loc[boundary, "gap"])
    assert np.isfinite(after.loc[snapshot.dates[310], "return_60"])


def test_action_boundary_invalidates_label_but_retains_prediction_sample(snapshot):
    bars = snapshot.tables["bars"]
    bars["sequence_id"] = bars.instrument_id
    symbol = "synthetic.xshe.000001"
    mask = (bars.instrument_id == symbol) & (bars.date >= snapshot.dates[204])
    bars.loc[mask, "sequence_id"] = symbol + "/unverified-action"
    panel = build_panel(snapshot, 5, risk_policy="include")
    row = panel.samples.loc[(panel.samples.instrument_id == symbol) &
                            (panel.samples.date == snapshot.dates[200])].iloc[0]
    assert row.entry_status == row.exit_status == "available"
    assert row.label_status == "unverified_corporate_action_boundary"
    assert np.isnan(row.forward_return) and np.isnan(row.target)
