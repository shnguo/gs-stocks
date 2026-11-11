import copy

import pandas as pd
import pytest

from quant_research.earnings_walkforward import (
    historical_development_folds,
    partition_sample_metadata,
)


def calendar():
    return pd.bdate_range("2015-09-01", "2026-09-04").strftime("%Y-%m-%d").tolist()


def test_folds_require_full_three_plus_one_years_quarter_and_normal_exit_tail():
    dates = calendar()
    folds = historical_development_folds(dates, "2025-08-07")
    assert folds[0]["fold_id"] == "2019Q4"
    assert folds[0]["train_start"] == "2015-10-01"
    assert folds[0]["validation_start"] == "2018-10-01"
    assert folds[-1]["fold_id"] == "2025Q2"
    assert len(folds) == 23
    assert all(fold["normal_exit_observed_through"] < "2025-08-07" for fold in folds)
    assert historical_development_folds(dates[:100], "2025-08-07") == []
    shortened = [d for d in dates if d <= "2025-07-10"]
    assert historical_development_folds(shortened, "2025-08-07")[-1]["fold_id"] == "2025Q1"


def metadata_and_fold():
    dates = calendar()
    fold = historical_development_folds(dates, "2025-08-07")[-1]
    boundary = dates.index("2025-04-01")
    rows = [dict(sample_id="val-eligible", date="2024-10-01", feature_end="2024-10-01",
                 label_end=dates[boundary - 21], label_status="available"),
            dict(sample_id="val-gap", date="2024-10-01", feature_end="2024-10-01",
                 label_end=dates[boundary - 20], label_status="available"),
            dict(sample_id="val-delayed-exit", date="2024-10-01", feature_end="2024-10-01",
                 label_end="2025-04-02", label_status="available"),
            dict(sample_id="val-unknown", date="2024-10-01", feature_end="2024-10-01",
                 label_end="", label_status="missing_execution_endpoint"),
            dict(sample_id="train-eligible", date="2023-01-03", feature_end="2023-01-03",
                 label_end="2023-02-01", label_status="available"),
            dict(sample_id="test-unknown", date="2025-06-30", feature_end="2025-06-30",
                 label_end="", label_status="unknown_terminal_value"),
            dict(sample_id="test-sealed-exit", date="2025-06-27", feature_end="2025-06-27",
                 label_end="2025-08-08", label_status="available")]
    return pd.DataFrame(rows), fold, dates


def test_actual_endpoint_purge_clear_session_gap_and_unknown_evaluation_retention():
    samples, fold, dates = metadata_and_fold()
    result = partition_sample_metadata(samples, fold, dates, "2025-08-07")
    assert result["train"].sample_id.tolist() == ["train-eligible"]
    assert result["validation"].sample_id.tolist() == ["val-eligible"]
    assert set(result["test"].sample_id) == {"test-unknown", "test-sealed-exit"}
    reasons = result["audit"].set_index("sample_id").partition_reason
    assert reasons["val-gap"] == "fewer_than_20_clear_sessions_after_label_end"
    assert reasons["val-delayed-exit"] == "label_crosses_next_partition"
    assert reasons["val-unknown"] == "label_unavailable_not_fit"
    assert reasons["test-sealed-exit"] == "evaluation_endpoint_enters_sealed_interval_do_not_read_outcome"
    assert not result["historical_return_experiment_run"]


def test_outcome_columns_cannot_pass_through_metadata_partitioning():
    samples, fold, dates = metadata_and_fold()
    samples["target"] = .5
    with pytest.raises(ValueError, match="metadata columns only"):
        partition_sample_metadata(samples, fold, dates, "2025-08-07")


@pytest.mark.parametrize("change", ["duplicate", "future_feature", "unobserved_endpoint", "sealed_date", "gap"])
def test_invalid_clocks_duplicate_ids_and_relaxed_gap_are_rejected(change):
    samples, fold, dates = metadata_and_fold()
    if change == "duplicate":
        samples.loc[1, "sample_id"] = samples.loc[0, "sample_id"]
    elif change == "future_feature":
        samples.loc[0, "feature_end"] = "2024-10-02"
    elif change == "unobserved_endpoint":
        samples.loc[0, "label_end"] = "2099-01-01"
    elif change == "sealed_date":
        samples.loc[0, ["date", "feature_end"]] = "2025-08-08"
    else:
        fold = copy.deepcopy(fold)
        fold["embargo_sessions"] = 19
    with pytest.raises(ValueError):
        partition_sample_metadata(samples, fold, dates, "2025-08-07")
