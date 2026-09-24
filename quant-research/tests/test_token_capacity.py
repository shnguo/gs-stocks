import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def comparison(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("compare_token_capacity")


def test_frequency_baseline_uses_only_known_training_targets(comparison):
    train = {"s1": np.array([[3, 0, 0], [3, 0, 2], [3, 1, 1]]),
             "s2": np.array([[3, 0, 0], [3, 0, 2], [3, 1, 1]]),
             "valid": np.array([[True, True], [True, False], [True, True]])}
    evaluation = {"s1": np.array([[3, 0, 0], [3, 2, 2], [3, 0, 0]]),
                  "s2": np.array([[3, 0, 0], [3, 2, 2], [3, 0, 0]]),
                  "valid": np.array([[True, True], [True, True], [False, False]])}
    first = comparison.frequency_baseline(train, ["a", "a", "b"], evaluation, 1, [2, 2])
    assert np.isfinite(first[:2]).all() and np.isnan(first[2]).all()
    assert (first[0] < first[1]).all()
    train["s1"][1, 2] = train["s2"][1, 2] = 3
    altered = comparison.frequency_baseline(train, ["a", "a", "b"], evaluation, 1, [2, 2])
    np.testing.assert_array_equal(first, altered)


def test_paired_interval_has_expected_sign_for_constant_difference(comparison):
    result = comparison.paired_summary([-0.1]*8)
    assert result["dates"] == 8
    assert result["mean_difference"] == pytest.approx(-0.1)
    np.testing.assert_allclose(result["two_date_block_95"], [-0.1, -0.1])
    with pytest.raises(ValueError, match="finite"):
        comparison.paired_summary([np.nan])
