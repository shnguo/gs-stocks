import numpy as np
import pandas as pd
from review_daily_early_stage import mature


def test_mature_attaches_selected_and_ideal_outcomes():
    dates = pd.date_range("2026-01-01", periods=6).strftime("%Y-%m-%d").tolist()
    bars = pd.DataFrame({
        "instrument_id": ["cn.xshg.600001"]*6,
        "date": dates,
        "open": [10, 10, 10, 10, 10, 10],
        "high": [10.2, 10.5, 11, 12, 11.5, 11],
        "low": [9.8, 9.5, 9.7, 9.8, 9.9, 10],
        "close": [10]*6,
        "volume": [100]*6,
        "amount": [1000]*6,
        "factor": [1]*6,
        "source_trade_status": [1]*6,
    })
    cohort = pd.DataFrame({
        "local_row": [1],
        "date": [dates[0]],
        "instrument_id": ["cn.xshg.600001"],
        "sell_reference_date": [dates[3]],
    })
    found = mature(cohort, bars, dates[1:], .0025).iloc[0]
    assert found.label_known
    np.testing.assert_allclose(found.actual_execution_return, .1975)
    np.testing.assert_allclose(found.actual_extrema_scenario, 12/9.5-1-.0025)
    np.testing.assert_allclose(found.actual_ideal_execution_return, .1975)
    np.testing.assert_allclose(found.actual_adverse, -.05)


def test_mature_keeps_unknown_rows_without_imputation():
    dates = pd.date_range("2026-01-01", periods=6).strftime("%Y-%m-%d").tolist()
    bars = pd.DataFrame({
        "instrument_id": ["cn.xshg.600001"]*6,
        "date": dates,
        "open": [10]*6,
        "high": [11]*6,
        "low": [9]*6,
        "close": [10]*6,
        "volume": [100]*6,
        "amount": [1000]*6,
        "factor": [1]*6,
        "source_trade_status": [1, 1, 1, 0, 1, 1],
    })
    cohort = pd.DataFrame({
        "local_row": [1],
        "date": [dates[0]],
        "instrument_id": ["cn.xshg.600001"],
        "sell_reference_date": [dates[3]],
    })
    found = mature(cohort, bars, dates[1:], .0025).iloc[0]
    assert not found.label_known
    assert np.isnan(found.actual_execution_return)
