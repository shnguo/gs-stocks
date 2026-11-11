import pandas as pd
import pytest

from quant_research.us_fundamentals import (instant_asof, mapping_on, shares_asof, ttm_asof,
                                            value_yields)


def facts(rows):
    return pd.DataFrame(rows, columns=["field", "start", "end", "val", "filed", "accn"])


def test_ttm_rolls_annual_forward_with_ytd_and_respects_filing_date():
    f = facts([
        ("net_income", "2022-01-01", "2022-12-31", 100.0, "2023-02-15", "a1"),
        ("net_income", "2022-01-01", "2022-06-30", 40.0, "2022-08-01", "q2-22"),
        ("net_income", "2023-01-01", "2023-06-30", 70.0, "2023-08-01", "q2-23"),
        ("net_income", "2023-01-01", "2023-12-31", 150.0, "2024-02-15", "a2"),
    ])
    assert ttm_asof(f, ["net_income"], "2023-03-01") == 100.0
    assert ttm_asof(f, ["net_income"], "2023-08-01") == pytest.approx(100 + 70 - 40)
    assert ttm_asof(f, ["net_income"], "2024-03-01") == 150.0
    assert ttm_asof(f, ["net_income"], "2023-01-31") is None  # nothing annual filed yet


def test_ttm_uses_latest_restatement_known_at_date():
    f = facts([
        ("net_income", "2022-01-01", "2022-12-31", 100.0, "2023-02-15", "a1"),
        ("net_income", "2022-01-01", "2022-12-31", 90.0, "2023-05-01", "a1-amended"),
    ])
    assert ttm_asof(f, ["net_income"], "2023-03-01") == 100.0
    assert ttm_asof(f, ["net_income"], "2023-06-01") == 90.0


def test_shares_sum_classes_from_latest_filing():
    f = facts([
        ("shares", None, "2023-01-20", 500.0, "2023-02-01", "k"),
        ("shares", None, "2023-04-20", 300.0, "2023-05-01", "q"),
        ("shares", None, "2023-04-20", 200.0, "2023-05-01", "q"),
    ])
    assert shares_asof(f, "2023-03-01") == (500.0, "2023-01-20")
    assert shares_asof(f, "2023-06-01") == (500.0, "2023-04-20")
    assert shares_asof(f, "2023-01-01") == (None, None)


def test_instant_and_yields():
    f = facts([("equity", None, "2023-03-31", 80.0, "2023-05-01", "q"),
               ("equity", None, "2020-03-31", 50.0, "2020-05-01", "old")])
    assert instant_asof(f, ["equity"], "2023-06-01") == 80.0
    assert instant_asof(f, ["equity"], "2023-04-15") is None  # 2020 value too old
    y = value_yields({"net_income": 10.0, "equity": None, "revenue": 50.0, "operating_cash": -5.0}, 100.0)
    assert y["earnings_yield"] == 0.1 and y["sales_yield"] == 0.5 and y["cashflow_yield"] == -0.05
    assert pd.isna(y["book_to_price"])


def test_mapping_on_uses_latest_snapshot_before_date():
    history = pd.DataFrame({"snapshot": ["2020-01-01", "2020-06-01"], "cik": [1, 1],
                            "ticker": ["FB", "META"], "title": ["x", "x"]})
    assert mapping_on(history, "2020-03-01").ticker.tolist() == ["FB"]
    assert mapping_on(history, "2021-01-01").ticker.tolist() == ["META"]
    assert mapping_on(history, "2019-01-01").ticker.tolist() == ["FB"]
