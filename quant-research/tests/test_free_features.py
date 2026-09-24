import numpy as np
import pandas as pd
import pytest

from quant_research.free_features import FEATURES, RAW_FIELDS, align


def frames():
    rows = pd.DataFrame(
        {"instrument_id": ["cn.xshg.600036", "cn.xshe.000022"], "date": ["2024-01-02"] * 2}
    )
    bars = rows.assign(close=10.0, volume=100.0)
    source = rows.iloc[:1].copy()
    for k in RAW_FIELDS:
        source[k] = 10.0
    source["float_shares"] = 1000.0
    source["total_shares"] = 2000.0
    source["pe_ttm"] = -5.0
    source["pb"] = 0.0
    return rows, bars, source


def test_missing_retired_member_is_preserved_without_zero_imputation():
    rows, bars, source = frames()
    x, c, r = align(rows, bars, source)
    assert len(x) == 2 and np.isnan(x[1, :-1]).all() and x[1, -1] == 0
    assert c["source_missing"] == 1 and r.tolist() == ["available", "source_missing"]
    assert x[0, FEATURES.index("turnover_pct")] == 10
    assert x[0, FEATURES.index("earnings_yield")] == np.float32(-0.2)
    assert np.isnan(x[0, FEATURES.index("book_to_price")])


def test_future_share_record_cannot_fill_past_date():
    rows, bars, source = frames()
    source["date"] = "2024-01-03"
    x, c, _ = align(rows, bars, source)
    assert c["source_available"] == 0 and np.isnan(x[:, :-1]).all()


def test_mismatched_quote_blocks_all_new_values():
    rows, bars, source = frames()
    source["close"] = 11.0
    x, c, r = align(rows, bars, source)
    assert c["price_mismatch"] == 1 and r[0] == "price_mismatch" and np.isnan(x[0, :-1]).all()


def test_duplicate_sources_are_rejected():
    rows, bars, source = frames()
    with pytest.raises(ValueError, match="Duplicate"):
        align(rows, bars, pd.concat([source, source]))


def test_sealed_dates_are_rejected():
    rows, bars, source = frames()
    rows["date"] = "2025-08-07"
    with pytest.raises(ValueError, match="Sealed"):
        align(rows, bars, source)
