import numpy as np
import pandas as pd
import pytest

from quant_research.daily_input import reference_chain


def chain():
    anchor = pd.Series({"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "factor": 2.0})
    updates = pd.DataFrame(
        [
            {
                "date": "2026-09-07",
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "pre_close": 10.0,
                "volume": 100.0,
                "amount": 1000.0,
            },
            {
                "date": "2026-09-08",
                "open": 10.5,
                "high": 11.0,
                "low": 10.0,
                "close": 10.8,
                "pre_close": 10.5,
                "volume": 100.0,
                "amount": 1000.0,
            },
        ]
    ).set_index("date")
    return anchor, anchor.copy(), updates, ["2026-09-07", "2026-09-08"]


def test_unchanged_reference_allows_extension_without_fabricating_an_event():
    a, o, u, d = chain()
    assert reference_chain(a, o, u, d) == (True, "source_reference_continuity_only")


@pytest.mark.parametrize(
    "kind",
    [
        "reference_change",
        "missing_reference",
        "missing_day",
        "halt",
        "anchor_mismatch",
        "factor_missing",
    ],
)
def test_reference_or_missing_barrier_cannot_be_imputed(kind):
    a, o, u, d = chain()
    if kind == "reference_change":
        u.loc[d[1], "pre_close"] = 9.5
    if kind == "missing_reference":
        u.loc[d[1], "pre_close"] = np.nan
    if kind == "missing_day":
        u = u.drop(d[0])
    if kind == "halt":
        u.loc[d[0], "volume"] = 0.0
    if kind == "anchor_mismatch":
        o["close"] = 10.02
    if kind == "factor_missing":
        a["factor"] = np.nan
    assert not reference_chain(a, o, u, d)[0]
