import importlib.util
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "bench", Path(__file__).parents[1] / "scripts/validate_lowvol_benchmarks.py"
)
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


def data(n=9, amount=2000000.0):
    dates = pd.bdate_range("2021-08-02", periods=22 + n).strftime("%Y-%m-%d").tolist()
    f = pd.DataFrame(
        {
            "open": 10.0,
            "close": 10.0,
            "high": 10.0,
            "low": 10.0,
            "volume": 10000.0,
            "amount": amount,
        },
        index=dates,
    )
    return dates[22:], f


def test_partial_entry_is_carried_and_minimum_fees_paid():
    dates, f = data(n=5)
    r, frames = b.etf_account({"x": f}, {"x": {}}, dates, {"x": 0.9}, "base", False)
    buys = frames["fills"]
    assert len(buys) == 5
    assert (buys.notional <= 10000).all()
    assert frames["holdings"].shares.tolist() == [900, 1800, 2700, 3600, 4500]
    assert r["commission"] == 25
    assert frames["nav"].equity.iloc[-1] == pytest.approx(200000 - 25 - buys.slippage.sum())


def test_split_preserves_wealth_and_suspension_preserves_position():
    dates, f = data(n=5, amount=100000000.0)
    f.loc[f.index >= dates[3], ["open", "close", "high", "low"]] = 5.0
    f = f.drop(dates[2])
    _, frames = b.etf_account(
        {"x": f},
        {"x": {dates[3]: {"split": 2.0, "dividend": 0.0}}},
        dates,
        {"x": 0.9},
        "base",
        False,
    )
    nav = frames["nav"]
    held = frames["holdings"]
    assert nav.equity.nunique() == 1
    assert held.shares.iloc[3] == 2 * held.shares.iloc[1]


def test_dividend_counts_once_and_receivable_not_spendable():
    dates, f = data(n=5, amount=100000000.0)
    f.loc[f.index >= dates[2], ["open", "close", "high", "low"]] = 9.0
    _, frames = b.etf_account(
        {"x": f},
        {"x": {dates[2]: {"split": 1.0, "dividend": 1.0}}},
        dates,
        {"x": 0.9},
        "base",
        False,
    )
    nav = frames["nav"]
    held = frames["holdings"]
    assert nav.equity.nunique() == 1
    assert nav.receivable.iloc[-1] == held.shares.iloc[0]
    assert nav.cash.iloc[-1] == nav.cash.iloc[0]
