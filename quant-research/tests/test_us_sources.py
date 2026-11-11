import pandas as pd

from quant_research.us_sources import alpaca_symbol, gap_segments, membership_runs, ndx_membership


def test_ndx_membership_walks_changes_backwards():
    current = {"A", "B", "NEW"}
    changes = pd.DataFrame({"date": ["2020-03-15", "2021-06-01"],
                            "added": ["B", "NEW"], "removed": ["OLD", None]})
    m = ndx_membership(current, changes, "2026-01-01", ["2020-01-31", "2020-03-31", "2021-06-30"])
    assert m["2021-06-30"] == {"A", "B", "NEW"}
    assert m["2020-03-31"] == {"A", "B"}
    assert m["2020-01-31"] == {"A", "OLD"}
    # A change on the date itself is already in effect.
    assert ndx_membership(current, changes, "2026-01-01", ["2021-06-01"])["2021-06-01"] == current


def test_membership_runs_split_on_gaps_and_renames():
    monthly = {"2020-01": {"FB", "X"}, "2020-02": {"FB"}, "2020-03": {"META", "X"}}
    runs = membership_runs(monthly)
    got = sorted(map(tuple, runs.to_numpy().tolist()))
    assert got == [("FB", "2020-01", "2020-02"), ("META", "2020-03", "2020-03"),
                   ("X", "2020-01", "2020-01"), ("X", "2020-03", "2020-03")]


def test_alpaca_symbol_class_shares():
    assert alpaca_symbol("BRK-B") == "BRK.B" and alpaca_symbol("BF.B") == "BF.B"


def test_gap_segments_split_long_gaps_only():
    calendar = [f"d{i:03d}" for i in range(100)]
    dates = pd.Series(["d000", "d001", "d015", "d016", "d060", "d061"])
    assert gap_segments(dates, calendar, 20).tolist() == [0, 0, 0, 0, 1, 1]
