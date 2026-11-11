from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from quant_research.factors import price_panel
from quant_research.scorecard import (
    NEW_YORK,
    basket_members,
    holding_returns,
    independent,
    last_complete_day,
    score_run,
    summarize,
    timing,
    verified_files,
)
from quant_research.storage import file_hash

CFG = {"holding_sessions": 3, "cost_one_way": 0.0005, "min_stocks": 2, "price_check_tolerance": 0.05,
       "baskets": [{"key": "top", "column": "drift_quintile", "equals": 5},
                   {"key": "first", "column": "rank", "max": 1},
                   {"key": "mild", "column": "pullback_mild", "equals": True}]}


def bars(prices: dict[str, list], dates: list[str]) -> pd.DataFrame:
    """One (open, close) pair per session; None = no bar that day."""
    rows = []
    for name, series in prices.items():
        for date, pair in zip(dates, series):
            if pair is not None:
                rows.append(dict(instrument_id=name, date=date, open=pair[0], high=max(pair), low=min(pair),
                                 close=pair[1], volume=1000.0, factor=1.0))
    return pd.DataFrame(rows)


def panel_for(sessions: int):
    dates = list(pd.bdate_range("2026-11-06", periods=6).strftime("%Y-%m-%d"))[:sessions]
    prices = {
        "benchmark": [(100, 100), (100, 101), (101, 102), (102, 103), (104, 104), (105, 105)],
        "UP":        [(10, 10), (10, 11), (11, 12), (12, 12), (13, 13), (14, 14)],        # +30% open to open
        "DOWN":      [(20, 20), (20, 19), (19, 18), (18, 18), (16, 16), (15, 15)],        # -20%
        "GONE":      [(50, 50), (50, 55), (55, 60), None, None, None],                    # stops trading: last close 60
        "LATE":      [(30, 30), None, (30, 31), (31, 32), (33, 33), (34, 34)],            # no bar on the entry session
    }
    return price_panel(bars({k: v[:sessions] for k, v in prices.items()}, dates)), dates


def ranking() -> pd.DataFrame:
    return pd.DataFrame({"rank": [1, 2, 3, 4], "symbol": ["UP", "GONE", "DOWN", "LATE"],
                         "drift_percentile": [1.0, 0.75, 0.5, 0.25], "drift_quintile": [5, 5, 1, 1],
                         "last_close": [10.0, 50.0, 20.0, 60.0]})


def test_mature_run_uses_entry_open_to_exit_open():
    panel, dates = panel_for(6)
    values, state = holding_returns(panel, dates[0], 3)
    got = dict(zip(panel.ids, values))
    assert state == {"status": "mature", "entry_date": dates[1], "exit_date": dates[4], "sessions_held": 3}
    assert got["UP"] == pytest.approx(0.30) and got["DOWN"] == pytest.approx(-0.20)
    assert got["GONE"] == pytest.approx(60 / 50 - 1)      # valued at its last traded close
    assert np.isnan(got["LATE"])                          # could not be bought at the entry open
    assert got["benchmark"] == pytest.approx(0.04)


def test_unfinished_run_is_marked_at_the_last_close_and_never_called_mature():
    panel, dates = panel_for(4)                           # exit open (session 5) not seen yet
    values, state = holding_returns(panel, dates[0], 3)
    got = dict(zip(panel.ids, values))
    assert state["status"] == "interim" and state["exit_date"] is None and state["sessions_held"] == 3
    assert got["UP"] == pytest.approx(0.20) and got["GONE"] == pytest.approx(0.20)
    panel, dates = panel_for(1)
    values, state = holding_returns(panel, dates[0], 3)
    assert state["status"] == "waiting" and np.isnan(values).all()


def test_timing_needs_the_run_written_before_the_entry_open():
    assert timing("2026-11-09T03:23:34+00:00", "2026-11-09") == "prospective"   # Sunday night New York
    assert timing("2026-11-09T14:29:59+00:00", "2026-11-09") == "prospective"   # 09:29:59 New York
    assert timing("2026-11-09T14:30:00+00:00", "2026-11-09") == "late"
    assert timing("2026-11-06T07:58:12+00:00", "2026-11-02") == "late"          # written days after entry


def test_last_complete_day_waits_for_the_close():
    assert last_complete_day(datetime(2026, 11, 9, 16, 59, tzinfo=NEW_YORK)) == "2026-11-08"
    assert last_complete_day(datetime(2026, 11, 9, 17, 0, tzinfo=NEW_YORK)) == "2026-11-09"


def test_score_run_baskets_costs_and_missing_columns():
    panel, dates = panel_for(6)
    outcome, stocks = score_run(ranking(), panel, "benchmark", dates[0], CFG)
    universe = (0.30 + 0.20 - 0.20) / 3                   # LATE has no return and is left out
    assert outcome["status"] == "mature" and outcome["sessions_left"] == 0
    assert outcome["priced"] == 3 and outcome["scored"] == 4
    assert outcome["universe_return"] == pytest.approx(universe, abs=1e-6)
    assert outcome["benchmark_return"] == pytest.approx(0.04)
    top = outcome["baskets"]["top"]
    assert top["listed"] == 2 and top["stocks"] == 2
    assert top["gross"] == pytest.approx(0.25) and top["net"] == pytest.approx(0.249)   # full round trip
    assert top["excess_universe"] == pytest.approx(0.249 - universe, abs=1e-6)
    assert top["excess_benchmark"] == pytest.approx(0.209)
    assert outcome["baskets"]["first"]["net"] is None      # one stock is under min_stocks
    assert "mild" not in outcome["baskets"]                # older runs have no such column
    assert outcome["quintiles"][4] == pytest.approx(0.25) and outcome["quintiles"][1] is None
    assert outcome["price_mismatches"] == 1                # LATE's frozen close does not match its bar
    assert stocks.set_index("symbol")["return"]["DOWN"] == pytest.approx(-0.20)


def test_score_run_interim_counts_sessions_left():
    panel, dates = panel_for(3)
    outcome, _ = score_run(ranking(), panel, "benchmark", dates[0], CFG)
    assert outcome["status"] == "interim" and outcome["sessions_held"] == 2 and outcome["sessions_left"] == 2
    panel, dates = panel_for(1)
    outcome, _ = score_run(ranking(), panel, "benchmark", dates[0], CFG)
    assert outcome["status"] == "waiting" and outcome["baskets"] == {} and outcome["universe_return"] is None


def test_basket_members_reads_text_booleans():
    frame = pd.DataFrame({"flag": ["True", "False", None], "rank": [1, 60, 3]})
    assert basket_members(frame, {"column": "flag", "equals": True}).tolist() == [True, False, False]
    assert basket_members(frame, {"column": "rank", "max": 50}).tolist() == [True, False, True]
    assert basket_members(frame, {"column": "absent", "equals": True}) is None


def run(as_of, entry, exit_, timing_, excess, status="mature"):
    return {"as_of": as_of, "entry_date": entry, "exit_date": exit_, "status": status, "timing": timing_,
            "rank_ic": 0.02, "baskets": {"top": {"net": excess + 0.01, "excess_universe": excess,
                                                 "excess_benchmark": excess - 0.01}}}


def test_summary_keeps_late_and_unfinished_runs_out_of_the_record():
    runs = [run("d01", "d02", "d23", "prospective", 0.02), run("d02", "d03", "d24", "prospective", -0.01),
            run("d22", "d23", "d44", "prospective", 0.04), run("d00", "d01", "d22", "late", 0.50),
            run("d30", "d31", None, "prospective", 0.90, status="interim")]
    assert [r["as_of"] for r in independent(runs[:3])] == ["d01", "d22"]
    record = summarize(runs, ["top"], min_independent=2)
    forward = record["prospective"]
    assert forward["runs"] == 3 and forward["independent_runs"] == 2 and forward["enough"]
    assert forward["baskets"]["top"]["all"]["mean_excess_universe"] == pytest.approx(0.05 / 3, abs=1e-6)
    assert forward["baskets"]["top"]["independent"]["mean_excess_universe"] == pytest.approx(0.03)
    assert forward["baskets"]["top"]["all"]["share_beating_universe"] == pytest.approx(2 / 3, abs=1e-6)
    assert record["late"]["runs"] == 1 and not record["late"]["enough"]
    assert summarize(runs, ["top"], min_independent=3)["prospective"]["enough"] is False


def test_verified_files_detects_a_changed_file(tmp_path):
    (tmp_path / "ranking.csv").write_text("a\n1\n")
    files = {"ranking.csv": file_hash(tmp_path / "ranking.csv")}
    verified_files(tmp_path, files)
    (tmp_path / "ranking.csv").write_text("a\n2\n")
    with pytest.raises(ValueError):
        verified_files(tmp_path, files)
