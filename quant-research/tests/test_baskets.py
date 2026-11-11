import importlib.util
from pathlib import Path

import numpy as np

from quant_research.baskets import BasketBook

BASE = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, BASE / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_basket_book_excess_turnover_and_small_baskets():
    ids = np.array([f"s{i}" for i in range(10)])
    returns = np.arange(10) / 100.0          # pool mean 4.5%
    valid = np.ones(10, dtype=bool)
    book = BasketBook(["top", "tiny"], cost_one_way=0.001)
    top = np.arange(10) >= 5                 # s5..s9, mean 7%
    book.add("2020-01-31", ids, returns, valid, {"top": top, "tiny": np.arange(10) == 0})
    shifted = (np.arange(10) >= 4) & (np.arange(10) <= 8)  # s4..s8 next month: one name changes of five
    book.add("2020-02-29", ids, returns, valid, {"top": shifted, "tiny": np.arange(10) == 0})
    frame = book.frame()
    assert np.isclose(frame.top_excess[0], 0.025)
    assert np.isclose(frame.top_excess_net[0], 0.025 - 2 * 0.001 * 1.0)  # first month: full turnover
    assert np.isclose(frame.top_turnover[1], 0.2)
    assert np.isnan(frame.tiny_excess[0])    # fewer than 5 stocks: skipped


def test_checklist_signals_and_missing_inputs():
    tf = load("turnaround_fundamental")
    nan = np.nan
    now = {"net_income": np.array([10.0, -5.0, 10.0]), "operating_cash": np.array([15.0, -6.0, 15.0]),
           "revenue": np.array([110.0, 90.0, 110.0]), "gross_profit": np.array([44.0, 27.0, nan]),
           "assets": np.array([100.0, 100.0, 100.0]), "assets_current": np.array([30.0, 10.0, nan]),
           "liabilities_current": np.array([15.0, 10.0, nan]), "lt_debt": np.array([20.0, 50.0, nan])}
    prev = {"net_income": np.array([-2.0, 5.0, 8.0]), "operating_cash": np.array([5.0, 5.0, 8.0]),
            "revenue": np.array([100.0, 100.0, 100.0]), "gross_profit": np.array([35.0, 35.0, nan]),
            "assets": np.array([100.0, 100.0, 100.0]), "assets_current": np.array([20.0, 20.0, nan]),
            "liabilities_current": np.array([20.0, 10.0, nan]), "lt_debt": np.array([30.0, 40.0, nan])}
    out = tf.checklist(now, prev, np.array([0.0, 0.10, nan]), 0.01)
    # Company 0 improved on everything; company 1 got worse on everything.
    assert out["signals_available"][0] == 9 and out["signals_passed"][0] == 9
    assert out["signals_available"][1] == 9 and out["signals_passed"][1] == 0
    # Company 2 lacks balance-sheet detail and share data: only 5 signals can be judged.
    assert out["signals_available"][2] == 5
    assert np.isclose(out["delta_roa"][0], 0.12) and np.isclose(out["revenue_growth"][1], -0.10)


def test_top_n_plain_and_sector_capped():
    from quant_research.baskets import top_n
    score = np.array([9.0, 8.0, 7.0, 6.0, 5.0, np.nan, 4.0])
    mask = np.array([True, True, True, True, True, True, False])
    assert top_n(score, mask, 3).tolist() == [True, True, True, False, False, False, False]
    assert top_n(score, mask, 10).sum() == 5                 # NaN and masked-out never picked
    groups = np.array(["a", "a", "a", "b", None, "b", "b"], dtype=object)
    capped = top_n(score, mask, 4, groups, cap=2)            # third "a" skipped, None is its own bucket
    assert capped.tolist() == [True, True, False, True, True, False, False]


def test_shortlist_rules_pick_inside_their_group():
    sc = load("shortlist_check")
    cfg = {"sizes": [10, 20], "sector_cap_share": 0.2}
    rng = np.random.default_rng(0)
    n = 300
    drift, controlled, vol = rng.normal(size=n), rng.normal(size=n), rng.random(n)
    sector = rng.integers(1, 13, size=n).astype(object)
    pool = np.ones(n, dtype=bool)
    mild = np.zeros(n, dtype=bool)
    mild[:60] = True
    picks = sc.shortlist_picks(drift, controlled, vol, sector, {"depth_0_10": mild}, pool, cfg)
    assert set(picks) == set(sc.basket_names({**cfg, "rules": dict.fromkeys(
        ["top_drift", "mild_drift", "mild_controlled", "mild_sector", "mild_lowvol"], "")}))
    for name, pick in picks.items():
        size = int(name.rsplit("_", 1)[1])
        assert pick.sum() == size or (name.startswith("mild_sector") and pick.sum() <= size)
        if name.startswith("mild_"):
            assert not (pick & ~mild).any()
    assert picks["top_drift_10"][np.argmax(drift)]
    assert picks["mild_lowvol_10"][np.argmin(np.where(mild, vol, np.inf))]
    counts = np.unique(sector[picks["mild_sector_20"]], return_counts=True)[1]
    assert counts.max() <= 4
