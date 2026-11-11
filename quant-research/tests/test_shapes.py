import numpy as np
import pandas as pd

from quant_research.factors import price_panel
from quant_research.shapes import shape_scores

CFG = {"jump_sigma": 3.0, "min_traded_share": 0.8, "prior_sessions": 189, "periods": {"3m": 63}}


def test_shapes_pick_the_matching_paths():
    rng = np.random.default_rng(11)
    dates = list(pd.bdate_range("2020-01-01", periods=320).strftime("%Y-%m-%d"))
    paths = {
        "rise": np.full(320, 0.002),                                  # up all the way
        "turn": np.r_[np.full(257, -0.003), np.full(63, 0.004)],      # down, then up for 3 months
        "fall": np.full(320, -0.002),                                 # down all the way
        "broke": np.r_[np.full(257, 0.004), np.full(63, -0.004)],     # up, then down for 3 months
    }
    rows = []
    for name, drift in paths.items():
        r = drift + rng.normal(0, 0.004, 320)
        price = 10 * np.exp(np.cumsum(r))
        rows += [dict(instrument_id=name, date=d, open=p, high=p, low=p, close=p, volume=1000.0, factor=1.0)
                 for d, p in zip(dates, price)]
    for i in range(16):  # filler stocks so percentiles have a crowd
        price = 10 * np.exp(np.cumsum(rng.normal(0, 0.01, 320)))
        rows += [dict(instrument_id=f"x{i:02d}", date=d, open=p, high=p, low=p, close=p, volume=1000.0, factor=1.0)
                 for d, p in zip(dates, price)]
    panel = price_panel(pd.DataFrame(rows))
    mask = np.ones(len(panel.ids), dtype=bool)
    date = dates[-1]
    drift_12_1 = np.array([{"rise": 0.5, "broke": 0.9}.get(i, -0.1) for i in panel.ids])
    out = shape_scores(panel, date, mask, drift_12_1, CFG)
    idx = {name: list(panel.ids).index(name) for name in paths}
    _, turning = out["turning_up_3m"]
    assert turning[idx["turn"]] and not turning[idx["rise"]] and not turning[idx["fall"]]
    rising_score, _ = out["rising_now_3m"]
    assert rising_score[idx["turn"]] > rising_score[idx["broke"]]
    _, uptrend = out["long_uptrend_3m"]
    assert uptrend[idx["rise"]] and not uptrend[idx["broke"]]
