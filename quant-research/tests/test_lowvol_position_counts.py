import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_lowvol_position_counts import simulate_equal_target  # noqa: E402


def sample(count):
    dates = pd.bdate_range("2022-01-03", periods=43).strftime("%Y-%m-%d").tolist()
    names = [f"cn.xshe.{i:06d}" for i in range(1, 51)]
    shape = (len(dates), len(names))
    arrays = {k: np.full(shape, 10.0) for k in ["open", "high", "low", "close"]}
    arrays.update(
        volume=np.full(shape, 1e8),
        amount=np.full(shape, 1e9),
        source_is_st=np.zeros(shape),
        source_trade_status=np.ones(shape),
        label_sequence_id=np.zeros(shape, dtype=int),
    )
    inst = pd.DataFrame({"instrument_id": names, "delisted_at": [""] * len(names)})
    cfg = dict(
        initial_cash=200000.0,
        evaluation_start=dates[1],
        rebalance_sessions=20,
        retention_buffer_rank=2 * count,
        max_positions=count,
        position_budget_fraction=0.9 / count,
        adv_participation=0.005,
    )
    sc = dict(
        name="base", rebalance_offset=0, delay_sessions=0, slippage_bps=10, commission_multiplier=1
    )
    score = np.tile(np.arange(50, 0, -1), (len(dates), 1)).astype(float)
    return dates, names, inst, arrays, cfg, sc, score


@pytest.mark.parametrize("count", [6, 10, 20])
def test_same_initial_market_exposure_and_real_minimum_commission(tmp_path, count):
    dates, names, inst, arrays, cfg, sc, score = sample(count)
    _, nav = simulate_equal_target(
        dates,
        names,
        inst,
        arrays,
        {},
        score,
        arrays["amount"],
        cfg,
        sc,
        "test",
        tmp_path / "account",
    )
    fills = pd.read_parquet(tmp_path / "account/fills.parquet")
    first = fills[fills.date.eq(dates[1])]
    assert len(first) == count
    assert nav.market_value.iloc[0] == pytest.approx(180000)
    assert first.commission.sum() == pytest.approx(100 if count == 20 else 90.09)
    assert nav.cash.min() >= 0
    assert (first.shares % 100 == 0).all()


def test_retained_winner_is_reduced_and_others_topped_up(tmp_path):
    dates, names, inst, arrays, cfg, sc, score = sample(10)
    for field in ["open", "high", "low", "close"]:
        arrays[field][15:, 0] = 20.0
    _, nav = simulate_equal_target(
        dates,
        names,
        inst,
        arrays,
        {},
        score,
        arrays["amount"],
        cfg,
        sc,
        "test",
        tmp_path / "account",
    )
    fills = pd.read_parquet(tmp_path / "account/fills.parquet")
    resize = fills[fills.date.eq(dates[21])]
    assert ((resize.side == "sell") & (resize.instrument_id == names[0])).any()
    assert (resize.side == "buy").any()
    held = pd.read_parquet(tmp_path / "account/holdings.parquet")
    today = held[held.date.eq(dates[21])]
    target = 0.09 * nav.loc[nav.date.eq(dates[20]), "equity"].iloc[0]
    assert all(abs(row.value - target) <= row.mark * 100 + 1e-7 for row in today.itertuples())


def test_blocked_sale_is_retained_and_cannot_fund_new_positions(tmp_path):
    dates, names, inst, arrays, cfg, sc, score = sample(20)
    score[20:, :20] = -np.inf
    arrays["source_trade_status"][21:, 0] = 0
    result, nav = simulate_equal_target(
        dates,
        names,
        inst,
        arrays,
        {},
        score,
        arrays["amount"],
        cfg,
        sc,
        "test",
        tmp_path / "account",
    )
    held = pd.read_parquet(tmp_path / "account/holdings.parquet")
    assert names[0] in held.loc[held.date.eq(dates[-1]), "instrument_id"].tolist()
    assert nav.cash.min() >= 0 and nav.positions.max() <= 20
    assert result["blocked_reasons"]["no_verified_trading_bar"] > 0


def test_split_dividend_and_delayed_signal_keep_original_semantics(tmp_path):
    dates, names, inst, arrays, cfg, sc, score = sample(6)
    sc["delay_sessions"] = 1
    sc["dividend_delay_days"] = 10000
    for field in ["open", "high", "low", "close"]:
        arrays[field][3:, 0] = 4.95
    _, nav = simulate_equal_target(
        dates,
        names,
        inst,
        arrays,
        {3: [(0, 2.0, 0.1)]},
        score,
        arrays["amount"],
        cfg,
        sc,
        "test",
        tmp_path / "account",
    )
    fills = pd.read_parquet(tmp_path / "account/fills.parquet")
    assert fills.signal_date.iloc[0] == dates[0]
    assert fills.date.iloc[0] == dates[2]
    q = fills.loc[fills.instrument_id.eq(names[0]), "shares"].iloc[0]
    before = nav.loc[nav.date.eq(dates[2]), "equity"].iloc[0]
    after = nav.loc[nav.date.eq(dates[3]), "equity"].iloc[0]
    assert after == pytest.approx(before - q * 0.02)
    assert nav.loc[nav.date.eq(dates[3]), "receivable"].iloc[0] == pytest.approx(q * 0.08)
