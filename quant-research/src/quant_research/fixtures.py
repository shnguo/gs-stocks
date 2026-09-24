"""Deterministic artificial market, for engineering verification only."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .storage import TABLES, freeze


def create_fixture(source: Path, destination: Path, stocks: int = 18, days: int = 400):
    if source.exists() or destination.exists():
        raise FileExistsError("Fixture destinations must be new")
    source.mkdir(parents=True)
    dates = pd.bdate_range("2023-01-02", periods=days).strftime("%Y-%m-%d").tolist()
    rng = np.random.default_rng(17029)
    instruments, bars, events, actions = [], [], [], []
    for stock in range(stocks):
        exchange = ["XSHG", "XSHE", "XBSE"][stock % 3]
        instrument_id = f"synthetic.{exchange.lower()}.{stock:06d}"
        instruments.append(dict(instrument_id=instrument_id, exchange=exchange,
                                board=exchange, listed_at="2020-01-01", delisted_at="",
                                industry=f"synthetic_industry_{stock % 4}"))
        t = np.arange(days)
        returns = (0.0001 + 0.001 * np.sin(t / 18 + stock) + rng.normal(0, 0.008, days))
        close = (15 + stock) * np.exp(np.cumsum(returns))
        opening = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 0.002, days))
        factor = np.ones(days)
        if stock == 2 and days > 335:
            factor[335:] = 2
            close[335:] /= 2
            opening[335:] /= 2
            actions.append(dict(instrument_id=instrument_id, ex_date=dates[335],
                                pay_date=dates[335], split_ratio=2.0, cash_per_share=0.0))
        for i, date in enumerate(dates):
            previous = close[max(0, i - 1)]
            if stock == 2 and i == 335:
                previous /= 2
            volume = float(rng.integers(800_000, 1_400_000))
            bars.append(dict(instrument_id=instrument_id, date=date, open=opening[i],
                             close=close[i], high=max(close[i], opening[i]) * 1.005,
                             low=min(close[i], opening[i]) * 0.995, volume=volume,
                             amount=volume * close[i], factor=factor[i],
                             upper_limit=previous * 1.1, lower_limit=previous * 0.9))
        if stock == 0 and days > 350:
            events.append(dict(instrument_id=instrument_id, start_date=dates[330],
                               end_date=dates[349], status="*ST",
                               known_at=dates[328] + "T10:00:00Z"))
    tables = {
        "instruments": instruments, "bars": bars, "risk_events": events, "actions": actions,
        "calendar": [dict(date=d, exchange=e, is_open=True) for d in dates
                     for e in ["XSHG", "XSHE", "XBSE"]],
        "risk_coverage": [dict(date=d, exchange=e, complete=True, known_at=d + "T01:00:00Z")
                          for d in dates for e in ["XSHG", "XSHE", "XBSE"]],
        "rules": [dict(board=e, start_date=dates[0], end_date="", min_buy=100, buy_step=100,
                       sell_step=1, t_plus=1, buy_fee=0.0003, sell_fee=0.0003,
                       stamp_duty=0.0005, minimum_fee=5.0, slippage_bps=5.0)
                  for e in ["XSHG", "XSHE", "XBSE"]],
    }
    for name, rows in tables.items():
        pd.DataFrame(rows, columns=sorted(TABLES[name])).to_parquet(source / f"{name}.parquet",
                                                                  index=False)
    declaration = dict(vintage="synthetic", source="deterministic_engineering_fixture",
                       synthetic=True, st_history_verified=False, actions_verified=False,
                       rules_verified=False, historical_universe_verified=False,
                       industry_history_verified=False)
    return freeze(source, destination, declaration)

