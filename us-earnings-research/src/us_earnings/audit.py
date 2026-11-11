"""Independent reconciliation of saved trades against cached source prices."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def audit_run(root: Path, config_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    out = root / "artifacts" / config.get("artifact_subdir", "")
    source = pd.read_csv(root / "data/prices.csv", parse_dates=["date"])
    frames = {symbol: group.set_index("date").sort_index() for symbol, group in source.groupby("ticker")}
    days = frames[config["benchmark"]].index
    cost = config["cost_bps_per_side"] / 10000
    trade_count = 0
    residuals = {}
    for arm in ["price", "earnings", "guidance", "ml"]:
        trades = pd.read_csv(out / f"trades-{arm}.csv", parse_dates=["signal_date", "entry_date", "exit_date"])
        pnl = []
        for trade in trades.itertuples():
            frame = frames[trade.ticker]
            entry = frame.at[trade.entry_date, "open"] * (1 + cost)
            assert abs(entry - trade.entry_price) < 1e-9
            assert trade.entry_date > trade.signal_date
            window = frame.loc[:trade.signal_date].tail(config["atr_sessions"] + 1)
            previous_close = window.close.shift().to_numpy()
            ranges = np.maximum.reduce([(window.high - window.low).to_numpy(),
                                         np.abs(window.high.to_numpy() - previous_close),
                                         np.abs(window.low.to_numpy() - previous_close)])
            atr = float(ranges[-config["atr_sessions"]:].mean())
            stop = frame.at[trade.entry_date, "open"] - config["stop_atr_multiple"] * atr
            bars = frame.loc[trade.entry_date:trade.exit_date]
            if trade.exit_reason == "stop":
                assert bars.low.iloc[-1] <= stop + 1e-9
                assert (bars.low.iloc[:-1] > stop - 1e-9).all()
                exit_price = min(bars.open.iloc[-1], stop) * (1 - cost)
            else:
                assert (bars.low > stop - 1e-9).all()
                exit_price = bars.close.iloc[-1] * (1 - cost)
            assert abs(exit_price - trade.exit_price) < 1e-8
            assert abs((exit_price / entry - 1) - trade.net_return) < 1e-10
            assert abs(trade.shares * entry - trade.invested) < 1e-10
            assert days.get_loc(trade.exit_date) - days.get_loc(trade.entry_date) + 1 == trade.holding_sessions
            pnl.append(trade.shares * (exit_price - entry))
        curve = pd.read_csv(out / f"equity-{arm}.csv")
        residual = float(curve.nav.iloc[-1] - (1 + sum(pnl)))
        assert abs(residual) < 1e-9
        assert curve.cash.min() >= -1e-9
        assert curve.positions.max() <= config["max_positions"]
        residuals[arm] = residual
        trade_count += len(trades)
    events = pd.read_csv(out / "events.csv", keep_default_na=False)
    verified = 0
    for event in events.loc[events.status == "ok"].itertuples():
        key = hashlib.sha256(event.source_url.encode()).hexdigest()
        body = (root / "data/raw" / (key + ".bin")).read_bytes()
        assert hashlib.sha256(body).hexdigest() == event.source_sha256
        assert pd.Timestamp(event.signal_date) > pd.Timestamp(event.filingDate)
        verified += 1
    model = json.loads((out / "text-model-metadata.json").read_text())
    assert model["training_last_date"] <= model["settings"]["training_end"]
    predictions = pd.read_csv(out / "text-model-predictions.csv")
    assert predictions.filingDate.min() >= model["settings"]["evaluation_start"]
    result = {"passed": True, "trade_returns_recomputed_from_source_prices": trade_count,
              "source_exhibit_byte_hashes_verified": verified, "nav_residuals": residuals,
              "model_temporal_boundary_checked": True}
    (out / "independent-audit.json").write_text(json.dumps(result, indent=2))
    return result
