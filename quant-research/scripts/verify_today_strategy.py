"""Independent raw-record, feature-formula and price-level acceptance checks."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read(p):
    return json.loads(p.read_text())


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    info = read(run / "run.json")
    source = Path(info["input"]["source_root"])
    n_hash = 0
    for root in [run, source / "snapshot", source / "panel-h5"]:
        for name, expected in read(root / "manifest.json")["files"].items():
            assert sha(root / name) == expected
            n_hash += 1
    for item in read(source / "lineage.json")["captures"]:
        p = Path(item["path"])
        m = read(p / "manifest.json")
        assert sha(p / "manifest.json") == item["manifest_sha256"]
        assert sha(Path(m["capture"]) / "response.json") == item["raw_sha256"]
        n_hash += 2
    rows = pd.read_parquet(run / "rows.parquet")
    symbols = rows.instrument_id.tolist()
    panel = read(source / "panel-h5/manifest.json")
    s = info["input"]["signal_date"]
    t = panel["dates"].index(s)
    dates = panel["dates"][t - 60 : t + 1]
    bars = pd.read_parquet(source / "snapshot/bars.parquet")
    assert bars.date.max() == s
    fields = ["open", "high", "low", "close", "factor", "volume", "amount"]
    f = bars.set_index(["instrument_id", "date"]).reindex(
        pd.MultiIndex.from_product([symbols, dates])
    )
    a = f[fields].to_numpy(float).reshape(len(rows), 61, 7)
    adjusted = a[:, :, 3] * a[:, :, 4]
    returns = adjusted[:, 1:] / adjusted[:, :-1] - 1
    expected = {}
    for w in [1, 2, 3, 5, 10, 20, 60]:
        expected[f"return_{w}"] = adjusted[:, -1] / adjusted[:, -w - 1] - 1
    for w in [5, 10, 20, 60]:
        expected[f"volatility_{w}"] = np.std(returns[:, -w:], axis=1, ddof=1)
        expected[f"price_mean_{w}"] = adjusted[:, -1] / np.mean(adjusted[:, -w:], axis=1) - 1
    o, h, low, c, factor, v, amount = [a[:, -1, j] for j in range(7)]
    expected.update(
        range=(h - low) / c,
        body=c / o - 1,
        gap=o * factor / adjusted[:, -2] - 1,
        close_position=np.divide(c - low, h - low, out=np.full(len(c), 0.5), where=h != low),
        log_volume=np.log1p(v),
        log_amount=np.log1p(amount),
    )
    for w in [5, 20]:
        expected[f"volume_mean_{w}"] = v / np.mean(a[:, -w:, 5], axis=1) - 1
        expected[f"amount_mean_{w}"] = amount / np.mean(a[:, -w:, 6], axis=1) - 1
    expected.update(suspended=np.zeros(len(rows)), unpriced_suspension=np.zeros(len(rows)))
    x = np.load(run / "input.npz")["features"]
    for j, name in enumerate(panel["feature_names"]):
        np.testing.assert_allclose(x[:, j], expected[name], rtol=2e-6, atol=2e-6, err_msg=name)
    values = np.load(source / "panel-h5/values.npy", mmap_mode="r")
    assert np.isnan(values[:, t + 1 :]).all()
    ledger = pd.read_parquet(source / "refresh-coverage.parquet").set_index("instrument_id")
    assert ledger.loc[symbols, "continuity_passed"].all()
    assert ledger.loc[symbols, "current_member"].all()
    q = np.load(run / "lightgbm-forecast.npy")
    prices = np.exp(q) * c[:, None, None, None]
    plans = pd.read_parquet(run / "plans.parquet")
    orders = plans.loc[plans.model == "lightgbm"].set_index("instrument_id").loc[symbols]
    np.testing.assert_allclose(
        orders.buy, np.floor(np.minimum(c, prices[:, 0, 2, 1]) / 0.01) * 0.01, atol=1e-7
    )
    np.testing.assert_allclose(
        orders.take_profit, np.floor(prices[:, 4, 3, 1] / 0.01) * 0.01, atol=1e-7
    )
    np.testing.assert_allclose(
        orders.stop, np.floor(prices[:, 1:, 2, 0].min(1) / 0.01) * 0.01, atol=1e-7
    )
    assert not plans.executable.any() and not plans.actual_trade_recorded.any()
    result = {
        "input_rows_verified": len(rows),
        "feature_formulas_verified": len(expected),
        "feature_values_verified": int(x.size),
        "price_levels_verified": 3 * len(rows),
        "hashes_verified": n_hash,
        "future_features_all_missing": True,
        "lightgbm_conditions_met": int(orders.signal.eq("conditional_candidate").sum()),
        "trade_date": info["input"]["trade_date"],
        "signal_date": s,
        "publication_mode": info["input"]["mode"],
    }
    target = run.parents[2] / "verification.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
