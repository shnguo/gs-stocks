"""Independent checks of the recorded two-day integration run."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, required=True)
    args = parser.parse_args()
    store = args.store
    acceptance = read(store / "acceptance-runs.json")
    hashes = 0
    for section in ["models", "runs", "reviews"]:
        for manifest in (store / section).rglob("manifest.json"):
            for name, expected in read(manifest)["files"].items():
                h = hashlib.sha256((manifest.parent / name).read_bytes()).hexdigest()
                assert h == expected, (manifest, name)
                hashes += 1
    run = Path(acceptance["replay"])
    meta = read(run / "run.json")
    rows = pd.read_parquet(run / "rows.parquet")
    assert rows.instrument_id.is_unique
    package = Path(acceptance["package"])
    pilot = Path(read(package / "model.json")["source_pilot"])
    prior_rows = pd.read_parquet(pilot / "evaluation-rows.parquet")
    prior_rows["output_index"] = np.arange(len(prior_rows))
    overlap = rows.reset_index().merge(
        prior_rows.loc[prior_rows.date == meta["input"]["signal_date"]], on="instrument_id"
    )
    pred = np.load(run / "lightgbm-forecast.npy")
    prior = np.load(pilot / "lightgbm-evaluation.npy")
    np.testing.assert_allclose(
        pred[overlap["index"]], prior[overlap.output_index], rtol=0, atol=1e-12
    )
    checked = 0
    for key in ["partial_review", "mature_review"]:
        directory = Path(acceptance[key])
        review = read(directory / "review.json")
        labels = np.load(directory / "labels.npz")
        actual = pd.read_parquet(
            Path(meta["input"]["source_root"]) / "snapshot/bars.parquet",
            filters=[("date", "in", meta["input"]["horizon_dates"])],
        )
        actual = actual.set_index(["instrument_id", "date"])
        for metric in review["metrics"]:
            d = metric["day_offset"]
            y = labels["targets"][:, d, 3]
            mask = np.isfinite(y)
            assert int(mask.sum()) == metric["known_rows"]
            if metric["date"] > review["observed_through"]:
                assert not mask.any() and metric["pinball"] is None
                continue
            q = np.load(run / f"{metric['model']}-forecast.npy")[:, d, 3]
            if mask.any():
                raw = actual.close.reindex(
                    pd.MultiIndex.from_arrays([rows.instrument_id, [metric["date"]] * len(rows)])
                ).to_numpy()
                expected = np.log(raw / labels["reference"])
                np.testing.assert_allclose(y[mask], expected[mask], rtol=0, atol=1e-12)
                residual = y[mask, None] - q[mask]
                loss = np.where(
                    residual >= 0,
                    residual * np.array([0.1, 0.5, 0.9]),
                    -residual * np.array([0.9, 0.5, 0.1]),
                ).sum() / (3 * mask.sum())
                coverage = (
                    np.count_nonzero((q[mask, 0] <= y[mask]) & (y[mask] <= q[mask, 2])) / mask.sum()
                )
                assert abs(loss - metric["pinball"]) < 1e-12
                assert abs(coverage - metric["coverage80"]) < 1e-12
            checked += 1
        outcomes = pd.read_parquet(directory / "outcomes.parquet")
        assert (
            outcomes.loc[outcomes.state.isin(["pending", "unknown"]), "hypothetical_net_return"]
            .isna()
            .all()
        )
        assert not outcomes.actual_trade_recorded.any()
        assert sum(r["count"] for r in review["states"]) == len(outcomes)
    blocked = Path(acceptance["current_blocked"])
    assert (
        read(blocked / "run.json")["status"] == "blocked"
        and not (blocked / "plans.parquet").exists()
    )
    result = {
        "hashes_verified": hashes,
        "checkpoint_rows_reproduced": len(overlap),
        "close_metrics_independently_recomputed": checked,
        "pending_unknown_preserved": True,
        "actual_trades_created": False,
        "blocked_live_run_has_no_forecast": True,
        "quality_promotion": False,
    }
    (store / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
