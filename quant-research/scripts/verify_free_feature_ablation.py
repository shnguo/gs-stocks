"""Independent same-date feature arithmetic, model reload and cost/choice checks."""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from verify_category_ablation import verify_metrics
from verify_industry_decisions import restore, verify_decisions
from verify_plan_asof_fit import check, read
from verify_plan_prequential import calibrator_check, load, verify_comparisons

from quant_research.plan_targets import plan_targets
from quant_research.plan_value import HEADS, head_targets
from quant_research.price_strategy import TradeAssumptions
from quant_research.storage import file_hash, utc_now, write_json


def independent_context(rows, records, bars):
    keys = ["instrument_id", "date"]
    assert not records.duplicated(keys).any() and not bars.duplicated(keys).any()
    joined = (
        rows[keys]
        .assign(_order=np.arange(len(rows)))
        .merge(records, on=keys, how="left", validate="one_to_one")
        .merge(
            bars.rename(columns={"close": "reference", "volume": "volume_shares"}),
            on=keys,
            how="left",
            validate="one_to_one",
        )
        .sort_values("_order")
    )
    available = (
        joined.close.notna()
        & joined.reference.notna()
        & ((joined.close - joined.reference).abs() <= 0.010001)
    )
    for col in [
        "total_cap",
        "float_cap",
        "total_shares",
        "float_shares",
        "pe_ttm",
        "pb",
        "ps_ttm",
        "pcf_ttm",
    ]:
        joined.loc[~available, col] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        columns = [joined.volume_shares / joined.float_shares * 100]
        columns.extend(
            np.log(joined[k].where(joined[k] > 0))
            for k in ["total_cap", "float_cap", "total_shares", "float_shares"]
        )
        columns.append(joined.float_shares / joined.total_shares)
        columns.extend(
            1 / joined[k].where(joined[k].abs() > 1e-8)
            for k in ["pe_ttm", "pb", "ps_ttm", "pcf_ttm"]
        )
        columns.append(available.astype(float))
    return np.column_stack(columns).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    root = parser.parse_args().output.resolve()
    if (root / "verification.json").exists():
        raise FileExistsError("Keep prior verification")
    state, exp, protocol = [
        read(root / p) for p in ["run-status.json", "experiment.json", "protocol.json"]
    ]
    assert state["status"] == "completed"
    counts = dict(
        hashes=check(root, state["files"]),
        feature_values=0,
        restored_values=0,
        calibrators=0,
        chosen_rows=0,
        metric_rows=0,
        cost_dates=0,
    )
    prior = Path(exp["prior"])
    source = Path(exp["source"])
    for parent, key in [(prior, "parent_status_sha256"), (source, "source_status_sha256")]:
        assert file_hash(parent / "run-status.json") == exp[key]
        counts["hashes"] += check(parent, read(parent / "run-status.json")["files"])
    index = read(source / "index.json")
    for fold in protocol["fold_indices"]:
        parent = prior / f"fold-{fold:02d}"
        directory = root / f"fold-{fold:02d}"
        config = read(parent / "config.json")
        snapshot = Path(config["root"]) / "snapshot"
        parts = {}
        for part in ["train", "selection", "calibration", "evaluation"]:
            rows = pd.read_parquet(parent / f"{part}-rows.parquet")
            assert rows.label_end.max() < "2025-08-07"
            assert sorted(rows.date.unique()) == config["dates"][part]
            records = []
            for day in sorted(rows.date.unique()):
                path = source / "daily" / f"{day}.parquet"
                assert file_hash(path) == index[day]["parquet_sha256"]
                records.append(pd.read_parquet(path))
            bars = pd.read_parquet(
                snapshot / "bars.parquet",
                columns=["instrument_id", "date", "close", "volume"],
                filters=[("date", "in", sorted(rows.date.unique()))],
            )
            expected = independent_context(rows, pd.concat(records, ignore_index=True), bars)
            recorded = np.load(directory / f"{part}-context.npy")
            np.testing.assert_allclose(recorded, expected, rtol=0, atol=0, equal_nan=True)
            coverage = read(directory / f"{part}-coverage.json")
            assert coverage["rows"] == len(rows) and coverage["source_available"] == int(
                expected[:, -1].sum()
            )
            assert coverage["source_available"] / len(rows) >= protocol["min_available_fraction"]
            status = pd.read_parquet(directory / f"{part}-source-status.parquet")
            pd.testing.assert_frame_equal(
                status[["instrument_id", "date"]], rows[["instrument_id", "date"]]
            )
            counts["feature_values"] += expected.size
            parts[part] = (rows, recorded)
        rows, context = parts["evaluation"]
        data = load(parent / "evaluation.npz")
        cal = load(parent / "calibration.npz")
        base = plan_targets(data)
        assumptions = TradeAssumptions()
        stress = plan_targets(
            data,
            replace(
                assumptions,
                commission=assumptions.commission * 2,
                minimum_fee=assumptions.minimum_fee * 2,
                sell_tax=assumptions.sell_tax * 2,
                slippage_bps=assumptions.slippage_bps * 2,
            ),
        )
        metadata = read(directory / "models.json")
        raw = load(directory / "daily_information_raw-evaluation.npz")
        corrected = load(directory / "daily_information-evaluation.npz")
        sample = np.unique(np.linspace(0, len(rows) - 1, 32, dtype=int))
        restored = restore(
            directory, np.column_stack([data["x"][sample], context[sample]]), metadata
        )
        for head in HEADS:
            np.testing.assert_allclose(restored[head], raw[head][sample], atol=1e-12, rtol=1e-12)
            counts["restored_values"] += restored[head].size
        cal_pred = restore(
            directory, np.column_stack([cal["x"], parts["calibration"][1]]), metadata
        )
        counts["calibrators"] += calibrator_check(
            cal_pred,
            head_targets(plan_targets(cal)),
            parts["calibration"][0].date.to_numpy(),
            read(directory / "calibration.json"),
            raw,
            corrected,
        )
        forecasts = dict(daily_information_raw=raw, daily_information=corrected)
        counts["metric_rows"] += verify_metrics(
            rows, head_targets(base), forecasts, pd.read_csv(directory / "head-metrics.csv")
        )
        for name, pred in forecasts.items():
            n, days = verify_decisions(
                rows,
                pred,
                base,
                stress,
                pd.read_parquet(directory / f"{name}-chosen.parquet"),
                pd.read_csv(directory / f"{name}-decision-metrics.csv"),
            )
            counts["chosen_rows"] += n
            counts["cost_dates"] += days
        print("Independently verified information fold", fold, flush=True)
    verify_comparisons(pd.read_csv(root / "head-metrics.csv"), read(root / "assessment.json"))
    counts["hashes"] += check(root, state["files"])
    write_json(
        root / "verification.json",
        dict(
            passed=True,
            verified_at=utc_now(),
            counts=counts,
            scope="All new feature values independently recalculated, same original cohorts, checkpoint sample restored, all calibrators refitted, daily metrics/choices/doubled cost scenarios and block comparisons verified. No original publication-time or profitability certification.",
        ),
    )
    print(counts, flush=True)


if __name__ == "__main__":
    main()
