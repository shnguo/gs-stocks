from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import simulate
from .config import digest
from .evaluation import paired_rank_ic_difference
from .features import Standardizer, build_panel
from .models import (
    predict_transformer,
    rank_ic,
    save_transformer,
    tabular_values,
    train_transformer,
)
from .pool import require_full_pool
from .splits import Fold
from .storage import Snapshot, file_hash, utc_now, write_json
from .tree_runner import isolated_tree
from .universe import audit_snapshot


def run_experiment(snapshot: Snapshot, config: dict, fold: Fold, horizon: int,
                   seed: int, output: Path, engineering: bool = False,
                   equal_information: bool = False, training_only: bool = False,
                   allow_partial_universe: bool = False) -> dict:
    if output.exists():
        raise FileExistsError("Experiment outputs are immutable; use a new run directory")
    if horizon not in config["horizons"] or seed not in config["seeds"]:
        raise ValueError("Horizon or seed is outside the registered configuration")
    if allow_partial_universe and not engineering:
        raise ValueError("Partial universes are allowed only for explicit engineering test fixtures")
    pool = ({"scope": "engineering_test_fixture", "data_ready": False}
            if allow_partial_universe else require_full_pool(snapshot, config))
    risk_policy = config.get("training_risk_policy", "exclude_st")
    audit = audit_snapshot(snapshot, config["signal_time"], config["min_history_years"],
                           training_risk_policy=risk_policy)
    blockers = list(audit["training_data_blockers"] if training_only else audit["formal_blockers"])
    blockers.append("final_holdout_and_multiyear_walkforward_pending")
    if not training_only:
        blockers.append("historical_industry_portfolio_engine_pending")
        if snapshot.manifest["declaration"].get("purpose") == "training":
            raise ValueError("Training-only snapshot requires --training-only")
    if not engineering:
        raise ValueError("Formal experiment is not ready: " + ", ".join(blockers))
    if fold.purpose != "development":
        raise ValueError("This runner cannot inspect the sealed final holdout")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    package = Path(__file__).parent
    sources = {p.name: p.read_text() for p in sorted(package.glob("*.py"))}
    lock = package.parents[1] / "uv.lock"
    software = {"source_sha256": digest(sources), "lock_sha256": file_hash(lock),
                "versions": {name: importlib.metadata.version(name)
                             for name in ["numpy", "pandas", "torch", "lightgbm", "pyarrow"]}}
    identity = {"dataset_id": snapshot.dataset_id, "config": config, "fold": fold.to_dict(),
                "universe_id": pool.get("universe_id"), "universe_scope": pool["scope"],
                "allow_partial_universe": allow_partial_universe,
                "horizon": horizon, "seed": seed, "engineering": engineering,
                "training_only": training_only,
                "equal_information": equal_information, "software": software}
    # Reserve an append-only trial before training, including failed attempts in the budget.
    registry = output.parent / "trial-registry"
    registry.mkdir(exist_ok=True)
    trial_key = digest({"dataset_id": snapshot.dataset_id, "horizon": horizon,
                        "model": config["model"], "equal_information": equal_information,
                        "training_risk_policy": risk_policy})
    registered = list(registry.glob(f"h{horizon}-*.json"))
    trial_path = registry / f"h{horizon}-{trial_key}.json"
    if not trial_path.exists():
        if len(registered) >= config["max_trials_per_horizon"]:
            raise ValueError("Registered hyperparameter trial budget exhausted")
        with trial_path.open("x") as handle:
            json.dump({"trial_key": trial_key, "identity": identity, "reserved_at": utc_now()}, handle)
    status = {"run_id": digest(identity), "created_at": utc_now(), "identity": identity,
              "status": "running", "formal_ready": False, "formal_blockers": blockers}
    write_json(output / "run.json", status)
    write_json(output / "source-code.json", sources)
    try:
        panel = build_panel(snapshot, horizon, config["lookback"], config["signal_time"],
                            risk_policy=risk_policy)
        if (panel.samples.label_status == "missing_execution_endpoint").any():
            blockers.append("missing_label_endpoints_need_censoring_review")
        splits = fold.select(panel.samples)
        panel.samples.to_parquet(output / "samples.parquet", index=False)
        panel.exclusions.to_parquet(output / "exclusions.parquet", index=False)
        write_json(output / "data-audit.json", audit)
        write_json(output / "pool-coverage.json", pool)
        write_json(output / "features.json", {"names": panel.feature_names,
                   "lookback": panel.lookback, "horizon": horizon,
                   "training_risk_policy": risk_policy,
                   "risk_status_usage": "audit_metadata_only_not_a_model_feature",
                   "missing_endpoint_samples": int((panel.samples.label_status ==
                                                     "missing_execution_endpoint").sum())})
        endpoint_audit = (panel.samples.assign(year=panel.samples.date.str[:4])
                          .groupby(["year", "entry_status", "exit_status"], dropna=False)
                          .size().rename("samples").reset_index())
        endpoint_audit.to_parquet(output / "label-coverage.parquet", index=False)
        scaler = Standardizer.fit(panel, splits["train"])
        write_json(output / "scaler.json", {"mean": scaler.mean.tolist(),
                   "scale": scaler.scale.tolist(), "fit_partition": "train_only"})
        test = splits["test"].copy()

        def train_tree(flatten: bool = False) -> np.ndarray:
            return isolated_tree(tabular_values(panel, splits["train"], scaler, flatten),
                                 splits["train"].target.to_numpy(),
                                 tabular_values(panel, splits["validation"], scaler, flatten),
                                 splits["validation"].target.to_numpy(),
                                 tabular_values(panel, test, scaler, flatten), seed,
                                 output / ("lightgbm-sequence.txt" if flatten else "lightgbm.txt"))

        lgbm_scores = train_tree()
        transformer, training_log = train_transformer(
            panel, splits["train"], splits["validation"], scaler, config["model"], seed)
        save_transformer(output / "transformer.pt", transformer, panel, scaler, config["model"], seed)
        write_json(output / "training-log.json", training_log)
        scores = {
            "momentum": panel.values[test.stock_index.to_numpy(dtype=int),
                                     test.date_index.to_numpy(dtype=int),
                                     panel.feature_names.index("return_20")],
            "lightgbm": lgbm_scores,
            "transformer": predict_transformer(transformer, panel, test, scaler,
                                                config["model"]["batch_size"]),
        }
        if equal_information:
            scores["lightgbm_sequence"] = train_tree(True)
        scored = test[["date"]].copy()
        for name in ["lightgbm", "transformer"]:
            scored[name] = scores[name]
        scores["ensemble"] = (scored.groupby("date").lightgbm.rank(pct=True).to_numpy() +
                              scored.groupby("date").transformer.rank(pct=True).to_numpy()) / 2
        metrics, trading_metrics, portfolios = {}, {}, {}
        trading_mask = test.trading_eligible.to_numpy(dtype=bool)
        for name, values in scores.items():
            predictions = test.copy()
            predictions["score"] = np.asarray(values)
            predictions.to_parquet(output / f"predictions-{name}.parquet", index=False)
            metrics[name] = rank_ic(test, values)
            trading_metrics[name] = rank_ic(test.loc[trading_mask], np.asarray(values)[trading_mask])
            if not training_only:
                result = simulate(snapshot, predictions, horizon, config["portfolio"])
                portfolios[name] = result["summary"]
                for table, frame in result.items():
                    if isinstance(frame, pd.DataFrame):
                        frame.to_parquet(output / f"{name}-{table}.parquet", index=False)
        stress = {}
        for name in ([] if training_only else ["lightgbm", "transformer", "ensemble"]):
            predictions = test.copy()
            predictions["score"] = scores[name]
            stress[name] = {}
            for variant, multiplier, delay in [("double_cost", 2.0, 0), ("one_day_delay", 1.0, 1)]:
                stress[name][variant] = simulate(snapshot, predictions, horizon, config["portfolio"],
                                                 multiplier, delay)["summary"]
        summary = {"mode": "engineering_only", "synthetic": snapshot.manifest["declaration"]["synthetic"],
                   "universe_scope": pool["scope"], "universe_id": pool.get("universe_id"),
                   "training_only": training_only,
                   "dataset_coverage": {"instruments": audit["instruments"],
                       "bar_rows": audit["bar_rows"], "first_date": audit["first_date"],
                       "last_date": audit["last_date"],
                       "exchanges": sorted(snapshot.tables["instruments"].exchange.unique()),
                       "all_ashare_coverage_verified": snapshot.manifest["declaration"].get(
                           "all_ashare_coverage_verified", False)},
                   "horizon": horizon, "seed": seed, "formal_ready": False,
                   "formal_blockers": blockers, "partition_rows": {k: len(v) for k, v in splits.items()},
                   "training_risk_policy": risk_policy, "trading_risk_policy": "exclude_st",
                   "training_data_blockers": audit["training_data_blockers"],
                   "backtest_data_blockers": audit["backtest_data_blockers"],
                   "partition_risk_status_counts": {k: {
                       status: int(count) for status, count in rows.risk_status.value_counts().items()
                   } for k, rows in splits.items()},
                   "partition_source_st_counts": {k: {
                       "flagged_st": int((rows.source_is_st == 1).sum()),
                       "flagged_normal": int((rows.source_is_st == 0).sum()),
                       "source_flag_missing": int(rows.source_is_st.isna().sum())
                   } for k, rows in splits.items()},
                   "test_label_coverage": {
                       "scored_samples": len(test),
                       "available_labels": int((test.label_status == "available").sum()),
                       "unavailable_labels": int((test.label_status != "available").sum())},
                   "trading_universe_metrics": trading_metrics,
                   "metric_scope": "training_universe_at_signal",
                   "metrics": metrics, "portfolios": portfolios, "stress": stress,
                   "paired_transformer_minus_lightgbm_ic": paired_rank_ic_difference(
                       metrics["transformer"], metrics["lightgbm"]),
                   "architecture_attribution_ready": False,
                   "checkpoint_selection": "validation_rank_ic_only",
                   "ensemble_weights": {"lightgbm": 0.5, "transformer": 0.5}}
        write_json(output / "summary.json", summary)
        status.update(status="completed", completed_at=utc_now(), files={
            path.name: file_hash(path) for path in sorted(output.iterdir()) if path.name != "run.json"})
        write_json(output / "run.json", status)
        return summary
    except Exception as error:
        status.update(status="failed", failed_at=utc_now(), error_type=type(error).__name__,
                      error=str(error)[:500])
        write_json(output / "run.json", status)
        raise
