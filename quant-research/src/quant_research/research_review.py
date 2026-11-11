"""Compact decision evidence, conservative execution reviews and a public research summary."""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import simulate
from .daily_loop import digest, read, verify, write_manifest
from .execution_snapshot import ExecutionEvidence, build_execution_snapshot
from .publication import publication_status
from .storage import TABLES, Snapshot, file_hash, read_table, utc_now, write_json

ARM_LABELS = {
    "control": "收益主榜", "start_stage": "上涨初期榜",
    "event_overlay": "事件窗口榜", "confirmed_event": "事件确认榜",
}
EXPLANATIONS = {
    "training_snapshot_has_no_execution_contract": "当前数据仅用于训练，尚无成交模拟契约",
    "historical_information_vintage_unverified": "历史信息可见时间尚未验证",
    "st_history_verified_missing": "缺少经过验证的历史风险状态",
    "actions_verified_missing": "缺少经过验证的公司行动资料",
    "rules_verified_missing": "缺少经过验证的成交规则",
    "historical_universe_verified_missing": "历史股票池尚未验证",
    "industry_history_verified_missing": "历史行业资料尚未验证",
    "price_limits_unverified": "预测窗口缺少经过验证的涨跌停价格",
    "outcomes_pending": "五个预测交易日尚未全部成熟",
    "synthetic_data_is_engineering_validation_only": "合成数据仅用于工程验证",
    "risk_status_unknown_in_expected_universe": "部分候选的风险状态未知",
    "future_price_coverage_incomplete": "部分候选缺少完整预测窗口行情",
    "late_publication_excluded": "迟发报告不进入前瞻成交模拟",
    "execution_data_contract_incomplete": "正式成交数据尚有缺口，请查看逐股票检查",
}


def records(frame):
    # pandas JSON already encodes missing and non-finite numbers as null.
    return json.loads(frame.to_json(orient="records"))


class TrainingOutcomeSnapshot(Snapshot):
    """The incremental feed's hashed training format cannot authorize simulation."""

    def __init__(self, path):
        self.path = Path(path)
        self.manifest = verify(self.path)
        if (self.manifest["declaration"].get("purpose") != "training" or
                set(self.manifest["files"]) != {name + ".parquet" for name in TABLES}):
            raise ValueError("Unsupported incremental training snapshot contract")
        self.tables = {name: read_table(self.path, name) for name in TABLES}
        # This feed deliberately preserves NaN at unknown corporate-action/identity
        # boundaries. Reject corrupt known values, retain unknowns, and block execution.
        bars = self.tables["bars"]
        if bars.duplicated(["instrument_id", "date"]).any():
            raise ValueError("Duplicate incremental daily prices")
        if not set(bars.instrument_id) <= set(self.tables["instruments"].instrument_id):
            raise ValueError("Unknown incremental instrument")
        for key in ("open", "high", "low", "close", "factor", "volume", "amount"):
            bars[key] = pd.to_numeric(bars[key], errors="raise")
            known = bars[key].dropna()
            if (not np.isfinite(known).all() or
                    (known.lt(0) if key in {"volume", "amount"} else known.le(0)).any()):
                raise ValueError("Invalid known incremental price or volume")
        known = bars.dropna(subset=["open", "high", "low", "close"])
        if (known.high.lt(known[["open", "low", "close"]].max(axis=1)).any() or
                known.low.gt(known[["open", "high", "close"]].min(axis=1)).any()):
            raise ValueError("Invalid incremental OHLC ordering")


def outcome_snapshot(path):
    return (Snapshot(path) if "schema_version" in read(Path(path) / "manifest.json")
            else TrainingOutcomeSnapshot(path))


def match_candidates(cohort, config):
    """Freeze outcome-blind, same-board controls using the original model rank."""
    baseline = cohort[cohort.arm.eq("control")].dropna(
        subset=["predicted_rank", "volatility_rank", "liquidity_rank"])
    pairs = []
    limits = config["matching"]["calipers"]
    for arm, treated in cohort[~cohort.arm.eq("control")].groupby("arm"):
        treated = treated.sort_values(["rank", "instrument_id"]).head(config["top_n"])
        excluded = set(cohort[cohort.arm.eq(arm)].instrument_id)
        controls = baseline[~baseline.instrument_id.isin(excluded)]
        available = set(controls.index)
        for row in treated.itertuples():
            choices = controls.loc[sorted(available)]
            choices = choices[choices.board.eq(row.board)].copy()
            if choices.empty:
                continue
            distance = pd.Series(0.0, index=choices.index)
            valid = pd.Series(True, index=choices.index)
            for field, caliper in limits.items():
                gap = (choices[field] - getattr(row, field)).abs()
                valid &= gap.le(caliper)
                distance += (gap / caliper)**2
            choices["distance"] = distance
            choices = choices[valid].sort_values(["distance", "instrument_id"])
            if choices.empty:
                continue
            chosen = choices.iloc[0]
            available.remove(chosen.name)
            pairs.append({"arm": arm, "treated": row.instrument_id,
                          "control": chosen.instrument_id, "distance": float(chosen.distance),
                          **{field + "_gap": float(getattr(row, field) - chosen[field])
                             for field in limits}})
    return pd.DataFrame(pairs, columns=["arm", "treated", "control", "distance",
                                       *[f + "_gap" for f in limits]])


def frozen_features(source, signal, symbols):
    """Only the run's original input vintage and completed signal-close bars are read."""
    verify(source / "snapshot")
    bars = pd.read_parquet(source / "snapshot/bars.parquet")
    master = pd.read_parquet(source / "snapshot/instruments.parquet")
    bars = bars[bars.instrument_id.isin(symbols) & bars.date.le(signal)].sort_values("date")
    history = bars.groupby("instrument_id", group_keys=False).tail(60)
    rows = []
    for symbol, group in history.groupby("instrument_id"):
        adjusted = group.close.tail(21).to_numpy(float) * group.factor.tail(21).to_numpy(float)
        rows.append({
            "instrument_id": symbol,
            "volatility_20": float(np.std(np.diff(np.log(adjusted)), ddof=1))
            if len(group) >= 21 else None,
            "liquidity_20": float(group.amount.tail(20).mean()) if len(group) >= 20 else None,
        })
    features = pd.DataFrame(rows, columns=["instrument_id", "volatility_20", "liquidity_20"])
    features["volatility_rank"] = features.volatility_20.rank(pct=True)
    features["liquidity_rank"] = features.liquidity_20.rank(pct=True)
    columns = [x for x in ("instrument_id", "board", "industry", "exchange") if x in master]
    return features.merge(master[columns], on="instrument_id", validate="one_to_one"), history


def load_publications(run, overlays=()):
    store = run.parent.parent
    info = read(run / "run.json")
    run_hash = file_hash(run / "manifest.json")
    items = []
    for arm, directory, rank_column in [
        ("control", "reference-publications", "rank"),
        ("start_stage", "start-stage-publications", "start_stage_rank"),
    ]:
        pointer = store / directory / (run.name + ".json")
        if not pointer.exists():
            continue
        ref = read(pointer)
        path = Path(ref["report"])
        path = path.parent if path.suffix == ".md" else path
        verify(path)
        delivery = read(path / "delivery.json")
        if file_hash(path / "manifest.json") != ref["manifest_sha256"]:
            raise ValueError("Publication pointer hash changed")
        if delivery["run_manifest_sha256"] != run_hash:
            raise ValueError("Publication belongs to another model run")
        frame = pd.read_csv(path / "ranking.csv")
        frame["rank"] = frame[rank_column]
        items.append((arm, frame, delivery.get("publication_at", delivery.get("at")),
                      {"manifest_sha256": ref["manifest_sha256"], "delivery": delivery}))
    for path in map(Path, overlays):
        verify(path)
        meta = read(path / "metadata.json")
        if meta["signal_date"] != info["signal_date"]:
            continue
        # The bundle must preserve the model control cohort, not merely share a date.
        control = pd.read_csv(path / "control-ranking.csv")
        original = pd.read_csv(run / "ranking.csv")
        fields = ["instrument_id", "expected_net_return"]
        a, b = [f[fields].sort_values("instrument_id").reset_index(drop=True)
                for f in (control, original)]
        if (not a.instrument_id.equals(b.instrument_id) or
                not np.allclose(a.expected_net_return, b.expected_net_return, rtol=1e-12)):
            raise ValueError("Event overlay control does not match the frozen model ranking")
        for arm, name, column in [
            ("event_overlay", "event-overlay-ranking.csv", "event_overlay_rank"),
            ("confirmed_event", "confirmed-event-ranking.csv", "confirmed_event_rank"),
        ]:
            frame = pd.read_csv(path / name)
            frame["rank"] = frame[column]
            items.append((arm, frame, meta["decision_at"], {
                "manifest_sha256": file_hash(path / "manifest.json"), "delivery": meta,
            }))
    return items


def freeze_decision(run, config, overlays=()):
    run = Path(run)
    verify(run)
    info, binding = read(run / "run.json"), read(run / "binding.json")
    items = load_publications(run, overlays)
    if not items:
        raise ValueError("No published reference ranking exists")
    pointer = run.parent.parent / "decision-publications" / (run.name + ".json")
    if pointer.exists():
        ref = read(pointer)
        previous = Path(ref["bundle"])
        verify(previous)
        if file_hash(previous / "manifest.json") != ref["manifest_sha256"]:
            raise ValueError("Frozen decision pointer hash changed")
        original = read(previous / "decision.json")
        # Deploying review code must not recreate an already prospective matched cohort.
        if (original["protocol"] == config and
                original["run_manifest_sha256"] == file_hash(run / "manifest.json") and
                original["publications"] == {arm: lineage["manifest_sha256"] for arm, _, _, lineage in items}):
            return previous
    identity = {
        "version": config["version"], "protocol": config,
        "code_hashes": {name: file_hash(Path(__file__).with_name(name))
                        for name in ("research_review.py", "backtest.py", "publication.py",
                                     "universe.py", "storage.py", "daily_loop.py")},
        "run_manifest_sha256": file_hash(run / "manifest.json"),
        "publications": {arm: lineage["manifest_sha256"] for arm, _, _, lineage in items},
    }
    dest = run.parent.parent / "decisions" / run.name / digest(identity)[:16]
    if dest.exists():
        verify(dest)
        return dest
    source = Path(binding["source"])
    symbols = set().union(*(set(frame.instrument_id) for _, frame, _, _ in items))
    features, history = frozen_features(source, info["signal_date"], symbols)
    primary = next(frame for arm, frame, _, _ in items if arm == "control")
    primary_rank = primary.set_index("instrument_id")["rank"].rank(pct=True)
    stages = run.parent.parent / "reports" / (run.name + "-start-stage-v1/all-candidates.parquet")
    stage = pd.read_parquet(stages) if stages.exists() else None
    frozen, arms = [], []
    for arm, frame, published, lineage in items:
        status = publication_status(info["signal_date"], info["horizon_dates"], published)
        status["prospective"] = status["prospective"] and bool(info["prospective"])
        frame = frame.drop(columns=[c for c in features if c != "instrument_id" and c in frame])
        frame = frame.merge(features, on="instrument_id", how="left", validate="one_to_one")
        frame["predicted_rank"] = frame.instrument_id.map(primary_rank)
        frame["volatility_band"] = pd.cut(frame.volatility_rank, [0, .2, .4, .6, .8, 1],
                                        labels=["q1", "q2", "q3", "q4", "q5"]).astype("string")
        if stage is not None:
            cols = [c for c in ("instrument_id", "early_stage_score", "right_side", "return_20",
                               "ma20_distance", "extension_flags") if c in stage and c not in frame]
            if "instrument_id" not in cols:
                cols.insert(0, "instrument_id")
            frame = frame.merge(stage[cols], on="instrument_id", how="left", validate="one_to_one")
        frame["arm"] = arm
        frame["signal_date"] = info["signal_date"]
        frame["planned_exit_date"] = info["horizon_dates"][-1]
        frame["risk_gate"] = ("历史风险状态已验证" if read(source / "snapshot/manifest.json")["declaration"]["st_history_verified"]
                              else "历史风险状态待验证")
        frozen.append(frame)
        arms.append({"arm": arm, "label": ARM_LABELS[arm], "count": len(frame),
                     "publication": status, "lineage": lineage})
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".decision-", dir=dest.parent))
    try:
        recorded_at = utc_now()
        cohort = pd.concat([frame for frame in frozen if len(frame)], ignore_index=True)
        cohort.to_csv(staging / "cohort.csv", index=False)
        match_candidates(cohort, config).to_csv(staging / "matching-pairs.csv", index=False)
        history.to_parquet(staging / "history.parquet", index=False)
        control_report = Path(read(run.parent.parent / "reference-publications" / (run.name + ".json"))["report"])
        if (control_report / "daily-ohlcva.csv").exists():
            shutil.copyfile(control_report / "daily-ohlcva.csv", staging / "daily-predictions.csv")
        # Small canonical tables are retained with the contemporaneous source declaration.
        for name in ("instruments", "rules", "risk_events", "risk_coverage", "actions"):
            shutil.copyfile(source / f"snapshot/{name}.parquet", staging / f"{name}.parquet")
        write_json(staging / "decision.json", {
            **identity, "signal_date": info["signal_date"], "horizon_dates": info["horizon_dates"],
            "recorded_at": recorded_at, "arms": arms,
            "matching_prospective": bool(pd.Timestamp(recorded_at) <
                pd.Timestamp(info["horizon_dates"][0] + "T09:15:00", tz="Asia/Shanghai")),
            "source_manifest": read(source / "snapshot/manifest.json"),
            "model_config": binding["config"],
            "interpretation": "Frozen publication evidence; late captures do not become prospective.",
        })
        write_manifest(staging)
        staging.rename(dest)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return dest


def scenario_outcomes(cohort, decision, snapshot):
    """Reference-price diagnostics retain the frozen exit date and unknown outcomes."""
    rows = []
    bars = snapshot.tables["bars"].set_index(["instrument_id", "date"])
    groups = {symbol: group for symbol, group in snapshot.tables["bars"].groupby("instrument_id")}
    entry_date = decision["horizon_dates"][0]
    for row in cohort.itertuples():
        value = None
        fixed, adverse = None, None
        exit_date = getattr(row, "sell_reference_date", "")
        if isinstance(exit_date, str) and (row.instrument_id, exit_date) in bars.index:
            if (row.instrument_id, entry_date) in bars.index:
                entry, exit_bar = bars.loc[(row.instrument_id, entry_date)], bars.loc[(row.instrument_id, exit_date)]
                block = groups[row.instrument_id]
                block = block[block.date.between(entry_date, exit_date)]
                expected = [d for d in decision["horizon_dates"] if d <= exit_date]
                unchanged = (len(block) == len(expected) and np.allclose(block.factor, entry.factor) and
                             np.isfinite(block[["open", "high", "low", "close", "factor"]].to_numpy()).all())
                for key in ("sequence_id", "label_sequence_id"):
                    if key in block:
                        unchanged = unchanged and block[key].notna().all() and block[key].nunique() == 1
                if "source_trade_status" in block:
                    unchanged = unchanged and block.source_trade_status.eq(1).all()
                if unchanged and block.volume.gt(0).all():
                    method = decision["model_config"].get("ranking_method", "open_to_close")
                    buy, sell = ("low", "high") if method == "t_low_to_future_high" else ("open", "close")
                    value = float(exit_bar[sell] / entry[buy] - 1 -
                                  decision["model_config"]["round_trip_cost_scenario"])
        fixed_exit = decision["horizon_dates"][-1]
        if (row.instrument_id, entry_date) in bars.index and (row.instrument_id, fixed_exit) in bars.index:
            block = groups[row.instrument_id]
            block = block[block.date.isin(decision["horizon_dates"])]
            unchanged_identity = all(block[key].notna().all() and block[key].nunique() == 1
                                     for key in ("sequence_id", "label_sequence_id") if key in block)
            if (len(block) == 5 and unchanged_identity and block.volume.gt(0).all() and
                    np.isfinite(block[["open", "high", "low", "close", "factor"]].to_numpy()).all() and
                    np.allclose(block.factor, block.factor.iloc[0]) and
                    ("source_trade_status" not in block or block.source_trade_status.eq(1).all())):
                entry = bars.loc[(row.instrument_id, entry_date)]
                fixed = float(bars.loc[(row.instrument_id, fixed_exit)].close / entry.open - 1 -
                              decision["model_config"]["round_trip_cost_scenario"])
                adverse = float(block.low.min() / entry.open - 1)
        rows.append({"arm": row.arm, "instrument_id": row.instrument_id,
                     "realized_net_scenario": value, "fixed_open_close_scenario": fixed,
                     "adverse_scenario": adverse})
    values = pd.DataFrame(rows)
    for field in ("realized_net_scenario", "fixed_open_close_scenario", "adverse_scenario"):
        values[field] = pd.to_numeric(values[field], errors="coerce")
    return cohort.merge(values, on=["arm", "instrument_id"], validate="one_to_one")


def diagnostics(frame):
    known = frame.realized_net_scenario.notna()
    selected = frame[known]
    result = {"count": len(frame), "known": int(known.sum()), "pending": int((~known).sum())}
    if len(selected):
        error = selected.expected_net_return - selected.realized_net_scenario
        result.update(mean_net_scenario=float(selected.realized_net_scenario.mean()),
                      mean_overestimate=float(error.mean()), mean_absolute_error=float(error.abs().mean()),
                      positive_fraction=float(selected.realized_net_scenario.gt(0).mean()))
        if "positive_fraction" in selected:
            probability = pd.to_numeric(selected.positive_fraction, errors="coerce")
            valid = probability.between(0, 1)
            if valid.any():
                result["direction_brier"] = float((probability[valid] -
                    selected.loc[valid, "realized_net_scenario"].gt(0).astype(float)).pow(2).mean())
        if (len(selected) > 2 and selected.expected_net_return.nunique() > 1 and
                selected.realized_net_scenario.nunique() > 1):
            result["rank_ic"] = float(selected.expected_net_return.rank().corr(
                selected.realized_net_scenario.rank()))
    return result


def review_decision(bundle, source, config, snapshot=None, execution_report=None):
    bundle, source = Path(bundle), Path(source)
    verify(bundle)
    decision = read(bundle / "decision.json")
    snapshot = snapshot or outcome_snapshot(source / "snapshot")
    observed = snapshot.tables["bars"].date.max()
    cohort = pd.read_csv(bundle / "cohort.csv")
    identity = {"decision_sha256": file_hash(bundle / "manifest.json"),
                "outcome_dataset_id": snapshot.dataset_id,
                "outcome_manifest_sha256": file_hash(source / "snapshot/manifest.json"),
                "review_code_sha256": file_hash(Path(__file__)),
                "protocol_sha256": digest(config)}
    if execution_report:
        identity["execution_report_sha256"] = file_hash(Path(execution_report) / "gap-report.json")
    dest = bundle.parents[2] / "research-reviews" / decision["signal_date"] / digest(identity)[:16]
    if dest.exists():
        verify(dest)
        return dest
    outcomes = scenario_outcomes(cohort, decision, snapshot)
    matured = observed >= decision["horizon_dates"][-1]
    blockers = snapshot.formal_blockers(min_years=0)
    if execution_report and not read(Path(execution_report) / "gap-report.json")["snapshot_generated"]:
        blockers.append("execution_data_contract_incomplete")
    if config.get("allow_synthetic_for_tests") and snapshot.manifest["declaration"]["synthetic"]:
        blockers = []
    if not matured:
        blockers.append("outcomes_pending")
    future = snapshot.tables["bars"]
    future = future[future.instrument_id.isin(cohort.instrument_id) &
                    future.date.isin(decision["horizon_dates"])]
    if len(future) and (future[["upper_limit", "lower_limit"]].isna().any().any() or
                        future[["upper_limit", "lower_limit"]].le(0).any().any()):
        blockers.append("price_limits_unverified")
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".review-", dir=dest.parent))
    outcomes.to_csv(staging / "outcomes.csv", index=False)
    cards = []
    for arm in decision["arms"]:
        frame = outcomes[outcomes.arm.eq(arm["arm"])].sort_values("rank")
        top = frame.head(config["top_n"])
        arm_blockers = list(blockers)
        if not arm["publication"]["prospective"]:
            arm_blockers.append("late_publication_excluded")
        if matured and len(future[future.instrument_id.isin(top.instrument_id)]) != len(top)*5:
            arm_blockers.append("future_price_coverage_incomplete")
        card = {"arm": arm["arm"], "label": arm["label"], "publication": arm["publication"],
                "matured": bool(matured), "diagnostics": diagnostics(top),
                "execution": {"status": "blocked" if arm_blockers else "ready",
                              "blockers": sorted(set(arm_blockers))},
                "high_volatility_share": float(top.volatility_band.eq("q5").mean()) if len(top) else None,
                "strata": {key: [{"group": str(group), **diagnostics(part)}
                                 for group, part in frame.groupby(key, dropna=False)]
                           for key in ("board", "industry", "volatility_band") if key in frame}}
        if not arm_blockers and len(top) and arm["publication"]["prospective"]:
            predictions = top[["signal_date", "instrument_id", "rank", "planned_exit_date"]].rename(
                columns={"signal_date": "date"})
            predictions["score"] = -predictions["rank"]
            predictions["decision_at"] = arm["publication"]["published_at"]
            runs = []
            for multiplier in config["cost_multipliers"]:
                result = simulate(snapshot, predictions, 5, config["portfolio"], cost_multiplier=multiplier)
                prefix = arm["arm"] + "-cost-" + str(multiplier)
                for name in ("nav", "orders", "fills", "holdings", "cash_events"):
                    if name in result:
                        result[name].to_csv(staging / (prefix + "-" + name + ".csv"), index=False)
                orders = result["orders"]
                buy_orders = orders[orders.side.eq("buy")] if len(orders) else orders
                filled = int(buy_orders.status.isin(["filled", "partial"]).sum()) if len(buy_orders) else 0
                runs.append({"cost_multiplier": multiplier, **result["summary"],
                             "filled_instruments": filled,
                             "candidate_fill_coverage": filled / len(top),
                             "unfilled_candidates": len(top) - filled,
                             "ending_positions": int(result["nav"].positions.iloc[-1])})
            card["execution"] = {"status": "simulated", "blockers": [], "scenarios": runs,
                                 "interpretation": "Next-session open entry and frozen fifth-session close exit; simulated fills only."}
        cards.append(card)
    pairs = pd.read_csv(bundle / "matching-pairs.csv")
    control = outcomes[outcomes.arm.eq("control")].set_index("instrument_id")
    matched = []
    for pair in pairs.itertuples():
        treated = outcomes[outcomes.arm.eq(pair.arm)].set_index("instrument_id").loc[pair.treated]
        other = control.loc[pair.control]
        matched.append({"arm": pair.arm, "treated": pair.treated, "control": pair.control,
                        "known": bool(pd.notna(treated.fixed_open_close_scenario) and
                                      pd.notna(other.fixed_open_close_scenario)),
                        "open_close_scenario_difference": treated.fixed_open_close_scenario - other.fixed_open_close_scenario,
                        "adverse_difference": treated.adverse_scenario - other.adverse_scenario})
    matched = pd.DataFrame(matched, columns=["arm", "treated", "control", "known",
                                           "open_close_scenario_difference", "adverse_difference"])
    matched.to_csv(staging / "matched-outcomes.csv", index=False)
    write_json(staging / "review.json", {**identity, "signal_date": decision["signal_date"],
        "observed_through": observed, "created_at": utc_now(), "cards": cards,
        "matching": [{"arm": arm, "pairs": len(group), "known_pairs": int(group.known.sum()),
                      "prospective": decision["matching_prospective"] and next(a["publication"]["prospective"] for a in cards if a["arm"] == arm),
                      "mean_open_close_scenario_difference": float(group.loc[group.known, "open_close_scenario_difference"].mean()) if group.known.any() else None,
                      "mean_adverse_difference": float(group.loc[group.known, "adverse_difference"].mean()) if group.known.any() else None,
                      "mean_rank_gaps": {field: float(pairs.loc[pairs.arm.eq(arm), field + "_gap"].abs().mean())
                                         for field in config["matching"]["calipers"]}}
                     for arm, group in matched.groupby("arm")],
        "synthetic": snapshot.manifest["declaration"]["synthetic"],
        "comparison": "Different cohorts are descriptive shadow comparisons, not causal effects."})
    write_manifest(staging)
    staging.rename(dest)
    return dest


def refresh_research(store, source, config, overlays=(), output=None, execution_evidence=None, analysis_only=False):
    from .research_analysis import (
        aggregate_reports,
        build_analysis_review,
        data_freshness,
        refresh_analysis,
    )
    if analysis_only:
        if execution_evidence:
            raise ValueError("Execution imports require the explicit execution review mode")
        return refresh_analysis(store, source, config, overlays, output)
    from .execution_sources import public_observations
    store = Path(store)
    snapshot = outcome_snapshot(Path(source) / "snapshot")
    execution_pointer = store / "execution-evidence-publication.json"
    if execution_evidence:
        evidence = ExecutionEvidence(execution_evidence)
        write_json(execution_pointer, {"bundle": str(evidence.root), "identity": evidence.identity})
    elif execution_pointer.exists():
        ref = read(execution_pointer)
        evidence = ExecutionEvidence(ref["bundle"])
        if evidence.identity != ref["identity"]:
            raise ValueError("Registered execution evidence changed; supply a new immutable version")
    else:
        evidence = None
    summaries, bundles, latest = [], [], None
    analysis_reports = []
    registry_path = store / "event-overlay-publications.json"
    registered = read(registry_path) if registry_path.exists() else {}
    provided = set()
    for overlay in map(Path, overlays):
        verify(overlay)
        signal = read(overlay / "metadata.json")["signal_date"]
        if signal in provided:
            raise ValueError("Provide only one active event overlay per signal date")
        provided.add(signal)
        registered[signal] = {"bundle": str(overlay.resolve()),
                              "manifest_sha256": file_hash(overlay / "manifest.json")}
    decisions = {}
    for pointer in (store / "decision-publications").glob("*.json"):
        ref = read(pointer)
        bundle = Path(ref["bundle"])
        verify(bundle)
        if file_hash(bundle / "manifest.json") != ref["manifest_sha256"]:
            raise ValueError("Compact decision pointer changed")
        decisions[pointer.stem] = bundle
    for run in sorted((store / "runs").glob("*/run.json")):
        if not (store / "reference-publications" / (run.parent.name + ".json")).exists():
            continue
        active = []
        if run.parent.name in registered:
            event_ref = registered[run.parent.name]
            event = Path(event_ref["bundle"])
            if not event.exists():
                raise FileNotFoundError("Restore the registered event bundle before refreshing this model run")
            verify(event)
            if file_hash(event / "manifest.json") != event_ref["manifest_sha256"]:
                raise ValueError("Registered event overlay changed")
            active.append(event)
        bundle = freeze_decision(run.parent, config, active)
        decisions[run.parent.name] = bundle
        write_json(store / "decision-publications" / (run.parent.name + ".json"), {
            "bundle": str(bundle.resolve()), "manifest_sha256": file_hash(bundle / "manifest.json")})
    if provided:
        write_json(registry_path, registered)
    for signal, bundle in sorted(decisions.items()):
        analysis = build_analysis_review(bundle, snapshot)
        analysis_reports.append({**read(analysis / "analysis.json"), "manifest_sha256": file_hash(analysis / "manifest.json")})
        bundles.append(analysis)
        frozen_protocol = read(bundle / "decision.json")["protocol"]
        execution = build_execution_snapshot(bundle, snapshot, evidence)
        gaps = read(execution / "gap-report.json")
        execution_source = execution if gaps["snapshot_generated"] else Path(source)
        active_snapshot = outcome_snapshot(execution / "snapshot") if gaps["snapshot_generated"] else snapshot
        review = review_decision(bundle, execution_source, frozen_protocol,
                                 snapshot=active_snapshot, execution_report=execution)
        decision, result = read(bundle / "decision.json"), read(review / "review.json")
        cohort = pd.read_csv(bundle / "cohort.csv")
        trial = {"model_config": decision["model_config"], "review_protocol": frozen_protocol,
                 "arms": {a["arm"]: a["lineage"]["delivery"].get("config_sha256") for a in decision["arms"]}}
        trial_id = digest(trial)
        trial_path = store / "experiments" / (trial_id + ".json")
        if not trial_path.exists():
            write_json(trial_path, {"trial_id": trial_id, "registered_at": utc_now(),
                                   "protocol": trial, "status": "shadow_only"})
        summaries.append(result)
        bundles.extend([bundle, review, trial_path, execution])
        compact_candidates = []
        for row in gaps["candidates"]:
            grouped = {}
            for gap in row["gaps"]:
                grouped.setdefault((gap["component"], gap["reason"]), []).append(gap["date"])
            compact_candidates.append({**row, "gaps": [
                {"component": component, "reason": reason, "dates": sorted(set(dates))}
                for (component, reason), dates in grouped.items()]})
        latest = {"signal_date": decision["signal_date"], "horizon_dates": decision["horizon_dates"],
                  "arms": [{**a, "candidates": records(cohort[cohort.arm.eq(a["arm"])].head(config["top_n"]))}
                           for a in result["cards"]],
                  "decision_sha256": file_hash(bundle / "manifest.json"),
                  "review_sha256": file_hash(review / "manifest.json")}
        latest["execution_data"] = {**{key: gaps[key] for key in (
            "status", "snapshot_generated", "evidence_supplied", "component_labels",
            "reason_labels", "global_checks", "acceptance")},
            "candidates": compact_candidates, "report_sha256": file_hash(execution / "gap-report.json")}
    operations = [read(p) for p in store.glob("operations/*.json")]
    operations.sort(key=lambda row: row["started_at"])
    matched_summary = []
    for arm in ("start_stage", "event_overlay", "confirmed_event"):
        points = [m for s in summaries if not s["synthetic"] for m in s["matching"]
                  if m["arm"] == arm and m["prospective"] and m["known_pairs"]]
        matched_summary.append({"arm": arm, "mature_dates": len(points),
                                "known_pairs": sum(p["known_pairs"] for p in points),
                                "minimum_dates": config["matching"]["minimum_dates"],
                                "minimum_pairs": config["matching"]["minimum_pairs"],
                                "mean_open_close_scenario_difference": float(np.mean([p["mean_open_close_scenario_difference"] for p in points])) if points else None,
                                "improved_date_fraction": float(np.mean([p["mean_open_close_scenario_difference"] > 0 for p in points])) if points else None,
                                "mean_adverse_difference": float(np.mean([p["mean_adverse_difference"] for p in points])) if points else None,
                                "balance": {field: float(np.mean([p["mean_rank_gaps"][field] for p in points])) if points else None
                                            for field in config["matching"]["calipers"]},
                                "promotion_eligible": False,
                                "status": "accumulating_frozen_shadow_evidence"})
    summary = {"schema_version": 1, "generated_at": utc_now(), "matching": matched_summary,
               "mode": "analysis_and_execution", "analysis_reports": analysis_reports,
               "analysis_overview": aggregate_reports(analysis_reports), "freshness": data_freshness(snapshot),
               "source_observations": public_observations(store),
               "engineering_only": snapshot.manifest["declaration"]["synthetic"],
               "observed_through": summaries[-1]["observed_through"] if summaries else None,
               "latest": latest, "history": summaries,
               "prospective_mature_dates": len({s["signal_date"] for s in summaries if not s["synthetic"] for a in s["cards"]
                                                if a["publication"]["prospective"] and a["matured"]}),
               "experiment_count": len(list((store / "experiments").glob("*.json"))),
               "prospective_simulated_dates": len({s["signal_date"] for s in summaries
                   if not s["synthetic"] for a in s["cards"]
                   if a["publication"]["prospective"] and a["execution"]["status"] == "simulated"}),
               "operations": operations[-10:], "blocker_labels": EXPLANATIONS,
               "policy": "Late reports are excluded from forward performance; scenario returns and simulated fills are separate."}
    write_json(store / "research-latest.json", summary)
    if output:
        write_json(Path(output), summary)
    return {"summary": str(store / "research-latest.json"),
            "evidence": [str(p) for p in bundles], "prospective_mature_dates": summary["prospective_mature_dates"]}
