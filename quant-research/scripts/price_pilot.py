"""Bounded five-day price pilot. Each stage runs in its own interpreter.

No rank training, no final holdout access, no automatic queue or cloud writes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.price_strategy import (
    FIELDS,
    GRID,
    PlanCalibrator,
    TradeAssumptions,
    candidate_plans,
    choose_plan,
    forecast_metrics,
    ordered_quantiles,
    price_labels,
    trade_diagnostic,
)
from quant_research.storage import file_hash, utc_now, write_json


def verified_json(path):
    return json.loads(path.read_text())


def load_arrays(path):
    # NpzFile decompresses on every __getitem__. Materialize once before loops.
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def verify_files(root, files):
    for name, expected in files.items():
        p = (root / name).resolve()
        if not p.is_relative_to(root.resolve()) or file_hash(p) != expected:
            raise ValueError(f"Artifact hash mismatch: {name}")


def pick_dates(dates, start, end, count, boundary):
    eligible = [d for i, d in enumerate(dates) if start <= d <= end
                and i + 5 < len(dates) and dates[i + 5] < boundary]
    if not eligible or count < 1:
        raise ValueError("No dates available")
    return [eligible[i] for i in np.unique(np.linspace(0, len(eligible) - 1,
                                                     min(count, len(eligible)), dtype=int))]


def prepare(args):
    root, out = args.root.resolve(), args.output.resolve()
    if out.exists():
        raise FileExistsError("Use a new output directory")
    if min(args.train_dates, args.validation_dates, args.eval_dates,
           args.iterations, args.kronos_per_exchange) < 1 or args.path_samples < 8:
        raise ValueError("Invalid bounded pilot configuration")
    acceptance = verified_json(root / "development-acceptance.json")
    if (not acceptance.get("passed") or acceptance.get("formal_ready") is not False
            or acceptance.get("mode") != "reconstructed_history_development_training"):
        raise ValueError("Unaccepted development dataset")
    verify_files(root, acceptance["evidence"])
    schedule = verified_json(root / "schedule.json")
    fold = schedule["development"][args.fold]
    sealed = schedule["sealed_holdout_start"]
    if fold["purpose"] != "development" or fold["test_end"] >= sealed:
        raise ValueError("Sealed holdout cannot be opened")
    panel_root = root / "panel-h5"
    if file_hash(panel_root / "manifest.json") != acceptance["panels"]["5"]:
        raise ValueError("Panel not accepted")
    panel = verified_json(panel_root / "manifest.json")
    verify_files(panel_root, panel["files"])
    if panel["dataset_id"] != acceptance["dataset_id"] or not panel["completed"]:
        raise ValueError("Panel dataset mismatch")
    snapshot = root / "snapshot"
    manifest = verified_json(snapshot / "manifest.json")
    verify_files(snapshot, manifest["files"])
    if manifest["dataset_id"] != acceptance["dataset_id"]:
        raise ValueError("Snapshot dataset mismatch")
    dates = panel["dates"]
    validation_days = [d for d in dates if fold["validation_start"] <= d < fold["test_start"]]
    cal_start = validation_days[len(validation_days) // 2]
    train_start = max(fold["train_start"], str((pd.Timestamp(fold["validation_start"])
                                             - pd.DateOffset(years=1)).date()))
    calibration_count = args.calibration_dates or args.validation_dates
    if calibration_count < 1:
        raise ValueError("Invalid calibration dates")
    intervals = {
        "train": (train_start, fold["validation_start"], args.train_dates, fold["validation_start"]),
        "selection": (fold["validation_start"], cal_start, args.validation_dates, cal_start),
        "calibration": (cal_start, fold["test_start"], calibration_count, fold["test_start"]),
        "evaluation": (fold["test_start"], fold["test_end"], args.eval_dates, fold["test_end"]),
    }
    chosen = {k: pick_dates(dates, *v) for k, v in intervals.items()}
    if getattr(args, "date_plan", None):
        from quant_research.rolling_price import validate_date_plan
        chosen = verified_json(args.date_plan)
        boundaries = validate_date_plan(chosen, dates, fold, sealed)
        intervals = {k: (v[0], v[-1], len(v), boundaries[k]) for k, v in chosen.items()}
    selected_dates = sum(chosen.values(), [])
    if len(selected_dates) != len(set(selected_dates)):
        raise ValueError("Partition overlap")
    all_rows = pd.read_parquet(panel_root / "samples.parquet",
        filters=[("date", "in", selected_dates)],
        columns=["date", "instrument_id", "stock_index", "date_index", "risk_status",
                 "trading_eligible", "source_is_st"])
    meta = pd.read_parquet(snapshot / "instruments.parquet")
    needed = sorted({dates[t + h] for t in all_rows.date_index.unique() for h in range(6)})
    if max(needed) >= sealed:
        raise ValueError("Attempted holdout label read")
    bars = pd.read_parquet(snapshot / "bars.parquet", filters=[("date", "in", needed)])
    actions = pd.read_parquet(snapshot / "actions.parquet")
    values = np.load(panel_root / "values.npy", mmap_mode="r")
    out.mkdir(parents=True)
    config = {"created_at": utc_now(), "dataset_id": acceptance["dataset_id"],
        "root": str(root), "fold": fold, "sealed_holdout_start": sealed,
        "horizon": 5, "horizon_definition": "next session day1 through day5 close",
        "dates": chosen, "seed": 17, "quantiles": [.1, .5, .9],
        "iterations": args.iterations, "candidate_grid": GRID,
        "assumptions": asdict(TradeAssumptions()), "kronos_per_exchange": args.kronos_per_exchange,
        "path_samples": args.path_samples, "label_units": "log(raw future quote / signal close)",
        "mode": "bounded_development_price_and_hypothetical_trade_diagnostic",
        "formal_ready": False, "executable": False,
        "limitations": ["training-only snapshot; no verified execution contract",
            "missing historical risk/price-limit/corporate-action evidence",
            "no portfolio NAV, capital, board-lot or liquidity-capacity simulation",
            "daily OHLC touching is not proof of a fill",
            "Kronos pretraining overlap unverified", "small date and Kronos identity samples"],
        "source_hashes": {"acceptance": file_hash(root / "development-acceptance.json"),
                          "panel_manifest": file_hash(panel_root / "manifest.json"),
                          "snapshot_manifest": file_hash(snapshot / "manifest.json")}}
    config["matched_calibration"] = args.matched_calibration
    if getattr(args, "date_plan", None):
        config["date_plan_sha256"] = file_hash(args.date_plan)
    if args.prior_pilot:
        prior = verified_json(args.prior_pilot / "config.json")
        if prior["dataset_id"] != config["dataset_id"]:
            raise ValueError("Prior pilot dataset differs")
        config["prior_price_evaluation_dates"] = prior["dates"]["evaluation"]
        config["prior_pilot"] = {"path": str(args.prior_pilot.resolve()),
            "config_sha256": file_hash(args.prior_pilot / "config.json")}
    config["stability_protocol"] = {
        "day_weighting": "equal signal date; no IID-stock significance test",
        "cost_stress": "double commission/minimum_fee/sell_tax/slippage; fixed chosen orders",
        "primary_price_target": "day5 close pinball",
        "calibration_comparison": "same finite-forecast rows" if args.matched_calibration else "model-specific",
        "evaluation_status": "development quarter previously used for ranking; not sealed test"}
    write_json(out / "config.json", config)
    # Fixed, exchange-stratified identity sampling before future labels/predictions.
    subset = []
    rng = np.random.default_rng(17)
    for exchange, group in meta.sort_values("instrument_id").groupby("exchange"):
        candidates = group.loc[group.listed_at <= min(chosen["calibration"]), "instrument_id"]
        subset.extend(rng.choice(candidates, min(args.kronos_per_exchange, len(candidates)),
                                 replace=False).tolist())
    write_json(out / "kronos-identities.json", {"identities": subset,
        "selection": "seed17 exchange-stratified historical identities; no outcome/availability filter"})
    coverage = {}
    for partition, days in chosen.items():
        rows = all_rows.loc[all_rows.date.isin(days)].sort_values(["date", "instrument_id"]).copy()
        # Save the entire selected cohort, including signals with unusable raw reference.
        refs = bars.set_index(["instrument_id", "date"]).close
        reference = refs.reindex(pd.MultiIndex.from_frame(rows[["instrument_id", "date"]])).to_numpy()
        input_good = np.isfinite(reference) & (reference > 0)
        rows["input_status"] = np.where(input_good, "available", "missing_signal_quote")
        rows.to_parquet(out / f"{partition}-cohort.parquet", index=False)
        rows = rows.loc[input_good].reset_index(drop=True)
        rows["kronos_selected"] = rows.instrument_id.isin(subset)
        labels = price_labels(bars, rows, dates, actions, intervals[partition][3])
        x = values[rows.stock_index.to_numpy(int), rows.date_index.to_numpy(int)].copy()
        if not np.isfinite(x).all():
            raise ValueError("Past panel features invalid")
        rows["label_end"] = [dates[t + 5] for t in rows.date_index]
        rows.to_parquet(out / f"{partition}-rows.parquet", index=False)
        np.savez_compressed(out / f"{partition}.npz", x=x, **labels)
        coverage[partition] = {"cohort": int(len(input_good)), "input_rows": len(rows),
            "all_five_days_labelled": int(labels["valid"].all(axis=1).sum()),
            "labelled_by_day": labels["valid"].sum(axis=0).tolist(),
            "kronos_selected_rows": int(rows.kronos_selected.sum())}
        print(partition, coverage[partition], flush=True)
    write_json(out / "coverage.json", coverage)
    # Source snapshot for reproducibility. Execution must use this saved source.
    import shutil
    code = out / "code"
    shutil.copytree(Path(__file__).resolve().parents[1] / "src" / "quant_research",
                    code / "quant_research", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(__file__, code / "price_pilot.py")
    shutil.copy2(Path(__file__).resolve().parents[1] / "uv.lock", out / "uv.lock")
    write_json(out / "prepared-manifest.json", {str(p.relative_to(out)): file_hash(p)
        for p in out.rglob("*") if p.is_file() and p.name != "prepared-manifest.json"})


def tree(args):
    import lightgbm as lgb
    out = args.output
    cfg = verified_json(out / "config.json")
    train = load_arrays(out / "train.npz")
    selection = load_arrays(out / "selection.npz")
    data = {p: load_arrays(out / f"{p}.npz") for p in ("calibration", "evaluation")}
    predictions = {p: np.empty((len(d["x"]), 5, 4, 3)) for p, d in data.items()}
    model_dir = out / "lightgbm-models"
    model_dir.mkdir()
    rows = pd.read_parquet(out / "train-rows.parquet")
    weights = 1 / rows.groupby("date").date.transform("size").to_numpy()
    val_rows = pd.read_parquet(out / "selection-rows.parquet")
    val_weights = 1 / val_rows.groupby("date").date.transform("size").to_numpy()
    log = []
    for h in range(5):
        for f in range(4):
            y, v = train["targets"][:, h, f], selection["targets"][:, h, f]
            good, vg = np.isfinite(y), np.isfinite(v)
            if good.sum() < 100 or vg.sum() < 20:
                raise ValueError("Insufficient observed training/selection labels")
            for j, q in enumerate(cfg["quantiles"]):
                model = lgb.LGBMRegressor(objective="quantile", alpha=q,
                    n_estimators=cfg["iterations"], learning_rate=.05, num_leaves=15,
                    min_child_samples=100, reg_lambda=1., n_jobs=4, random_state=cfg["seed"],
                    verbosity=-1, deterministic=True, force_col_wise=True)
                model.fit(train["x"][good], y[good], sample_weight=weights[good],
                    eval_set=[(selection["x"][vg], v[vg])], eval_sample_weight=[val_weights[vg]],
                    callbacks=[lgb.early_stopping(10, verbose=False)])
                model.booster_.save_model(str(model_dir / f"d{h+1}-{FIELDS[f]}-q{q}.txt"))
                for p, d in data.items():
                    predictions[p][:, h, f, j] = model.predict(d["x"])
                log.append({"day": h+1, "field": FIELDS[f], "q": q,
                            "best_iteration": model.best_iteration_})
            print(f"LightGBM day {h+1}, {FIELDS[f]} complete", flush=True)
    for p, pred in predictions.items():
        np.save(out / f"lightgbm-{p}.npy", ordered_quantiles(pred))
    # Constant empirical train-only quantiles are a mandatory naive price baseline.
    baseline = np.nanquantile(train["targets"], cfg["quantiles"], axis=0).transpose(1, 2, 0)
    for p, d in data.items():
        np.save(out / f"naive-{p}.npy", np.broadcast_to(baseline, (len(d["x"]), 5, 4, 3)))
    write_json(out / "lightgbm-training.json", {"models": log, "objective": "quantile_pinball",
        "selection": "early stopping on selection only; calibration and evaluation untouched"})


def kronos(args):
    from quant_research.price_kronos import native_paths, path_quantiles
    out, bundle = args.output, args.bundle.resolve()
    cfg = verified_json(out / "config.json")
    m = verified_json(bundle / "inputs/manifest.json")
    if m["dataset_id"] != cfg["dataset_id"] or not m["completed"]:
        raise ValueError("Kronos input identity mismatch")
    if file_hash(bundle / "inputs/values.npy") != m["values_sha256"]:
        raise ValueError("Kronos input hash mismatch")
    raw = np.load(bundle / "inputs/values.npy", mmap_mode="r")
    stock_axis = {s: i for i, s in enumerate(m["instruments"])}
    date_axis = {d: i for i, d in enumerate(m["dates"])}
    all_windows, past, future, slots = [], [], [], []
    arrays, ledgers = {}, {}
    for part in ("calibration", "evaluation"):
        rows = pd.read_parquet(out / f"{part}-rows.parquet")
        arrays[part] = np.full((len(rows), cfg["path_samples"], 5, 6), np.nan, np.float32)
        ledgers[part] = []
        for i, row in rows.loc[rows.kronos_selected].iterrows():
            t, s = date_axis[row.date], stock_axis[row.instrument_id]
            window = raw[s, t - 59:t + 1]
            if (len(window) != 60 or t + 5 >= len(m["dates"])
                    or m["dates"][t+5] >= cfg["sealed_holdout_start"]):
                raise ValueError("Kronos window/calendar boundary")
            good = bool(np.isfinite(window).all() and (window[:, :4] > 0).all())
            ledgers[part].append({"row_index": int(i), "date": row.date,
                "instrument_id": row.instrument_id, "input_status": "available" if good else "missing"})
            if good:
                all_windows.append(window.copy())
                past.append(m["dates"][t-59:t+1])
                future.append(m["dates"][t+1:t+6])
                slots.append((part, i))
    if not slots:
        raise ValueError("No selected Kronos histories available")
    paths = native_paths(bundle, np.stack(all_windows), past, future,
        samples=cfg["path_samples"], seed=cfg["seed"], device=args.device)
    for (part, i), p in zip(slots, paths):
        arrays[part][i] = p
    for part, paths in arrays.items():
        labels = np.load(out / f"{part}.npz")
        q, good = path_quantiles(paths, labels["reference"])
        # Sparse rows only for path storage; predictions remain on the full cohort axis.
        ids = np.array([r["row_index"] for r in ledgers[part]], dtype=int)
        np.savez_compressed(out / f"kronos-{part}-paths.npz", row_indices=ids,
                            paths=paths[ids], valid_paths=good[ids])
        np.save(out / f"kronos-{part}.npy", q)
        write_json(out / f"kronos-{part}-coverage.json", {"rows": ledgers[part],
            "valid_paths": good[ids].sum(axis=1).tolist(), "min_paths_for_forecast": 8})
    write_json(out / "kronos-native.json", {"adaptation": "none; original pretrained weights",
        "device": args.device, "sample_count": cfg["path_samples"], "paths_averaged": False,
        "provenance": verified_json(bundle / "pretrained-provenance.json"),
        "inputs_manifest_sha256": file_hash(bundle / "inputs/manifest.json"),
        "probabilities_calibrated": False})


def outcomes_for(labels, assumptions):
    return [[trade_diagnostic(labels["future"][i], labels["valid"][i], plan, assumptions,
                             labels["upper"][i], labels["lower"][i])
             for plan in candidate_plans(ref, assumptions)]
            for i, ref in enumerate(labels["reference"])]


def report(args):
    out = args.output
    cfg = verified_json(out / "config.json")
    assumptions = TradeAssumptions(**cfg["assumptions"])
    parts = {p: load_arrays(out / f"{p}.npz") for p in ("calibration", "evaluation")}
    rows = pd.read_parquet(out / "evaluation-rows.parquet")
    outcomes = {}
    for p, d in parts.items():
        print(f"Simulating fixed candidate plans: {p}", flush=True)
        outcomes[p] = outcomes_for(d, assumptions)
    forecasts = {m: {p: np.load(out / f"{m}-{p}.npy") for p in parts}
                 for m in ("naive", "lightgbm", "kronos")}
    common = np.logical_and.reduce([np.isfinite(v["evaluation"]).all(axis=(1, 2, 3))
                                    for v in forecasts.values()])
    cal_common = np.logical_and.reduce([np.isfinite(v["calibration"]).all(axis=(1, 2, 3))
                                       for v in forecasts.values()])
    calibration_mask = cal_common if cfg.get("matched_calibration") else np.ones(len(cal_common), bool)
    summary, all_plans, all_forecasts = {}, [], []
    for model, pred in forecasts.items():
        print(f"Calibrating and evaluating {model}", flush=True)
        cal = PlanCalibrator().fit(pred["calibration"][calibration_mask],
            [o for o, keep in zip(outcomes["calibration"], calibration_mask) if keep])
        estimates = cal.estimate(pred["evaluation"])
        plan_rows = []
        for i, row in rows.iterrows():
            choice = choose_plan(estimates[i])
            record = {"model": model, "date": row.date, "instrument_id": row.instrument_id,
                "common_comparison": bool(common[i]), "risk_status": row.risk_status,
                "executable": False, "execution_blocker": "training_snapshot_no_execution_contract",
                "signal": "research_candidate" if choice is not None else "observe",
                "forecast_available": bool(np.isfinite(pred["evaluation"][i]).all()),
                "status": "no_order", "net_return": 0., "filled": False}
            record["decision_reason"] = (
                "positive_conditional_edge" if choice is not None else
                "forecast_unavailable" if not record["forecast_available"] else
                "calibration_support_insufficient" if all(e is None for e in estimates[i]) else
                "fill_or_risk_adjusted_edge_below_threshold")
            if choice is not None:
                plan = candidate_plans(parts["evaluation"]["reference"][i], assumptions)[choice]
                record.update(asdict(plan), candidate_index=choice, **estimates[i][choice],
                              **outcomes["evaluation"][i][choice])
            plan_rows.append(record)
        # Vectorize the expanded quote table instead of millions of Python dicts.
        available = np.isfinite(pred["evaluation"]).all(axis=(1, 2, 3))
        quote_rows = rows.loc[available]
        prices = (np.exp(pred["evaluation"][available])
                  * parts["evaluation"]["reference"][available, None, None, None]).reshape(-1, 3)
        all_forecasts.append(pd.DataFrame({"model": model,
            "date": np.repeat(quote_rows.date.to_numpy(), 20),
            "instrument_id": np.repeat(quote_rows.instrument_id.to_numpy(), 20),
            "day": np.tile(np.repeat(np.arange(1, 6), 4), len(quote_rows)),
            "field": np.tile(FIELDS, len(quote_rows) * 5),
            "q10": prices[:, 0], "q50": prices[:, 1], "q90": prices[:, 2], "executable": False}))
        frame = pd.DataFrame(plan_rows)
        # Exact next-session validity and common day-5 time exit from the saved calendar.
        axis = verified_json(Path(cfg["root"]) / "panel-h5/manifest.json")["dates"]
        frame["buy_valid_until"] = [axis[t+1] for t in rows.date_index]
        frame["time_exit_date"] = rows.label_end.to_numpy()
        all_plans.append(frame)
        result = {"price_all_available": forecast_metrics(parts["evaluation"]["targets"], pred["evaluation"]),
                  "price_common_cohort": forecast_metrics(parts["evaluation"]["targets"][common],
                                                           pred["evaluation"][common])}
        for label, f in (("all", frame), ("common", frame.loc[common])):
            orders = f.loc[f.signal == "research_candidate"]
            filled = orders.loc[orders.filled.eq(True)]
            known = orders.net_return.dropna()
            fill_known = orders.loc[orders.filled.notna()]
            fill_brier = (float(((fill_known.fill_probability - fill_known.filled.astype(float)) ** 2).mean())
                          if len(fill_known) else None)
            result[label + "_trade_diagnostic"] = {
                "signals": len(f), "candidate_orders": len(orders), "filled": len(filled),
                "unresolved": int(orders.net_return.isna().sum()),
                "selected_order_fill_brier": fill_brier,
                "mean_net_per_resolved_order": float(known.mean()) if len(known) else None,
                "mean_net_per_resolved_fill": float(filled.net_return.mean())
                    if filled.net_return.notna().any() else None,
                "not_a_portfolio_return": True}
        summary[model] = result
        print(f"Completed {model} diagnostics", flush=True)
    pd.concat(all_plans, ignore_index=True).to_csv(out / "trade-plans.csv", index=False)
    pd.concat(all_forecasts, ignore_index=True).to_parquet(out / "price-forecasts.parquet", index=False)
    write_json(out / "comparison.json", {"models": summary, "common_rows": int(common.sum()),
        "matched_calibration": bool(cfg.get("matched_calibration")),
        "common_calibration_rows": int(cal_common.sum()),
        "evaluation_cohort": len(rows), "formal_ready": False, "executable": False})
    lines = ["# 5 日价格预测：快速开发验证", "", "此报告为历史开发诊断，所有价格均非当前交易建议。",
        "数据缺少已核验的成交契约，候选计划均标记为不可执行；没有生成组合净值。", "",
        "| 模型 | 共同预测样本 | 其中收盘可评分 | 5 日收盘分位数损失 | 80% 区间覆盖 | 候选订单 | 未解决订单 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for model, r in summary.items():
        metric = r["price_common_cohort"]["targets"][-1]
        diag = r["common_trade_diagnostic"]
        loss = f"{metric['pinball']:.6f}" if metric['pinball'] is not None else "无数据"
        coverage = (f"{metric['interval_80_coverage']:.1%}"
                    if metric['interval_80_coverage'] is not None else "无数据")
        lines.append(f"| {model} | {int(common.sum())} | {metric['scored_rows']} | {loss} | "
                     f"{coverage} | {diag['candidate_orders']} | {diag['unresolved']} |")
    lines += ["", "分位数损失越低越好；80% 区间覆盖应在更大样本上接近 80%。",
        "", "| 全量价格评估 | 5 日收盘可评分样本 | 分位数损失 | 80% 区间覆盖 |",
        "|---|---:|---:|---:|"]
    for model in ("naive", "lightgbm"):
        m = summary[model]["price_all_available"]["targets"][-1]
        loss = f"{m['pinball']:.6f}" if m['pinball'] is not None else "无数据"
        coverage = f"{m['interval_80_coverage']:.1%}" if m['interval_80_coverage'] is not None else "无数据"
        lines.append(f"| {model} | {m['scored_rows']} | {loss} | {coverage} |")
    lines += ["", "共同样本按相同股票、信号日期和标签比较；全量 LightGBM 指标单独保留。",
        "训练、早停、交易校准和评估采用互不重叠的时间区间，并按真实第 5 日标签结束日隔离。",
        "Kronos 使用原始预训练权重；各采样路径保留，预训练与历史评估重叠尚未排除。",
        "", "文件：trade-plans.csv 包含买价、止盈、止损、有效期及观望；",
        "price-forecasts.parquet 包含未来每日 OHLC 的 10/50/90% 价格分位数；",
        "comparison.json 和 coverage.json 包含完整指标及覆盖缺口。"]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "tree", "kronos", "report"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--root", type=Path)
    p.add_argument("--bundle", type=Path)
    p.add_argument("--fold", type=int, default=14)
    p.add_argument("--train-dates", type=int, default=24)
    p.add_argument("--date-plan", type=Path)
    p.add_argument("--validation-dates", type=int, default=4)
    p.add_argument("--calibration-dates", type=int)
    p.add_argument("--matched-calibration", action="store_true")
    p.add_argument("--prior-pilot", type=Path)
    p.add_argument("--eval-dates", type=int, default=2)
    p.add_argument("--iterations", type=int, default=60)
    p.add_argument("--kronos-per-exchange", type=int, default=6)
    p.add_argument("--path-samples", type=int, default=16)
    p.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = p.parse_args()
    if args.stage != "prepare":
        verify_files(args.output, verified_json(args.output / "prepared-manifest.json"))
        if args.stage == "report":
            for stage in ("tree", "kronos"):
                state = verified_json(args.output / f"{stage}-status.json")
                if state["status"] != "completed":
                    raise ValueError("Prediction stages must complete before evaluation")
                verify_files(args.output, state["files"])
        marker = args.output / f"{args.stage}-status.json"
        if marker.exists():
            raise FileExistsError("Stage already attempted; use a new pilot directory")
        write_json(marker, {"status": "running", "pid": os.getpid(), "started_at": utc_now()})
    try:
        globals()[args.stage](args)
    except BaseException as e:
        if args.stage != "prepare":
            write_json(marker, {"status": "interrupted" if isinstance(e, KeyboardInterrupt) else "failed",
                                "error": repr(e), "updated_at": utc_now()})
        raise
    if args.stage != "prepare":
        patterns = {"tree": ["lightgbm-models/*.txt", "lightgbm-*.npy", "naive-*.npy",
                             "lightgbm-training.json"],
                    "kronos": ["kronos-*.npy", "kronos-*-paths.npz", "kronos-*-coverage.json",
                               "kronos-native.json"],
                    "report": ["comparison.json", "trade-plans.csv", "price-forecasts.parquet",
                               "report.md"]}
        files = {str(f.relative_to(args.output)): file_hash(f)
                 for pattern in patterns[args.stage] for f in args.output.glob(pattern)}
        write_json(marker, {"status": "completed", "updated_at": utc_now(), "files": files})


if __name__ == "__main__":
    sys.exit(main())
