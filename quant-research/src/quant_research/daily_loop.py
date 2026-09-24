"""Research-only premarket prediction, immutable drafts and append-only reviews."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .daily_models import adapter, validate_forecast
from .price_strategy import Plan, TradeAssumptions, price_labels, trade_diagnostic
from .storage import file_hash, utc_now, write_json

TZ = ZoneInfo("Asia/Shanghai")
POLICY = {
    "version": "conditional-draft-v1",
    "tick": 0.01,
    "min_reward_risk": 1.0,
    "cost_buffer_fraction": 0.0025,
    "entry_expiry": "T close",
    "exit_horizon": "T+4",
    "opening_invalidation": "open <= stop",
    "status": "unvalidated_research_rule",
    "max_display": 10,
}


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def verify(root, filename="manifest.json"):
    info = read(root / filename)
    for name, expected in info["files"].items():
        f = (root / name).resolve()
        if not f.is_relative_to(root.resolve()) or file_hash(f) != expected:
            raise ValueError(f"Artifact hash mismatch: {name}")
    return info


def write_manifest(root):
    write_json(
        root / "manifest.json",
        {
            "files": {
                str(f.relative_to(root)): file_hash(f)
                for f in root.rglob("*")
                if f.is_file() and f.name != "manifest.json"
            }
        },
    )


def source_code():
    names = [
        "daily_loop.py",
        "daily_models.py",
        "daily_input.py",
        "features.py",
        "price_strategy.py",
        "storage.py",
    ]
    return {name: file_hash(Path(__file__).with_name(name)) for name in names}


def commit_directory(stage, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(stage)
    try:
        os.rename(stage, target)
    except OSError:
        if not target.exists():
            raise
        verify(target)
        shutil.rmtree(stage)
    return target


def register_model(pilot, store, quality):
    pilot, store = Path(pilot).resolve(), Path(store).resolve()
    prepared = read(pilot / "prepared-manifest.json")
    for name in ["config.json", "train-rows.parquet", "selection-rows.parquet", "uv.lock"]:
        if file_hash(pilot / name) != prepared[name]:
            raise ValueError(f"Changed preparation artifact: {name}")
    cfg = read(pilot / "config.json")
    status = read(pilot / "tree-status.json")
    if status["status"] != "completed":
        raise ValueError("Unfinished model training")
    for name, expected in status["files"].items():
        path = (pilot / name).resolve()
        if not path.is_relative_to(pilot) or file_hash(path) != expected:
            raise ValueError("Changed training artifact")
    panel = read(Path(cfg["root"]) / "panel-h5/manifest.json")
    if panel["dataset_id"] != cfg["dataset_id"]:
        raise ValueError("Training data identity mismatch")
    # Includes selection labels, which selected the saved iteration counts.
    learned_through = max(
        pd.read_parquet(pilot / f"{s}-rows.parquet").label_end.max() for s in ["train", "selection"]
    )
    quality_data = read(quality)
    metadata = {
        "adapter": "lightgbm",
        "feature_names": panel["feature_names"],
        "lookback": 60,
        "horizon": 5,
        "trained_labels_through": learned_through,
        "training_dataset_id": cfg["dataset_id"],
        "source_pilot": str(pilot),
        "source_tree_status_sha256": file_hash(pilot / "tree-status.json"),
        "quality_evidence_sha256": file_hash(Path(quality)),
        "quality_passed": quality_data.get("models", {}).get("lightgbm", {}).get("passed", False),
        "strategy_validated": False,
        "model_files": {
            n: h for n, h in status["files"].items() if n.startswith("lightgbm-models/")
        },
        "registered_at": utc_now(),
    }
    expected = {
        f"lightgbm-models/d{d}-{f}-q{q}.txt"
        for d in range(1, 6)
        for f in ["open", "high", "low", "close"]
        for q in [0.1, 0.5, 0.9]
    }
    if set(metadata["model_files"]) != expected:
        raise ValueError("Incomplete five-session quantile model package")
    metadata["naive_source_sha256"] = status["files"]["naive-evaluation.npy"]
    metadata["training_lock_sha256"] = prepared["uv.lock"]
    metadata["prepared_manifest_sha256"] = file_hash(pilot / "prepared-manifest.json")
    identity = {k: v for k, v in metadata.items() if k != "registered_at"}
    model_id = digest(identity)[:20]
    target = store / "models" / model_id
    if target.exists():
        verify(target)
        return target
    store.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".model-", dir=store))
    try:
        (stage / "weights").mkdir()
        for name in metadata["model_files"]:
            shutil.copy2(pilot / name, stage / "weights" / Path(name).name)
        q = np.load(pilot / "naive-evaluation.npy", mmap_mode="r")[0]
        np.save(stage / "naive.npy", q)
        shutil.copy2(quality, stage / "quality-evidence.json")
        shutil.copy2(pilot / "uv.lock", stage / "uv.lock")
        write_json(stage / "model.json", {**metadata, "model_id": model_id})
        return commit_directory(stage, target)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def inspect_input(root, model, trade_date, cutoff, mode, now=None):
    root = Path(root).resolve()
    now = pd.Timestamp(now or utc_now())
    cutoff = pd.Timestamp(cutoff)
    if cutoff.tzinfo is None or now.tzinfo is None:
        raise ValueError("Timezone-aware cutoff and clock required")
    local = cutoff.tz_convert("Asia/Shanghai")
    if (
        mode not in {"replay", "prospective", "intraday_research"}
        or local.strftime("%Y-%m-%d") != trade_date
    ):
        raise ValueError("Invalid mode or trade-date cutoff")
    panel = read(root / "panel-h5/manifest.json")
    snapshot = read(root / "snapshot/manifest.json")
    dates = panel["dates"]
    issues = []
    if mode in {"prospective", "intraday_research"}:
        if now < cutoff:
            issues.append("cutoff_not_reached")
        if (
            now.tz_convert("Asia/Shanghai").strftime("%Y-%m-%d") != trade_date
            or (
                mode == "prospective"
                and now.tz_convert("Asia/Shanghai").strftime("%H:%M") >= "09:15"
            )
            or (
                mode == "intraday_research"
                and not "09:15" <= now.tz_convert("Asia/Shanghai").strftime("%H:%M") < "15:00"
            )
        ):
            issues.append("outside_premarket_generation_window")
        if pd.Timestamp(panel["created_at"]) > cutoff:
            issues.append("features_not_available_at_cutoff")
        if pd.Timestamp(snapshot["created_at"]) > cutoff:
            issues.append("snapshot_not_available_at_cutoff")
        if pd.Timestamp(model["registered_at"]) > cutoff:
            issues.append("model_not_available_at_cutoff")
    if mode != "intraday_research" and local.strftime("%H:%M") >= "09:15":
        issues.append("cutoff_after_premarket_window")
    if mode == "intraday_research" and not "09:15" <= local.strftime("%H:%M") < "15:00":
        issues.append("invalid_intraday_publication_cutoff")
    horizon, signal = [], None
    if trade_date not in dates:
        issues.append("trade_date_not_in_source_calendar")
    else:
        i = dates.index(trade_date)
        if i < model["lookback"] or i + 4 >= len(dates):
            issues.append("insufficient_past_or_five_session_calendar")
        else:
            horizon, signal = dates[i : i + 5], dates[i - 1]
            if model["trained_labels_through"] >= signal:
                issues.append("model_selection_labels_not_before_signal")
    sealed = read(root / "schedule.json")["sealed_holdout_start"]
    if mode == "replay" and horizon and horizon[-1] >= sealed:
        issues.append("sealed_holdout_replay_forbidden")
    if model["feature_names"] != panel["feature_names"]:
        issues.append("feature_schema_mismatch")
    info = {
        "mode": mode,
        "trade_date": trade_date,
        "signal_date": signal,
        "horizon_dates": horizon,
        "cutoff": cutoff.isoformat(),
        "source_root": str(root),
        "source_latest_calendar_date": dates[-1],
        "source_created_at": snapshot["created_at"],
        "source_manifest_sha256": file_hash(root / "snapshot/manifest.json"),
        "panel_manifest_sha256": file_hash(root / "panel-h5/manifest.json"),
        "source_dataset_id": snapshot["dataset_id"],
        "issues": issues,
        "availability_status": "retrospective_reconstruction_not_PIT"
        if mode == "replay"
        else "snapshot_availability_checked",
        "risk_evidence": "unverified",
        "news_evidence": "not_connected",
        "sealed_holdout_start": sealed,
    }
    return info


def load_input(root, model, info):
    root = Path(root)
    panel = read(root / "panel-h5/manifest.json")
    snap = read(root / "snapshot/manifest.json")
    if panel["dataset_id"] != snap["dataset_id"]:
        raise ValueError("Panel and snapshot identity mismatch")
    # Hash verification is not outcome access. No sample/label table is decoded here.
    for base, manifest in [(root / "panel-h5", panel), (root / "snapshot", snap)]:
        for rel, expected in manifest["files"].items():
            f = (base / rel).resolve()
            if not f.is_relative_to(base.resolve()) or file_hash(f) != expected:
                raise ValueError("Changed source snapshot")
    meta = pd.read_parquet(root / "snapshot/instruments.parquet").set_index("instrument_id")
    symbols = panel["instruments"]
    signal, day = info["signal_date"], info["trade_date"]
    t = panel["dates"].index(signal)
    values = np.load(root / "panel-h5/values.npy", mmap_mode="r")
    # Slice only information at or before S. Do not use future-labelled samples for eligibility.
    history = values[:, t - model["lookback"] + 1 : t + 1].copy()
    bars = pd.read_parquet(root / "snapshot/bars.parquet", filters=[("date", "==", signal)])
    if bars.duplicated(["instrument_id", "date"]).any():
        raise ValueError("Duplicate reference bars")
    reference = bars.set_index("instrument_id").reindex(symbols)
    metadata = meta.reindex(symbols)
    eligible = (metadata.listed_at <= signal) & (
        metadata.delisted_at.fillna("").eq("") | (metadata.delisted_at >= day)
    )
    valid_history = np.isfinite(history).all((1, 2))
    quote = reference[["open", "high", "low", "close", "factor", "volume", "amount"]].to_numpy(
        float
    )
    valid_quote = np.isfinite(quote).all(1) & (quote > 0).all(1)
    valid_quote &= (quote[:, 1] >= quote[:, :4].max(1)) & (quote[:, 2] <= quote[:, :4].min(1))
    valid_quote &= (
        reference.get("source_trade_status", pd.Series(1.0, index=reference.index)).ne(0).to_numpy()
    )
    if "current_member" in metadata:
        eligible &= metadata.current_member.fillna(False)
    good = eligible.to_numpy() & valid_history & valid_quote
    reason = np.select(
        [~eligible.to_numpy(), ~valid_quote, ~valid_history],
        [
            "inactive_or_not_listed",
            "signal_quote_missing_or_suspended",
            "past_feature_history_missing",
        ],
        default="available",
    )
    ledger = pd.DataFrame(
        {
            "instrument_id": symbols,
            "date": signal,
            "input_status": reason,
            "forecast_available": good,
            "risk_status": "unknown",
        }
    )
    ledger["name"] = metadata.get("name", pd.Series("", index=metadata.index)).fillna("").to_numpy()
    selected = ledger.loc[good].reset_index(drop=True)
    selected["date_index"] = t
    anchor = reference.loc[np.array(symbols)[good]].reset_index()
    # reindex retains the source date column; missing reference quotes were excluded above.
    return {
        "rows": selected,
        "ledger": ledger,
        "features": history[good, -1],
        "history": history[good],
        "reference": anchor.close.to_numpy(float),
        "anchor": anchor,
    }


def conditional_plan(pred, reference, policy):
    quote = np.exp(pred) * reference
    tick = policy["tick"]
    bid = np.floor(min(reference, quote[0, 2, 1]) / tick) * tick
    stop = np.floor(quote[1:, 2, 0].min() / tick) * tick
    target = np.floor(quote[4, 3, 1] / tick) * tick
    risk = bid - stop + bid * policy["cost_buffer_fraction"]
    reward = target - bid - bid * policy["cost_buffer_fraction"]
    ratio = reward / risk if risk > 0 else None
    candidate = 0 < stop < bid < target and ratio is not None and ratio >= policy["min_reward_risk"]
    return {
        "signal": "conditional_candidate" if candidate else "watch",
        "buy": round(bid, 8),
        "take_profit": round(target, 8),
        "stop": round(stop, 8),
        "reward_risk": ratio,
        "scenario_reward_fraction": reward / bid if bid > 0 else None,
        "scenario_risk_fraction": risk / bid if bid > 0 else None,
        "decision_reason": "hypothesis_conditions_met"
        if candidate
        else "median_target_insufficient_for_downside",
        "risk_status": "unknown",
        "quality_status": "not_approved_for_recommendations",
        "opening_condition": "cancel if opening price <= stop; otherwise do not pay above buy limit",
        "position_size": None,
        "actual_trade_recorded": False,
        "executable": False,
    }


def generate(root, package, store, trade_date, cutoff, mode="prospective", policy=None, now=None):
    root, package, store = Path(root).resolve(), Path(package).resolve(), Path(store).resolve()
    verify(package)
    model = read(package / "model.json")
    policy = dict(POLICY if policy is None else policy)
    if policy["tick"] <= 0 or policy["min_reward_risk"] < 0:
        raise ValueError("Invalid research rule")
    info = inspect_input(root, model, trade_date, cutoff, mode, now)
    code = source_code()
    series = digest({"model_id": model["model_id"], "policy": policy, "code": code, "mode": mode})[
        :20
    ]
    identity = {
        "series_id": series,
        "input": info,
        "model_id": model["model_id"],
        "policy": policy,
        "code": code,
    }
    run_id = digest(identity)[:24]
    target = store / "runs" / trade_date / run_id
    if target.exists():
        verify(target)
        return target
    store.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".daily-", dir=store))
    try:
        previous = (
            sorted((store / "runs" / trade_date).glob("*/run.json"))
            if (store / "runs" / trade_date).exists()
            else []
        )
        metadata = {
            **identity,
            "run_id": run_id,
            "created_at": utc_now(),
            "package": str(package),
            "supersedes_candidates": [read(f)["run_id"] for f in previous],
            "status": "blocked" if info["issues"] else "research_draft",
            "model_quality_passed": model["quality_passed"],
            "strategy_validated": False,
            "portfolio_return_available": False,
        }
        write_json(stage / "run.json", metadata)
        (stage / "code").mkdir()
        for name in code:
            shutil.copy2(Path(__file__).with_name(name), stage / "code" / name)
        if info["issues"]:
            (stage / "report.md").write_text(
                "# 盘前研究任务：数据或时点不满足要求\n\n"
                + f"目标日期：{trade_date}；来源日历截至 {info['source_latest_calendar_date']}。\n\n"
                + "\n".join("- " + r for r in info["issues"])
                + "\n\n本次没有生成价格预测或交易策略，不使用旧行情冒充当日输入。\n"
            )
            return commit_directory(stage, target)
        data = load_input(root, model, info)
        if not len(data["rows"]):
            metadata["status"] = "blocked"
            metadata["blocking_issues"] = ["no_usable_past_only_input"]
            write_json(stage / "run.json", metadata)
            data["ledger"].to_parquet(stage / "coverage.parquet", index=False)
            (stage / "report.md").write_text(
                "输入覆盖不足，没有生成预测。详见 coverage.parquet。\n"
            )
            return commit_directory(stage, target)
        data["ledger"].to_parquet(stage / "coverage.parquet", index=False)
        data["rows"].to_parquet(stage / "rows.parquet", index=False)
        data["anchor"].to_parquet(stage / "reference-bars.parquet", index=False)
        np.savez_compressed(
            stage / "input.npz",
            features=data["features"],
            history=data["history"],
            reference=data["reference"],
        )
        predictions = {
            name: adapter(name, package).predict(data["features"], data["history"])
            for name in [model["adapter"], "naive"]
        }
        records = []
        for name, pred in predictions.items():
            validate_forecast(pred, len(data["rows"]))
            np.save(stage / f"{name}-forecast.npy", pred)
            for i, row in enumerate(data["rows"].itertuples()):
                plan = conditional_plan(pred[i], data["reference"][i], policy)
                records.append(
                    {
                        **plan,
                        "model": name,
                        "instrument_id": row.instrument_id,
                        "name": getattr(row, "name", ""),
                        "report_date": trade_date,
                        "signal_date": info["signal_date"],
                        "buy_valid_until": trade_date,
                        "time_exit_date": info["horizon_dates"][-1],
                        "reference_close": data["reference"][i],
                    }
                )
        # A frozen price rule is a strategy control, not an extra forecast model.
        for i, row in enumerate(data["rows"].itertuples()):
            ref = data["reference"][i]
            bid = np.floor(ref * 0.99 / policy["tick"]) * policy["tick"]
            records.append(
                {
                    "model": "fixed_rule",
                    "instrument_id": row.instrument_id,
                    "signal": "conditional_candidate",
                    "buy": bid,
                    "take_profit": np.floor(bid * 1.03 / policy["tick"]) * policy["tick"],
                    "stop": np.floor(bid * 0.98 / policy["tick"]) * policy["tick"],
                    "name": getattr(row, "name", ""),
                    "report_date": trade_date,
                    "signal_date": info["signal_date"],
                    "buy_valid_until": trade_date,
                    "time_exit_date": info["horizon_dates"][-1],
                    "reference_close": ref,
                    "decision_reason": "frozen_control_minus1_plus3_minus2",
                    "executable": False,
                    "actual_trade_recorded": False,
                    "risk_status": "unknown",
                }
            )
        plans = pd.DataFrame(records)
        flagged = plans["name"].str.contains("ST|退", case=False, regex=True) & plans.model.ne(
            "fixed_rule"
        )
        plans.loc[flagged, "signal"] = "watch"
        plans.loc[flagged, "decision_reason"] = "current_name_risk_flag"
        plans.to_parquet(stage / "plans.parquet", index=False)
        plans.to_csv(stage / "plans.csv", index=False)
        write_json(
            stage / "generation-summary.json",
            {
                "cohort": len(data["ledger"]),
                "predicted": len(data["rows"]),
                "models": list(predictions),
                "conditions_met": plans.groupby("model")
                .signal.apply(lambda s: int(s.eq("conditional_candidate").sum()))
                .to_dict(),
                "quality_passed": model["quality_passed"],
                "recommendation_ready": False,
            },
        )
        primary = (
            plans.loc[
                plans.model.eq(model["adapter"])
                & ~plans["name"].str.contains("ST|退", case=False, regex=True)
            ]
            .sort_values(["reward_risk", "instrument_id"], ascending=[False, True])
            .head(policy["max_display"])
        )
        n_candidates = int(
            plans.loc[plans.model.eq(model["adapter"]), "signal"].eq("conditional_candidate").sum()
        )
        conclusion = (
            "今日策略：不新开仓，保留现金，等待出现满足收益风险条件的机会。"
            if n_candidates == 0
            else f"今日有 {n_candidates} 个条件候选，须核对发布后的价格、公告与风险状态后再决定是否参与。"
        )
        lines = [
            "# 五日条件策略：研究草案",
            "",
            conclusion,
            "",
            "历史回放，非当时真实发布的建议。"
            if mode == "replay"
            else (
                "盘中补发研究版：仅使用上一完整交易日行情；发布前的触价不追认为成交。"
                if mode == "intraday_research"
                else "前向研究草案，模型与策略尚未通过建议质量验收。"
            ),
            f"报告交易日：{trade_date}；信号收盘日：{info['signal_date']}；信息截止：{info['cutoff']}。",
            f"覆盖：{'、'.join(info['horizon_dates'])}；预测 {len(data['rows'])} / {len(data['ledger'])} 条身份。",
            "公告层尚未接入，风险状态未核验；不推定实际持仓，不给出准确股数。",
            "以下条件用于检验假设，分位数不是已经校准的真实概率。价格单位为人民币元。",
            "",
            "| 股票身份 | 研究状态 | 买入限价 | 目标价 | 风险失效价 | 情景收益风险比 |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for r in primary.itertuples():
            ratio = "无有效比值" if pd.isna(r.reward_risk) else f"{r.reward_risk:.2f}"
            state = "条件候选" if r.signal == "conditional_candidate" else "观察"
            lines.append(
                f"| {r.instrument_id} {r.name} | {state} | {r.buy:.2f} | {r.take_profit:.2f} | {r.stop:.2f} | {ratio} |"
            )
        row_index = {s: i for i, s in enumerate(data["rows"].instrument_id)}
        forecast_rows = []
        for symbol in primary.instrument_id:
            i = row_index[symbol]
            quotes = np.exp(predictions[model["adapter"]][i]) * data["reference"][i]
            for d, date in enumerate(info["horizon_dates"]):
                record = {"instrument_id": symbol, "date": date, "day_offset": d}
                for f, field in enumerate(["open", "high", "low", "close"]):
                    for j, q in enumerate([10, 50, 90]):
                        record[f"{field}_q{q}"] = float(quotes[d, f, j])
                forecast_rows.append(record)
        pd.DataFrame(forecast_rows).to_csv(stage / "display-forecasts.csv", index=False)
        lines += [
            "",
            "五个交易日的收盘价格区间（仅展示上表股票；完整 OHLC 见 display-forecasts.csv）：",
            "",
            "| 股票身份 | 交易日 | q10 | q50 | q90 |",
            "|---|---|---:|---:|---:|",
        ]
        lines += [
            f"| {r['instrument_id']} | {r['date']} | {r['close_q10']:.2f} | {r['close_q50']:.2f} | {r['close_q90']:.2f} |"
            for r in forecast_rows
        ]
        lines += [
            "",
            (
                "盘中补发条件：只观察发布后的价格；发布前出现过买点不追认，缺少发布后分时证据时不认定成交。"
                if mode == "intraday_research"
                else "统一条件：开盘不高于失效价时放弃买入；买入不高于限价，未触发当日失效。"
            ),
            "已有持仓的处理需要用户提供持仓与可卖数量；本草案不假定已经买入。",
            "触发后上涨：达到目标价时复核退出；横盘：到原 T+4 节点复核；下跌：失效价触发风险检查。",
            "触发价不保证成交，跳空、停牌或可卖限制可能扩大损失；新报告不覆盖原报告或自动延长原期限。",
            "",
            "全部股票条件保存在 plans.csv；逐日 OHLC 分位数保存在 forecast 数组；复盘将单独保存。",
        ]
        (stage / "report.md").write_text("\n".join(lines) + "\n")
        return commit_directory(stage, target)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def review(run, source, store, observed_through, now=None):
    run, source, store = Path(run).resolve(), Path(source).resolve(), Path(store).resolve()
    verify(run)
    original = read(run / "run.json")
    info = original["input"]
    if original["status"] != "research_draft":
        raise ValueError("Blocked run has no forecast to review")
    now = pd.Timestamp(now or utc_now()).tz_convert("Asia/Shanghai")
    if info["mode"] in {"prospective", "intraday_research"} and (
        observed_through > now.strftime("%Y-%m-%d")
        or (observed_through == now.strftime("%Y-%m-%d") and now.strftime("%H:%M") < "16:00")
    ):
        raise ValueError("Requested outcome day is not complete")
    horizon = info["horizon_dates"]
    end = min(observed_through, horizon[-1])
    if info["mode"] == "replay" and end >= info["sealed_holdout_start"]:
        raise ValueError("Sealed holdout cannot be scored by replay")
    manifest = read(source / "snapshot/manifest.json")
    if (
        info["mode"] in {"prospective", "intraday_research"}
        and pd.Timestamp(manifest["created_at"]) > now
    ):
        raise ValueError("Review snapshot is from the future")
    for name in ["bars.parquet", "actions.parquet"]:
        if file_hash(source / "snapshot" / name) != manifest["files"][name]:
            raise ValueError("Changed review source")
    review_id = digest(
        {
            "run_id": original["run_id"],
            "source": file_hash(source / "snapshot/manifest.json"),
            "observed_through": observed_through,
            "code": source_code(),
        }
    )[:24]
    target = store / "reviews" / original["run_id"] / review_id
    if target.exists():
        verify(target)
        return target
    rows = pd.read_parquet(run / "rows.parquet")
    anchor = pd.read_parquet(run / "reference-bars.parquet")
    future = pd.read_parquet(
        source / "snapshot/bars.parquet",
        filters=[("date", ">", info["signal_date"]), ("date", "<=", end)],
    )
    actions = pd.read_parquet(
        source / "snapshot/actions.parquet",
        filters=[("ex_date", ">", info["signal_date"]), ("ex_date", "<=", end)],
    )
    # Reference is frozen. Detect retrospective corrections rather than silently changing the denominator.
    fresh = pd.read_parquet(
        source / "snapshot/bars.parquet", filters=[("date", "==", info["signal_date"])]
    )
    old = anchor.set_index("instrument_id")
    new = fresh.set_index("instrument_id").reindex(old.index)
    changed = ~np.isclose(
        old[["open", "high", "low", "close", "factor"]].to_numpy(float),
        new[["open", "high", "low", "close", "factor"]].to_numpy(float),
        equal_nan=True,
    ).all(1)
    axis = [info["signal_date"]] + horizon
    local_rows = rows.assign(date_index=0, date=info["signal_date"])
    labels = price_labels(
        pd.concat([anchor, future], ignore_index=True), local_rows, axis, actions, "9999-12-31"
    )
    labels["valid"][changed] = False
    labels["targets"][changed] = np.nan
    plans = pd.read_parquet(run / "plans.parquet")
    ids = {s: i for i, s in enumerate(rows.instrument_id)}
    records = []
    assumptions = TradeAssumptions()
    for plan in plans.itertuples():
        i = ids[plan.instrument_id]
        state = "no_order"
        outcome = {"net_return": 0.0, "filled": False, "status": "no_order", "ambiguous": False}
        if plan.signal == "conditional_candidate" and info["mode"] == "intraday_research":
            outcome = {"net_return": None, "filled": None, "status": "unknown", "ambiguous": False}
            state = "unknown"
        elif plan.signal == "conditional_candidate":
            if labels["valid"][i, 0] and labels["future"][i, 0, 0] <= plan.stop:
                outcome = {
                    "net_return": 0.0,
                    "filled": False,
                    "status": "cancelled_at_open",
                    "ambiguous": False,
                }
            else:
                outcome = trade_diagnostic(
                    labels["future"][i],
                    labels["valid"][i],
                    Plan(plan.buy, plan.take_profit, plan.stop),
                    assumptions,
                    labels["upper"][i],
                    labels["lower"][i],
                )
            state = outcome["status"]
            if state == "unresolved":
                elapsed = np.array(horizon) <= end
                state = (
                    "unknown"
                    if end >= horizon[-1] or (~labels["valid"][i, elapsed]).any()
                    else "pending"
                )
        records.append(
            {
                "model": plan.model,
                "instrument_id": plan.instrument_id,
                "signal": plan.signal,
                "state": state,
                "hypothetical_net_return": outcome["net_return"],
                "hypothetical_filled": outcome["filled"],
                "intraday_order_ambiguous": outcome["ambiguous"],
                "reference_revised": bool(changed[i]),
                "actual_trade_recorded": False,
                "interpretation": "daily_bar_hypothesis_not_user_fill",
            }
        )
    outcomes = pd.DataFrame(records)
    metrics = []
    for model in read(run / "generation-summary.json")["models"]:
        pred = np.load(run / f"{model}-forecast.npy")
        for day in range(5):
            known = np.isfinite(labels["targets"][:, day, 3])
            y = labels["targets"][known, day, 3]
            q = pred[known, day, 3]
            e = y[:, None] - q
            metrics.append(
                {
                    "model": model,
                    "date": horizon[day],
                    "day_offset": day,
                    "known_rows": int(known.sum()),
                    "pending_day": horizon[day] > end,
                    "pinball": float(
                        np.maximum(
                            np.array([0.1, 0.5, 0.9]) * e, np.array([-0.9, -0.5, -0.1]) * e
                        ).mean()
                    )
                    if known.any()
                    else None,
                    "coverage80": float(((y >= q[:, 0]) & (y <= q[:, 2])).mean())
                    if known.any()
                    else None,
                }
            )
    comparisons = []
    for name, group in outcomes.groupby("model"):
        candidate = group.loc[group.signal.eq("conditional_candidate")]
        resolved = candidate.loc[~candidate.state.isin(["pending", "unknown"])]
        closed = candidate.loc[candidate.state.eq("closed")]
        comparisons.append(
            {
                "model": name,
                "input_rows": len(group),
                "candidate_rows": len(candidate),
                "resolved_candidate_rows": len(resolved),
                "closed_rows": len(closed),
                "unknown_rows": int(candidate.state.eq("unknown").sum()),
                "pending_rows": int(candidate.state.eq("pending").sum()),
                "ambiguous_rows": int(candidate.intraday_order_ambiguous.sum()),
                "known_candidate_mean_net": float(resolved.hypothetical_net_return.mean())
                if len(resolved)
                else None,
                "closed_mean_net": float(closed.hypothetical_net_return.mean())
                if len(closed)
                else None,
                "interpretation": "individual_hypotheses_not_portfolio_return",
            }
        )
    summary = {
        "created_at": utc_now(),
        "run_id": original["run_id"],
        "series_id": original["series_id"],
        "observed_through": observed_through,
        "horizon_complete": end >= horizon[-1],
        "source_manifest_sha256": file_hash(source / "snapshot/manifest.json"),
        "states": outcomes.groupby(["model", "state"])
        .size()
        .rename("count")
        .reset_index()
        .to_dict("records"),
        "reference_revisions": int(changed.sum()),
        "metrics": metrics,
        "comparisons": comparisons,
        "actual_fills_available": False,
        "portfolio_return_available": False,
        "assumptions": asdict(assumptions),
        "code": source_code(),
    }
    store.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".review-", dir=store))
    try:
        (stage / "code").mkdir()
        for name in summary["code"]:
            shutil.copy2(Path(__file__).with_name(name), stage / "code" / name)
        outcomes.to_parquet(stage / "outcomes.parquet", index=False)
        np.savez_compressed(stage / "labels.npz", **labels)
        write_json(stage / "review.json", summary)
        pd.DataFrame(metrics).to_csv(stage / "metrics.csv", index=False)
        pd.DataFrame(comparisons).to_csv(stage / "comparisons.csv", index=False)
        lines = [
            "# 研究草案复盘",
            "",
            f"原始计划：{original['run_id']}；观察至 {observed_through}。",
            "原计划、预测和期限均未修改。未完成日期标记等待，成熟但不可判定的结果保留未知。",
            "以下为日线条件诊断；触价、假设成交和用户真实成交严格区分。",
            "",
            "| 模型 | 状态 | 数量 |",
            "|---|---|---:|",
        ]
        lines += [f"| {r['model']} | {r['state']} | {r['count']} |" for r in summary["states"]]
        lines += [
            "",
            "对照包含 LightGBM、朴素预测条件规则，以及固定回撤 1% 买入、上涨 3% 目标、下跌 2% 失效规则。",
            "候选数、已知结果均值、等待与未知数量见 comparisons.csv；各组选择率不同，均值不能直接证明策略优劣。",
            "逐日收盘分位损失与覆盖见 metrics.csv，详细条件状态见 outcomes.parquet。",
            "复盘仅评估冻结草案，不重新选股、不修改当时买卖价、不将未知结果填零。",
        ]
        (stage / "report.md").write_text("\n".join(lines) + "\n")
        return commit_directory(stage, target)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
