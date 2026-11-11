"""Publication-bound price diagnostics, independent of execution eligibility."""
from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .daily_loop import digest, read, verify, write_manifest
from .storage import file_hash, utc_now, write_json
from .universe import cutoff

FIELDS = {"open": "开盘价", "high": "最高价", "low": "最低价", "close": "收盘价"}
STATES = {"comparable": "可对照", "pending_close": "等待收盘", "price_missing": "缺少已收盘行情",
          "prediction_missing": "缺少冻结预测", "boundary_unknown": "复权或身份边界待核验",
          "inactive_session": "当日无有效成交行情", "anchor_missing": "缺少信号日价格"}
BOARD_LABELS = {"shanghai_main": "沪市主板", "shenzhen_main": "深市主板", "star": "科创板",
                "chinext": "创业板", "chi_next": "创业板", "beijing": "北交所", "北交所": "北交所"}


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def aware(value):
    result = pd.Timestamp(value)
    if result.tzinfo is None or pd.isna(result):
        raise ValueError("Analysis time must include a timezone")
    return result


def price_state(anchor, bar):
    if anchor is None or number(anchor.get("close")) is None or number(anchor.get("close")) <= 0:
        return "anchor_missing"
    if number(anchor.get("factor")) is None or number(anchor.get("factor")) <= 0:
        return "boundary_unknown"
    if bar is None:
        return "price_missing"
    if any(number(bar.get(field)) is None or number(bar.get(field)) <= 0 for field in FIELDS):
        return "price_missing"
    if number(bar.get("factor")) is None or number(bar.get("factor")) <= 0:
        return "boundary_unknown"
    trade_status = number(bar.get("source_trade_status"))
    if number(bar.get("volume")) is None:
        return "price_missing"
    if (number(bar.get("volume")) <= 0 or
            (trade_status is not None and trade_status != 1)):
        return "inactive_session"
    if not math.isclose(float(bar["factor"]), float(anchor["factor"]), rel_tol=1e-10, abs_tol=1e-12):
        return "boundary_unknown"
    for key in ("sequence_id", "label_sequence_id"):
        if key in anchor or key in bar:
            if pd.isna(anchor.get(key)) or pd.isna(bar.get(key)) or anchor.get(key) != bar.get(key):
                return "boundary_unknown"
    return "comparable"


def summarize(rows, field="close"):
    selected = [row for row in rows if row["field"] == field]
    known = [row for row in selected if row["status"] == "comparable"]
    errors = [(row["predicted"] / row["actual"] - 1) for row in known]
    bands = [row for row in known if row["q10"] is not None and row["q90"] is not None]
    directions = [row for row in known if row["anchor"] is not None]
    baseline_errors = [abs(row["anchor"] / row["actual"] - 1) for row in directions]
    return {"field": field, "label": FIELDS[field], "expected": len(selected), "known": len(known),
            "pending": sum(row["status"] == "pending_close" for row in selected),
            "missing": sum(row["status"] not in {"pending_close", "comparable"} for row in selected),
            "absolute_error_sum": sum(abs(error) for error in errors), "bias_sum": sum(errors),
            "mean_absolute_percentage_error": float(np.mean(np.abs(errors))) if errors else None,
            "mean_bias": float(np.mean(errors)) if errors else None,
            "baseline_known": len(baseline_errors), "baseline_absolute_error_sum": sum(baseline_errors),
            "baseline_mean_absolute_percentage_error": float(np.mean(baseline_errors)) if baseline_errors else None,
            "band_known": len(bands), "band_hits": sum(row["q10"] <= row["actual"] <= row["q90"] for row in bands),
            "band_hit_rate": sum(row["q10"] <= row["actual"] <= row["q90"] for row in bands) / len(bands) if bands else None,
            "direction_known": len(directions),
            "direction_hits": int(sum(np.sign(row["predicted"] - row["anchor"]) == np.sign(row["actual"] - row["anchor"]) for row in directions)),
            "direction_agreement": int(sum(np.sign(row["predicted"] - row["anchor"]) == np.sign(row["actual"] - row["anchor"]) for row in directions)) / len(directions) if directions else None}


def trend(history, signal, expected_dates=None):
    history = history[history.date.le(signal)].sort_values("date").tail(21)
    if history.empty or history.date.iloc[-1] != signal:
        return {"status": "anchor_missing", "return_5": None, "return_20": None, "ma20_distance": None}
    anchor = history.iloc[-1]
    valid = (len(history) == 21 and (expected_dates is None or history.date.tolist() == expected_dates) and
             all(price_state(anchor, row) == "comparable" for _, row in history.iterrows()))
    if not valid:
        return {"status": "boundary_unknown", "return_5": None, "return_20": None, "ma20_distance": None}
    close = history.close.to_numpy(float)
    return {"status": "comparable", "return_5": float(close[-1] / close[-6] - 1),
            "return_20": float(close[-1] / close[0] - 1), "ma20_distance": float(close[-1] / close[-20:].mean() - 1)}


def build_analysis_review(bundle, snapshot, now=None):
    bundle = Path(bundle)
    verify(bundle)
    decision = read(bundle / "decision.json")
    now = aware(now or utc_now())
    dates, signal = decision["horizon_dates"], decision["signal_date"]
    completed = [date for date in dates if now >= cutoff(date, "15:00")]
    identity = {"decision_sha256": file_hash(bundle / "manifest.json"),
                "outcome_manifest_sha256": file_hash(snapshot.path / "manifest.json"),
                "analysis_code_sha256": file_hash(Path(__file__)), "completed_sessions": completed}
    destination = bundle.parents[2] / "analysis-reviews" / signal / digest(identity)[:16]
    if destination.exists():
        verify(destination)
        return destination
    cohort = pd.read_csv(bundle / "cohort.csv").sort_values(["arm", "rank"])
    top = cohort.groupby("arm", sort=False).head(decision["protocol"]["top_n"])
    symbols = set(top.instrument_id)
    history = pd.read_parquet(bundle / "history.parquet")
    history = history[history.instrument_id.isin(symbols) & history.date.le(signal)]
    histories = {symbol: frame for symbol, frame in history.groupby("instrument_id")}
    calendar = snapshot.tables["calendar"]
    expected_history = sorted(set(calendar.loc[calendar.is_open.astype(bool) & calendar.date.le(signal), "date"]))[-21:]
    bars = snapshot.tables["bars"]
    bars = bars[bars.instrument_id.isin(symbols) & bars.date.isin(dates)]
    actual = {(row.instrument_id, row.date): row for _, row in bars.iterrows()}
    predictions = {}
    if (bundle / "daily-predictions.csv").exists():
        forecast = pd.read_csv(bundle / "daily-predictions.csv")
        forecast = forecast[forecast.instrument_id.isin(symbols) & forecast.field.isin(FIELDS)]
        if forecast.duplicated(["instrument_id", "date", "field"]).any():
            raise ValueError("Duplicate frozen daily prediction")
        for row in forecast.itertuples():
            if row.date not in dates or row.day != dates.index(row.date):
                raise ValueError("Frozen prediction date differs from its horizon")
            bounds = [number(getattr(row, key)) for key in ("q10", "q50", "q90")]
            known = [value for value in bounds if value is not None]
            if any(value <= 0 for value in known) or known != sorted(known):
                raise ValueError("Invalid frozen price quantiles")
            predictions[(row.instrument_id, row.date, row.field)] = bounds
    observations, arms = [], []
    for arm in decision["arms"]:
        candidates, arm_rows = [], []
        selected = top[top.arm.eq(arm["arm"])]
        for row in selected.itertuples():
            past = histories.get(row.instrument_id, pd.DataFrame(columns=history.columns))
            anchor_rows = past[past.date.eq(signal)]
            anchor = anchor_rows.iloc[-1] if len(anchor_rows) else None
            anchor_close = number(anchor.get("close")) if anchor is not None else None
            own, path = [], []
            for date in dates:
                bar = actual.get((row.instrument_id, date))
                base_state = "pending_close" if date not in completed else price_state(anchor, bar)
                point = {"date": date, "actual": number(bar.close) if bar is not None and base_state == "comparable" else None,
                         "status": base_state}
                for field in FIELDS:
                    q10, q50, q90 = predictions.get((row.instrument_id, date, field), [None, None, None])
                    status = "prediction_missing" if q50 is None and base_state == "comparable" else base_state
                    item = {"arm": arm["arm"], "instrument_id": row.instrument_id, "date": date, "field": field,
                            "status": status, "predicted": q50, "q10": q10, "q90": q90, "anchor": anchor_close,
                            "actual": number(bar[field]) if bar is not None and base_state == "comparable" else None}
                    own.append(item)
                    if field == "close":
                        point.update(predicted=q50, q10=q10, q90=q90, status=status)
                path.append(point)
            arm_rows.extend(own)
            features = trend(past, signal, expected_history)
            candidate = {key: getattr(row, key, None) for key in (
                "instrument_id", "name", "rank", "board", "industry", "expected_net_return",
                "return_q10", "return_q90", "model_disagreement", "volatility_20", "liquidity_20",
                "volatility_rank", "liquidity_rank", "early_stage_score", "risk_gate", "event_type", "event_score")}
            # CSV unknowns are encoded as null rather than JSON NaN.
            candidate = json.loads(pd.DataFrame([candidate]).to_json(orient="records"))[0]
            risks = []
            if getattr(row, "volatility_band", None) == "q5":
                risks.append("历史波动率位于股票池最高20%分组")
            if getattr(row, "liquidity_rank", 1) <= .2:
                risks.append("历史成交额位于股票池最低20%分组")
            if not candidate["industry"] or candidate["industry"] == "unknown":
                risks.append("信号日行业资料未知")
            if not decision["source_manifest"]["declaration"]["st_history_verified"]:
                risks.append("历史风险状态未核验")
            if anchor is not None and number(anchor.get("source_trade_status")) is None:
                risks.append("来源未提供信号日交易状态；对照使用完整量价")
            if features["status"] != "comparable":
                risks.append("历史趋势资料不完整或跨资料边界")
            candidate.update(trend=features, risks=risks, anchor_close=anchor_close, path=path,
                             quality=summarize(own), board_label=BOARD_LABELS.get(candidate["board"], candidate["board"] or "未知"))
            candidates.append(candidate)
        observations.extend(arm_rows)
        arms.append({"arm": arm["arm"], "label": arm["label"], "publication": arm["publication"],
                     "candidate_count": len(selected), "ranking_population": arm["count"],
                     "completed_sessions": len(completed), "total_sessions": len(dates),
                     "quality": summarize(arm_rows), "fields": [summarize(arm_rows, field) for field in FIELDS],
                     "horizons": [{"date": date, **summarize([item for item in arm_rows if item["date"] == date])}
                                  for date in dates],
                     "candidates": candidates,
                     "strata": {key: [{"group": str(group), "candidate_count": len(part),
                         **summarize([item for item in arm_rows if item["instrument_id"] in set(part.instrument_id)])}
                         for group, part in selected.groupby(key, dropna=False)]
                         for key in ("board", "industry", "volatility_band") if key in selected}})
    result = {**identity, "signal_date": signal, "horizon_dates": dates, "created_at": utc_now(),
              "observed_through": data_freshness(snapshot, now)["observed_through"],
              "synthetic": (snapshot.manifest["declaration"]["synthetic"] or
                            decision["source_manifest"]["declaration"]["synthetic"]), "arms": arms,
              "state_labels": STATES,
              "definitions": {"price_error": "模型参考价除以实测价格减1；平均绝对值为价格误差。",
                  "direction": "预测和实测价格相对信号日收盘的涨跌方向是否一致。",
                  "band": "冻结的各模型10%至90%分位值均值；其命中比例是描述性指标，不代表已校准的80%概率。",
                  "baseline": "朴素基线使用信号日收盘价预测后续价格，与模型使用相同可对照记录。",
                  "boundary": "对照要求完整量价及相同来源复权因子和价格序列；公司行动条款仍需独立核验。",
                  "sample": "每条样本为一个信号日、一只候选股票、一个预测交易日、一个价格字段；重叠预测窗口不是独立样本。"}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".analysis-", dir=destination.parent))
    try:
        write_json(staging / "analysis.json", result)
        pd.DataFrame(observations).to_parquet(staging / "price-comparisons.parquet", index=False)
        selected_predictions = pd.DataFrame([{"instrument_id": symbol, "date": date, "field": field,
                                             "q10": bounds[0], "q50": bounds[1], "q90": bounds[2]}
                                            for (symbol, date, field), bounds in predictions.items()])
        selected_predictions.to_parquet(staging / "frozen-price-predictions.parquet", index=False)
        history.to_parquet(staging / "frozen-history.parquet", index=False)
        write_manifest(staging)
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination


def aggregate_reports(reports):
    """Weight by forecast records, never average day-level rates or arm overlaps."""
    result = []
    for scope in ("prospective", "retrospective"):
        for arm in sorted({item["arm"] for report in reports for item in report["arms"]}):
            groups = [(report, item) for report in reports if not report["synthetic"] for item in report["arms"]
                      if item["arm"] == arm and item["publication"]["prospective"] == (scope == "prospective")]
            stats = [item["quality"] for _, item in groups]
            known = sum(row["known"] for row in stats)
            bands, directions = sum(row["band_known"] for row in stats), sum(row["direction_known"] for row in stats)
            baseline_known = sum(row["baseline_known"] for row in stats)
            result.append({"scope": scope, "arm": arm, "signal_dates": len(groups), "known": known,
                           "mean_absolute_percentage_error": sum(row["absolute_error_sum"] for row in stats) / known if known else None,
                           "mean_bias": sum(row["bias_sum"] for row in stats) / known if known else None,
                           "baseline_mean_absolute_percentage_error": sum(row["baseline_absolute_error_sum"] for row in stats) / baseline_known if baseline_known else None,
                           "band_hit_rate": sum(row["band_hits"] for row in stats) / bands if bands else None,
                           "direction_agreement": sum(row["direction_hits"] for row in stats) / directions if directions else None})
    return result


def data_freshness(snapshot, now=None):
    now = aware(now or utc_now())
    calendar = snapshot.tables["calendar"]
    closed = sorted(date for date in set(calendar.loc[calendar.is_open.astype(bool), "date"])
                    if now >= cutoff(date, "15:00"))
    bars = snapshot.tables["bars"]
    observed = sorted(set(bars.loc[bars.date.isin(closed) & bars.close.notna(), "date"]))
    expected, last = (closed[-1] if closed else None), (observed[-1] if observed else None)
    missing = [date for date in closed if last is None or date > last]
    return {"expected_through": expected, "observed_through": last, "missing_sessions": missing,
            "status": "current" if expected is not None and expected == last else "stale",
            "calendar_basis": snapshot.manifest["declaration"].get("calendar_basis", "snapshot_calendar")}


def refresh_analysis(store, source, config, overlays=(), output=None):
    from .execution_sources import public_observations
    from .research_review import freeze_decision, outcome_snapshot
    store = Path(store)
    snapshot = outcome_snapshot(Path(source) / "snapshot")
    decisions = {}
    for pointer in (store / "decision-publications").glob("*.json"):
        reference = read(pointer)
        bundle = Path(reference["bundle"])
        verify(bundle)
        if file_hash(bundle / "manifest.json") != reference["manifest_sha256"]:
            raise ValueError("Frozen analysis publication changed")
        decisions[pointer.stem] = bundle
    registry_path = store / "event-overlay-publications.json"
    registry = read(registry_path) if registry_path.exists() else {}
    seen = set()
    for overlay in map(Path, overlays):
        verify(overlay)
        signal = read(overlay / "metadata.json")["signal_date"]
        if signal in seen:
            raise ValueError("Provide one event overlay per signal date")
        seen.add(signal)
        registry[signal] = {"bundle": str(overlay.resolve()), "manifest_sha256": file_hash(overlay / "manifest.json")}
    for run in sorted((store / "runs").glob("*/run.json")):
        if not (store / "reference-publications" / (run.parent.name + ".json")).exists():
            continue
        active = []
        if run.parent.name in registry:
            reference = registry[run.parent.name]
            overlay = Path(reference["bundle"])
            verify(overlay)
            if file_hash(overlay / "manifest.json") != reference["manifest_sha256"]:
                raise ValueError("Registered overlay changed")
            active.append(overlay)
        bundle = freeze_decision(run.parent, config, active)
        decisions[run.parent.name] = bundle
        write_json(store / "decision-publications" / (run.parent.name + ".json"),
                   {"bundle": str(bundle.resolve()), "manifest_sha256": file_hash(bundle / "manifest.json")})
    if seen:
        write_json(registry_path, registry)
    reports, evidence = [], []
    for signal, bundle in sorted(decisions.items()):
        analysis = build_analysis_review(bundle, snapshot)
        reports.append({**read(analysis / "analysis.json"), "manifest_sha256": file_hash(analysis / "manifest.json")})
        evidence.extend([str(bundle), str(analysis)])
    operations = sorted((read(path) for path in store.glob("operations/*.json")), key=lambda row: row["started_at"])
    summary = {"schema_version": 1, "mode": "analysis", "generated_at": utc_now(),
               "engineering_only": snapshot.manifest["declaration"]["synthetic"],
               "observed_through": data_freshness(snapshot)["observed_through"], "freshness": data_freshness(snapshot),
               "analysis_reports": reports, "analysis_overview": aggregate_reports(reports),
               "source_observations": public_observations(store), "operations": operations[-10:],
               "latest": reports[-1] if reports else None,
               "prospective_mature_dates": len({report["signal_date"] for report in reports if not report["synthetic"]
                    for arm in report["arms"] if arm["publication"]["prospective"] and arm["quality"]["known"] == arm["quality"]["expected"] and arm["quality"]["expected"]}),
               "policy": "Analysis only. Late publications remain retrospective. Execution simulation is not run."}
    write_json(store / "research-latest.json", summary)
    if output:
        write_json(Path(output), summary)
    return {"summary": str(store / "research-latest.json"), "evidence": evidence,
            "prospective_mature_dates": summary["prospective_mature_dates"]}
