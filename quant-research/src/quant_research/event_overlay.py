"""Point-in-time event overlays for immutable token-model rankings."""
from __future__ import annotations

import ast
import json
import tempfile
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .daily_loop import commit_directory, digest, verify
from .storage import write_json

MODEL_COLUMNS = {
    "instrument_id",
    "expected_net_return",
    "return_q10",
    "model_disagreement",
}
RADAR_COLUMNS = {
    "event_id",
    "event_type",
    "scope",
    "instrument_id",
    "scheduled_date",
    "actual_date",
    "available_at",
    "status",
    "event_score",
    "score_complete",
    "radar_state",
    "confirmation_passed",
    "blockers",
}
ARM_FILES = {
    "control": ("control-ranking.csv", "control_rank"),
    "event_overlay": ("event-overlay-ranking.csv", "event_overlay_rank"),
    "confirmed_event": ("confirmed-event-ranking.csv", "confirmed_event_rank"),
}


def load_overlay_config(path: Path) -> dict[str, object]:
    return normalize_overlay_config(json.loads(Path(path).read_text()))


def normalize_overlay_config(value: Mapping[str, object]) -> dict[str, object]:
    allowed = {
        "version",
        "weights",
        "risk_weights",
        "minimum_event_score",
        "post_event_max_calendar_days",
        "hard_blockers",
        "top_n",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown event-overlay config fields: {sorted(unknown)}")
    result = {
        "version": value.get("version", "event-overlay-v1"),
        "weights": value.get("weights", {"model": 0.7, "event": 0.2, "risk": 0.1}),
        "risk_weights": value.get(
            "risk_weights", {"downside": 0.5, "agreement": 0.5}
        ),
        "minimum_event_score": value.get("minimum_event_score", 50.0),
        "post_event_max_calendar_days": value.get("post_event_max_calendar_days", 7),
        "hard_blockers": value.get("hard_blockers", []),
        "top_n": value.get("top_n", 20),
    }
    if not isinstance(result["version"], str) or not result["version"]:
        raise ValueError("event-overlay version must be a non-empty string")
    result["weights"] = _weights(
        result["weights"], {"model", "event", "risk"}, "weights"
    )
    result["risk_weights"] = _weights(
        result["risk_weights"], {"downside", "agreement"}, "risk_weights"
    )
    minimum = float(result["minimum_event_score"])
    if not np.isfinite(minimum) or not 0 <= minimum <= 100:
        raise ValueError("minimum_event_score must be between 0 and 100")
    result["minimum_event_score"] = minimum
    post_days = int(result["post_event_max_calendar_days"])
    if post_days < 0:
        raise ValueError("post_event_max_calendar_days must not be negative")
    result["post_event_max_calendar_days"] = post_days
    top_n = int(result["top_n"])
    if top_n < 1:
        raise ValueError("top_n must be positive")
    result["top_n"] = top_n
    blockers = result["hard_blockers"]
    if not isinstance(blockers, list) or not all(
        isinstance(item, str) and item for item in blockers
    ):
        raise ValueError("hard_blockers must be a list of non-empty strings")
    result["hard_blockers"] = sorted(set(blockers))
    return result


def _weights(value: object, expected: set[str], label: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} must contain exactly {sorted(expected)}")
    result = {key: float(value[key]) for key in sorted(expected)}
    if not all(np.isfinite(item) and item >= 0 for item in result.values()):
        raise ValueError(f"{label} must be finite and non-negative")
    if not np.isclose(sum(result.values()), 1.0, atol=1e-12, rtol=0):
        raise ValueError(f"{label} must sum to one")
    return result


def _iso_date(value: object, field: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an ISO date") from exc


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value in {1, 1.0, "1", "true", "True", "yes"}:
        return True
    if value in {0, 0.0, "0", "false", "False", "no", "", None}:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def _parse_blockers(value: object) -> tuple[str, ...]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ()
    if isinstance(value, (list, tuple, set)):
        return tuple(sorted({str(item) for item in value if str(item)}))
    text = str(value).strip()
    if not text or text in {"[]", "()"}:
        return ()
    parsed: object
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = [item for item in text.split("|") if item]
    if isinstance(parsed, str):
        parsed = [parsed]
    if not isinstance(parsed, (list, tuple, set)):
        raise ValueError("event blockers must be a list or pipe-delimited string")
    return tuple(sorted({str(item) for item in parsed if str(item)}))


def _model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = MODEL_COLUMNS - set(frame)
    if missing:
        raise ValueError(f"model ranking missing columns: {sorted(missing)}")
    result = frame.copy()
    result["instrument_id"] = result.instrument_id.astype(str)
    if result.instrument_id.eq("").any() or result.instrument_id.duplicated().any():
        raise ValueError("model ranking requires unique non-empty instrument_id values")
    numeric = ["expected_net_return", "return_q10", "model_disagreement"]
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(result[numeric].to_numpy(float)).all():
        raise ValueError("model ranking contains non-finite scores")
    if result.model_disagreement.lt(0).any():
        raise ValueError("model_disagreement must not be negative")
    if "rank" in result:
        source_rank = pd.to_numeric(result["rank"], errors="raise")
        if source_rank.lt(1).any() or source_rank.duplicated().any():
            raise ValueError("model rank must be unique and positive")
        result["model_rank"] = source_rank.astype(int)
    else:
        ordered = result.sort_values(
            ["expected_net_return", "instrument_id"], ascending=[False, True]
        ).index
        result["model_rank"] = pd.Series(
            np.arange(1, len(result) + 1), index=ordered
        ).sort_index()
    result["model_strength"] = result.expected_net_return.rank(
        method="average", pct=True
    )
    result["downside_strength"] = result.return_q10.rank(method="average", pct=True)
    result["agreement_strength"] = result.model_disagreement.rank(
        method="average", ascending=False, pct=True
    )
    return result


def _validate_timing(
    signal_date: str, horizon_dates: Sequence[str], decision_at: str
) -> tuple[date, list[date], pd.Timestamp]:
    signal = _iso_date(signal_date, "signal_date")
    horizons = [_iso_date(item, "horizon date") for item in horizon_dates]
    if not horizons or horizons != sorted(set(horizons)):
        raise ValueError("horizon_dates must be non-empty, unique and increasing")
    if horizons[0] <= signal:
        raise ValueError("forecast horizon must start after signal_date")
    decision = pd.Timestamp(decision_at)
    if decision.tzinfo is None:
        raise ValueError("decision_at must include a timezone")
    decision = decision.tz_convert("UTC")
    cutoff = pd.Timestamp(
        horizons[0].isoformat() + "T09:15:00", tz="Asia/Shanghai"
    ).tz_convert("UTC")
    if decision >= cutoff:
        raise ValueError("event overlay must be frozen before the first forecast session")
    return signal, horizons, decision


def _alignment(
    row: Mapping[str, object], signal: date, horizons: Sequence[date], post_days: int
) -> tuple[str, str, int | None, int | None]:
    status = str(row.get("status", ""))
    scheduled = _iso_date(row.get("scheduled_date"), "scheduled_date")
    if status == "completed":
        raw_actual = row.get("actual_date")
        actual = _iso_date(raw_actual, "actual_date") if raw_actual else scheduled
        age = (signal - actual).days
        if 0 <= age <= post_days:
            return "post_event", actual.isoformat(), None, -age
        return "outside_window", actual.isoformat(), None, -age
    if horizons[0] <= scheduled <= horizons[-1]:
        offset = next(
            (index + 1 for index, horizon in enumerate(horizons) if horizon >= scheduled),
            len(horizons),
        )
        return "future_window", scheduled.isoformat(), offset, (scheduled - signal).days
    return "outside_window", scheduled.isoformat(), None, (scheduled - signal).days


def _event_diagnostics(
    radar: pd.DataFrame,
    model_ids: set[str],
    signal: date,
    horizons: Sequence[date],
    decision: pd.Timestamp,
    cfg: Mapping[str, object],
) -> pd.DataFrame:
    missing = RADAR_COLUMNS - set(radar)
    if missing:
        raise ValueError(f"event radar missing columns: {sorted(missing)}")
    hard_blockers = set(cfg["hard_blockers"])
    records: list[dict[str, object]] = []
    for source in radar.to_dict("records"):
        row = dict(source)
        blockers = _parse_blockers(row.get("blockers"))
        alignment, event_date, offset, distance = _alignment(
            row, signal, horizons, int(cfg["post_event_max_calendar_days"])
        )
        instrument_id = str(row.get("instrument_id", ""))
        score_raw = row.get("event_score")
        try:
            score = float(score_raw)
        except (TypeError, ValueError):
            score = float("nan")
        score_complete = _as_bool(row.get("score_complete"))
        confirmation = _as_bool(row.get("confirmation_passed"))
        available = pd.Timestamp(str(row.get("available_at", "")))
        if available.tzinfo is None:
            raise ValueError("event available_at must include a timezone")
        known_at_decision = available.tz_convert("UTC") <= decision
        hard = sorted(set(blockers).intersection(hard_blockers))
        reason = "eligible"
        if str(row.get("scope")) != "instrument":
            reason = "non_instrument_scope"
        elif instrument_id not in model_ids:
            reason = "not_in_model_universe"
        elif not known_at_decision:
            reason = "not_available_at_decision"
        elif alignment == "outside_window":
            reason = "outside_model_window"
        elif str(row.get("status")) in {"cancelled", "postponed"}:
            reason = "inactive_event_status"
        elif str(row.get("radar_state")) == "avoid":
            reason = "radar_avoid"
        elif hard:
            reason = "hard_blocker"
        elif not score_complete or not np.isfinite(score):
            reason = "incomplete_event_score"
        elif score < float(cfg["minimum_event_score"]):
            reason = "event_score_below_minimum"
        elif str(row.get("status")) == "completed" and not confirmation:
            reason = "post_event_confirmation_pending"
        eligible = reason == "eligible"
        row.update({
            "instrument_id": instrument_id,
            "event_score": score,
            "event_score_normalized": score / 100 if np.isfinite(score) else np.nan,
            "blockers": "|".join(blockers),
            "hard_blockers": "|".join(hard),
            "known_at_decision": known_at_decision,
            "event_alignment": alignment,
            "aligned_event_date": event_date,
            "event_session_offset": offset,
            "event_calendar_distance": distance,
            "confirmation_passed": confirmation,
            "event_overlay_eligible": eligible,
            "event_overlay_reason": reason,
        })
        records.append(row)
    if records:
        return pd.DataFrame(records)
    result = radar.copy()
    empty_columns = {
        "event_score_normalized": float,
        "hard_blockers": object,
        "known_at_decision": bool,
        "event_alignment": object,
        "aligned_event_date": object,
        "event_session_offset": float,
        "event_calendar_distance": float,
        "event_overlay_eligible": bool,
        "event_overlay_reason": object,
    }
    for column, dtype in empty_columns.items():
        if column not in result:
            result[column] = pd.Series(dtype=dtype)
    return result


def _primary_events(frame: pd.DataFrame, *, confirmed: bool = False) -> pd.DataFrame:
    selected = frame.loc[frame.event_overlay_eligible].copy()
    if confirmed:
        selected = selected.loc[selected.confirmation_passed]
    if selected.empty:
        return selected
    counts = selected.groupby("instrument_id").size().rename("eligible_event_count")
    selected["confirmation_priority"] = selected.confirmation_passed.astype(int)
    selected = selected.sort_values(
        [
            "instrument_id",
            "confirmation_priority",
            "event_score",
            "event_calendar_distance",
            "event_id",
        ],
        ascending=[True, False, False, True, True],
    ).drop_duplicates("instrument_id", keep="first")
    selected = selected.merge(counts, on="instrument_id", validate="one_to_one")
    return selected.drop(columns="confirmation_priority")


def _rank_arm(
    model: pd.DataFrame,
    events: pd.DataFrame,
    cfg: Mapping[str, object],
    rank_column: str,
) -> pd.DataFrame:
    event_columns = [
        "instrument_id",
        "event_id",
        "event_type",
        "title",
        "status",
        "scheduled_date",
        "actual_date",
        "aligned_event_date",
        "event_alignment",
        "event_session_offset",
        "event_calendar_distance",
        "event_score",
        "event_score_normalized",
        "radar_state",
        "confirmation_passed",
        "reward_risk",
        "eligible_event_count",
        "blockers",
    ]
    available = [column for column in event_columns if column in events]
    if events.empty:
        result = model.iloc[:0].copy()
        for column in available:
            if column != "instrument_id":
                result[column] = pd.Series(dtype=events[column].dtype)
        result["risk_quality"] = pd.Series(dtype=float)
        result["ranking_score"] = pd.Series(dtype=float)
        result[rank_column] = pd.Series(dtype=int)
        return result
    result = model.merge(events[available], on="instrument_id", validate="one_to_one")
    risk_weights = cfg["risk_weights"]
    result["risk_quality"] = (
        float(risk_weights["downside"]) * result.downside_strength
        + float(risk_weights["agreement"]) * result.agreement_strength
    )
    weights = cfg["weights"]
    result["ranking_score"] = (
        float(weights["model"]) * result.model_strength
        + float(weights["event"]) * result.event_score_normalized
        + float(weights["risk"]) * result.risk_quality
    )
    result = result.sort_values(
        ["ranking_score", "model_rank", "instrument_id"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    result[rank_column] = np.arange(1, len(result) + 1)
    return result


def build_event_overlay(
    model_ranking: pd.DataFrame,
    radar: pd.DataFrame,
    *,
    signal_date: str,
    horizon_dates: Sequence[str],
    decision_at: str,
    config: Mapping[str, object],
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Return control, event-window and confirmed-event rankings without retraining."""
    cfg = normalize_overlay_config(config)
    model = _model_frame(model_ranking)
    signal, horizons, decision = _validate_timing(signal_date, horizon_dates, decision_at)
    diagnostics = _event_diagnostics(
        radar,
        set(model.instrument_id),
        signal,
        horizons,
        decision,
        cfg,
    )
    eligible = _primary_events(diagnostics)
    confirmed = _primary_events(diagnostics, confirmed=True)
    risk_weights = cfg["risk_weights"]
    control = model.sort_values(["model_rank", "instrument_id"]).reset_index(drop=True)
    control["risk_quality"] = (
        float(risk_weights["downside"]) * control.downside_strength
        + float(risk_weights["agreement"]) * control.agreement_strength
    )
    control["control_rank"] = np.arange(1, len(control) + 1)
    outputs = {
        "control": control,
        "event_overlay": _rank_arm(
            model, eligible, cfg, "event_overlay_rank"
        ),
        "confirmed_event": _rank_arm(
            model, confirmed, cfg, "confirmed_event_rank"
        ),
        "diagnostics": diagnostics.sort_values(
            ["event_overlay_eligible", "event_score", "event_id"],
            ascending=[False, False, True],
            na_position="last",
        ).reset_index(drop=True),
    }
    reasons = (
        diagnostics.event_overlay_reason.value_counts().sort_index().to_dict()
        if len(diagnostics)
        else {}
    )
    metadata = {
        "version": cfg["version"],
        "signal_date": signal.isoformat(),
        "horizon_dates": [item.isoformat() for item in horizons],
        "decision_at": decision.isoformat().replace("+00:00", "Z"),
        "model_rows": len(model),
        "radar_events": len(radar),
        "eligible_events": int(diagnostics.event_overlay_eligible.sum()) if len(diagnostics) else 0,
        "event_overlay_instruments": len(outputs["event_overlay"]),
        "confirmed_event_instruments": len(outputs["confirmed_event"]),
        "exclusion_reasons": reasons,
        "config": cfg,
        "config_sha256": digest(cfg),
        "scope": "research_only_shadow_rankings_no_trade_instruction",
    }
    return outputs, metadata


def render_overlay_report(
    outputs: Mapping[str, pd.DataFrame], metadata: Mapping[str, object]
) -> str:
    top_n = int(metadata["config"]["top_n"])
    text = "# 事件增强模型观察名单\n\n"
    text += (
        f"信号日：{metadata['signal_date']}。未来交易日："
        + "、".join(metadata["horizon_dates"])
        + f"。冻结时间：{metadata['decision_at']}。\n\n"
    )
    text += (
        f"纯模型 {metadata['model_rows']} 只；事件窗口合格 "
        f"{metadata['event_overlay_instruments']} 只；事件后确认 "
        f"{metadata['confirmed_event_instruments']} 只。\n\n"
    )
    text += "## 事件增强 Top 列表\n\n"
    text += _report_table(outputs["event_overlay"], "event_overlay_rank", top_n)
    text += "\n## 事件后确认 Top 列表\n\n"
    text += _report_table(outputs["confirmed_event"], "confirmed_event_rank", top_n)
    text += (
        "\n纯模型排序完整保存在 control-ranking.csv。事件增强结果是研究用影子排名，"
        "不改变模型检查点，不构成交易指令。\n"
    )
    return text


def _report_table(frame: pd.DataFrame, rank_column: str, top_n: int) -> str:
    if frame.empty:
        return "当前没有符合条件的事件标的。\n"
    text = (
        "| 排名 | 代码 | 名称 | 模型排名 | 事件 | 事件分 | 联合分 | 模型预期收益 |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
    )
    for row in frame.head(top_n).itertuples():
        name = getattr(row, "name", "")
        text += (
            f"| {getattr(row, rank_column)} | {row.instrument_id} | {name} | "
            f"{row.model_rank} | {row.event_type} | {row.event_score:.1f} | "
            f"{row.ranking_score:.4f} | {row.expected_net_return:.2%} |\n"
        )
    return text


def write_event_overlay_bundle(
    destination: Path,
    outputs: Mapping[str, pd.DataFrame],
    metadata: Mapping[str, object],
) -> Path:
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("event overlay outputs use a new directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".event-overlay-", dir=destination.parent))
    outputs["control"].to_csv(stage / ARM_FILES["control"][0], index=False)
    outputs["event_overlay"].to_csv(stage / ARM_FILES["event_overlay"][0], index=False)
    outputs["confirmed_event"].to_csv(
        stage / ARM_FILES["confirmed_event"][0], index=False
    )
    outputs["diagnostics"].to_csv(stage / "event-diagnostics.csv", index=False)
    write_json(stage / "metadata.json", dict(metadata))
    (stage / "report.md").write_text(
        render_overlay_report(outputs, metadata), encoding="utf-8"
    )
    return commit_directory(stage, destination)


def review_event_overlay(
    bundle: Path, realized: pd.DataFrame, *, top_n: int | None = None
) -> dict[str, object]:
    """Score mature shadow arms without changing the frozen publication."""
    bundle = Path(bundle)
    verify(bundle)
    required = {"instrument_id", "realized_net_return"}
    missing = required - set(realized)
    if missing:
        raise ValueError(f"realized outcomes missing columns: {sorted(missing)}")
    outcomes = realized[list(required)].copy()
    outcomes["instrument_id"] = outcomes.instrument_id.astype(str)
    if outcomes.instrument_id.eq("").any() or outcomes.instrument_id.duplicated().any():
        raise ValueError("realized outcomes require unique non-empty instrument_id values")
    outcomes["realized_net_return"] = pd.to_numeric(
        outcomes.realized_net_return, errors="raise"
    )
    metadata = json.loads((bundle / "metadata.json").read_text())
    limit = int(top_n or metadata["config"]["top_n"])
    if limit < 1:
        raise ValueError("top_n must be positive")
    arms: dict[str, object] = {}
    for arm, (filename, rank_column) in ARM_FILES.items():
        frame = pd.read_csv(bundle / filename)
        selected = frame.sort_values(rank_column).head(limit)
        merged = selected.merge(outcomes, on="instrument_id", how="left", validate="one_to_one")
        known = merged.realized_net_return.notna()
        values = merged.loc[known, "realized_net_return"]
        score_column = "ranking_score" if "ranking_score" in merged else "model_strength"
        rank_ic = None
        if known.sum() >= 3:
            scores = merged.loc[known, score_column]
            if scores.nunique() > 1 and values.nunique() > 1:
                rank_ic = float(scores.rank().corr(values.rank()))
        arms[arm] = {
            "published": len(frame),
            "selected": len(selected),
            "known": int(known.sum()),
            "coverage": float(known.mean()) if len(known) else None,
            "mean_realized_net_return": float(values.mean()) if len(values) else None,
            "median_realized_net_return": float(values.median()) if len(values) else None,
            "positive_fraction": float(values.gt(0).mean()) if len(values) else None,
            "rank_ic": rank_ic,
        }
    return {
        "version": metadata["version"],
        "signal_date": metadata["signal_date"],
        "top_n": limit,
        "arms": arms,
        "interpretation": (
            "descriptive shadow-arm comparison; differing candidate sets are not a causal estimate"
        ),
    }
