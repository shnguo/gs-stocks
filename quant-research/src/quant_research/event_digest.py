from __future__ import annotations

from datetime import date, timedelta

import pandas as pd


def _display_score(value: object) -> str:
    if value is None or pd.isna(value):
        return "待研究"
    return f"{float(value):.1f}"


def _display_blockers(value: object) -> str:
    if isinstance(value, list):
        items = value
    elif value is None or (isinstance(value, float) and pd.isna(value)):
        items = []
    else:
        items = [str(value)]
    return "、".join(str(item) for item in items) or "无"


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "暂无事件。"
    rows = ["| 日期 | 标的 | 事件 | 状态 | 分数 | 关键阻碍 |",
            "| --- | --- | --- | --- | ---: | --- |"]
    for row in frame.sort_values(["scheduled_date", "event_id"]).head(100).itertuples():
        values = row._asdict()
        rows.append("| {scheduled_date} | {instrument_id} | {title} | {radar_state} | "
                    "{score} | {blockers_display} |".format(
                        **values,
                        score=_display_score(values.get("event_score")),
                        blockers_display=_display_blockers(values.get("blockers")),
                    ))
    if len(frame) > 100:
        rows.append(f"\n仅展示前100条；该分组共{len(frame)}条。")
    return "\n".join(rows)


def render_daily_digest(radar: pd.DataFrame, radar_date: str, *,
                        completed_through: str | None = None) -> str:
    today = date.fromisoformat(radar_date)
    today_timestamp = pd.Timestamp(today)
    frame = radar.copy()
    if frame.empty:
        frame = pd.DataFrame(columns=["scheduled_date", "actual_date", "status", "event_id",
                                      "instrument_id", "title", "radar_state", "event_score",
                                      "blockers"])
    frame["scheduled_date"] = frame.scheduled_date.astype(str)
    scheduled = pd.to_datetime(frame.scheduled_date, errors="raise")
    actual = pd.to_datetime(frame.actual_date.replace("", pd.NA), errors="coerce")
    future_60 = frame.loc[(scheduled >= today_timestamp) &
                          (scheduled <= today_timestamp + timedelta(days=60))]
    next_10 = frame.loc[(scheduled >= today_timestamp) &
                        (scheduled <= today_timestamp + timedelta(days=10))]
    next_3 = frame.loc[(scheduled >= today_timestamp) &
                       (scheduled <= today_timestamp + timedelta(days=3))]
    post_5 = frame.loc[(actual >= today_timestamp - timedelta(days=5)) &
                       (actual <= today_timestamp)]
    state_counts = frame.radar_state.value_counts().to_dict() if len(frame) else {}
    lines = [
        f"# A股事件雷达 {radar_date}",
        "",
        "研究用途，不构成交易指令。事件、评分和技术位置必须分别验证。",
        "",
        f"价格完整收盘数据截至：{completed_through or '未提供'}",
        f"事件总数：{len(frame)}；观察：{state_counts.get('observe', 0)}；"
        f"等待确认：{state_counts.get('wait_confirmation', 0)}；"
        f"回避：{state_counts.get('avoid', 0)}。",
        "",
        "## 未来60天事件总览",
        "",
        _table(future_60),
        "",
        "## 未来10天重点研究",
        "",
        _table(next_10),
        "",
        "## 未来3天高风险窗口",
        "",
        _table(next_3),
        "",
        "## 事件后5天价格确认",
        "",
        _table(post_5),
        "",
    ]
    return "\n".join(lines)
