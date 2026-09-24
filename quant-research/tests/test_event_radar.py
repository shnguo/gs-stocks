import json
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from quant_research.event_digest import render_daily_digest
from quant_research.event_radar import (
    build_radar,
    classify_event,
    completed_technical_snapshot,
    event_score,
)
from quant_research.event_sources import (
    cninfo_periodic_event,
    fetch_cninfo_periodic_rows,
    fetch_sse_periodic_rows,
    sse_periodic_event,
)
from quant_research.event_store import EventStore, normalize_event
from quant_research.event_study import event_study


def sample_event(**changes):
    event = {
        "event_id": "sse:600000:2026h1",
        "event_type": "earnings_report",
        "scope": "instrument",
        "instrument_id": "cn.xshg.600000",
        "title": "2026 half-year report",
        "scheduled_date": "2026-09-20",
        "actual_date": "",
        "published_at": "2026-09-01T08:00:00Z",
        "observed_at": "2026-09-01T08:05:00Z",
        "availability_basis": "verified_source_timestamp",
        "source_name": "Shanghai Stock Exchange",
        "source_url": "https://www.sse.com.cn/example",
        "source_tier": "primary_exchange",
        "status": "scheduled",
        "certainty": 23,
        "transmission": 20,
        "expectation_gap": 13,
        "flow_impact": 9,
        "price_confirmation": 10,
        "crowding_penalty": 2,
        "gap_penalty": 2,
        "liquidity_penalty": 1,
        "support_price": 99,
        "target_price": 120,
        "failure_price": 98,
        "price_basis": "qfq",
        "expected_metric": "net_profit_cny",
        "expected_value": 1_000_000_000,
        "actual_value": "",
        "unit": "CNY",
        "notes": "",
        "source_document_sha256": "",
        "expected_window_end": "",
    }
    event.update(changes)
    return event


def rising_bars(instrument_id="cn.xshg.600000", periods=30):
    dates = pd.bdate_range("2026-08-12", periods=periods)
    close = np.linspace(100, 105, periods)
    volume = np.full(periods, 1_000_000.0)
    volume[-1] = 1_200_000
    return pd.DataFrame({
        "instrument_id": instrument_id,
        "date": dates.strftime("%Y-%m-%d"),
        "open": close - 0.2,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": volume,
        "factor": 1.0,
    })


def test_event_schema_requires_explicit_availability_and_completed_date():
    with pytest.raises(ValueError, match="timezone offset"):
        normalize_event(sample_event(published_at="2026-09-01 08:00:00"))
    with pytest.raises(ValueError, match="completed events require"):
        normalize_event(sample_event(status="completed"))
    with pytest.raises(ValueError, match="between 0 and 25"):
        normalize_event(sample_event(certainty=26))
    with pytest.raises(ValueError, match="price_basis=qfq"):
        normalize_event(sample_event(price_basis=""))


def test_event_store_is_append_only_point_in_time_and_idempotent(tmp_path):
    store = EventStore(tmp_path / "events.sqlite")
    first = sample_event()
    result = store.ingest([first], source_path="first.json", source_sha256="a" * 64,
                          ingested_at="2026-09-01T09:00:00Z")
    assert result["inserted"] == 1
    repeated = store.ingest([first], source_path="first.json", source_sha256="a" * 64,
                            ingested_at="2026-09-02T09:00:00Z")
    assert repeated["inserted"] == 0 and repeated["idempotent"]

    revised = sample_event(scheduled_date="2026-09-25", status="confirmed",
                           published_at="2026-09-03T08:00:00Z",
                           observed_at="2026-09-03T08:02:00Z")
    store.ingest([revised], source_path="second.json", source_sha256="b" * 64,
                 ingested_at="2026-09-03T09:00:00Z")
    before = store.as_of("2026-09-02T00:00:00Z")
    after = store.as_of("2026-09-04T00:00:00Z")
    assert before[0]["scheduled_date"] == "2026-09-20"
    assert after[0]["scheduled_date"] == "2026-09-25"
    assert len(store.history(first["event_id"])) == 2
    audit = store.audit("2026-09-04T00:00:00Z")
    assert audit["events"] == 1 and audit["revisions"] == 2
    assert not audit["source_integrity_passed"]


def test_local_observation_prevents_backfilled_source_timestamp_leakage(tmp_path):
    store = EventStore(tmp_path / "events.sqlite")
    event = sample_event(availability_basis="local_observation",
                         observed_at="2026-09-10T08:00:00Z")
    store.ingest([event], source_path="archive.json", source_sha256="c" * 64,
                 ingested_at="2026-09-10T09:00:00Z")
    assert store.as_of("2026-09-09T23:59:59Z") == []
    assert len(store.as_of("2026-09-10T08:00:00Z")) == 1


def test_completed_technical_snapshot_uses_adjusted_completed_bars():
    bars = rising_bars()
    result = completed_technical_snapshot(bars, bars.date.iloc[-1]).iloc[0]
    expected_ma5 = (bars.close * bars.factor).iloc[-5:].mean()
    assert result.ma5 == pytest.approx(expected_ma5)
    assert result.above_ma5 and result.ma5_rising
    assert not result.overextended
    assert result.volume_ratio_20 == pytest.approx(1.2)


def test_event_score_and_post_event_confirmation():
    bars = rising_bars()
    technical = completed_technical_snapshot(bars, bars.date.iloc[-1]).iloc[0].to_dict()
    actual = bars.date.iloc[-3]
    event = normalize_event(sample_event(status="completed", actual_date=actual))
    scored = event_score(event)
    assert scored["event_score"] == pytest.approx(70)
    decision = classify_event(event, technical, bars.date.iloc[-1])
    assert decision["radar_state"] == "observe"
    assert decision["confirmation_passed"]
    assert decision["reward_risk"] >= 2


def test_event_radar_never_turns_pre_event_watch_into_buy_instruction():
    bars = rising_bars()
    as_of = bars.date.iloc[-1]
    scheduled = (pd.Timestamp(as_of).date() + timedelta(days=10)).isoformat()
    event = normalize_event(sample_event(scheduled_date=scheduled))
    radar = build_radar([event], bars, as_of, completed_through=as_of)
    assert radar.loc[0, "radar_state"] == "observe"
    assert "buy" not in json.dumps(radar.to_dict("records")).lower()


def test_stale_bars_cannot_pass_post_event_confirmation():
    bars = rising_bars()
    completed_through = bars.date.iloc[-1]
    radar_date = (pd.Timestamp(completed_through).date() + timedelta(days=10)).isoformat()
    event = normalize_event(sample_event(status="completed", actual_date=bars.date.iloc[-3],
                                         scheduled_date=bars.date.iloc[-3]))
    radar = build_radar([event], bars, radar_date, completed_through=completed_through,
                        past_days=20)
    assert radar.loc[0, "radar_state"] == "wait_confirmation"
    assert "price_data_stale" in radar.loc[0, "blockers"]
    assert not radar.loc[0, "confirmation_passed"]


def test_overdue_scheduled_event_waits_for_source_resolution():
    bars = rising_bars()
    as_of = bars.date.iloc[-1]
    overdue = (pd.Timestamp(as_of).date() - timedelta(days=2)).isoformat()
    event = normalize_event(sample_event(scheduled_date=overdue))
    radar = build_radar([event], bars, as_of, completed_through=as_of, past_days=5)
    assert radar.loc[0, "radar_state"] == "wait_confirmation"
    assert "scheduled_event_overdue" in radar.loc[0, "blockers"]


def test_post_event_study_enters_only_after_publication():
    bars = rising_bars(periods=30)
    scheduled = bars.date.iloc[10]
    event = normalize_event(sample_event(
        scheduled_date=scheduled,
        status="completed",
        actual_date=scheduled,
        published_at=f"{scheduled}T08:30:00Z",
        observed_at=f"{scheduled}T08:31:00Z",
    ))
    rows, summary = event_study([event], bars, mode="post", horizon_sessions=2)
    assert len(rows) == 1
    assert rows.iloc[0].entry_date == bars.date.iloc[11]
    assert rows.iloc[0].exit_date == bars.date.iloc[13]
    assert summary["execution_assumption"] == "adjusted_open_to_adjusted_open_without_costs"


def test_pre_event_study_excludes_events_not_known_by_entry_cutoff():
    bars = rising_bars(periods=30)
    scheduled = bars.date.iloc[10]
    event = normalize_event(sample_event(
        scheduled_date=scheduled,
        published_at=f"{bars.date.iloc[9]}T08:00:00Z",
        observed_at=f"{bars.date.iloc[9]}T08:01:00Z",
    ))
    rows, summary = event_study([event], bars, mode="pre", horizon_sessions=5)
    assert rows.empty
    assert summary["exclusions"]["event_not_known_before_entry"] == 1


def test_sse_periodic_transform_preserves_appointment_history_without_backdating():
    row = {
        "companyCode": "600000",
        "companyAbbr": "浦发银行",
        "publishYear": "2026",
        "bulletinType": "L012",
        "publishDate0": "2026-08-31",
        "publishDate1": "2026-08-29",
        "publishDate2": "2026-08-28",
        "publishDate3": "",
        "actualDate": "2026-08-28",
    }
    event = sse_periodic_event(row, observed_at="2026-09-22T08:00:00Z",
                               source_document_sha256="d" * 64)
    assert event["scheduled_date"] == "2026-08-28"
    assert event["actual_date"] == "2026-08-28"
    assert event["status"] == "completed"
    assert event["instrument_id"] == "cn.xshg.600000"
    assert event["availability_basis"] == "local_observation"
    assert event["published_at"] == "2026-09-22T08:00:00Z"


def test_sse_periodic_fetch_paginates_and_rejects_duplicate_business_keys():
    def fetch(_url, params):
        page = int(params["pageHelp.pageNo"])
        row = {"companyCode": f"60000{page}", "bulletinType": "L012",
               "publishYear": "2026"}
        return {"sqlId": "SSE_SZSGG_DQBGYYQK_CAST_NEW", "result": [row],
                "pageHelp": {"pageCount": 2}}

    result = fetch_sse_periodic_rows("2026", "L012", fetch=fetch)
    assert [row["companyCode"] for row in result["rows"]] == ["600001", "600002"]


def test_cninfo_periodic_transform_maps_shenzhen_identity_and_latest_change():
    row = {
        "seccode": "000001",
        "secname": "平安银行",
        "f001d_0102": "2026-06-30",
        "f002d_0102": "2026-08-15",
        "f003d_0102": "2026-08-16",
        "f004d_0102": "",
        "f005d_0102": "",
        "f006d_0102": "2026-08-16",
    }
    event = cninfo_periodic_event(row, market="sz", observed_at="2026-09-22T08:00:00Z",
                                  source_document_sha256="e" * 64)
    assert event["instrument_id"] == "cn.xshe.000001"
    assert event["scheduled_date"] == "2026-08-16"
    assert event["actual_date"] == "2026-08-16"
    assert event["source_tier"] == "designated_disclosure"
    assert event["availability_basis"] == "local_observation"


def test_cninfo_periodic_fetch_paginates():
    def fetch(_url, params):
        page = int(params["pagenum"])
        row = {"seccode": f"00000{page}", "f001d_0102": "2026-06-30"}
        return {"prbookinfos": [row], "totalPages": 2, "totalRows": 2}

    result = fetch_cninfo_periodic_rows("2026-06-30", market="sz", fetch=fetch)
    assert [row["seccode"] for row in result["rows"]] == ["000001", "000002"]


def test_daily_digest_has_all_four_operating_lists():
    radar = pd.DataFrame([{
        "scheduled_date": "2026-09-25",
        "actual_date": "",
        "status": "scheduled",
        "event_id": "one",
        "instrument_id": "cn.xshg.600000",
        "title": "test event",
        "radar_state": "observe",
        "event_score": 66,
        "blockers": [],
    }])
    text = render_daily_digest(radar, "2026-09-22", completed_through="2026-09-21")
    for heading in ["未来60天事件总览", "未来10天重点研究", "未来3天高风险窗口",
                    "事件后5天价格确认"]:
        assert heading in text
    assert "不构成交易指令" in text
