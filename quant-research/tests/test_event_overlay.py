import json

import pandas as pd
import pytest

from quant_research.cli import main
from quant_research.event_overlay import (
    build_event_overlay,
    normalize_overlay_config,
    review_event_overlay,
    write_event_overlay_bundle,
)
from quant_research.storage import write_json

SIGNAL_DATE = "2026-09-22"
HORIZON_DATES = [
    "2026-09-23",
    "2026-09-24",
    "2026-09-25",
    "2026-09-28",
    "2026-09-29",
]
DECISION_AT = "2026-09-22T09:00:00Z"


def overlay_config(**changes):
    value = {
        "version": "test-overlay-v1",
        "weights": {"model": 0.7, "event": 0.2, "risk": 0.1},
        "risk_weights": {"downside": 0.5, "agreement": 0.5},
        "minimum_event_score": 50,
        "post_event_max_calendar_days": 7,
        "hard_blockers": [
            "price_overextended",
            "event_cancelled",
            "event_postponed",
            "first_post_event_close_pending",
        ],
        "top_n": 2,
    }
    value.update(changes)
    return value


def model_ranking():
    return pd.DataFrame([
        {
            "instrument_id": "cn.xshg.600001",
            "name": "A",
            "expected_net_return": 0.05,
            "return_q10": -0.02,
            "model_disagreement": 0.03,
            "rank": 3,
        },
        {
            "instrument_id": "cn.xshg.600002",
            "name": "B",
            "expected_net_return": 0.08,
            "return_q10": 0.01,
            "model_disagreement": 0.01,
            "rank": 2,
        },
        {
            "instrument_id": "cn.xshg.600003",
            "name": "C",
            "expected_net_return": 0.12,
            "return_q10": -0.04,
            "model_disagreement": 0.08,
            "rank": 1,
        },
    ])


def radar_event(instrument_id, event_id, **changes):
    row = {
        "event_id": event_id,
        "event_type": "earnings_report",
        "scope": "instrument",
        "instrument_id": instrument_id,
        "title": "test event",
        "scheduled_date": "2026-09-24",
        "actual_date": "",
        "available_at": "2026-09-22T08:00:00Z",
        "status": "scheduled",
        "event_score": 70,
        "score_complete": True,
        "radar_state": "observe",
        "confirmation_passed": False,
        "reward_risk": 2.5,
        "blockers": [],
    }
    row.update(changes)
    return row


def build(radar):
    return build_event_overlay(
        model_ranking(),
        pd.DataFrame(radar),
        signal_date=SIGNAL_DATE,
        horizon_dates=HORIZON_DATES,
        decision_at=DECISION_AT,
        config=overlay_config(),
    )


def test_future_events_align_to_model_window_and_hard_blocks_stay_out():
    outputs, metadata = build([
        radar_event("cn.xshg.600001", "eligible", event_score=90),
        radar_event(
            "cn.xshg.600002",
            "blocked",
            event_score=95,
            blockers=["price_overextended"],
        ),
        radar_event(
            "cn.xshg.600003",
            "too-late",
            scheduled_date="2026-10-10",
            event_score=100,
        ),
    ])
    assert outputs["control"].instrument_id.tolist() == [
        "cn.xshg.600003",
        "cn.xshg.600002",
        "cn.xshg.600001",
    ]
    assert outputs["event_overlay"].event_id.tolist() == ["eligible"]
    assert outputs["event_overlay"].event_session_offset.tolist() == [2]
    assert outputs["confirmed_event"].empty
    reasons = outputs["diagnostics"].set_index("event_id").event_overlay_reason
    assert reasons["blocked"] == "hard_blocker"
    assert reasons["too-late"] == "outside_model_window"
    assert metadata["event_overlay_instruments"] == 1


def test_completed_events_require_confirmation_and_get_a_separate_arm():
    outputs, metadata = build([
        radar_event("cn.xshg.600001", "future", event_score=85),
        radar_event(
            "cn.xshg.600002",
            "confirmed",
            status="completed",
            actual_date="2026-09-20",
            scheduled_date="2026-09-20",
            event_score=75,
            confirmation_passed=True,
        ),
        radar_event(
            "cn.xshg.600003",
            "pending",
            status="completed",
            actual_date="2026-09-21",
            scheduled_date="2026-09-21",
            confirmation_passed=False,
            radar_state="wait_confirmation",
        ),
    ])
    assert set(outputs["event_overlay"].event_id) == {"future", "confirmed"}
    assert outputs["confirmed_event"].event_id.tolist() == ["confirmed"]
    assert metadata["confirmed_event_instruments"] == 1
    reasons = outputs["diagnostics"].set_index("event_id").event_overlay_reason
    assert reasons["pending"] == "post_event_confirmation_pending"


def test_overlay_rejects_future_information_and_late_publication():
    outputs, _ = build([
        radar_event(
            "cn.xshg.600001",
            "late-information",
            available_at="2026-09-22T10:00:00Z",
        )
    ])
    assert outputs["event_overlay"].empty
    assert outputs["diagnostics"].event_overlay_reason.tolist() == [
        "not_available_at_decision"
    ]
    with pytest.raises(ValueError, match="before the first forecast session"):
        build_event_overlay(
            model_ranking(),
            pd.DataFrame([radar_event("cn.xshg.600001", "one")]),
            signal_date=SIGNAL_DATE,
            horizon_dates=HORIZON_DATES,
            decision_at="2026-09-23T01:15:00Z",
            config=overlay_config(),
        )


def test_csv_blockers_and_weight_validation_are_strict():
    outputs, _ = build([
        radar_event(
            "cn.xshg.600001",
            "csv-blockers",
            blockers="['price_overextended']",
        )
    ])
    assert outputs["event_overlay"].empty
    with pytest.raises(ValueError, match="sum to one"):
        normalize_overlay_config(
            overlay_config(weights={"model": 0.8, "event": 0.2, "risk": 0.2})
        )


def test_empty_radar_preserves_the_full_control():
    empty = pd.DataFrame(columns=[
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
    ])
    outputs, metadata = build_event_overlay(
        model_ranking(),
        empty,
        signal_date=SIGNAL_DATE,
        horizon_dates=HORIZON_DATES,
        decision_at=DECISION_AT,
        config=overlay_config(),
    )
    assert len(outputs["control"]) == 3
    assert outputs["event_overlay"].empty
    assert outputs["confirmed_event"].empty
    assert metadata["radar_events"] == 0


def test_overlay_bundle_is_immutable_and_reviewable(tmp_path):
    outputs, metadata = build([
        radar_event("cn.xshg.600001", "future-a", event_score=80),
        radar_event("cn.xshg.600002", "future-b", event_score=90),
    ])
    destination = tmp_path / "overlay"
    write_event_overlay_bundle(destination, outputs, metadata)
    assert (destination / "manifest.json").is_file()
    assert "不构成交易指令" in (destination / "report.md").read_text()
    with pytest.raises(FileExistsError):
        write_event_overlay_bundle(destination, outputs, metadata)
    realized = pd.DataFrame({
        "instrument_id": [
            "cn.xshg.600001",
            "cn.xshg.600002",
            "cn.xshg.600003",
        ],
        "realized_net_return": [0.04, -0.01, 0.02],
    })
    review = review_event_overlay(destination, realized)
    assert review["arms"]["control"]["known"] == 2
    assert review["arms"]["event_overlay"]["known"] == 2
    assert review["arms"]["confirmed_event"]["known"] == 0
    assert "not a causal estimate" in review["interpretation"]
    json.dumps(review, allow_nan=False)


def test_event_overlay_cli_writes_a_lineage_checked_bundle(tmp_path, monkeypatch, capsys):
    run = tmp_path / "run"
    run.mkdir()
    model_ranking().to_csv(run / "ranking.csv", index=False)
    write_json(run / "run.json", {
        "signal_date": SIGNAL_DATE,
        "horizon_dates": HORIZON_DATES,
    })
    radar = tmp_path / "radar.csv"
    pd.DataFrame([
        radar_event("cn.xshg.600001", "cli-event", event_score=88)
    ]).to_csv(radar, index=False)
    write_json(radar.with_suffix(".csv.json"), {"as_of": DECISION_AT})
    config = tmp_path / "config.json"
    write_json(config, overlay_config())
    output = tmp_path / "overlay-cli"
    monkeypatch.setattr("sys.argv", [
        "quant-research",
        "event-overlay",
        str(run),
        str(radar),
        "--overlay-config",
        str(config),
        "--output",
        str(output),
    ])
    main()
    response = json.loads(capsys.readouterr().out)
    assert response["event_overlay_instruments"] == 1
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["lineage"]["model_ranking_sha256"]
    assert pd.read_csv(output / "event-overlay-ranking.csv").event_id.tolist() == [
        "cli-event"
    ]
