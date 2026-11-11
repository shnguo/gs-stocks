import copy
import inspect
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

import quant_research.earnings_continuation as continuation
from quant_research.config import digest
from quant_research.earnings_fixture import create_earnings_bundle, seal_artificial_bundle
from quant_research.storage import file_hash

PROTOCOL = Path(__file__).parents[1] / "configs/earnings-drift-validation-20261012-v1.json"


def extend_input(source, output):
    shutil.copytree(source, output)
    spec = json.loads((output / "inputs.json").read_text())
    calendar = pd.read_parquet(output / "snapshot/calendar.parquet")
    dates = sorted(calendar.loc[calendar.is_open, "date"].unique())
    next_day = dates[dates.index(spec["end"]) + 1]
    freezes = pd.read_parquet(output / spec["tables"]["freezes"])
    freezes = pd.concat([freezes, pd.DataFrame([dict(date=next_day, decision_at=next_day + "T10:00:00Z")])], ignore_index=True)
    freezes.to_parquet(output / spec["tables"]["freezes"], index=False)
    model = pd.read_parquet(output / spec["tables"]["model_ranks"])
    extra = model.loc[model.date.eq(spec["end"])].assign(date=next_day, available_at=next_day + "T09:00:00Z")
    pd.concat([model, extra], ignore_index=True).to_parquet(output / spec["tables"]["model_ranks"], index=False)
    spec["end"] = next_day
    (output / "inputs.json").write_text(json.dumps(spec))
    seal_artificial_bundle(output)


@pytest.fixture(scope="module")
def replay(tmp_path_factory):
    root = tmp_path_factory.mktemp("earnings-continuation")
    create_earnings_bundle(root / "first", PROTOCOL)
    extend_input(root / "first", root / "second")
    first = continuation.continue_engineering_replay(root / "journal", root / "first")
    second = continuation.continue_engineering_replay(root / "journal", root / "second")
    return root, first, second


def test_next_session_keeps_all_36_prefixes_and_fills_the_previously_frozen_pair(replay):
    root, first, second = replay
    records = continuation.read_journal(root / "journal", expected_head=second["record_sha256"])
    assert len(records) == 2
    old = root / "journal/attempts" / first["attempt_id"]
    new = root / "journal/attempts" / second["attempt_id"]
    details = json.loads((new / "attempt.json").read_text())
    assert details["continuation"]["account_prefixes_verified"] == 36
    assert details["continuation"]["prior_sessions"] == 31 and details["continuation"]["current_sessions"] == 32
    pending = pd.read_parquet(old / "result/h20/base/E1/pending_entries.parquet")
    fills = pd.read_parquet(new / "result/h20/base/E1/fills.parquet")
    assert set(pending.trade_id) <= set(fills.trade_id)
    old_pairs = pd.read_parquet(old / "result/h20/frozen-pairs.parquet")
    new_pairs = pd.read_parquet(new / "result/h20/frozen-pairs.parquet")
    pd.testing.assert_frame_equal(old_pairs, new_pairs)
    assert not details["forward_timeliness_claim"] and not details["formal_profit_qualified"]


def test_history_revision_is_retained_as_rejected_and_does_not_promote_an_account(replay):
    root, _, second = replay
    bad = root / "history-revision"
    extend_input(root / "second", bad)
    spec = json.loads((bad / "inputs.json").read_text())
    bars_path = bad / spec["snapshot"] / "bars.parquet"
    bars = pd.read_parquet(bars_path)
    bars.loc[0, "close"] += .001
    bars.to_parquet(bars_path, index=False)
    manifest_path = bad / spec["snapshot"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["bars.parquet"] = file_hash(bars_path)
    manifest["dataset_id"] = digest({k: manifest[k] for k in ["schema_version", "declaration", "files"]})
    manifest_path.write_text(json.dumps(manifest))
    seal_artificial_bundle(bad)
    with pytest.raises(ValueError, match="Frozen history changed: bars"):
        continuation.continue_engineering_replay(root / "journal", bad)
    rows = continuation.read_journal(root / "journal")
    assert rows[-1]["status"] == "rejected_attempt"
    accepted = [row for row in rows if row["status"] == "engineering_replay_committed"]
    assert accepted[-1]["record_sha256"] == second["record_sha256"]
    rejected = root / "journal/attempts" / rows[-1]["attempt_id"]
    assert (rejected / "result/run-receipt.json").exists()
    assert not json.loads((rejected / "attempt.json").read_text())["formal_profit_qualified"]


def test_user_authorized_protocol_revision_still_requires_a_separate_journal(replay):
    root, _, _ = replay
    changed = root / "v2-revision"
    extend_input(root / "second", changed)
    shutil.copyfile(PROTOCOL.with_name("earnings-drift-validation-20261012-v2.json"), changed / "protocol.json")
    seal_artificial_bundle(changed)
    with pytest.raises(ValueError, match="Changed protocol"):
        continuation.continue_engineering_replay(root / "journal", changed)
    rows = continuation.read_journal(root / "journal")
    assert rows[-1]["status"] == "rejected_attempt"


@pytest.mark.parametrize("change", ["file", "extra_file", "sequence", "retained_head", "orphan"])
def test_committed_attempt_mutation_and_uncommitted_state_are_rejected(replay, tmp_path, change):
    root, _, second = replay
    store = tmp_path / "journal"
    shutil.copytree(root / "journal", store)
    rows = continuation.read_journal(store)
    attempt = store / "attempts" / rows[0]["attempt_id"]
    if change == "file":
        (attempt / "attempt.json").write_text("{}")
    elif change == "extra_file":
        (attempt / "injected.txt").write_text("new")
    elif change == "sequence":
        content = copy.deepcopy(rows)
        content[0]["sequence"] = 99
        (store / "journal.jsonl").write_text("\n".join(json.dumps(row) for row in content) + "\n")
    elif change == "retained_head":
        with pytest.raises(ValueError, match="externally retained head"):
            continuation.read_journal(store, expected_head="a" * 64)
        return
    else:
        (store / "attempts/uncommitted-attempt").mkdir()
        with pytest.raises(ValueError, match="Uncommitted attempt"):
            continuation.continue_engineering_replay(store, root / "second")
        return
    with pytest.raises(ValueError):
        continuation.read_journal(store, expected_head=second["record_sha256"])


@pytest.mark.parametrize("clock_case", ["timely", "primary_deadline", "naive_source", "future_source"])
def test_rejected_input_observation_uses_runtime_clock_and_never_claims_a_cash_day(tmp_path, monkeypatch, clock_case):
    root = tmp_path / "input"
    inputs = create_earnings_bundle(root, PROTOCOL)
    manifest_path = root / "snapshot/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["declaration"]["purpose"] = "training"
    path = root / "snapshot/bars.parquet"
    bars = pd.read_parquet(path)
    bars.loc[bars.date.le(inputs["end"])].to_parquet(path, index=False)
    manifest["files"]["bars.parquet"] = file_hash(path)
    manifest["dataset_id"] = digest({k: manifest[k] for k in ["schema_version", "declaration", "files"]})
    calendar = pd.read_parquet(root / "snapshot/calendar.parquet")
    dates = sorted(calendar.loc[calendar.is_open, "date"].unique())
    next_day = dates[dates.index(inputs["end"]) + 1]
    clock = next_day + "T01:00:00Z" if clock_case == "primary_deadline" else inputs["end"] + "T10:00:00Z"
    manifest["created_at"] = (inputs["end"] + "T10:00:01Z" if clock_case == "future_source"
                              else inputs["end"] + "T09:00:00" if clock_case == "naive_source"
                              else inputs["end"] + "T09:00:00Z")
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(continuation, "utc_now", lambda: clock)
    row = continuation.observe_unaccepted_inputs(tmp_path / "journal", root / "snapshot", PROTOCOL)
    details = json.loads((tmp_path / "journal/attempts" / row["attempt_id"] / "attempt.json").read_text())
    assert details["status"] == "real_input_observation_blocked"
    assert details["preparation_before_primary_deadline"] == (clock_case != "primary_deadline")
    assert details["recorded_before_primary_deadline"] == (clock_case != "primary_deadline")
    if clock_case in {"naive_source", "future_source"}:
        assert "source_creation_clock_missing_naive_or_after_observation" in details["blockers"]
    if clock_case == "primary_deadline":
        assert "outside_completed_close_to_0900_freeze_window" in details["blockers"]
    assert "training_snapshot_cannot_enter_account" in details["blockers"]
    assert details["candidate_state"] == "not_evaluated_source_rejected"
    assert details["account_state"] == "not_run" and not details["cash_day_claim"]
    assert not details["prospective_start_registered"]
    assert "now" not in inspect.signature(continuation.observe_unaccepted_inputs).parameters


def test_missing_journal_cannot_satisfy_a_retained_head(tmp_path):
    with pytest.raises(ValueError, match="externally retained head"):
        continuation.read_journal(tmp_path / "missing", expected_head="a" * 64)
