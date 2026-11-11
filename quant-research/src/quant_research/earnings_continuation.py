"""Immutable research attempts and prefix-preserving account continuation.

Engineering replay remains synthetic. Real input observations record current-clock
rejections, never cash accounts or a retrospective prospective start.
"""
from __future__ import annotations

import fcntl
import json
import os
import shutil
from pathlib import Path
from uuid import uuid4

import pandas as pd

from .config import digest
from .daily_loop import read, verify
from .earnings_runner import SUPPORTED_PROTOCOL_DIGESTS, run_bundle
from .storage import TABLES, file_hash, utc_now, write_json


def _hashes(root):
    return {str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*")) if p.is_file()}


def _check_files(root, files):
    if {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()} != set(files):
        raise ValueError("Committed attempt file set changed")
    for name, expected in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or file_hash(path) != expected:
            raise ValueError("Committed attempt file changed: " + name)


def read_journal(root, expected_head=None):
    root = Path(root).resolve()
    path = root / "journal.jsonl"
    rows, previous = [], None
    if not path.exists():
        if expected_head is not None:
            raise ValueError("Research journal differs from the externally retained head")
        return rows
    for line in path.read_text().splitlines():
        row = json.loads(line)
        content = {k: v for k, v in row.items() if k != "record_sha256"}
        if (row["sequence"] != len(rows) + 1 or row["previous_sha256"] != previous
                or digest(content) != row["record_sha256"]):
            raise ValueError("Research journal chain changed")
        attempt = root / "attempts" / row["attempt_id"]
        if not attempt.resolve().is_relative_to((root / "attempts").resolve()):
            raise ValueError("Attempt path escaped the journal")
        _check_files(attempt, row["files"])
        rows.append(row)
        previous = row["record_sha256"]
    if expected_head is not None and previous != expected_head:
        raise ValueError("Research journal differs from the externally retained head")
    return rows


def _same(old, new, label):
    if old.empty and new.empty:
        return
    if set(old.columns) != set(new.columns):
        raise ValueError("Frozen history changed: " + label)
    old, new = old.reindex(sorted(old.columns), axis=1), new.reindex(sorted(old.columns), axis=1)
    try:
        pd.testing.assert_frame_equal(old.reset_index(drop=True), new.reset_index(drop=True),
                                      check_exact=True, check_dtype=False)
    except AssertionError as error:
        raise ValueError("Frozen history changed: " + label) from error


def check_continuation(prior_input, prior_result, current_input, current_result):
    """Check old source observations, plans, pairs and every account's observed prefix."""
    old_spec, new_spec = read(prior_input / "inputs.json"), read(current_input / "inputs.json")
    old_summary, new_summary = read(prior_result / "research-summary.json"), read(current_result / "research-summary.json")
    old_dates, new_dates = old_summary["observed_dates"], new_summary["observed_dates"]
    if (old_spec["start"] != new_spec["start"] or len(new_dates) <= len(old_dates)
            or new_dates[:len(old_dates)] != old_dates):
        raise ValueError("Continuation must extend the same complete observed calendar")
    if (file_hash(prior_input / old_spec["protocol"]) != file_hash(current_input / new_spec["protocol"])
            or read(prior_result / "run-receipt.json")["runner_source_hashes"]
                != read(current_result / "run-receipt.json")["runner_source_hashes"]):
        raise ValueError("Changed protocol or runner code needs a separate registered version")
    end = old_dates[-1]
    for label, old_path, new_path in [("bars", prior_input / old_spec["snapshot"] / "bars.parquet",
                                      current_input / new_spec["snapshot"] / "bars.parquet"),
        *[(name, prior_input / old_spec["tables"][name], current_input / new_spec["tables"][name])
          for name in ["freezes", "model_ranks"]]]:
        old, new = pd.read_parquet(old_path), pd.read_parquet(new_path)
        _same(old.loc[old.date.le(end)], new.loc[new.date.le(end)], label)
    cutoff = pd.to_datetime(pd.read_parquet(prior_input / old_spec["tables"]["freezes"]).decision_at, utc=True).max()
    old_events = pd.read_parquet(prior_input / old_spec["tables"]["events"])
    new_events = pd.read_parquet(current_input / new_spec["tables"]["events"])
    _same(old_events.loc[pd.to_datetime(old_events.available_at, utc=True).le(cutoff)],
          new_events.loc[pd.to_datetime(new_events.available_at, utc=True).le(cutoff)], "already observed events")
    for relative in ["features.parquet", *[f"candidates/{arm}.parquet" for arm in ["P", "M", "E0", "E1"]],
                     "h20/frozen-pairs.parquet", "h5/frozen-pairs.parquet"]:
        old, new = pd.read_parquet(prior_result / relative), pd.read_parquet(current_result / relative)
        _same(old, new.loc[new.date.le(end)], relative)
    accounts = 0
    for path in sorted(prior_result.glob("h*/**/summary.json")):
        folder = path.parent.relative_to(prior_result)
        for name in ["nav", "fills", "orders", "holdings", "cash_events", "selections"]:
            old, new = pd.read_parquet(path.parent / (name + ".parquet")), pd.read_parquet(current_result / folder / (name + ".parquet"))
            _same(old, new.loc[new.date.le(end)] if "date" in new else new, str(folder / name))
        accounts += 1
    if accounts != 36:
        raise ValueError("Every one of the 36 prior account prefixes must be checked")
    return dict(prior_observed_end=end, current_observed_end=new_dates[-1],
                prior_sessions=len(old_dates), current_sessions=len(new_dates), account_prefixes_verified=accounts,
                terminal_plans_and_pairs_preserved=True, formal_profit_qualified=False)


def _archive_code(attempt):
    names = ["earnings_continuation.py", "earnings_runner.py", "earnings_ledger.py", "earnings_selection.py",
        "earnings_comparison.py", "earnings_validation.py", "storage.py", "daily_loop.py", "config.py",
        "event_radar.py", "universe.py", "backtest.py"]
    target = attempt / "code"
    target.mkdir()
    for name in names:
        shutil.copyfile(Path(__file__).with_name(name), target / name)


def _attempt(store, kind, operation):
    store = Path(store).resolve()
    store.mkdir(parents=True, exist_ok=True)
    (store / "attempts").mkdir(exist_ok=True)
    with (store / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = read_journal(store)
        committed = {r["attempt_id"] for r in previous}
        if any(p.name not in committed for p in (store / "attempts").iterdir()):
            raise ValueError("Uncommitted attempt exists; inspect its process and artifacts before recovery")
        identifier = uuid4().hex
        attempt = store / "attempts" / identifier
        attempt.mkdir()
        _archive_code(attempt)
        body = dict(schema_version=1, kind=kind, started_at=utc_now(), formal_profit_qualified=False,
                    formal_daily_runner_implemented=False, prospective_start_registered=False,
                    complete_environment_registered=False)
        error = None
        try:
            body.update(operation(attempt, previous))
        except Exception as exc:
            body.update(status="rejected_attempt", error_type=type(exc).__name__, reason=str(exc))
            error = exc
        body["recorded_at"] = utc_now()
        if kind == "unaccepted_real_input_observation" and "primary_deadline" in body:
            body["recorded_before_primary_deadline"] = bool(
                pd.Timestamp(body["recorded_at"]) < pd.Timestamp(body["primary_deadline"]))
        write_json(attempt / "attempt.json", body)
        record = dict(schema_version=1, sequence=len(previous) + 1, attempt_id=identifier,
            kind=kind, status=body["status"], recorded_at=body["recorded_at"],
            previous_sha256=previous[-1]["record_sha256"] if previous else None,
            files=_hashes(attempt), formal_profit_qualified=False)
        record["record_sha256"] = digest(record)
        with (store / "journal.jsonl").open("a") as journal:
            journal.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            journal.flush()
            os.fsync(journal.fileno())
        if error:
            raise error
        return record


def continue_engineering_replay(store, bundle):
    bundle = Path(bundle).resolve()
    if Path(store).resolve().is_relative_to(bundle):
        raise ValueError("Journal cannot modify its source bundle")

    def operation(attempt, records):
        verify(bundle)
        shutil.copytree(bundle, attempt / "input")
        frozen_input = attempt / "input"
        envelope = verify(frozen_input)
        if envelope != verify(bundle):
            raise ValueError("Source input changed while being archived")
        run_bundle(frozen_input, attempt / "result")
        if verify(frozen_input) != envelope or verify(bundle) != envelope:
            raise ValueError("Research input changed during execution")
        for name, expected in read(attempt / "result/run-receipt.json")["runner_source_hashes"].items():
            if file_hash(attempt / "code" / name) != expected:
                raise ValueError("Runner code changed after its attempt archive")
        accepted = [r for r in records if r["kind"] == "engineering_replay" and r["status"] == "engineering_replay_committed"]
        continuation = None
        if accepted:
            old = Path(store).resolve() / "attempts" / accepted[-1]["attempt_id"]
            if _hashes(old / "code") != _hashes(attempt / "code"):
                raise ValueError("Changed research orchestration needs a separate registered version")
            continuation = check_continuation(old / "input", old / "result", frozen_input, attempt / "result")
        return dict(status="engineering_replay_committed", continuation=continuation,
            evidence_role="synthetic_engineering_only", forward_timeliness_claim=False,
            account_run_receipt_sha256=file_hash(attempt / "result/run-receipt.json"))

    return _attempt(store, "engineering_replay", operation)


def observe_unaccepted_inputs(store, snapshot, protocol):
    snapshot, protocol = Path(snapshot).resolve(), Path(protocol).resolve()
    if Path(store).resolve().is_relative_to(snapshot):
        raise ValueError("Journal cannot modify its input snapshot")

    def operation(attempt, records):
        manifest = verify(snapshot)
        protocol_hash = file_hash(protocol)
        if set(manifest["files"]) != {name + ".parquet" for name in TABLES}:
            raise ValueError("Observation needs every required market table")
        shutil.copytree(snapshot, attempt / "input-snapshot")
        shutil.copyfile(protocol, attempt / "protocol.json")
        if manifest != verify(attempt / "input-snapshot") or manifest != verify(snapshot):
            raise ValueError("Source snapshot changed while being archived")
        if file_hash(attempt / "protocol.json") != protocol_hash or file_hash(protocol) != protocol_hash:
            raise ValueError("Protocol changed while being archived")
        config = read(attempt / "protocol.json")
        calendar = pd.read_parquet(attempt / "input-snapshot/calendar.parquet")
        dates = sorted(calendar.loc[calendar.is_open.astype(bool), "date"].unique())
        bars = pd.read_parquet(attempt / "input-snapshot/bars.parquet", columns=["date"])
        signal = bars.date.max()
        offset = dates.index(signal)
        if offset + 1 >= len(dates):
            raise ValueError("Source calendar lacks the next session")
        prepared = pd.Timestamp(utc_now())
        close = pd.Timestamp(signal + "T15:00:00", tz="Asia/Shanghai")
        deadline = pd.Timestamp(dates[offset + 1] + "T09:00:00", tz="Asia/Shanghai")
        timely = bool(close <= prepared < deadline)
        blockers = ["independent_source_acceptance_and_registration_required"]
        source_created = pd.to_datetime(manifest.get("created_at"), errors="coerce")
        if pd.isna(source_created) or source_created.tzinfo is None or source_created > prepared:
            blockers.append("source_creation_clock_missing_naive_or_after_observation")
        declaration = manifest["declaration"]
        if declaration.get("purpose") == "training":
            blockers.append("training_snapshot_cannot_enter_account")
        if digest(config) not in SUPPORTED_PROTOCOL_DIGESTS:
            blockers.append("protocol_semantics_not_implemented")
        if not config["costs"]["minimum_commission_user_confirmed"]:
            blockers.append("minimum_commission_unconfirmed")
        if not timely:
            blockers.append("outside_completed_close_to_0900_freeze_window")
        return dict(status="real_input_observation_blocked", input_dataset_id=manifest["dataset_id"],
            signal_date=signal, next_session=dates[offset + 1], clock_checked_at=prepared.isoformat(),
            source_created_at=manifest.get("created_at"),
            preparation_before_primary_deadline=timely, primary_deadline=deadline.isoformat(),
            hard_deadline=pd.Timestamp(dates[offset + 1] + "T09:15:00", tz="Asia/Shanghai").isoformat(),
            blockers=blockers, candidate_state="not_evaluated_source_rejected", account_state="not_run",
            cash_day_claim=False, actual_observation_only=True,
            formal_return_experiment_run=False, prior_committed_attempts=len(records))

    return _attempt(store, "unaccepted_real_input_observation", operation)
