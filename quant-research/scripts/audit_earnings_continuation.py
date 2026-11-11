"""Persist CLI continuation checks and a current-clock rejected real-input observation."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.config import digest
from quant_research.daily_loop import read, verify
from quant_research.earnings_continuation import read_journal
from quant_research.earnings_fixture import create_earnings_bundle, seal_artificial_bundle
from quant_research.storage import file_hash, utc_now, write_json


def _extend(source, target):
    shutil.copytree(source, target)
    spec = read(target / "inputs.json")
    calendar = pd.read_parquet(target / "snapshot/calendar.parquet")
    dates = sorted(calendar.loc[calendar.is_open, "date"].unique())
    next_day = dates[dates.index(spec["end"]) + 1]
    freezes_path = target / spec["tables"]["freezes"]
    freezes = pd.read_parquet(freezes_path)
    pd.concat([freezes, pd.DataFrame([dict(date=next_day, decision_at=next_day + "T10:00:00Z")])],
              ignore_index=True).to_parquet(freezes_path, index=False)
    model_path = target / spec["tables"]["model_ranks"]
    model = pd.read_parquet(model_path)
    extra = model.loc[model.date.eq(spec["end"])].assign(date=next_day, available_at=next_day + "T09:00:00Z")
    pd.concat([model, extra], ignore_index=True).to_parquet(model_path, index=False)
    spec["end"] = next_day
    write_json(target / "inputs.json", spec)
    seal_artificial_bundle(target)


def audit(output, snapshot, protocol):
    output, snapshot, protocol = map(lambda p: Path(p).resolve(), [output, snapshot, protocol])
    if output.exists():
        raise FileExistsError("Audit destination must be new")
    if output.is_relative_to(snapshot):
        raise ValueError("Audit cannot alter its real source snapshot")
    output.mkdir(parents=True)
    project = Path(__file__).resolve().parents[1]
    cli = project / "scripts/record_earnings_day.py"
    original = verify(snapshot)
    original_manifest_hash = file_hash(snapshot / "manifest.json")

    def run(label, args, expected_code):
        with (output / (label + ".log")).open("w") as log:
            result = subprocess.run([sys.executable, str(cli), *args], cwd=project,
                                    stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode != expected_code:
            raise ValueError(label + " unexpected CLI exit; attempt preserved")

    create_earnings_bundle(output / "input-first", protocol)
    _extend(output / "input-first", output / "input-second")
    engineering = output / "engineering-journal"
    for label in ["first", "second"]:
        run(label, ["engineering-replay", "--store", str(engineering),
                    "--inputs", str(output / ("input-" + label))], 0)
    rows = read_journal(engineering)
    first, second = [engineering / "attempts" / row["attempt_id"] for row in rows]
    old_end = read(first / "input/inputs.json")["end"]
    new_end = read(second / "input/inputs.json")["end"]
    accounts = []
    for path in sorted((first / "result").glob("h*/**/summary.json")):
        relative = path.parent.relative_to(first / "result")
        before = pd.read_parquet(path.with_name("nav.parquet"))
        after = pd.read_parquet(second / "result" / relative / "nav.parquet")
        prefix = after.loc[after.date.le(old_end)].reset_index(drop=True)
        if before.date.tolist() != prefix.date.tolist():
            raise ValueError("Independent account prefix dates changed")
        for name in ["nav", "cash", "market_value", "receivables", "dividend_tax_reserve"]:
            if not np.array_equal(before[name].to_numpy(), prefix[name].to_numpy()):
                raise ValueError("Independent account prefix values changed: " + name)
        if not np.allclose(after.nav, after.cash + after.market_value + after.receivables
                           - after.dividend_tax_reserve, rtol=0, atol=1e-8):
            raise ValueError("Independent continued NAV identity failed")
        old_plan = pd.read_parquet(path.with_name("pending_entries.parquet"))
        new_selected = pd.read_parquet(second / "result" / relative / "selections.parquet")
        old_selected = pd.read_parquet(path.with_name("selections.parquet"))
        for planned in old_plan.itertuples():
            previous = old_selected.loc[old_selected.trade_id.eq(planned.trade_id)]
            current = new_selected.loc[new_selected.trade_id.eq(planned.trade_id)]
            for name in ["instrument_id", "budget", "requested_budget", "requested_shares", "entry_date", "decision_at"]:
                if previous[name].tolist() != current[name].tolist():
                    raise ValueError("Independent pending plan changed: " + name)
        accounts.append(dict(account=str(relative), old_sessions=len(before), new_sessions=len(after),
                             pending_plans_preserved=len(old_plan)))
    if len(accounts) != 36:
        raise ValueError("Expected every one of the 36 account prefixes")
    old_pairs = pd.read_parquet(first / "result/h20/frozen-pairs.parquet")
    new_pairs = pd.read_parquet(second / "result/h20/frozen-pairs.parquet")
    pd.testing.assert_frame_equal(old_pairs, new_pairs, check_exact=True)
    old_pending = pd.read_parquet(first / "result/h20/base/E1/pending_entries.parquet")
    fills = pd.read_parquet(second / "result/h20/base/E1/fills.parquet")
    if not len(old_pending) or not set(old_pending.trade_id) <= set(fills.trade_id):
        raise ValueError("Previously frozen E1 pending order did not fill in the artificial next session")
    result_counts = []
    for attempt in [first, second]:
        receipt = read(attempt / "result/run-receipt.json")
        for name, expected in receipt["output_files"].items():
            if file_hash(attempt / "result" / name) != expected:
                raise ValueError("Persisted experiment output changed")
        result_counts.append(len(receipt["output_files"]))
    second_head = rows[-1]["record_sha256"]
    write_json(output / "retained-engineering-head-before-rejection.json", dict(head=second_head, retained_at=utc_now()))

    _extend(output / "input-second", output / "input-history-revision")
    bars_path = output / "input-history-revision/snapshot/bars.parquet"
    bars = pd.read_parquet(bars_path)
    bars.loc[0, "close"] += .001
    bars.to_parquet(bars_path, index=False)
    manifest_path = bars_path.with_name("manifest.json")
    manifest = read(manifest_path)
    manifest["files"]["bars.parquet"] = file_hash(bars_path)
    manifest["dataset_id"] = digest({k: manifest[k] for k in ["schema_version", "declaration", "files"]})
    write_json(manifest_path, manifest)
    seal_artificial_bundle(output / "input-history-revision")
    run("history-revision-rejected", ["engineering-replay", "--store", str(engineering),
        "--inputs", str(output / "input-history-revision")], 1)
    records = read_journal(engineering)
    rejected = read(engineering / "attempts" / records[-1]["attempt_id"] / "attempt.json")
    accepted = [row for row in records if row["status"] == "engineering_replay_committed"]
    if accepted[-1]["record_sha256"] != second_head or rejected["reason"] != "Frozen history changed: bars":
        raise ValueError("Historical revision was not retained behind the original accepted head")

    observations = output / "real-observation-journal"
    run("actual-observation", ["observe-unaccepted", "--store", str(observations),
        "--snapshot", str(snapshot), "--protocol", str(protocol)], 0)
    observation_records = read_journal(observations)
    observation = read(observations / "attempts" / observation_records[-1]["attempt_id"] / "attempt.json")
    if (observation["account_state"] != "not_run" or observation["cash_day_claim"]
            or observation["prospective_start_registered"]):
        raise ValueError("Rejected actual inputs gained an account or a prospective start")
    if verify(snapshot) != original or file_hash(snapshot / "manifest.json") != original_manifest_hash:
        raise ValueError("Real source snapshot changed during the read-only observation")
    result = dict(schema_version=1, created_at=utc_now(), status="engineering_continuation_verified_real_sources_blocked",
        old_end=old_end, new_end=new_end, artificial_account_prefixes=accounts, output_file_counts=result_counts,
        frozen_pairs=len(old_pairs), previously_pending_E1_orders_filled=len(old_pending),
        rejected_history_revision_retained=True, last_accepted_engineering_head=second_head,
        current_engineering_journal_head=records[-1]["record_sha256"], actual_observation=observation,
        actual_observation_journal_head=observation_records[-1]["record_sha256"],
        real_source_bytes_preserved=True, formal_profit_qualified=False, formal_daily_runner_implemented=False,
        prospective_start_registered=False, complete_environment_registered=False,
        source_hashes={str(p): file_hash(p) for p in [protocol, Path(__file__), cli, snapshot / "manifest.json",
            engineering / "journal.jsonl", observations / "journal.jsonl"]},
        limitations=["Artificial prefixes and plan arithmetic provide engineering evidence only.",
            "Actual observation is always blocked; preparation and recording clocks cannot establish accepted strategy decisions.",
            "The retained head is a local audit copy, not an independent signed append-only witness."])
    write_json(output / "audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.output, args.snapshot, args.protocol)
    print(json.dumps(dict(status=result["status"], account_prefixes=len(result["artificial_account_prefixes"]),
                          actual_observation_status=result["actual_observation"]["status"], formal_profit_qualified=False)))


if __name__ == "__main__":
    main()
