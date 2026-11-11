"""Unified offline research orchestration; formal daily operation remains unimplemented.

Run all four arms, frozen matched controls, friction/delay/slippage scenarios and
the descriptive five-session comparison. Synthetic runs verify engineering only.
Current real inputs require independent source acceptance and a registered runner.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import digest
from .daily_loop import verify
from .earnings_comparison import paired_decisions, reconcile_paired_accounts
from .earnings_ledger import FEE_COLUMNS, simulate_earnings_account
from .earnings_selection import (
    frozen_selection_features,
    match_frozen_event_selections,
    select_earnings_arms,
)
from .earnings_validation import account_metrics, evaluate_research_diagnostics
from .storage import TABLES, Snapshot, file_hash, utc_now

SUPPORTED_PROTOCOL_DIGESTS = {
    "003b988630bb305d0c13fe2fc442be1946e5f3028b530b56eff9ec824b382817",  # legacy v1 replay
    "0d6bc910a850715f4f6a8e6ae3095429ae326f82acfc6a5f96fac045f3e59201",  # v2, sole 504-session look
}
INPUT_TABLES = {"freezes", "events", "industries", "capitalisations", "model_ranks", "event_coverage", "fees"}


def observed_window(snapshot, start, end):
    dates = snapshot.dates
    bars = snapshot.tables["bars"].date
    if (bars.empty or start not in dates or end not in dates or start >= end
            or start < bars.min() or end > bars.max()):
        raise ValueError("Research window needs at least two actually observed exchange sessions")
    return [d for d in dates if start <= d <= end]


def _executable_decisions(frame, dates, end, delay):
    index = {d: i for i, d in enumerate(dates)}
    executable = frame.date.map(lambda d: index[d] + 1 + delay < len(dates)
                               and dates[index[d] + 1 + delay] <= end).astype(bool)
    pending = frame.loc[~executable].copy()
    pending["pending_reason"] = "entry_session_not_yet_in_observed_window"
    return frame.loc[executable].copy(), pending


def _options(protocol, horizon):
    p = protocol["portfolio"]
    return dict(capital=protocol["capital_cny"], horizon=horizon,
        max_positions=p["maximum_positions"], slot_weight=p["maximum_initial_weight_per_position"],
        gross_weight=p["maximum_gross_weight"], industry_weight=p["maximum_industry_weight"],
        adv_participation=p["adv_participation"], drawdown_pause=p["research_drawdown_pause"])


def run_earnings_research(snapshot, freezes, events, industries, capitalisations, model_ranks,
                          event_coverage, fees, protocol, start, end):
    """Offline engineering stage of the full protocol, never a registered profit test.

    Allocation uses each full account's causal past holdings. Matching consumes every
    base E1 selection made before its opening outcome, not its successful fills. The
    resulting pair identities/budgets are held fixed across primary cost sensitivities.
    Terminal candidate decisions are retained as pending rather than silently deleted.
    """
    declaration = snapshot.manifest["declaration"]
    if declaration.get("purpose") == "training":
        raise ValueError("Training-only data cannot enter the unified account experiment")
    if declaration.get("synthetic") is not True:
        raise ValueError("Real-input acceptance and registered daily runner remain required; real-return execution is blocked")
    if digest(protocol) not in SUPPORTED_PROTOCOL_DIGESTS:
        raise ValueError("Only implemented v1/v2 protocol semantics are supported; changed rules need implementation and registration")
    dates = observed_window(snapshot, start, end)
    if {"date", "decision_at"} - set(freezes) or freezes.date.tolist() != dates:
        raise ValueError("Freeze calendar must retain every observed close, including no-signal and terminal days")
    if snapshot.dates.index(end) + 2 >= len(snapshot.dates):
        raise ValueError("Calendar needs two next sessions to freeze terminal base and delayed plans")
    if FEE_COLUMNS - set(fees):
        raise ValueError("Missing independently supplied effective fee schedule")
    if (not np.isfinite(fees[["commission_rate", "minimum_commission", "slippage_bps"]].to_numpy(float)).all()
            or not fees.commission_rate.eq(protocol["costs"]["broker_commission_rate"]).all()
            or not fees.minimum_commission.eq(protocol["costs"]["minimum_commission_cny_placeholder"]).all()
            or not fees.slippage_bps.eq(protocol["costs"]["base_slippage_bps_each_side"]).all()):
        raise ValueError("Base fee inputs differ from the frozen commission/minimum/slippage assumptions")
    feature_frames = [frozen_selection_features(snapshot, row.date, row.decision_at, industries, capitalisations)
                      for row in freezes.itertuples()]
    features = pd.concat(feature_frames, ignore_index=True)
    if features.empty:
        raise ValueError("No observable instrument cross-sections; missing input is not a cash signal")
    model = model_ranks.loc[model_ranks.date.isin(dates)].copy()
    candidates = select_earnings_arms(features, events, snapshot.dates, model_ranks=model)
    eligible_keys = set(map(tuple, features.loc[features.confirmation_eligible, ["date", "instrument_id"]].to_numpy()))
    supplied_keys = set(map(tuple, model[["date", "instrument_id"]].to_numpy()))
    if not eligible_keys <= supplied_keys or candidates["model_exclusions"]:
        raise ValueError("Missing, late or invalid frozen M ranks cannot be interpreted as a cash control")
    arms = candidates["arms"]
    accounts, pending, comparisons, matches_by_horizon = {}, {}, {}, {}
    settings = [("base", 10., 1., 0), ("stress", 10., 2., 0),
                ("delayed", 10., 1., 1), ("slippage_5", 5., 1., 0), ("slippage_20", 20., 1., 0)]
    for horizon in [20, 5]:
        horizon_accounts, horizon_pending = {}, {}
        selected_settings = settings if horizon == 20 else settings[:1]
        for name, slippage, multiplier, delay in selected_settings:
            scenario_fees = fees.copy()
            scenario_fees["slippage_bps"] = slippage
            scenario, waiting = {}, {}
            for arm, frame in arms.items():
                _, waiting[arm] = _executable_decisions(frame, snapshot.dates, end, delay)
                scenario[arm] = simulate_earnings_account(snapshot, frame, scenario_fees, industries,
                    start, end, **_options(protocol, horizon), friction_multiplier=multiplier, delay_sessions=delay)
            if name == "base":
                matched = match_frozen_event_selections(scenario["E1"]["selections"], features, events,
                                                       event_coverage, snapshot.dates)
                matches_by_horizon[horizon] = matched
                paired = paired_decisions(matched["pairs"], features)
            else:
                # Never rematch after seeing stress, delayed, or alternative-slippage outcomes.
                paired = paired_decisions(matches_by_horizon[horizon]["pairs"], features)
            for arm, frame in paired.items():
                _, waiting[arm] = _executable_decisions(frame, snapshot.dates, end, delay)
                scenario[arm] = simulate_earnings_account(snapshot, frame, scenario_fees, industries,
                    start, end, **_options(protocol, horizon), friction_multiplier=multiplier, delay_sessions=delay)
            horizon_accounts[name], horizon_pending[name] = scenario, waiting
            comparison = reconcile_paired_accounts(scenario["E1_matched"], scenario["P_matched_exposure"],
                                                    matches_by_horizon[horizon]["pairs"])
            comparisons[horizon, name] = comparison
        accounts[horizon], pending[horizon] = horizon_accounts, horizon_pending
    main = accounts[20]
    scenarios = {name: {"full": main[name]["E1"], "treated": main[name]["E1_matched"],
                        "control": main[name]["P_matched_exposure"]} for name in ["base", "stress", "delayed"]}
    diagnostics = evaluate_research_diagnostics(scenarios, matches_by_horizon[20]["pairs"], arms["E1"], dates, protocol)
    metrics = {str(h): {name: {arm: account_metrics(account, dates) for arm, account in rows.items()}
                        for name, rows in scenarios.items()} for h, scenarios in accounts.items()}
    return {"status": "synthetic_offline_research_complete_not_formal_acceptance",
            "protocol_version": protocol["version"],
            "single_formal_look_sessions": protocol["validation"]["single_formal_look_sessions"],
            "formal_profit_qualified": False, "formal_return_experiment_run": False,
            "formal_daily_runner_implemented": False, "prospective_start_registered": False,
            "features": features, "event_audit": candidates["event_audit"], "candidates": arms,
            "accounts": accounts, "pending": pending, "matches": matches_by_horizon,
            "comparisons": comparisons, "primary_diagnostics": diagnostics, "account_metrics": metrics,
            "observed_dates": dates, "freeze_rows": len(freezes),
            "limits": ["Synthetic inputs verify engineering, not any realized strategy return.",
                "This is offline orchestration; source acceptance, immutable real-time decisions and single-look registration remain required.",
                "Terminal selections, budgets and matches freeze before future fills; pending orders have no invented executions.",
                "Five-session accounts and alternative slippage scenarios are descriptive and cannot promote a strategy."]}


def _input_file(bundle, relative, declared):
    path = (bundle / relative).resolve()
    if not path.is_relative_to(bundle) or relative not in declared:
        raise ValueError("Every research input must be inside and hashed by its bundle")
    return path


def run_bundle(bundle: Path, output: Path):
    """Immutable artifact entry point, with rejected attempts preserved and never retried."""
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Experiment output must be new")
    if output.is_relative_to(bundle):
        raise ValueError("Experiment output cannot modify its input bundle")
    envelope = verify(bundle)
    spec = json.loads(_input_file(bundle, "inputs.json", envelope["files"]).read_text())
    if spec.get("schema_version") != 1 or set(spec["tables"]) != INPUT_TABLES:
        raise ValueError("Incomplete research experiment input specification")
    protocol_path = _input_file(bundle, spec["protocol"], envelope["files"])
    snapshot_path = (bundle / spec["snapshot"]).resolve()
    if not snapshot_path.is_relative_to(bundle):
        raise ValueError("Snapshot must remain inside the research input bundle")
    required_snapshot_paths = [spec["snapshot"] + "/manifest.json", *[spec["snapshot"] + "/" + name + ".parquet" for name in TABLES]]
    if not set(required_snapshot_paths) <= envelope["files"].keys():
        raise ValueError("Bundle must bind the canonical snapshot identity and all tables")
    paths = {name: _input_file(bundle, value, envelope["files"]) for name, value in spec["tables"].items()}
    output.mkdir(parents=True)
    runner_sources = {name: file_hash(Path(__file__).with_name(name)) for name in [
        "earnings_runner.py", "earnings_selection.py", "earnings_ledger.py", "earnings_comparison.py",
        "earnings_validation.py", "event_radar.py", "universe.py", "backtest.py", "storage.py", "config.py", "daily_loop.py"]}
    receipt = {"schema_version": 1, "started_at": utc_now(), "formal_profit_qualified": False,
        "formal_return_experiment_run": False, "formal_daily_runner_implemented": False,
        "input_manifest_sha256": file_hash(bundle / "manifest.json"), "input_files": envelope["files"],
        "runner_source_hashes": runner_sources, "complete_environment_registered": False}
    try:
        snapshot = Snapshot(snapshot_path)
        # Reject training/real sources before converting them into performance evidence.
        if snapshot.manifest["declaration"].get("purpose") == "training":
            raise ValueError("Training-only data cannot enter the unified account experiment")
        tables = {name: pd.read_parquet(path) for name, path in paths.items()}
        result = run_earnings_research(snapshot, **tables, protocol=json.loads(protocol_path.read_text()),
                                       start=spec["start"], end=spec["end"])
        def save_frame(relative, frame):
            path = output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
        save_frame("features.parquet", result["features"])
        save_frame("event-audit.parquet", result["event_audit"])
        for arm, frame in result["candidates"].items():
            save_frame(f"candidates/{arm}.parquet", frame)
        for horizon, scenarios in result["accounts"].items():
            for name, arms in scenarios.items():
                for arm, account in arms.items():
                    for table, frame in account.items():
                        if isinstance(frame, pd.DataFrame):
                            save_frame(f"h{horizon}/{name}/{arm}/{table}.parquet", frame)
                    (output / f"h{horizon}/{name}/{arm}/summary.json").write_text(json.dumps(account["summary"], indent=2) + "\n")
                    save_frame(f"h{horizon}/{name}/{arm}/pending-candidates.parquet", result["pending"][horizon][name][arm])
                comparison = result["comparisons"][horizon, name]
                for table in ["daily_accounts", "daily_pair_exposure", "pair_diagnostics"]:
                    save_frame(f"h{horizon}/{name}/paired/{table}.parquet", comparison[table])
                for side in ["treated", "control"]:
                    for table in ["daily_trade_attribution", "terminal_trade_attribution", "daily_reconciliation"]:
                        save_frame(f"h{horizon}/{name}/paired/{side}-{table}.parquet", comparison[side + "_attribution"][table])
            save_frame(f"h{horizon}/frozen-pairs.parquet", result["matches"][horizon]["pairs"])
            save_frame(f"h{horizon}/unmatched.parquet", result["matches"][horizon]["unmatched"])
        summary = {k: v for k, v in result.items() if k not in {
            "features", "event_audit", "candidates", "accounts", "pending", "matches", "comparisons"}}
        (output / "research-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        receipt.update(status=result["status"], completed_at=utc_now(),
                       output_files={str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob("*")) if p.is_file()})
    except Exception as exc:
        receipt.update(status="rejected_or_failed_research_attempt", completed_at=utc_now(),
                       error_type=type(exc).__name__, reason=str(exc))
        (output / "run-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
        raise
    (output / "run-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    return receipt
