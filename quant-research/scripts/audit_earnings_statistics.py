"""Reproduce statistical and sample-capacity diagnostics using artificial accounts only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.earnings_ledger import simulate_earnings_account
from quant_research.earnings_validation import (
    evaluate_research_diagnostics,
    normal_holding_capacity,
    paired_moving_block_interval,
)
from quant_research.fixtures import create_fixture


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(output: Path, protocol_path: Path | None = None):
    root = Path(__file__).resolve().parents[1]
    protocol_path = Path(protocol_path).resolve() if protocol_path else root / "configs/earnings-drift-validation-20261012-v2.json"
    protocol = json.loads(protocol_path.read_text())
    sessions = protocol["validation"]["single_formal_look_sessions"]
    output.mkdir(parents=True, exist_ok=False)
    snapshot = create_fixture(output / "artificial-source", output / "artificial-snapshot", stocks=12, days=sessions + 70)
    for name, value in {"open": 10., "close": 10., "high": 10.1, "low": 9.9, "factor": 1.,
                        "upper_limit": 11., "lower_limit": 9., "volume": 4_000_000., "amount": 40_000_000.}.items():
        snapshot.tables["bars"][name] = value
    snapshot.tables["actions"] = snapshot.tables["actions"].iloc[:0].copy()
    snapshot.tables["risk_events"] = snapshot.tables["risk_events"].iloc[:0].copy()
    overridden = output / "actual-artificial-inputs"
    overridden.mkdir()
    for name, frame in snapshot.tables.items():
        frame.to_parquet(overridden / (name + ".parquet"), index=False)
    (overridden / "scope.json").write_text(json.dumps({
        "synthetic": True, "purpose": "engineering_capacity_and_statistics_only",
        "exchange_calendar_verified": False, "formal_profit_qualified": False,
        "note": "These are the actual in-memory fixture overrides; the original fixture remains preserved."
    }, indent=2) + "\n")
    symbols = snapshot.tables["instruments"].instrument_id.tolist()
    industries = pd.DataFrame([dict(instrument_id=s, industry=f"artificial_{i % 3}",
        start_date=snapshot.dates[0], end_date="", known_at="2020-01-01T00:00:00Z") for i, s in enumerate(symbols)])
    fees = pd.DataFrame([dict(board=b, start_date=snapshot.dates[0], end_date="", commission_rate=.0005,
        minimum_commission=5., transfer_rate=.00001, stamp_duty=.0005, slippage_bps=10.)
        for b in snapshot.tables["rules"].board])
    dates = snapshot.dates[50:50 + sessions]
    decision_indices = list(range(50, 50 + sessions - 1, 20))
    quarters = pd.date_range("2010-03-31", periods=len(decision_indices), freq="QE").strftime("%Y-%m-%d").tolist()
    treated, controls, pairs, events = [], [], [], []
    for cycle, index in enumerate(decision_indices):
        date = snapshot.dates[index]
        clock = date + "T10:00:00Z"
        for slot in range(6):
            trade, control_trade = f"E1/artificial/{cycle}/{slot}", f"P_matched/artificial/{cycle}/{slot}"
            decision = dict(date=date, decision_at=clock, selection_order=slot, signal_atr14_raw=.2)
            treated.append({**decision, "instrument_id": symbols[slot], "trade_id": trade})
            controls.append({**decision, "instrument_id": symbols[slot + 6], "trade_id": control_trade})
            pairs.append(dict(date=date, decision_at=clock, treated=symbols[slot], control=symbols[slot + 6],
                treated_trade_id=trade, control_trade_id=control_trade, selection_order=slot, intended_budget=30000.))
            events.append({**treated[-1], "fiscal_period": quarters[cycle]})
    decisions, control_decisions, pairs, events = map(pd.DataFrame, [treated, controls, pairs, events])
    scenarios = {}
    for name, multiplier, delay in [("base", 1, 0), ("stress", 2, 0), ("delayed", 1, 1)]:
        accounts = {}
        for arm, selected in [("treated", decisions), ("control", control_decisions)]:
            account = simulate_earnings_account(snapshot, selected, fees, industries, dates[0], dates[-1],
                capital=200000, horizon=20, friction_multiplier=multiplier, delay_sessions=delay)
            accounts[arm] = account
            directory = output / name / arm
            directory.mkdir(parents=True)
            for table, frame in account.items():
                if isinstance(frame, pd.DataFrame):
                    frame.to_parquet(directory / (table + ".parquet"), index=False)
            (directory / "summary.json").write_text(json.dumps(account["summary"], indent=2) + "\n")
        accounts["full"] = accounts["treated"]
        scenarios[name] = accounts
    for name, frame in [("frozen-event-identities", events), ("frozen-pairs", pairs),
                        ("hypothetical-effective-fees", fees), ("hypothetical-industry-history", industries)]:
        frame.to_parquet(output / (name + ".parquet"), index=False)
    pd.DataFrame({"date": dates}).to_csv(output / "expected-artificial-calendar.csv", index=False)
    result = evaluate_research_diagnostics(scenarios, pairs, events, dates, protocol)
    buys = scenarios["base"]["full"]["fills"].query("side == 'buy'")
    # Separate counting path: raw filled entries and explicit session list, no maturity helper.
    independently_mature = sum(dates.index(str(d)) + 20 <= len(dates) - 1 for d in buys.date)
    expected_mature = sum(6 for entry in range(1, sessions, 20) if entry + 20 < sessions)
    assert independently_mature == result["mature_unique_filled_events"] == expected_mature
    assert len(buys) == len(decision_indices) * 6
    capacity = []
    for sessions in [120, 252, 342, 504]:
        row = normal_holding_capacity(sessions)
        by_enumeration = sum(6 for entry in range(1, sessions, 20) if entry + 20 < sessions)
        assert row["no_early_exit_mature_trade_upper_bound"] == by_enumeration
        capacity.append(row)
    # Nonzero, artificial series checks the bootstrap independently of flat-price accounts.
    left = .001 + np.sin(np.arange(sessions) / 13) * .002
    right = np.cos(np.arange(sessions) / 17) * .001
    interval = paired_moving_block_interval(left, right)
    rng = np.random.default_rng(20261012)
    samples = []
    for _ in range(10000):
        indices = [i for start in rng.integers(0, sessions - 20 + 1, size=math.ceil(sessions / 20))
                   for i in range(start, start + 20)][:sessions]
        samples.append(sum(float(left[i] - right[i]) for i in indices) * 252 / len(indices))
    independent_limits = np.quantile(samples, [.025, .975])
    assert np.allclose(independent_limits, [interval["interval_lower"], interval["interval_upper"]], atol=1e-12, rtol=0)
    pd.DataFrame({"artificial_treated": left, "artificial_control": right}).to_csv(
        output / "bootstrap-artificial-inputs.csv", index=False)
    source_paths = [protocol_path, Path(__file__), root / "src/quant_research/earnings_validation.py",
        root / "src/quant_research/earnings_comparison.py", root / "src/quant_research/earnings_ledger.py",
        root / "tests/test_earnings_validation.py"]
    report = {"scope": "artificial_engineering_evidence_only_not_real_strategy_returns",
        "formal_profit_qualified": False, "prospective_start_registered": False,
        "protocol_version": protocol["version"], "formal_look_sessions": sessions,
        "source_hashes": {str(p.relative_to(root)): digest(p) for p in source_paths},
        "output_hashes": {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file()},
        "normal_holding_capacity": capacity, "independent_mature_fill_count": independently_mature,
        "raw_buy_fill_count": len(buys), "statistics": result,
        "independent_nonzero_bootstrap_check": {"maximum_absolute_limit_difference": float(np.max(np.abs(
            independent_limits - [interval["interval_lower"], interval["interval_upper"]]))),
            "repetitions": 10000, "formal_profit_qualified": False},
        "caveats": ["Artificial event identities and unlimited qualifying opportunities assume a saturation bound only.",
            "Fixture market prices, industry, fees and periods are invented, not accepted historical inputs.",
            "Actual artificial inputs preserve all deterministic overrides separately from the original fixture.",
            "Forced risk exits can increase trade count; the capacity bound applies only to normal 20-session holding.",
            "The v2 504-session design retains 100 mature events and 60 mature entry dates; hypothetical saturation does not ensure real signal counts.",
            "Protocol and environment registration remain incomplete; this is no prospective or historical profit result."]}
    (output / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    args = parser.parse_args()
    result = audit(args.output, args.protocol)
    print(json.dumps({"output": str(args.output), "mature_artificial_fills": result["independent_mature_fill_count"],
        "formal_profit_qualified": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
