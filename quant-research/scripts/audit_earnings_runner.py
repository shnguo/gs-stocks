"""Build and independently inspect an artificial offline experiment; no market returns."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.daily_loop import read, verify
from quant_research.earnings_fixture import create_earnings_bundle
from quant_research.storage import Snapshot, file_hash, utc_now


def audit(output, legacy, failed_adapter, protocol=None):
    output, legacy, failed_adapter = Path(output).resolve(), Path(legacy).resolve(), Path(failed_adapter).resolve()
    if output.exists():
        raise FileExistsError("Audit destination must be new")
    output.mkdir(parents=True)
    project = Path(__file__).resolve().parents[1]
    protocol = Path(protocol).resolve() if protocol else project / "configs/earnings-drift-validation-20261012-v2.json"
    create_earnings_bundle(output / "input", protocol)
    with (output / "cli.log").open("w") as log:
        completed = subprocess.run([sys.executable, str(project / "scripts/run_earnings_experiment.py"),
            "--inputs", str(output / "input"), "--output", str(output / "experiment")],
            cwd=project, stdout=log, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise RuntimeError("Artificial runner failed; attempt log preserved")
    receipt = read(output / "experiment/run-receipt.json")
    for name, expected in receipt["output_files"].items():
        if file_hash(output / "experiment" / name) != expected:
            raise ValueError("Research output hash mismatch")
    spec = read(output / "input/inputs.json")
    calendar = pd.read_parquet(output / "input/snapshot/calendar.parquet")
    dates = sorted(calendar.loc[calendar.is_open & calendar.date.ge(spec["start"]) & calendar.date.le(spec["end"]), "date"].unique())
    summaries = []
    for summary_path in sorted((output / "experiment").glob("h*/**/summary.json")):
        summary = read(summary_path)
        nav = pd.read_parquet(summary_path.with_name("nav.parquet"))
        if nav.date.tolist() != dates or not np.allclose(nav.nav,
                nav.cash + nav.market_value + nav.receivables - nav.dividend_tax_reserve):
            raise ValueError("Independent daily calendar or NAV identity failed")
        computed = float(nav.nav.iloc[-1] / 200_000 - 1)
        if not np.isclose(computed, summary["total_return"], rtol=0, atol=1e-12):
            raise ValueError("Independent terminal NAV return differs")
        summaries.append(dict(account=str(summary_path.parent.relative_to(output / "experiment")),
                              observed_sessions=len(nav), terminal_nav_return=computed,
                              formal_profit_qualified=False))
    if len(summaries) != 36:
        raise ValueError("Expected 36 artificial accounts")
    for path in sorted((output / "experiment").glob("h*/**/paired/daily_accounts.parquet")):
        daily = pd.read_parquet(path)
        if daily.date.tolist() != dates or not np.allclose(daily.daily_net_return_difference,
                daily.daily_return_treated - daily.daily_return_control, atol=1e-12, rtol=0):
            raise ValueError("Independent paired daily difference failed")
    source = verify(legacy / "snapshot")
    bars = pd.read_parquet(legacy / "snapshot/bars.parquet")
    numeric = bars[["open", "high", "low", "close", "volume", "amount", "factor"]].apply(pd.to_numeric, errors="raise")
    nonfinite = ~np.isfinite(numeric)
    original_adapter = read(failed_adapter / "adapter-receipt.json")
    adapted_manifest = read(failed_adapter / "manifest.json")
    if (not all(adapted_manifest["declaration"][k] == v for k, v in source["declaration"].items())
            or adapted_manifest["source_created_at"] != source["created_at"]):
        raise ValueError("Adapter changed original declarations or observation clock")
    if not all(file_hash(failed_adapter / name) == expected for name, expected in source["files"].items()):
        raise ValueError("Failed adapter did not preserve table bytes")
    try:
        Snapshot(failed_adapter)
    except ValueError as error:
        rejection = str(error)
    else:
        raise ValueError("Expected original malformed training input to remain rejected")
    result = dict(schema_version=1, created_at=utc_now(), status="engineering_only_verified_actual_input_rejected",
        artificial_accounts=summaries, artificial_observed_sessions=len(dates),
        output_file_count=len(receipt["output_files"]),
        actual_input=dict(rows=len(bars), nonfinite_by_column={k: int(v) for k, v in nonfinite.sum().items()},
            affected_rows=int(nonfinite.any(axis=1).sum()), canonical_rejection=rejection,
            original_adapter_status=original_adapter["status"], source_bytes_preserved=True,
            purpose=source["declaration"]["purpose"], original_declarations_unchanged=True),
        protocol_sha256=file_hash(protocol), formal_profit_qualified=False, formal_return_experiment_run=False,
        formal_daily_runner_implemented=False, prospective_start_registered=False,
        method="Independent calendar, NAV identity and terminal-return arithmetic from persisted account tables.",
        limits=["Artificial returns are not estimates, forecasts or results for any market strategy.",
                "This audit does not constitute independent source acceptance or complete environment registration.",
                "Original failed adapter receipt is preserved; its missing error text is recorded here from strict read-only reload."],
        source_hashes={str(p): file_hash(p) for p in [protocol, Path(__file__), legacy / "snapshot/manifest.json",
            legacy / "snapshot/bars.parquet", failed_adapter / "adapter-receipt.json", output / "input/manifest.json",
            output / "experiment/run-receipt.json"]})
    (output / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--failed-adapter", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    args = parser.parse_args()
    result = audit(args.output, args.legacy, args.failed_adapter, args.protocol)
    print(json.dumps({"status": result["status"], "accounts": len(result["artificial_accounts"]),
                      "actual_missing_factor_rows": result["actual_input"]["affected_rows"], "formal_profit_qualified": False}))


if __name__ == "__main__":
    main()
