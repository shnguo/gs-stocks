"""Research-only independent recheck of a source-bound automatic arithmetic screen.

No financial extraction, collection, source qualification or order generation occurs.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from audit_earnings_preflight import automatic_numeric_observation

from quant_research.storage import file_hash


def audit(screen_path, candidates_path, packet_path, preflight_path):
    screen_path, candidates_path, packet_path, preflight_path = map(
        Path, [screen_path, candidates_path, packet_path, preflight_path])
    observation = automatic_numeric_observation(screen_path, candidates_path, packet_path)
    screen = json.loads(screen_path.read_text())
    candidates = json.loads(candidates_path.read_text())
    preflight = json.loads(preflight_path.read_text())
    if preflight.get("human_review_required") is not False or preflight.get("formal_profit_qualified") is not False:
        raise ValueError("Expected unqualified automatic preflight")
    for name, expected in preflight["source_hashes"].items():
        if file_hash(Path(name)) != expected:
            raise ValueError("Preflight input changed: " + name)
    maximum_error = Decimal(0)
    checked = growth_gate = 0
    for row, original in zip(screen["records"], candidates["records"], strict=True):
        if row["check"]["status"] != "pass":
            continue
        values, detail = original["candidate"], row["check"]["details"]
        prior = Decimal(str(values["prior_same_period_profit_cny"]))
        widths = [Decimal(str(v)) for v in detail["currency_rounding_half_widths_cny"]]
        reported = [Decimal(str(values[name])) for name in ["disclosed_yoy_lower", "disclosed_yoy_upper"]]
        percent_width = Decimal(str(detail["disclosed_percent_rounding_half_width"]))
        for index, name in enumerate(["net_profit_lower_cny", "net_profit_upper_cny"]):
            amount = Decimal(str(values[name]))
            expected = [(amount - widths[index]) / (prior + widths[2]) - 1,
                        (amount + widths[index]) / (prior - widths[2]) - 1]
            for recomputed, actual in zip(expected, detail["computed_yoy_rounding_intervals"][index], strict=True):
                maximum_error = max(maximum_error, abs(recomputed - Decimal(str(actual))))
            if (reported[index] + percent_width + Decimal("1e-10") < expected[0]
                    or reported[index] - percent_width - Decimal("1e-10") > expected[1]):
                raise ValueError("Decimal arithmetic disagrees with a pass")
        checked += 1
        gate = reported[0] >= Decimal("0.3")
        if detail["reported_primary_growth_gate_met"] != gate or detail["missing_yoy_imputed"] is not False:
            raise ValueError("Primary disclosed gate or missing-value policy changed")
        growth_gate += gate
    if maximum_error >= Decimal("1e-12"):
        raise ValueError("Decimal interval verification differs from Rust")
    fee = next(v for v in preflight["components"] if v["id"] == "actual_broker_minimum_fee")["observed"]
    return {"verified_at": datetime.now(timezone.utc).isoformat(), **observation,
            "decimal_recheck_documents": checked, "decimal_max_interval_endpoint_error": str(maximum_error),
            "arithmetic_compatible_with_reported_growth_at_least_30_percent": growth_gate,
            "minimum_commission_cny": fee["minimum_commission_cny"],
            "minimum_commission_user_confirmed": fee["minimum_confirmed"],
            "preflight_unmet_components": len(preflight["components"]),
            "preflight_bound_sources": len(preflight["source_hashes"]),
            "qualified_historical_signals": 0, "formal_profit_qualified": False,
            "formal_return_experiment_run": False,
            "limit": "Decimal checks recompute arithmetic, not extraction semantics, source accuracy or profitability.",
            "source_hashes": {str(p.resolve()): file_hash(p) for p in [
                screen_path, candidates_path, packet_path, preflight_path, Path(__file__)]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["screen", "candidates", "packet", "preflight", "output"]:
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Audit output must be new")
    value = audit(args.screen, args.candidates, args.packet, args.preflight)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"documents": value["documents"], "decimal_rechecked": value["decimal_recheck_documents"],
                      "qualified_historical_signals": 0}))


if __name__ == "__main__":
    main()
