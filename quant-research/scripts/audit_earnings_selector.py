"""Replay source-bound numeric candidates through the first-event research gate.

This audit preserves all unverified flags and never runs an account-return experiment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from audit_earnings_numeric import audit
from prepare_earnings_review import sha256

from quant_research import earnings_selection
from quant_research.earnings_selection import EVENT_COLUMNS, first_event_audit
from quant_research.storage import utc_now


def run(capture, packet, candidates, reference, snapshot, output):
    numeric_audit = audit(capture, packet, candidates, reference)
    source = json.loads(packet.read_text())
    extracted = json.loads(candidates.read_text())
    original = {r["announcement_id"]: r for r in source["documents"]}
    manifest = json.loads((snapshot / "manifest.json").read_text())
    calendar_path = snapshot / "calendar.parquet"
    if sha256(calendar_path) != manifest["files"]["calendar.parquet"]:
        raise ValueError("Exchange calendar bytes differ from the bound input snapshot")
    through = manifest["declaration"].get("price_data_through")
    if not through:
        raise ValueError("Actual observed price boundary must be declared for this audit")
    calendar = pd.read_parquet(calendar_path)
    dates = sorted(calendar.loc[calendar.is_open & calendar.date.le(through), "date"].unique())
    canonical = []
    for row in extracted["records"]:
        doc, number = original[row["announcement_id"]], row["candidate"]
        item = {key: number.get(key) for key in EVENT_COLUMNS}
        item.update(announcement_id=row["announcement_id"], instrument_id=row["instrument_id"],
                    available_at=row["available_at"], document_kind=doc["document_kind"],
                    availability_basis=row["availability_basis"], pdf_sha256=row["pdf_sha256"],
                    financial_fields_verified=False, first_numeric_version_verified=False,
                    historical_pit_verified=False)
        canonical.append(item)
    frame = pd.DataFrame(canonical, columns=sorted(EVENT_COLUMNS))
    checked = first_event_audit(frame, dates, historical=True)
    summary = {"schema_version": 1, "audited_at": utc_now(), "observed_calendar_through": dates[-1],
               "source_documents": len(frame), "numeric_extraction_statuses": numeric_audit["extraction_statuses"],
               "first_event_gate_reasons": {k: int(v) for k, v in checked.reason.value_counts().items()},
               "qualified_historical_signals": int(checked.eligible.sum()),
               "formal_profit_qualified": False, "return_experiment_run": False,
               "independent_100_document_acceptance": False,
               "sources": {str(path): sha256(path) for path in [packet, candidates, reference,
                   snapshot / "manifest.json", calendar_path, Path(earnings_selection.__file__), Path(__file__)]},
               "conclusion": "No event is promoted by extraction, a claimed old announcement date, or a future calendar session."}
    output.mkdir(parents=True, exist_ok=False)
    frame.to_parquet(output / "canonical-events.parquet", index=False)
    checked.to_parquet(output / "first-event-audit.parquet", index=False)
    files = {p.name: sha256(p) for p in output.iterdir()}
    summary["output_files"] = files
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["capture", "packet", "candidates", "reference", "snapshot", "output"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = run(**vars(args))
    print(json.dumps({key: result[key] for key in ["source_documents", "qualified_historical_signals",
          "first_event_gate_reasons", "return_experiment_run"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
