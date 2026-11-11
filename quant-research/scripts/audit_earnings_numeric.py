"""Independent read-only comparison of initial source labels and Rust candidates."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from prepare_earnings_review import audit_capture, sha256


def audit(capture: Path, packet: Path, candidates: Path, reference: Path) -> dict:
    manifest, captured_documents = audit_capture(capture)
    source = json.loads(packet.read_text())
    extracted = json.loads(candidates.read_text())
    labels = json.loads(reference.read_text())
    if (extracted["source_review_packet_sha256"] != sha256(packet)
            or extracted["financial_fields_verified"] is not False
            or extracted["historical_pit_verified"] is not False
            or source["source_index_sha256"] != sha256(capture / "index.json")
            or source["capture_manifest_sha256"] != sha256(capture / "manifest.json")):
        raise ValueError("Source binding or candidate qualification differs")
    rows = {r["announcement_id"]: r for r in extracted["records"]}
    documents = {r["announcement_id"]: r for r in source["documents"]}
    if (len(rows) != len(extracted["records"])
            or len(documents) != len(source["documents"])
            or set(rows) != set(documents)
            or set(documents) != {r["announcement_id"] for r in captured_documents}):
        raise ValueError("Missing or duplicate document candidates")
    observed_records = json.loads((capture / "index.json").read_text())["records"]
    for identifier, row in rows.items():
        original = documents[identifier]
        if (original["pdf_sha256"] != sha256(capture / "pdfs" / f"{identifier}.pdf")
                or original["text_sha256"] != sha256(packet.parent / "text" / f"{identifier}.txt")
                or original["actual_available_at"] != manifest["finished_at"]
                or not any(r["announcement_id"] == identifier
                    and r["instrument_id"] == original["instrument_id"]
                    and r["title"] == original["title"] for r in observed_records)):
            raise ValueError("Review packet differs from actual archived document or source identity")
        for key in ["instrument_id", "pdf_sha256", "text_sha256"]:
            if row[key] != original[key]:
                raise ValueError("Candidate lineage differs from observed document")
        if row["available_at"] != original["actual_available_at"]:
            raise ValueError("Candidate availability changed")
        candidate = row["candidate"]
        if any(candidate.get(k) is not False for k in ["financial_fields_verified",
                "historical_pit_verified", "first_numeric_version_verified"]):
            raise ValueError("Automatic extraction cannot qualify a historical signal")
    checks = []
    reference_ids = [r["announcement_id"] for r in labels["records"]]
    if (len(reference_ids) != len(set(reference_ids))
            or not set(reference_ids).issubset(rows)
            or labels["reviewed_documents"] != len(reference_ids)):
        raise ValueError("Missing, duplicate, or inconsistent source reference count")
    for expected in labels["records"]:
        identifier = expected["announcement_id"]
        row = rows[identifier]
        if row["instrument_id"] != expected["instrument_id"]:
            raise ValueError("Reference label security identity differs")
        if row["pdf_sha256"] != expected["pdf_sha256"] or row["text_sha256"] != expected["text_sha256"]:
            raise ValueError("Reference label document identity differs")
        candidate = row["candidate"]
        supported = candidate["status"] == "candidate_extracted_requires_independent_review"
        fields = []
        expected_outcome = expected.get("expected_parser_outcome")
        if expected_outcome == "unsupported":
            if supported or candidate.get("reason") not in expected["expected_unsupported_reasons"]:
                raise ValueError(f"Reviewed exclusion changed: {identifier}")
        elif expected_outcome not in {None, "supported"}:
            raise ValueError("Unknown reference parser outcome")
        elif supported:
            expected_fields = {key: expected[key] for key in ["fiscal_period_start", "fiscal_period", "profit_metric"]}
            expected_fields.update(expected["expected_numeric_fields"])
            for key, value in expected_fields.items():
                if key.startswith("previous_"):
                    continue
                actual = candidate.get(key)
                tolerance = 0.005 if key.endswith("_cny") else 1e-8
                matches = (math.isclose(actual, value, rel_tol=0., abs_tol=tolerance)
                           if isinstance(value, (int, float)) and isinstance(actual, (int, float))
                           else actual == value)
                if not matches:
                    raise ValueError(f"Initial source reference mismatch: {identifier}/{key}")
                fields.append(key)
        elif expected["case"] not in {
            "table_layout_should_remain_unsupported_until_parser_is_validated",
            "correction_excluded_from_primary_first_forecast",
        }:
            raise ValueError("Previously supported reference case became unsupported")
        checks.append({"announcement_id": identifier, "case": expected["case"],
                       "candidate_extracted": supported, "checked_fields": fields,
                       "source_values_recorded_but_not_parsed": expected_outcome == "unsupported",
                       "result": "matches_initial_source_label" if supported else "remains_explicitly_unsupported"})
    status = Counter(r["candidate"]["status"] for r in rows.values())
    reasons = Counter(r["candidate"].get("reason") for r in rows.values()
                      if r["candidate"]["status"] == "unsupported_requires_review")
    return {"schema_version": 1, "source_listing_announcement_rows": manifest["index_records"],
            "pdfs": len(rows), "extraction_statuses": dict(status),
            "unsupported_reasons": dict(reasons), "initial_reference_checks": checks,
            "reference_documents_inspected": len(checks), "independent_100_document_acceptance": False,
            "qualified_historical_signals": 0, "formal_strategy_return_test_run": False,
            "source_hashes": {p.name: sha256(p) for p in [packet, candidates, reference]},
            "conclusion": "The extractor is not ready for a representative event-return experiment; unsupported layouts remain explicit and no historical timestamp is certified."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["capture", "packet", "candidates", "reference", "output"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.capture, args.packet, args.candidates, args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
