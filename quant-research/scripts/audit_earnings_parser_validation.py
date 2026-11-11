"""Research audit of the frozen new-original parser check, not trading acceptance."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from audit_earnings_secondary import audit as audit_secondary
from prepare_earnings_review import audit_capture

from quant_research.storage import file_hash


def instant(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Parser validation clocks must be timezone aware")
    return result


def audit(root, owning_repo=None):
    root = Path(root)
    def read(name):
        return json.loads((root / name).read_text())
    plan, frozen, capture = read("selection.json"), read("parser-freeze.json"), read("capture/manifest.json")
    secondary_folder = frozen.get("secondary_folder", "ocr")
    native = secondary_folder == "native"
    packet, first, pixels, compared = [read(name) for name in [
        "review/review-packet.json", "primary-candidates.json", secondary_folder + "/manifest.json", "comparison.json"]]
    if (file_hash(root / "selection.json") != frozen["selection_sha256"]
            or file_hash(root / "capture/manifest.json") != frozen["new_capture_manifest_sha256"]
            or frozen["new20_financial_contents_inspected"] is not False
            or frozen["no_post_result_tuning_in_this_validation"] is not True):
        raise ValueError("Fixed selection or freeze declaration changed")
    for name, expected in frozen["source_code_hashes"].items():
        if file_hash(root / "code-freeze" / name) != expected:
            raise ValueError("Frozen financial implementation changed")
        if owning_repo is not None and file_hash(Path(owning_repo) / name) != expected:
            raise ValueError("Current owning implementation differs from validation freeze")
    if owning_repo is not None and file_hash(Path(owning_repo) / "target/debug/mootdx-cf-rs") != frozen["binary_sha256"]:
        raise ValueError("Current validation executable differs from freeze")
    clocks = [instant(plan["created_at"]), instant(capture["started_at"]), instant(capture["finished_at"]),
              instant(frozen["frozen_at"]), instant(packet["created_at"]), instant(first["created_at"]),
              instant(pixels["started_at"]), instant(pixels["finished_at"]), instant(compared["created_at"])]
    if clocks != sorted(clocks):
        raise ValueError("Financial extraction or validation preceded the code freeze")
    expected = {row["announcement_id"]: row for row in plan["selected"]}
    actual = {row["announcement_id"]: row for row in packet["documents"]}
    if len(expected) != 20 or expected.keys() != actual.keys():
        raise ValueError("New-original sample replaced, dropped or duplicated a fixed id")
    if set(expected).intersection(plan["previous_local_pdf_ids"]):
        raise ValueError("Selected originals were already local at sample declaration")
    for identity, row in expected.items():
        for key in ["instrument_id", "title", "pdf_url", "source_announcement_date"]:
            actual_key = "source_announcement_date_claim" if key == "source_announcement_date" else key
            if row[key] != actual[identity][actual_key]:
                raise ValueError("New original identity differs from declared source metadata")
    # This independent source replay includes all inherited raw index pages and
    # fresh PDF hashes. Reused pages remain explicitly late archive observations.
    audit_capture(root / "capture")
    if capture["index_observation_basis"] != "late_reobservation_of_hash_bound_archived_pages":
        raise ValueError("Archived index cannot be relabeled as a new live source query")
    if capture["request"]["capture_mode"] != "fixed_attachments_from_archived_index_no_new_index_request":
        raise ValueError("Unexpected source query or acquisition mode")
    for language, source in pixels.get("traineddata_hashes_before_run", {}).items():
        if file_hash(Path(source["path"])) != source["sha256"]:
            raise ValueError("OCR model file changed after the checked run: " + language)
    result = audit_secondary(root / "comparison.json", root / secondary_folder, root / "review/review-packet.json",
                             root / "primary-candidates.json", root / "code-freeze/crates/lakehouse-sources/src/earnings_secondary.rs")
    if native:
        from audit_earnings_native import audit_decimal_arithmetic, audit_native
        result["native_integrity"] = audit_native(root / "native", root / "code-freeze")
        result["decimal_arithmetic"] = audit_decimal_arithmetic(root / "comparison.json", root / "primary-candidates.json")
        if pixels["decoder"] != frozen["external_tools"]["mutool"]:
            raise ValueError("Native decoder differs from frozen tool")
        for tool, source in frozen["external_tools"].items():
            if file_hash(Path(source["path"])) != source["sha256"]:
                raise ValueError("Frozen external tool changed: " + tool)
    if result["documents"] != 20:
        raise ValueError("Full fixed validation sample was not evaluated")
    result.update({"freeze_and_extraction_chronology_verified": True, "selected_ids_previously_local": 0,
                   "all_declared_originals_evaluated": True, "current_source_and_executable_equal_to_freeze": owning_repo is not None,
                   "archived_source_index_reused_without_new_listing_requests": True,
                   "formal_strategy_registration_completed": False,
                   "role": "new20_parser_validation_not_strategy_returns_or_unbiased_population_accuracy"})
    for path in [root / "selection.json", root / "parser-freeze.json", root / "capture/manifest.json", Path(__file__)]:
        result["source_hashes"][str(path.resolve())] = file_hash(path)
    result["audited_at"] = datetime.now(timezone.utc).isoformat()
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--owning-repo", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Validation audit output must be new")
    value = audit(args.root, args.owning_repo)
    with args.output.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: value[k] for k in ["documents", "pages", "agreement_documents",
                                           "freeze_and_extraction_chronology_verified", "qualified_historical_signals"]}))
