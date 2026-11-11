"""Research-only integrity and comparison recheck; never source or profit acceptance."""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from quant_research.storage import file_hash


def audit(comparison, ocr_root, packet, candidates, secondary_source=None):
    comparison, ocr_root, packet, candidates = map(Path, [comparison, ocr_root, packet, candidates])
    ocr_root = ocr_root.resolve()
    secondary_source = Path(secondary_source) if secondary_source else comparison.parent / "secondary-parser-v3.rs"
    value = json.loads(comparison.read_text())
    source = json.loads(packet.read_text())
    primary = json.loads(candidates.read_text())
    manifest = json.loads((ocr_root / "manifest.json").read_text())
    native = value.get("secondary_input_backend") == "MuPDF_mutool_stext_XML"
    manifest_key = "source_native_manifest_sha256" if native else "source_ocr_manifest_sha256"
    sources_key = "native_source_hashes" if native else "ocr_source_hashes"
    if (value["source_review_packet_sha256"] != file_hash(packet)
            or value[manifest_key] != file_hash(ocr_root / "manifest.json")
            or manifest["source_review_packet_sha256"] != file_hash(packet)
            or value["primary_source_sha256"] != primary["extractor_source_sha256"]
            or value["secondary_source_sha256"] != file_hash(secondary_source)):
        raise ValueError("Secondary comparison source identity differs")
    if manifest["status"] != ("captured_unqualified_native_evidence" if native else "captured_unqualified_pixel_evidence"):
        raise ValueError("Expected complete observed pixel evidence")
    for item in [value, manifest]:
        for flag in ["financial_fields_verified", "historical_pit_verified", "first_numeric_version_verified", "formal_profit_qualified"]:
            if item[flag] is not False:
                raise ValueError("Development evidence cannot promote qualification")
    for name, expected in value[sources_key].items():
        target = (ocr_root / name).resolve()
        if not target.is_relative_to(ocr_root) or file_hash(target) != expected:
            raise ValueError("OCR source bytes or containment changed")
    if len(value["records"]) != len(source["documents"]) or len(primary["records"]) != len(source["documents"]):
        raise ValueError("Fixed original coverage differs")
    counts, reasons, narrative = Counter(), Counter(), Counter()
    agreed_growth = agreed = pages = 0
    for row, original, first, pixels in zip(value["records"], source["documents"], primary["records"], manifest["records"], strict=True):
        for key in ["announcement_id", "instrument_id", "pdf_sha256"]:
            if row[key] != original[key] or first[key] != original[key] or pixels[key] != original[key]:
                raise ValueError("Original identity differs")
        if row["available_at"] != original["actual_available_at"]:
            raise ValueError("Original availability was backdated")
        pages += len(pixels["pages"])
        counts[row["status"]] += 1
        second = row["secondary"]
        reasons[second.get("reason", "secondary_candidate_extracted")] += 1
        if second.get("narrative_failure"):
            narrative[second["narrative_failure"]] += 1
        if row["automatic_financial_acceptance"] is not False or row["formal_profit_qualified"] is not False:
            raise ValueError("Per-document qualification changed")
        if row["status"] != "fields_agree_arithmetic_passed_not_qualified":
            continue
        if row["field_differences"] or any(row[k]["status"] != "pass" for k in ["primary_arithmetic_check", "secondary_arithmetic_check"]):
            raise ValueError("Agreement lacks arithmetic or has field conflicts")
        first = first["candidate"]
        for key in ["fiscal_period_start", "fiscal_period", "profit_metric", "original_unit"]:
            if first[key] != second[key]:
                raise ValueError("Declared semantic agreement differs")
        for key in ["net_profit_lower_cny", "net_profit_upper_cny", "prior_same_period_profit_cny", "disclosed_yoy_lower", "disclosed_yoy_upper"]:
            tolerance = Decimal("0.005") if key.endswith("cny") else Decimal("1e-8")
            if abs(Decimal(str(first[key])) - Decimal(str(second[key]))) > tolerance:
                raise ValueError("Decimal field comparison differs")
        agreed += 1
        agreed_growth += Decimal(str(second["disclosed_yoy_lower"])) >= Decimal("0.3")
    if dict(counts) != value["status_counts"] or dict(reasons) != value["secondary_reason_counts"]:
        raise ValueError("Comparison counts differ")
    return {"verified_at": datetime.now(timezone.utc).isoformat(), "documents": len(value["records"]),
            "pages": pages, ("native_sources_hash_verified" if native else "ocr_sources_hash_verified"): len(value[sources_key]),
            "status_counts": dict(counts), "secondary_reason_counts": dict(reasons),
            "narrative_failure_counts": dict(narrative), "agreement_documents": agreed,
            "agreed_growth_at_least_30_percent": agreed_growth, "human_review_required": False,
            "qualified_historical_signals": 0, "formal_profit_qualified": False,
            "formal_return_experiment_run": False,
            "limits": ["The fixed development originals do not provide unbiased semantic accuracy.",
                       "Agreement is not first-version, historical-time, execution or profit acceptance."],
            "source_hashes": {str(p.resolve()): file_hash(p) for p in [comparison, ocr_root / "manifest.json", packet, candidates, secondary_source, Path(__file__)]}}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["comparison", "ocr-root", "packet", "candidates", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--secondary-source", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Research audit output must be new")
    result = audit(args.comparison, args.ocr_root, args.packet, args.candidates, args.secondary_source)
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ["documents", "pages", "agreement_documents", "qualified_historical_signals"]}))
