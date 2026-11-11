"""Independent research verification of native PDF geometry and financial comparisons."""
from __future__ import annotations

import csv
import io
import json
import xml.etree.ElementTree as ET
from decimal import Decimal
from pathlib import Path

from quant_research.storage import file_hash


def audit_native(root, code_root):
    root, code_root = Path(root), Path(code_root)
    manifest = json.loads((root / "manifest.json").read_text())
    progress = json.loads((root / "progress.json").read_text())
    if (manifest["status"] != "captured_unqualified_native_evidence"
            or progress["status"] != manifest["status"]
            or progress["documents_completed"] != len(manifest["records"])
            or manifest["adapter_source_sha256"] != file_hash(code_root / "crates/lakehouse-sources/src/earnings_native_pdf.rs")
            or manifest["ocr_performed"] is not False):
        raise ValueError("Native decoder provenance differs")
    decoded = pages = glyphs = abstained = 0
    for record in manifest["records"]:
        if record["status"] == "native_decode_abstain":
            if not record.get("failure"):
                raise ValueError("Abstention has no diagnostic")
            abstained += 1
            continue
        if record["status"] != "native_pages_decoded":
            raise ValueError("Unexpected native status")
        xml = root / record["xml"]["file"]
        if file_hash(xml) != record["xml"]["sha256"]:
            raise ValueError("Native XML changed")
        raw = xml.read_bytes()
        if b"<!DOCTYPE" in raw or b"<!ENTITY" in raw or len(raw) > 16 * 1024 * 1024:
            raise ValueError("Unsupported native XML declaration or size")
        source_pages = ET.fromstring(raw).findall("page")
        if len(source_pages) != len(record["pages"]) or len(source_pages) != record["declared_pages"]:
            raise ValueError("Native page coverage changed")
        for page_number, (source, output) in enumerate(zip(source_pages, record["pages"], strict=True), 1):
            if source.attrib["id"] != "page" + str(page_number) or output["page"] != page_number:
                raise ValueError("Native page order differs")
            text = "".join("".join(char.attrib["c"] for char in line.iter("char")) + "\n"
                           for line in source.iter("line"))
            if (root / output["text"]["file"]).read_text() != text:
                raise ValueError("Native text does not follow original XML characters")
            rows = list(csv.DictReader(io.StringIO((root / output["tsv"]["file"]).read_text()), delimiter="\t"))
            chars = list(source.iter("char"))
            if len(rows) != len(chars) or len(chars) != output["glyphs"]:
                raise ValueError("Native glyph count changed")
            for char, row in zip(chars, rows, strict=True):
                quad = [Decimal(v) for v in char.attrib["quad"].split()]
                if len(quad) != 8 or any(not v.is_finite() for v in quad):
                    raise ValueError("Invalid native quadrilateral")
                x, y = quad[::2], quad[1::2]
                geometry = [min(x), min(y), max(x) - min(x), max(y) - min(y)]
                if (row["text"] != char.attrib["c"] or row["conf"] != "-1"
                        or row["page_num"] != str(page_number) or row["level"] != "5"):
                    raise ValueError("Native token meaning changed")
                for name, value in zip(["left", "top", "width", "height"], geometry, strict=True):
                    # Only tolerance for f64 serialization of source coordinates, in PDF points.
                    if abs(Decimal(row[name]) - value) > Decimal("1e-9"):
                        raise ValueError("Native coordinate differs from original XML")
            pages += 1
            glyphs += len(chars)
        decoded += 1
    return {"decoded_documents": decoded, "decode_abstentions": abstained,
            "pages_independently_replayed": pages, "glyphs_independently_replayed": glyphs,
            "independent_replay_is_research_verification_not_a_third_financial_parser": True}


def audit_decimal_arithmetic(comparison, candidates):
    comparisons = json.loads(Path(comparison).read_text())
    primary = json.loads(Path(candidates).read_text())
    count = 0
    max_error = Decimal(0)
    for row, first in zip(comparisons["records"], primary["records"], strict=True):
        a, b = first["candidate"], row["secondary"]
        expected = "fields_agree_arithmetic_passed_not_qualified"
        if a["status"] != "candidate_extracted_requires_independent_review":
            expected = "primary_unsupported_abstain"
        elif b["status"] != "secondary_candidate_extracted_requires_consensus":
            expected = "secondary_unsupported_abstain"
        else:
            different = any(a[k] != b[k] for k in ["fiscal_period_start", "fiscal_period", "profit_metric", "original_unit"])
            for key in ["net_profit_lower_cny", "net_profit_upper_cny", "prior_same_period_profit_cny", "disclosed_yoy_lower", "disclosed_yoy_upper"]:
                tolerance = Decimal("0.005") if key.endswith("cny") else Decimal("1e-8")
                different |= (a.get(key) is None or b.get(key) is None
                              or abs(Decimal(str(a[key])) - Decimal(str(b[key]))) > tolerance)
            if different:
                expected = "field_conflict_abstain"
            elif row["primary_arithmetic_check"]["status"] != "pass" or row["secondary_arithmetic_check"]["status"] != "pass":
                expected = "arithmetic_not_passed_abstain"
        if row["status"] != expected:
            raise ValueError("Independent result classification differs")
        for values, check in [(first["candidate"], row["primary_arithmetic_check"]),
                              (row["secondary"], row["secondary_arithmetic_check"])]:
            if check["status"] != "pass":
                continue
            detail = check["details"]
            prior = Decimal(str(values["prior_same_period_profit_cny"]))
            widths = [Decimal(str(v)) for v in detail["currency_rounding_half_widths_cny"]]
            reported = [Decimal(str(values[k])) for k in ["disclosed_yoy_lower", "disclosed_yoy_upper"]]
            half = Decimal(str(detail["disclosed_percent_rounding_half_width"]))
            for i, key in enumerate(["net_profit_lower_cny", "net_profit_upper_cny"]):
                amount = Decimal(str(values[key]))
                interval = [(amount - widths[i]) / (prior + widths[2]) - 1,
                            (amount + widths[i]) / (prior - widths[2]) - 1]
                for value, existing in zip(interval, detail["computed_yoy_rounding_intervals"][i], strict=True):
                    max_error = max(max_error, abs(value - Decimal(str(existing))))
                if reported[i] + half + Decimal("1e-10") < interval[0] or reported[i] - half - Decimal("1e-10") > interval[1]:
                    raise ValueError("Independent Decimal computation rejects arithmetic pass")
            count += 1
    if max_error > Decimal("1e-12"):
        raise ValueError("Independent Decimal intervals differ")
    return {"passing_path_records_recomputed": count, "maximum_interval_endpoint_error": str(max_error),
            "all_document_result_categories_recomputed": len(comparisons["records"])}
