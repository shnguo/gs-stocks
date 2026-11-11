"""Read-only source audit and review packet; never certify extracted financial values."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_capture(capture: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((capture / "manifest.json").read_text())
    if (manifest.get("status") != "captured_research_observations"
            or manifest.get("historical_pit_verified") is not False
            or manifest.get("complete_announcement_coverage_verified") is not False):
        raise ValueError("Only a completed, explicitly unqualified capture can be audited")
    root = capture.resolve()
    for name, expected in manifest["files"].items():
        source = (root / name).resolve()
        if not source.is_relative_to(root) or sha256(source) != expected:
            raise ValueError("Archived source hash or containment check failed")
    index = json.loads((root / "index.json").read_text())
    if (index.get("numeric_fields_verified") is not False
            or index.get("historical_pit_verified") is not False
            or index.get("observed_at") != manifest.get("finished_at")):
        raise ValueError("Observation qualification or clock differs")
    records = index["records"]
    if "stock" in manifest.get("request", {}):
        parts = manifest["request"]["stock"].split(",")
        if (len(parts) != 2 or not re.fullmatch(r"[06]\d{5}", parts[0])
                or parts[0].startswith(("688", "689"))
                or not re.fullmatch(r"[A-Za-z0-9_]{1,64}", parts[1])
                or any(r["security_code"] != parts[0] or r.get("org_id") != parts[1]
                       or r.get("scope_exclusion") is not None for r in records)):
            raise ValueError("Original source rows differ from explicitly requested stock scope")
    keys = {(r["security_code"], r["announcement_id"]) for r in records}
    if len(keys) != len(records) or manifest["index_records"] != len(records):
        raise ValueError("Duplicate or missing listing/announcement records")
    raw_keys = []
    for name in sorted(n for n in manifest["files"] if re.fullmatch(r"index-\d{3}\.json", n)):
        page = json.loads((root / name).read_text())
        if page["totalAnnouncement"] != manifest["claimed_total"]:
            raise ValueError("Source total changed between pages")
        source_rows = page["announcements"]
        if source_rows is None and page["totalAnnouncement"] == 0:
            source_rows = []
        if not isinstance(source_rows, list):
            raise ValueError("Missing announcement list for nonempty source total")
        raw_keys.extend((r["secCode"], r["announcementId"]) for r in source_rows)
    if len(raw_keys) != len(set(raw_keys)) or set(raw_keys) != keys:
        raise ValueError("Raw pagination disagrees with normalized identities")
    if manifest["index_pagination_complete"] != (len(raw_keys) == manifest["claimed_total"]):
        raise ValueError("Pagination completion claim differs from observed rows")
    documents = {}
    for row in records:
        if row["available_at"] != index["observed_at"] or row["availability_basis"] != "local_observation":
            raise ValueError("Old source dates must not replace actual availability")
        name = f"pdfs/{row['announcement_id']}.pdf"
        captured = name in manifest["files"]
        if row["pdf_captured"] != captured:
            raise ValueError("PDF capture flag differs from archived evidence")
        if captured:
            if not (root / name).read_bytes().startswith(b"%PDF-"):
                raise ValueError("Archived PDF header is missing")
            documents.setdefault(row["announcement_id"], row)
    if len(documents) != manifest["pdfs_captured"]:
        raise ValueError("Unique PDF count differs from capture receipt")
    return manifest, sorted(documents.values(), key=lambda r: r["announcement_id"])


def excerpt(text: str) -> str:
    paragraphs = text.splitlines()
    selected = set()
    for i, line in enumerate(paragraphs):
        if any(word in line for word in ["归属", "净利润", "单位", "预告期间", "比上年", "更正"]):
            selected.update(range(max(0, i - 2), min(len(paragraphs), i + 4)))
    return "\n".join(paragraphs[i] for i in sorted(selected))[:6000]


def refined_classifications(capture: Path, classification: Path | None) -> dict:
    if classification is None:
        return {}
    value = json.loads(classification.read_text())
    if (value.get("source_index_sha256") != sha256(capture / "index.json")
            or value.get("numeric_fields_verified") is not False
            or value.get("classification_version") != 2):
        raise ValueError("Derived classifications must bind to the same unqualified source index")
    index = json.loads((capture / "index.json").read_text())["records"]
    expected = {(r["security_code"], r["announcement_id"]): r for r in index}
    result = {}
    for row in value["records"]:
        key = row["security_code"], row["announcement_id"]
        if (key in result or key not in expected or row["title"] != expected[key]["title"]
                or row["original_document_kind"] != expected[key]["document_kind"]
                or row.get("financial_fields_verified") is not False):
            raise ValueError("Derived classification identity or qualification differs")
        result[key] = row["document_kind"]
    if set(result) != set(expected):
        raise ValueError("Derived classifications must cover every original index record")
    return result


def prepare(capture: Path, output: Path, pdftotext: str, pdfinfo: str,
            classification: Path | None = None) -> dict:
    manifest, documents = audit_capture(capture)
    refined = refined_classifications(capture, classification)
    if output.exists():
        raise FileExistsError("Refusing to overwrite a review packet")
    output.mkdir(parents=True)
    (output / "text").mkdir()
    rows = []
    for row in documents:
        identifier = row["announcement_id"]
        kind = refined.get((row["security_code"], identifier), row["document_kind"])
        pdf = capture / "pdfs" / f"{identifier}.pdf"
        text_file = output / "text" / f"{identifier}.txt"
        result = subprocess.run([pdftotext, "-layout", "-nopgbrk", str(pdf), str(text_file)],
                                capture_output=True, timeout=30, check=False)
        info = subprocess.run([pdfinfo, str(pdf)], capture_output=True,
                              timeout=30, check=False)
        page_match = re.search(rb"^Pages:\s*(\d+)", info.stdout, re.MULTILINE)
        size = text_file.stat().st_size if text_file.exists() else 0
        text = text_file.read_text(errors="replace") if 0 < size <= 2_000_000 else ""
        status = "awaiting_independent_numeric_review"
        if result.returncode or not text.strip() or "\ufffd" in text:
            status = "text_unavailable_or_invalid_requires_visual_review"
        if row["scope_exclusion"] is not None:
            status = "excluded_listing_scope"
        elif kind != "earnings_forecast":
            status = "non_primary_document_type_requires_separate_review"
        rows.append({
            "announcement_id": identifier, "instrument_id": row["instrument_id"],
            "title": row["title"], "document_kind": kind,
            "original_document_kind": row["document_kind"],
            "source_announcement_date_claim": row["source_announcement_date"],
            "actual_available_at": row["available_at"], "availability_basis": "local_observation",
            "pdf_url": row["pdf_url"], "pdf_sha256": sha256(pdf),
            "text_sha256": sha256(text_file) if text_file.exists() else None,
            "pages": int(page_match[1]) if page_match else None,
            "pdf_security_code_text_found": row["security_code"] in text,
            "extract_status": "text_observed" if text else "unavailable",
            "review_status": status, "financial_fields_verified": False,
            "historical_pit_verified": False, "first_numeric_version_verified": False,
            "fiscal_period": None, "net_profit_lower_cny": None,
            "net_profit_upper_cny": None, "prior_same_period_profit_cny": None,
            "disclosed_yoy_lower": None, "disclosed_yoy_upper": None,
            "original_unit": None, "profit_metric": None,
            "reviewer": None, "reviewed_at": None, "financial_evidence_excerpt": excerpt(text),
        })
    packet = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "capture_manifest_sha256": sha256(capture / "manifest.json"),
              "source_index_sha256": sha256(capture / "index.json"),
              "classification_sha256": sha256(classification) if classification else None,
              "selection": manifest["pdf_selection"], "documents": rows,
              "automated_checks": "source hashes, pagination identities, PDF headers, text availability",
              "financial_numeric_acceptance": False, "accepted_financial_documents": 0}
    (output / "review-packet.json").write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n")
    columns = [k for k in rows[0] if k != "financial_evidence_excerpt"] if rows else []
    with (output / "review-checklist.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = {"source_listing_announcement_rows": manifest["index_records"],
               "source_category_pagination_complete": manifest["index_pagination_complete"],
               "pdf_documents": len(rows), "text_observed": sum(r["extract_status"] == "text_observed" for r in rows),
               "document_types": dict(Counter(r["document_kind"] for r in rows)),
               "review_statuses": dict(Counter(r["review_status"] for r in rows)),
               "financial_documents_accepted": 0, "historical_pit_verified": False,
               "complete_event_coverage_verified": False,
               "files": {p.name: sha256(p) for p in output.iterdir() if p.is_file()},
               "note": "Downloading and extracting text is not an independent numeric review."}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pdftotext", default="pdftotext")
    parser.add_argument("--pdfinfo", default="pdfinfo")
    parser.add_argument("--classification", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.capture, args.output, args.pdftotext, args.pdfinfo,
                             args.classification),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
