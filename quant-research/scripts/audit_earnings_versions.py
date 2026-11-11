"""Independent read-only source and as-of audit; never certify financial events."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from prepare_earnings_review import audit_capture, sha256

QUALIFICATIONS = {
    "financial_fields_verified", "historical_pit_verified", "first_numeric_version_verified",
    "complete_announcement_coverage_verified", "formal_profit_qualified",
    "formal_return_experiment_run",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def stamp(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(result.tzinfo is not None, "Timezone-aware evidence clock required")
    return result


def unqualified(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in QUALIFICATIONS:
                require(child is False, f"Qualification upgraded: {key}")
            unqualified(child)
    elif isinstance(value, list):
        for child in value:
            unqualified(child)


def compact_json_bytes(raw: bytes) -> bytes:
    """Remove only JSON syntax whitespace, preserving original numeric tokens."""
    quoted = escaped = False
    result = bytearray()
    for byte in raw:
        if quoted:
            result.append(byte)
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                quoted = False
        elif byte == 34:
            quoted = True
            result.append(byte)
        elif byte not in b" \n\r\t":
            result.append(byte)
    require(not quoted, "Unclosed JSON string")
    return bytes(result)


def identity(row: dict) -> tuple[str, str]:
    return row["security_code"], row["announcement_id"]


def numeric_key(observation: dict) -> tuple[str, str]:
    return (json.dumps(observation["row"], sort_keys=True, ensure_ascii=False),
            observation["extractor_source_sha256"])


def audit_registry(registry: Path, captures: list[Path],
                   attachments: list[tuple[Path, Path]]) -> list[dict]:
    require(registry.is_dir() and not registry.is_symlink(), "Registry directory required")
    sources = {}
    for capture in captures:
        manifest, _ = audit_capture(capture)
        digest = sha256(capture / "manifest.json")
        require(digest not in sources, "Duplicate capture argument")
        sources[digest] = capture, manifest
    numbers = {}
    for packet, candidate in attachments:
        digest = sha256(candidate)
        require(digest not in numbers, "Duplicate candidate argument")
        numbers[digest] = packet, candidate
    entries = sorted(registry.iterdir())
    batches = []
    previous = previous_time = None
    for sequence, folder in enumerate(entries, 1):
        require(folder.name == f"batch-{sequence:06}" and folder.is_dir()
                and not folder.is_symlink(), "Missing, partial, locked or unexpected batch")
        require({p.name for p in folder.iterdir()} == {"manifest.json", "observations.json"},
                "Partial or unexpected batch evidence")
        for name in ("manifest.json", "observations.json"):
            require(not (folder / name).is_symlink(), "Symlink is not registry evidence")
        raw = (folder / "observations.json").read_bytes()
        manifest = json.loads((folder / "manifest.json").read_bytes())
        value = json.loads(raw)
        unqualified(manifest)
        unqualified(value)
        require(all(manifest.get(k) is False for k in (
            "historical_pit_verified", "first_numeric_version_verified", "formal_profit_qualified")),
            "Missing registry qualification safeguards")
        require(manifest["sequence"] == sequence and manifest["schema_version"] == 1
                and manifest["previous_manifest_sha256"] == previous
                and manifest["observations_sha256"] == hashlib.sha256(raw).hexdigest()
                and manifest["import_sha256"] == hashlib.sha256(compact_json_bytes(raw)).hexdigest(),
                "Registry count or hash chain differs")
        registered = stamp(manifest["registered_at"])
        require(previous_time is None or previous_time <= registered,
                "Registration clock moved backwards")
        source = value["source"]
        require(source["capture_manifest_sha256"] in sources, "Unbound capture manifest")
        capture, original = sources[source["capture_manifest_sha256"]]
        require(source == {
            "provider": "cninfo", "capture_manifest_sha256": sha256(capture / "manifest.json"),
            "source_index_sha256": sha256(capture / "index.json"),
            "observed_at": original["finished_at"], "request": original["request"],
            "index_records": original["index_records"],
            "index_pagination_complete": original["index_pagination_complete"],
            "historical_pit_verified": False, "complete_announcement_coverage_verified": False,
        }, "Registered source metadata differs from actual capture")
        require(stamp(source["observed_at"]) <= registered, "Observation occurs after registration")
        rows = json.loads((capture / "index.json").read_bytes())["records"]
        require(len(rows) == len(value["records"]), "Source row count differs")
        for row, frozen in zip(rows, value["records"], strict=True):
            copied = {k: v for k, v in frozen.items() if k not in {
                "registered_document_kind", "title_period_claim", "archived_pdf_sha256"}}
            require(copied == row, "Registered listing identity, title or clock differs")
            require(stamp(row["source_published_at_claim"]) <= stamp(source["observed_at"]),
                    "Source publication claim occurs after actual observation")
            expected_pdf = original["files"].get(f"pdfs/{row['announcement_id']}.pdf")
            require(frozen["archived_pdf_sha256"] == expected_pdf, "Archived PDF binding differs")
        numeric = value["numeric"]
        if numeric is not None:
            require(numeric["candidates_sha256"] in numbers, "Unbound numeric attachment")
            packet_path, candidate_path = numbers[numeric["candidates_sha256"]]
            packet = json.loads(packet_path.read_bytes())
            candidate = json.loads(candidate_path.read_bytes())
            require(numeric["review_packet_sha256"] == sha256(packet_path)
                    == candidate["source_review_packet_sha256"]
                    and packet["capture_manifest_sha256"] == source["capture_manifest_sha256"]
                    and packet["source_index_sha256"] == source["source_index_sha256"]
                    and packet["financial_numeric_acceptance"] is False,
                    "Numeric packet lineage or qualification differs")
            for key in ("created_at", "extractor_version", "extractor_source_sha256", "records"):
                require(numeric[key] == candidate[key], "Persisted numeric attachment changed")
            require(stamp(source["observed_at"]) <= stamp(numeric["created_at"]) <= registered,
                    "Numeric version used outside its actual observation clocks")
            documents = {(d["instrument_id"], d["announcement_id"]): d
                         for d in packet["documents"]}
            require(len(documents) == len(packet["documents"]) == len(numeric["records"])
                    == original["pdfs_captured"], "Missing or duplicate numeric documents")
            for number in numeric["records"]:
                key = number["instrument_id"], number["announcement_id"]
                require(key in documents, "Unbound numeric identity")
                document = documents.pop(key)
                identifier = number["announcement_id"]
                require(number["pdf_sha256"] == document["pdf_sha256"]
                        == sha256(capture / "pdfs" / f"{identifier}.pdf")
                        and number["text_sha256"] == document["text_sha256"]
                        == sha256(packet_path.parent / "text" / f"{identifier}.txt"),
                        "Numeric document bytes differ from original archives")
        digest = sha256(folder / "manifest.json")
        batches.append({"manifest": manifest, "import": value, "digest": digest})
        previous, previous_time = digest, registered
    require(bool(batches), "No registered evidence")
    return batches


def audit_view(view: dict, batches: list[dict]) -> dict:
    unqualified(view)
    require(view.get("schema_version") == 1 and all(view.get(k) is False for k in (
        "historical_pit_verified", "first_numeric_version_verified", "formal_profit_qualified",
        "formal_return_experiment_run")), "Missing query qualification safeguards")
    cutoff = stamp(view["as_of"])
    visible = [b for b in batches if stamp(b["manifest"]["registered_at"]) <= cutoff]
    sources = [{"sequence": b["manifest"]["sequence"], "manifest_sha256": b["digest"],
                "registered_at": b["manifest"]["registered_at"], "source": b["import"]["source"]}
               for b in visible]
    require(view["sources"] == sources and view["registered_batches_in_view"] == len(sources),
            "Query contains missing, future or altered source batches")
    rows, numbers = [], {}
    for batch in visible:
        registered = batch["manifest"]["registered_at"]
        for row in batch["import"]["records"]:
            require(stamp(row["available_at"]) <= cutoff, "Query contains future source observation")
            rows.append({**row, "registry_available_at": registered,
                         "batch_manifest_sha256": batch["digest"]})
        numeric = batch["import"]["numeric"]
        for row in numeric["records"] if numeric is not None else []:
            observation = {"row": row, "registry_available_at": registered,
                           "numeric_created_at": numeric["created_at"],
                           "extractor_version": numeric["extractor_version"],
                           "extractor_source_sha256": numeric["extractor_source_sha256"],
                           "batch_manifest_sha256": batch["digest"]}
            numbers.setdefault(numeric_key(observation), observation)
    require(view["source_listing_observations"] == len(rows)
            and view["unique_listing_announcement_identities"] == len({identity(r) for r in rows})
            and view["numeric_observation_versions"] == len(numbers) == len(view["groups"]),
            "Query observation counts differ from available registered evidence")
    seen = set()
    for group in view["groups"]:
        require(all(group.get(k) is False for k in (
            "financial_fields_verified", "historical_pit_verified", "first_numeric_version_verified")),
            "Missing numeric query qualification safeguards")
        observation = group["numeric_observation"]
        key = numeric_key(observation)
        require(key in numbers and key not in seen and observation == numbers[key],
                "Missing, duplicated or changed numeric version in query")
        seen.add(key)
        target = next(r for r in rows if r["instrument_id"] == group["instrument_id"]
                      and r["announcement_id"] == group["announcement_id"]
                      and r["batch_manifest_sha256"] == observation["batch_manifest_sha256"])
        require(target["announcement_id"] == observation["row"]["announcement_id"]
                and target["instrument_id"] == observation["row"]["instrument_id"]
                and group["title_period_claim"] == target["title_period_claim"]
                and group["source_observed_at"] == target["available_at"],
                "Numeric query identity or source clock differs")
        expected = {name: {} for name in ("earlier_same_period_documents",
                    "same_claimed_time_documents", "ambiguous_period_predecessors")}
        for row in rows:
            if (row["instrument_id"] != target["instrument_id"]
                    or row["announcement_id"] == target["announcement_id"]
                    or stamp(row["source_published_at_claim"]) > stamp(target["source_published_at_claim"])):
                continue
            if row["title_period_claim"] is None or target["title_period_claim"] is None:
                name = "ambiguous_period_predecessors"
            elif row["title_period_claim"] != target["title_period_claim"]:
                continue
            elif stamp(row["source_published_at_claim"]) < stamp(target["source_published_at_claim"]):
                name = "earlier_same_period_documents"
            else:
                name = "same_claimed_time_documents"
            expected[name].setdefault(row["announcement_id"], {
                "announcement_id": row["announcement_id"], "title": row["title"],
                "document_kind": row["registered_document_kind"],
                "source_published_at_claim": row["source_published_at_claim"],
                "source_observed_at": row["available_at"],
                "registry_available_at": row["registry_available_at"], "pdf_url": row["pdf_url"],
            })
        for name, values in expected.items():
            require(group[name] == [values[k] for k in sorted(values)],
                    "Predecessor links differ from the available registered index")
        finding = "no_earlier_match_in_registered_sources_not_first_certification"
        if expected["earlier_same_period_documents"]:
            finding = "earlier_same_period_source_claim_observed_not_first"
        elif (expected["same_claimed_time_documents"] or expected["ambiguous_period_predecessors"]
              or target["title_period_claim"] is None):
            finding = "ambiguous_predecessor_or_same_claimed_time"
        require(group["finding"] == finding, "First-version discovery finding differs")
    return {"registered_batches": len(visible), "source_listing_observations": len(rows),
            "numeric_observation_versions": len(numbers),
            "findings": dict(Counter(g["finding"] for g in view["groups"]))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--capture", type=Path, action="append", required=True)
    parser.add_argument("--attachment", type=Path, nargs=2, action="append", default=[])
    parser.add_argument("--view", type=Path, action="append", required=True)
    parser.add_argument("--invariant-views", type=Path, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Audit output must be new")
    batches = audit_registry(args.registry, args.capture, args.attachment)
    views = {str(path): audit_view(json.loads(path.read_bytes()), batches) for path in args.view}
    first, replay = [json.loads(path.read_bytes()) for path in args.invariant_views]
    require(first == replay, "Late imports changed the earlier query view")
    inputs = ([args.registry / f"batch-{i:06}" / name for i in range(1, len(batches) + 1)
               for name in ("manifest.json", "observations.json")]
              + [p / name for p in args.capture for name in ("manifest.json", "index.json")]
              + [p for pair in args.attachment for p in pair]
              + args.view + args.invariant_views)
    result = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "status": "source_and_as_of_audit_passed_not_financial_acceptance",
              "registered_batches": len(batches), "views": views,
              "late_import_prior_view_unchanged": True,
              "independent_code_path": "Python source-byte audit of Rust registry outputs",
              "independent_human_acceptance": False, "financial_fields_verified": False,
              "historical_pit_verified": False, "first_numeric_version_verified": False,
              "formal_profit_qualified": False, "formal_return_experiment_run": False,
              "source_hashes": {str(p): sha256(p) for p in inputs},
              "audit_source_sha256": sha256(Path(__file__)),
              "limits": ["Title period labels remain unverified claims, not company/report identities.",
                         "No complete market coverage or historical availability certification.",
                         "A hash chain without an external head witness cannot detect tail deletion."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "source_hashes"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
