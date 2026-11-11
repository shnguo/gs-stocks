import hashlib
import json

import pytest
from prepare_earnings_review import audit_capture


def write_json(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def capture(tmp_path):
    root = tmp_path / "capture"
    root.mkdir()
    (root / "pdfs").mkdir()
    (root / "pdfs/123.pdf").write_bytes(b"%PDF-test-fixture-not-a-real-document")
    observed = "2026-10-12T09:00:00Z"
    # One issuer document may be listed under two securities; the grain remains explicit.
    records = [{"security_code": code, "instrument_id": f"cn.xshe.{code}",
                "title": "Synthetic earnings forecast", "announcement_id": "123", "available_at": observed,
                "availability_basis": "local_observation", "pdf_captured": True}
               for code in ["000550", "200550"]]
    write_json(root / "index.json", {"records": records, "observed_at": observed,
                                     "numeric_fields_verified": False,
                                     "historical_pit_verified": False})
    write_json(root / "index-001.json", {"totalAnnouncement": 2, "announcements": [
        {"secCode": r["security_code"], "announcementId": r["announcement_id"]}
        for r in records]})
    manifest = {"status": "captured_research_observations", "historical_pit_verified": False,
                "complete_announcement_coverage_verified": False, "finished_at": observed,
                "index_records": 2, "claimed_total": 2, "index_pagination_complete": True,
                "pdfs_captured": 1, "files": {}}
    for name in ["index.json", "index-001.json", "pdfs/123.pdf"]:
        manifest["files"][name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    write_json(root / "manifest.json", manifest)
    return root


def test_listing_rows_do_not_inflate_unique_original_document_count(capture):
    manifest, documents = audit_capture(capture)
    assert manifest["index_records"] == 2
    assert len(documents) == 1
    assert manifest["historical_pit_verified"] is False


def test_zero_total_with_null_announcements_is_a_source_bound_empty_observation(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    observed = "2026-10-12T09:00:00Z"
    write_json(root / "index.json", {"records": [], "observed_at": observed,
               "numeric_fields_verified": False, "historical_pit_verified": False})
    write_json(root / "index-001.json", {"totalAnnouncement": 0, "announcements": None})
    manifest = {"status": "captured_research_observations", "historical_pit_verified": False,
                "complete_announcement_coverage_verified": False, "finished_at": observed,
                "index_records": 0, "claimed_total": 0, "index_pagination_complete": True,
                "pdfs_captured": 0, "files": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                    for name in ["index.json", "index-001.json"]}}
    write_json(root / "manifest.json", manifest)
    checked, documents = audit_capture(root)
    assert documents == [] and checked["index_records"] == 0
    assert checked["complete_announcement_coverage_verified"] is False


def test_null_announcements_cannot_hide_positive_source_total(capture):
    page = {"totalAnnouncement": 2, "announcements": None}
    write_json(capture / "index-001.json", page)
    manifest = json.loads((capture / "manifest.json").read_text())
    manifest["files"]["index-001.json"] = hashlib.sha256((capture / "index-001.json").read_bytes()).hexdigest()
    write_json(capture / "manifest.json", manifest)
    with pytest.raises(ValueError, match="nonempty source total"):
        audit_capture(capture)


def test_modified_raw_identity_is_rejected_even_if_its_hash_is_recomputed(capture):
    page = json.loads((capture / "index-001.json").read_text())
    page["announcements"][0]["announcementId"] = "999"
    write_json(capture / "index-001.json", page)
    manifest = json.loads((capture / "manifest.json").read_text())
    manifest["files"]["index-001.json"] = hashlib.sha256(
        (capture / "index-001.json").read_bytes()).hexdigest()
    write_json(capture / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Raw pagination"):
        audit_capture(capture)


def test_source_metadata_cannot_be_upgraded_to_historical_certification(capture):
    manifest = json.loads((capture / "manifest.json").read_text())
    manifest["historical_pit_verified"] = True
    write_json(capture / "manifest.json", manifest)
    with pytest.raises(ValueError, match="explicitly unqualified"):
        audit_capture(capture)


@pytest.mark.parametrize("stock", ["000550,org", "000550", "000550,org;other"])
def test_source_rows_or_invalid_filter_cannot_escape_requested_stock_scope(capture, stock):
    manifest = json.loads((capture / "manifest.json").read_text())
    manifest["request"] = {"stock": stock}
    write_json(capture / "manifest.json", manifest)
    with pytest.raises(ValueError, match="stock scope"):
        audit_capture(capture)


def test_evidence_paths_cannot_escape_the_capture(capture):
    external = capture.parent / "external.txt"
    external.write_text("unrelated file")
    manifest = json.loads((capture / "manifest.json").read_text())
    manifest["files"]["../external.txt"] = hashlib.sha256(external.read_bytes()).hexdigest()
    write_json(capture / "manifest.json", manifest)
    with pytest.raises(ValueError, match="containment"):
        audit_capture(capture)


@pytest.fixture
def numeric_reference(capture, tmp_path):
    from prepare_earnings_review import sha256

    review = tmp_path / "review"
    review.mkdir()
    (review / "text").mkdir()
    text_path = review / "text/123.txt"
    text_path.write_text("Synthetic parent profit 10000000000 yuan, not a source acceptance")
    observed = "2026-10-12T09:00:00Z"
    document = {"announcement_id": "123", "instrument_id": "cn.xshe.000550",
                "title": "Synthetic earnings forecast", "actual_available_at": observed,
                "pdf_sha256": sha256(capture / "pdfs/123.pdf"), "text_sha256": sha256(text_path)}
    packet = review / "review-packet.json"
    write_json(packet, {"capture_manifest_sha256": sha256(capture / "manifest.json"),
                        "source_index_sha256": sha256(capture / "index.json"),
                        "documents": [document]})
    fields = {"net_profit_lower_cny": 10_000_000_000., "net_profit_upper_cny": 11_000_000_000.,
              "prior_same_period_profit_cny": 5_000_000_000., "disclosed_yoy_lower": 1.,
              "disclosed_yoy_upper": 1.2, "original_unit": "元"}
    period = {"fiscal_period_start": "2026-01-01", "fiscal_period": "2026-06-30",
              "profit_metric": "attributable_to_parent_same_reporting_period"}
    qualification = {"financial_fields_verified": False, "historical_pit_verified": False,
                     "first_numeric_version_verified": False}
    candidate = {**document, "available_at": observed,
                 "candidate": {**period, **fields, **qualification,
                               "status": "candidate_extracted_requires_independent_review"}}
    candidates = tmp_path / "candidates.json"
    write_json(candidates, {"source_review_packet_sha256": sha256(packet),
                           "financial_fields_verified": False, "historical_pit_verified": False,
                           "records": [candidate]})
    reference = tmp_path / "reference.json"
    write_json(reference, {"reviewed_documents": 1, "records": [{**document, **period,
        "expected_numeric_fields": fields, "case": "synthetic_large_amount_reference"}]})
    return capture, packet, candidates, reference


def test_reference_comparison_does_not_hide_yuan_errors_in_large_amounts(numeric_reference):
    from audit_earnings_numeric import audit

    assert audit(*numeric_reference)["reference_documents_inspected"] == 1
    candidates = numeric_reference[2]
    value = json.loads(candidates.read_text())
    value["records"][0]["candidate"]["net_profit_lower_cny"] += 1.
    write_json(candidates, value)
    with pytest.raises(ValueError, match="net_profit_lower_cny"):
        audit(*numeric_reference)


def test_duplicate_reference_labels_cannot_inflate_reviewed_count(numeric_reference):
    from audit_earnings_numeric import audit

    reference = numeric_reference[3]
    value = json.loads(reference.read_text())
    value["records"] *= 2
    value["reviewed_documents"] = 2
    write_json(reference, value)
    with pytest.raises(ValueError, match="reference count"):
        audit(*numeric_reference)


def test_coordinated_packet_and_candidate_rehash_cannot_replace_observed_text(numeric_reference):
    from audit_earnings_numeric import audit
    from prepare_earnings_review import sha256

    packet, candidates, reference = numeric_reference[1:]
    forged_hash = "a" * 64
    value = json.loads(packet.read_text())
    value["documents"][0]["text_sha256"] = forged_hash
    write_json(packet, value)
    value = json.loads(candidates.read_text())
    value["source_review_packet_sha256"] = sha256(packet)
    value["records"][0]["text_sha256"] = forged_hash
    write_json(candidates, value)
    value = json.loads(reference.read_text())
    value["records"][0]["text_sha256"] = forged_hash
    write_json(reference, value)
    with pytest.raises(ValueError, match="actual archived document"):
        audit(*numeric_reference)


def test_reviewed_exclusion_cannot_silently_become_an_accepted_parser_case(numeric_reference):
    from audit_earnings_numeric import audit

    reference = numeric_reference[3]
    value = json.loads(reference.read_text())
    value["records"][0].update(expected_parser_outcome="unsupported",
                               expected_unsupported_reasons=["ambiguous_prior_basis"])
    write_json(reference, value)
    with pytest.raises(ValueError, match="Reviewed exclusion changed"):
        audit(*numeric_reference)
