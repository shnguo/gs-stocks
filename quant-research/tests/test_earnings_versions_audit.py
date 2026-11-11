from copy import deepcopy

import pytest
from audit_earnings_versions import audit_view, compact_json_bytes


def evidence():
    source = {"observed_at": "2026-10-12T10:00:00Z"}
    row = {"security_code": "600001", "instrument_id": "cn.xshg.600001",
           "announcement_id": "200", "available_at": source["observed_at"],
           "title_period_claim": "2026H1", "source_published_at_claim": "2026-07-15T00:00:00Z"}
    number = {"announcement_id": "200", "instrument_id": row["instrument_id"],
              "candidate": {"net_profit_lower_cny": 10000000.0}}
    numeric = {"records": [number], "created_at": "2026-10-12T11:00:00Z",
               "extractor_version": 9, "extractor_source_sha256": "e" * 64}
    batch = {"manifest": {"registered_at": "2026-10-12T12:00:00Z", "sequence": 1},
             "import": {"source": source, "records": [row], "numeric": numeric},
             "digest": "b" * 64}
    observation = {"row": number, "registry_available_at": batch["manifest"]["registered_at"],
                   "numeric_created_at": numeric["created_at"], "extractor_version": 9,
                   "extractor_source_sha256": numeric["extractor_source_sha256"],
                   "batch_manifest_sha256": batch["digest"]}
    view = {"schema_version": 1, "as_of": "2026-10-12T13:00:00Z",
            "historical_pit_verified": False, "first_numeric_version_verified": False,
            "formal_profit_qualified": False, "formal_return_experiment_run": False,
            "sources": [{"sequence": 1, "manifest_sha256": batch["digest"],
                         "registered_at": batch["manifest"]["registered_at"], "source": source}],
            "registered_batches_in_view": 1, "source_listing_observations": 1,
            "unique_listing_announcement_identities": 1, "numeric_observation_versions": 1,
            "groups": [{"announcement_id": "200", "instrument_id": row["instrument_id"],
                        "numeric_observation": observation, "title_period_claim": "2026H1",
                        "source_observed_at": source["observed_at"],
                        "earlier_same_period_documents": [], "same_claimed_time_documents": [],
                        "ambiguous_period_predecessors": [], "financial_fields_verified": False,
                        "historical_pit_verified": False, "first_numeric_version_verified": False,
                        "finding": "no_earlier_match_in_registered_sources_not_first_certification"}]}
    return deepcopy(view), deepcopy([batch])


def test_original_numeric_tokens_and_string_whitespace_survive_hash_compaction():
    raw = b'{\n  "text": "a b \\\" c", "number": 1.2300000000000000e+09\n}'
    assert compact_json_bytes(raw) == b'{"text":"a b \\\" c","number":1.2300000000000000e+09}'


@pytest.mark.parametrize("field", ["as_of", "numeric", "qualification"])
def test_future_batches_numeric_mutation_and_qualification_upgrade_rejected(field):
    view, batches = evidence()
    assert audit_view(view, batches)["numeric_observation_versions"] == 1
    if field == "as_of":
        view["as_of"] = "2026-10-12T11:59:59Z"
    elif field == "numeric":
        view["groups"][0]["numeric_observation"]["row"]["candidate"]["net_profit_lower_cny"] += 1
    else:
        view["first_numeric_version_verified"] = True
    with pytest.raises(ValueError):
        audit_view(view, batches)


def test_observed_unknown_period_predecessor_cannot_be_hidden_by_no_match_claim():
    view, batches = evidence()
    earlier = {**batches[0]["import"]["records"][0], "announcement_id": "100",
               "title_period_claim": None, "source_published_at_claim": "2026-07-01T00:00:00Z",
               "title": "更正提示", "registered_document_kind": "earnings_correction",
               "pdf_url": "https://example.invalid/100.pdf"}
    batches[0]["import"]["records"].append(earlier)
    view["source_listing_observations"] = view["unique_listing_announcement_identities"] = 2
    with pytest.raises(ValueError, match="Predecessor links"):
        audit_view(view, batches)


def test_later_registered_index_is_invisible_to_prior_cutoff():
    view, batches = evidence()
    later = deepcopy(batches[0])
    later["manifest"].update(sequence=2, registered_at="2026-10-12T14:00:00Z")
    later["import"]["numeric"] = None
    later["digest"] = "c" * 64
    assert audit_view(view, batches + [later]) == audit_view(view, batches)
