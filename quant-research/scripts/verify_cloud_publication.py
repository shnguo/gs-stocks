"""Read-only verification of warehouse recovery without changing original capture evidence."""
import argparse
import concurrent.futures
import datetime
import json
from pathlib import Path

from verify_cloud_migration import query, settings


def main():
    parser = argparse.ArgumentParser()
    for name in ['config', 'baseline', 'output']:
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    config = settings(args.config)
    baseline = json.loads(args.baseline.read_text())
    cid = config['RESEARCH_CAMPAIGN_ID']
    campaign = query(config, 'SELECT * FROM research_campaigns WHERE campaign_id=?', [cid])[0]

    def page(offset):
        return query(config, 'SELECT * FROM research_jobs WHERE campaign_id=? ORDER BY ordinal LIMIT 1000 OFFSET ?', [cid, offset])

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        rows = [row for group in pool.map(page, range(0, campaign['expected_jobs'], 1000)) for row in group]
    assert len(rows) == len(baseline['jobs']) == campaign['expected_jobs']
    preserved_capture = preserved_publication = preserved_review = 0
    recovered = []
    new_publications = []
    for old, new in zip(baseline['jobs'], rows):
        for key in ['job_id', 'identity_json', 'ordinal']:
            assert old[key] == new[key], (old['job_id'], key)
        if old['status'] == 'captured_raw_verified':
            for key in ['status', 'rows', 'raw_key', 'raw_sha256', 'capture_path', 'capture_identity_json']:
                assert old[key] == new[key], (old['job_id'], key)
            preserved_capture += 1
            if old['warehouse_status'] == 'pending':
                assert new['warehouse_status'] == 'published' and new['commit_id'], old['job_id']
                recovered.append(new)
        if old['warehouse_status'] == 'published':
            assert new['warehouse_status'] == 'published' and new['commit_id'] == old['commit_id'], old['job_id']
            preserved_publication += 1
        if old['status'] in ['blocked_source_api_requires_review', 'failed_requires_review']:
            assert new == old, old['job_id']
            preserved_review += 1
        if old['warehouse_status'] != 'published' and new['warehouse_status'] == 'published':
            new_publications.append(new)
    attempts = query(config, 'SELECT * FROM research_publication_attempts WHERE campaign_id=? ORDER BY started_at', [cid])
    assert len(attempts) == len({a['attempt_id'] for a in attempts})
    assert any(a['status'] == 'legacy_failed' and a['evidence_key'] for a in attempts)
    report = {
        'verified_at': datetime.datetime.now(datetime.UTC).isoformat(), 'campaign': campaign,
        'immutable_jobs_verified': len(rows), 'original_capture_receipts_preserved': preserved_capture,
        'original_publication_receipts_preserved': preserved_publication,
        'original_failures_and_restrictions_preserved': preserved_review,
        'recovered_pending_jobs': len(recovered), 'recovered_pending_rows': sum(j['rows'] for j in recovered),
        'newly_published_jobs': new_publications, 'publication_attempts': attempts,
        'published_jobs': sum(j['warehouse_status'] == 'published' for j in rows),
        'published_rows': sum(j['rows'] for j in rows if j['warehouse_status'] == 'published'),
        'data_ready': False,
    }
    with args.output.open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k not in ['newly_published_jobs', 'publication_attempts']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
