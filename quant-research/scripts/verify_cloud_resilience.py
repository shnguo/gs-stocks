"""Read-only full-plan, preserved-receipt and retry-lineage verification."""
import argparse
import collections
import concurrent.futures
import datetime
import json
from pathlib import Path
from verify_cloud_migration import query, settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('plan', type=Path)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    config = settings(args.config)
    cid = config['RESEARCH_CAMPAIGN_ID']
    campaign = query(config, 'SELECT * FROM research_campaigns WHERE campaign_id=?', [cid])[0]
    def page(offset):
        return query(config, 'SELECT * FROM research_jobs WHERE campaign_id=? ORDER BY ordinal LIMIT 1000 OFFSET ?', [cid, offset])
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        rows = [row for group in pool.map(page, range(0, campaign['expected_jobs'], 1000)) for row in group]
    plan = json.loads(args.plan.read_text())
    baseline = json.loads(args.baseline.read_text())
    assert len(rows) == len(plan['jobs']) == len(baseline['jobs']) == campaign['expected_jobs']
    preserved = collections.Counter()
    failed = []
    for row, planned, old in zip(rows, plan['jobs'], baseline['jobs']):
        assert row['job_id'] == planned['job_id'] == old['job_id']
        assert json.loads(row['identity_json']) == planned['identity'] == json.loads(old['identity_json'])
        assert row['ordinal'] == old['ordinal']
        if old['status'] in ('captured_raw_verified', 'blocked_source_api_requires_review'):
            for field in ('status', 'rows', 'raw_key', 'raw_sha256', 'capture_path', 'warehouse_status', 'commit_id'):
                assert row[field] == old[field], (row['job_id'], field)
            preserved[old['status']] += 1
        if old['status'] == 'failed_requires_review':
            failed.append(old)
    attempts = query(config, "SELECT * FROM research_attempts WHERE campaign_id=? AND attempt_id NOT LIKE 'legacy-%'", [cid])
    for old in failed:
        current = next(r for r in rows if r['job_id'] == old['job_id'])
        if current['raw_key'] != old['raw_key']:
            assert any(a['job_id'] == old['job_id'] and a['raw_key'] == old['raw_key'] and a['raw_sha256'] == old['raw_sha256'] and a['last_error'] == old['last_error'] for a in attempts), old['job_id']
    assert len({a['attempt_id'] for a in attempts}) == len(attempts)
    new = [a for a in attempts if a['kind'] == 'capture']
    published = {r['job_id']: r for r in rows if r['warehouse_status'] == 'published'}
    report = {
        'verified_at': datetime.datetime.now(datetime.UTC).isoformat(), 'campaign': campaign,
        'immutable_jobs_verified': len(rows), 'protected_receipts_preserved': dict(preserved),
        'original_failed_evidence_preserved': len(failed),
        'new_capture_outcomes': dict(collections.Counter(a['status'] for a in new)),
        'new_capture_sources': dict(collections.Counter(json.loads(a['identity_json'])['provider'] for a in new)),
        'new_published_jobs': [r for r in published.values() if r['job_id'] in {a['job_id'] for a in new}],
        'published_jobs': len(published), 'published_rows': sum(r['rows'] for r in published.values()),
        'data_ready': False,
    }
    with args.output.open('x') as out:
        json.dump(report, out, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != 'new_published_jobs'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
