"""Read-only whole-campaign comparison after explicit free-factor enrollment."""
import argparse
import concurrent.futures
import datetime
import json
from pathlib import Path

from verify_cloud_migration import query, settings


def main():
    parser = argparse.ArgumentParser()
    for name in ('config', 'baseline', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    c = settings(args.config)
    cid = c['RESEARCH_CAMPAIGN_ID']
    before = json.loads(args.baseline.read_text())
    campaign = query(c, 'SELECT * FROM research_campaigns WHERE campaign_id=?', [cid])[0]
    assert campaign['plan_sha256'] == before['campaign']['plan_sha256']
    runs = query(c, "SELECT * FROM research_validation_runs WHERE campaign_id=? AND run_id LIKE 'enable-free-factors%' AND status='completed' ORDER BY started_at", [cid])
    routes = {}
    for run in runs:
        report = json.loads(run['report_json'])
        assert report['applied_jobs'] == len(report['routes'])
        assert report['before_state_key'] and report['qualification_key'] and run['evidence_key']
        for route in report['routes']:
            job_id = route['before']['job_id']
            assert job_id not in routes
            routes[job_id] = route

    def page(offset):
        return query(c, 'SELECT * FROM research_jobs WHERE campaign_id=? ORDER BY ordinal LIMIT 1000 OFFSET ?', [cid, offset])
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [j for group in pool.map(page, range(0, campaign['expected_jobs'], 1000)) for j in group]
    assert len(jobs) == len(before['jobs']) == 11624
    preserved_captures = preserved_publications = preserved_failures = 0
    factor_jobs = []
    for old, new in zip(before['jobs'], jobs):
        for key in ('job_id', 'ordinal', 'identity_json'):
            assert new[key] == old[key], (old['job_id'], key)
        assert new['attempt_count'] >= old['attempt_count']
        if old['status'] == 'captured_raw_verified':
            for key in ('status', 'rows', 'raw_key', 'raw_sha256', 'capture_path', 'capture_identity_json'):
                assert old[key] == new[key], (old['job_id'], key)
            preserved_captures += 1
        if old['warehouse_status'] == 'published':
            assert old['commit_id'] == new['commit_id'] and new['warehouse_status'] == 'published'
            preserved_publications += 1
        if old['job_id'] in routes:
            route = routes[old['job_id']]
            assert old == route['before'], old['job_id']
            original = json.loads(old['identity_json'])
            effective = json.loads(new['capture_identity_json'])
            assert effective == json.loads(route['effective_identity_json'])
            assert original['provider'] == 'tushare' and original['query']['api'] == 'adj_factor'
            assert effective['provider'] == 'sina' and effective['query']['api'] == 'hfq_factor'
            assert effective['instrument_id'] == original['instrument_id']
            assert effective['query']['params'] == original['query']['params']
            factor_jobs.append(new)
        elif old['status'] in ('failed_requires_review', 'blocked_source_api_requires_review'):
            assert new == old, old['job_id']
            preserved_failures += 1
    assert len(factor_jobs) == len(routes)
    report = {'verified_at': datetime.datetime.now(datetime.UTC).isoformat(), 'campaign': campaign,
              'immutable_jobs_verified': len(jobs), 'original_capture_receipts_preserved': preserved_captures,
              'original_publication_receipts_preserved': preserved_publications,
              'unchanged_original_failures': preserved_failures, 'enrolled_factor_jobs': len(routes),
              'enrolled_original_rejections_preserved_in_r2_before_states': len(routes),
              'factor_jobs': factor_jobs, 'enrollment_evidence_keys': [r['evidence_key'] for r in runs],
              'published_jobs': sum(j['warehouse_status'] == 'published' for j in jobs),
              'published_rows': sum(j['rows'] for j in jobs if j['warehouse_status'] == 'published'),
              'data_ready': False}
    with args.output.open('x') as out:
        json.dump(report, out, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != 'factor_jobs'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
