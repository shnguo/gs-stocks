"""Independent research audit of restored source bytes; no live fetch or writes to the lake."""
import argparse
import bisect
import hashlib
import json
from pathlib import Path


def verify(root, job):
    inventory = json.loads((root / 'ARCHIVE-MANIFEST.json').read_text())['files']
    for name, entry in inventory.items():
        raw = (root / name).read_bytes()
        assert len(raw) == entry['bytes'] and hashlib.sha256(raw).hexdigest() == entry['sha256']
    path = root / job['capture_path']
    m = json.loads((path / 'manifest.json').read_text())
    identity = json.loads(job['capture_identity_json'])
    assert m['provider'] == 'sina' and m['api'] == 'hfq_factor'
    assert m['params'] == identity['query']['params']
    p = m['params']
    code = p['ts_code'][:6]
    text = (path / 'sina-hfq.js').read_text()
    prefix = 'var bj' + code + 'hfq='
    assert text.strip().startswith(prefix)
    factors, _ = json.JSONDecoder().raw_decode(text.strip()[len(prefix):])
    assert factors['total'] == len(factors['data'])
    series = sorted((f['d'], float(f['f'])) for f in factors['data'] if f['d'] != '1900-01-01')
    assert all(f > 0 for _, f in series)
    ref = m['daily_reference']['identity']
    assert ref['query']['params'] == p and ref['instrument_id'] == identity['instrument_id']
    raw = json.loads((path / 'daily-reference/response.json').read_text())
    assert ref['provider'] == 'tushare' and raw['code'] == 0
    records = [dict(zip(raw['data']['fields'], row)) for row in raw['data']['items']]
    assert all(r['ts_code'] == p['ts_code'] for r in records)
    prices = {r['trade_date'][:4] + '-' + r['trade_date'][4:6] + '-' + r['trade_date'][6:]: float(r['close']) for r in records}
    days = sorted(prices)
    start = p['start_date'][:4] + '-' + p['start_date'][4:6] + '-' + p['start_date'][6:]
    end = p['end_date'][:4] + '-' + p['end_date'][4:6] + '-' + p['end_date'][6:]
    anchors = [i for i, (day, _) in enumerate(series) if day <= days[0]]
    assert anchors
    anchor = max(anchors)
    assert series[anchor][0] <= start
    actions = json.loads((path / 'distributions.json').read_text())
    assert actions['success'] and actions['code'] == 0 and actions['result']['pages'] == 1
    events = {}
    for event in actions['result']['data']:
        assert event['SECURITY_CODE'] == code and event['SECUCODE'] == p['ts_code']
        if not event['EX_DIVIDEND_DATE']:
            continue
        day = event['EX_DIVIDEND_DATE'][:10]
        if days[0] < day <= end and ((event['PRETAX_BONUS_RMB'] or 0) > 0 or (event['BONUS_IT_RATIO'] or 0) > 0):
            assert event['ASSIGN_PROGRESS'] == '实施分配' and day not in events
            events[day] = event
    previous = series[anchor][1]
    residuals = []
    for day, value in series[anchor + 1:]:
        if day > end:
            break
        event = events.pop(day)
        prior = days[bisect.bisect_left(days, day) - 1]
        close = prices[prior]
        cash = (event['PRETAX_BONUS_RMB'] or 0) / 10
        stock = (event['BONUS_IT_RATIO'] or 0) / 10
        assert abs(stock * 10 - (event['BONUS_RATIO'] or 0) - (event['IT_RATIO'] or 0)) < 1e-6
        expected = close * (1 + stock) / (close - cash)
        residual = abs((value / previous) / expected - 1)
        assert residual <= 1e-5
        residuals.append(residual)
        previous = value
    assert not events and residuals
    assert job['rows'] == 1 + len(residuals)
    return {'job_id': job['job_id'], 'code': code, 'raw_key': job['raw_key'], 'files_verified': len(inventory),
            'factor_rows': job['rows'], 'events_verified': len(residuals), 'max_relative_residual': max(residuals)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('artifacts', type=Path)
    args = parser.parse_args()
    jobs = json.loads((args.artifacts / 'canary-jobs-final.json').read_text())
    assert len(jobs) == 3 and all(j['warehouse_status'] == 'published' for j in jobs)
    results = [verify(args.artifacts / ('canary-raw-' + json.loads(j['identity_json'])['query']['params']['ts_code'][:6]), j) for j in jobs]
    report = {'passed': True, 'samples': results, 'total_factor_rows': sum(r['factor_rows'] for r in results),
              'total_events_verified': sum(r['events_verified'] for r in results), 'data_ready': False}
    with (args.artifacts / 'independent-raw-verification.json').open('x') as out:
        json.dump(report, out, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
