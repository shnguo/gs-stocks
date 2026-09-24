"""Small Rust-acquired historical-feature pilot with independent raw replay."""
import argparse
import json
import shutil
import subprocess
import zlib
from pathlib import Path

import numpy as np

from quant_research.storage import file_hash, utc_now, write_json

FIELDS = 'date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST'.split(',')
END = b'<![CDATA[]]>\n'


def read(path):
    return json.loads(path.read_text())


def decode(raw):
    if not raw.endswith(END) or len(raw) < 33:
        raise ValueError('Invalid raw frame')
    header = raw[:21].decode()
    version, kind, size = header.split('\x01')
    if version not in ['00.9.00', '00.9.30'] or not size.isdigit():
        raise ValueError('Invalid response header')
    body = raw[21:-len(END)]
    if kind == '96':
        if int(size) > len(body):
            raise ValueError('Truncated compressed frame')
        trailer = body[int(size):].removesuffix(b'\n')
        if trailer and (trailer[:1] != b'\x01' or not trailer[1:].isdigit()):
            raise ValueError('Invalid compressed trailer')
        decoder = zlib.decompressobj()
        text = decoder.decompress(body[:int(size)])+decoder.flush()
        if not decoder.eof or decoder.unused_data:
            raise ValueError('Invalid compressed body')
        text = text.decode()
    else:
        text, crc = body.decode().rsplit('\x01', 1)
        if len(text) != int(size) or zlib.crc32((header+text).encode()) != int(crc):
            raise ValueError('Invalid source checksum')
    return kind, text.split('\x01')


def replay(directory):
    if (directory/'job.json').exists():
        job = read(directory/'job.json')
        if job['status'] != 'completed' or not job.get('selected_attempt'):
            raise ValueError('Incomplete recovery job')
        selected = None
        for attempt in job['attempts']:
            path = (directory/attempt['directory']).resolve()
            if not path.is_relative_to(directory.resolve()):
                raise ValueError('Invalid recovery path')
            if file_hash(path/'manifest.json') != attempt['manifest_sha256']:
                raise ValueError('Changed recovery evidence')
            manifest = read(path/'manifest.json')
            if manifest['query'] != job['query']:
                raise ValueError('Recovery query changed')
            if attempt['directory'] == job['selected_attempt']:
                if attempt['status'] != 'completed':
                    raise ValueError('Selected failed attempt')
                selected = path
        if selected is None:
            raise ValueError('Missing recovery selection')
        directory = selected
    meta = read(directory/'manifest.json')
    if meta['status'] != 'completed' or meta['provider'] != 'baostock':
        raise ValueError('Incomplete capture')
    records = read(directory/'records.json')
    if file_hash(directory/'records.json') != meta['records_sha256'] or not records['pagination_complete']:
        raise ValueError('Changed records or incomplete pagination')
    rows = []
    expected_page = 1
    q = meta['query']
    for f in meta['raw_files']:
        path = (directory/f['file']).resolve()
        if not path.is_relative_to(directory.resolve()) or file_hash(path) != f['sha256']:
            raise ValueError('Changed raw file')
        kind, parts = decode(path.read_bytes())
        if parts[0] != '0':
            raise ValueError('Provider error')
        if f['file'] == 'login.bin':
            assert kind == '01' and parts[2:4] == ['login', 'anonymous']
            continue
        assert kind == '96' and parts[2:4] == ['query_history_k_data_plus', 'anonymous']
        assert int(parts[4]) == expected_page and int(parts[5]) == 2000
        expected_page += 1
        expected_fields = FIELDS if q['api'] == 'daily_features' else [
            FIELDS[j] for j in [*range(10), 11, 17]]
        assert q['api'] in ['daily', 'daily_features'] and records['fields'] == expected_fields
        assert [parts[7], [f.strip() for f in parts[8].split(',')], *parts[9:]] == [
            q['code'], expected_fields, q['start_date'], q['end_date'], 'd', '3']
        page = json.loads(parts[6])['record']
        assert len(page) <= 2000
        rows.extend(page)
    assert rows == records['rows'] and len(rows) == meta['rows']
    assert expected_page-1 == meta['pages'] == records['pages']
    assert len(page) < 2000
    assert len({r[0] for r in rows}) == len(rows)
    for row in rows:
        assert len(row) == len(records['fields']) and row[1] == q['code']
        assert q['start_date'] <= row[0] <= q['end_date'] < '2025-08-07'
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rust-repo', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-attempts', type=int, choices=[1, 2, 3], default=3)
    a = p.parse_args()
    repo, out = a.rust_repo.resolve(), a.output.resolve()
    out.mkdir()
    shutil.copy2(repo/'target/debug/mootdx-cf-rs', out/'collector')
    (out/'code').mkdir()
    for name in ['crates/lakehouse-sources/src/baostock_research.rs', 'src/research_sources.rs', 'src/main.rs', 'Cargo.lock']:
        dest = out/'code'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo/name, dest)
    shutil.copy2(Path(__file__), out/'code/daily_feature_pilot.py')
    queries = [
        dict(api='daily_features', code='sh.600036', start_date='2024-01-02', end_date='2024-01-31'),
        dict(api='daily_features', code='sz.000001', start_date='2024-01-02', end_date='2024-01-31'),
        dict(api='daily_features', code='sh.600053', start_date='2022-03-28', end_date='2022-03-31'),
        dict(api='daily_features', code='sz.000022', start_date='2018-12-26', end_date='2018-12-28')]
    queries += [{**q, 'api':'daily'} for q in queries[:2]]
    for i,q in enumerate(queries):
        write_json(out/f'query-{i:02d}.json', q)
    write_json(out/'capture-policy.json', dict(max_attempts=a.max_attempts,
        backoff_seconds=[5, 15], retry='transport failures only; separate immutable attempts'))
    write_json(out/'frozen-manifest.json', {str(f.relative_to(out)):file_hash(f) for f in out.rglob('*') if f.is_file()})
    write_json(out/'run-status.json', dict(status='running', started_at=utc_now()))
    try:
        evidence = []
        for i,q in enumerate(queries):
            dest = out/f'capture-{i:02d}'
            with (out/f'capture-{i:02d}.log').open('x') as log:
                subprocess.run([str(out/'collector'), 'research-capture-baostock-resilient', str(out/f'query-{i:02d}.json'), str(dest), str(a.max_attempts)],
                    stdout=log, stderr=subprocess.STDOUT, check=True)
            parsed = replay(dest)
            if q['api'] == 'daily_features':
                assert parsed['fields'] == FIELDS
                for row in parsed['rows']:
                    for j in [10, 12, 13, 14, 15, 16]:
                        assert row[j] == '' or np.isfinite(float(row[j]))
                    assert row[10] == '' or float(row[10]) >= 0
                evidence.append(dict(query=q, rows=len(parsed['rows']),
                    missing={FIELDS[j]:sum(r[j] == '' for r in parsed['rows']) for j in [10,13,14,15,16]},
                    suspended=sum(r[11] == '0' for r in parsed['rows'])))
            print('Captured and independently replayed', q['code'], q['api'], len(parsed['rows']), flush=True)
        common = 0
        for features_id, daily_id in [(0,4), (1,5)]:
            extra, base = replay(out/f'capture-{features_id:02d}'), replay(out/f'capture-{daily_id:02d}')
            positions = [extra['fields'].index(f) for f in base['fields']]
            assert [[r[j] for j in positions] for r in extra['rows']] == base['rows']
            common += len(base['rows'])*len(positions)
        for name, expected in read(out/'frozen-manifest.json').items():
            assert file_hash(out/name) == expected
        write_json(out/'verification.json', dict(passed=True, verified_at=utc_now(), samples=evidence,
            common_raw_values=common, scope='Six bounded captures replayed from raw frames, two legacy daily cross-schema controls. Not complete history or original publication-time verification; no BSE or directly observed float cap.',
            source_documentation='https://pypi.org/project/baostock/', executable=False))
        write_json(out/'run-status.json', dict(status='completed', completed_at=utc_now(),
            files={str(f.relative_to(out)):file_hash(f) for f in out.rglob('*') if f.is_file() and f.name != 'run-status.json'}))
    except BaseException as exc:
        write_json(out/'run-status.json', dict(status='failed', error=repr(exc), updated_at=utc_now()))
        raise


if __name__ == '__main__':
    main()
