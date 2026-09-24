"""Close the external benchmark with immutable artifact references."""
from pathlib import Path

from token_external_benchmark import BASE, PRIOR, ROOT, read

from quant_research.storage import file_hash, utc_now, write_json


def main():
    for name in ['run-completed.json', 'independent-verification.json', 'tests.json']:
        assert read(ROOT / name)['passed']
    assert file_hash(ROOT / 'tests.log') == read(ROOT / 'tests.json')['log_sha256']
    note = BASE / 'docs/token-external-benchmark-20260915.md'
    assert note.read_text() == (ROOT / 'report.md').read_text()
    decision = read(ROOT / 'research-decision.json')
    entry = ('2026-09-15: [External benchmark](docs/token-external-benchmark-20260915.md) completed on 76 dates in April–July 2025. '
        + decision['status_summary'] + ' Frozen Kronos Base and historical-volatility baselines each generated 155,648 five-day paths; '
        'three self-built forecast sets were reused exactly. Independent path scores, baseline reconstruction, common cohorts, '
        '576 paired estimates and intervals, 256 exact Kronos reload paths and 14 focused tests passed. '
        'No model training or calibration fitting; the sealed holdout remains unused.\n\n')
    for name in ['README.md', 'STATUS.md']:
        p = BASE / name
        text = p.read_text()
        if entry not in text:
            heading, rest = text.split('\n\n', 1)
            p.write_text(heading + '\n\n' + entry + rest)
    previous = 0
    for name, h in read(PRIOR / 'final-verification.json')['files'].items():
        if Path(name).is_relative_to(BASE / 'artifacts'):
            assert file_hash(Path(name)) == h
            previous += 1
    files = {}
    for done in ROOT.rglob('completed.json'):
        meta = read(done)
        assert meta['passed']
        for name, h in meta['files'].items():
            assert file_hash(done.parent / name) == h
        files[str(done)] = file_hash(done)
    for name in ['protocol.json', 'sources.json', 'prepared.json', 'inputs.npz', 'row-ids.npy', 'rows.parquet',
                 'runtime.json', 'run-completed.json', 'independent-verification.json', 'research-decision.json',
                 'tests.json', 'tests.log', 'report.md']:
        files[str(ROOT / name)] = file_hash(ROOT / name)
    for name in ['token_external_benchmark.py', 'verify_token_external_benchmark.py', 'report_token_external_benchmark.py', 'finalize_token_external_benchmark.py']:
        files[str(BASE / 'scripts' / name)] = file_hash(BASE / 'scripts' / name)
    for name in ['docs/token-external-benchmark-20260915.md', 'README.md', 'STATUS.md']:
        files[str(BASE / name)] = file_hash(BASE / name)
    for p in (ROOT / 'code').rglob('*.py'):
        files[str(p)] = file_hash(p)
    write_json(ROOT / 'final-verification.json', dict(passed=True, at=utc_now(), files=files,
        prior_artifact_files_unchanged=previous, bounded_runs_complete=True, tests_passed=14,
        trained=False, calibration_refitted=False, model_defaults_changed=False, sealed_holdout_opened=False))
    print('Closed external benchmark:', len(files), 'bound files', flush=True)


if __name__ == '__main__':
    main()
