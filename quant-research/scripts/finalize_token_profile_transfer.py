"""Close the frozen-profile transfer comparison and preserve prior decisions."""
from pathlib import Path

from token_profile_transfer_run import BASE, OLD, PRIOR, ROOT, read

from quant_research.storage import file_hash, utc_now, write_json


def main():
    for name in ['run-completed.json', 'independent-verification.json', 'tests.json']:
        assert read(ROOT / name)['passed']
    assert file_hash(ROOT / 'tests.log') == read(ROOT / 'tests.json')['log_sha256']
    decision = read(ROOT / 'research-decision.json')
    assert (ROOT / 'report.md').read_text() == (BASE / 'docs/token-profile-transfer-20260915.md').read_text()
    entry = ('2026-09-15: [Frozen-profile transfer](docs/token-profile-transfer-20260915.md) completed on 76 dates in April–July 2025. '
        + decision['status_summary'] + ' Six forecast runs generated 933,888 five-day paths without predictor retraining or calibration refitting. '
        'Independent raw-path scores, matched cohorts, 360 paired estimates and confidence intervals, and six exact token/price replays passed; '
        'nine focused regression tests passed.\n\n')
    for name in ['README.md', 'STATUS.md']:
        p = BASE / name
        text = p.read_text()
        if entry not in text:
            heading, rest = text.split('\n\n', 1)
            p.write_text(heading + '\n\n' + entry + rest)
    prior_count = 0
    for prior in [PRIOR, OLD]:
        for name, digest in read(prior / 'final-verification.json')['files'].items():
            if Path(name).is_relative_to(BASE / 'artifacts'):
                assert file_hash(Path(name)) == digest, name
                prior_count += 1
    files = {}
    for done in ROOT.rglob('completed.json'):
        meta = read(done)
        assert meta['passed']
        for name, digest in meta['files'].items():
            assert file_hash(done.parent / name) == digest
        files[str(done)] = file_hash(done)
    for name in ['protocol.json', 'profiles.json', 'source-manifest.json', 'preflight.json', 'row-ids.npy',
                 'run-completed.json', 'independent-verification.json', 'tests.json', 'tests.log',
                 'research-decision.json', 'runtime.json', 'report.md']:
        files[str(ROOT / name)] = file_hash(ROOT / name)
    for name in ['scripts/token_profile_transfer_run.py', 'scripts/verify_token_profile_transfer.py',
                 'scripts/report_token_profile_transfer.py', 'scripts/finalize_token_profile_transfer.py',
                 'configs/token-profile-transfer-v1.json', 'docs/token-profile-transfer-20260915.md', 'README.md', 'STATUS.md']:
        files[str(BASE / name)] = file_hash(BASE / name)
    for path in (ROOT / 'code').rglob('*.py'):
        files[str(path)] = file_hash(path)
    write_json(ROOT / 'final-verification.json', dict(passed=True, at=utc_now(), files=files,
        prior_artifact_files_unchanged=prior_count, bounded_runs_complete=True, tests_passed=9,
        predictor_refitted=False, calibration_refitted=False, model_defaults_changed=False, sealed_holdout_opened=False))
    print('Closed verified transfer experiment:', len(files), 'bound files')


if __name__ == '__main__':
    main()
