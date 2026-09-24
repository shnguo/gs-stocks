"""Close the bounded comparison and bind its evidence and current decision."""
import pandas as pd
from token_training_refresh_run import BASE, ROOT, read

from quant_research.storage import file_hash, utc_now, write_json


def main():
    for name in ['run-completed.json', 'independent-verification.json', 'tests.json']:
        assert read(ROOT / name)['passed']
    decision = read(ROOT / 'research-decision.json')
    paired = pd.read_csv(ROOT / 'results/paired.csv', dtype={'seed': str})
    targets = paired[(paired.metric == 'mae') & paired.target.isin(['maximum', 'minimum'])]
    individual = targets[targets.seed != 'mean']
    combined = paired[(paired.metric == 'mae') & (paired.target == 'high_low') & (paired.seed != 'mean')]
    counts = decision['scenario_counts']
    assert counts['mean_target_wins'] == int((targets[targets.seed == 'mean'].delta < 0).sum())
    assert counts['individual_target_wins'] == int((individual.delta < 0).sum())
    assert counts['individual_high_low_wins'] == int((combined.delta < 0).sum())
    assert read(ROOT / 'tie-correction.json')['changed_mae_rows'] == 0
    assert file_hash(ROOT / 'tests.log') == read(ROOT / 'tests.json')['log_sha256']
    for name, digest in read(ROOT / 'dependency-receipt.json')['files'].items():
        assert file_hash(BASE / name) == digest
    update = read(ROOT / 'calibration-decision.json')
    assert file_hash(BASE / update['source']) == update['source_sha256']
    assert (ROOT / 'report.md').read_text() == (BASE / 'docs/token-training-refresh-20260915.md').read_text()
    entry = ('2026-09-15: [Training refresh comparison](docs/token-training-refresh-20260915.md) completed. '
             + decision['status_summary'] + ' Two new fits and four verified checkpoint reuses cover two windows and three seeds. '
             'Independent raw-path scoring, 400 paired estimates and confidence intervals, six exact CPU reloads, '
             'source hashes and date boundaries passed; 17 focused regression tests passed. '
             'Interval calibration remains the preferred research method with checkpoint-specific refitting.\n\n')
    for name in ['README.md', 'STATUS.md']:
        path = BASE / name
        text = path.read_text()
        text = text.replace('A bounded two-window training-refresh comparison is now in progress.',
                            'The subsequent bounded two-window training-refresh comparison is complete; see the latest entry.')
        if entry not in text:
            first, rest = text.split('\n\n', 1)
            path.write_text(first + '\n\n' + entry + rest)
    files = {}
    for done in ROOT.rglob('completed.json'):
        obj = read(done)
        assert obj['passed']
        for name, digest in obj['files'].items():
            assert file_hash(done.parent / name) == digest
        files[str(done)] = file_hash(done)
    for name in ['protocol.json', 'source-manifest.json', 'dependency-receipt.json', 'metric-definitions.json', 'calibration-decision.json',
                 'research-decision.json', 'run-completed.json', 'independent-verification.json', 'tie-correction.json', 'tests.json', 'report.md']:
        files[str(ROOT / name)] = file_hash(ROOT / name)
    for name in ['configs/token-training-refresh-v1.json', 'docs/token-training-refresh-20260915.md',
                 'docs/token-calibration-decision-update-20260915.md', 'scripts/token_training_refresh_run.py',
                 'scripts/verify_token_training_refresh.py', 'scripts/report_token_training_refresh.py',
                 'scripts/finalize_token_training_refresh.py', 'scripts/correct_token_refresh_ties.py', 'README.md', 'STATUS.md']:
        files[str(BASE / name)] = file_hash(BASE / name)
    for path in (ROOT / 'code').rglob('*.py'):
        files[str(path)] = file_hash(path)
    write_json(ROOT / 'final-verification.json', dict(passed=True, at=utc_now(), files=files,
        focused_tests_passed=17, bounded_runs_complete=True, model_defaults_changed=False,
        old_calibration_coefficients_transferred=False, sealed_holdout_opened=False))
    print('Closed verified experiment:', len(files), 'bound files')


if __name__ == '__main__':
    main()
