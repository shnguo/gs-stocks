"""Bind the verified research profile and publish the completed project report."""
from pathlib import Path

from token_refreshed_calibration_run import (
    BASE,
    DECODER,
    OLD,
    REFRESH,
    ROOT,
    SEEDS,
    predictor,
    read,
)

from quant_research.storage import file_hash, utc_now, write_json


def main():
    for name in ['run-completed.json', 'independent-verification.json', 'tests.json']:
        assert read(ROOT / name)['passed']
    assert file_hash(ROOT / 'tests.log') == read(ROOT / 'tests.json')['log_sha256']
    cfg = read(ROOT / 'protocol.json')
    decision = read(ROOT / 'research-decision.json')
    profile = dict(status=decision['status'], scope='historical research', model_kind='hierarchical_token_decoder_v1',
        lookback=60, forecast_horizon=5, interval_horizons=[2, 5], nominal_coverage=.8,
        decoder=dict(path=str(DECODER), sha256=file_hash(DECODER)), sampling=cfg['sampling'],
        calibration=cfg['fit'], model_defaults_changed=False, checkpoints={})
    for window in cfg['windows']:
        fit = ROOT / window / 'fit.json'
        obj = read(fit)
        entries = []
        for seed in SEEDS:
            checkpoint = predictor(window, seed) / 'training/best.pt'
            digest = file_hash(checkpoint)
            assert obj['checkpoint_sha256'][str(seed)] == digest
            entries.append(dict(seed=seed, path=str(checkpoint), sha256=digest,
                factor_lookup=dict(seed=seed, remaining_keys=['horizon', 'target'])))
        profile['checkpoints'][window] = dict(predictors=entries, fit_path=str(fit), fit_sha256=file_hash(fit),
            calibration_period=cfg['windows'][window]['calibration']['dates'])
    write_json(ROOT / 'research-profile.json', profile)
    assert (ROOT / 'report.md').read_text() == (BASE / 'docs/token-refreshed-calibration-20260915.md').read_text()
    entry = ('2026-09-15: [Calibration after training refresh](docs/token-refreshed-calibration-20260915.md) completed. '
        + decision['status_summary'] + ' Thirty-six checkpoint-specific factors were fitted on the corresponding H1 forecasts. '
        'Four new validation forecast runs, two verified token reuses, and six reused evaluation sets required no neural-network retraining. '
        'Independent quantiles, factors, score/cohort calculations, 640 paired estimates and confidence intervals passed; '
        'nine focused regression tests passed. A checkpoint-bound research profile is retained.\n\n')
    for name in ['README.md', 'STATUS.md']:
        p = BASE / name
        text = p.read_text()
        if entry not in text:
            heading, rest = text.split('\n\n', 1)
            p.write_text(heading + '\n\n' + entry + rest)
    prior_count = 0
    for prior in [REFRESH, OLD]:
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
    for window in cfg['windows']:
        for name in ['fit.json', 'protocol.json']:
            files[str(ROOT / window / name)] = file_hash(ROOT / window / name)
    for name in ['protocol.json', 'source-manifest.json', 'fits-frozen.json', 'run-completed.json',
                 'independent-verification.json', 'tests.json', 'tests.log', 'research-decision.json',
                 'research-profile.json', 'report.md']:
        files[str(ROOT / name)] = file_hash(ROOT / name)
    for name in ['scripts/token_refreshed_calibration_run.py', 'scripts/verify_token_refreshed_calibration.py',
                 'scripts/report_token_refreshed_calibration.py', 'scripts/finalize_token_refreshed_calibration.py',
                 'configs/token-refreshed-calibration-v1.json', 'docs/token-refreshed-calibration-20260915.md',
                 'README.md', 'STATUS.md']:
        files[str(BASE / name)] = file_hash(BASE / name)
    for path in (ROOT / 'code').rglob('*.py'):
        files[str(path)] = file_hash(path)
    write_json(ROOT / 'final-verification.json', dict(passed=True, at=utc_now(), files=files,
        prior_artifact_files_unchanged=prior_count, bounded_runs_complete=True, tests_passed=9,
        medians_unchanged=True, old_calibration_coefficients_transferred=False,
        model_defaults_changed=False, sealed_holdout_opened=False))
    print('Closed verified calibration experiment:', len(files), 'bound files')


if __name__ == '__main__':
    main()
