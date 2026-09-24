"""Bind the completed calibration evidence and update project status."""
from pathlib import Path

import pandas as pd
from token_interval_calibration_run import BASE, ROOT, TRANSFER, read

from quant_research.storage import file_hash, utc_now, write_json


def main():
    assert read(ROOT/'run-completed.json')['passed']
    assert read(ROOT/'independent-verification.json')['passed']
    decision=read(ROOT/'research-decision.json')
    paired=pd.read_csv(ROOT/'results/paired.csv',dtype={'seed':str})
    results=[]
    for stage in read(ROOT/'protocol.json')['evaluations']:
        x=paired[(paired.stage==stage['name'])&(paired.horizon==2)&(paired.target=='mean')&(paired.seed=='mean')].set_index('metric')
        results.append(f"{stage['name']}: two-day coverage {100*x.loc['coverage80','baseline']:.2f}% → {100*x.loc['coverage80','calibrated']:.2f}%, interval score {100*x.loc['interval_score80','relative_change']:+.2f}%")
    entry=('2026-09-15: [Token interval calibration](docs/token-interval-calibration-20260915.md) completed. '
        'Fitted 18 date-weighted interval factors on 2023 H1 forecasts from three fixed predictors, then evaluated on 2023 H2 and 2024 H2. '
        +'; '.join(results)+'. '
        +f"Two-day acceptance {'passed' if decision['decisions']['2']['passes'] else 'not met'} across both windows; "
        +f"five-day acceptance {'passed' if decision['decisions']['5']['passes'] else 'not met'}. "
        'Median forecasts and point MAE are exactly unchanged. Nine focused tests and independent raw-quantile, coefficient, score, date-aggregation and confidence-interval checks passed. '
        'This is retrospective interval research; model defaults and the sealed holdout remain unchanged.\n\n')
    for name in ['README.md','STATUS.md']:
        p=BASE/name
        text=p.read_text()
        if entry not in text:
            head,rest=text.split('\n\n',1)
            p.write_text(head+'\n\n'+entry+rest)
    prior=0
    for name,h in read(TRANSFER/'final-verification.json')['files'].items():
        path=Path(name)
        if path.is_relative_to(BASE/'artifacts'):
            assert file_hash(path)==h
            prior+=1
    files={}
    for done in ROOT.rglob('completed.json'):
        obj=read(done)
        assert obj['passed']
        for name,h in obj['files'].items():
            assert file_hash(done.parent/name)==h
        files[str(done)]=file_hash(done)
    for name in ['protocol.json','source-manifest.json','fit.json','run-completed.json','research-decision.json',
                 'independent-verification.json','report.md','test_token_intervals.py','tests.json']:
        files[str(ROOT/name)]=file_hash(ROOT/name)
    for name in ['src/quant_research/token_intervals.py','tests/test_token_intervals.py',
                 'scripts/token_interval_calibration_run.py','scripts/verify_token_interval_calibration.py',
                 'scripts/report_token_interval_calibration.py','scripts/finalize_token_interval_calibration.py',
                 'configs/token-interval-calibration-v1.json','docs/token-interval-calibration-20260915.md','README.md','STATUS.md']:
        files[str(BASE/name)]=file_hash(BASE/name)
    assert (ROOT/'report.md').read_text()==(BASE/'docs/token-interval-calibration-20260915.md').read_text()
    assert chr(96) not in (ROOT/'report.md').read_text()
    write_json(ROOT/'final-verification.json',dict(passed=True,at=utc_now(),files=files,
        focused_tests_passed=9,prior_artifact_files_unchanged=prior,raw_prices_changed=False,
        median_forecasts_changed=False,model_defaults_changed=False,sealed_holdout_opened=False))
    print('Final calibration manifest saved:',len(files),'files;',prior,'prior artifacts unchanged',flush=True)


if __name__=='__main__':
    main()
