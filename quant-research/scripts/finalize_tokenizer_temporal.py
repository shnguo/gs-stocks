"""Record the completed transfer result without changing model defaults."""
from pathlib import Path

import pandas as pd
from tokenizer_temporal_run import BASE, PRIOR, ROOT, read

from quant_research.storage import file_hash, utc_now, write_json


def main():
    assert read(ROOT/'run-completed.json')['passed']
    assert read(ROOT/'independent-verification.json')['passed']
    assert read(ROOT/'common-draw-sensitivity/independent-verification.json')['passed']
    decision=read(ROOT/'research-decision.json')
    assert not decision['default_model_changed'] and not decision['daily_model_promoted']
    pairs=pd.read_csv(ROOT/'forecast-results/paired.csv',dtype={'seed':str})
    rows=pairs[pairs.seed=='mean'].set_index(['horizon','metric'])
    result=decision['confirmation']
    pre=read(ROOT/'preflight.json')
    entry=(f"2026-09-15: [Fixed-decoder temporal validation](docs/tokenizer-temporal-20260915.md) completed. "
        f"The exact prior decoder and three predictor checkpoints were evaluated on {pre['evaluation_dates']} dates in 2023 H2, "
        f"with {pre['evaluation_inputs']:,} inputs per predictor and no refitting. "
        f"Two/five-day CRPS changes {100*rows.loc[(2,'crps'),'relative_change']:+.2f}% / {100*rows.loc[(5,'crps'),'relative_change']:+.2f}%; "
        f"median high/low/range MAE changes {100*rows.loc[(2,'mae'),'relative_change']:+.2f}% / {100*rows.loc[(5,'mae'),'relative_change']:+.2f}%. "
        f"Prespecified confirmation conditions: two-day {'pass' if result['2']['passes'] else 'not met'}, "
        f"five-day {'pass' if result['5']['passes'] else 'not met'}. "
        "Independent raw-path scores, target-specific intervals, shared-path sensitivity, reloads and lineage checks passed. "
        "This is retrospective fixed-weight transfer evidence. The adapted decoder remains a research candidate; model defaults and sealed holdout are unchanged.\n\n")
    for name in ['README.md','STATUS.md']:
        p=BASE/name
        text=p.read_text()
        if entry not in text:
            head,rest=text.split('\n\n',1)
            p.write_text(head+'\n\n'+entry+rest)
    checked=0
    for name,h in read(PRIOR/'final-verification.json')['files'].items():
        path=Path(name)
        if path.is_relative_to(BASE/'artifacts'):
            assert file_hash(path)==h,path
            checked+=1
    files={}
    for done in ROOT.rglob('completed.json'):
        obj=read(done)
        assert obj['passed']
        for name,h in obj['files'].items():
            assert file_hash(done.parent/name)==h
        files[str(done)]=file_hash(done)
    for name in ['protocol.json','source-manifest.json','preflight.json','splits.json','run-completed.json','decoder-selection/selection.json',
                 'research-decision.json','independent-verification.json','common-draw-sensitivity/independent-verification.json','report.md']:
        files[str(ROOT/name)]=file_hash(ROOT/name)
    for name in ['scripts/tokenizer_temporal_run.py','scripts/verify_tokenizer_temporal.py','scripts/report_tokenizer_temporal.py',
                 'scripts/finalize_tokenizer_temporal.py','scripts/tokenizer_forecast_sensitivity.py','scripts/verify_tokenizer_sensitivity.py',
                 'scripts/verify_token_pipeline_audit.py','scripts/verify_tokenizer_reconstruction.py','scripts/report_tokenizer_reconstruction.py',
                 'configs/tokenizer-temporal-v1.json','docs/tokenizer-temporal-20260915.md','README.md','STATUS.md']:
        files[str(BASE/name)]=file_hash(BASE/name)
    report=(ROOT/'report.md').read_text()
    assert chr(96) not in report
    assert report==(BASE/'docs/tokenizer-temporal-20260915.md').read_text()
    write_json(ROOT/'final-verification.json',dict(passed=True,at=utc_now(),files=files,
        prior_artifact_files_unchanged=checked,weights_refitted=False,model_defaults_changed=False,
        sealed_holdout_opened=False))
    print('Final manifest recorded:',len(files),'files;',checked,'prior artifact files unchanged',flush=True)


if __name__=='__main__':
    main()
