"""Finalize the verified midpoint/width diagnostic and its research decision."""
from pathlib import Path

import pandas as pd
from token_range_decomposition import BASE, PARENT, ROOT, SEEDS, checked, read

from quant_research.storage import file_hash, utc_now, write_json


def table(rows,cols):
    return '\n'.join(['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']+
        ['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    checked()
    proof=read(ROOT/'independent-verification.json')
    assert proof['passed'] and read(ROOT/'tests.json')['passed']
    decision=read(ROOT/'research-decision.json')
    paired=pd.read_csv(ROOT/'paired.csv',dtype={'seed':str})
    attribution=pd.read_csv(ROOT/'attribution-summary.csv',dtype={'seed':str})
    summary=pd.read_csv(ROOT/'summary.csv')
    rows,attrs,mse,seeds=[],[],[],[]
    labels={'center_mae':'Midpoint from endpoint medians','half_width_mae':'Half-width from endpoint medians',
        'path_center_mae':'Median sampled midpoint','path_center_crps':'Sampled midpoint CRPS',
        'path_range_mae':'Median sampled full range','path_range_crps':'Sampled full-range CRPS',
        'endpoint_mae':'Mean high/low endpoint MAE'}
    for h in [2,5]:
        p=paired[(paired.against=='historical')&(paired.seed=='mean')&(paired.horizon==h)].set_index('metric')
        for metric,label in labels.items():
            x=p.loc[metric]
            rows.append([h,label,f'{x.baseline:.4f}',f'{x.candidate:.4f}',f'{100*x.relative_change:+.2f}%',f'[{x.ci_low:+.4f}, {x.ci_high:+.4f}]'])
        a=attribution[(attribution.against=='historical')&(attribution.seed=='mean')&(attribution.horizon==h)].set_index('component')
        attrs.append([h,*[f'{a.loc[k,"delta"]:+.4f}' for k in ['center','half_width','total']],
            f'[{a.loc["center","ci_low"]:+.4f}, {a.loc["center","ci_high"]:+.4f}]',
            f'[{a.loc["half_width","ci_low"]:+.4f}, {a.loc["half_width","ci_high"]:+.4f}]'])
        mse.append([h,*[f'{p.loc[k,"delta"]:+.4f}' for k in ['center_mse','half_width_mse','endpoint_mse']]])
        for seed in SEEDS:
            s=paired[(paired.against=='historical')&(paired.seed==str(seed))&(paired.horizon==h)].set_index('metric')
            seeds.append([seed,h,*[f'{100*s.loc[k,"relative_change"]:+.2f}%' for k in ['center_mae','half_width_mae','path_center_mae','path_range_mae']]])
    bias=[]
    for name in ['historical','ours_mean','kronos']:
        for row in summary[summary.variant==name].itertuples():
            bias.append([name,row.horizon,f'{row.center_error:+.4f}',f'{2*row.half_width_error:+.4f}'])
    body='\n\n'.join([
        '# Midpoint versus width: locating the remaining high/low forecast gap',
        'Date: 2026-09-15 (America/Los_Angeles). Completed frozen-path diagnostic. No new inference, training, calibration or prediction-rule changes.',
        '## Conclusion',decision['conclusion'],decision['limitations'],
        '## Question and matched evidence',
        'The preceding external benchmark found stronger high/low forecasts than Kronos Base, but weaker high/low forecasts than historical volatility despite better sampled-range accuracy. This diagnostic tests whether the remaining error concerns the position of the range or its size.',
        'It reuses the same five sets of 64-path forecasts: three self-built training seeds, frozen Kronos Base and historical volatility. The exact shared cohort has 2,372 two-day and 2,253 five-day stock-date inputs across 76 signal dates, April 1 through July 22, 2025. Labels end by July 29. Every model uses complete known labels and at least 16 legal paths. Stocks are weighted equally within date, then dates equally; self-built means average the three seed scores, not their paths.',
        '## What is being decomposed?',
        'Let H and L be the forecast medians of the future highest high and lowest low. Midpoint C = (H + L)/2; half-width W = (H − L)/2. The realized targets use the actual future high and low. Prices are expressed as percentage-point returns from the signal close. This midpoint describes where the price range sits; it is not the last-day close or a direct measure of up/down trading accuracy.',
        'Writing the midpoint error as eC and half-width error as eW gives exact identities: mean high/low absolute error = max(|eC|, |eW|); mean high/low squared error = eC² + eW². Absolute errors therefore cannot simply be added across the two components.',
        'Illustration only: actual high 12 and low 8 give midpoint 10 and width 4. Forecast high 14 and low 10 also give width 4, but midpoint 12. The width is correct while both endpoints miss by 2.',
        'The median of sampled ranges need not equal median(high) minus median(low); the same distinction applies to midpoint medians. The report checks both definitions, and the independently computed sampled-range scores match the preceding benchmark.',
        '## Comparison with historical volatility',
        'Lower MAE and CRPS are better. Negative relative changes favor the self-built model. All errors and confidence-interval endpoints below are in percentage points of signal close, except percentage changes.',
        table(rows,['Days','Metric','Historical baseline','Self-built mean','Change','95% CI of absolute difference']),
        '## How the components account for the endpoint error gap',
        'For absolute error, start with the historical forecast components and replace them with the self-built components in each of the two possible orders. Average each component’s incremental change across those orders. The two contributions add exactly to the observed endpoint-MAE difference, including component interaction. These are arithmetic contributions, not causal effects of training and not a selected hybrid forecasting rule.',
        table(attrs,['Days','Midpoint contribution','Half-width contribution','Net endpoint error gap','95% CI, midpoint contribution','95% CI, half-width contribution']),
        'A second check uses the exact additive squared-error identity. Units here are squared percentage points; these values cannot be compared directly with the preceding MAE table.',
        table(mse,['Days','Midpoint MSE difference','Half-width MSE difference','Endpoint MSE difference']),
        '## Seed consistency',
        table(seeds,['Seed','Days','Midpoint MAE change','Half-width MAE change','Sampled-midpoint MAE change','Sampled-range MAE change']),
        decision['local_regressions'],
        '## Signed error',
        'Negative midpoint bias means the predicted range is too low on average. Negative width bias means the gap between endpoint medians is too narrow. These are averages over this period, not adjustments to apply to future forecasts.',
        table(bias,['Model','Days','Midpoint bias','Full-width bias from endpoint medians']),
        '## Interpretation and limits',
        'The result locates the observed gap in price-level placement. It does not establish whether the cause is the token objective, training-period mismatch, momentum information, calibration, or another mechanism. A midpoint is affected by drift and high/low asymmetry, so this is not proof of a particular directional-signal failure.',
        'The confidence intervals use 2,000 circular resamples of 10 dates and remain conditional on this previously examined development block, trained seeds and fixed sampled paths. Subgroup comparisons are descriptive and not adjusted for multiplicity. Pretrained tokenizer/predictor date coverage remains unresolved. The results do not establish attainable trades or profitability. Previous gains over the old self-built model and Kronos Base are retained.',
        '## Verification',
        f'Independent raw-path formulas and pairwise CRPS reproduced all {proof["row_components"]:,} model/stock-date/horizon component records. The shared cohort was independently reconstructed from label masks and all five models’ legal-path masks. Every endpoint MAE and sampled-range MAE/CRPS matched the previous benchmark. Both algebraic identities passed row by row. All {proof["attribution_rows"]:,} attribution rows, {proof["paired_estimates"]} paired estimates and {proof["attribution_estimates"]} attributed estimates and their block intervals were independently verified.',
        'Eight focused tests passed, including translation versus width errors, nonadditive component interactions, median noncommutativity and invalid endpoints. All source hashes match. Saved forecasts, model weights, calibration factors and the daily strategy model remain unchanged; the sealed holdout beginning August 7, 2025 remains unopened.',
        '## Research decision and next experiment',decision['next_action'],
        'Artifacts: artifacts/token-range-decomposition-20260915-v1 contains the protocol frozen before scoring, source hashes, frozen code, rows.parquet, daily.csv, summary.csv, paired.csv, attribution rows/daily/summary, test receipt, independent verification and final manifest.',
        'Sources: [external benchmark](token-external-benchmark-20260915.md), [later-period transfer](token-profile-transfer-20260915.md). Component definitions and the averaging/attribution procedures are reproduced in the frozen scripts and src/quant_research/range_decomposition.py.',
    ])+'\n'
    assert chr(96) not in body
    note=BASE/'docs/token-range-decomposition-20260915.md'
    note.write_text(body)
    (ROOT/'report.md').write_text(body)
    entry=('2026-09-15: [Midpoint/width diagnostic](docs/token-range-decomposition-20260915.md) completed. '+decision['status_summary']+
        ' Saved paths only; no inference, training or calibration changes. All 23,125 component records, 27,750 attribution rows, 272 estimates and intervals, and eight focused tests passed independent verification.\n\n')
    for name in ['README.md','STATUS.md']:
        p=BASE/name
        text=p.read_text()
        if entry not in text:
            head,rest=text.split('\n\n',1)
            p.write_text(head+'\n\n'+entry+rest)
    prior_count=0
    for name,h in read(PARENT/'final-verification.json')['files'].items():
        if Path(name).is_relative_to(BASE/'artifacts'):
            assert file_hash(Path(name))==h
            prior_count+=1
    files={str(p):file_hash(p) for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['final-verification.json','run.log','verification.log']}
    for name in ['README.md','STATUS.md','docs/token-range-decomposition-20260915.md','src/quant_research/range_decomposition.py',
        'tests/test_range_decomposition.py','scripts/token_range_decomposition.py','scripts/verify_token_range_decomposition.py','scripts/report_token_range_decomposition.py']:
        files[str(BASE/name)]=file_hash(BASE/name)
    write_json(ROOT/'final-verification.json',dict(passed=True,at=utc_now(),files=files,prior_artifact_files_unchanged=prior_count,
        tests_passed=8,bounded_diagnostic_complete=True,new_inference=False,trained=False,calibration_refitted=False,
        model_defaults_changed=False,sealed_holdout_opened=False))
    print('Verified diagnostic report and final manifest saved:',len(files),'bound files',flush=True)


if __name__=='__main__':
    main()
