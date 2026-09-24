"""Render empirical interval-calibration results and prespecified acceptance."""
from pathlib import Path

import numpy as np
import pandas as pd
from report_tokenizer_reconstruction import table
from token_interval_calibration_run import BASE, ROOT, read

from quant_research.storage import file_hash, utc_now, write_json


def main():
    assert read(ROOT/'independent-verification.json')['passed']
    cfg=read(ROOT/'protocol.json')
    fitted=read(ROOT/'fit.json')
    paired=pd.read_csv(ROOT/'results/paired.csv',dtype={'seed':str})
    band=cfg['acceptance']['coverage_band']
    decisions={}
    for h in cfg['horizons']:
        checks={}
        for stage in cfg['evaluations']:
            p=paired[(paired.stage==stage['name'])&(paired.horizon==h)]
            mean=p[(p.seed=='mean')&(p.target=='mean')].set_index('metric')
            cover=p[(p.seed=='mean')&(p.target!='mean')&(p.metric=='coverage80')]
            seeds=p[(p.seed!='mean')&(p.target=='mean')&(p.metric=='interval_score80')]
            checks[stage['name']]=dict(
                each_target_coverage_in_band=bool(cover.calibrated.between(*band).all()),
                mean_interval_score_improves_with_interval=bool(mean.loc['interval_score80','ci_high']<=0),
                each_seed_interval_score_noninferior=bool((seeds.delta<=0).all()),
                median_mae_unchanged=bool(mean.loc['mae','delta']==0))
        decisions[str(h)]=dict(passes=all(all(x.values()) for x in checks.values()),checks=checks)
    write_json(ROOT/'research-decision.json',dict(at=utc_now(),status='research_only',decisions=decisions,
        calibration_fit_sha256=file_hash(ROOT/'fit.json'),default_model_changed=False,daily_model_promoted=False,
        raw_paths_changed=False,median_forecasts_changed=False,sealed_holdout_opened=False,
        next_check='Compare refreshed training history against the fixed old predictors on maximum-high and minimum-low forecast accuracy; this is a proposed experiment, not a demonstrated cause.'))
    lines=['# Token forecast interval calibration','',
        'Run date: 2026-09-15.','',
        '## Outcome','',
        f"The fixed interval calibration {'passes' if decisions['2']['passes'] else 'does not pass'} the two-day acceptance conditions across both evaluation windows. Five-day conditions {'pass' if decisions['5']['passes'] else 'do not pass'}. Model weights, generated prices and median forecasts are unchanged.",'',
        '## Experiment','',
        'The preceding decoder experiment improved two-day distribution forecasts, while nominal 80% interval coverage remained below 70% in the later window. This experiment fits interval widths to observed validation errors and checks whether the improvement survives later dates.','',
        '- Fit on 2023 H1 only: 16 stock/date inputs per signal date, all three fixed predictor seeds, and the previously selected price-consistency decoder.',
        '- Evaluate on 2023 H2 using retained raw paths and on 2024 H2 using newly generated paths, with 32 inputs per date and predictor.',
        '- Keep 60 history bars, five generated days, 64 sampled paths, temperature 1.0 and top-p 1.0. The two-day metric scores the first two forecast days.',
        '- Both interval variants use exactly the same adapted-decoder paths and usable inputs: known full-horizon labels and at least 16 legal draws. Earlier original/adapted comparisons used their intersection of usable inputs, so their aggregate numbers need not match this raw control.','',
        'The three predictors were fitted on 2016–2021 and selected on 2022 H1. No neural weights are refitted here. Calibration is a separate statistical transformation of interval endpoints, not a new prediction head or a correction to generated OHLC bars.','',
        '## Calibration method','',
        'For each predictor seed, horizon and target, compute how far each validation outcome lies from the predicted median, divided by the corresponding original half-width. Give each signal date equal total weight and take the empirical 80th percentile of these scores, with a minimum factor of one. This yields 18 fixed widening factors. A numerical half-width floor of 0.000001 percentage points handles degenerate intervals.','',
        'Apply each factor to the original lower and upper half-widths. Keep the median exactly fixed. Intersect the resulting intervals with the known mathematical support: nonnegative range and returns no lower than -100%. This intersection cannot remove a valid observed target; it is not a repair to the generated prices.','',
        'Illustration only: an original price interval of 100–104 with median 102 becomes 99.5–104.5 under a fitted factor of 1.25. The point forecast remains 102; the interval expresses greater uncertainty.','',
        'The procedure targets total interval coverage. It does not guarantee 10% errors in each tail, so lower and upper misses are reported separately. It does not define a newly calibrated full distribution, and no improved CRPS is claimed. Serial dependence, shared market shocks and retrospective choices prevent an iid or conformal coverage guarantee.','',
        'Acceptance requires 75–85% coverage for every target, lower mean interval score with its date-block confidence interval at or below zero, non-worsening interval score in all predictor seeds, and identical median MAE, separately in both windows. Two days is primary; five days has a separate secondary decision. Wider intervals incur a width penalty in the interval score.','',
        '## Evaluation size and eligibility','']
    rows=[]
    for stage in [cfg['calibration']]+cfg['evaluations']:
        dest=ROOT/'intervals'/stage['name']
        source=Path(read(dest/'source.json')['source'])
        q=pd.read_parquet(dest/'quantiles.parquet')
        for h in [2,5]:
            cov=pd.concat([pd.read_csv(source/f'predictors/seed{s}/dense_3720k_equal/decoder-forecast/coverage.csv')
                for s in cfg['forecast_seeds']])
            cov=cov[(cov.variant=='adapted')&(cov.horizon==h)]
            count=len(np.load(source/'predictors/seed17/dense_3720k_equal/decoder-forecast/row-ids.npy'))
            rows.append([stage['name'],h,q.date.nunique(),count,f'{cov.usable.mean():.1f}',
                f'{100*cov.usable_fraction.mean():.2f}%',f'{100*(cov.valid_paths/cov.total_paths).mean():.2f}%'])
    lines += [table(['Window','Days','Signal dates','Inputs / predictor','Mean usable / predictor','Usable fraction','Legal draws'],rows),'',
        'Calibration uses fewer stocks per date to bound its inference cost while retaining every signal date. All retained rows contribute to the date-weighted fit; evaluation keeps the previously used 32-input-per-date design.','',
        '## Frozen factors','']
    rows=[]
    for p in fitted['parameters']:
        rows.append([p['seed'],p['horizon'],p['target'],f"{p['scale']:.4f}",p['rows'],p['dates'],p['degenerate_halfwidths']])
    lines += [table(['Seed','Days','Target','Width factor','Fit rows','Fit dates','Degenerate half-width rows'],rows),'',
        '## Evaluation averages','']
    rows=[]
    for stage in cfg['evaluations']:
        for h in [2,5]:
            p=paired[(paired.stage==stage['name'])&(paired.horizon==h)&(paired.target=='mean')&(paired.seed=='mean')].set_index('metric')
            rows.append([stage['name'],h,
                f"{100*p.loc['coverage80','baseline']:.2f}% → {100*p.loc['coverage80','calibrated']:.2f}%",
                f"{100*p.loc['width80','relative_change']:+.2f}%",
                f"{p.loc['interval_score80','baseline']:.4f} → {p.loc['interval_score80','calibrated']:.4f}",
                f"{100*p.loc['interval_score80','relative_change']:+.2f}%",
                f"[{p.loc['interval_score80','ci_low']:+.4f}, {p.loc['interval_score80','ci_high']:+.4f}]"])
    lines += [table(['Window','Days','80% coverage','Width change','Interval score','Score change','95% difference interval'],rows),'',
        'Lower interval score is better. Metrics average within each signal date, then equally over dates, targets and predictor seeds. Price widths and errors are in percentage points of signal-close price.','',
        '## High, low and range separately','']
    rows=[]
    for stage in cfg['evaluations']:
        daily=pd.read_csv(ROOT/'intervals'/stage['name']/'daily.csv')
        tails=daily.groupby(['variant','horizon','target'])[['lower_miss','upper_miss']].mean()
        for h in [2,5]:
            for target in ['maximum','minimum','range']:
                p=paired[(paired.stage==stage['name'])&(paired.horizon==h)&(paired.target==target)&(paired.seed=='mean')].set_index('metric')
                miss=tails.loc[('calibrated',h,target)]
                rows.append([stage['name'],h,target,
                    f"{100*p.loc['coverage80','baseline']:.2f}% → {100*p.loc['coverage80','calibrated']:.2f}%",
                    f"{100*p.loc['interval_score80','relative_change']:+.2f}%",
                    f'{100*miss.lower_miss:.2f}% / {100*miss.upper_miss:.2f}%'])
    lines += [table(['Window','Days','Target','Coverage','Interval score change','Lower / upper miss'],rows),'',
        '## Check conditions','']
    rows=[]
    for h,decision in decisions.items():
        for stage,checks in decision['checks'].items():
            rows.append([h,stage,*checks.values()])
    lines += [table(['Days','Window','Every target 75–85%','Score difference interval ≤ 0','Every seed non-worse','Identical median MAE'],rows),'',
        '## Interpretation and next direction','',
        'Coverage improves for every target in both windows. The tradeoff is approximately 27% wider two-day intervals and 19% wider five-day intervals. Mean interval scores improve by roughly 1–2%, but their 2024 H2 difference intervals include zero. The low-price interval score worsens slightly in that window: 0.65% at two days and 0.35% at five days. Better coverage alone therefore does not satisfy the acceptance gate.','',
        'Retain the fitted factors and all evidence as a research candidate. They are tied to these exact predictor checkpoints, decoder and sampling settings, and should not be applied automatically to a different model. Neither the daily model nor its defaults are changed.','',
        'The next controlled question is whether refreshing the training history improves maximum-high and minimum-low point forecasts. The predictors were deliberately held at their 2016–2021 fits to isolate decoder and calibration changes. A matched comparison with refreshed fits can test data freshness; this experiment does not establish staleness as the cause of the remaining errors.','',
        '## Verification and limitations','',
        '- Nine focused tests passed, including date-weight invariance, median preservation, degenerate intervals, support intersection and the width penalty.',
        '- Raw quantiles and actual high/low/range targets are independently reconstructed from retained sampled paths. Fitted scales satisfy independently calculated date-weighted quantile conditions.',
        '- Interval endpoints, all six metric fields, date means, 256 comparisons and confidence intervals are independently checked. Median forecasts and their MAE match exactly before and after calibration.',
        '- All fits use only the 2023 H1 forecast errors. Coefficients are frozen before applying them to either evaluation period. New generation replays its first batch exactly; original checkpoints reload with identical CPU logits.',
        '- Confidence intervals resample 2,000 circular blocks of ten trading dates. They are conditional on the fixed models, selected stocks and fitted calibration factors; they do not include uncertainty from refitting the calibration.',
        '- These are retrospective development windows that have appeared in earlier research. Official tokenizer pretraining coverage remains unresolved. The sealed holdout starting 2025-08-07 is unused.',
        '- This tests interval calibration, not improved high/low point estimates, a retrained two-day model, or the official Kronos Base predictor.','',
        '## Artifacts','',f'- [Frozen protocol]({ROOT}/protocol.json)',f'- [Fitted factors]({ROOT}/fit.json)',
        f'- [Independent verification]({ROOT}/independent-verification.json)',f'- [Research decision]({ROOT}/research-decision.json)',
        f'- [Detailed comparisons]({ROOT}/results/paired.csv)','']
    report=BASE/'docs/token-interval-calibration-20260915.md'
    report.write_text('\n'.join(lines))
    (ROOT/'report.md').write_text(report.read_text())
    print('Interval report complete',decisions,flush=True)


if __name__=='__main__':
    main()
