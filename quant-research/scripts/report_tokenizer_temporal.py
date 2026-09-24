"""Report the fixed-weight chronological transfer check from verified outputs."""
import pandas as pd
from report_tokenizer_reconstruction import table
from tokenizer_temporal_run import BASE, PRIOR, ROOT, read

from quant_research.storage import file_hash, utc_now, write_json


def main():
    assert read(ROOT/'independent-verification.json')['passed']
    cfg=read(ROOT/'protocol.json')
    pre=read(ROOT/'preflight.json')
    pairs=pd.read_csv(ROOT/'forecast-results/paired.csv',dtype={'seed':str})
    target=pd.read_csv(ROOT/'forecast-results/target-paired.csv',dtype={'seed':str})
    cov=pd.concat([pd.read_csv(ROOT/f'predictors/seed{s}/dense_3720k_equal/decoder-forecast/coverage.csv').assign(seed=s)
        for s in cfg['forecast_seeds']],ignore_index=True)
    cov['legal_fraction']=cov.valid_paths/cov.total_paths
    rules=cfg['confirmation']
    assessments={}
    for h in cfg['horizons']:
        means=pairs[(pairs.seed=='mean')&(pairs.horizon==h)].set_index('metric')
        seed_crps=pairs[(pairs.seed!='mean')&(pairs.horizon==h)&(pairs.metric=='crps')]
        legal=cov[cov.horizon==h].pivot(index='seed',columns='variant',values='legal_fraction')
        checks=dict(all_predictor_seeds_lower_crps=bool((seed_crps.delta<0).all()),
            mean_crps_difference_interval_below_zero=bool(means.loc['crps','ci_high']<0),
            median_mae_guard=bool(means.loc['mae','relative_change']<=rules['maximum_relative_median_mae_increase']),
            interval_score_noninferior=bool(means.loc['interval_score80','delta']<=0),
            legal_draw_fraction_nondecreasing=bool((legal.adapted>=legal.frozen).all()))
        assessments[str(h)]=dict(passes=all(checks.values()),checks=checks,
            crps_relative_change=float(means.loc['crps','relative_change']),
            median_mae_relative_change=float(means.loc['mae','relative_change']))
    decision=dict(at=utc_now(),status='research_candidate_only',confirmation=assessments,
        weights_refitted=False,default_model_changed=False,daily_model_promoted=False,sealed_holdout_opened=False,
        protocol_sha256=file_hash(ROOT/'protocol.json'),evidence_sha256=file_hash(ROOT/'forecast-results/target-paired.csv'))
    write_json(ROOT/'research-decision.json',decision)
    lines=['# Fixed-decoder temporal validation','',
        'Run date: 2026-09-15.','',
        '## Outcome','',
        f"The fixed decoder {'passes' if assessments['2']['passes'] else 'does not pass'} the prespecified two-day confirmation conditions in 2023 H2. Five-day conditions {'pass' if assessments['5']['passes'] else 'do not pass'}. This is a transfer test of the exact prior checkpoints; there is no retraining or reselection.",'',
        '## What was held fixed','',
        '- The price-consistency tokenizer decoder, selected at epoch 15 on 2022 H1 validation. The encoder and binary token semantics are unchanged.',
        '- The three 3,720,448-parameter predictor checkpoints from seeds 17, 29 and 43, fitted on 2016–2021 inputs and selected on 2022 H1 token CE.',
        '- Sixty history bars, five generated days, 64 independent token trajectories per input, temperature 1.0, top-p 1.0 and top-k disabled.',
        '- The two-day result scores the first two days of the same five-day generator. It does not test a separately trained two-day model.',
        '- The same generated token trajectories are decoded by the original and adapted tokenizers. There is no post-decoding price repair.','',
        'The control uses the original Kronos tokenizer with the self-built predictor. This is not a new comparison against the official Kronos Base predictor.','',
        f"Only the evaluation period changes: {pre['first_signal']} through {pre['last_signal']}, {pre['evaluation_dates']} signal dates and {pre['evaluation_inputs']:,} stock/date inputs per predictor. Last target: {pre['last_target']}. Across three predictors, {pre['evaluation_inputs']*64*3:,} generated trajectories are each decoded twice.",'',
        'The new evaluation is disjoint from the 2022 H2 decoder experiment. However, 2023 H2 has already appeared in earlier sampling research and is retrospective development data. Official tokenizer pretraining coverage is unresolved. These qualifications prevent treating it as a pristine test or evidence of live trading performance. The sealed holdout starting 2025-08-07 remains unused.','',
        '## Prespecified interpretation','',
        'Two days is primary. Confirmation requires lower CRPS in all three predictor seeds, a date-block 95% interval below zero for the mean CRPS difference, median MAE degradation no greater than 1%, non-worsening interval score and non-decreasing legal-path fractions. Five days uses the same descriptive checks as a secondary horizon. No daily-model promotion follows automatically. The protocol was saved before forecasts began.','',
        'CRPS measures error across the forecast distribution; MAE measures the error of its median. Price metrics are in percentage points of signal-close price, averaged equally over maximum-high, minimum-low and range. Lower is better. Coverage is the fraction of actual values inside nominal 80% intervals.','',
        '## Forecast results','']
    rows=[]
    for x in pairs[(pairs.seed=='mean')].itertuples(index=False):
        scale=100 if x.metric=='coverage80' else 1
        rows.append([x.horizon,x.metric,f'{x.baseline*scale:.4f}',f'{x.adapted*scale:.4f}',
            f'{100*x.relative_change:+.2f}%',f'[{x.ci_low*scale:+.4f}, {x.ci_high*scale:+.4f}]'])
    lines += [table(['Days','Metric','Original','Adapted','Relative change','95% interval for difference'],rows),'',
        'Coverage values and their difference intervals are percentages and percentage points respectively. Its relative-change column is a relative percentage change, not percentage points.','']
    rows=[]
    for x in pairs[(pairs.seed!='mean')&(pairs.metric=='crps')].itertuples(index=False):
        rows.append([x.seed,x.horizon,f'{100*x.relative_change:+.2f}%',f'[{x.ci_low:+.4f}, {x.ci_high:+.4f}]'])
    lines += [table(['Predictor seed','Days','CRPS change','95% interval for difference'],rows),'',
        '## Maximum-high, minimum-low and range','']
    rows=[]
    for x in target[(target.seed=='mean')&(target.target!='mean')&(target.metric.isin(['crps','mae']))].itertuples(index=False):
        rows.append([x.horizon,x.target,x.metric,f'{x.baseline:.4f}',f'{x.adapted:.4f}',
            f'{100*x.relative_change:+.2f}%',f'[{x.ci_low:+.4f}, {x.ci_high:+.4f}]'])
    lines += [table(['Days','Target','Metric','Original','Adapted','Change','95% difference interval'],rows),'',
        'For the two-day maximum-high and minimum-low point estimates individually, both difference intervals still include zero. The clearest point-accuracy gain is in the range target. The significant average does not establish a separate high-price or low-price point-accuracy improvement.','',
        'Intervals use 2,000 circular resamples of ten trading dates, shared across seeds. They are conditional on these three fitted predictors, one fitted decoder and the sampled stock cohorts. Target-specific intervals are descriptive and are not adjusted for multiple comparisons.','',
        '## Legal paths and coverage','']
    rows=[]
    for (name,h),g in cov.groupby(['variant','horizon']):
        rows.append([name,h,f'{100*g.legal_fraction.mean():.2f}%',f'{100*g.usable_fraction.mean():.2f}%'])
    lines += [table(['Decoder','Days','Legal generated draws','Usable inputs / all inputs'],rows),'',
        'The primary comparison uses common stock/date inputs with at least 16 legal draws under each decoder. Each decoder can retain different draws. The following sensitivity retains exactly the same legal token trajectories under both decoders, also requiring at least 16.','']
    shared=pd.read_csv(ROOT/'common-draw-sensitivity/metrics.csv').groupby(['variant','horizon'])[['crps','mae','coverage80']].mean()
    rows=[]
    for h in cfg['horizons']:
        b,a=shared.loc[('frozen',h)],shared.loc[('adapted',h)]
        rows.append([h,f'{100*(a.crps/b.crps-1):+.2f}%',f'{100*(a.mae/b.mae-1):+.2f}%',
            f'{100*b.coverage80:.2f}% → {100*a.coverage80:.2f}%'])
    lines += [table(['Days','Shared-path CRPS change','Shared-path median MAE change','Shared-path 80% coverage'],rows),'',
        '## Comparison with the preceding window','']
    rows=[]
    for label,root in [('2022 H2',PRIOR),('2023 H2',ROOT)]:
        values=pd.read_csv(root/'forecast-results/paired.csv',dtype={'seed':str})
        for h in cfg['horizons']:
            x=values[(values.seed=='mean')&(values.horizon==h)].set_index('metric')
            rows.append([label,h,f"{100*x.loc['crps','relative_change']:+.2f}%",f"{100*x.loc['mae','relative_change']:+.2f}%",
                f"{100*x.loc['coverage80','baseline']:.2f}% → {100*x.loc['coverage80','adapted']:.2f}%"])
    lines += [table(['Evaluation window','Days','CRPS change','Median MAE change','80% coverage'],rows),'',
        'These are separate windows with exactly the same model weights. Their errors are not pooled into a single independent sample or used to choose a replacement checkpoint.','',
        '## Interpretation and next priority','',
        ('The two-day distribution gain transfers to the later window under the prespecified checks. Its size remains modest, and the earlier window showed little point-accuracy improvement.'
         if assessments['2']['passes'] else
         'The later window does not satisfy all two-day confirmation conditions. Keep the observed reconstruction gain separate from claims of stable forecasting improvement.'),'',
        'Five-day results must be read alongside the earlier inconclusive window and the separate high/low/range rows. A gain in average distribution score does not imply that every price target improves.','',
        'The next priority is prediction-interval calibration with parameters fitted only on validation data, while monitoring high/low median accuracy. Keep this decoder as a research candidate and the existing control available; this transfer check does not establish reliable trading intervals or justify a larger model.','',
        '## Reconstruction diagnostic','']
    rows=[]
    for name in ['frozen','adapted']:
        x=read(ROOT/'reconstruction-evaluation'/name/'summary.json')
        rows.append([name,f"{x['mae']:.4f}",f"{100*x['legal']:.2f}%",f"{x['normalized_mae']:.4f}",f"{x['turnover_mae']:.4f}"])
    lines += [table(['Decoder','Extrema MAE (pp)','Legal reconstructions','Normalized six-field MAE','Normalized volume/amount MAE'],rows),'',
        'Reconstruction uses actual future tokens and is an offline diagnostic. It is not forecast accuracy and did not select this checkpoint.','',
        '## Verification and limits','',
        '- Independently recomputed all raw forecast scores, date means, 128 target/seed contrasts and block intervals. The shared-path sensitivity also passes independent explicit pairwise-CRPS checks.',
        '- Predictor checkpoint copies are byte-identical to the prior experiment. CPU reload logits reproduce the saved references exactly, and generated first-batch replay is exact.',
        '- Training/selection row IDs are unchanged, evaluation periods are disjoint, and all five-day labels end within the specified evaluation boundary.',
        '- The inherited frozen implementation passed 31 regression tests in the preceding experiment. This run verifies unchanged source/checkpoint hashes and adds independent checks of the new outputs.',
        '- One decoder fit and three predictor fits do not measure uncertainty across decoder-training seeds. Fixed old predictors test transfer without refresh; they do not answer whether periodic retraining helps.','',
        '## Artifacts','',f'- [Frozen protocol]({ROOT}/protocol.json)',f'- [Decision and conditions]({ROOT}/research-decision.json)',
        f'- [Independent verification]({ROOT}/independent-verification.json)',f'- [Per-target comparisons]({ROOT}/forecast-results/target-paired.csv)','']
    report=BASE/'docs/tokenizer-temporal-20260915.md'
    report.write_text('\n'.join(lines))
    (ROOT/'report.md').write_text(report.read_text())
    print('Report complete',assessments,flush=True)


if __name__=='__main__':
    main()
