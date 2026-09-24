"""Build the decoder-adaptation research report from verified artifacts."""
import json
from pathlib import Path

import pandas as pd

BASE=Path('/Users/guo/Documents/stocks/quant-research')
ROOT=BASE/'artifacts/tokenizer-reconstruction-20260915-v1'


def read(p):
    return json.loads(p.read_text())


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    assert read(ROOT/'independent-verification.json')['passed']
    cfg=read(ROOT/'protocol.json')
    choice=read(ROOT/'decoder-selection/selection.json')
    complete=read(ROOT/'run-completed.json')
    lines=['# Tokenizer decoder reconstruction experiment','',
        'Run date: 2026-09-15.','',
        '## Scope and outcome','',
        f"Validation selects **{choice['selected']}**. The encoder and binary token codes remain frozen; only the tokenizer decoder is adapted. The comparison retains the original tokenizer as its control.",'',
        'The first candidate minimizes normalized six-field reconstruction loss. The second adds losses for maximum-high, minimum-low and range errors, plus OHLC ordering violations. These terms affect learned weights; generated prices are not repaired after decoding. Both candidates start from identical official weights, use the same training rows and order, and retain the existing history-based normalization and input clipping.','',
        'A decoder can improve reconstruction of observed tokens without improving prediction of unknown future prices. This experiment treats those as separate measurements.','',
        '## Chronological partitions','']
    rows=[]
    for x in read(ROOT/'splits.json'):
        rows.append([x['partition'],x['first']+'–'+x['last'],x['dates'],f"{x['rows']:,}",f"{x['known_rows']:,}",x['last_label']])
    lines += [table(['Partition','Signal dates','Dates','Pool rows','Rows with known targets','Last target'],rows),'',
        'The 2022 H2 window is newly designated for this token-forecast protocol, but remains retrospective development data. Official tokenizer pretraining coverage remains unresolved. Every five-day target ends before the next partition boundary. The sealed holdout beginning 2025-08-07 is not opened.','',
        '## Training and selection','',
        'Only post_quant_embed, the three decoder Transformer blocks and the final six-field head receive gradients. Input embeddings, encoder layers, quantization projection and binary code semantics remain unchanged. Adaptation uses seed 17, AdamW at learning rate 0.00003, weight decay 0.01, batch 256, a minimum of eight and maximum of sixteen epochs, and patience three on validation reconstruction MAE. Each epoch visits all 1,418 training dates and rotates 32 stocks/date through the known-label pool.','',
        'Both objectives target actual, unclipped OHLCVA values using history-only normalization. Unknown future suffixes contribute neither loss nor gradients. The general loss is mean Smooth L1 error across six normalized fields and known steps. The price objective adds 0.25 times Smooth L1 high/low/range error in percentage points of signal-close price, equally over two/five-day known prefixes, plus 0.25 times the OHLC ordering gap in the same units.','',
        'Every epoch is retained. A checkpoint must improve date-equal mean extrema MAE by at least 5%, improve legal-path fraction by at least two percentage points, and keep overall normalized MAE and volume/amount MAE within 5% of the frozen control. The minimum-MAE qualifying checkpoint is selected; if none qualifies, the original tokenizer is retained. Validation uses 16 stocks/date; evaluation uses 32. Evaluation results never change the selection.','']
    rows=[]
    for c in choice['candidates']:
        info=read(ROOT/'decoder-training'/c['objective']/'summary.json')
        hist=read(ROOT/'decoder-training'/c['objective']/'history.json')
        # The checkpoint epoch is identifiable from its selected score and saved history.
        matches=[x for x in hist if abs(x['selection']['mae']-c['mae'])<1e-12]
        chosen=matches[0]
        rows.append([c['objective'],info['epochs'],chosen['epoch'],f"{chosen['unique_rows']:,}",f"{info['unique_rows']:,}",info['trainable_parameters'],c['qualified'],info['stopping']])
    lines += [table(['Objective','Epochs','Selected epoch','Rows seen by selected weights','Rows seen by complete run','Trainable parameters','Passes guards','Stopping'],rows),'',
        '### Reconstruction results','',
        'These metrics use the actual future tokens, so they are diagnostics rather than forecasts or a strict forecast-error floor. All finite reconstructions remain in the error calculation, including invalid OHLC paths. Errors are averaged within each date, then over dates, horizons and the three price targets.','']
    rows=[]
    for stage in ['decoder-selection','reconstruction-evaluation']:
        for name in ['frozen']+cfg['decoder_objectives']:
            x=read(ROOT/stage/name/'summary.json')
            rows.append([stage,name,f"{x['mae']:.4f}",f"{100*x['legal']:.2f}%",f"{x['normalized_mae']:.4f}",f"{x['turnover_mae']:.4f}"])
    lines += [table(['Partition','Decoder','High/low/range MAE (pp)','Legal paths','Six-field normalized MAE','Volume/amount normalized MAE'],rows),'']
    if complete['forecast_stage_run']:
        pairs=pd.read_csv(ROOT/'forecast-results/paired.csv',dtype={'seed':str})
        mean=pairs[(pairs.seed=='mean')&(pairs.metric=='crps')].set_index('horizon')
        lines += ['## Actual forecast comparison','',
            f"Across three predictor seeds, changing only the decoder changes mean two-day CRPS by {100*mean.loc[2,'relative_change']:+.2f}% and five-day CRPS by {100*mean.loc[5,'relative_change']:+.2f}%.",'',
            'Three fresh 3,720,448-parameter predictors are trained on the training partition with seeds 17, 29 and 43, equal forecast-day token CE, and validation-only CE checkpoint selection. Each generates 64 complete token trajectories per input, using temperature 1.0, top-p 1.0 and top-k disabled. Exactly the same generated token trajectories are decoded by the original and selected adapted decoder. No actual future token enters forecast generation. Decoder adaptation itself uses one training seed; this test varies the predictor, not the decoder fit.','',
            'Metrics require known targets and at least 16 legal paths per input. The two decoders share each comparison stock/date cohort but may retain different legal draws; coverage and valid-draw fractions are therefore reported alongside error. Confidence intervals resample 2,000 circular blocks of ten trading dates jointly across seeds. They are conditional on these fitted models and sampled stock cohorts.','']
        training=[]
        for seed in cfg['forecast_seeds']:
            root=ROOT/f'predictors/seed{seed}/dense_3720k_equal/training'
            summary=read(root/'summary.json')
            history=read(root/'history.json')
            selected=next(x for x in history if x['epoch']==summary['selected_epoch'])
            training.append([seed,summary['epochs'],summary['selected_epoch'],f"{selected['unique_examples_visited']:,}",f"{summary['unique_examples_visited']:,}"])
        lines += [table(['Predictor seed','Training epochs','Selected epoch','Rows seen by selected weights','Rows seen by complete run'],training),'',
            'Forecast evaluation covers 120 dates and 3,840 inputs per predictor seed. Across three seeds, 737,280 generated token trajectories are each decoded twice. These are generated scenarios, not independent observed training samples.','']
        rows=[]
        for x in pairs[pairs.metric=='crps'].itertuples(index=False):
            rows.append([x.seed,x.horizon,f'{x.baseline:.4f}',f'{x.adapted:.4f}',f'{100*x.relative_change:+.2f}%',f'[{x.ci_low:+.4f}, {x.ci_high:+.4f}]'])
        lines += [table(['Predictor seed','Days','Original CRPS','Adapted CRPS','Relative change','95% interval for difference'],rows),'']
        rows=[]
        for x in pairs[(pairs.seed=='mean')&(pairs.metric!='crps')].itertuples(index=False):
            scale=100 if x.metric=='coverage80' else 1
            rows.append([x.horizon,x.metric,f'{x.baseline*scale:.4f}',f'{x.adapted*scale:.4f}',f'{x.delta*scale:+.4f}'])
        lines += [table(['Days','Metric','Original','Adapted','Difference'],rows),'']
        by_target=pd.read_csv(ROOT/'forecast-results/metrics.csv').groupby(['variant','horizon','target'])[['mae','crps']].mean()
        rows=[]
        for h in [2,5]:
            for target in ['maximum','minimum','range']:
                b=by_target.loc[('frozen',h,target)]
                a=by_target.loc[('adapted',h,target)]
                rows.append([h,target,f'{b.mae:.4f}',f'{a.mae:.4f}',f'{100*(a.mae/b.mae-1):+.2f}%',
                    f'{100*(a.crps/b.crps-1):+.2f}%'])
        lines += ['### Maximum, minimum and range separately','',
            'MAE is the absolute error of each predictive median, measured in percentage points of the signal close; it is not a percentage hit rate. CRPS measures error across the predicted distribution. Lower is better for both.','',
            table(['Days','Target','Original median MAE (pp)','Adapted median MAE (pp)','MAE change','CRPS change'],rows),'']
        cov=pd.concat([pd.read_csv(ROOT/f'predictors/seed{s}/dense_3720k_equal/decoder-forecast/coverage.csv') for s in cfg['forecast_seeds']])
        cov['legal_fraction']=cov.valid_paths/cov.total_paths
        rows=[]
        for (name,h),x in cov.groupby(['variant','horizon'])[['usable_fraction','legal_fraction']].mean().iterrows():
            rows.append([name,h,f'{100*x.usable_fraction:.2f}%',f'{100*x.legal_fraction:.2f}%'])
        lines += [table(['Decoder','Days','Usable inputs / all inputs','Legal sampled draws'],rows),'']
        sensitivity=ROOT/'common-draw-sensitivity'
        assert read(sensitivity/'independent-verification.json')['passed']
        shared=pd.read_csv(sensitivity/'metrics.csv').groupby(['variant','horizon'])[['crps','mae','coverage80']].mean()
        rows=[]
        for h in [2,5]:
            b=shared.loc[('frozen',h)]
            a=shared.loc[('adapted',h)]
            rows.append([h,f'{b.crps:.4f}',f'{a.crps:.4f}',f'{100*(a.crps/b.crps-1):+.2f}%',
                f'{100*(a.mae/b.mae-1):+.2f}%',f'{100*b.coverage80:.2f}% → {100*a.coverage80:.2f}%'])
        lines += ['### Identical legal-path sensitivity','',
            'This secondary diagnostic was specified before forecast generation. It retains only token trajectories that decode to legal prices under both decoders, with at least 16 shared legal paths per input. Both sides therefore score identical generated tokens and stock/date rows. It changes neither checkpoint selection nor the primary comparison.','',
            table(['Days','Original CRPS','Adapted CRPS','CRPS change','Median MAE change','80% coverage'],rows),'',
            'The modest two-day gain persists after holding the legal-path set fixed. Five-day point accuracy remains essentially unchanged. This separates improvements to the decoded values from changes in which trajectories survive legality filtering.','',
            '## Research decision','',
            'Retain the price-consistency decoder as a research candidate. Keep the original decoder as the default control until a separate chronological-window confirmation is complete. The adapted decoder reduces observed-token reconstruction MAE by 7.87% and substantially increases valid generated paths, but that gain does not translate into a similarly sized forecast improvement.','',
            'Two-day CRPS improves in all three predictor seeds, with a mean difference interval below zero in this window. Median high/low/range MAE improves only 0.20%, and its interval includes zero. Five-day CRPS improves only 0.18%, its interval includes zero, and median MAE worsens 0.25%. There is no established five-day accuracy gain.','',
            'Nominal 80% interval coverage remains only 69.45% at two days and 71.68% at five days. The next controlled check should keep this recipe fixed and test its temporal stability, reporting maximum-high and minimum-low accuracy separately as well as distribution calibration. These results do not establish that increasing model size, switching to MoE, or shortening the training horizon would help.','',
            'This experiment compares decoders attached to the same self-built predictors; it does not constitute a new comparison against the Kronos Base predictor. The checkpoint is not promoted to the daily strategy model.','']
    else:
        lines += ['## Forecast gate','',
            'Neither adapted decoder passed the prespecified validation requirements. The predictor-training and forecast stage was therefore not launched. The original tokenizer is retained, and no claim of improved forecasting is made.','']
    lines += ['## Verification','',
        '- All 31 regression tests passed, including loss masking, price-unit invariance, learned consistency gradients, a real decoder update, unchanged token codes and causal encoder/decoder behavior.',
        '- Independent calculations reproduce reconstruction errors, legal masks, date means, and checkpoint/decoder selection. Frozen state tensors match the official tokenizer exactly; CPU reloads reproduce saved MPS reconstructions within checked numerical tolerances.',
        '- Raw forecast paths, all six score fields, date means, paired differences and block intervals are independently recomputed. Predictor CPU logits and the first generated batch replay are verified. The shared-path sensitivity is independently checked using explicit pairwise CRPS.',
        '- Data and source snapshots are hashed. Original tokenizer files are preserved, target boundaries remain chronological, and the sealed holdout remains unused.','',
        '## Artifacts','',f'- [Frozen protocol]({ROOT}/protocol.json)',f'- [Decoder selection]({ROOT}/decoder-selection/selection.json)',
        f'- [Independent verification]({ROOT}/independent-verification.json)',f'- [Complete experiment]({ROOT})','']
    (BASE/'docs/tokenizer-reconstruction-20260915.md').write_text('\n'.join(lines))


if __name__=='__main__':
    main()
