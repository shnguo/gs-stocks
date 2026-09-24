"""Assemble a source-linked report after all experiment and verification stages pass."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE=Path('/Users/guo/Documents/stocks/quant-research')
AUDIT=BASE/'artifacts/token-pipeline-audit-20260915-v1'
CONF=BASE/'artifacts/token-pipeline-confirmation-20260915-v1'


def read(p):
    return json.loads(p.read_text())


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(str(x) for x in row)+' |' for row in rows])


def main():
    for root in [AUDIT,CONF]:
        assert read(root/'independent-verification.json')['passed']
    assert read(CONF/'model-verification.json')['passed']
    assert read(CONF/'run-completed.json')['passed']
    recon=pd.read_csv(AUDIT/'reconstruction/metrics.csv')
    paired=pd.read_csv(CONF/'results/paired-comparisons.csv',dtype={'seed':str})
    audit_choice=read(AUDIT/'sampling/selection.json')
    chrono_choice=read(CONF/'sampler-selection/selection.json')
    mean_sampling=paired[(paired.seed=='mean')&(paired.variant=='sampling')&(paired.metric=='crps')].set_index('horizon')
    outcome=(f"Full-distribution sampling changes mean two-day CRPS by {100*mean_sampling.loc[2,'relative_change']:+.2f}% "
             f"and five-day CRPS by {100*mean_sampling.loc[5,'relative_change']:+.2f}% across three training seeds. "
             'This is a distribution-quality result; median-price accuracy and invalid-path costs are reported separately.')
    lines=['# Tokenizer, sampling and price-checkpoint experiment','',
        'Run dates: 2026-09-14–15 PDT / 2026-09-15 UTC.','',
        '## Outcome','',outcome,'',
        'Representation reconstruction is the next targeted optimization to test. Removing clipping did not help the frozen tokenizer, and wider sampling retains a cost in invalid paths. These diagnostics justify a controlled tokenizer comparison; they do not prove that tokenizer fine-tuning will improve forecasts. Additional market features, expanded stock/history sampling and capacity changes were not run in this stage.','',
        '## Research setting','',
        'Use the saved research preset: 64 paths, temperature 1.0, top-p 1.0 and top-k disabled. The preset contains the sampling keyword arguments accepted by generate_tokens. Keep equal forecast-day loss, the existing clipping threshold and CE checkpoint selection. The daily strategy model is not promoted by this experiment.','',
        f'[Sampling preset]({BASE}/configs/token-sampling-research-v2.json)','',
        'Nominal 80% intervals still cover only 64.6% of two-day and 67.1% of five-day outcomes in the confirmation. Mean legal five-day draw rate falls from 87.3% to 84.6%. These limitations remain despite lower CRPS.','',
        '## Completed scope','',
        'Completed the staged pipeline audit, validation-only sampling comparisons and three-seed chronological confirmation. The existing 3,720,448-parameter causal token decoder, 60-day OHLCVA inputs, frozen official tokenizer and equal forecast-day training loss are retained throughout these comparisons. No predictor pretraining, new market features, capacity change or trading-rule change is mixed into these experiments.','',
        'The audit identifies representation distortion and tests two inexpensive changes independently: sampling from a wider token distribution, and choosing saved weights by high/low/range distribution error instead of token cross-entropy. Further data-feature and capacity experiments remain gated on these results.','',
        '## Representation audit','',
        'Each of 11,520 fixed stock/date inputs is tested at two- and five-day horizons. The three offline stages are: clipping alone; encoding and decoding the clipped true future; encoding and decoding the unclipped true future with the same frozen tokenizer. Historical means, scales and cached tokens are checked against the original data.','',
        'CRPS measures error in the whole predicted distribution; lower is better. Median MAE measures the error of its central price estimate. Interval score penalizes both excessive interval width and observations outside the interval. The following errors are percentage points of the signal-day close, averaged within each date and then across dates. Finite reconstructions with invalid OHLC relationships remain included, and their invalidity is counted separately. Actual future tokens are available only for this diagnostic: these figures are neither forecasts nor a strict lower bound on achievable forecast error.','']
    rr=[]
    for fold in ['2024h2','2025q2']:
        for h in [2,5]:
            x=recon[(recon.fold==fold)&(recon.partition=='evaluation')&(recon.horizon==h)&(recon.target=='range')].set_index('stage')
            rr.append([fold,h,f"{x.loc['clip_only','mae']:.3f}",f"{x.loc['tokenizer_clipped','mae']:.3f}",
                f"{x.loc['tokenizer_unclipped_future','mae']:.3f}",f"{100*x.loc['tokenizer_clipped','valid_ohlc']:.1f}%"])
    lines += [table(['Previously examined window','Days','Clipping-only range MAE','Clipped tokenizer range MAE','Unclipped tokenizer range MAE','Valid reconstructed paths'],rr),'',
        'Most invalid tokenizer reconstructions exceed a 0.01% signal-price violation threshold. Among invalid paths, the median largest OHLC ordering violation is 0.24–0.34 percentage points of signal price across the four inspected partitions, so this is not merely floating-point noise. No tolerance or repair changes the original masks. Simply removing clipping worsens range reconstruction in both displayed windows. The supplementary component analysis separates clipping distortion from tokenizer distortion around clipped values; absolute component errors do not add. Exact invalid-path counts and comparisons with the original forecaster on matched stock/date cohorts are saved under distortion-analysis.','',
        '## Sampling audit on the existing validation partitions','',
        'The four prespecified settings are baseline temperature 1.0 / top-p 0.9; full distribution at temperature 1.0; full distribution at 0.9; and full distribution at 1.1. Top-k is disabled. Every input retains 64 independently sampled paths; nested first-32 results diagnose finite-sample effects. They do not constitute a separate training experiment.','',
        'Selection holds the draw count at 64. The primary score averages maximum-high, minimum-low and range CRPS equally across two/five days and both validation folds. A candidate must improve average relative CRPS by at least 1%, keep MAE within 1%, avoid a usable-input coverage drop exceeding one percentage point, and have no worse 80% interval score in either fold. All metrics use common stock/date cohorts.','',
        f"The existing-window audit selects: **{audit_choice['selected']}**. This choice is diagnostic; it does not select the setting for the earlier chronological confirmation window.",'']
    m=pd.read_csv(AUDIT/'sampling/metrics.csv').groupby(['fold','variant'])[['crps','mae','coverage80','interval_score80']].mean()
    rr=[]
    for fold in ['2024h2','2025q2']:
        for variant in ['baseline','full_distribution','full_cooler','full_warmer']:
            x=m.loc[(fold,variant)]
            rr.append([fold,variant,f'{x.crps:.4f}',f'{x.mae:.4f}',f'{100*x.coverage80:.1f}%',f'{x.interval_score80:.3f}'])
    lines += [table(['Validation fold','Sampler','CRPS','Median MAE','80% interval coverage','80% interval score'],rr),'']
    cov=pd.read_csv(AUDIT/'sampling/coverage.csv')
    rr=[]
    for _,x in cov[(cov.horizon==5)&cov.variant.isin(['baseline','full_distribution'])].iterrows():
        rr.append([x.fold,x.variant,f'{100*x.valid_paths/x.total_paths:.1f}%',f'{int(x.usable)}/{int(x.known)}'])
    lines += [table(['Validation fold','Sampler','Legal five-day draws','Usable inputs / known labels'],rr),'',
        'Wider sampling increases invalid raw draws even where useful interval coverage improves. Scores describe the retained legal-path distribution and common usable inputs; they do not conceal discarded paths or establish unconditional calibration.','',
        '## Chronological confirmation','',
        'The newly designated window is new to this token-path evaluation protocol. It remains retrospective development data, and the tokenizer pretraining provenance limits still apply. Sampling settings for this confirmation are chosen only from 2023 H1 validation using seed 17, then frozen before any 2023 H2 evaluation and reused for seeds 29 and 43. Later-window audit outcomes do not select this setting.','',
        table(['Partition','Signal dates','Dates','Selected pool rows'],[
            ['Training','2016-03-03–2022-12-23','1,660','318,720'],
            ['Selection','2023-01-03–2023-06-21','113','21,696'],
            ['Evaluation','2023-07-03–2023-12-22','119','22,848']]),'',
        'There are 307,768 training rows with at least one known future day. Each epoch samples 32 stocks per training date, rotating through the pool. Seeds 17, 29 and 43 change initialization and training order; evaluation stocks and generation randomness remain fixed. Every five-day target ends before the next partition boundary. The eligible pool is not the number of examples actually used for a gradient update.','',
        'Training uses AdamW, learning rate 0.0003, weight decay 0.01, batch 256, at least eight and at most 24 epochs, and token-CE patience four. Every epoch is saved. Price-checkpoint candidates are every second completed epoch plus the CE-best checkpoint. Validation uses eight stocks/date and 32 paths/input, the same metric and guards as sampler selection, and falls back to the CE-best checkpoint if no candidate qualifies. CE monitoring uses 64 stocks/date.','',
        f"Chronological validation selects sampler **{chrono_choice['selected']}**, using only {chrono_choice['dates'][0]} through {chrono_choice['dates'][1]}.",'']
    rr=[]
    replacements=[]
    for seed in [17,29,43]:
        r=CONF/f'seed{seed}/dense_3720k_equal'
        x=read(r/'training/summary.json')
        c=read(r/'checkpoint-selection/selection.json')
        if c['selected_epoch']!=c['ce_epoch']:
            replacements.append(seed)
        history=read(r/'training/history.json')
        seen={e['epoch']:e['unique_examples_visited'] for e in history}
        rr.append([seed,x['epochs'],x['selected_epoch'],c['selected_epoch'],f"{seen[x['selected_epoch']]:,}",f"{seen[c['selected_epoch']]:,}",f"{x['unique_examples_visited']:,}",x['stopping']])
    checkpoint_outcome=(f'Qualifying price-checkpoint replacements occur for seeds {replacements}.' if replacements else
        'No seed finds a qualifying replacement for the CE checkpoint. Consequently, the price-checkpoint evaluation reuses identical baseline weights and forecasts. The candidate search found no improvement that passed the frozen guards; it does not establish that CE is generally optimal for price forecasts.')
    lines += [table(['Seed','Completed epochs','CE epoch','Price-selected epoch','Rows seen by CE checkpoint','Rows seen by price checkpoint','Rows seen by complete run','Stopping'],rr),'',checkpoint_outcome,'',
        'Evaluation uses 32 stocks per date (3,808 inputs per seed), 64 paths/input, and separate comparisons changing only the sampler or only the checkpoint. The sampler comparison holds CE weights fixed; the checkpoint comparison holds the baseline sampler fixed. When a selection falls back to the baseline, identical outputs are reused explicitly. Legal-path filtering is evaluated separately at two/five days; at least 16 of 64 paths are required, and usable-input coverage is reported.','',
        '### Distribution error by seed','',
        'Negative changes mean lower CRPS. Confidence intervals resample circular blocks of ten shared trading dates 2,000 times. Seed-mean intervals are conditional on these three fitted models and the sampled stock cohorts; they do not estimate uncertainty across all possible training seeds.','']
    rr=[]
    for _,x in paired[paired.metric=='crps'].iterrows():
        rr.append([x.seed,x.horizon,x.variant,f'{x.baseline:.4f}',f'{x.candidate:.4f}',f'{100*x.relative_change:+.2f}%',f'[{x.ci_low:+.4f}, {x.ci_high:+.4f}]'])
    lines += [table(['Seed','Days','Change tested','Baseline CRPS','Candidate CRPS','Relative change','95% interval for CRPS difference'],rr),'',
        '### Accuracy and uncertainty checks, averaged across seeds','']
    rr=[]
    for _,x in paired[(paired.seed=='mean')&(paired.metric!='crps')].iterrows():
        scale=100 if x.metric=='coverage80' else 1
        rr.append([x.horizon,x.variant,x.metric,f'{scale*x.baseline:.4f}',f'{scale*x.candidate:.4f}',f'{scale*x.delta:+.4f}'])
    lines += [table(['Days','Change tested','Metric','Baseline','Candidate','Difference'],rr),'',
        'Coverage is shown as a percentage; MAE and interval score use percentage points of signal-close price. Full target-specific metrics, input availability and date-level scores are retained in the artifacts.','',
        '## Verification and limitations','',
        'The audit retains 671,744 sampled five-day paths and confirmation retains 2,387,968, for a total of 3,059,712. These are generated scenarios, not independent labelled training examples.','',
        '- All 25 relevant regression tests passed, including the pinned official tokenizer integration and causal masks in both training and evaluation.','- Independent verification recomputes legal masks, extrema, explicit pairwise empirical CRPS, median errors, interval endpoints and scores on all common row cohorts. It also verifies split boundaries, known training rows, immutable manifests and generation chunk hashes.','- All three selected CE checkpoints reproduce their saved CPU logits exactly. Encoding four real 60-day inputs produces the original cached tokens and remains unchanged when an arbitrary five-day suffix is appended. Every generation run replays its first full batch exactly.','- Original completed experiments pass their unchanged file hashes. The source dataset ends before the sealed holdout beginning 2025-08-07. No holdout observations are opened.','- The stock sample is balanced by exchange and does not represent a natural market-cap-weighted universe. Membership, risk-vintage and tokenizer pretraining provenance remain retrospective limitations.','- This is forecast research, with no execution or profitability claim.','',
        '## Reproducible artifacts','',
        f'- Audit: [{AUDIT.name}]({AUDIT})',f'- Confirmation: [{CONF.name}]({CONF})',
        f'- Frozen audit protocol: [protocol.json]({AUDIT}/protocol.json)',
        f'- Frozen confirmation protocol: [protocol.json]({CONF}/protocol.json)',
        f'- Paired comparisons: [paired-comparisons.csv]({CONF}/results/paired-comparisons.csv)',
        f'- Audit verification: [independent-verification.json]({AUDIT}/independent-verification.json)',
        f'- Confirmation verification: [independent-verification.json]({CONF}/independent-verification.json)','']
    (BASE/'docs/token-pipeline-audit-20260915.md').write_text('\n'.join(lines))
    sizes={}
    for root in [AUDIT,CONF]:
        sizes[root.name]=sum(int(np.prod(np.load(p,mmap_mode='r').shape[:2])) for p in root.rglob('paths.npy'))
    print('Saved five-day path counts:',sizes)


if __name__=='__main__':
    main()
