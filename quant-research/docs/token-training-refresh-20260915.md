# Training refresh: high- and low-price forecast comparison

Date: 2026-09-15. Completed retrospective research on two historical windows and three training seeds.

## Current conclusion

Retain expanding-history training refresh as a useful research improvement. Across the two tested windows, maximum-high and minimum-low MAE both improve after averaging three seeds. Combined high/low MAE falls 1.24% and 1.14% in 2023 H2, and 1.72% and 3.31% in 2024 H2, at two and five days respectively. All 12 individual seed/window/horizon high-low averages improve. The largest average target improvement is 2024 H2 five-day minimum-low MAE, down 8.26%.

Maximum-high and minimum-low MAE improve in 8 of 8 window/horizon/target combinations after averaging the three seeds, and in 20 of 24 individual seed combinations. These counts describe the tested cases, not the probability of future success.

The evidence is strongest for minimum-low and range forecasts. High-price gains are smaller (0.48–0.82% on average), and four of 24 individual high/low seed cases regress, all on maximum-high forecasts; the largest regression is 0.59%. The 2023 H2 combined high/low confidence intervals include zero, while both 2024 H2 intervals are below zero. This qualifies certainty without negating the consistent direction of improvement. The experiment jointly changes training recency, history size, optimization exposure and checkpoint-selection period, so it does not isolate recency alone. These are previously examined historical windows.

## What was compared

Control: the exact three predictors trained on 2016–2021 and selected on 2022 H1, using archived forecasts. Refreshed: train on 2016–2022 and select on 2023 H1 for 2023 H2; train on 2016–2023 and select on 2024 H1 for 2024 H2. Every future label stays inside its partition. Each seed uses the same preselected 32 stocks per signal date, 119 dates in 2023 H2 and 120 dates in 2024 H2.

Both sides use the same 3,720,448 parameter autoregressive token Transformer, 60 history bars, equal five-day token loss, fixed adapted decoder, and 64 sampled paths with temperature 1 and full token distribution. Generation batches contain four inputs and use seed 17 plus batch offset. The decoder was trained through 2021 and selected on 2022 H1. Its weights are identical on both sides.

Four matching completed predictor fits were reused. Two new fits were completed for 2024 H2, seeds 29 and 43. For 2023 H2, saved generated tokens were re-decoded with the fixed adapted decoder after validating source hashes; first-batch generation and decoded prices replay exactly. The 2024 H2 forecasts were newly generated.

This tests the practical refresh package. It jointly changes recent-history coverage, dataset size, optimization exposure, and the validation period used for checkpoint selection. It does not identify the isolated causal effect of newer observations. Both evaluation windows were examined in earlier research; they are not untouched tests. Official tokenizer pretraining coverage remains unresolved.

## Point forecast errors

Point forecasts are medians across legal sampled paths of the future maximum high, minimum low, or high-minus-low range. Errors and confidence-interval endpoints are percentage points of the signal-date close. For example, 1.0 pp is a price error of 1 on a signal close of 100. Relative changes below zero mean lower error. No buy/sell rules enter these targets.

| Window | Days | Target | Old MAE, pp | Refreshed MAE, pp | Change | 95% CI of difference, pp |
| --- | --- | --- | --- | --- | --- | --- |
| 2023h2 | 2 | maximum | 2.3444 | 2.3312 | -0.56% | [-0.0368, +0.0119] |
| 2023h2 | 2 | minimum | 1.6718 | 1.6351 | -2.20% | [-0.0742, -0.0039] |
| 2023h2 | 2 | range | 2.0365 | 2.0093 | -1.34% | [-0.0456, -0.0107] |
| 2023h2 | 5 | maximum | 4.0821 | 4.0623 | -0.48% | [-0.0584, +0.0238] |
| 2023h2 | 5 | minimum | 2.4383 | 2.3835 | -2.25% | [-0.1351, +0.0157] |
| 2023h2 | 5 | range | 3.7643 | 3.6993 | -1.73% | [-0.1102, -0.0287] |
| 2024h2 | 2 | maximum | 4.0542 | 4.0216 | -0.80% | [-0.0764, +0.0164] |
| 2024h2 | 2 | minimum | 2.5119 | 2.4312 | -3.21% | [-0.1331, -0.0292] |
| 2024h2 | 2 | range | 3.1751 | 3.0978 | -2.43% | [-0.1233, -0.0366] |
| 2024h2 | 5 | maximum | 7.3467 | 7.2862 | -0.82% | [-0.1428, +0.0381] |
| 2024h2 | 5 | minimum | 3.6859 | 3.3813 | -8.26% | [-0.5322, -0.1033] |
| 2024h2 | 5 | range | 6.2679 | 6.0465 | -3.53% | [-0.3954, -0.0966] |

## Combined high/low evidence

MAE measures the median point forecast. CRPS scores the full sampled distribution; smaller is better. Each date has equal weight, then seeds and the high/low targets have equal weight. Date win rate is the fraction of dates with a lower averaged error, not a trading win rate.

| Window | Days | Metric | Old | Refreshed | Change | Dates improved |
| --- | --- | --- | --- | --- | --- | --- |
| 2023h2 | 2 | mae | 2.0081 | 1.9831 | -1.24% | 60.5% |
| 2023h2 | 2 | crps | 1.5030 | 1.4869 | -1.07% | 59.7% |
| 2023h2 | 5 | mae | 3.2602 | 3.2229 | -1.14% | 58.8% |
| 2023h2 | 5 | crps | 2.4712 | 2.4462 | -1.01% | 59.7% |
| 2024h2 | 2 | mae | 3.2830 | 3.2264 | -1.72% | 59.2% |
| 2024h2 | 2 | crps | 2.4737 | 2.4197 | -2.18% | 64.2% |
| 2024h2 | 5 | mae | 5.5163 | 5.3337 | -3.31% | 67.5% |
| 2024h2 | 5 | crps | 4.2233 | 4.0781 | -3.44% | 71.7% |

The 95% intervals use 2,000 circular block bootstrap samples with blocks of 10 dates. They preserve local time dependence but are conditional on these three trained seeds and historical windows; they do not capture all model-selection or market-regime uncertainty. Multiple target comparisons are descriptive.

## Data coverage and training completion

| Window | Seed | Train dates | Known pool | Unique visited | Total presentations | Best epoch | Epochs | Fit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023h2 | 17 | 1660 | 307768 | 307768 | 478080 | 5 | 9 | reused |
| 2023h2 | 29 | 1660 | 307768 | 307768 | 584320 | 7 | 11 | reused |
| 2023h2 | 43 | 1660 | 307768 | 307768 | 531200 | 6 | 10 | reused |
| 2024h2 | 17 | 1902 | 354007 | 354007 | 547776 | 5 | 9 | reused |
| 2024h2 | 29 | 1902 | 354007 | 354007 | 608640 | 6 | 10 | new fit |
| 2024h2 | 43 | 1902 | 354007 | 354007 | 608640 | 6 | 10 | new fit |

A training epoch visits 32 stocks per date with rotating coverage; selection uses up to 64 per date. At least 16 of the 64 paths must be legal for a row/horizon to be scored. Comparisons use the intersection of usable inputs between the two predictors within each seed. The following coverage counts are totals across the three seeds, so the same stock/date can appear three times.

| Window | Days | Predictor | Usable inputs | Known-label inputs | Legal paths |
| --- | --- | --- | --- | --- | --- |
| 2023h2 | 2 | old | 11240 | 11361 | 96.05% |
| 2023h2 | 2 | refreshed | 11248 | 11361 | 96.10% |
| 2023h2 | 5 | old | 11056 | 11289 | 91.57% |
| 2023h2 | 5 | refreshed | 11055 | 11289 | 91.64% |
| 2024h2 | 2 | old | 11388 | 11430 | 95.91% |
| 2024h2 | 2 | refreshed | 11387 | 11430 | 95.83% |
| 2024h2 | 5 | old | 11130 | 11319 | 90.87% |
| 2024h2 | 5 | refreshed | 11117 | 11319 | 90.82% |

## Calibration decision carried forward

Date-weighted interval calibration remains the preferred research interval method because it improves average coverage and interval score across most tested scenarios. The earlier strict-gate result remains preserved as historical evidence. This comparison uses raw forecast distributions to measure the predictor refresh. The 18 old calibration coefficients are tied to the old predictors and were not transferred to new weights. Refreshed predictors require calibration fitted on their own earlier validation forecasts.

## Verification

All six checkpoint CPU logit reloads match exactly. Split boundaries, checkpoint selection, training-row membership, source and chunk hashes, forecast assembly, legal-path counts, and common cohorts were verified. Independent formulas reproduced 268,326 row/target score records, all date-equal aggregates, and 400 paired estimates and block intervals. Prior calibration artifacts are unchanged.

The audit caught a rounding issue in the descriptive date-frequency counter: a mathematically tied coverage value could be counted differently under a different averaging order. Differences within 1e-12 are now treated as ties. The original summary is retained, and the correction changes no forecast, error estimate, confidence interval, or MAE date-improvement count. The correction script and receipt are included with the frozen run.

The daily strategy model was not replaced. The sealed holdout beginning 2025-08-07 remains unopened. All jobs for this bounded comparison completed.

## Next action

Fit checkpoint-specific interval calibration using the refreshed predictors on their own earlier validation periods, then verify that improved high/low point estimates and calibrated interval coverage coexist on the same two evaluation windows. Retain the decoder, full-distribution sampler and equal output loss. Do not transfer the old 18 calibration coefficients to these new weights.

Artifacts: artifacts/token-training-refresh-20260915-v1 contains the frozen protocol and source code, source manifest, training checkpoints, generated tokens and paths, common-cohort scores, paired estimates, independent verification and final manifest. The complete per-seed breakdown is results/paired.csv.

Related: [calibration decision update](token-calibration-decision-update-20260915.md), [calibration measurements](token-interval-calibration-20260915.md), [fixed-weight temporal validation](tokenizer-temporal-20260915.md).
