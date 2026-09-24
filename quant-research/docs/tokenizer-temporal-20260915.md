# Fixed-decoder temporal validation

Run date: 2026-09-15.

## Outcome

The fixed decoder passes the prespecified two-day confirmation conditions in 2023 H2. Five-day conditions do not pass. This is a transfer test of the exact prior checkpoints; there is no retraining or reselection.

## What was held fixed

- The price-consistency tokenizer decoder, selected at epoch 15 on 2022 H1 validation. The encoder and binary token semantics are unchanged.
- The three 3,720,448-parameter predictor checkpoints from seeds 17, 29 and 43, fitted on 2016–2021 inputs and selected on 2022 H1 token CE.
- Sixty history bars, five generated days, 64 independent token trajectories per input, temperature 1.0, top-p 1.0 and top-k disabled.
- The two-day result scores the first two days of the same five-day generator. It does not test a separately trained two-day model.
- The same generated token trajectories are decoded by the original and adapted tokenizers. There is no post-decoding price repair.

The control uses the original Kronos tokenizer with the self-built predictor. This is not a new comparison against the official Kronos Base predictor.

Only the evaluation period changes: 2023-07-03 through 2023-12-22, 119 signal dates and 3,808 stock/date inputs per predictor. Last target: 2023-12-29. Across three predictors, 731,136 generated trajectories are each decoded twice.

The new evaluation is disjoint from the 2022 H2 decoder experiment. However, 2023 H2 has already appeared in earlier sampling research and is retrospective development data. Official tokenizer pretraining coverage is unresolved. These qualifications prevent treating it as a pristine test or evidence of live trading performance. The sealed holdout starting 2025-08-07 remains unused.

## Prespecified interpretation

Two days is primary. Confirmation requires lower CRPS in all three predictor seeds, a date-block 95% interval below zero for the mean CRPS difference, median MAE degradation no greater than 1%, non-worsening interval score and non-decreasing legal-path fractions. Five days uses the same descriptive checks as a secondary horizon. No daily-model promotion follows automatically. The protocol was saved before forecasts began.

CRPS measures error across the forecast distribution; MAE measures the error of its median. Price metrics are in percentage points of signal-close price, averaged equally over maximum-high, minimum-low and range. Lower is better. Coverage is the fraction of actual values inside nominal 80% intervals.

## Forecast results

| Days | Metric | Original | Adapted | Relative change | 95% interval for difference |
| --- | --- | --- | --- | --- | --- |
| 2 | crps | 1.5586 | 1.5136 | -2.89% | [-0.0948, -0.0126] |
| 2 | mae | 2.0424 | 2.0142 | -1.38% | [-0.0611, -0.0032] |
| 2 | coverage80 | 64.8928 | 67.3671 | +3.81% | [+1.7653, +3.3438] |
| 2 | interval_score80 | 11.3271 | 10.7043 | -5.50% | [-1.3001, -0.2093] |
| 5 | crps | 2.6524 | 2.6173 | -1.32% | [-0.1049, +0.0128] |
| 5 | mae | 3.4285 | 3.4227 | -0.17% | [-0.0525, +0.0345] |
| 5 | coverage80 | 67.4539 | 69.7398 | +3.39% | [+1.7205, +2.9124] |
| 5 | interval_score80 | 19.5824 | 18.9763 | -3.10% | [-1.5248, -0.0399] |

Coverage values and their difference intervals are percentages and percentage points respectively. Its relative-change column is a relative percentage change, not percentage points.

| Predictor seed | Days | CRPS change | 95% interval for difference |
| --- | --- | --- | --- |
| 17 | 2 | -2.72% | [-0.0911, -0.0107] |
| 29 | 2 | -3.03% | [-0.1008, -0.0134] |
| 43 | 2 | -2.92% | [-0.0938, -0.0130] |
| 17 | 5 | -1.44% | [-0.1061, +0.0051] |
| 29 | 5 | -1.41% | [-0.1159, +0.0183] |
| 43 | 5 | -1.12% | [-0.0944, +0.0173] |

## Maximum-high, minimum-low and range

| Days | Target | Metric | Original | Adapted | Change | 95% difference interval |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | maximum | crps | 1.8199 | 1.7727 | -2.59% | [-0.1335, +0.0036] |
| 2 | maximum | mae | 2.3619 | 2.3429 | -0.81% | [-0.0709, +0.0200] |
| 2 | minimum | crps | 1.2571 | 1.2291 | -2.23% | [-0.0545, -0.0075] |
| 2 | minimum | mae | 1.6883 | 1.6683 | -1.19% | [-0.0527, +0.0043] |
| 2 | range | crps | 1.5988 | 1.5389 | -3.74% | [-0.1142, -0.0193] |
| 2 | range | mae | 2.0768 | 2.0314 | -2.19% | [-0.0923, -0.0093] |
| 5 | maximum | crps | 3.2370 | 3.1760 | -1.88% | [-0.1783, +0.0157] |
| 5 | maximum | mae | 4.1044 | 4.0921 | -0.30% | [-0.0775, +0.0454] |
| 5 | minimum | crps | 1.7708 | 1.7638 | -0.40% | [-0.0347, +0.0215] |
| 5 | minimum | mae | 2.4251 | 2.4291 | +0.16% | [-0.0285, +0.0373] |
| 5 | range | crps | 2.9493 | 2.9121 | -1.26% | [-0.1232, +0.0229] |
| 5 | range | mae | 3.7560 | 3.7468 | -0.24% | [-0.0837, +0.0470] |

For the two-day maximum-high and minimum-low point estimates individually, both difference intervals still include zero. The clearest point-accuracy gain is in the range target. The significant average does not establish a separate high-price or low-price point-accuracy improvement.

Intervals use 2,000 circular resamples of ten trading dates, shared across seeds. They are conditional on these three fitted predictors, one fitted decoder and the sampled stock cohorts. Target-specific intervals are descriptive and are not adjusted for multiple comparisons.

## Legal paths and coverage

| Decoder | Days | Legal generated draws | Usable inputs / all inputs |
| --- | --- | --- | --- |
| adapted | 2 | 96.05% | 98.39% |
| adapted | 5 | 91.57% | 96.78% |
| frozen | 2 | 92.47% | 97.55% |
| frozen | 5 | 84.39% | 94.07% |

The primary comparison uses common stock/date inputs with at least 16 legal draws under each decoder. Each decoder can retain different draws. The following sensitivity retains exactly the same legal token trajectories under both decoders, also requiring at least 16.

| Days | Shared-path CRPS change | Shared-path median MAE change | Shared-path 80% coverage |
| --- | --- | --- | --- |
| 2 | -2.86% | -1.47% | 64.84% → 66.53% |
| 5 | -1.45% | -0.55% | 67.34% → 68.40% |

## Comparison with the preceding window

| Evaluation window | Days | CRPS change | Median MAE change | 80% coverage |
| --- | --- | --- | --- | --- |
| 2022 H2 | 2 | -0.82% | -0.20% | 67.28% → 69.45% |
| 2022 H2 | 5 | -0.18% | +0.25% | 69.65% → 71.68% |
| 2023 H2 | 2 | -2.89% | -1.38% | 64.89% → 67.37% |
| 2023 H2 | 5 | -1.32% | -0.17% | 67.45% → 69.74% |

These are separate windows with exactly the same model weights. Their errors are not pooled into a single independent sample or used to choose a replacement checkpoint.

## Interpretation and next priority

The two-day distribution gain transfers to the later window under the prespecified checks. Its size remains modest, and the earlier window showed little point-accuracy improvement.

Five-day results must be read alongside the earlier inconclusive window and the separate high/low/range rows. A gain in average distribution score does not imply that every price target improves.

The next priority is prediction-interval calibration with parameters fitted only on validation data, while monitoring high/low median accuracy. Keep this decoder as a research candidate and the existing control available; this transfer check does not establish reliable trading intervals or justify a larger model.

## Reconstruction diagnostic

| Decoder | Extrema MAE (pp) | Legal reconstructions | Normalized six-field MAE | Normalized volume/amount MAE |
| --- | --- | --- | --- | --- |
| frozen | 1.0291 | 92.06% | 0.2369 | 0.3896 |
| adapted | 0.8650 | 96.44% | 0.2094 | 0.3417 |

Reconstruction uses actual future tokens and is an offline diagnostic. It is not forecast accuracy and did not select this checkpoint.

## Verification and limits

- Independently recomputed all raw forecast scores, date means, 128 target/seed contrasts and block intervals. The shared-path sensitivity also passes independent explicit pairwise-CRPS checks.
- Predictor checkpoint copies are byte-identical to the prior experiment. CPU reload logits reproduce the saved references exactly, and generated first-batch replay is exact.
- Training/selection row IDs are unchanged, evaluation periods are disjoint, and all five-day labels end within the specified evaluation boundary.
- The inherited frozen implementation passed 31 regression tests in the preceding experiment. This run verifies unchanged source/checkpoint hashes and adds independent checks of the new outputs.
- One decoder fit and three predictor fits do not measure uncertainty across decoder-training seeds. Fixed old predictors test transfer without refresh; they do not answer whether periodic retraining helps.

## Artifacts

- [Frozen protocol](/Users/guo/Documents/stocks/quant-research/artifacts/tokenizer-temporal-20260915-v1/protocol.json)
- [Decision and conditions](/Users/guo/Documents/stocks/quant-research/artifacts/tokenizer-temporal-20260915-v1/research-decision.json)
- [Independent verification](/Users/guo/Documents/stocks/quant-research/artifacts/tokenizer-temporal-20260915-v1/independent-verification.json)
- [Per-target comparisons](/Users/guo/Documents/stocks/quant-research/artifacts/tokenizer-temporal-20260915-v1/forecast-results/target-paired.csv)
