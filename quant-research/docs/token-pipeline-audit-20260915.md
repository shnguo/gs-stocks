# Tokenizer, sampling and price-checkpoint experiment

Run dates: 2026-09-14–15 PDT / 2026-09-15 UTC.

## Outcome

Full-distribution sampling changes mean two-day CRPS by -1.86% and five-day CRPS by -3.48% across three training seeds. This is a distribution-quality result; median-price accuracy and invalid-path costs are reported separately.

Representation reconstruction is the next targeted optimization to test. Removing clipping did not help the frozen tokenizer, and wider sampling retains a cost in invalid paths. These diagnostics justify a controlled tokenizer comparison; they do not prove that tokenizer fine-tuning will improve forecasts. Additional market features, expanded stock/history sampling and capacity changes were not run in this stage.

## Research setting

Use the saved research preset: 64 paths, temperature 1.0, top-p 1.0 and top-k disabled. The preset contains the sampling keyword arguments accepted by generate_tokens. Keep equal forecast-day loss, the existing clipping threshold and CE checkpoint selection. The daily strategy model is not promoted by this experiment.

[Sampling preset](/Users/guo/Documents/stocks/quant-research/configs/token-sampling-research-v2.json)

Nominal 80% intervals still cover only 64.6% of two-day and 67.1% of five-day outcomes in the confirmation. Mean legal five-day draw rate falls from 87.3% to 84.6%. These limitations remain despite lower CRPS.

## Completed scope

Completed the staged pipeline audit, validation-only sampling comparisons and three-seed chronological confirmation. The existing 3,720,448-parameter causal token decoder, 60-day OHLCVA inputs, frozen official tokenizer and equal forecast-day training loss are retained throughout these comparisons. No predictor pretraining, new market features, capacity change or trading-rule change is mixed into these experiments.

The audit identifies representation distortion and tests two inexpensive changes independently: sampling from a wider token distribution, and choosing saved weights by high/low/range distribution error instead of token cross-entropy. Further data-feature and capacity experiments remain gated on these results.

## Representation audit

Each of 11,520 fixed stock/date inputs is tested at two- and five-day horizons. The three offline stages are: clipping alone; encoding and decoding the clipped true future; encoding and decoding the unclipped true future with the same frozen tokenizer. Historical means, scales and cached tokens are checked against the original data.

CRPS measures error in the whole predicted distribution; lower is better. Median MAE measures the error of its central price estimate. Interval score penalizes both excessive interval width and observations outside the interval. The following errors are percentage points of the signal-day close, averaged within each date and then across dates. Finite reconstructions with invalid OHLC relationships remain included, and their invalidity is counted separately. Actual future tokens are available only for this diagnostic: these figures are neither forecasts nor a strict lower bound on achievable forecast error.

| Previously examined window | Days | Clipping-only range MAE | Clipped tokenizer range MAE | Unclipped tokenizer range MAE | Valid reconstructed paths |
| --- | --- | --- | --- | --- | --- |
| 2024h2 | 2 | 0.361 | 1.692 | 1.786 | 94.6% |
| 2024h2 | 5 | 1.301 | 2.614 | 2.861 | 89.8% |
| 2025q2 | 2 | 0.097 | 1.193 | 1.248 | 95.1% |
| 2025q2 | 5 | 0.420 | 1.532 | 1.659 | 90.3% |

Most invalid tokenizer reconstructions exceed a 0.01% signal-price violation threshold. Among invalid paths, the median largest OHLC ordering violation is 0.24–0.34 percentage points of signal price across the four inspected partitions, so this is not merely floating-point noise. No tolerance or repair changes the original masks. Simply removing clipping worsens range reconstruction in both displayed windows. The supplementary component analysis separates clipping distortion from tokenizer distortion around clipped values; absolute component errors do not add. Exact invalid-path counts and comparisons with the original forecaster on matched stock/date cohorts are saved under distortion-analysis.

## Sampling audit on the existing validation partitions

The four prespecified settings are baseline temperature 1.0 / top-p 0.9; full distribution at temperature 1.0; full distribution at 0.9; and full distribution at 1.1. Top-k is disabled. Every input retains 64 independently sampled paths; nested first-32 results diagnose finite-sample effects. They do not constitute a separate training experiment.

Selection holds the draw count at 64. The primary score averages maximum-high, minimum-low and range CRPS equally across two/five days and both validation folds. A candidate must improve average relative CRPS by at least 1%, keep MAE within 1%, avoid a usable-input coverage drop exceeding one percentage point, and have no worse 80% interval score in either fold. All metrics use common stock/date cohorts.

The existing-window audit selects: **full_distribution**. This choice is diagnostic; it does not select the setting for the earlier chronological confirmation window.

| Validation fold | Sampler | CRPS | Median MAE | 80% interval coverage | 80% interval score |
| --- | --- | --- | --- | --- | --- |
| 2024h2 | baseline | 2.6734 | 3.5470 | 59.5% | 19.074 |
| 2024h2 | full_distribution | 2.6125 | 3.5316 | 68.1% | 17.932 |
| 2024h2 | full_cooler | 2.6328 | 3.5257 | 63.8% | 18.401 |
| 2024h2 | full_warmer | 2.6737 | 3.6156 | 71.2% | 18.117 |
| 2025q2 | baseline | 2.5528 | 3.4016 | 64.5% | 17.996 |
| 2025q2 | full_distribution | 2.4898 | 3.3844 | 71.9% | 17.012 |
| 2025q2 | full_cooler | 2.5062 | 3.3775 | 68.4% | 17.224 |
| 2025q2 | full_warmer | 2.5335 | 3.4346 | 74.8% | 17.332 |

| Validation fold | Sampler | Legal five-day draws | Usable inputs / known labels |
| --- | --- | --- | --- |
| 2024h2 | baseline | 85.0% | 1623/1727 |
| 2024h2 | full_distribution | 81.2% | 1618/1727 |
| 2025q2 | baseline | 89.3% | 797/829 |
| 2025q2 | full_distribution | 85.7% | 796/829 |

Wider sampling increases invalid raw draws even where useful interval coverage improves. Scores describe the retained legal-path distribution and common usable inputs; they do not conceal discarded paths or establish unconditional calibration.

## Chronological confirmation

The newly designated window is new to this token-path evaluation protocol. It remains retrospective development data, and the tokenizer pretraining provenance limits still apply. Sampling settings for this confirmation are chosen only from 2023 H1 validation using seed 17, then frozen before any 2023 H2 evaluation and reused for seeds 29 and 43. Later-window audit outcomes do not select this setting.

| Partition | Signal dates | Dates | Selected pool rows |
| --- | --- | --- | --- |
| Training | 2016-03-03–2022-12-23 | 1,660 | 318,720 |
| Selection | 2023-01-03–2023-06-21 | 113 | 21,696 |
| Evaluation | 2023-07-03–2023-12-22 | 119 | 22,848 |

There are 307,768 training rows with at least one known future day. Each epoch samples 32 stocks per training date, rotating through the pool. Seeds 17, 29 and 43 change initialization and training order; evaluation stocks and generation randomness remain fixed. Every five-day target ends before the next partition boundary. The eligible pool is not the number of examples actually used for a gradient update.

Training uses AdamW, learning rate 0.0003, weight decay 0.01, batch 256, at least eight and at most 24 epochs, and token-CE patience four. Every epoch is saved. Price-checkpoint candidates are every second completed epoch plus the CE-best checkpoint. Validation uses eight stocks/date and 32 paths/input, the same metric and guards as sampler selection, and falls back to the CE-best checkpoint if no candidate qualifies. CE monitoring uses 64 stocks/date.

Chronological validation selects sampler **full_distribution**, using only 2023-01-03 through 2023-06-21.

| Seed | Completed epochs | CE epoch | Price-selected epoch | Rows seen by CE checkpoint | Rows seen by price checkpoint | Rows seen by complete run | Stopping |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 17 | 9 | 5 | 5 | 265,600 | 265,600 | 307,768 | early_stopping |
| 29 | 11 | 7 | 7 | 307,768 | 307,768 | 307,768 | early_stopping |
| 43 | 10 | 6 | 6 | 307,768 | 307,768 | 307,768 | early_stopping |

No seed finds a qualifying replacement for the CE checkpoint. Consequently, the price-checkpoint evaluation reuses identical baseline weights and forecasts. The candidate search found no improvement that passed the frozen guards; it does not establish that CE is generally optimal for price forecasts.

Evaluation uses 32 stocks per date (3,808 inputs per seed), 64 paths/input, and separate comparisons changing only the sampler or only the checkpoint. The sampler comparison holds CE weights fixed; the checkpoint comparison holds the baseline sampler fixed. When a selection falls back to the baseline, identical outputs are reused explicitly. Legal-path filtering is evaluated separately at two/five days; at least 16 of 64 paths are required, and usable-input coverage is reported.

### Distribution error by seed

Negative changes mean lower CRPS. Confidence intervals resample circular blocks of ten shared trading dates 2,000 times. Seed-mean intervals are conditional on these three fitted models and the sampled stock cohorts; they do not estimate uncertainty across all possible training seeds.

| Seed | Days | Change tested | Baseline CRPS | Candidate CRPS | Relative change | 95% interval for CRPS difference |
| --- | --- | --- | --- | --- | --- | --- |
| 17 | 2 | sampling | 1.5570 | 1.5275 | -1.90% | [-0.0479, -0.0138] |
| 29 | 2 | sampling | 1.5648 | 1.5458 | -1.21% | [-0.0376, -0.0051] |
| 43 | 2 | sampling | 1.5871 | 1.5478 | -2.47% | [-0.0560, -0.0250] |
| mean | 2 | sampling | 1.5696 | 1.5404 | -1.86% | [-0.0466, -0.0146] |
| 17 | 2 | price_checkpoint | 1.5570 | 1.5570 | +0.00% | [+0.0000, +0.0000] |
| 29 | 2 | price_checkpoint | 1.5648 | 1.5648 | +0.00% | [+0.0000, +0.0000] |
| 43 | 2 | price_checkpoint | 1.5871 | 1.5871 | +0.00% | [+0.0000, +0.0000] |
| mean | 2 | price_checkpoint | 1.5696 | 1.5696 | +0.00% | [+0.0000, +0.0000] |
| 17 | 5 | sampling | 2.7055 | 2.6084 | -3.59% | [-0.1508, -0.0488] |
| 29 | 5 | sampling | 2.6823 | 2.6258 | -2.11% | [-0.1105, -0.0063] |
| 43 | 5 | sampling | 2.7816 | 2.6512 | -4.69% | [-0.1900, -0.0791] |
| mean | 5 | sampling | 2.7231 | 2.6285 | -3.48% | [-0.1495, -0.0463] |
| 17 | 5 | price_checkpoint | 2.7055 | 2.7055 | +0.00% | [+0.0000, +0.0000] |
| 29 | 5 | price_checkpoint | 2.6823 | 2.6823 | +0.00% | [+0.0000, +0.0000] |
| 43 | 5 | price_checkpoint | 2.7816 | 2.7816 | +0.00% | [+0.0000, +0.0000] |
| mean | 5 | price_checkpoint | 2.7231 | 2.7231 | +0.00% | [+0.0000, +0.0000] |

### Accuracy and uncertainty checks, averaged across seeds

| Days | Change tested | Metric | Baseline | Candidate | Difference |
| --- | --- | --- | --- | --- | --- |
| 2 | sampling | mae | 2.0145 | 2.0162 | +0.0017 |
| 2 | sampling | coverage80 | 57.1511 | 64.6009 | +7.4499 |
| 2 | sampling | interval_score80 | 11.8216 | 11.2089 | -0.6127 |
| 2 | price_checkpoint | mae | 2.0145 | 2.0145 | +0.0000 |
| 2 | price_checkpoint | coverage80 | 57.1511 | 57.1511 | +0.0000 |
| 2 | price_checkpoint | interval_score80 | 11.8216 | 11.8216 | +0.0000 |
| 5 | sampling | mae | 3.4245 | 3.3861 | -0.0384 |
| 5 | sampling | coverage80 | 57.5239 | 67.0786 | +9.5547 |
| 5 | sampling | interval_score80 | 21.1984 | 19.5261 | -1.6723 |
| 5 | price_checkpoint | mae | 3.4245 | 3.4245 | +0.0000 |
| 5 | price_checkpoint | coverage80 | 57.5239 | 57.5239 | +0.0000 |
| 5 | price_checkpoint | interval_score80 | 21.1984 | 21.1984 | +0.0000 |

Coverage is shown as a percentage; MAE and interval score use percentage points of signal-close price. Full target-specific metrics, input availability and date-level scores are retained in the artifacts.

## Verification and limitations

The audit retains 671,744 sampled five-day paths and confirmation retains 2,387,968, for a total of 3,059,712. These are generated scenarios, not independent labelled training examples.

- All 25 relevant regression tests passed, including the pinned official tokenizer integration and causal masks in both training and evaluation.
- Independent verification recomputes legal masks, extrema, explicit pairwise empirical CRPS, median errors, interval endpoints and scores on all common row cohorts. It also verifies split boundaries, known training rows, immutable manifests and generation chunk hashes.
- All three selected CE checkpoints reproduce their saved CPU logits exactly. Encoding four real 60-day inputs produces the original cached tokens and remains unchanged when an arbitrary five-day suffix is appended. Every generation run replays its first full batch exactly.
- Original completed experiments pass their unchanged file hashes. The source dataset ends before the sealed holdout beginning 2025-08-07. No holdout observations are opened.
- The stock sample is balanced by exchange and does not represent a natural market-cap-weighted universe. Membership, risk-vintage and tokenizer pretraining provenance remain retrospective limitations.
- This is forecast research, with no execution or profitability claim.

## Reproducible artifacts

- Audit: [token-pipeline-audit-20260915-v1](/Users/guo/Documents/stocks/quant-research/artifacts/token-pipeline-audit-20260915-v1)
- Confirmation: [token-pipeline-confirmation-20260915-v1](/Users/guo/Documents/stocks/quant-research/artifacts/token-pipeline-confirmation-20260915-v1)
- Frozen audit protocol: [protocol.json](/Users/guo/Documents/stocks/quant-research/artifacts/token-pipeline-audit-20260915-v1/protocol.json)
- Frozen confirmation protocol: [protocol.json](/Users/guo/Documents/stocks/quant-research/artifacts/token-pipeline-confirmation-20260915-v1/protocol.json)
- Paired comparisons: [paired-comparisons.csv](/Users/guo/Documents/stocks/quant-research/artifacts/token-pipeline-confirmation-20260915-v1/results/paired-comparisons.csv)
- Audit verification: [independent-verification.json](/Users/guo/Documents/stocks/quant-research/artifacts/token-pipeline-audit-20260915-v1/independent-verification.json)
- Confirmation verification: [independent-verification.json](/Users/guo/Documents/stocks/quant-research/artifacts/token-pipeline-confirmation-20260915-v1/independent-verification.json)
