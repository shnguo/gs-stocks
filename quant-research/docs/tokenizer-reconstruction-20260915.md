# Tokenizer decoder reconstruction experiment

Run date: 2026-09-15.

## Scope and outcome

Validation selects **price_consistency**. The encoder and binary token codes remain frozen; only the tokenizer decoder is adapted. The comparison retains the original tokenizer as its control.

The first candidate minimizes normalized six-field reconstruction loss. The second adds losses for maximum-high, minimum-low and range errors, plus OHLC ordering violations. These terms affect learned weights; generated prices are not repaired after decoding. Both candidates start from identical official weights, use the same training rows and order, and retain the existing history-based normalization and input clipping.

A decoder can improve reconstruction of observed tokens without improving prediction of unknown future prices. This experiment treats those as separate measurements.

## Chronological partitions

| Partition | Signal dates | Dates | Pool rows | Rows with known targets | Last target |
| --- | --- | --- | --- | --- | --- |
| train | 2016-03-03–2021-12-24 | 1418 | 272,256 | 261,563 | 2021-12-31 |
| selection | 2022-01-04–2022-06-23 | 112 | 21,504 | 21,321 | 2022-06-30 |
| evaluation | 2022-07-01–2022-12-23 | 120 | 23,040 | 22,979 | 2022-12-30 |

The 2022 H2 window is newly designated for this token-forecast protocol, but remains retrospective development data. Official tokenizer pretraining coverage remains unresolved. Every five-day target ends before the next partition boundary. The sealed holdout beginning 2025-08-07 is not opened.

## Training and selection

Only post_quant_embed, the three decoder Transformer blocks and the final six-field head receive gradients. Input embeddings, encoder layers, quantization projection and binary code semantics remain unchanged. Adaptation uses seed 17, AdamW at learning rate 0.00003, weight decay 0.01, batch 256, a minimum of eight and maximum of sixteen epochs, and patience three on validation reconstruction MAE. Each epoch visits all 1,418 training dates and rotates 32 stocks/date through the known-label pool.

Both objectives target actual, unclipped OHLCVA values using history-only normalization. Unknown future suffixes contribute neither loss nor gradients. The general loss is mean Smooth L1 error across six normalized fields and known steps. The price objective adds 0.25 times Smooth L1 high/low/range error in percentage points of signal-close price, equally over two/five-day known prefixes, plus 0.25 times the OHLC ordering gap in the same units.

Every epoch is retained. A checkpoint must improve date-equal mean extrema MAE by at least 5%, improve legal-path fraction by at least two percentage points, and keep overall normalized MAE and volume/amount MAE within 5% of the frozen control. The minimum-MAE qualifying checkpoint is selected; if none qualifies, the original tokenizer is retained. Validation uses 16 stocks/date; evaluation uses 32. Evaluation results never change the selection.

| Objective | Epochs | Selected epoch | Rows seen by selected weights | Rows seen by complete run | Trainable parameters | Passes guards | Stopping |
| --- | --- | --- | --- | --- | --- | --- | --- |
| reconstruction | 16 | 13 | 261,563 | 261,563 | 1977606 | False | early_stopping |
| price_consistency | 16 | 15 | 261,563 | 261,563 | 1977606 | True | budget_limited |

### Reconstruction results

These metrics use the actual future tokens, so they are diagnostics rather than forecasts or a strict forecast-error floor. All finite reconstructions remain in the error calculation, including invalid OHLC paths. Errors are averaged within each date, then over dates, horizons and the three price targets.

| Partition | Decoder | High/low/range MAE (pp) | Legal paths | Six-field normalized MAE | Volume/amount normalized MAE |
| --- | --- | --- | --- | --- | --- |
| decoder-selection | frozen | 1.0258 | 84.59% | 0.1157 | 0.1507 |
| decoder-selection | reconstruction | 0.9691 | 85.98% | 0.1050 | 0.1321 |
| decoder-selection | price_consistency | 0.9245 | 95.62% | 0.1090 | 0.1399 |
| reconstruction-evaluation | frozen | 0.7639 | 91.02% | 0.1317 | 0.1760 |
| reconstruction-evaluation | reconstruction | 0.7226 | 88.45% | 0.1165 | 0.1428 |
| reconstruction-evaluation | price_consistency | 0.7038 | 96.32% | 0.1212 | 0.1515 |

## Actual forecast comparison

Across three predictor seeds, changing only the decoder changes mean two-day CRPS by -0.82% and five-day CRPS by -0.18%.

Three fresh 3,720,448-parameter predictors are trained on the training partition with seeds 17, 29 and 43, equal forecast-day token CE, and validation-only CE checkpoint selection. Each generates 64 complete token trajectories per input, using temperature 1.0, top-p 1.0 and top-k disabled. Exactly the same generated token trajectories are decoded by the original and selected adapted decoder. No actual future token enters forecast generation. Decoder adaptation itself uses one training seed; this test varies the predictor, not the decoder fit.

Metrics require known targets and at least 16 legal paths per input. The two decoders share each comparison stock/date cohort but may retain different legal draws; coverage and valid-draw fractions are therefore reported alongside error. Confidence intervals resample 2,000 circular blocks of ten trading dates jointly across seeds. They are conditional on these fitted models and sampled stock cohorts.

| Predictor seed | Training epochs | Selected epoch | Rows seen by selected weights | Rows seen by complete run |
| --- | --- | --- | --- | --- |
| 17 | 11 | 7 | 261,563 | 261,563 |
| 29 | 11 | 7 | 261,563 | 261,563 |
| 43 | 9 | 5 | 226,880 | 261,563 |

Forecast evaluation covers 120 dates and 3,840 inputs per predictor seed. Across three seeds, 737,280 generated token trajectories are each decoded twice. These are generated scenarios, not independent observed training samples.

| Predictor seed | Days | Original CRPS | Adapted CRPS | Relative change | 95% interval for difference |
| --- | --- | --- | --- | --- | --- |
| 17 | 2 | 1.4678 | 1.4538 | -0.95% | [-0.0198, -0.0086] |
| 29 | 2 | 1.4699 | 1.4618 | -0.55% | [-0.0136, -0.0023] |
| 43 | 2 | 1.4553 | 1.4414 | -0.96% | [-0.0192, -0.0080] |
| mean | 2 | 1.4643 | 1.4523 | -0.82% | [-0.0167, -0.0069] |
| 17 | 5 | 2.3294 | 2.3189 | -0.45% | [-0.0306, +0.0072] |
| 29 | 5 | 2.3346 | 2.3355 | +0.04% | [-0.0177, +0.0185] |
| 43 | 5 | 2.3271 | 2.3240 | -0.13% | [-0.0220, +0.0139] |
| mean | 5 | 2.3304 | 2.3262 | -0.18% | [-0.0224, +0.0121] |

| Days | Metric | Original | Adapted | Difference |
| --- | --- | --- | --- | --- |
| 2 | mae | 1.9470 | 1.9431 | -0.0039 |
| 2 | coverage80 | 67.2848 | 69.4466 | +2.1618 |
| 2 | interval_score80 | 10.3294 | 10.1250 | -0.2044 |
| 5 | mae | 3.1172 | 3.1251 | +0.0079 |
| 5 | coverage80 | 69.6516 | 71.6835 | +2.0319 |
| 5 | interval_score80 | 16.2521 | 16.1015 | -0.1507 |

### Maximum, minimum and range separately

MAE is the absolute error of each predictive median, measured in percentage points of the signal close; it is not a percentage hit rate. CRPS measures error across the predicted distribution. Lower is better for both.

| Days | Target | Original median MAE (pp) | Adapted median MAE (pp) | MAE change | CRPS change |
| --- | --- | --- | --- | --- | --- |
| 2 | maximum | 2.1394 | 2.1375 | -0.09% | -0.69% |
| 2 | minimum | 1.7687 | 1.7684 | -0.02% | -0.68% |
| 2 | range | 1.9328 | 1.9233 | -0.49% | -1.08% |
| 5 | maximum | 3.4575 | 3.4504 | -0.20% | -0.78% |
| 5 | minimum | 2.6382 | 2.6465 | +0.31% | -0.24% |
| 5 | range | 3.2560 | 3.2783 | +0.69% | +0.51% |

| Decoder | Days | Usable inputs / all inputs | Legal sampled draws |
| --- | --- | --- | --- |
| adapted | 2 | 98.61% | 95.91% |
| adapted | 5 | 96.74% | 90.90% |
| frozen | 2 | 97.09% | 91.82% |
| frozen | 5 | 93.54% | 83.15% |

### Identical legal-path sensitivity

This secondary diagnostic was specified before forecast generation. It retains only token trajectories that decode to legal prices under both decoders, with at least 16 shared legal paths per input. Both sides therefore score identical generated tokens and stock/date rows. It changes neither checkpoint selection nor the primary comparison.

| Days | Original CRPS | Adapted CRPS | CRPS change | Median MAE change | 80% coverage |
| --- | --- | --- | --- | --- | --- |
| 2 | 1.4649 | 1.4552 | -0.66% | -0.19% | 67.26% → 68.26% |
| 5 | 2.3335 | 2.3282 | -0.23% | +0.03% | 69.50% → 70.02% |

The modest two-day gain persists after holding the legal-path set fixed. Five-day point accuracy remains essentially unchanged. This separates improvements to the decoded values from changes in which trajectories survive legality filtering.

## Research decision

Retain the price-consistency decoder as a research candidate. Keep the original decoder as the default control until a separate chronological-window confirmation is complete. The adapted decoder reduces observed-token reconstruction MAE by 7.87% and substantially increases valid generated paths, but that gain does not translate into a similarly sized forecast improvement.

Two-day CRPS improves in all three predictor seeds, with a mean difference interval below zero in this window. Median high/low/range MAE improves only 0.20%, and its interval includes zero. Five-day CRPS improves only 0.18%, its interval includes zero, and median MAE worsens 0.25%. There is no established five-day accuracy gain.

Nominal 80% interval coverage remains only 69.45% at two days and 71.68% at five days. The next controlled check should keep this recipe fixed and test its temporal stability, reporting maximum-high and minimum-low accuracy separately as well as distribution calibration. These results do not establish that increasing model size, switching to MoE, or shortening the training horizon would help.

This experiment compares decoders attached to the same self-built predictors; it does not constitute a new comparison against the Kronos Base predictor. The checkpoint is not promoted to the daily strategy model.

## Verification

- All 31 regression tests passed, including loss masking, price-unit invariance, learned consistency gradients, a real decoder update, unchanged token codes and causal encoder/decoder behavior.
- Independent calculations reproduce reconstruction errors, legal masks, date means, and checkpoint/decoder selection. Frozen state tensors match the official tokenizer exactly; CPU reloads reproduce saved MPS reconstructions within checked numerical tolerances.
- Raw forecast paths, all six score fields, date means, paired differences and block intervals are independently recomputed. Predictor CPU logits and the first generated batch replay are verified. The shared-path sensitivity is independently checked using explicit pairwise CRPS.
- Data and source snapshots are hashed. Original tokenizer files are preserved, target boundaries remain chronological, and the sealed holdout remains unused.

## Artifacts

- [Frozen protocol](/Users/guo/Documents/stocks/quant-research/artifacts/tokenizer-reconstruction-20260915-v1/protocol.json)
- [Decoder selection](/Users/guo/Documents/stocks/quant-research/artifacts/tokenizer-reconstruction-20260915-v1/decoder-selection/selection.json)
- [Independent verification](/Users/guo/Documents/stocks/quant-research/artifacts/tokenizer-reconstruction-20260915-v1/independent-verification.json)
- [Complete experiment](/Users/guo/Documents/stocks/quant-research/artifacts/tokenizer-reconstruction-20260915-v1)
