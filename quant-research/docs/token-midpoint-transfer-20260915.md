# Midpoint loss: calibration transfers, predictor gains depend on seed

Date: 2026-09-15 (America/Los_Angeles). Completed checkpoint-specific calibration and frozen transfer evaluation. No neural-network training occurred in this step.

## Decision

Retain the midpoint-loss direction and its verified earlier gains. Do not replace the current calibrated research reference with the complete new checkpoint set yet. On the 2025 transfer window, seeds 17 and 29 improve high/low point accuracy at both horizons, while seed 43 regresses enough to worsen the three-seed average. This is a concentrated stability problem, not evidence that every instance of the change is ineffective.

Across the two examined development windows, the full midpoint continuation improves average high/low MAE in 10 of 12 seed/horizon comparisons versus its original checkpoints. The loss-specific comparison against equal-budget token-loss continuation improves 7 of 12 high/low MAE cases and 8 of 12 midpoint-CRPS cases. Those counts support continued investigation. They do not erase the size of seed 43's regressions or establish 12 independent market experiments.

Interval calibration remains useful: it improves average high/low interval score in five of six seed/horizon cases and preserves point forecasts exactly. Preserve the current calibrated reference, all candidate checkpoints and both control arms. No seed was dropped or selected using the 2025 outcomes.

## What ran

All predictor weights, the frozen decoder and the selected midpoint-loss coefficient 0.05 remained fixed. The previous eight-stock-per-date validation set had only 896 inputs, below the existing 1,000-usable-row requirement for calibration. Both continuation arms therefore generated complete 16-stock-per-date sets on the current reference's exact 1,792 inputs across 112 dates in 2024 H1.

Thirty-six new factors were fitted using the existing date-equal 80th-percentile method, bound to their respective checkpoints and frozen before new transfer generation. Every factor used at least 1,693 valid observations and all 112 dates. Factors range from 1.1626 to 1.3950. The current profile's previous factors were reused unchanged.

Transfer evaluation covers 2,432 input stock-dates across 76 dates, April 1–July 22, 2025, with labels through July 29. Each model generates 64 paths per input. Common usable cohorts contain 2,381 two-day inputs per seed and 2,299–2,301 five-day inputs per seed. Comparisons use identical cohorts across all three profiles, raw and calibrated variants. Stocks, dates and seeds receive the documented equal weighting. Seed means describe average performance, not a price-forecast ensemble.

Six calibration forecast runs and six transfer forecast runs generated 1,622,016 paths. Three existing reference transfer sets were reused exactly. See the [frozen method](token-midpoint-transfer-method.md) for definitions and chronology.

## Calibration works, but does not change point accuracy

| Midpoint candidate | Two days | Five days |
| --- | ---: | ---: |
| Raw average high/low coverage | 71.40% | 72.51% |
| Calibrated average high/low coverage | 82.35% | 83.19% |
| Interval-score change from calibration | −0.40% | −0.29% |
| Interval-width change from calibration | +29.49% | +26.51% |
| Point-MAE change from calibration | Exactly zero | Exactly zero |

Nominal coverage is 80%. The table averages separate high-price and low-price coverage rates; it is not joint coverage of both endpoints or a whole path. Calibration improves coverage substantially but slightly overshoots the target. Its mean interval-score gains have wide confidence intervals, and seed 17 has a five-day interval-score regression. These qualifications do not negate the gains in the other five cases. Interval scaling does not define a full calibrated distribution, so CRPS is reported only for raw forecast samples.

## Three-seed transfer averages

Negative error changes are better. The current reference is the previously retained calibrated profile; the matched CE control received the same two extra training epochs as the midpoint model.

| Metric | Midpoint versus current: 2d / 5d | Midpoint versus matched CE: 2d / 5d |
| --- | ---: | ---: |
| Average high/low MAE | +0.64% / +1.15% | +0.26% / +0.29% |
| High/low interval score | +0.80% / +1.62% | +0.34% / +1.17% |
| Raw midpoint MAE | +0.90% / +1.51% | +0.21% / +0.51% |
| Raw midpoint CRPS | +0.86% / +1.55% | +0.26% / +0.46% |
| Range MAE | +0.88% / +1.25% | +0.43% / +1.05% |

The current reference's average high/low coverage is 81.56% / 82.85%, compared with 82.35% / 83.19% for the new candidate. More coverage alone is not a victory: the candidate has wider intervals and worse average interval score. At five days, the interval-score difference versus current is +0.369824 percentage points, with a paired 95% block-bootstrap interval of [+0.192217, +0.551589]. The loss-specific difference versus matched CE is +0.268702, with interval [+0.100607, +0.425160].

## The average hides a large seed-specific difference

| Training seed | High/low MAE versus current: 2d / 5d | High/low MAE versus matched CE: 2d / 5d |
| --- | ---: | ---: |
| 17 | −0.27% / −0.65% | −0.30% / −0.42% |
| 29 | −1.85% / −3.22% | −1.03% / −2.02% |
| 43 | +3.96% / +7.04% | +2.02% / +3.04% |

Seed 43's high/low MAE regressions have difference intervals entirely above zero at both horizons, against both controls. Its loss-specific interval-score regressions also have positive intervals. Its earlier gains versus the original checkpoint in 2024 H2 did not carry to the later period. The current evidence locates the failure; it does not establish whether optimization, training-period fit, market regime or another mechanism caused it.

For context, the common-three-profile 2024 H2 comparison showed high/low MAE changes of −0.590% / −0.623% versus current and −0.088% / −0.189% versus matched CE. Thus the earlier improvement remains observed evidence, while the later pooled improvement is not confirmed. Range MAE improves in only 4 of 12 loss-specific seed/horizon/window cases, an additional tradeoff to retain in the assessment.

## Verification and interpretation

The independent audit passed all 36 factor reconstructions, 8,160 forecast-chunk checks, 15 exact token/price replays totaling 3,840 paths, 253,128 raw/calibrated row-score checks, 42,129 midpoint-score checks and 608 paired estimates with uncertainty intervals. It verified factor-to-checkpoint bindings, unchanged medians and MAE, source hashes and freezing before transfer generation. Nine focused regression tests passed. The [evidence tables](token-midpoint-transfer-evidence-20260915.md) retain target-specific results, coverage, factors, monthly descriptions and cross-window consistency.

This is retrospective development evidence. Both windows were previously examined, calibration H1 also selected earlier checkpoints and the loss coefficient, and tokenizer pretraining coverage is unresolved. Confidence intervals are pointwise, without multiplicity adjustment, and condition on fixed models and forecast draws. No formal coverage, untouched-test or profitability claim follows. The sealed period beginning August 7, 2025 remains unused.

## Next step

Use the saved paths and pre-transfer checkpoint evidence to diagnose why seed 43 behaves differently across periods and why its continuation amplifies the error. Compare signed midpoint error, range width, target, month and input characteristics against its original checkpoint and the other seeds. Keep all seeds in the reported comparison and do not choose a replacement using the 2025 result. This bounded diagnosis should precede another loss-weight search or an architecture change.
