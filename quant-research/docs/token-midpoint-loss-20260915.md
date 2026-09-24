# Midpoint loss: a small incremental improvement

Date: 2026-09-15 (America/Los_Angeles). Completed matched continuation experiment: six fits, twelve forecast sets and independent verification.

## Conclusion

Retain the midpoint objective as a modest research improvement, especially for five-day distribution forecasts. It does not solve the previously observed midpoint weakness. Relative to equal-budget token-loss continuation, five-day midpoint CRPS improves 0.39% and average high/low MAE improves 0.19%. Two-day high/low MAE improves 0.09%, while two-day midpoint CRPS and range errors regress slightly. These tradeoffs remain part of the result.

The new checkpoints also improve on their original warm starts, but much of that improvement comes from additional training. Keep the new objective and exact checkpoints as a frozen, uncalibrated research candidate. The existing calibrated research profile remains the reference until checkpoint-specific calibration and transfer validation are complete. No daily-model default changed.

## What changed

The model retains its 3,720,448 parameters, 60-day history, five-day token generation, frozen decoder and sampling settings. The added training objective scores the midpoint of future highest high and lowest low. It differentiates sampled token log probabilities using decoded real token paths; it adds no price-adjustment rule or prediction head.

Three seeds each received two matched continuations: unchanged token cross-entropy, and cross-entropy plus midpoint-distribution loss with coefficient 0.05. Both arms used identical initial weights and the same 121,728 training rows, drawn from a 354,007-row eligible pool across 1,902 dates. Each auxiliary run applied midpoint gradients to 3,808 of those rows, using four draws each. See the [method](token-midpoint-loss-method.md) for the estimator and its limitations.

## Selection and evaluation

Validation used 896 inputs across 112 dates in 2024 H1. The prespecified mean midpoint CRPS selected coefficient 0.05: 2.440267 → 2.437625, a 0.1083% reduction. The choice was written and hashed before evaluation generation began.

Evaluation used 3,840 input stock-dates across 120 dates in 2024 H2, with 64 forecast paths per input and model. Scoring uses complete labels and at least 16 legal paths, on paired common cohorts within each seed and horizon. Dates and seeds receive equal weight. The following changes are relative to matched token-loss continuation; negative error changes are improvements.

| Metric | Two days | Five days |
| --- | ---: | ---: |
| Midpoint MAE | −0.239% | −0.188% |
| Midpoint CRPS | +0.080% | −0.391% |
| Average high/low MAE | −0.087% | −0.191% |
| Range MAE | +0.107% | +0.044% |
| Range CRPS | +0.131% | −0.008% |

Five-day midpoint CRPS improves in two of three seeds and four of six months; 73 of 120 date-level averages improve. Its mean absolute difference is −0.015094 percentage points, with a paired ten-date block-bootstrap 95% interval of [−0.031592, −0.000367]. Seed 43 has a tiny five-day CRPS regression. Two-day midpoint CRPS worsens on average despite improving in two seeds, because seed 17's regression is larger.

Average high/low MAE improves in three of six seed/horizon cells. Its mean confidence intervals cross zero at both horizons; this qualifies the magnitude rather than automatically rejecting the observed gain. Monthly and individual-seed results are retained in the [evidence tables](token-midpoint-loss-evidence-20260915.md).

Raw average high/low 80% interval coverage changes from 68.26% to 68.54% at two days and from 69.76% to 69.68% at five days. These remain undercovered. Previous calibration factors were not transferred to the changed checkpoints.

## Separate the loss effect from extra training

On a separate common cohort including the original checkpoints:

| Change versus original checkpoint | Two-day high/low MAE | Five-day high/low MAE |
| --- | ---: | ---: |
| Token-loss continuation | −0.502% | −0.435% |
| Midpoint-loss continuation | −0.590% | −0.623% |

The midpoint continuation reduces midpoint MAE by 0.732% / 0.805% and midpoint CRPS by 0.384% / 0.838% versus the originals. Do not attribute all of these gains to the auxiliary objective: the matched continuation comparison above isolates its incremental effect. The three-arm cohort differs slightly from the primary two-arm cohort.

## Verification and limits

All 55,246 primary row-score records and 112 paired estimates were independently recomputed using explicit pairwise CRPS distances. The original-checkpoint reference adds 22,459 scored rows and 112 paired estimates. Verification covered all 7,104 forecast chunks, 12 exact token/price replays totaling 3,072 paths, six exact CPU checkpoint reloads, identical training exposure, source hashes and selection chronology. All 28 consumed source files and 62 previous artifact files were unchanged. The focused suite passed 28 tests; one optional tokenizer integration test was skipped in that environment, while actual pinned-decoder gradient and causal-inference checks passed separately.

This is a small two-epoch, four-draw auxiliary experiment. Validation was reused from warm-start selection, 2024 H2 has appeared in earlier development, and tokenizer pretraining date coverage remains unresolved. The training risk includes all finite decoded paths, while evaluation masks illegal OHLC paths. Results are retrospective and conditional on these fitted models and sampled paths. They establish neither profitability nor a broadly solved midpoint-forecasting problem. The sealed period beginning August 7, 2025 remains unused.

## Next step

Freeze the selected checkpoints, fit checkpoint-specific intervals using earlier validation, and compare the complete frozen forecast pipeline on the existing 2025 transfer window. Preserve the current range improvements and the calibrated reference. This checks whether the small objective gain survives transfer before considering another loss coefficient or a larger model.
