# Midpoint-loss calibration and transfer protocol

The experiment completes the frozen midpoint-loss candidate's interval-calibration and later-period comparison. It does not train another predictor or change the selected loss weight.

## Three fixed profiles

- Current reference: the refreshed predictors and their existing checkpoint-specific 2024 H1 interval factors. Reuse the verified April–July 2025 paths exactly.
- Token-loss continuation: the three fixed checkpoints trained for two additional epochs with token cross-entropy alone.
- Midpoint-loss continuation: the three matched checkpoints trained with coefficient 0.05, already selected on 2024 H1 before the preceding 2024 H2 evaluation.

All profiles share the 3,720,448-parameter architecture, 60-day context, five-day forecast, frozen decoder and sampling method. Seeds are 17, 29 and 43. Each input produces 64 paths using temperature 1, full-distribution sampling and the existing seed-plus-four-row-offset convention.

## Earlier-period calibration

The preceding loss-selection experiment used eight stocks per date, only 896 inputs. The established calibration method requires at least 1,000 usable rows and 80 dates per factor. Generate complete 16-stock-per-date calibration sets for both continuation arms, matching the current reference's exact 1,792 inputs across 112 signal dates from January 2 to June 21, 2024. Labels end by June 28.

For each seed, horizon and target, extract the 10th, 50th and 90th percentiles of legal generated draws. Targets are the future maximum high, minimum low and total range at two and five trading days, expressed as percentage-point returns from the signal-date close. Complete labels and at least 16 legal paths are required. Each model fits its own usable inputs from the same prespecified stock-date pool.

Calculate a residual divided by the appropriate median-to-bound distance. Fit its date-equal weighted 80th percentile, with a minimum scale of one. This widens intervals around an exactly preserved median. A small positive denominator handles degenerate bounds; lower support is −100 percentage points for high and low returns and zero for range. Use the existing algorithm and settings without tuning. Each of the six continuation checkpoints receives six factors, for 36 newly fitted factors in total.

Bind every factor set to the exact predictor and decoder hashes, then freeze it before any new 2025 generation or scoring. The reference's previous factors remain unchanged. Calibration adjusts interval bounds only; it does not define a calibrated full probability distribution. Consequently report CRPS only for raw sampled distributions.

## Later-period comparison

Generate both continuation arms on the fixed 2,432 stock-date inputs across 76 dates from April 1 to July 22, 2025. Labels end by July 29. Compare all three profiles on the intersection of usable inputs within each training seed and horizon. Use the same common cohort for raw and calibrated scores, including raw midpoint accuracy.

Weight stocks equally within a date, dates equally within a seed, and seeds equally in the pooled result. Report maximum-high and minimum-low scores individually and their equal average. Mean endpoint coverage is the average of two marginal coverage rates, not joint coverage of both endpoints or an entire price path.

The complete midpoint pipeline versus the current calibrated reference measures the overall change. The midpoint pipeline versus matched token-loss continuation isolates the incremental loss-design change, including the corresponding calibration response. Raw-versus-calibrated comparisons isolate interval scaling and must preserve point MAE exactly. Also report range accuracy, raw distribution CRPS, midpoint MAE/CRPS, interval widths, miss rates, usable input counts and legal-path rates.

Use paired ten-date circular block-bootstrap intervals with 2,000 replicates and seed 314159. These are pointwise intervals without multiplicity adjustment, conditional on the fitted models and forecast samples. Assess magnitude, seed and target consistency and local regressions together. A confidence interval crossing zero alone is not a veto. Monthly summaries are descriptive and never select factors, weights or checkpoints.

## Scope and verification

The 2025 window has already appeared in development research. The calibration period also selected earlier checkpoints and the midpoint coefficient. Original tokenizer pretraining date coverage is unresolved. This is retrospective transfer evidence, not an untouched final test, formal coverage guarantee or profitability claim. The sealed period beginning August 7, 2025 remains unused.

Independent checks recompute raw quantiles, date-weighted factors, midpoint and endpoint CRPS with explicit pairwise distances, interval metrics, common cohorts, date aggregation and paired uncertainty intervals. Verify all forecast chunks, exact token-and-price replays, factor-to-checkpoint bindings, preserved medians and source hashes. The current calibrated reference and daily-model defaults remain unchanged during the experiment.
