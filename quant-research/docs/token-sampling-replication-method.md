# Frozen-checkpoint sampling replication

## Question

Does seed 43's downward shift and greater low-price error persist when the same checkpoints generate different sampled futures? Compare all three training seeds and all three checkpoint owners so the diagnosis retains the improvements previously seen in other seeds.

## Fixed design

- Original, token-only continuation and midpoint-loss continuation checkpoints, each with training seeds 17, 29 and 43.
- Reuse generation seed 17 and add generation seeds 100017 and 200017. Each four-input batch uses the generation seed plus its input offset. The three seed ranges do not overlap.
- Keep the decoder, 60-day input context, five-day forecast, sampling temperature 1, full token distribution, four-input batch shape and 64 paths per input unchanged.
- Generate all 18 new sets regardless of interim results. This yields 2,801,664 new five-day paths. The nine original sets contribute another 1,400,832 saved paths.
- Use exactly the existing 2,432 inputs: 32 stock-date inputs on each of 76 dates from April 1 through July 22, 2025. Last labels end July 29. No new stocks or dates are selected.
- No model fitting, checkpoint selection, seed dropping, price correction, calibration fitting or access to the sealed period beginning August 7, 2025.

## Comparisons

Primary comparisons use one common input set per horizon: known labels and at least 16 legal paths in each of the 27 forecast sets. This prevents different stock/date eligibility from changing comparisons between training seeds or draw replicates. Report all exclusions relative to the original nine-model shared cohort. As a sensitivity analysis, also score the three-owner common cohort separately for each training seed and draw replicate.

Report the two- and five-day high/low mean absolute error, high and low errors separately, signed midpoint and half-width errors implied by endpoint medians, direct sampled-midpoint MAE and CRPS, sampled-range MAE and legal-path fraction. CRPS compares the complete sampled distribution with the realized value, using both distance to the realization and pairwise sample distances.

Compare current → token-only continuation, token-only → midpoint-loss continuation, and current → midpoint-loss continuation. Average stocks within date, then dates equally. Across replicates, average metrics from each 64-path forecast; do not pool paths into a 192-path distribution or average prices into an ensemble. Keep each training seed visible, as well as the equal-seed average.

## Uncertainty and limits

For each draw replicate and for their mean, calculate pointwise 95% intervals from 2,000 circular bootstrap resamples of ten consecutive dates, with bootstrap seed 314159. Report the range across the three draw replicates descriptively. Three replicates do not establish a precise sampling-error confidence interval, and date bootstrap intervals do not include uncertainty from retraining models.

This is diagnosis on previously examined development dates. Repeating sampling can test sensitivity to Monte Carlo draws; it does not supply a new market period or establish out-of-sample trading performance. It also cannot identify a causal optimizer mechanism. Retain the existing tokenizer provenance limitations.

## Verification

Bind the exact input IDs, checkpoint and decoder hashes, original forecast hashes, data arrays, protocol and executable code before generation. Preserve every new chunk's row IDs, coarse/fine tokens, prices and legality mask. Reconstruct every score independently, using explicit pairwise-distance CRPS; independently verify common cohorts, date/draw/seed weighting and every confidence interval. Check every saved chunk and replay the first and last batches of each new forecast set after checkpoint reload. Reconcile reused results against the preceding diagnostic.

The final report should state whether the regression's direction is stable across draw seeds, whether other seeds retain gains, and what uncertainty remains. The current profile changes only after a separate justified decision; this experiment itself makes no promotion.
