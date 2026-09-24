# Frozen-checkpoint sampling replication

## Conclusion

Seed 43's deterioration persists across all three forecast sampling runs. Its downward range-center movement and extra low-price error are not explained by the original single draw set. This supports a stability problem in the learned continuation checkpoints on this development period; it does not identify the causal optimizer mechanism or establish that this training seed is intrinsically bad.

Retain the midpoint-loss direction and the gains in seeds 17 and 29. Keep the current calibrated reference preferred. No training seed or draw replicate was discarded, and no production/default model changed.

## Experiment

Reused the nine original forecast sets and completed eighteen new sets: original, token-only continuation and midpoint-loss continuation checkpoints, each with training seeds 17, 29 and 43, under two additional forecast RNG seeds. Decoder, predictor weights, input IDs, sampling settings, four-input batch shape and 64 paths per input were unchanged. The additional RNG seed ranges are disjoint from each other and from the original.

All 2,801,664 new five-day OHLCVA paths are saved. Together with the original forecasts, the analysis covers 4,202,496 paths across 27 sets. There were no training runs or calibration fits.

The same 2,432 inputs cover 76 signal dates from April 1 through July 22, 2025, with labels ending July 29. The primary cohort requires known labels and at least 16 legal paths in every set. It contains 2,376 two-day inputs and 2,287 five-day inputs on all 76 dates. This excludes only three/seven inputs from the previous all-nine-model shared cohort. A separate within-draw/seed comparison checks sensitivity to that restriction.

Stocks have equal weight within date, dates have equal weight, and sampling replicates have equal weight. We average the errors from each 64-path forecast; this is neither an ensemble of forecast prices nor a pooled 192-path distribution. Negative error changes are better. Absolute differences are percentage points of the signal close. In the tables, draw 0 denotes the three-draw score average, and seed 0 denotes the three-training-seed score average.

## Average result by training seed

Midpoint-loss continuation versus the original/current checkpoint:

| seed | horizon | baseline | candidate | relative_change_pct | delta | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 17 | 2 | 2.7046 | 2.6916 | -0.4784 | -0.0129 | -0.0418 | 0.0157 |
| 17 | 5 | 4.2015 | 4.1568 | -1.0642 | -0.0447 | -0.1032 | 0.0068 |
| 29 | 2 | 2.7208 | 2.6693 | -1.8945 | -0.0515 | -0.0868 | -0.0229 |
| 29 | 5 | 4.2230 | 4.0894 | -3.1633 | -0.1336 | -0.2685 | -0.0316 |
| 43 | 2 | 2.7688 | 2.8882 | 4.3116 | 0.1194 | 0.0445 | 0.2166 |
| 43 | 5 | 4.3949 | 4.7381 | 7.8074 | 0.3431 | 0.1291 | 0.6278 |

The gains in seeds 17 and 29 remain. Seed 43's larger downside makes the equal-seed mean worse by 0.67% / 1.29% at two/five days. The two-day date-bootstrap interval crosses zero; the five-day interval is slightly above zero. These intervals are conditional on this fixed collection of models and draws and are not adjusted for multiple comparisons.

## Seed 43: both continuation effects reproduce

| draw | horizon | comparison | relative_change_pct | delta | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 2 | ce_vs_current | 2.3499 | 0.0651 | 0.0218 | 0.1173 |
| 0 | 2 | midpoint_vs_ce | 1.9166 | 0.0543 | 0.0149 | 0.1040 |
| 0 | 2 | midpoint_vs_current | 4.3116 | 0.1194 | 0.0445 | 0.2166 |
| 0 | 5 | ce_vs_current | 4.4348 | 0.1949 | 0.0545 | 0.3718 |
| 0 | 5 | midpoint_vs_ce | 3.2294 | 0.1482 | 0.0440 | 0.2754 |
| 0 | 5 | midpoint_vs_current | 7.8074 | 0.3431 | 0.1291 | 0.6278 |
| 17 | 2 | ce_vs_current | 1.9148 | 0.0532 | 0.0144 | 0.0999 |
| 17 | 2 | midpoint_vs_ce | 2.0324 | 0.0576 | 0.0167 | 0.1106 |
| 17 | 2 | midpoint_vs_current | 3.9861 | 0.1108 | 0.0373 | 0.2045 |
| 17 | 5 | ce_vs_current | 3.9796 | 0.1759 | 0.0196 | 0.3681 |
| 17 | 5 | midpoint_vs_ce | 3.0588 | 0.1405 | 0.0442 | 0.2521 |
| 17 | 5 | midpoint_vs_current | 7.1601 | 0.3164 | 0.1069 | 0.5987 |
| 100017 | 2 | ce_vs_current | 2.5137 | 0.0693 | 0.0232 | 0.1279 |
| 100017 | 2 | midpoint_vs_ce | 1.9157 | 0.0542 | 0.0117 | 0.1050 |
| 100017 | 2 | midpoint_vs_current | 4.4775 | 0.1235 | 0.0484 | 0.2289 |
| 100017 | 5 | ce_vs_current | 4.2058 | 0.1843 | 0.0572 | 0.3469 |
| 100017 | 5 | midpoint_vs_ce | 3.7849 | 0.1729 | 0.0509 | 0.3352 |
| 100017 | 5 | midpoint_vs_current | 8.1499 | 0.3572 | 0.1335 | 0.6695 |
| 200017 | 2 | ce_vs_current | 2.6233 | 0.0726 | 0.0208 | 0.1306 |
| 200017 | 2 | midpoint_vs_ce | 1.8022 | 0.0512 | 0.0099 | 0.0975 |
| 200017 | 2 | midpoint_vs_current | 4.4728 | 0.1238 | 0.0451 | 0.2186 |
| 200017 | 5 | ce_vs_current | 5.1230 | 0.2245 | 0.0810 | 0.4020 |
| 200017 | 5 | midpoint_vs_ce | 2.8487 | 0.1312 | 0.0291 | 0.2468 |
| 200017 | 5 | midpoint_vs_current | 8.1176 | 0.3558 | 0.1452 | 0.6314 |

Across the three draws, total high/low MAE deterioration ranges from 3.99% to 4.48% at two days and 7.16% to 8.15% at five days. Relative to matched CE continuation, the midpoint objective worsens it in all six draw/horizon cases: 1.80–2.03% at two days and 2.85–3.78% at five days.

On the three-draw average, token-only continuation adds 2.35% / 4.43% error relative to current; midpoint loss adds 1.92% / 3.23% relative to CE. Their combined result is +4.31% / +7.81% versus current. The percentage changes have different denominators and must not be added.

Original-draw numbers differ slightly from the preceding report because every row here uses the stricter common 27-set cohort. The original paths were reused unchanged and their row-level scores reconcile exactly with the preceding diagnostic.

## The downward shift persists

Change in implied range-center bias for seed 43, midpoint loss versus CE:

| draw | horizon | delta | ci_low | ci_high |
| --- | --- | --- | --- | --- |
| 0 | 2 | -0.1287 | -0.1793 | -0.0814 |
| 0 | 5 | -0.3253 | -0.4622 | -0.1861 |
| 17 | 2 | -0.1146 | -0.1708 | -0.0596 |
| 17 | 5 | -0.2855 | -0.4207 | -0.1454 |
| 100017 | 2 | -0.1414 | -0.1947 | -0.0928 |
| 100017 | 5 | -0.3664 | -0.5311 | -0.2084 |
| 200017 | 2 | -0.1302 | -0.1734 | -0.0864 |
| 200017 | 5 | -0.3240 | -0.4513 | -0.2032 |

The mean center moves downward by 0.129 / 0.325 pp at two/five days relative to CE, with the same sign in every draw. Relative to current, the total downward movement is 0.194 / 0.483 pp. On average, seed 43's predicted-low MAE increases 8.46% / 15.91% versus current, and 3.51% / 6.49% versus CE. Direct sampled-midpoint CRPS also worsens 1.87% / 3.27% versus CE. The finding therefore appears in the sampled midpoint distribution as well as the endpoint medians.

The other seeds do not show the same average loss-specific center movement: seed 29 moves upward at both horizons, while seed 17's movement is small. Seed 17's two-day loss-specific MAE improvement is weak and flips slightly positive in one draw (+0.075%); its three-draw mean still improves. Seed 29 improves against CE at both horizons in every draw. This preserves the distinction between a useful direction and its sensitivity to training initialization/checkpoint state.

## Equal-seed average and cohort sensitivity

| horizon | comparison | relative_change_pct | delta | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- |
| 2 | ce_vs_current | 0.3734 | 0.0102 | -0.0090 | 0.0321 |
| 2 | midpoint_vs_ce | 0.2954 | 0.0081 | -0.0029 | 0.0213 |
| 2 | midpoint_vs_current | 0.6699 | 0.0183 | -0.0018 | 0.0437 |
| 5 | ce_vs_current | 0.9450 | 0.0404 | -0.0116 | 0.0979 |
| 5 | midpoint_vs_ce | 0.3376 | 0.0146 | -0.0126 | 0.0416 |
| 5 | midpoint_vs_current | 1.2858 | 0.0549 | 0.0019 | 0.1180 |

The isolated loss-specific average worsens 0.30% / 0.34% versus CE, with date intervals crossing zero. Both the individual-seed gains and the pooled downside remain relevant. Three draw replicates on the same dates are not three independent market tests and must not be counted as new independent scenarios supporting or rejecting the method.

Within-draw/seed sensitivity, averaged across draws:

| seed | horizon | comparison | relative_change_pct |
| --- | --- | --- | --- |
| 0 | 2 | ce_vs_current | 0.3960 |
| 0 | 2 | midpoint_vs_ce | 0.3120 |
| 0 | 2 | midpoint_vs_current | 0.7092 |
| 0 | 5 | ce_vs_current | 1.0042 |
| 0 | 5 | midpoint_vs_ce | 0.3264 |
| 0 | 5 | midpoint_vs_current | 1.3338 |
| 17 | 2 | ce_vs_current | -0.2892 |
| 17 | 2 | midpoint_vs_ce | -0.1109 |
| 17 | 2 | midpoint_vs_current | -0.3998 |
| 17 | 5 | ce_vs_current | -0.5072 |
| 17 | 5 | midpoint_vs_ce | -0.3850 |
| 17 | 5 | midpoint_vs_current | -0.8903 |
| 29 | 2 | ce_vs_current | -0.8940 |
| 29 | 2 | midpoint_vs_ce | -0.9660 |
| 29 | 2 | midpoint_vs_current | -1.8513 |
| 29 | 5 | ce_vs_current | -0.9844 |
| 29 | 5 | midpoint_vs_ce | -2.1731 |
| 29 | 5 | midpoint_vs_current | -3.1361 |
| 43 | 2 | ce_vs_current | 2.3360 |
| 43 | 2 | midpoint_vs_ce | 1.9333 |
| 43 | 2 | midpoint_vs_current | 4.3145 |
| 43 | 5 | ce_vs_current | 4.3520 |
| 43 | 5 | midpoint_vs_ce | 3.2482 |
| 43 | 5 | midpoint_vs_current | 7.7415 |

## Next optimization

Further resampling is no longer the first priority: the current bounded replication reproduces the main finding. The next controlled training experiment should focus on continuation stability and checkpoint selection, while keeping the model architecture and useful midpoint objective.

Specifically, save every continuation epoch and evaluate price-space metrics, signed center bias and low-price error on predeclared development subperiods. Compare selecting a checkpoint by midpoint-distribution validation with the existing fixed-final continuation, including the original checkpoint as an explicit baseline. Keep all training seeds and fix the selection rule before evaluating later dates. The earlier artifacts saved no per-epoch price-validation trajectory, so this experiment is needed to determine whether earlier stopping can preserve the gains without the seed-43 drift. No such training was started here, and this diagnosis is not evidence that earlier stopping will necessarily solve it.

## Verification and limits

Independent reconstruction passed for all 126,550 usable score rows and all 1,728 paired estimates and date-bootstrap intervals. Every one of 16,416 saved chunks was checked against its row IDs, path array, tokens, legality mask and source hash. Forty-five exact checkpoint-reload replays reproduced 11,520 token/price paths, including the first and last batch of every new set. Nine focused tests passed. Inputs, source forecasts, checkpoints and decoder hashes remained unchanged.

Ready within this scope. This is previously examined development data, not a fresh market holdout. Three RNG replicates test sampling sensitivity; they do not quantify uncertainty from retraining or establish performance in another market period. The date intervals are pointwise and conditional. Existing tokenizer pretraining-provenance limitations remain. The sealed period beginning August 7, 2025 was not accessed.

- [Method](token-sampling-replication-method.md)
- [Detailed evidence](token-sampling-replication-evidence-20260915.md)
- [Frozen protocol](../artifacts/token-sampling-replication-20260915-v1/protocol.json)
- [Independent arithmetic](../artifacts/token-sampling-replication-20260915-v1/arithmetic-verification.json)
- [Path and replay verification](../artifacts/token-sampling-replication-20260915-v1/replay-verification.json)
- [Final receipt](../artifacts/token-sampling-replication-20260915-v1/final-verification.json)
