# Seed 43: continuation and midpoint-loss diagnostic

## Conclusion

Retain the midpoint-loss direction and the gains in seeds 17 and 29. Seed 43 has a consistent direction of forecast movement across the saved periods: the midpoint-loss continuation moves the predicted price-range center downward relative to matched token-only continuation in all three periods. In the 2025 transfer window this increases error, especially for the predicted lows.

The deterioration has two parts. Extra token-only training already worsens seed 43; the midpoint objective adds further deterioration. This is evidence about these fixed checkpoints and sampled paths. It does not identify a defective random seed, prove the objective always fails, or isolate the underlying optimization mechanism.

Keep the current calibrated reference preferred while retaining the new candidate and all three seeds. No checkpoint was reselected, prediction corrected or seed dropped.

## Scope and comparability

Recomputed 27 saved forecast sets: three predictor seeds × three checkpoint owners × three periods. Current means the original 2024-vintage checkpoint. CE means two additional epochs of token-only training. Midpoint means the matched two-epoch continuation with the 0.05 midpoint-distribution objective. Each set contains 64 sampled five-day OHLCVA paths per input.

Periods: 112 signal dates in 2024 H1, 120 in 2024 H2, and 76 in April–July 2025. H1 is reused development/selection data. The 2025 window was already inspected; this is retrospective diagnosis, not a fresh confirmation set. The sealed period beginning 2025-08-07 was not used.

For each seed and horizon, all three owners use the same known-label inputs with at least 16 legal paths per owner. Error averages weight stocks equally within a date and dates equally. Units are percentage points of signal close. Negative error changes are better. Calibrated intervals preserve the endpoint medians, so this raw-path point-error diagnosis applies to the calibrated profiles too; it does not re-estimate interval calibration.

## 1. Extra training and the new loss both contribute

Seed 43 high/low mean absolute error (MAE):

| comparison | horizon | base_mae | new_mae | delta | mae_change_pct | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ce_vs_current | 2 | 2.7802 | 2.8331 | 0.0529 | 1.9028 | 0.0142 | 0.0991 |
| ce_vs_current | 5 | 4.4492 | 4.6221 | 0.1729 | 3.8868 | 0.0143 | 0.3676 |
| midpoint_vs_ce | 2 | 2.8331 | 2.8904 | 0.0573 | 2.0222 | 0.0165 | 0.1102 |
| midpoint_vs_ce | 5 | 4.6221 | 4.7625 | 0.1404 | 3.0378 | 0.0458 | 0.2513 |
| midpoint_vs_current | 2 | 2.7802 | 2.8904 | 0.1102 | 3.9635 | 0.0368 | 0.2042 |
| midpoint_vs_current | 5 | 4.4492 | 4.7625 | 0.3133 | 7.0427 | 0.1017 | 0.5969 |

The absolute changes add: current → CE plus CE → midpoint equals current → midpoint. The percentage changes have different denominators and must not be added. At two/five days, token-only continuation worsens MAE by 1.90% / 3.89%; the midpoint continuation adds 2.02% / 3.04% versus CE. The resulting total is 3.96% / 7.04% versus current.

The 95% intervals above use 2,000 circular resamples of ten consecutive dates, conditional on these checkpoints and draws. They are pointwise, without a multiple-comparison correction. They do not measure variability across new training runs or new forecast samples.

## 2. Most extra endpoint error comes from range placement and predicted lows

We decompose the two endpoint medians into center = (high + low)/2 and half-width = (high − low)/2. Mean high/low absolute error equals the larger of the absolute center error and absolute half-width error. Averaging the two orders of replacing these components gives contributions that add exactly to the MAE change. This is arithmetic attribution, not causal identification or a deployable mixed model.

| horizon | center_share_pct | low_error_share_pct | loss_center_share_pct |
| --- | --- | --- | --- |
| 2 | 83.94 | 83.43 | 92.05 |
| 5 | 81.56 | 90.52 | 94.51 |

Thus center placement accounts for 83.94% / 81.56% of the total deterioration and 92.05% / 94.51% of the loss-specific deterioration. Predicted-low errors account for 83.43% / 90.52% of the total increase in high/low mean MAE.

Seed 43 mean signed center error changes from −0.804 to −0.990 pp at two days and from −2.082 to −2.556 pp at five days. Relative to current, the mean high prediction changes only −0.028 / −0.043 pp, while the mean low prediction moves −0.345 / −0.905 pp. The implied width increases mainly toward the low side. Smaller average width underprediction does not imply better individual width estimates: its arithmetic contribution to endpoint error is also positive.

Actual sampled-midpoint medians and CRPS are scored separately from centers implied by endpoint medians. Those medians need not be equal. Both diagnostics agree on deterioration for seed 43 in 2025; direct midpoint CRPS is in the evidence tables.

## 3. Concentrated in April, but not confined to a few dates or stocks

| comparison | horizon | positive_dates | dates | total_delta | after_top5_removed | after_worst10_removed |
| --- | --- | --- | --- | --- | --- | --- |
| ce_vs_current | 2 | 47 | 76 | 0.0529 | 0.0253 | 0.0275 |
| midpoint_vs_ce | 2 | 52 | 76 | 0.0573 | 0.0337 | 0.0291 |
| midpoint_vs_current | 2 | 56 | 76 | 0.1102 | 0.0740 | 0.0655 |
| ce_vs_current | 5 | 52 | 76 | 0.1729 | 0.1107 | 0.0826 |
| midpoint_vs_ce | 5 | 52 | 76 | 0.1404 | 0.0993 | 0.0896 |
| midpoint_vs_current | 5 | 56 | 76 | 0.3133 | 0.2368 | 0.1793 |

The full midpoint continuation worsens seed 43 on 56/76 dates at both horizons. The isolated midpoint objective worsens it on 52/76 dates. Removing either the five largest positive daily differences or the worst consecutive ten-date block leaves all six seed-43 continuation comparisons worse. These removals are sensitivity checks only; no observations were removed from the reported primary results.

| horizon | group | dates | rows | endpoint_mae_delta | global_delta_contribution |
| --- | --- | --- | --- | --- | --- |
| 2 | 2025-04 | 21 | 668 | 0.2847 | 0.0787 |
| 2 | 2025-05 | 19 | 597 | 0.0593 | 0.0148 |
| 2 | 2025-06 | 20 | 615 | 0.0372 | 0.0098 |
| 2 | 2025-07 | 16 | 501 | 0.0329 | 0.0069 |
| 5 | 2025-04 | 21 | 656 | 0.7720 | 0.2133 |
| 5 | 2025-05 | 19 | 572 | 0.1544 | 0.0386 |
| 5 | 2025-06 | 20 | 579 | 0.0871 | 0.0229 |
| 5 | 2025-07 | 16 | 492 | 0.1829 | 0.0385 |

All four months worsen versus current. April contributes about 71% / 68% of the net two/five-day error increase. Relative to CE, the five-day midpoint objective improves in May, but worsens in April, June and July.

The two/five-day cohorts contain 1,640 / 1,590 distinct instruments. Of these, 955 / 999 have a positive average difference versus current, but the median instrument appears only once. Individual-stock results are too sparse to support stock-specific rules. The five largest positive instrument contributions explain only about 8.1% / 6.6% of the total deterioration.

## 4. Input characteristics

Past volatility uses twenty adjusted-close log returns; trend uses the corresponding twenty-day adjusted-close return. All inputs end on the signal date. Volatility tercile thresholds are calculated solely from the 1,792 H1 inputs. This rule was fixed before computing the diagnostic cuts; the transfer outcomes were already known and this is not a prospective test.

| horizon | cut | group | dates | rows | endpoint_mae_delta | global_delta_contribution |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | volatility_group | high | 76 | 708 | 0.0905 | 0.0651 |
| 2 | volatility_group | low | 73 | 949 | 0.0525 | 0.0115 |
| 2 | volatility_group | middle | 76 | 724 | 0.1154 | 0.0335 |
| 2 | trend_group | negative | 76 | 1028 | 0.1524 | 0.0855 |
| 2 | trend_group | nonnegative | 76 | 1353 | 0.0550 | 0.0247 |
| 5 | volatility_group | high | 76 | 687 | 0.4184 | 0.1947 |
| 5 | volatility_group | low | 73 | 911 | 0.1294 | 0.0395 |
| 5 | volatility_group | middle | 76 | 701 | 0.2775 | 0.0791 |
| 5 | trend_group | negative | 76 | 998 | 0.3966 | 0.2282 |
| 5 | trend_group | nonnegative | 76 | 1301 | 0.2142 | 0.0851 |

Each listed group worsens versus current at both horizons. High-volatility observations contribute about 59% / 62% of the total deterioration. Negative past-trend observations contribute about 78% / 73%. These are overlapping cuts, not additive independent causes or a validated routing policy.

The group error column averages within group and date, then over dates where the group exists. The global contribution column preserves the full-cohort date weights and sums to the overall difference within each cut. Group error multiplied by a simple overall row share need not equal its global contribution when group composition changes over dates.

## 5. What could earlier validation tell us?

The original loss selection used 896 inputs, eight per date, and selected midpoint loss on the mean midpoint CRPS across three seeds and two horizons. Its average advantage was only 0.1083%. Seed 43 improved on that criterion at both horizons; it was not the failing seed on the selection metric.

Original selection changes, midpoint versus CE:

| seed | horizon | endpoint_mae_change_pct | midpoint_crps_change_pct |
| --- | --- | --- | --- |
| 17 | 2 | 0.7719 | 0.5529 |
| 17 | 5 | 0.6331 | 0.8939 |
| 29 | 2 | -0.9124 | -0.7953 |
| 29 | 5 | 0.8016 | 0.4315 |
| 43 | 2 | -0.3390 | -0.3709 |
| 43 | 5 | -0.7072 | -1.3876 |

On the larger, matched 1,792-input H1 calibration forecast set, seed 43 also improves endpoint MAE versus CE by 0.22% / 0.37%, while remaining worse than its original checkpoint by 1.17% / 0.32%. Token-only continuation was already worse than the original by 1.39% / 0.70%. This was a warning about continuation stability, not evidence that the midpoint objective should have been rejected.

A more specific warning is the consistent signed center movement relative to CE:

| window | horizon | delta | ci_low | ci_high |
| --- | --- | --- | --- | --- |
| 2024h1 | 2 | -0.1193 | -0.1901 | -0.0404 |
| 2024h1 | 5 | -0.2482 | -0.3721 | -0.1038 |
| 2024h2 | 2 | -0.1010 | -0.1960 | -0.0252 |
| 2024h2 | 5 | -0.2127 | -0.3989 | -0.0612 |
| 2025transfer | 2 | -0.1153 | -0.1708 | -0.0609 |
| 2025transfer | 5 | -0.2861 | -0.4221 | -0.1471 |

The downward movement is present in all three periods. Its sign alone is not a failure criterion: it can improve forecasts in a period where downward movement is helpful. In 2024 H2, midpoint continuation still improves seed 43 versus the original at both horizons, although its five-day loss-specific MAE worsens by 0.38%. The 2025 result therefore cannot justify retroactively selecting or rejecting a seed based on its identity.

Saved continuation histories contain training CE, sampled auxiliary loss and legal-path fractions, but no per-epoch out-of-sample price forecast trajectory. Both continuation arms used a fixed final two-epoch checkpoint. The original warm-start selection CE and continuation training CE are different quantities; their numerical levels do not establish improved validation. All three midpoint runs have a larger second-epoch sampled training midpoint CRPS on different sampled rows, so that trace is not a unique warning for seed 43. The saved artifacts cannot show whether an earlier continuation checkpoint would avoid the regression.

## 6. Controls, uncertainty and next experiment

Seeds 17 and 29 retain their 2025 gains:

| seed | comparison | horizon | mae_change_pct |
| --- | --- | --- | --- |
| 17 | ce_vs_current | 2 | 0.0315 |
| 17 | ce_vs_current | 5 | -0.2300 |
| 29 | ce_vs_current | 2 | -0.8267 |
| 29 | ce_vs_current | 5 | -1.2279 |
| 43 | ce_vs_current | 2 | 1.9028 |
| 43 | ce_vs_current | 5 | 3.8868 |
| 17 | midpoint_vs_ce | 2 | -0.3010 |
| 17 | midpoint_vs_ce | 5 | -0.4203 |
| 29 | midpoint_vs_ce | 2 | -1.0313 |
| 29 | midpoint_vs_ce | 5 | -2.0207 |
| 43 | midpoint_vs_ce | 2 | 2.0222 |
| 43 | midpoint_vs_ce | 5 | 3.0378 |
| 17 | midpoint_vs_current | 2 | -0.2696 |
| 17 | midpoint_vs_current | 5 | -0.6494 |
| 29 | midpoint_vs_current | 2 | -1.8495 |
| 29 | midpoint_vs_current | 5 | -3.2238 |
| 43 | midpoint_vs_current | 2 | 3.9635 |
| 43 | midpoint_vs_current | 5 | 7.0427 |

The all-nine-model shared-input sensitivity preserves the finding: seed 43 worsens 3.97% / 7.14%, while seeds 17 and 29 improve at both horizons. Unequal legal-input cohorts do not explain the sign pattern. Legal-path fractions remain around 97% at two days and 93% at five days for seed 43; the error increase is not accompanied by a collapse in usable paths.

Retain the previously observed gains: full continuation improved high/low MAE in 10/12 seed/horizon cases across the two evaluation periods; the isolated midpoint loss improved 7/12. Those counts do not erase seed 43's larger downside, and the downside does not erase the other improvements.

The next bounded experiment should repeat forecast sampling from the same original, CE and midpoint checkpoints with additional fixed draw seeds, retaining all three training seeds and the existing date/stock cohort. Compare signed center movement and low-price MAE as well as overall error. This will test whether the observed drift is stable across sampled paths before changing the training objective again. If it persists, the next training comparison should save per-epoch price validation and test a constrained continuation, with its rule fixed on development data. Neither follow-up was launched here.

## Verification and artifacts

Ready within the reviewed scope, with the development-data and fixed-draw limitations above. Independent code reconstructed every raw-path endpoint, center/width decomposition and midpoint CRPS using explicit pairwise distances; reconciled prior 2024 H2 and 2025 scores; checked input-only features, all attribution rows, date/group weights, concentration removals, shared-seed cohorts, original selection and every paired estimate/interval. Eight focused tests passed. No neural training, forecasting or calibration fitting ran.

- [Detailed evidence](token-seed43-diagnostic-evidence-20260915.md)
- [Protocol](../artifacts/token-seed43-diagnostic-20260915-v1/protocol.json)
- [Independent verification](../artifacts/token-seed43-diagnostic-20260915-v1/independent-verification.json)
- [Final source-bound receipt](../artifacts/token-seed43-diagnostic-20260915-v1/final-verification.json)
- [Analysis script](../scripts/token_seed43_diagnostic.py)
- [Independent verifier](../scripts/verify_token_seed43_diagnostic.py)
