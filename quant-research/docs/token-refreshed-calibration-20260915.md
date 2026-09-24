# Checkpoint-specific interval calibration after training refresh

Date: 2026-09-15 (America/Los_Angeles). Completed historical research on 2023 H2 and 2024 H2, three predictor seeds per window.

## Conclusion

Retain checkpoint-specific interval calibration together with training refresh. For refreshed predictors, average maximum-high/minimum-low coverage rises from 68.15% to 77.38% and from 69.47% to 78.01% in 2023 H2, and from 69.04% to 79.00% and from 69.94% to 79.82% in 2024 H2, at two and five days respectively. Average interval score improves in all four window/horizon combinations and all 12 individual seed/window/horizon combinations. The full refreshed-and-calibrated pipeline also improves average interval score over the old calibrated pipeline in all 12 seed/window/horizon combinations, while exactly preserving the earlier point-MAE gains.

Calibration widens the refreshed high/low intervals by 20.87–26.81%. Average interval-score improvements from calibration alone are 0.86–2.01%; the 2024 H2 confidence intervals include zero. Across the 12 target/window/horizon averages, calibration improves interval score in 10, and worsens it in two. Coverage near 80% in the mean does not imply that each individual target is equally calibrated. These limitations qualify the result without negating the improvement across most tested scenarios.

## Calibration effect on the refreshed predictors

This table equally averages maximum-high and minimum-low metrics across three seeds. Coverage is the fraction of realized targets inside the nominal 80% interval. Lower interval score is better: it accounts for both interval width and misses. Negative score changes indicate improvement. The confidence interval is for the score difference in percentage points of signal-date close.

| Window | Days | Raw coverage | Calibrated coverage | Width change | Interval score change | 95% CI of score difference |
| --- | --- | --- | --- | --- | --- | --- |
| 2023h2 | 2 | 68.15% | 77.38% | +24.41% | -2.01% | [-0.3588, -0.0695] |
| 2023h2 | 5 | 69.47% | 78.01% | +20.87% | -1.52% | [-0.5155, -0.0269] |
| 2024h2 | 2 | 69.04% | 79.00% | +26.81% | -1.53% | [-0.9049, +0.2394] |
| 2024h2 | 5 | 69.94% | 79.82% | +23.86% | -0.86% | [-1.1764, +0.5028] |

Calibration changes only the lower and upper bounds around the existing median. Every median and every row-level point MAE is exactly unchanged. It does not calibrate the full forecast distribution, so this report makes no claim of improved CRPS from calibration.

## Complete refreshed pipeline versus the previous calibrated models

The comparison below uses exactly matched stock-date inputs. The old models were trained through 2021, and their calibration factors were fitted on 2023 H1 and held fixed for both evaluation windows. Refreshed models were trained through 2022 or 2023 and calibrated on the corresponding 2023 H1 or 2024 H1 forecasts. The adapted decoder and sampler are identical. Thus this comparison includes both predictor refresh and calibration refresh.

| Window | Days | Old calibrated coverage | Refreshed calibrated coverage | Interval score change | Point MAE change | Width change |
| --- | --- | --- | --- | --- | --- | --- |
| 2023h2 | 2 | 77.87% | 77.37% | -1.37% | -1.24% | -3.61% |
| 2023h2 | 5 | 77.89% | 78.03% | -1.04% | -1.14% | -3.37% |
| 2024h2 | 2 | 77.94% | 79.01% | -3.31% | -1.72% | -3.58% |
| 2024h2 | 5 | 76.85% | 79.84% | -2.51% | -3.31% | -3.03% |

The point-MAE changes reproduce the previous training-refresh comparison on its original common cohorts. The first table uses every usable refreshed-model input; the second uses the old/new intersection. Coverage figures from these tables need not be identical because their cohorts differ.

## Individual targets

| Window | Days | Target | Raw coverage | Calibrated coverage | Interval score change |
| --- | --- | --- | --- | --- | --- |
| 2023h2 | 2 | maximum | 67.41% | 77.82% | -2.36% |
| 2023h2 | 2 | minimum | 68.89% | 76.94% | -1.48% |
| 2023h2 | 2 | range | 66.05% | 78.58% | -2.59% |
| 2023h2 | 5 | maximum | 68.12% | 78.28% | -1.87% |
| 2023h2 | 5 | minimum | 70.82% | 77.73% | -0.78% |
| 2023h2 | 5 | range | 69.84% | 78.30% | -1.41% |
| 2024h2 | 2 | maximum | 67.41% | 77.73% | -2.62% |
| 2024h2 | 2 | minimum | 70.66% | 80.28% | +0.50% |
| 2024h2 | 2 | range | 68.44% | 80.03% | -1.46% |
| 2024h2 | 5 | maximum | 67.48% | 76.04% | -2.22% |
| 2024h2 | 5 | minimum | 72.41% | 83.61% | +2.89% |
| 2024h2 | 5 | range | 72.45% | 79.04% | -1.09% |

The two average regressions from calibration alone are the 2024 H2 minimum-low target: interval score increases 0.50% at two days and 2.90% at five days. Five-day low-price coverage reaches 83.61%, while five-day high-price coverage is 76.04%, suggesting that the remaining calibration error differs between targets. Against the old calibrated pipeline, however, the refreshed calibrated minimum-low interval score improves 5.78% and 7.09% respectively. The complete pipeline improves all 12 target/window/horizon average interval scores and 33 of 36 individual seed/target cases; these are descriptive historical comparisons.

## Frozen method and chronology

There are 36 independently fitted factors: two checkpoint vintages × three training seeds × two forecast horizons × three targets. Each factor uses only its own checkpoint’s earlier H1 validation forecasts. The exact predictor hash, decoder hash and sampling settings are attached to each fit. The old 18 factors are not reused.

Within each fit, every signal date has equal total weight. The scale is the weighted empirical 80th percentile of the residual normalized by the left or right raw half-width, with a minimum of one. It widens the interval around the unchanged median. Lower bounds respect the target support: −100% for high/low returns, and zero for range. The numerical half-width floor is 0.000001 percentage points.

All 36 factors were frozen before this run scored either H2 evaluation. Signals and their entire five-day label windows stay inside the corresponding partition. Calibration uses 16 stocks per date, evaluation 32, selected before checking labels. A horizon requires complete known labels and at least 16 legal draws among 64 paths.

| Checkpoint vintage | Seed | Calibration dates | Usable calibration inputs | Forecast source |
| --- | --- | --- | --- | --- |
| 2023h2 | 17 | 113 | 1782 | re-decoded cached tokens |
| 2023h2 | 29 | 113 | 1782 | new generated tokens |
| 2023h2 | 43 | 113 | 1781 | new generated tokens |
| 2024h2 | 17 | 112 | 1748 | re-decoded cached tokens |
| 2024h2 | 29 | 112 | 1749 | new generated tokens |
| 2024h2 | 43 | 112 | 1747 | new generated tokens |

The neural networks were not retrained. Two archived validation token sets were verified and decoded with the fixed adapted decoder; four new validation token sets were generated. All six evaluation forecast sets came from the completed training-refresh experiment. Token generation uses 64 paths, temperature 1, full token distribution, batch size four and generation seed 17 plus batch offset.

## Interpretation limits

These are previously examined historical windows. H1 also served for checkpoint selection, so the calibration data are not independent of model selection. Official tokenizer pretraining coverage remains unresolved. The empirical method has no formal future coverage guarantee.

Confidence intervals use 2,000 circular block resamples of 10 dates and describe uncertainty within these historical windows and three trained seeds. They are not an all-subgroup acceptance veto. Width, misses, effect size, consistency and local regressions are considered together.

## Verification and artifacts

Independent formulas reproduced 197,070 raw quantile records and 394,140 raw/calibrated score records, all 36 date-weighted factors, date-equal aggregates and 640 paired estimates and confidence intervals. All six calibration token/price first-batch replays matched exactly. Source hashes, chunk assembly, chronological boundaries, factor-to-checkpoint binding and exact median preservation passed. The matched end-to-end point errors reproduce the prior verified research results.

Nine focused regression tests passed. No daily strategy model was replaced, and the sealed holdout beginning 2025-08-07 was not opened.

Artifacts are in artifacts/token-refreshed-calibration-20260915-v1. The protocol, frozen source, source manifest, two checkpoint-specific fit files, calibration paths, evaluation scores, common-cohort comparisons, verification receipt and final manifest are retained. research-profile.json binds the resulting checkpoint, decoder, sampler and interval-factor files for research reuse.

## Next bounded step

Keep this verified combined research profile fixed and evaluate its transfer to a later development period, without retuning on that evaluation. Retain the current calibration gain; investigate target-specific over-widening as a separate bounded comparison rather than discarding the working pipeline. The sealed holdout should remain reserved for a separately frozen final comparison.

## Fitted factors

| Checkpoint vintage | Seed | Days | Target | Scale | Dates | Rows |
| --- | --- | --- | --- | --- | --- | --- |
| 2023h2 | 17 | 2 | maximum | 1.2483 | 113 | 1782 |
| 2023h2 | 17 | 2 | minimum | 1.1935 | 113 | 1782 |
| 2023h2 | 17 | 2 | range | 1.3262 | 113 | 1782 |
| 2023h2 | 17 | 5 | maximum | 1.2261 | 113 | 1739 |
| 2023h2 | 17 | 5 | minimum | 1.1361 | 113 | 1739 |
| 2023h2 | 17 | 5 | range | 1.1808 | 113 | 1739 |
| 2023h2 | 29 | 2 | maximum | 1.2547 | 113 | 1782 |
| 2023h2 | 29 | 2 | minimum | 1.1760 | 113 | 1782 |
| 2023h2 | 29 | 2 | range | 1.2899 | 113 | 1782 |
| 2023h2 | 29 | 5 | maximum | 1.2044 | 113 | 1738 |
| 2023h2 | 29 | 5 | minimum | 1.1166 | 113 | 1738 |
| 2023h2 | 29 | 5 | range | 1.1873 | 113 | 1738 |
| 2023h2 | 43 | 2 | maximum | 1.3104 | 113 | 1781 |
| 2023h2 | 43 | 2 | minimum | 1.2708 | 113 | 1781 |
| 2023h2 | 43 | 2 | range | 1.3487 | 113 | 1781 |
| 2023h2 | 43 | 5 | maximum | 1.3075 | 113 | 1738 |
| 2023h2 | 43 | 5 | minimum | 1.2491 | 113 | 1738 |
| 2023h2 | 43 | 5 | range | 1.2392 | 113 | 1738 |
| 2024h2 | 17 | 2 | maximum | 1.2567 | 112 | 1748 |
| 2024h2 | 17 | 2 | minimum | 1.2049 | 112 | 1748 |
| 2024h2 | 17 | 2 | range | 1.3089 | 112 | 1748 |
| 2024h2 | 17 | 5 | maximum | 1.1790 | 112 | 1695 |
| 2024h2 | 17 | 5 | minimum | 1.1933 | 112 | 1695 |
| 2024h2 | 17 | 5 | range | 1.1454 | 112 | 1695 |
| 2024h2 | 29 | 2 | maximum | 1.2757 | 112 | 1749 |
| 2024h2 | 29 | 2 | minimum | 1.2733 | 112 | 1749 |
| 2024h2 | 29 | 2 | range | 1.3232 | 112 | 1749 |
| 2024h2 | 29 | 5 | maximum | 1.2173 | 112 | 1694 |
| 2024h2 | 29 | 5 | minimum | 1.2895 | 112 | 1694 |
| 2024h2 | 29 | 5 | range | 1.1811 | 112 | 1694 |
| 2024h2 | 43 | 2 | maximum | 1.2924 | 112 | 1747 |
| 2024h2 | 43 | 2 | minimum | 1.3050 | 112 | 1747 |
| 2024h2 | 43 | 2 | range | 1.3250 | 112 | 1747 |
| 2024h2 | 43 | 5 | maximum | 1.2511 | 112 | 1690 |
| 2024h2 | 43 | 5 | minimum | 1.3402 | 112 | 1690 |
| 2024h2 | 43 | 5 | range | 1.1838 | 112 | 1690 |

Related: [training refresh comparison](token-training-refresh-20260915.md), [calibration decision](token-calibration-decision-update-20260915.md), [previous calibration measurements](token-interval-calibration-20260915.md).
