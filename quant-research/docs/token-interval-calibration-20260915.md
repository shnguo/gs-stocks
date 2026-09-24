# Token forecast interval calibration

Run date: 2026-09-15.

## Outcome

The fixed interval calibration does not pass the two-day acceptance conditions across both evaluation windows. Five-day conditions do not pass. Model weights, generated prices and median forecasts are unchanged.

## Experiment

The preceding decoder experiment improved two-day distribution forecasts, while nominal 80% interval coverage remained below 70% in the later window. This experiment fits interval widths to observed validation errors and checks whether the improvement survives later dates.

- Fit on 2023 H1 only: 16 stock/date inputs per signal date, all three fixed predictor seeds, and the previously selected price-consistency decoder.
- Evaluate on 2023 H2 using retained raw paths and on 2024 H2 using newly generated paths, with 32 inputs per date and predictor.
- Keep 60 history bars, five generated days, 64 sampled paths, temperature 1.0 and top-p 1.0. The two-day metric scores the first two forecast days.
- Both interval variants use exactly the same adapted-decoder paths and usable inputs: known full-horizon labels and at least 16 legal draws. Earlier original/adapted comparisons used their intersection of usable inputs, so their aggregate numbers need not match this raw control.

The three predictors were fitted on 2016–2021 and selected on 2022 H1. No neural weights are refitted here. Calibration is a separate statistical transformation of interval endpoints, not a new prediction head or a correction to generated OHLC bars.

## Calibration method

For each predictor seed, horizon and target, compute how far each validation outcome lies from the predicted median, divided by the corresponding original half-width. Give each signal date equal total weight and take the empirical 80th percentile of these scores, with a minimum factor of one. This yields 18 fixed widening factors. A numerical half-width floor of 0.000001 percentage points handles degenerate intervals.

Apply each factor to the original lower and upper half-widths. Keep the median exactly fixed. Intersect the resulting intervals with the known mathematical support: nonnegative range and returns no lower than -100%. This intersection cannot remove a valid observed target; it is not a repair to the generated prices.

Illustration only: an original price interval of 100–104 with median 102 becomes 99.5–104.5 under a fitted factor of 1.25. The point forecast remains 102; the interval expresses greater uncertainty.

The procedure targets total interval coverage. It does not guarantee 10% errors in each tail, so lower and upper misses are reported separately. It does not define a newly calibrated full distribution, and no improved CRPS is claimed. Serial dependence, shared market shocks and retrospective choices prevent an iid or conformal coverage guarantee.

Acceptance requires 75–85% coverage for every target, lower mean interval score with its date-block confidence interval at or below zero, non-worsening interval score in all predictor seeds, and identical median MAE, separately in both windows. Two days is primary; five days has a separate secondary decision. Wider intervals incur a width penalty in the interval score.

## Evaluation size and eligibility

| Window | Days | Signal dates | Inputs / predictor | Mean usable / predictor | Usable fraction | Legal draws |
| --- | --- | --- | --- | --- | --- | --- |
| calibration2023h1 | 2 | 113 | 1808 | 1781.3 | 98.53% | 97.06% |
| calibration2023h1 | 5 | 113 | 1808 | 1735.7 | 96.00% | 92.78% |
| evaluation2023h2 | 2 | 119 | 3808 | 3746.7 | 98.39% | 96.05% |
| evaluation2023h2 | 5 | 119 | 3808 | 3685.3 | 96.78% | 91.57% |
| evaluation2024h2 | 2 | 120 | 3840 | 3796.0 | 98.85% | 95.91% |
| evaluation2024h2 | 5 | 120 | 3840 | 3710.0 | 96.61% | 90.87% |

Calibration uses fewer stocks per date to bound its inference cost while retaining every signal date. All retained rows contribute to the date-weighted fit; evaluation keeps the previously used 32-input-per-date design.

## Frozen factors

| Seed | Days | Target | Width factor | Fit rows | Fit dates | Degenerate half-width rows |
| --- | --- | --- | --- | --- | --- | --- |
| 17 | 2 | maximum | 1.3101 | 1780 | 113 | 0 |
| 17 | 2 | minimum | 1.2579 | 1780 | 113 | 0 |
| 17 | 2 | range | 1.3260 | 1780 | 113 | 0 |
| 17 | 5 | maximum | 1.2367 | 1734 | 113 | 0 |
| 17 | 5 | minimum | 1.1824 | 1734 | 113 | 0 |
| 17 | 5 | range | 1.1707 | 1734 | 113 | 0 |
| 29 | 2 | maximum | 1.2884 | 1782 | 113 | 0 |
| 29 | 2 | minimum | 1.2147 | 1782 | 113 | 0 |
| 29 | 2 | range | 1.3178 | 1782 | 113 | 0 |
| 29 | 5 | maximum | 1.2266 | 1738 | 113 | 0 |
| 29 | 5 | minimum | 1.1672 | 1738 | 113 | 0 |
| 29 | 5 | range | 1.1878 | 1738 | 113 | 0 |
| 43 | 2 | maximum | 1.2245 | 1782 | 113 | 0 |
| 43 | 2 | minimum | 1.1984 | 1782 | 113 | 0 |
| 43 | 2 | range | 1.2850 | 1782 | 113 | 0 |
| 43 | 5 | maximum | 1.2040 | 1735 | 113 | 0 |
| 43 | 5 | minimum | 1.1073 | 1735 | 113 | 0 |
| 43 | 5 | range | 1.2001 | 1735 | 113 | 0 |

## Evaluation averages

| Window | Days | 80% coverage | Width change | Interval score | Score change | 95% difference interval |
| --- | --- | --- | --- | --- | --- | --- |
| evaluation2023h2 | 2 | 67.36% → 77.76% | +26.99% | 10.7263 → 10.5236 | -1.89% | [-0.3809, -0.0533] |
| evaluation2023h2 | 5 | 69.75% → 78.07% | +18.95% | 19.0391 → 18.8173 | -1.17% | [-0.5062, -0.0025] |
| evaluation2024h2 | 2 | 68.15% → 78.37% | +26.98% | 17.7285 → 17.4998 | -1.29% | [-0.8650, +0.2569] |
| evaluation2024h2 | 5 | 69.72% → 77.38% | +18.96% | 33.0661 → 32.6628 | -1.22% | [-1.2788, +0.2841] |

Lower interval score is better. Metrics average within each signal date, then equally over dates, targets and predictor seeds. Price widths and errors are in percentage points of signal-close price.

## High, low and range separately

| Window | Days | Target | Coverage | Interval score change | Lower / upper miss |
| --- | --- | --- | --- | --- | --- |
| evaluation2023h2 | 2 | maximum | 67.39% → 78.19% | -1.85% | 10.23% / 11.57% |
| evaluation2023h2 | 2 | minimum | 69.07% → 77.53% | -0.93% | 10.89% / 11.58% |
| evaluation2023h2 | 2 | range | 65.62% → 77.56% | -2.66% | 11.89% / 10.55% |
| evaluation2023h2 | 5 | maximum | 69.26% → 78.77% | -1.51% | 9.35% / 11.88% |
| evaluation2023h2 | 5 | minimum | 70.10% → 76.98% | -0.70% | 11.27% / 11.75% |
| evaluation2023h2 | 5 | range | 69.91% → 78.45% | -1.04% | 10.40% / 11.15% |
| evaluation2024h2 | 2 | maximum | 67.59% → 77.43% | -2.21% | 10.12% / 12.45% |
| evaluation2024h2 | 2 | minimum | 69.38% → 78.45% | +0.65% | 7.85% / 13.70% |
| evaluation2024h2 | 2 | range | 67.48% → 79.25% | -1.49% | 11.58% / 9.16% |
| evaluation2024h2 | 5 | maximum | 67.92% → 76.43% | -1.93% | 8.51% / 15.05% |
| evaluation2024h2 | 5 | minimum | 70.40% → 77.24% | +0.35% | 6.76% / 16.01% |
| evaluation2024h2 | 5 | range | 70.85% → 78.46% | -1.12% | 10.08% / 11.46% |

## Check conditions

| Days | Window | Every target 75–85% | Score difference interval ≤ 0 | Every seed non-worse | Identical median MAE |
| --- | --- | --- | --- | --- | --- |
| 2 | evaluation2023h2 | True | True | True | True |
| 2 | evaluation2024h2 | True | False | True | True |
| 5 | evaluation2023h2 | True | True | True | True |
| 5 | evaluation2024h2 | True | False | True | True |

## Interpretation and next direction

Coverage improves for every target in both windows. The tradeoff is approximately 27% wider two-day intervals and 19% wider five-day intervals. Mean interval scores improve by roughly 1–2%, but their 2024 H2 difference intervals include zero. The low-price interval score worsens slightly in that window: 0.65% at two days and 0.35% at five days. Better coverage alone therefore does not satisfy the acceptance gate.

Retain the fitted factors and all evidence as a research candidate. They are tied to these exact predictor checkpoints, decoder and sampling settings, and should not be applied automatically to a different model. Neither the daily model nor its defaults are changed.

The next controlled question is whether refreshing the training history improves maximum-high and minimum-low point forecasts. The predictors were deliberately held at their 2016–2021 fits to isolate decoder and calibration changes. A matched comparison with refreshed fits can test data freshness; this experiment does not establish staleness as the cause of the remaining errors.

## Verification and limitations

- Nine focused tests passed, including date-weight invariance, median preservation, degenerate intervals, support intersection and the width penalty.
- Raw quantiles and actual high/low/range targets are independently reconstructed from retained sampled paths. Fitted scales satisfy independently calculated date-weighted quantile conditions.
- Interval endpoints, all six metric fields, date means, 256 comparisons and confidence intervals are independently checked. Median forecasts and their MAE match exactly before and after calibration.
- All fits use only the 2023 H1 forecast errors. Coefficients are frozen before applying them to either evaluation period. New generation replays its first batch exactly; original checkpoints reload with identical CPU logits.
- Confidence intervals resample 2,000 circular blocks of ten trading dates. They are conditional on the fixed models, selected stocks and fitted calibration factors; they do not include uncertainty from refitting the calibration.
- These are retrospective development windows that have appeared in earlier research. Official tokenizer pretraining coverage remains unresolved. The sealed holdout starting 2025-08-07 is unused.
- This tests interval calibration, not improved high/low point estimates, a retrained two-day model, or the official Kronos Base predictor.

## Artifacts

- [Frozen protocol](/Users/guo/Documents/stocks/quant-research/artifacts/token-interval-calibration-20260915-v1/protocol.json)
- [Fitted factors](/Users/guo/Documents/stocks/quant-research/artifacts/token-interval-calibration-20260915-v1/fit.json)
- [Independent verification](/Users/guo/Documents/stocks/quant-research/artifacts/token-interval-calibration-20260915-v1/independent-verification.json)
- [Research decision](/Users/guo/Documents/stocks/quant-research/artifacts/token-interval-calibration-20260915-v1/research-decision.json)
- [Detailed comparisons](/Users/guo/Documents/stocks/quant-research/artifacts/token-interval-calibration-20260915-v1/results/paired.csv)
