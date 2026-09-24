# Frozen-profile transfer to April–July 2025

Date: 2026-09-15 (America/Los_Angeles). Completed transfer research; no predictor retraining, calibration fitting or evaluation-based retuning.

## Conclusion

Retain the refreshed predictors and their checkpoint-specific calibration as the preferred research profile. On the later April–July 2025 development block, mean high/low point MAE improves 1.13% at two days and 2.02% at five days versus the old calibrated profile. Calibrated interval score improves 1.83% and 2.73%, and raw distribution CRPS improves 1.21% and 2.34%. All six target/horizon means improve in both point MAE and interval score. Across individual seed/target/horizon cases, point MAE improves in 13 of 18 and interval score in 17 of 18. These descriptive counts overlap and are not independent significance tests. This later-window evidence supports retaining the earlier gains without claiming uniform superiority.

The mean high/low MAE and interval-score difference intervals cross zero at both horizons, so their exact size remains uncertain in this 76-date block. The largest point gains come from seed 17; seed 43 has modest point-error regressions. The development block was already used in earlier experiments, and tokenizer pretraining-date coverage is unresolved. This is retrospective transfer evidence, not an untouched final test or a profitability result. No production or daily-strategy model is promoted.

## Comparison and frozen sources

Evaluation uses 76 signal dates from 2025-04-01 through 2025-07-22, with future labels ending by 2025-07-29. The same 2,432 distinct stock-date inputs are shared by all six runs, selected as the first 32 members of the outcome-independent daily pool. The full five-day label window stays inside the April–July development block.

The refreshed profile uses the three 3,720,448 parameter predictors trained through 2023, selected on 2024 H1, and their checkpoint-specific interval factors fitted on 2024 H1. The control uses three predictors trained through 2021, selected on 2022 H1, with factors fitted on 2023 H1. All weights and factors are the exact previously verified files.

Both vintages use the same adapted decoder, 60 history bars, equal output-loss training recipe, 64 generated paths, temperature 1, full token distribution, batch size four and generation seed 17 plus batch offset. Each comparison uses common usable stock-date/horizon inputs between vintages within each seed. Calibration leaves eligibility and the point median unchanged.

## Does the combined profile transfer?

Maximum-high and minimum-low errors are equally averaged across targets, dates and the three seeds. MAE, interval scores and confidence-interval endpoints are in percentage points of the signal-date close. Negative relative changes mean lower error. CRPS scores only the raw sampled distributions; it is not a claim that interval calibration improves the full distribution.

| Days | High/low MAE, old → refreshed | MAE change | Calibrated interval score change | 95% CI of score difference | Raw CRPS change |
| --- | --- | --- | --- | --- | --- |
| 2 | 2.7716 → 2.7403 | -1.13% | -1.83% | [-0.6007, +0.1139] | -1.21% |
| 5 | 4.3968 → 4.3082 | -2.02% | -2.73% | [-1.2882, +0.2009] | -2.34% |

## Does the old calibration still help the refreshed predictors?

This comparison holds predictor weights, generated paths, point medians and the common cohort fixed. It applies the frozen 2024 H1 interval factors without updating them on 2025 outcomes. The nominal coverage target is 80%. Lower interval score is better because it penalizes both width and misses.

| Days | Raw coverage | Calibrated coverage | Width change | Interval score change | 95% CI of score difference |
| --- | --- | --- | --- | --- | --- |
| 2 | 72.01% | 81.55% | +26.88% | -0.38% | [-0.7738, +0.4098] |
| 5 | 72.98% | 82.81% | +24.28% | -0.16% | [-1.0986, +0.6968] |

## Individual targets

| Days | Target | MAE change versus old | Interval score change versus old calibrated | Old → refreshed calibrated coverage | Calibration-only score change |
| --- | --- | --- | --- | --- | --- |
| 2 | maximum | -0.61% | -0.64% | 81.90% → 80.59% | -1.25% |
| 2 | minimum | -1.83% | -3.34% | 82.07% → 82.52% | +0.78% |
| 2 | range | -3.12% | -3.38% | 76.34% → 77.42% | -2.96% |
| 5 | maximum | -0.50% | -1.76% | 82.62% → 81.02% | -0.95% |
| 5 | minimum | -4.14% | -4.15% | 79.35% → 84.60% | +1.05% |
| 5 | range | -6.11% | -3.76% | 73.68% → 75.06% | -2.10% |

Calibration alone raises mean high/low coverage from 72.01% to 81.55% at two days and from 72.98% to 82.81% at five days. It widens intervals by 26.88% and 24.28%, while improving their mean score only 0.38% and 0.16%; point medians are exactly unchanged. Minimum-low interval scores worsen 0.78% and 1.05% under calibration, with coverage reaching 82.52% and 84.60%. Range coverage remains below 80% at 77.42% and 75.06%, despite improved scores. For the complete refreshed profile, seed 43 high/low MAE worsens 0.94% at two days and 1.02% at five days; its two-day interval score worsens 0.05%. Seed 29 five-day calibration-only high/low score worsens 0.65%. These cases qualify the broad improvements and motivate targeted follow-up rather than discarding the profile.

## Seed consistency

| Seed | Days | High/low MAE change | Calibrated interval score change | Refreshed calibrated coverage |
| --- | --- | --- | --- | --- |
| 17 | 2 | -3.94% | -3.42% | 80.17% |
| 29 | 2 | -0.30% | -2.08% | 82.12% |
| 43 | 2 | +0.94% | +0.05% | 82.37% |
| 17 | 5 | -6.42% | -5.80% | 81.00% |
| 29 | 5 | -0.50% | -1.36% | 83.67% |
| 43 | 5 | +1.02% | -0.95% | 83.76% |

## Usable inputs and valid paths

Counts below sum over three seeds, so repeated stock-date inputs are counted once per seed. A target horizon requires complete known labels and at least 16 legal draws out of 64. Scores use the common cohort even if the individual-vintage usable counts differ.

| Vintage | Days | Inputs across seeds | Known-label inputs | Usable inputs | Legal paths |
| --- | --- | --- | --- | --- | --- |
| old | 2 | 7296 | 7167 | 7146 | 96.81% |
| old | 5 | 7296 | 6972 | 6910 | 92.09% |
| refreshed | 2 | 7296 | 7167 | 7149 | 96.83% |
| refreshed | 5 | 7296 | 6972 | 6909 | 92.19% |

## Interpretation

This tests transfer of the entire frozen research profile. It does not estimate the isolated causal effect of data recency: the vintages differ in training history, training exposure, checkpoint-selection period and calibration period. The 2025 development block was used by earlier model experiments and is not an untouched final test. Official tokenizer pretraining coverage remains unresolved.

The 95% confidence intervals use 2,000 circular block resamples of 10 dates. They describe uncertainty in this 76-date block conditional on the trained seeds, not all market regimes or all model-selection uncertainty. Subgroup comparisons are descriptive. Local regressions and uncertain estimates qualify earlier gains rather than automatically negating them.

## Verification

Independent formulas reproduced 168,684 raw/calibrated row-target score records, 168,432 common-cohort records, all date-equal aggregates, and 360 paired estimates and confidence intervals. Raw quantiles and pairwise CRPS were recalculated from the saved paths. All six token-and-price first-batch replays matched exactly. Factor/checkpoint hashes, path assembly, legal-path masks, chronology and exact median preservation passed.

Nine focused regression tests passed. The run retained 933,888 five-day sampled paths. Both historical calibration fits and all model weights stayed unchanged. The daily strategy model was not replaced and the sealed holdout beginning 2025-08-07 was not opened.

## Research decision and next step

Keep the verified predictor weights, decoder, sampling recipe and calibration factors fixed. The next bounded comparison should benchmark this current profile against frozen Kronos Base and a simple historical-volatility baseline on identical development dates, histories, labels and eligible cohorts, with equal path budgets where applicable. Report raw distribution metrics separately from calibrated intervals and disclose the pretraining-date limitation. This tests whether the accumulated improvements close the external-baseline gap before adding model complexity. Target-specific calibration over-widening remains a separate research issue; do not retune factors on this evaluation block. Reserve the sealed holdout for a separately frozen final comparison.

Artifacts: artifacts/token-profile-transfer-20260915-v1 contains frozen source, protocol, profiles, source hashes, preflight bounds, token/path chunks, raw quantiles, calibrated scores, matched cohorts, results/paired.csv, independent verification and final manifest. Prior research artifacts remain intact.

Related: [checkpoint-specific calibration](token-refreshed-calibration-20260915.md), [training refresh](token-training-refresh-20260915.md), [calibration decision](token-calibration-decision-update-20260915.md).
