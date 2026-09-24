# Frozen self-built model versus Kronos Base and historical volatility

Date: 2026-09-15 (America/Los_Angeles). Completed retrospective external benchmark. No training, calibration fitting or evaluation-based retuning.

## Conclusion

Retain the refreshed self-built model and its checkpoint-specific calibration as a useful research profile. On this matched development cohort, raw mean high/low MAE is 8.07% lower than Kronos Base at two days and 10.25% lower at five days; raw CRPS is 9.28% and 11.46% lower. All six training-seed/horizon high-low averages improve in both metrics. Across individual targets, raw CRPS and interval score improve in all 18 seed/target/horizon comparisons against Base, while MAE improves in 15 of 18. The mean high/low MAE and CRPS difference intervals are below zero at both horizons. These results support a forecasting advantage over this frozen Kronos Base configuration in the tested period, not universal superiority or a causal claim about model size.

The historical-volatility baseline remains a stronger high/low point-forecast benchmark: self-built high/low MAE is 5.01% higher at two days and 6.47% higher at five days, and CRPS is 3.45% and 4.05% higher. These aggregate difference intervals cross zero, but all three self-built seeds show the same point-estimate disadvantage; this is an optimization target rather than proof that the methods are equivalent. Conversely, self-built range MAE is 11.84% lower at two days and 8.78% lower at five days than the historical baseline, with CRPS 11.31% and 9.43% lower; those mean difference intervals are below zero. Retain these target-specific strengths and the previously demonstrated improvements. This is a previously examined development block with unresolved pretraining-date coverage, not an untouched test or evidence of trading profitability.

## Matched scope and comparison

The source cohort contains 2,432 distinct stock-date inputs on 76 signal dates, 2025-04-01 through 2025-07-22, with labels ending no later than 2025-07-29. Both neural pipelines use the same 60 historical OHLCVA bars, signal-anchored price adjustment, normalization statistics and calendar; the historical baseline uses those same source prices. The first 32 members of the outcome-independent daily pool are retained before inspecting outcomes or forecast quality.

The self-built profile contains three frozen 3,720,448 parameter predictors trained through 2023 and selected on 2024 H1, the adapted decoder, and checkpoint-specific 2024 H1 interval factors. Existing 64-path forecasts are reused exactly. Self-built means below average the three models’ scores; they are not forecasts from a 192-path ensemble, and the three seeds are not treated as independent market-date observations.

Kronos Base uses its official tokenizer/decoder and frozen 102,310,592 parameter predictor, revision 2b554741eca47781b64468546e77fef3e85130e6. Its official native inference generates 64 independent five-day paths per input, temperature 1, top-p 1, top-k 0, clip 5 and batches of four. The averaging axis has size one; no paths are averaged away. The native decoder differs from the adapted self-built decoder: this is a full-pipeline comparison, not a capacity or architecture ablation. All 291,840 coarse/fine historical token values match the frozen self-built input prefixes.

The historical-volatility baseline samples 64 contiguous five-day blocks from 59 previous-close OHLC log-return vectors in each 60-day history. The historical mean close drift is subtracted from all four returns, and sampled returns are compounded from the signal close. A hash of seed 17 and stock/date fixes the block starts. Volume and amount are unused zero placeholders; this baseline evaluates prices only. Each external baseline has one frozen Monte Carlo realization, not three independently generated replicas.

## High/low accuracy and interval quality

Maximum future high and minimum future low are scored separately, then equally averaged. Rows are equally weighted within each date, followed by equal date weights. MAE is the absolute error of the sample median. CRPS measures the full sampled distribution; both are lower-is-better. Prices, errors, widths and interval scores are in percentage points of the signal-day close. Coverage is for separate nominal 80% intervals for each target, not a joint executable trading band.

| Days | Model | High/low MAE | Raw CRPS | 80% coverage | Interval width | Interval score |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | Self-built, raw | 2.7250 | 2.0643 | 72.12% | 7.1364 | 14.8363 |
| 2 | Self-built, calibrated | 2.7250 | — | 81.65% | 9.0547 | 14.7904 |
| 2 | Kronos Base, raw | 2.9642 | 2.2756 | 60.43% | 6.6371 | 16.7723 |
| 2 | Historical volatility | 2.5949 | 1.9956 | 77.42% | 7.9958 | 14.4907 |
| 5 | Self-built, raw | 4.2554 | 3.1981 | 73.18% | 11.3732 | 22.5934 |
| 5 | Self-built, calibrated | 4.2554 | — | 83.01% | 14.1347 | 22.5772 |
| 5 | Kronos Base, raw | 4.7415 | 3.6119 | 55.20% | 9.3254 | 26.3026 |
| 5 | Historical volatility | 3.9970 | 3.0737 | 76.69% | 13.1985 | 22.7183 |

Calibration preserves the self-built medians exactly. Its secondary interval comparison has unequal calibration treatment: neither external baseline is fitted to the self-built calibration period. A better calibrated interval score alone would not demonstrate a better predictor. Calibrated CRPS is omitted because only interval boundaries, not the entire predictive distribution, were calibrated.

## Raw self-built changes relative to each external baseline

Negative relative changes favor the self-built model. Confidence intervals contain absolute differences, in percentage points. Paired 95% intervals use 2,000 circular resamples of 10 signal dates; they are conditional on this development block, these trained seeds and fixed sampled paths. Target/seed comparisons are descriptive and overlap.

| Days | Baseline | MAE change | 95% CI, MAE difference | CRPS change | 95% CI, CRPS difference | Raw interval score change |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | Kronos Base, raw | -8.07% | [-0.3816, -0.0609] | -9.28% | [-0.3139, -0.0904] | -11.54% |
| 2 | Historical volatility | +5.01% | [-0.0080, +0.3061] | +3.45% | [-0.0372, +0.2039] | +2.38% |
| 5 | Kronos Base, raw | -10.25% | [-0.8828, -0.0242] | -11.46% | [-0.6801, -0.0947] | -14.10% |
| 5 | Historical volatility | +6.47% | [-0.0696, +0.6877] | +4.05% | [-0.1003, +0.4159] | -0.55% |

## Individual targets

| Days | Target | Model | MAE | Raw CRPS | 80% coverage | Interval score |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | maximum | Self-built, raw | 3.1310 | 2.3673 | 71.02% | 16.9946 |
| 2 | maximum | Self-built, calibrated | 3.1310 | — | 80.68% | 16.7886 |
| 2 | maximum | Kronos Base, raw | 3.1693 | 2.4514 | 60.25% | 18.2927 |
| 2 | maximum | Historical volatility | 3.0681 | 2.3483 | 78.37% | 16.8752 |
| 2 | minimum | Self-built, raw | 2.3189 | 1.7613 | 73.22% | 12.6780 |
| 2 | minimum | Self-built, calibrated | 2.3189 | — | 82.63% | 12.7921 |
| 2 | minimum | Kronos Base, raw | 2.7592 | 2.0998 | 60.61% | 15.2518 |
| 2 | minimum | Historical volatility | 2.1218 | 1.6429 | 76.46% | 12.1062 |
| 2 | range | Self-built, raw | 2.7886 | 2.1463 | 64.20% | 15.8015 |
| 2 | range | Self-built, calibrated | 2.7886 | — | 77.33% | 15.3206 |
| 2 | range | Kronos Base, raw | 3.3576 | 2.5229 | 52.21% | 17.8487 |
| 2 | range | Historical volatility | 3.1631 | 2.4201 | 66.27% | 17.7153 |
| 5 | maximum | Self-built, raw | 5.0392 | 3.8142 | 73.11% | 27.2075 |
| 5 | maximum | Self-built, calibrated | 5.0392 | — | 81.28% | 26.9660 |
| 5 | maximum | Kronos Base, raw | 5.0070 | 3.9305 | 56.91% | 30.1493 |
| 5 | maximum | Historical volatility | 4.9258 | 3.7854 | 77.49% | 27.4849 |
| 5 | minimum | Self-built, raw | 3.4717 | 2.5821 | 73.26% | 17.9793 |
| 5 | minimum | Self-built, calibrated | 3.4717 | — | 84.75% | 18.1885 |
| 5 | minimum | Kronos Base, raw | 4.4759 | 3.2933 | 53.48% | 22.4559 |
| 5 | minimum | Historical volatility | 3.0681 | 2.3620 | 75.89% | 17.9516 |
| 5 | range | Self-built, raw | 4.8654 | 3.6826 | 66.36% | 26.2519 |
| 5 | range | Self-built, calibrated | 4.8654 | — | 75.13% | 25.6929 |
| 5 | range | Kronos Base, raw | 5.6688 | 4.2831 | 46.02% | 30.7438 |
| 5 | range | Historical volatility | 5.3335 | 4.0659 | 65.38% | 28.8748 |

Against Kronos Base, five-day maximum-high MAE is 0.64% higher for the mean self-built result; its difference interval crosses zero. The stronger gains are in minimum-low MAE, down 15.96% at two days and 22.44% at five days, and range MAE, down 16.95% and 14.17%. Against historical volatility, maximum-high MAE is 2.05% / 2.30% higher and minimum-low MAE is 9.29% / 13.15% higher at two/five days. All six range seed/horizon comparisons favor the self-built model in both MAE and CRPS. The raw self-built interval score is 2.38% worse than historical volatility at two days and 0.55% better at five days, with uncertain differences. Calibration raises mean high/low coverage to 81.65% / 83.01%, but the minimum-low five-day interval reaches 84.75% coverage and range coverage is only 75.13%. Calibration therefore retains its documented coverage benefit without resolving every target's error or interval-width tradeoff.

## Training-seed consistency

| Self-built seed | Days | Baseline | High/low MAE change | High/low CRPS change |
| --- | --- | --- | --- | --- |
| 17 | 2 | Kronos Base, raw | -8.93% | -9.85% |
| 29 | 2 | Kronos Base, raw | -8.62% | -10.22% |
| 43 | 2 | Kronos Base, raw | -6.67% | -7.79% |
| 17 | 2 | Historical volatility | +4.03% | +2.80% |
| 29 | 2 | Historical volatility | +4.39% | +2.38% |
| 43 | 2 | Historical volatility | +6.61% | +5.15% |
| 17 | 5 | Kronos Base, raw | -11.79% | -12.68% |
| 29 | 5 | Kronos Base, raw | -11.31% | -12.67% |
| 43 | 5 | Kronos Base, raw | -7.64% | -9.02% |
| 17 | 5 | Historical volatility | +4.63% | +2.61% |
| 29 | 5 | Historical volatility | +5.20% | +2.62% |
| 43 | 5 | Historical volatility | +9.56% | +6.91% |

## Availability and common cohort

Each horizon requires all future labels to be known and at least 16 legal paths out of 64. Illegal generated prices are excluded, not repaired. Every reported model comparison uses the intersection of all three self-built predictors and both external baselines; raw/calibrated variants share that intersection. The tables below disclose the excluded coverage, so conditional forecast scores are not presented as whole-population results. This common cohort is smaller than the previous old-versus-refreshed comparison, so absolute self-built scores differ even though its paths and weights are unchanged.

| Variant | Days | Input rows | Known-label rows | Usable rows | Legal paths |
| --- | --- | --- | --- | --- | --- |
| kronos | 2 | 2432 | 2389 | 2378 | 87.00% |
| kronos | 5 | 2432 | 2324 | 2268 | 69.12% |
| historical | 2 | 2432 | 2389 | 2389 | 100.00% |
| historical | 5 | 2432 | 2324 | 2324 | 100.00% |
| ours_17 | 2 | 2432 | 2389 | 2384 | 96.94% |
| ours_17 | 5 | 2432 | 2324 | 2304 | 92.39% |
| ours_29 | 2 | 2432 | 2389 | 2384 | 96.88% |
| ours_29 | 5 | 2432 | 2324 | 2303 | 92.23% |
| ours_43 | 2 | 2432 | 2389 | 2381 | 96.69% |
| ours_43 | 5 | 2432 | 2324 | 2302 | 91.95% |

| Days | Common stock-date rows | Common signal dates |
| --- | --- | --- |
| 2 | 2372 | 76 |
| 5 | 2253 | 76 |

## Evidence and interpretation limits

This already-examined development period is not an untouched final test. The exact pretraining dates of the official tokenizer and Kronos predictor are not independently established, so temporal leakage cannot be excluded for their pretraining. Differing training histories, decoder weights and calibration prevent isolating an architecture effect. This study evaluates forecasting errors; it does not test fills, costs, attainable extrema or trading profitability.

The previous improvements against the older self-built profile remain valid evidence within their scope. External-baseline strengths or local regressions qualify their practical value; they do not erase those earlier measurements.

## Verification

Independent calculations reproduced 112,425 row-target scores, 111,000 common-cohort records, all date aggregates and 576 paired estimates and intervals. CRPS was independently recalculated using pairwise path distances. All 155,648 historical paths were reconstructed by sequential compounding. Source histories/statistics, calendar alignment, cohort membership and path validity were verified.

Kronos generation, path assembly and reload took 56.2 minutes on the local MPS device. Kronos weights were strictly loaded and SHA256-checked, and its first 256 native paths matched exactly after model reload. All three reused self-built runs and their calibration factors were independently rechecked. Fourteen focused regression tests passed. Prior weights, calibration factors and artifacts remain unchanged. The daily strategy model was not replaced, and the sealed holdout beginning 2025-08-07 was not opened.

## Research decision and next step

Keep the current predictor, decoder, sampler and calibration profile fixed. The next bounded diagnostic should use these saved paths to decompose high/low errors into the predicted interval centre and its width, comparing the self-built models with the historical-volatility baseline at both horizons and across the three seeds. Test the hypothesis that range size is learned better than directional price movement; do not assume it from the aggregate results. This diagnostic requires no new neural inference, training or calibration fit. Use its result to choose a subsequent learning-objective or data experiment. Preserve the current improvements, avoid architecture expansion before locating the remaining gap, and reserve the sealed holdout for a separately frozen final comparison.

Artifacts: artifacts/token-external-benchmark-20260915-v1 retains the protocol frozen before generation, source hashes, complete matched histories, raw Kronos and historical paths, historical block starts, scores, cohort records, comparison tables, independent verification, runtime and final manifest. The metric source is results/paired.csv and results/summary.csv; the source selection and transformations are reproducible with the frozen scripts in code/scripts.

Related: [later-period transfer](token-profile-transfer-20260915.md), [checkpoint calibration](token-refreshed-calibration-20260915.md), [earlier Kronos Base comparison](kronos-base-range-20260914.md).
