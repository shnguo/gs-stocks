# Midpoint versus width: locating the remaining high/low forecast gap

Date: 2026-09-15 (America/Los_Angeles). Completed frozen-path diagnostic. No new inference, training, calibration or prediction-rule changes.

## Conclusion

The remaining high/low gap versus historical volatility is concentrated in placement of the price range. Using the midpoint and half-width implied by the existing endpoint medians, self-built midpoint MAE is 9.94% higher at two days and 12.90% higher at five days, while half-width MAE is 11.32% lower and 5.84% lower. All three seeds show both directions at both horizons. Mean midpoint-error difference intervals are above zero, and mean half-width-error intervals are below zero. Scoring midpoint distributions directly gives the same qualitative result, so this is not just an artifact of taking endpoint medians separately. Retain the verified range improvements and the advantage over Kronos Base.

This locates an observed forecasting weakness; it does not identify its training or data cause. The midpoint is the centre of future high and low, not the terminal close or pure trend direction. Signed drift, asymmetry and response to market regimes can all affect it. The 76-date development block has already been examined, pretrained date coverage is unresolved, and uncertainty is conditional on the frozen models and paths. No profitability, production promotion or general market-regime claim follows.

## Question and matched evidence

The preceding external benchmark found stronger high/low forecasts than Kronos Base, but weaker high/low forecasts than historical volatility despite better sampled-range accuracy. This diagnostic tests whether the remaining error concerns the position of the range or its size.

It reuses the same five sets of 64-path forecasts: three self-built training seeds, frozen Kronos Base and historical volatility. The exact shared cohort has 2,372 two-day and 2,253 five-day stock-date inputs across 76 signal dates, April 1 through July 22, 2025. Labels end by July 29. Every model uses complete known labels and at least 16 legal paths. Stocks are weighted equally within date, then dates equally; self-built means average the three seed scores, not their paths.

## What is being decomposed?

Let H and L be the forecast medians of the future highest high and lowest low. Midpoint C = (H + L)/2; half-width W = (H − L)/2. The realized targets use the actual future high and low. Prices are expressed as percentage-point returns from the signal close. This midpoint describes where the price range sits; it is not the last-day close or a direct measure of up/down trading accuracy.

Writing the midpoint error as eC and half-width error as eW gives exact identities: mean high/low absolute error = max(|eC|, |eW|); mean high/low squared error = eC² + eW². Absolute errors therefore cannot simply be added across the two components.

Illustration only: actual high 12 and low 8 give midpoint 10 and width 4. Forecast high 14 and low 10 also give width 4, but midpoint 12. The width is correct while both endpoints miss by 2.

The median of sampled ranges need not equal median(high) minus median(low); the same distinction applies to midpoint medians. The report checks both definitions, and the independently computed sampled-range scores match the preceding benchmark.

## Comparison with historical volatility

Lower MAE and CRPS are better. Negative relative changes favor the self-built model. All errors and confidence-interval endpoints below are in percentage points of signal close, except percentage changes.

| Days | Metric | Historical baseline | Self-built mean | Change | 95% CI of absolute difference |
| --- | --- | --- | --- | --- | --- |
| 2 | Midpoint from endpoint medians | 2.3010 | 2.5298 | +9.94% | [+0.0833, +0.4188] |
| 2 | Half-width from endpoint medians | 1.5304 | 1.3572 | -11.32% | [-0.2518, -0.1056] |
| 2 | Median sampled midpoint | 2.3033 | 2.5330 | +9.97% | [+0.0860, +0.4174] |
| 2 | Sampled midpoint CRPS | 1.7743 | 1.9118 | +7.75% | [+0.0239, +0.2842] |
| 2 | Median sampled full range | 3.1631 | 2.7886 | -11.84% | [-0.5241, -0.1791] |
| 2 | Sampled full-range CRPS | 2.4201 | 2.1463 | -11.31% | [-0.3756, -0.1412] |
| 2 | Mean high/low endpoint MAE | 2.5949 | 2.7250 | +5.01% | [-0.0080, +0.3061] |
| 5 | Midpoint from endpoint medians | 3.5038 | 3.9558 | +12.90% | [+0.1145, +0.9015] |
| 5 | Half-width from endpoint medians | 2.4899 | 2.3444 | -5.84% | [-0.2485, -0.0211] |
| 5 | Median sampled midpoint | 3.5136 | 3.9559 | +12.59% | [+0.1179, +0.8707] |
| 5 | Sampled midpoint CRPS | 2.7288 | 2.9584 | +8.41% | [-0.0031, +0.5328] |
| 5 | Median sampled full range | 5.3335 | 4.8654 | -8.78% | [-0.7460, -0.0856] |
| 5 | Sampled full-range CRPS | 4.0659 | 3.6826 | -9.43% | [-0.5936, -0.1000] |
| 5 | Mean high/low endpoint MAE | 3.9970 | 4.2554 | +6.47% | [-0.0696, +0.6877] |

## How the components account for the endpoint error gap

For absolute error, start with the historical forecast components and replace them with the self-built components in each of the two possible orders. Average each component’s incremental change across those orders. The two contributions add exactly to the observed endpoint-MAE difference, including component interaction. These are arithmetic contributions, not causal effects of training and not a selected hybrid forecasting rule.

| Days | Midpoint contribution | Half-width contribution | Net endpoint error gap | 95% CI, midpoint contribution | 95% CI, half-width contribution |
| --- | --- | --- | --- | --- | --- |
| 2 | +0.2210 | -0.0910 | +0.1301 | [+0.0942, +0.3880] | [-0.1230, -0.0524] |
| 5 | +0.3836 | -0.1251 | +0.2585 | [+0.0979, +0.7628] | [-0.1771, -0.0555] |

A second check uses the exact additive squared-error identity. Units here are squared percentage points; these values cannot be compared directly with the preceding MAE table.

| Days | Midpoint MSE difference | Half-width MSE difference | Endpoint MSE difference |
| --- | --- | --- | --- |
| 2 | +3.0100 | -1.5196 | +1.4904 |
| 5 | +8.1967 | -1.6446 | +6.5521 |

## Seed consistency

| Seed | Days | Midpoint MAE change | Half-width MAE change | Sampled-midpoint MAE change | Sampled-range MAE change |
| --- | --- | --- | --- | --- | --- |
| 17 | 2 | +8.91% | -11.86% | +9.01% | -12.56% |
| 29 | 2 | +8.94% | -11.43% | +8.85% | -11.64% |
| 43 | 2 | +11.98% | -10.67% | +12.07% | -11.32% |
| 17 | 5 | +11.07% | -6.55% | +10.91% | -10.20% |
| 29 | 5 | +11.04% | -6.51% | +10.86% | -8.67% |
| 43 | 5 | +16.58% | -4.47% | +15.99% | -7.45% |

The independently scored sampled-midpoint MAE is 9.97% / 12.59% worse than historical volatility at two/five days; sampled-midpoint CRPS is 7.75% / 8.41% worse. The five-day mean midpoint-CRPS difference interval crosses zero. All six seed/horizon midpoint-MAE difference intervals are above zero. Half-width point estimates improve in all six cases, although seed 43's five-day interval crosses zero; the same seed's five-day sampled-range MAE interval also crosses zero. Sampled full-range MAE retains its 11.84% / 8.78% mean improvement. At five days, the self-built midpoint is too low by 1.6382 percentage points on average, versus 1.0520 for the historical baseline. This period-specific bias is a diagnostic, not an offset to apply to future predictions.

## Signed error

Negative midpoint bias means the predicted range is too low on average. Negative width bias means the gap between endpoint medians is too narrow. These are averages over this period, not adjustments to apply to future forecasts.

| Model | Days | Midpoint bias | Full-width bias from endpoint medians |
| --- | --- | --- | --- |
| historical | 2 | -0.4800 | -1.3566 |
| historical | 5 | -1.0520 | -1.8988 |
| ours_mean | 2 | -0.6388 | -1.1364 |
| ours_mean | 5 | -1.6382 | -1.7704 |
| kronos | 2 | -1.0594 | +0.4276 |
| kronos | 5 | -2.3308 | +0.7731 |

## Interpretation and limits

The result locates the observed gap in price-level placement. It does not establish whether the cause is the token objective, training-period mismatch, momentum information, calibration, or another mechanism. A midpoint is affected by drift and high/low asymmetry, so this is not proof of a particular directional-signal failure.

The confidence intervals use 2,000 circular resamples of 10 dates and remain conditional on this previously examined development block, trained seeds and fixed sampled paths. Subgroup comparisons are descriptive and not adjusted for multiplicity. Pretrained tokenizer/predictor date coverage remains unresolved. The results do not establish attainable trades or profitability. Previous gains over the old self-built model and Kronos Base are retained.

## Verification

Independent raw-path formulas and pairwise CRPS reproduced all 23,125 model/stock-date/horizon component records. The shared cohort was independently reconstructed from label masks and all five models’ legal-path masks. Every endpoint MAE and sampled-range MAE/CRPS matched the previous benchmark. Both algebraic identities passed row by row. All 27,750 attribution rows, 224 paired estimates and 48 attributed estimates and their block intervals were independently verified.

Eight focused tests passed, including translation versus width errors, nonadditive component interactions, median noncommutativity and invalid endpoints. All source hashes match. Saved forecasts, model weights, calibration factors and the daily strategy model remain unchanged; the sealed holdout beginning August 7, 2025 remains unopened.

## Research decision and next experiment

The next bounded optimization should compare the existing token cross-entropy objective with a centre-focused auxiliary training objective while preserving token generation, model size, historical inputs, decoder and sampling. Keep the existing equal horizon weights initially to isolate the new objective. Select its strength on an earlier validation partition, then compare fixed candidates across the same three seeds and two/five-day horizons, reporting both midpoint improvements and any range tradeoffs. Any differentiable surrogate must first be checked against errors from actual generated paths. Do not choose weights on this already-examined 2025 block, add a hand-set price offset, or expand to MoE as a response to this diagnostic. This prioritizes a targeted learning-objective experiment; it does not assume that the proposed loss will work. The current preferred research profile remains unchanged until the comparison is complete, and the sealed holdout stays reserved.

Artifacts: artifacts/token-range-decomposition-20260915-v1 contains the protocol frozen before scoring, source hashes, frozen code, rows.parquet, daily.csv, summary.csv, paired.csv, attribution rows/daily/summary, test receipt, independent verification and final manifest.

Sources: [external benchmark](token-external-benchmark-20260915.md), [later-period transfer](token-profile-transfer-20260915.md). Component definitions and the averaging/attribution procedures are reproduced in the frozen scripts and src/quant_research/range_decomposition.py.
