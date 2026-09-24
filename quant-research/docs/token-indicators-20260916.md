# BOLL, MACD and KDJ inputs for the token model

## Results

The five variants completed four matched training epochs and generated 722,240 five-day paths over 2,257 evaluation inputs and 71 dates. Results below use this experiment's freshly trained baseline and common usable stock-date cohorts.

| Variant | Return forecast MAE, percentage points ↓ | Top-ranked group extrema scenario ↑ | High/low mean error change vs baseline, percentage points ↓ |
| --- | ---: | ---: | ---: |
| Baseline | 4.2278 | 9.02% | — |
| BOLL | 4.3051 | 9.08% | +0.0580 |
| MACD | 4.2182 | 9.23% | +0.0338 |
| KDJ | 4.0909 | 9.01% | +0.0507 |
| All three | 4.3688 | 9.55% | +0.0493 |

- **KDJ is a candidate for better return-percentage prediction:** MAE fell 3.24%, improving on 40 of 71 dates (56.3%). Its ranked extrema scenario was approximately unchanged and high/low errors worsened.
- **The combined inputs are a candidate for ranking:** the top-group extrema scenario rose by 0.524 percentage points. Lift over the same model's whole sampled cohort rose from 3.544 to 3.918 percentage points, with higher lift on 53.5% of dates. Price and return-percentage forecast errors worsened.
- MACD gave smaller average ranking and return-error improvements. BOLL had no clear overall advantage in this run.

These are trade-offs, not a reason to discard every feature because one metric regresses. All trained candidates are retained. KDJ's return-error improvement and the combined model's ranking improvement have date-block bootstrap intervals spanning zero; this conveys uncertainty and is not used as an automatic rejection threshold. A date win fraction is not the probability of future effectiveness.

The current daily model and report pointer remain unchanged. This single-seed historical comparison does not establish which candidate will work best prospectively. In addition to historical-data limitations, repeated baseline training on MPS was not bit-identical; small effects should not be compared across runs as if they were feature effects.

The extrema scenario evaluates actual T low to the actual high on the exit date selected by the forecast, less 0.25% cost. It is not an executed return. The top group is the first 20% of the sampled stock cohort.

[Full metrics](../artifacts/token-indicators-20260916-v1/report.md) · [Verified per-date ranking comparisons](../artifacts/token-indicators-20260916-v1/verified-ranking-comparison.json) · [Independent verification](../artifacts/token-indicators-20260916-v1/verification.json)

## Implementation

The existing auxiliary-input branch now supports five variants: baseline, BOLL, MACD, KDJ and all three. Each single-indicator variant adds two continuous inputs plus their availability masks. The combined variant adds six inputs. The tokenizer, five-day OHLCVA outputs and learned forecast heads retain their existing contract.

Features are calculated from exactly the same 60 historical bars used by the market-token model. Every feature at a historical position uses only the prefix ending at that position. No extra pre-window price history is given to the indicator variants. No buy/sell threshold rules are introduced.

| Family | Inputs | Fixed convention | First available observation |
| --- | --- | --- | --- |
| BOLL | Position within bands; band width / middle band | 20-day mean, population standard deviation, bands at ±2 standard deviations | 20th |
| MACD | DIF / close; (DIF − signal) / close | Recursive EMA 12 and 26, signal EMA 9; EMA seeded with first available observation | DIF: 26th; histogram: 34th |
| KDJ | K / 100; D / 100 | 9-day RSV; K and D each recursively smoothed with weight 1/3; initial K=D=50 | 9th |

BOLL position is not clipped to the band boundaries. A zero-width band masks the undefined position while retaining zero band width. A zero-range KDJ window is masked and resets its smoothing state. Missing or invalid calendar bars break rolling windows and reset recursive state; there is no new price filling or compressed trading calendar.

J is omitted because it is a linear combination of K and D. The MACD histogram uses DIF−signal rather than the doubled display convention in some charting software. The precise initialization and scaling are frozen so training and inference agree. Recursive indicators restart at the beginning of each 60-bar window and can differ from a chart initialized years earlier.

Definitions: [Fidelity BOLL](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/bollinger-bands), [Fidelity MACD](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/macd), [Moomoo KDJ](https://www.moomoo.com/au/hans/support/topic3_142). These explain the indicator families; the initialization, masks and scale-normalized model inputs above are our explicit implementation choices.

## Data and training boundary

- Derive indicators from the existing signal-history price archive and adjustment factors, independently matching every batch's OHLCVA normalization statistics to the frozen token training dataset.
- Preserve the exact stock-date row IDs from the preceding feature experiment: 7,969 training rows over 251 dates, 716 selection rows over 45 dates and 2,257 evaluation rows over 71 dates.
- Training labels end before 2024; selection is 2024 H1; evaluation spans 2024 H2 through July 2025. Five-day label embargoes are preserved.
- Every variant starts from the same historical midpoint checkpoint, uses seed 17, four continuation epochs, the same batch order, equal five-day token CE plus 0.05 midpoint loss, and 64 forecast paths per input.
- The evaluation cohort's six signal-day indicators are all available. Earlier history availability reflects warm-up masks: approximately 68.3% for BOLL, 58.3% for MACD DIF, 45.0% for its histogram and 86.7% for K/D.
- Normalization is fitted only on training histories and saved in each model checkpoint. New input branches have zero initial output and therefore preserve the parent model's initial predictions.
- Future indicator values are always missing during both teacher-forced training and free-running generation. Actual future prices are never used to build auxiliary inputs.

This is a bounded historical development comparison, not an untouched or prospective evaluation. The original price archive and tokenizer provenance limitations remain. The sparse financial snapshots are not feature inputs in this experiment; their pre-existing cohort dates are reused for comparability.

## Manual operation

From any directory:

    /Users/guo/Documents/stocks/quant-research/scripts/run_token_features.sh --config /Users/guo/Documents/stocks/quant-research/configs/token-indicators-v1.json

Add --prepare-only to validate sources and prepare features without training. The script uses the existing pinned environment, prevents concurrent runs of the same experiment, validates retained results and reuses completed runs. A different protocol requires a new output directory.

The current daily workflow continues through scripts/run_daily_token.sh. The feature experiment does not replace daily checkpoints, change the latest report pointer, create an automation or place trades.

## Inference integration

The public predict_indicator_paths adapter derives the exact checkpoint-selected indicator features from raw historical OHLCVA plus factor data and calls the existing path predictor. The experiment runner uses the same indicator_features implementation when preparing training and evaluation inputs. Checkpoints retain feature names/order and training-only scaling buffers.

## Evaluation

Compare two-day and five-day maximum-high and minimum-low MAE on stock-date cohorts usable in all five variants. Retain per-arm legal-path coverage, CRPS, interval scores and per-date metrics. Report the size and frequency of improvement across dates and eight period/horizon/extrema scenarios, with date-block bootstrap intervals.

For ranking, use the existing reference-price definition: median T low for entry, the modal predicted peak date among T+1 through T+4 for exit (earliest tie), and median high on that chosen date for the exit price. Rank by exit / entry − 1 − 0.25%.

Outcome review freezes the predicted exit date and evaluates that day's actual high against actual T low. The top group is the first 20% of the sampled cohort, not the entire stock market. These extrema scenarios are not executed returns or guaranteed fills.

[Full experiment report](../artifacts/token-indicators-20260916-v1/report.md)

## Verification

Tests cover independent formula parity, prefix causality, split-adjustment and price-scale invariance, warm-up, calendar gaps, flat prices, checkpoint feature order, inference integration and existing model/daily-workflow regressions. Independent artifact verification recalculates features with pandas, checks source hashes and train-only scalers, replays sampled paths, recalculates high/low scores and reference-price rankings, and checks that daily workflow pointers and manifests remain unchanged.

Reproducibility note: repeating the preceding baseline with the same parent, rows and seed did not produce bit-identical trained weights on MPS; initially tiny numerical differences grew during sampled-loss continuation. All indicator comparisons use the freshly trained baseline in this run. Cross-experiment differences must not be attributed to features, and small effects require replication before a strong effectiveness claim. This differs from checkpoint inference replay, which is checked separately.

Final checks: 64 tests passed and one optional tokenizer-environment test was skipped. Independent verification covered 384 real feature windows, exact forecast replay for 20 inputs × 64 paths, 10,865 reference-ranking records and 1,420 daily high/low scores. The same completed manual invocation was rerun successfully without training again. Original daily configuration, latest pointer and run manifest hashes were preserved.
