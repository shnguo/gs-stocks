# Daily-history dense Transformer and output-loss experiment

Run date: 2026-09-14 PDT / 2026-09-15 UTC.

## Outcome

Both authorized stages and their independent audits are complete. Historical training now covers 1,902 and 2,144 signal dates in the two folds, with 354,007 and 400,198 distinct labelled rows visited per complete run. The larger dense model shows no consistent path-forecast advantage over the smaller model.

The tested output-loss weights [1.0, 0.8, 0.6, 0.4, 0.2] do not show a reliable improvement. Mean two-day high/low/range distribution error increases 1.64% in 2024 H2 and decreases only 0.04% in the 2025 window; five-day error increases 1.97% and 0.24%. All four mean-three paired CRPS intervals include zero. Keep equal weighting as the research reference for now.

The experiment completed six training runs and retained 602,112 sampled five-day paths. Twenty tests passed, the real-data tokenizer integration passed separately, all six CPU checkpoint reloads reproduced exactly, all six sampled-path replays passed, and independent raw-path calculations matched all 36 neural-model/horizon/target CRPS and MAE summaries. The final holdout remains sealed.

## Scope and order

The authorized experiment first expands historical date coverage and compares the existing 281,936-parameter and 3,720,448-parameter dense token decoders. After all four baseline training/generation runs and their path evaluation finish, it trains the 3,720,448-parameter model with decreasing forecast-day loss weights in both temporal folds.

Both models predict the same hierarchical coarse/fine tokens and decode them into five future days of open, high, low, close, volume and amount. The official Kronos tokenizer is frozen. Predictor weights start from scratch. Architecture, normalization, input history, token vocabulary, optimizer, initialization seed and sampling settings stay fixed for the loss comparison.

## Expanded data coverage

The approved source supplies 9,264,625 eligible historical stock/date windows across 2,282 dates, from 2016-03-03 through 2025-07-22. This is the eligible source universe, not a claim that every row received a gradient update.

A deterministic, outcome-independent sample selects 192 stocks per date, cycling across Shanghai, Shenzhen and Beijing exchange identifiers. The resulting encoded pool has 438,144 windows; 426,495 have at least one valid future label. Training visits every training date each epoch and rotates 32 stocks per date through the known-label pool. The report distinguishes eligible source windows, selected pool rows, and unique rows actually used by each model. The selected checkpoint can predate the end of training; both its visited-row count and the complete run's count are reported.

The inherited historical eligibility requirements determine the first available date. Historical context is 60 trading days. Normalization and the adjustment anchor use only those historical observations. No training-date recency weighting is introduced. Exchange sampling is deliberately balanced and does not represent a capitalization-weighted or naturally exchange-weighted market estimate.

| Fold | Training signal dates | Training dates | Selection signal dates | Selection dates | Evaluation signal dates | Evaluation dates |
| --- | --- | ---: | --- | ---: | --- | ---: |
| 2024 H2 | 2016-03-03–2023-12-22 | 1,902 | 2024-01-02–2024-06-21 | 112 | 2024-07-01–2024-12-24 | 120 |
| 2025 Apr–Jul | 2016-03-03–2024-12-24 | 2,144 | 2025-01-02–2025-03-24 | 52 | 2025-04-01–2025-07-22 | 76 |

Every five-day target must finish before the next partition boundary. The later fold expands its training history into the earlier evaluation period according to the protocol fixed before results were read. The final holdout beginning 2025-08-07 remains sealed; the latest target read is 2025-07-29.

All 12 month-embedding categories occur in both training histories. This addresses the previous 43-date experiment's unseen-month limitation.

## Output-loss design

For each row, compute the coarse-token and fine-token cross-entropy separately for forecast days 1 through 5. For either head:

L = sum(valid[h] × weight[h] × CE[h]) / sum(valid[h] × weight[h]).

Average the two head losses, then use equal weight per signal date. Unknown future labels invalidate the remaining suffix and receive no loss or gradient. Rows with no known targets remain in coverage records but do not train the model.

| Forecast day | Equal weighting | Decreasing weighting | Share for a fully known weighted row |
| --- | ---: | ---: | ---: |
| 1 | 1.0 | 1.0 | 33.33% |
| 2 | 1.0 | 0.8 | 26.67% |
| 3 | 1.0 | 0.6 | 20.00% |
| 4 | 1.0 | 0.4 | 13.33% |
| 5 | 1.0 | 0.2 | 6.67% |

The first two forecast days carry 60% of a fully known row's weighted loss. For a row with only two known days, the denominator is 1.8. These are weights on output-token errors; no input observations or historical dates are reweighted.

The best checkpoint is selected by unweighted, date-equal five-day token cross-entropy for every variant. Training uses AdamW, learning rate 0.0003, weight decay 0.01, batch size 256, seed 17, dropout 0.1, minimum 8 epochs, maximum 24 epochs, and patience 4. A run reaching the epoch limit is reported as budget-limited, not converged.

## Evaluation fixed before training

Each evaluation date contributes 32 stocks selected before checking future labels. Every model generates 32 independent five-day paths for each selected input, using temperature 1, top-p 0.9, and fixed batch boundaries and seeds.

Score the maximum high, minimum low, and high-minus-low range over the first two and all five forecast days. Errors use percentage points of the signal close. Report median MAE, distribution CRPS, nominal 80% interval coverage and width, interval score, and bias. CRPS measures distribution error; lower is better.

The historical-volatility baseline resamples 32 contiguous five-day blocks of historical OHLC log returns from the same 60-day input, after removing mean historical close drift. The deterministic persistence baseline keeps the signal close constant. Baseline samples are identical across compared models.

An evaluation row needs a complete known label prefix for its horizon and at least eight legal generated paths. Invalid prices are retained in the saved outputs, never repaired. Report each model's own coverage and metrics, then compare models on an identical common cohort. Coverage is broken out by exchange. Metrics average stocks within each signal date and then average dates. Results remain separate by fold.

The primary weighted-loss comparison is the mean CRPS across the three two-day extrema/range targets, together with the corresponding five-day tradeoff. Paired uncertainty intervals resample 10 consecutive trading dates in 2,000 circular block replicates, rather than treating overlapping daily outcomes or thousands of stocks as independent observations.

## Verification and limitations

- Equal-weight loss reproduces the original loss values and every parameter gradient in a controlled double-precision test.
- Tests cover chronological label purging, date-balanced rotation, exchange identifiers, unknown-suffix gradients and action/factor/episode barriers.
- The vectorized historical label builder matches the established independent price-label implementation on 192 historical inputs.
- The pinned tokenizer produces exactly the same first 60 tokens with or without future observations on the saved real-data probe.
- Training, generation, evaluation, reload checks and final results are recorded in the artifacts below.

These are two development windows with a single training seed. The frozen tokenizer's full pretraining-date manifest remains unverified, and the accepted source's historical identity and unresolved-boundary limitations remain. Neither a lower token loss nor better extrema forecasts establish realizable trading profit. This experiment does not test a model trained only on two future days.

## Artifacts

- Protocol: ../configs/token-history-v1.json
- Immutable dataset and executed code: ../artifacts/token-history-data-20260915-v1
- Training, paths and comparisons: ../artifacts/token-history-experiment-20260915-v1
- Runtime log: ../artifacts/token-history-experiment-20260915-v1.log
- Entry points: ../scripts/prepare_token_history.py, ../scripts/token_history_run.py, ../scripts/evaluate_token_history.py, ../scripts/verify_token_history.py

The first preparation attempt stopped before training because its sampler expected a different instrument-ID layout. Canonical exchange parsing was corrected, regression-tested and the failed preparation preserved separately. The successful dataset and executed code are hash-bound; no model trained on the failed pool.

### Reproduce or resume the frozen experiment

From the quant-research directory, the run uses the pinned tokenizer environment and the dataset's frozen source snapshot:

    PYTHONPATH=artifacts/token-history-data-20260915-v1/code/src artifacts/kronos-comparison-20260910-v1/venv/bin/python artifacts/token-history-data-20260915-v1/code/scripts/token_history_run.py --stage all

Completed phases are hash-verified and reused. Interrupted training resumes from an epoch checkpoint with optimizer and RNG state. The weighted phase requires a completed dense evaluation. The independent audit runs after the entire experiment completes:

    PYTHONPATH=artifacts/token-history-data-20260915-v1/code/src artifacts/kronos-comparison-20260910-v1/venv/bin/python scripts/verify_token_history.py

## Completed original dense comparison

All four training runs, path generations, fixed-seed path replays and the common-cohort dense evaluation completed before weighted training started.

| Fold | Model | Selected / completed epochs | Distinct rows seen by selected checkpoint | Distinct rows seen in complete run |
| --- | --- | ---: | ---: | ---: |
| 2024h2 | dense_282k | 24 / 24 | 354,007 | 354,007 |
| 2024h2 | dense_3720k | 5 / 9 | 304,320 | 354,007 |
| 2025q2 | dense_282k | 24 / 24 | 400,198 | 400,198 |
| 2025q2 | dense_3720k | 6 / 10 | 400,198 | 400,198 |

Both small-model runs reached the 24-epoch limit and selected the last epoch. Both larger models stopped after four non-improving epochs. Small-model convergence is therefore unresolved under this budget.

Mean distribution CRPS across maximum, minimum and range, in percentage points of signal close; lower is better:

| Fold | Horizon | 282k equal | 3.72M equal | Historical volatility | Persistence |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2024h2 | 2 days | 2.4887 | 2.4966 | 2.6569 | 5.5330 |
| 2024h2 | 5 days | 4.4388 | 4.4592 | 4.6449 | 9.4541 |
| 2025q2 | 2 days | 2.1325 | 2.1397 | 2.1781 | 4.4788 |
| 2025q2 | 5 days | 3.4084 | 3.3757 | 3.4922 | 7.2271 |

The larger model has no consistent advantage over the small model. Its two-day CRPS is about 0.3% higher in both windows. Five-day CRPS is 0.46% higher in 2024 H2 and 0.96% lower in the 2025 window. All corresponding paired date-block intervals include zero.

Both models have lower mean CRPS than the historical-volatility baseline in each window/horizon, but only the 2024 two-day mean-three comparisons exclude zero under the stated conditional intervals. Nominal 80% intervals cover only about 57–61% for the neural models, versus about 70–72% for historical volatility. Distribution calibration remains a material weakness.

These results use the newly fixed daily evaluation cohort. They should not be compared numerically with the earlier eight-date Kronos or short-history experiments as if the cohorts were identical. Isolating the effect of longer training history alone would require matched short-history runs on these same evaluation inputs.

## Completed weighted-loss comparison

The table below uses the common cohort of all three neural variants within each horizon. Its cohort can be slightly smaller than that of the original two-model dense comparison above. Positive changes mean higher error.

| Fold | Horizon | CRPS change | Mean CRPS difference, pp | Conditional 95% interval, pp | Median MAE change |
| --- | ---: | ---: | ---: | --- | ---: |
| 2024h2 | 2 days | +1.64% | +0.04089 | [-0.00651, +0.09145] | +0.53% |
| 2024h2 | 5 days | +1.97% | +0.08771 | [-0.03133, +0.21055] | +1.37% |
| 2025q2 | 2 days | -0.04% | -0.00089 | [-0.01291, +0.01221] | +0.48% |
| 2025q2 | 5 days | +0.24% | +0.00799 | [-0.03517, +0.06925] | +0.77% |

Median MAE is higher under weighting in all four fold/horizon comparisons. The near-zero 2025 two-day CRPS difference does not establish a short-horizon benefit. These results apply to this weight vector, seed, training pipeline and development cohort; they do not establish that every possible horizon-weighting design would fail.

### Individual extrema and range targets

Relative CRPS change for the weighted 3.72M model versus its equal-weight counterpart:

| Fold | Horizon | Maximum high | Minimum low | High-low range |
| --- | ---: | ---: | ---: | ---: |
| 2024h2 | 2 days | +2.49% | +2.35% | +0.07% |
| 2024h2 | 5 days | +2.19% | +3.93% | +0.77% |
| 2025q2 | 2 days | -0.50% | +0.78% | -0.17% |
| 2025q2 | 5 days | -0.43% | +3.04% | -0.78% |

### Interval coverage and usable inputs

Coverage below is the date-equal mean across the three target intervals. “Common rows” means complete known labels and at least eight valid paths for every neural variant.

| Fold | Horizon | Common rows | Equal-weight 80% interval coverage | Weighted 80% interval coverage |
| --- | ---: | ---: | ---: | ---: |
| 2024h2 | 2 days | 3,747 | 57.51% | 53.84% |
| 2024h2 | 5 days | 3,602 | 58.53% | 55.13% |
| 2025q2 | 2 days | 2,367 | 59.31% | 59.97% |
| 2025q2 | 5 days | 2,248 | 61.34% | 61.77% |

The weighted model has worse interval coverage in 2024 and a small coverage increase in 2025. Neither reaches nominal 80% coverage under the fixed sampling settings. The new loss therefore does not resolve the calibration weakness.

### Training and token-level diagnostics

The weighted model selected epoch 10 of 14 in 2024 and epoch 6 of 10 in 2025. The 2024 equal-weight model selected epoch 5 of 9, so the common stopping rule produced different update counts. Both 2025 variants selected epoch 6 of 10 and saw the same sampled training rows through that checkpoint. Their initial selection scores and model parameter counts matched exactly.

An independent pass computed per-day coarse/fine cross-entropy on every evaluation-pool input, totaling 112,896 model/input evaluations across the six runs. The weighted model’s day-1 and day-2 token CE also increased in both windows:

| Fold | Day 1 CE change | Day 2 CE change |
| --- | ---: | ---: |
| 2024h2 | +0.449% | +0.311% |
| 2025q2 | +0.070% | +0.116% |

Increasing a day’s loss weight changes optimization priority. It does not guarantee that held-out errors for that day will fall. Here, the tested design improved neither near-day token errors nor the primary price-distribution comparison reliably.

### Final evidence

- Dense comparison: ../artifacts/token-history-experiment-20260915-v1/dense-evaluation/summary.json
- Weighted comparison and paired intervals: ../artifacts/token-history-experiment-20260915-v1/weighted-evaluation/summary.json
- Per-day token diagnostics: ../artifacts/token-history-experiment-20260915-v1/per-day-token-ce.csv
- Full split/source/checkpoint audit: ../artifacts/token-history-experiment-20260915-v1/independent-verification.json
- Independent raw-path score audit: ../artifacts/token-history-experiment-20260915-v1/independent-path-verification.json
- Implementation tests: ../artifacts/token-history-experiment-20260915-v1/implementation-checks.json
- Final artifact manifest: ../artifacts/token-history-experiment-20260915-v1/final-verification.json
