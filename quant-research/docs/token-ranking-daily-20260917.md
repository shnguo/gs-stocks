# Manual daily continuation for the matched candidate

## What changed

The baseline and return/ranking candidate now have a separate manual continuation chain. Both start from the verified September 15 incremental bridge, then consume the same verified daily inputs in the same order. The existing operational model and its default report remain separate.

The new comparison forecasts the full eligible stock universe, replacing the bridge's 192-stock sample. It uses three seeds for each model family and 64 five-session paths per model/stock. Candidate rankings show expected net return as a percentage, T buy date/reference low, and the selected T+1–T+4 sell date/reference high.

## Run manually after the close

For the complete workflow—validate/refresh close data, update the operational model, generate its report, continue both comparison models, generate their report, and review mature outcomes:

    /Users/guo/Documents/stocks/quant-research/scripts/run_daily_token_matched.sh

To continue only the comparison using already completed and verified daily inputs:

    /Users/guo/Documents/stocks/quant-research/scripts/run_token_ranking_daily.sh

The latter defaults to the existing operational daily pointer. It accepts --run followed by an absolute completed daily-run path. It does not refresh market data or train the operational model again. No scheduler, recurring task, broker action or automatic model promotion is involved.

## Continuation and recovery

- The model weights and AdamW optimizer state both continue from the prior comparison close. Feature normalization stays fixed. Learning rate, loss weights, decoder, seeds and sampling protocol match the bridge.
- Each update uses the original daily pipeline's newly mature rows, recent replay and historical replay. Both comparison arms see every row once and have identical date-grouped batches per seed.
- The command follows the main pipeline's actual saved parent chain and processes all intervening completed updates in order. Calendar gaps with no separate main update do not create invented updates.
- It verifies parent manifests/checkpoint lineage, trained-through cutoffs, label maturity and replay roles. Previously consumed signal dates cannot be mislabeled newly mature. Backward or unrelated ancestry is rejected.
- A per-store process lock prevents concurrent comparison updates. Completed model and forecast stages are reused. Interrupted incomplete stages restart from the same fixed parent; their gradients are never stacked on a partial update.
- A completed date is immutable. Repeating it reuses its verification and report; an older date cannot regress the current comparison pointer. A run sealed before a crash but not yet linked by the pointer can be audited and linked without retraining.
- The comparison pointer advances only after independent verification. The operational checkpoint/report pointers are not changed by the comparison runner.

The comparison store is artifacts/token-ranking-daily-v1. Its current.json points to the most recent verified comparison report. Protocol changes require a different store; the initial model chain and existing reports remain traceable.

## Ranking and review

Expected net reference return is sell reference price / buy reference price − 1 − 0.25%. The buy reference is the equal-model mean of median predicted T lows. The sell date is the earliest modal peak day among T+1–T+4 under equal-model date frequencies; the sell reference averages each model's median high on that fixed day.

Both complete rankings are frozen before outcomes exist. Reports completed before T's 09:15 Shanghai opening auction are eligible for prospective review. Late reconstructions remain permanently excluded. The independent audit happens after forecast/report freezing; comparison-pointer advancement additionally requires it to pass.

On subsequent manual runs, the review uses all five mature sessions and the forecast-selected exit date. It retains original Top20 membership, reports unknown outcomes, and never substitutes a lower-ranked stock for a missing Top20 outcome. It reports each arm's own-universe return MAE and Top20 extrema scenario; different forecast eligibility can affect direct metric comparisons. No automatic promotion is inferred from those summaries. Daily-bar extrema are price references, not verified fills or executed profits.

## September 16 continuation

Uses the saved September 16 main inputs: 9,449 rows per fit, including 5,353 newly mature rows, 2,048 recent replay rows and 2,048 historical replay rows. Training labels stop at September 16 and the latest training signal is September 9. It forecasts 5,355 eligible stocks over September 17, 18, 21, 22 and 23: 2,056,320 five-session paths across the six updated checkpoints.

This continuation is produced after September 17's opening auction. It validates full-universe operation and is excluded from prospective scoring. Later timely manual runs can collect forward evidence. The operational September 16 model, original September 15 bridge, and earlier frozen-candidate artifacts are preserved.

## Evidence

- [Configuration](../configs/token-ranking-daily-v1.json)
- [Comparison report](../artifacts/token-ranking-daily-v1/runs/2026-09-16/report.md)
- [Independent verification](../artifacts/token-ranking-daily-v1/runs/2026-09-16-verification.json)
- [Current comparison state](../artifacts/token-ranking-daily-v1/current.json)
- [Mature-outcome review](../artifacts/token-ranking-daily-v1/review-latest.json)

Automated tests cover parent-chain ordering/tampering, mature versus replay labels, optimizer step continuation, protocol compatibility, required audit, pointer recovery, repeated-run reuse, frozen exit dates, missing outcomes and the existing daily/model functions. Actual checkpoint reload, optimizer step counts, frozen normalization, reference-price calculations and sampled forecast replay are checked independently on the completed real run.

74 focused tests passed; one optional tokenizer-dependency test was skipped. The real model run uses the pinned runtime with the tokenizer dependencies. Both command-line entry points and shell syntax checks passed.

## Completed result and verification

All six incremental fits and all 2,056,320 forecast paths completed. Baseline ranks 5,303 stocks and candidate ranks 5,307; their Top20 lists share 14 stocks. Mean forecast-implied Top20 net reference returns are 9.59% and 9.20%, respectively. These are the models' predictions on their own selected stocks, not observed performance or evidence of either model's accuracy.

Independent verification recalculated all 10,610 ranking records and exactly replayed 192 model inputs across all six checkpoints. It verified carried optimizer step counts, frozen normalization, label cutoffs, identical row/batch exposure, exact CPU reloads and unchanged operational state. The comparison pointer now targets the verified September 16 report; the outcome review correctly records late_excluded.

A repeated manual command reused the completed result: all checked hashes and checkpoint/forecast modification times stayed unchanged, with no training or forecasting repeated. See the [reuse receipt](../artifacts/token-ranking-daily-v1-reuse.json). All task processes have finished.
