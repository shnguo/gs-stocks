# Matched incremental update

## Purpose and scope

The retained return/ranking candidate had frozen pre-2024 weights, while the operational model already consumed recent labels. This bounded bridge gives the matched historical baseline and candidate identical recent/replay exposure before comparing their forecasts. It is an isolated manual experiment, not a live-model replacement.

The live pointer was already on September 16 when this task began. The bridge deliberately uses the saved September 15 initial daily update, which has a complete, reproducible bootstrap cohort. The September 16 main checkpoints and report pointers are preserved. Later daily updates are rejected by this bootstrap command; it does not silently skip intervening updates or claim to implement a continuing daily candidate chain.

## Fixed protocol

- Parents: baseline and indicators/ranking checkpoints from the completed matched historical experiment, seeds 17, 29 and 43.
- Per fit: all 9,728 original daily training rows exactly once: 7,680 recent rows and 2,048 old replay rows. One epoch, learning rate 0.00001, fresh AdamW with weight decay 0.01 for both arms.
- Both arms have identical date-grouped batches, at most 256 rows per batch. Larger date groups are split; sparse replay dates are packed. The operational main updater uses random-row batches, so this is a matched experimental control rather than a replay of the current operational checkpoint.
- Baseline loss: token CE plus 0.05 sampled midpoint loss. Candidate adds the already fixed return/ranking objective: two bags of eight paths, up to four rows per date and sixteen per batch, weights 0.01 each for return and ranking, five-percentage-point normalization. Sparse dates contribute return loss, but pair loss only uses same-date peers.
- Candidate feature normalization remains frozen from historical training. BOLL/MACD/KDJ histories are rebuilt causally for both recent and historical replay rows. Future features remain missing.
- Training labels end no later than September 15; the latest training signal is September 8. All features, labels, timestamps, row identities and source hashes are checked against saved daily inputs.
- Forecast cohort: 192 exchange-stratified, hash-selected stocks from the eligible daily universe, chosen without future outcomes. Each model samples 64 five-session paths. Six fits generate 73,728 paths.
- Ranking retains sell reference / buy reference − 1 − 0.25%, equal-model median prices, and the forecast-selected exit date. Prices and dates are retained in both complete ranking CSVs.

The first preparation exposed different numeric storage conventions: recent training histories use column-major views, daily forecast histories use C-order copies, and archived raw prices use float32 while historical target prices retain float64. Reconstruction now matches each source convention. Historical raw-to-target checks are exact after casting target prices to the archive's stored float32 precision; actual training still consumes the original saved float64 labels. Failed preparation receipts/logs are retained.

## Interpretation

This run occurs after the first forecast session began. It is permanently labeled late workflow validation and is not eligible for prospective performance scoring. Its forecast-implied Top20 percentages measure what the updated models predict, not realized return, accuracy improvement or executable profits. The 192-stock cohort is not the whole-market watchlist.

The bridge removes unequal incremental data exposure within this matched pair. It does not prove the ranking objective alone caused a change: the candidate also includes indicators, and the historical experiment contains the separate indicator control. Future effectiveness still requires timely, frozen forecasts and mature outcomes.

## Manual command

    /Users/guo/Documents/stocks/quant-research/scripts/run_token_ranking_incremental.sh

This runs or reuses the fixed bridge, then independently audits prices/ranks and replays one forecast batch per checkpoint. Completed fits and forecast artifacts are hash-verified and reused. Each checkpoint persists optimizer state for a later explicitly implemented continuation; this command itself accepts only the initial cohort. No scheduler or automatic promotion is added.

## Outputs

- [Fixed configuration](../configs/token-ranking-incremental-v1.json)
- [Comparison report](../artifacts/token-ranking-incremental-20260917-v1/report.md)
- [Result receipt](../artifacts/token-ranking-incremental-20260917-v1/result.json)
- [Independent verification](../artifacts/token-ranking-incremental-20260917-v1-verification.json)

## Verification

64 focused tests passed; one optional tokenizer-dependency test was skipped. The actual model run uses the pinned runtime with the tokenizer dependencies. Tests cover bounded matched batches, row coverage, maturity cutoffs, duplicate identities, rejection of incomplete continuation, return/ranking gradients, cached inference, publication timing, missing outcomes and the existing daily pipeline. Final real-run verification is recorded in the linked receipt.

## Completed run

All six updates completed. Baseline ranks 191 stocks and the candidate 190; Top20 overlap is 15/20. Their mean forecast-implied Top20 net returns are 5.97% and 5.76%, respectively. Those are predictions from each model’s own selected stocks, not evidence that either model is more accurate or profitable. Both report prices, percentages and selected buy/sell dates.

Repeating the command reused the sealed training and forecast artifacts. The ordinary daily workflow and frozen-candidate comparison remain as previously configured; this new bootstrap experiment does not replace their defaults.

Independent verification passed: all 381 published ranking records were recalculated from raw paths, 192 model inputs replayed exactly across the six checkpoints, both arms had identical batch hashes per seed, candidate normalization buffers were unchanged, and current September 16 live-state hashes were preserved. All task processes have finished.

## Follow-up

The [manual daily continuation](token-ranking-daily-20260917.md) now carries both matched models and their optimizer states across verified updates and forecasts the full eligible universe. The initial bridge and its fixed bootstrap command remain preserved.
