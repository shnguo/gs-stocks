# Three-idea validation

## Scope

1. Return and ranking auxiliary loss: compare indicators_ranking against indicators. Both have the same inputs, so this comparison isolates the added objective.
2. Full-universe Top20: evaluate all four frozen arms on every inherited input-eligible stock for the same 71 evaluation dates. This replaces the earlier cap of 192 stocks per date with 378,079 stock-date inputs.
3. Market and industry context: compare context_ranking against indicators_ranking. The new arm adds CSI300 returns, cohort returns, breadth, liquidity and dated industry peer features while preserving the training budget and loss design.

The full-universe evaluation and independent completion audit are now complete. All 12 model runs cover the 71 dates; the three ideas have separate conclusions. See the [final conclusions](/Users/guo/Documents/stocks/quant-research/artifacts/token-three-ideas-20260917-v1/completion-audit/conclusions.md) and [completion audit](/Users/guo/Documents/stocks/quant-research/artifacts/token-three-ideas-20260917-v1/completion-audit/verification.json). Completion does not mean that every change improves performance.

## Frozen protocol

- Config: configs/token-three-ideas-v1.json.
- Output: artifacts/token-three-ideas-20260917-v1.
- Existing models: artifacts/token-ranking-20260916-v2, reused without retraining.
- New context arm: original midpoint parent, seeds 17/29/43, 7,969 training rows over 251 dates, four fixed epochs, original learning rate and loss weights.
- Selection: 8,640 rows over 45 dates. Selection CE is diagnostic; there is no epoch or hyperparameter search.
- Evaluation: 378,079 rows over 71 dates, July 2024–July 2025; 64 paths per seed, five forecast sessions, batch size 32.
- Context features are available only at the signal-close position. Earlier token positions and unknown future positions do not receive backfilled context.
- Industry peers exclude the target stock; unknown or insufficient peers remain missing. Dated snapshots are no more than four calendar days old in this actual cohort.
- CSI300 data cover all 1,350 requested exchange-calendar sessions without forward-filling.
- All 13,632 shared sampled evaluation rows reproduce the previous history tokens, normalization and labels exactly, including when future encoder inputs are replaced by an unknown suffix.

## Manual execution and resumption

From /Users/guo/Documents/stocks/quant-research, execute:

PYTHONPATH=src:scripts artifacts/kronos-comparison-20260910-v1/venv/bin/python -u scripts/token_three_ideas.py

This is one manual process. It creates no scheduler or automation. A file lock prevents a second concurrent instance of this experiment. Completed model fits are retained; each forecast chunk is hash-verified before resumption. An interrupted unsealed chunk is recomputed using its fixed random seed.

The completed run used macOS caffeinate to prevent idle sleep. Its resumed manual process was detached from the tool session; the historical PID and command are recorded in run3-launch.json. Final logs are in artifacts/token-three-ideas-20260917-v1-run3.log; earlier logs are retained. All 12 forecast runs and the report are complete. The completed.json, validation/verification.json and completion-audit/verification.json files provide the completion evidence; prepared.json and unit tests alone do not establish performance.

## Evaluation interpretation

Publish mean and median effect, number and fraction of dates improved, date-block uncertainty intervals, individual seeds, calendar periods, poor-period results, native and paired universes, Top20 missing outcomes and volatility concentration. A wide interval alone does not negate an improvement; an improvement frequency above 50% is descriptive evidence rather than a calibrated probability of future success.

Rank by sell reference / buy reference − 1 − 0.25%. Buy reference uses the median predicted T low. The exit date uses forecast peak-day voting over T+1 through T+4, and the sell reference uses the median high on that chosen day. Freeze ranking and Top20 before attaching actual outcomes. Unknown outcomes do not admit replacement by rank 21. Actual daily extrema are scenarios, not executed returns.

Historical source membership, corporate-action provenance and unknown tokenizer pretraining overlap remain limitations. Industry snapshots are reconstructed historical records, not independently certified publication-time vintages. No sealed holdout is read, and no live model or report pointer is promoted by this experiment.
