# Daily token-model training, forecasts and stock selection

## Current execution mode: manual only

As requested, no model scheduler remains enabled. The new macOS LaunchAgent com.quant-research.daily-token was unloaded and its plist was moved out of Library/LaunchAgents into the verification archive. The Codex automation a-share-post-close-token-workflow is PAUSED. Scheduler installation scripts were retired. Existing data collectors com.mootdx.local-collector and com.mootdx.local-research are unchanged.

Run this command manually from any working directory:

    /Users/guo/Documents/stocks/quant-research/scripts/run_daily_token.sh

It uses the pinned Python and configuration automatically. The command archives and validates closed daily data/calendar/continuity, prepares only matured labels, incrementally trains all three checkpoints with replay, verifies checkpoint reloads, forecasts five sessions, publishes the reference-price return ranking and report, publishes a separate early-stage watchlist, and reviews prior mature predictions. The existing exclusive lock prevents concurrent updates. A completed close is verified and reused without another training update.

The early-stage watchlist is supplemental. It does not replace the return-maximizing ranking or any model checkpoint. It combines the existing forecast score with signal-close-only MA5/10/20, ATR14, recent-return, volume and pullback evidence. The report shows the forecast first-session open entry price, selected future-high exit price, net return, dates, prior ten-session return, MA20 distance and extension flags. Its historical validation covers 71 dates and 368,988 stock-date rows. The selected early-stage arm has a 3.88% mean open-entry scenario, improves the lower-tail return on 77.5% of dates and adverse excursion on 95.8%, and reduces Top20 overextension from 81.8% to 39.3%. Its average return remains below the original ranking, so both lists are retained. See [validation](../artifacts/token-early-stage-20260921-v2/report.md), [independent verification](../artifacts/token-early-stage-20260921-v2/independent-verification.json), and the [latest example](../artifacts/daily-token-live-v1/reports/2026-09-21-early-stage-v2/report.md).

The default chooses the latest trading date whose Shanghai 16:30 gate has passed. Before that time it can select the previous eligible close; check the returned signal date. To explicitly process a particular already-completed close:

    /Users/guo/Documents/stocks/quant-research/scripts/run_daily_token.sh --signal-date YYYY-MM-DD

An explicit date must be an open session whose 15:00 close has already occurred. Source data still must pass validation. Failed or incomplete inputs stop the run and retain evidence. Exit status and last-error.json report failures; latest.json links the completed report, and current.json binds its checkpoints. There is no automatic retry, model run or Codex notification. Re-run the same command manually after resolving a failure. Calendar evidence expires at the end of 2026 and must be updated manually before a forecast extends beyond it.

The manual-entry verification reused the completed September 15 run; it did not perform another training cycle or claim a fresh September 16 report. Retired scheduling files and the native test receipt are retained under artifacts/daily-token-live-v1/verification/manual-only.

## Current ranking: return from the displayed reference prices

The user confirmed that the desired return is the percentage implied by the stated buy/sell reference prices. The [current report](../artifacts/daily-token-live-v1/reports/2026-09-15-t_low_to_future_high-reference-return-v1/report.md) therefore ranks sell reference price / buy reference price − 1 − 0.0025, descending. The CSV's expected_net_return equals reference_price_net_return. Price calculations use full precision; the report shows four decimals. ST Zhuoran now shows 12.76%, not the former 61.27% pathwise mean, and ranks fourth in this saved cohort.

Keep the previously chosen dates and reference prices: T-low entry, the equal-model modal peak date among T+1–T+4, and that date's high-price median averaged across models. Do not silently change dates to raise the return. Path-average maxima and their previous ranks remain explicitly named diagnostic CSV columns; they are omitted from the main human-facing return field.

Each new reference-price ranking receives a real publication timestamp. A reference-publications pointer binds its immutable view to the source run. Forward scorecards use the published order, expected reference-price return and frozen exit dates, not the old raw-run path-mean ranking. At two days, price errors remain measurable while return outcomes for later selected exits remain pending; those pending rows are excluded from return averages and Brier scores. Full five-day reviews evaluate all observable selected exits. These are price-extrema scenarios, not guaranteed executed returns. Legacy runs without a reference publication retain their original review method.

All 5,328 stocks were recalculated with unchanged buy/sell dates and reference prices; 28 focused tests passed, including reversal of the old path-mean order and return-maturity handling. The original run/model artifacts and earlier reports remain unchanged. [Verification](../artifacts/daily-token-live-v1/verification/reference-return.json).

## Report fields: expected returns and entry/exit dates

The current [Chinese report](../artifacts/daily-token-live-v1/reports/2026-09-15-t_low_to_future_high-timing-v1/report.md) explicitly lists expected net interval return, buy date/price, sell reference date/price and the selected date's model-implied frequency. All 5,328 ranked stocks receive these fields in the CSV. Original ranks, forecasts and checkpoints are unchanged.

Buy date is T. For each model, count the date of the maximum high over T+1 through T+4 on each valid five-day path, resolving tied highs to the earliest day. Normalize within each model, then average the three frequency vectors. Choose the highest-frequency date, with the earliest date breaking ties. This respects equal model weights even when valid-path counts differ.

The sell reference price averages the three medians of that selected day's highs across all valid paths; it is not the median of maxima over different dates. The buy reference price averages T-low medians. Stock detail sections separately display the ranking's pathwise maximum-return mean, the expected return for the selected date, and the ratio computed from the reference prices. These are distinct summaries, not interchangeable return estimates. Four-date frequency distributions are displayed for every top-20 stock. Daily bars cannot locate an intraday time such as 10:30, and extrema do not guarantee fills.

The publisher creates a versioned timing-v1 view, preserving prior immutable reports and the original ranking publication time. The existing daily operator automatically uses this format on subsequent invocations. Twenty-seven focused tests passed; the actual report's date frequencies, reference prices and fixed-date expected returns were independently reconstructed for all 5,328 stocks. Evidence: [timing verification](../artifacts/daily-token-live-v1/verification/timing-report.json).

## Historical ranking revision: pathwise T low to later maximum high

On September 16, the user changed the ranking objective. The latest report is now the [ideal-timing ranking](../artifacts/daily-token-live-v1/reports/2026-09-15-t_low_to_future_high/report.md), with a [full CSV](../artifacts/daily-token-live-v1/reports/2026-09-15-t_low_to_future_high/ranking.csv). The September 15 original run and report remain immutable. The revised view reuses the same sampled paths and checkpoint hashes; no training or inference was repeated. Its ranking publication remains late and is not counted as prospective performance.

For each path: net potential = max(high on T+1, T+2, T+3, T+4) / low on T − 1 − 0.0025. Average valid path returns within each model, then average the three model means. Do not take the best sampled path or divide separately averaged entry/exit prices. Negative potential is retained.

New daily runs record the ranking method in their immutable run metadata, and their scorecards follow that recorded method. The operator can revise the presentation of an already-completed close without applying its gradient updates twice. The manual operator uses this ranking rule; recurring model tasks have been disabled.

Verification: 26 focused tests passed, covering same-day-high exclusion, path pairing, horizon distinction, negative outcomes, forward scorecards and preservation of existing reports. Independent reconstruction checked all 32,020 model/horizon scores and the order of 5,328 stocks, with all three checkpoint hashes and the original run manifest unchanged. Evidence: [ranking verification](../artifacts/daily-token-live-v1/verification/2026-09-15-ideal-timing.json).

## Operating decision

Prioritize a usable daily feedback loop. Retain improvements that work in a majority of meaningful cases; assess their average benefit, adverse cases and operating cost together. Agreement across every seed or a confidence interval excluding zero is not required to produce a research watchlist. Winning in 51% of cases is evidence of breadth, not by itself proof of positive net value: magnitude and downside remain visible.

The daily token workflow adopts the adapted decoder, full-distribution 64-path sampling and midpoint-loss direction. All three trained seeds participate equally. Daily updates run from the last completed operational checkpoints, with a small learning rate and replay. The existing experimental artifacts and older LightGBM workflow are preserved. This workflow has no fixed buy-limit formula and sends no orders.

## Daily sequence

1. On manual invocation, identify the latest completed trading session S whose 16:30 Asia/Shanghai gate has passed, or validate an explicitly requested completed close against the verified exchange calendar. Holidays and weekends do not create new training days.
2. Reuse the existing Rust source collector and normalizer to archive the current stock master and completed daily bars. Extend the research snapshot using verified price continuity. Preserve missing/uncertain instruments in the coverage ledger.
3. Train one small continuation epoch per model on newly matured five-day labels. A row whose signal is S−5 first becomes fully trainable at close S. Replay older recent examples and a fixed-size sample from the pre-2024 historical training pool.
4. Freeze three updated model checkpoints and retain their optimizer states and parent hashes. Each checkpoint must reload exactly. An OS process lock and immutable per-close completion receipt prevent concurrent or duplicate training.
5. Predict T through T+4, where T is the first trading session after S. Each model generates 64 OHLCVA paths for each eligible stock. Preserve raw paths, coarse/fine tokens, masks, daily quantiles, model scores and the coverage ledger.
6. Publish the common eligible stocks sorted by sell reference price / buy reference price − 1 − the declared 0.25% round-trip cost scenario. Keep the chosen sell reference date and date-specific reference price. Retain raw pathwise-maximum statistics only as diagnostics; they do not define the published order. Show the top 20, with model-implied positive-path frequency, low-side risk and disagreement. This cost scenario is a comparison assumption, not verified brokerage execution cost.
7. Review frozen forward forecasts as two- and five-session outcomes mature. Retain missing or action-affected outcomes as unknown. Report improvement frequency versus persistence, average error change, high/low MAE and CRPS, interval coverage/score, probability Brier score, ranking correlation, top-list return scenario and eligible-pool comparison.

The current published ranking measures the return implied by the specified reference prices and dates. The older pathwise-maximum aggregation is historical diagnostics only. Extremum prices are not guaranteed executable fills. The average potential may be influenced by unusually small predicted entry lows; no undeclared clipping or risk penalty is applied. The two-session variant buys at T low and sells at T+1 high. Prospective outcome reviews apply the same formula to observed extrema and label it potential, not actual trading profit. Historical open-to-close runs retain their original review formula. The positive-path frequency is not a calibrated real-world probability. Quantiles averaged across models are labeled as such; no mixture-distribution or 80% coverage guarantee is claimed.

## Incremental training settings

- Architecture and tokenizer remain the existing token model: approximately 3.72 million predictor parameters, 60-session history and five-session future.
- Three seeds: 17, 29 and 43; no seed is discarded using the 2025 results.
- One continuation epoch; learning rate 0.00001; midpoint objective weight 0.05; original equal-horizon token loss retained.
- Initial update: 128 eligible stocks per date over 60 recent mature signal dates, plus 2,048 pre-2024 historical replay rows. This is a recent-data adaptation, not a claim that every missing year was retrained.
- Subsequent closes: all available newly matured rows in the retained snapshot, up to 2,048 older recent replay rows, plus 2,048 historical replay rows.
- New checkpoint-specific interval factors are not fitted daily. The old calibrated factors belong to old checkpoints and are not silently applied to updated weights. Forward coverage scorecards show whether the new raw intervals need adjustment.

The operational model uses observable recent labels, including dates beyond the old research holdout boundary. That boundary no longer describes untouched data for this live model. Frozen historical experiment results remain unchanged, and new forward forecasts are scored only after publication and outcome maturity.

## Runtime and outputs

Run from /Users/guo/Documents/stocks/quant-research with the pinned research Python:

    PYTHONPATH=src:scripts artifacts/kronos-comparison-20260910-v1/venv/bin/python scripts/daily_token_cycle.py --config configs/daily-token-v1.json

The command checks Shanghai time itself, so the host's time zone and daylight-saving changes do not shift the market-close gate. Repeating a completed close returns the saved result. A concurrent invocation returns already_running. A process failure leaves the previous active model pointer intact until the new run completes; completed stages can be resumed.

Outputs are rooted at artifacts/daily-token-live-v1:

- current.json: active three-checkpoint chain, completed close and label watermark.
- latest.json: links to the latest report, complete ranking and daily OHLCVA forecast export.
- latest-early-stage.json: links to the supplemental early-stage watchlist; it never replaces latest.json.
- runs/S: immutable inputs, trained models, forecast paths and analytical output.
- reports/S/report.md: top-20 watchlist and five-session prices for each listed stock.
- reports/S/ranking.csv and daily-ohlcva.csv: complete ranking and all daily forecast quantiles.
- reports/S-early-stage-v2: supplemental early-stage ranking, all four research reranks and a report with entry/exit reference prices.
- scorecards: immutable two/five-session outcomes; evaluation-latest.json indexes the latest reviews.
- verification: independent first-run verification and implementation receipts.

Reports published after the first forecast day's 09:15 Shanghai cutoff are labeled late_research_watchlist and excluded from forward scorecards. The initial catch-up close is September 15, with forecasts for September 16, 17, 18, 21 and 22. This catch-up report is late; subsequent after-close runs target future sessions.

## Calendar and source limits

The calendar is verified through December 31, 2026 against the [Shanghai notice](https://www.sse.com.cn/disclosure/announcement/general/c/c_20251222_10802507.shtml), [Shenzhen notice](https://investor.szse.cn/disclosure/notice/general/t20251222_618087.html) and [Beijing notice](https://dataclouds.cninfo.com.cn/sjother2/regulatory/2025/20251222/55875a9937374da4ae2d4999dfda2722.pdf). The provider's dates must match weekday and holiday exclusions. Renewal is required before a forecast horizon extends outside that range.

The existing source-continuity checks exclude unverified factor changes and missing histories. They do not establish complete shareholder-entitlement, news, suspension or price-limit execution evidence. Policy and news are not modeled by this OHLCVA-only pipeline. The watchlist supports stock selection; it is not an executable trading ledger.

## Completion evidence

The first cycle completed on September 16 at 15:26 Shanghai using the September 15 close. Each of three models trained on 9,728 rows: 7,680 recent mature examples and 2,048 historical replay examples. Forecasting generated 1,033,920 five-session paths for 5,385 stocks. The common three-model ranking contains 5,328 stocks; 57 inputs did not have enough valid generated paths in all three models.

- [Top-20 report and five-day prices](../artifacts/daily-token-live-v1/reports/2026-09-15/report.md)
- [Full ranking](../artifacts/daily-token-live-v1/reports/2026-09-15/ranking.csv)
- [159,840 daily field/quantile rows](../artifacts/daily-token-live-v1/reports/2026-09-15/daily-ohlcva.csv)
- [Independent verification](../artifacts/daily-token-live-v1/verification/2026-09-15.json): all input chronology and normalization, 4,041 chunks, 32,020 model scores, 480,300 model/day/field quantiles, the ranking and six exact first/last-batch replays passed.
- [Engineering tests](../artifacts/daily-token-live-v1/verification/engineering-tests.json): 24 passed. [Repeated invocation](../artifacts/daily-token-live-v1/verification/idempotency.json) reused the run with unchanged checkpoint and completion hashes.
- [Original automation receipt](../artifacts/daily-token-live-v1/verification/automation.json) is historical. That Codex task is now paused; the model workflow is manual only.

Scheduling was subsequently canceled by the user. The native LaunchAgent successfully executed one already-completed-date check before being unloaded; this did not run a new close. Both recurring model entry points are now disabled, and only the manual command above should be used.

The first report remains a late research watchlist and has no forward scorecard. No prediction accuracy or realized profitability is claimed from this engineering run. A manual run for the September 16 close would forecast September 17, 18, 21, 22 and 23; it has not been initiated by this scheduling cancellation.
