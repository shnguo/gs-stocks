# A 股价格预测与交易计划研究

2026-09-23 event-model overlay: Added a point-in-time post-model overlay that preserves the
complete token-model control ranking and separately freezes event-window and confirmed-event
rankings. The initial shadow score combines 70% model-return percentile, 20% event score and
10% downside/agreement quality. Events must be known before the first forecast session and
pass explicit window, status and risk gates. Immutable bundles retain lineage, diagnostics and
a mature-outcome review command. This does not retrain or promote the model and does not create
orders. See [event and model overlay](docs/event-model-overlay-20260923.md).

2026-09-22 event radar MVP: Added an append-only, point-in-time event revision store,
official SSE and CNInfo periodic-report capture, a 60-day research radar, completed-close MA5 and
overextension gates, conservative pre/post event studies, source-integrity audits and a
four-list Chinese daily digest. The first live smoke capture normalized 2,321 SSE 2026
half-year-report records without backdating them; stale September 4 price data was blocked
from post-event confirmation. No scheduler, dashboard integration or trade instruction was
enabled. See [event radar operation and boundaries](docs/event-radar-mvp-20260922.md).

2026-09-22 matched early-stage test: Implemented an outcome-blind matched comparison controlling signal date, market segment, predicted-return rank, 20-day volatility and liquidity. A frozen rule requires at least 50 mature dates, 1,000 known pairs, positive mean executable-return difference on more than half of dates, non-worse adverse excursion and post-match rank gaps no larger than 0.05 before training an auxiliary head. Manual daily publications now retain compact matching cohorts and attach outcomes only after T+4 matures. The original 71-date replay cannot currently run because 144 of 147 required stock-level files were pruned; no result or model promotion is claimed. All 314 tests passed, with two optional skips, and a 10,000-row end-to-end synthetic replay passed independent verification. See [implementation and evidence](docs/token-early-stage-matched-20260922.md).

2026-09-21 early-stage watchlist: Added a causal second-stage reranker to address the primary list's concentration in already-surging stocks. A frozen 71-date, 368,988-row comparison shows that the selected early-stage arm improves lower-tail return on 77.5% of dates and adverse excursion on 95.8%, while reducing Top20 overextension from 81.8% to 39.3%. Mean open-entry scenario return falls from 5.98% to 3.88%, so it is published as a supplemental watchlist and does not replace the main return ranking. The manual daily script now produces both reports, with no scheduler or checkpoint promotion. See [validation and operation](docs/daily-token-live-20260916.md).

2026-09-17 manual matched daily continuation: Baseline and return/ranking candidate now carry both weights and AdamW state through verified daily updates, with lineage, duplicate-update and recovery checks. The September 16 continuation completed six fits on identical 9,449-row inputs and 2,056,320 paths for all 5,355 eligible stocks. Independent verification checked 10,610 ranking records and replayed 192 inputs; 74 tests passed, one optional test skipped. This reconstruction is late and excluded from prospective scoring. Operational model/report pointers remain unchanged; no scheduler or promotion. See [manual entry point and evidence](docs/token-ranking-daily-20260917.md).

2026-09-17 matched incremental bridge: Both historical baseline and return/ranking candidate completed the same initial daily update across three seeds: 9,728 rows per fit, one epoch, learning rate 0.00001, identical date-grouped exposure. Six fits generated 73,728 paths for an outcome-independent 192-stock sample; Top20 overlap is 15/20. This late reconstruction validates the workflow, not future accuracy or realized returns. The existing September 16 main checkpoints and report pointers remain unchanged. See [protocol, results and manual command](docs/token-ranking-incremental-20260917.md).

2026-09-16 return/ranking objective: Nine matched fits and 12,828,672 paths completed across three seeds and 192 stocks/date. Return forecast MAE improves 4.42% (51/71 dates); frozen Top20 observed extrema scenarios rise from 9.56% to 9.63%, with an uncertain mean gain and essentially flat returns on the 49 fully observed Top20 dates. Lower-tail outcomes improve. The candidate is retained for manual comparison, with no main-model replacement or scheduler. Independent path, fixed-rank, missing-outcome and replay checks passed; 61 tests passed and one optional test was skipped. See [results and manual operation](docs/token-ranking-20260916.md).

2026-09-16 technical indicators: Added BOLL, MACD and KDJ inputs and a manual five-arm comparison using the same 60-bar history and training budget. All five candidates completed training and 722,240 paths. KDJ reduced return forecast MAE by 3.24% (56.3% of dates improved); combined indicators raised the top-group historical extrema scenario from 9.02% to 9.55%, while high/low error worsened. Candidates are retained, with no daily-model replacement or scheduler change. Sixty-four tests passed; independent feature, ranking and forecast-replay checks passed. See [implementation and results](docs/token-indicators-20260916.md).

2026-09-16 auxiliary inputs: Added optional historical turnover/valuation embeddings with missing masks, training-only normalization, exact baseline warm start, and future-feature masking. The manual four-arm comparison completed on 7,969 training rows and 2,257 evaluation rows over 71 dates. Turnover reduced reference-return forecast MAE by 5.52%, while mean high/low error and sampled top-group extrema scenarios did not improve overall. Valuation improved 52.1% of dates but had slightly worse average error. All candidates are retained; live checkpoints and scheduling are unchanged. See [implementation and results](docs/token-features-20260916.md).

2026-09-16 manual-only execution: Model scheduling canceled at the user’s request. The new macOS model LaunchAgent was unloaded and archived; the Codex model automation is paused. Existing data collectors are unchanged. Use scripts/run_daily_token.sh manually for data validation, incremental training, forecasts, reference-price ranking, report generation and mature-outcome review. The entry point was checked against the completed September 15 cycle without repeating training; no new September 16 cycle was launched. See [manual operation](docs/daily-token-live-20260916.md).

2026-09-16 return correction: [Published ranking](docs/daily-token-live-20260916.md) now uses sell reference price / buy reference price − 1 − 0.25%, matching the user-confirmed figure. Buy/sell dates and prices are unchanged; all 5,328 stocks are re-ranked, with ST Zhuoran at 12.76% instead of the path-mean 61.27%. Forward scorecards follow the actual published ranking and frozen sell dates, keeping not-yet-mature return outcomes pending. Twenty-eight focused tests passed. Old reports and model artifacts remain immutable.

2026-09-16 report update: [Daily watchlists](docs/daily-token-live-20260916.md) now explicitly show expected net interval return, buy date/reference price and sell reference date/date-specific price in Chinese. Sale-date frequencies preserve equal model weights and exclude T. Fixed-date expected return and reference-price return are separately labeled. All 5,328 ranks and forecast/checkpoint artifacts are unchanged. Daily publication uses the new versioned format; 27 focused tests passed.

2026-09-16 ranking update: [Daily selection](docs/daily-token-live-20260916.md) now ranks mean ideal-timing net potential: T low entry and maximum T+1–T+4 high exit, less the 0.25% cost scenario. Same-day selling and cross-path extrema mixing are excluded. Re-ranked 5,328 stocks from existing paths without training or inference; original artifacts/checkpoints are unchanged. New daily runs and their outcome reviews use the recorded new method. Twenty-six tests and independent verification of all 32,020 model scores and final order passed. Extreme-price potential is not executed profit.

2026-09-16: [Daily token workflow](docs/daily-token-live-20260916.md) is now the operational priority, following the user’s instruction to complete the daily loop. Three predictors each continued on 9,728 mature/replay rows and generated 1,033,920 five-session paths for 5,385 stocks; 5,328 stocks enter the common ranking and top-20 report. The September 15 catch-up report is explicitly late and excluded from forward scoring. Twenty-four tests and independent verification of chronology, 4,041 chunks, 32,020 model scores, 480,300 daily quantiles, rankings and six exact replays passed; a repeated invocation left model/run hashes unchanged. The active heartbeat runs the time-gated operator hourly at minute 35, targeting the first eligible run after 16:30 Shanghai. The first scheduled execution remains pending. Daily watchlists use breadth, magnitude and downside evaluation without unanimous-seed/significance gates. Recent live labels are consumed by design, so old research holdout boundaries no longer describe unseen data for live weights. Existing frozen research artifacts are retained; the earlier experimental next-step suggestions below are historical, not an active queue.

2026-09-15: [Frozen-checkpoint sampling replication](docs/token-sampling-replication-20260915.md) completed. Eighteen new forecast sets (2,801,664 paths) and nine reused sets reproduce seed 43's regression across all three draws. Its mean high/low MAE worsens 4.31% / 7.81% versus current at two/five days; seeds 17 and 29 improve 0.48% / 1.06% and 1.89% / 3.16%. Loss-specific downward center movement persists; a single unlucky draw set does not explain it. Retain the objective direction and other-seed gains, keep the current calibrated reference preferred, and prioritize continuation/checkpoint stability. Independent verification covered 126,550 scores, 1,728 paired estimates, 16,416 chunks and 45 exact replays; nine tests passed. All jobs finished; no training, calibration fits, seed removal, default changes or sealed-holdout use.

2026-09-15: [Seed 43 continuation diagnostic](docs/token-seed43-diagnostic-20260915.md) completed from 27 saved forecast sets. Its 2025 high/low MAE deterioration is mainly center placement (84% / 82% of the two/five-day increase), concentrated in predicted lows. Token-only continuation already worsens MAE 1.90% / 3.89%; midpoint loss adds 2.02% / 3.04%. The full regression spans 56/76 dates and survives removal of the worst five dates or ten-date block. Signed downward movement versus CE was present in both earlier periods, although H1 midpoint scores improved. Retain the loss direction and other-seed gains; keep the current calibrated reference preferred. Independent verification covered 140,433 raw-path score rows, 140,433 attributions and 756 paired estimates; eight tests passed. No training, inference, calibration, seed selection or sealed-holdout use. Next: fixed-checkpoint sampling replication.

2026-09-15: [Midpoint calibration and transfer](docs/token-midpoint-transfer-20260915.md) completed. Retain the midpoint-loss direction and calibration gains; keep the current calibrated reference preferred while diagnosing seed 43. On 76 transfer dates, seeds 17 and 29 improve high/low MAE at both horizons, but seed 43 regresses 3.96% / 7.04%, making pooled MAE worse by 0.64% / 1.15% versus current. Across the two tested periods, high/low MAE improves in 10/12 full-continuation cases and 7/12 loss-specific cases. New calibration raises average high/low coverage from 71.40% / 72.51% to 82.35% / 83.19%, preserves point MAE exactly and improves interval score in 5/6 cases. Thirty-six factors, 12 new forecast sets and 3 reused reference sets completed without neural retraining. All 608 paired estimates, 15 exact replays, 8,160 chunks and nine focused tests passed verification. The sealed holdout and daily defaults remain unchanged.

2026-09-15: [Midpoint-loss comparison](docs/token-midpoint-loss-20260915.md) completed. Retain the auxiliary objective as a small research improvement, strongest at five days: versus equal-budget token-loss continuation, midpoint CRPS changes +0.080% / −0.391% and high/low MAE changes −0.087% / −0.191% at two/five days. Range MAE regresses slightly by +0.107% / +0.044%. Relative to original checkpoints, the midpoint continuation improves high/low MAE 0.590% / 0.623%; extra training contributes much of that gain. Six fits and twelve forecast sets completed. Independent verification covered 55,246 score rows, 7,104 chunks, 12 exact replays, six CPU reloads and 112 primary paired estimates; 28 focused tests passed. Selected weights are retained as an uncalibrated candidate for checkpoint-specific calibration and transfer validation. Current calibrated reference, daily defaults and sealed holdout remain unchanged.

2026-09-15: [Midpoint/width diagnostic](docs/token-range-decomposition-20260915.md) completed. The gap versus historical volatility is concentrated in price-range placement: midpoint MAE is 9.94% / 12.90% higher at two/five days, while implied half-width MAE is 11.32% / 5.84% lower. All three seeds show this pattern. Arithmetic attribution confirms width gains partially offset the midpoint disadvantage. Retain current gains and prioritize a controlled centre-focused learning-objective comparison. Saved paths only; no inference, training or calibration changes. All 23,125 component records, 27,750 attribution rows, 272 estimates and intervals, and eight focused tests passed independent verification.

2026-09-15: [External benchmark](docs/token-external-benchmark-20260915.md) completed on 76 dates in April–July 2025. Self-built raw high/low MAE is 8.07% / 10.25% lower than Kronos Base at two/five days, with raw CRPS 9.28% / 11.46% lower. All three seeds improve on Base in both high/low means. Historical volatility still has lower high/low point errors, while self-built range MAE is 11.84% / 8.78% lower than that baseline. Retain the research profile and its target-specific strengths; diagnose price-centre versus range-width error next. Frozen Kronos Base and historical-volatility baselines each generated 155,648 five-day paths; three self-built forecast sets were reused exactly. Independent path scores, baseline reconstruction, common cohorts, 576 paired estimates and intervals, 256 exact Kronos reload paths and 14 focused tests passed. No model training or calibration fitting; the sealed holdout remains unused.

2026-09-15: [Frozen-profile transfer](docs/token-profile-transfer-20260915.md) completed on 76 dates in April–July 2025. Retain the frozen refreshed/calibrated research profile: high/low MAE improves 1.13% / 2.02% and interval score improves 1.83% / 2.73% at two/five days versus the older calibrated profile. All six target/horizon means improve; seed 43 point regressions and minimum-low interval over-widening remain documented. Mean high/low confidence intervals cross zero, so effect sizes remain uncertain. Calibration yields 81.55% / 82.81% mean coverage without changing point forecasts. Six forecast runs generated 933,888 five-day paths without predictor retraining or calibration refitting. Independent raw-path scores, matched cohorts, 360 paired estimates and confidence intervals, and six exact token/price replays passed; nine focused regression tests passed.

2026-09-15: [Calibration after training refresh](docs/token-refreshed-calibration-20260915.md) completed. Retain refreshed predictors plus checkpoint-specific calibration. Mean high/low coverage becomes 77.38% / 78.01% in 2023 H2 and 79.00% / 79.82% in 2024 H2 for two/five days. Point-MAE gains are exactly preserved. The complete pipeline improves high/low interval score in all 12 seed/window/horizon cases; local 2024 low-price calibration regressions remain documented. Thirty-six checkpoint-specific factors were fitted on the corresponding H1 forecasts. Four new validation forecast runs, two verified token reuses, and six reused evaluation sets required no neural-network retraining. Independent quantiles, factors, score/cohort calculations, 640 paired estimates and confidence intervals passed; nine focused regression tests passed. A checkpoint-bound research profile is retained.

2026-09-15: [Training refresh comparison](docs/token-training-refresh-20260915.md) completed. Retain training refresh: combined high/low MAE improves 1.24% / 1.14% in 2023 H2 and 1.72% / 3.31% in 2024 H2 for two/five days. All 12 seed/window/horizon high-low averages improve; individual high-price regressions are small. Two new fits and four verified checkpoint reuses cover two windows and three seeds. Independent raw-path scoring, 400 paired estimates and confidence intervals, six exact CPU reloads, source hashes and date boundaries passed; 17 focused regression tests passed. Interval calibration remains the preferred research method with checkpoint-specific refitting.

2026-09-15: [Research decision update](docs/token-calibration-decision-update-20260915.md): retain interval calibration as a useful improvement and the preferred research interval method. The earlier strict-gate result is historical evidence, not a rejection of usefulness. The subsequent bounded two-window training-refresh comparison is complete; see the latest entry.

2026-09-15: [Token interval calibration](docs/token-interval-calibration-20260915.md) completed. Fitted 18 date-weighted interval factors on 2023 H1 forecasts from three fixed predictors, then evaluated on 2023 H2 and 2024 H2. evaluation2023h2: two-day coverage 67.36% → 77.76%, interval score -1.89%; evaluation2024h2: two-day coverage 68.15% → 78.37%, interval score -1.29%. Two-day acceptance not met across both windows; five-day acceptance not met. Median forecasts and point MAE are exactly unchanged. Nine focused tests and independent raw-quantile, coefficient, score, date-aggregation and confidence-interval checks passed. This is retrospective interval research; model defaults and the sealed holdout remain unchanged.

2026-09-15: [Fixed-decoder temporal validation](docs/tokenizer-temporal-20260915.md) completed. The exact prior decoder and three predictor checkpoints were evaluated on 119 dates in 2023 H2, with 3,808 inputs per predictor and no refitting. Two/five-day CRPS changes -2.89% / -1.32%; median high/low/range MAE changes -1.38% / -0.17%. Prespecified confirmation conditions: two-day pass, five-day not met. Independent raw-path scores, target-specific intervals, shared-path sensitivity, reloads and lineage checks passed. This is retrospective fixed-weight transfer evidence. The adapted decoder remains a research candidate; model defaults and sealed holdout are unchanged.

2026-09-15: [Tokenizer decoder adaptation](docs/tokenizer-reconstruction-20260915.md) completed. Two decoder objectives and three fresh predictor seeds were evaluated on 120 dates. Price-consistency adaptation improves observed-token reconstruction MAE 7.87% and five-day valid generated paths from 83.15% to 90.90%. Actual forecast CRPS improves 0.82% at two days; the 0.18% five-day change is inconclusive. Median extrema MAE changes -0.20% / +0.25%. A shared-token-path check confirms a modest two-day distribution gain. Retain the adapted decoder as a research candidate; the original control and daily model remain in place. All 31 regression tests, independent raw-score calculations, causality and reload checks passed. All experiment jobs completed; the sealed holdout remains unused.

2026-09-15: [Tokenizer and sampling validation](docs/token-pipeline-audit-20260915.md) completed. Three training seeds over 1,660 training dates and 119 evaluation dates confirm lower distribution error with full-distribution sampling: two-day CRPS -1.86%, five-day -3.48%; median MAE is essentially unchanged at two days and improves 1.12% at five days. All three checkpoint searches retain CE-best. A reusable 64-path sampling preset is saved; tokenizer reconstruction is the next controlled optimization. 25 regression tests, independent path/score/interval checks, three CPU reloads and causal token-prefix verification passed. 3,059,712 generated five-day paths are retained. All experiment processes exited; the sealed holdout remains unused. The daily strategy model is not promoted, and additional feature/capacity experiments remain gated.

2026-09-15 UTC / 2026-09-14 PDT：[扩展历史与输出损失对照](docs/token-history-loss-20260915.md) 已完成。两个窗口覆盖 1,902/2,144 个训练日期，完整运行分别遍历 354,007/400,198 条已知标签样本；完成 6 次训练及 602,112 条五日路径生成。大模型未显示稳定容量优势。递减输出权重使两日高低价/振幅平均 CRPS 在 2024 窗口上升 1.64%、2025 窗口下降 0.04%，五日分别上升 1.97%/0.24%；四项主要差值区间均跨零，继续保留等权研究基线。20 项测试、真实 tokenizer 集成、六模型 CPU 重载及路径重放、36 组独立路径指标复算通过，封存留出集未使用。

2026-09-14 后续：[自建 Decoder 完整训练与容量对照](docs/token-capacity-20260914.md) 已完成。使用同一批 22.86 万条训练输入，372 万/28 万参数版本分别训练 9/39 轮，完成独立 token 评价和每模型约 4.9 万条五日路径生成。大模型分布预测较好，但两者收盘点预测均未胜过持平基线，不晋级每日模型。30 项测试、检查点重载与独立指标复核通过。

2026-09-14：按用户确认，自建神经网络研究版本改为 [Decoder-only 行情 token 生成](docs/token-transformer-20260914.md)，使用 Kronos 粗细 token 机制，从头训练自建主体，生成未来多日六项行情及独立采样路径。新入口为 scripts/token_transformer_run.py，不使用固定买卖计划标签。25 项相关测试和小批真实数据训练/生成/重载验证通过；尚未进行全量效果训练。旧计划模型保留为历史基线，每日模型未替换。

2026-09-14：[数据接入与今日策略](docs/daily-data-and-strategy-20260914.md) 已完成。行情更新至 9 月 11 日，5,361 只股票产生 T 至 T+4 预测，今日模型策略为观察、不新开仓。报告为 10:31 盘中补发研究版；118 项测试和全部输入指标独立重算通过。已提供可复用增量入口，模型建议质量仍未晋级。

当前目标：按 [每日 token 工作流](docs/daily-token-live-20260916.md)，每个交易日收盘后增量训练，预测下一个交易日起 T 至 T+4 的六项行情，输出股票排序与前 20 名观察名单，并在结果成熟后复盘。以多数场景的改善幅度和风险综合评价，不再要求所有场景一致改善才能发布研究观察名单。券商接入与自动交易不在范围。

2026-09-13：[每日研究闭环](docs/daily-research-loop-20260913.md) 已用真实历史跑通预测、条件草案、冻结版本和到期复盘，保留朴素预测与固定规则对照。110 项测试通过，5,317 行检查点输出重现一致。当前快照仅到 9 月 4 日，真实每日前向输入与调度尚待接通；当前日期会明确阻断，模型没有通过建议质量验收。

首批 [三窗口模型质量实验](docs/model-quality-20260913.md) 已完成：分位损失增益跨窗口存在，但覆盖与中位数指标未同时达标。当前不发布盘前建议，继续验证风险预测的稳定性。

2026-09-13 后续：[价格区间校准与 Transformer 对照](docs/price-calibration-transformer-20260913.md) 已完成。小规模 Transformer 未优于全量 LightGBM；固定区间扩宽过度且不改变当前选单，继续保留原始 LightGBM 研究基线。没有启用实盘或恢复旧训练队列。

2026-09-13：已完成 [扩大日期验证与固定订单成本压力](docs/price-stability-20260913.md)。LightGBM 在新增 10 个日期均优于朴素价格基线；Kronos 区间覆盖不足。当前产物仍是历史开发诊断，所有计划不可执行，旧排名队列保持停止。

2026-09-12 当前方向：优先验证未来 5 日价格分布、买入限价、止盈和止损计划。新增独立的小规模 LightGBM 分位数与 Kronos 原生零样本流程，旧排名队列保持停止。实施目标、运行命令和成交诊断边界见 [5 日价格实验](docs/price-pilot-20260912.md)。下文原排序实验作为保留基线，其训练入口不代表当前默认任务。

目标是沪、深、北全部 A 股的横截面排序，分别研究持有 5 和 20 个交易日。股票池由独立全市场证券目录定义，与个人自选列表无关；行情可用性也不能决定池中成员。当前 v2 目录截至 2026-09-07，包含 5,558 个市场成员，完整名单、历史代码候选和采集计划见 [全市场股票池](docs/full-market-pool.md)。

按用户 2026-09-07 的调整，训练、验证和测试样本包含 ST、*ST 及风险状态未知的股票。历史 ST 完整性不再阻挡该训练池的数据验收；证券身份、历史价格、复权、标签和日期隔离要求继续有效。

训练范围与模拟买入范围独立：模拟买入仍按当时已知且已生效的状态排除 ST、*ST 和未知状态。风险状态原样保存为审计元数据，不作为首版模型特征；没有记录不代表正常。证券历史身份需要包含退市证券及代码变更。

当前已建立全市场目录及默认覆盖门槛，完整历史数据仍在准备。此前合成演示和八股真实训练仅用于工程验证，不能作为选股池或证明策略有效。查看 [开发状态](STATUS.md) 和 [长历史接入验收](docs/p1-long-history.md)。

## 环境与运行

Python 3.12–3.13，依赖使用 uv.lock 固定。在本目录执行：

    uv sync --locked --python 3.12
    uv run --locked pytest -q
    uv run --locked ruff check src tests scripts
    uv run --locked quant-research demo --output artifacts/demo-new --epochs 2 --equal-information

macOS 上 LightGBM 需要 libomp。本机已安装 Homebrew libomp 23.1.0。LightGBM 在新解释器进程中运行，避免与 PyTorch 的 OpenMP 运行时发生冲突；不修改全局线程库链接或关闭冲突检查。

演示默认保留 2 层、128 隐藏维度、4 注意力头、60 日窗口，将训练轮数降至 2，并把持仓上限改为 6，以检查排名确实影响选股。它覆盖 18 只虚构股票、400 个虚构交易日和两个目标周期，使用种子 17。正式种子 17/29/43 和最多 8 组参数登记在配置中，但尚未执行真实数据上的完整多种子实验。

## 已实现的接口

CLI 提供 pool、pool-audit、freeze、audit、folds、run、demo、report、event-overlay 和
event-overlay-review。每个接口可以加 --help 查看参数。

    uv run --locked quant-research pool
    uv run --locked quant-research pool-audit /path/to/snapshot --output artifacts/new-pool-audit.json
    uv run --locked quant-research freeze /path/to/canonical /path/to/new-snapshot /path/to/declaration.json
    uv run --locked quant-research audit /path/to/snapshot --output artifacts/audit.json
    uv run --locked quant-research folds /path/to/snapshot --horizon 20 --output artifacts/folds.json
    uv run --locked quant-research run /path/to/snapshot /path/to/fold.json --horizon 5 --output artifacts/new-run --engineering
    uv run --locked quant-research run /path/to/training-snapshot /path/to/fold.json --horizon 5 --output artifacts/new-training-run --engineering --training-only

run 默认必须提供全市场目录，并检查研究日期内每个预期成员及交易日覆盖。配置中的 universe_catalog 相对于配置文件目录解析；当前固定到已保存并核验哈希的完整目录。缺数据的股票留在覆盖台账，默认入口拒绝不完整快照，不从实际下载的股票反推研究池。

run 当前还要求显式 engineering 模式，正式实验仍受其他数据与工程门槛限制。测试夹具必须额外传入 --test-fixture；该选项仅在工程模式有效，并将实验范围标记为 engineering_test_fixture，不能用于宣称完成全市场研究。demo 自动使用该标记。最后 12 个月的保留集不由该入口开放；folds 只生成开发期的季度滚动窗口。

默认配置 training_risk_policy 为 include，训练池不按 ST 状态过滤；改为 exclude_st 可重跑严格剔除对照。缺少此字段的旧配置保留原来的剔除含义。exclude_statuses 和 unknown_status_policy 约束模拟买入，不因训练扩容而放宽。

audit 分列 training_data_blockers 与 backtest_data_blockers：包含 ST 的训练不要求完整 ST 史，非 ST 组合回测仍需完整历史状态。formal_ready 表示整套实验的数据条件，不能用训练数据门槛代替回测验收。其他历史数据缺口和剩余工程条件仍需完成。

--training-only 只训练模型并评估排序，不运行模拟成交或成本压力测试，不产生净值。声明 purpose 为 training 的快照允许 rules 空表，且必须使用该模式；成交引擎拒绝训练专用快照。训练审计不再要求行业与成交规则证据，价格、身份、公司行动与历史数据版本门槛仍保留。

Fold 文件包含 train_start、validation_start、test_start、test_end、purpose 五个字段；purpose 必须为 development。训练和验证样本的真实标签结束日必须早于下一分区起点。

## 数据契约

每个表使用同名 Parquet 或 CSV 文件，字段见 [storage.py](src/quant_research/storage.py)。日期采用 YYYY-MM-DD；known_at 为带时区的真实可用时间。价格为原始人民币价格，volume 为股，amount 为人民币元，factor 为用于相对收益的复权因子。不要使用复权价格撮合。

| 表 | 用途 |
| --- | --- |
| instruments | 历史稳定证券身份、交易所、板块、上市和退市日期；当前 industry 仅供工程测试 |
| calendar | 每个交易所的完整交易日历 |
| bars | OHLC、股数、金额、复权因子、当日有效涨跌停价格 |
| risk_events | 风险状态的生效区间和当时可用时间，包含 normal、ST、*ST、unknown |
| risk_coverage | 对应交易所、日期的状态覆盖是否完整，以及完整性证据可用时间 |
| rules | 按板块和生效区间的手数、T+1、佣金、税费、最低费用及滑点假设 |
| actions | 除权日期、支付日期、每旧股变更后的股数比例、每旧股现金权益 |

声明文件必须显式提供 source、vintage、synthetic，以及 st_history_verified、actions_verified、rules_verified、historical_universe_verified、industry_history_verified。布尔值必须是 JSON 布尔值。声明本身不是核验证据；真实数据验收还需要逐项源文件及独立复核。vintage 为 verified_pit、reconstructed 或 synthetic。

没有 ST 来源时，risk_events 与 risk_coverage 可以是保留字段的空表，st_history_verified 仍填 false；不要伪造 normal 事件、完整覆盖或历史可用时间。包含模式下这类证券可生成训练样本，模拟买入仍会排除。

可选的 source_trade_status 与 source_is_st 字段须为明确 0/1。来源日 ST 标记保留为辅助元数据，不替代事件可用时间或直接许可买入。因子导出必须有不晚于行情日的历史锚点，禁止从未来回填。

导入拒绝重复价格、缺失身份、反向区间、无效 OHLC、错误费用、重叠规则和日历之外的数据。快照对所有表计算 SHA-256，读取时重验，禁止覆盖已有目录。

## 模型与回测

25 个价格、波动率、量额特征全部向后计算。60 日序列最长需要约 120 个原始交易日；停牌缺日会导致特征或序列不完整，并进入剔除账本。5 日标签为 T+1 开盘至 T+6 开盘，20 日标签为 T+1 至 T+21。包含模式下，ST 状态不改变入池、特征或排名标签；未来执行端点缺失仍记录为待处理的删失样本，不填零。

端点如果停牌、无成交量额、缺价格或已经退市，不会因存在沿用价格而获得收益标签。样本保留 entry_status / exit_status，label-coverage.parquet 按年份记录缺口；测试保留评分，Rank IC 仅使用已知结果，并明确披露标签覆盖。未知终值及缺失选择偏差仍是正式验收依赖。

模型包括 20 日动量、LightGBM、共享单股时序 Transformer、两者固定等权的排名融合，以及可选的完整序列展平 LightGBM 对照。Transformer 使用独立初始化的编码层、Huber 排名目标、AdamW 和验证集 Rank IC 早停。

研究回测保存现金、应收分红、持仓、成交及被阻止的订单。ST 生效后发起退出；停牌、跌停或结算限制导致无法退出时继续计入持仓和损益。买入检查次日开盘时的风险状态，使用前 20 日成交额约束容量，考虑手数、佣金、卖出税和滑点。缺失限价字段会阻止成交。

日线开盘成交是研究假设，不能证明排队成交。停牌使用最后有效估值并报告陈旧估值天数；退市结算、复杂公司行动、历史行业暴露和基准尚需补齐，不能据此声称可实盘。费用及行业配置中的数值是工程默认参数，未认证为跨年份真实交易规则。

## 实验产物

每次实验保存数据版本、配置、代码内容及哈希、锁文件哈希、依赖版本、模型检查点、归一化参数、数据过滤明细、分区样本、逐日预测、训练日志、模拟账本和报告。检查点只由验证集选择。

样本和预测保留 risk_status 与 trading_eligible；检查点和实验身份保存 training_risk_policy。报告分列训练池 Rank IC、非 ST 信号子集 Rank IC，以及受限模拟组合收益，保存各分区的风险状态样本数。未知状态样本不混称非 ST，训练评分也不表示可以买入。

已实现两倍成本、延迟一天的压力测试，以及成对日 Rank IC 差异的 20 日区块自助法。历史不足时不给置信区间。最终保留集、多种子滚动比较、显著性和容量验收仍需真实数据执行。

artifacts 和 data 默认不提交，包含完整本地实验文件。报告是离线 HTML，可由 report 命令重新渲染并核验原始实验文件完整性。

## Rust 数据边界

生产采集和金融解析仍属于 /Users/guo/github/mootdx-cf。本目录 Python 仅做独立研究、夹具与只读数据核验，不写入现有数据湖。

新增 Rust 命令在 mootdx-cf 目录执行：

    cargo run --locked -- research-source-capture namechange /path/to/params.json /path/to/new-capture /path/to/env-file
    cargo run --locked -- research-audit-names /path/to/response.json /path/to/new-audit.json
    cargo run --locked -- research-normalize-tushare /path/to/capture /path/to/new-normalized
    cargo run --locked -- research-capture-baostock /path/to/query.json /path/to/new-capture
    cargo run --locked -- research-export-baostock-training /path/to/input-manifest.json /path/to/new-canonical
    cargo run --locked -- research-build-universe /path/to/current-market-capture /path/to/delisted-capture /path/to/baostock-master-capture 2015-09-01 2026-09-04 /path/to/new-universe
    cargo run --locked -- research-backfill-init /path/to/universe/manifest.json /path/to/new-campaign
    cargo run --locked -- research-backfill-run /path/to/campaign 600 500 /path/to/env-file
    cargo run --locked -- research-backfill-inspect /path/to/campaign
    cargo run --locked -- research-backfill-retry /path/to/campaign JOB-ID '写明核验后的失败原因和新尝试依据'
    cargo run --locked -- research-backfill-import /path/to/new-campaign /path/to/previous-campaign

capture 只支持明确列出的研究数据接口，每次单次有界请求，保存原始响应及安全请求元数据，保留拒绝响应，不自动重试。参数示例在 configs/capture-*.json。遵守账号实际限额，不能循环强刷；环境文件和请求 token 不进入日志或产物。

历史名称仅输出候选风险状态，不自动认证历史完整性；公告只有日期时不伪造盘前或收盘前可用时间。北交所改码通过历史身份表另行关联。a-stock-data 保持只读。

BaoStock 目前支持证券资料、日历、逐股不复权日线和复权事件的有界分页采集。代码、日期、字段、分页与原始响应哈希均校验，导出重放原始页；原始文件和失败目录不覆盖。明确停牌且所有量价为空的来源记录保留在审计中，不生成虚假日线。configs/real-pilot-export.json 仅供复现八股工程测试，不能定义研究范围。

全市场队列已生成 11,624 个采集任务，覆盖区间内全部 5,812 条历史代码候选，支持限速、断点续采、按接口暂停和保留原始证据的审查后重试。新目录可导入请求范围完全相同的旧采集；失败记录和来源接口暂停一并保留，不把升级版本当成重新请求的理由。沪深使用 BaoStock，北交所日线使用 Tushare；北交所因子受到账号频率限制，日历和代码映射仍待补齐。详见 [全市场历史数据采集](docs/full-market-backfill.md)。

## 云端采集与恢复

全市场历史任务由 Cloudflare Worker 和 Rust Container 执行，原始数据进入 R2，进度与入库回执进入 D1，来源数据进入 Parquet/Iceberg 数仓。退出本机后云端继续采集。本地数据是可恢复副本；最新进度、凭据配置和换机命令见 [云端采集与换机恢复](docs/cloud-research.md)。来源数据的持久化不代表完整训练质量验收已通过。

当前采用双路连续采集和独立入库，共享来源限速并复用 BaoStock 会话。80 项云端实测全部成功，双路实测速率约 11.39 项/分钟。详见 [优化结果与性能进度命令](docs/cloud-optimization-20260907.md)。
