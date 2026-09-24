# 开发状态

2026-09-22 matched early-stage test: Outcome-blind matching, frozen sample/balance gates, independent replay verification, compact daily cohort retention and T+4 maturation are implemented. The original historical run is data-blocked because 144 of 147 required stock-level files were pruned; date-level summaries are insufficient for matching, so no effect conclusion or auxiliary-head training was produced. The manual workflow remains unscheduled. All 314 tests passed with two optional skips; a 10,000-row synthetic end-to-end run and independent replay passed. See [implementation and evidence](docs/token-early-stage-matched-20260922.md).

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

2026-09-14 后续：[完整开发窗口容量对照](docs/token-capacity-20260914.md) 已完成。372 万和 28 万参数 Decoder 共同使用 228,553 条训练输入（43 日期），其中 227,199 条有已知未来标签；分别训练 9/39 轮并早停，选择第 3/33 轮。完成全部 42,863 条评价输入的 token 评价，以及每模型 49,088 条五日路径生成。大模型在共同样本上的第五日收盘分布 CRPS 比小模型低 11.68%，但两者收盘中位数 MAE 均高于持平基线；token 误差主要比较的区间跨零，未证明容量带来稳定优势。30 项测试、两个模型重载及独立指标复核通过。另发现月份 embedding 有未见类别，限制解释范围。全部阶段正常退出，旧计划收益优化队列保持停止，每日模型未替换。

2026-09-14：按随后确认的 token 生成改造要求，完成自建 Decoder-only、粗细 token 预测头及新的训练/多路径生成入口。25 项相关测试通过，并以 32 行真实训练样本做两轮工程验证，完成六项行情输出及独立重载核验。全量训练与模型质量评价未启动，此前停止的优化队列保持停止；每日模型未替换。详见 [token 生成改造](docs/token-transformer-20260914.md)。

2026-09-14 18:24 PDT：按用户要求停止所有模型训练。本轮优化已完成 6 次训练，第 7 次（种子 43 的参考模型）中途停止；训练队列与等待中的评价/验证启动器均已退出，未继续运行剩余任务。已有检查点和日志保留，实验标记为用户停止，不作为完整验证结果。未经用户重新指示不得恢复。见 [训练诊断记录](docs/transformer-optimization-20260914.md)。

2026-09-14：[新增信息 Transformer 对照](docs/context-transformer-20260914.md) 已完成。60 日行情序列加信号日 11 项上下文特征，三个匹配窗口使用 MPS GPU 各训练 8 轮，均按选择集早停并选择第 1 轮。原始条件收益 MSE 比原 Transformer 高 0.3481%，区块区间跨零；校准后与原 Transformer 基本相同，比同信息 LightGBM 高 6.1549%。成交概率改善未转化为收益预测优势，不晋级每日模型。8 项相关回归、CPU 检查点重载、全部校准/选单/成本独立复核通过；本轮没有微调 Kronos，训练进程均已结束。

2026-09-14：[免费信息特征接入与增量实验](docs/free-feature-implementation-20260914.md) 已完成。Rust 按日分页取得 371 日期、174.6 万条来源记录；新增 11 项特征，三窗口 24 日期对照与全部特征/校准/选单独立复核通过。原始条件收益 MSE 仅改善 0.0016%，区块区间跨零；校准后双倍成本选单情景三个窗口均负，不晋级每日模型。新浪辅助核对通过，暂不需购买 Tushare；原始披露时点、财务修订和退市缺失模式仍是限制。

2026-09-14：[数据接入与今日策略](docs/daily-data-and-strategy-20260914.md) 已完成。行情更新至 9 月 11 日，5,361 只股票产生 T 至 T+4 预测，今日模型策略为观察、不新开仓。报告为 10:31 盘中补发研究版；118 项测试和全部输入指标独立重算通过。已提供可复用增量入口，模型建议质量仍未晋级。

2026-09-13 每日研究闭环实施：LightGBM 注册、T 至 T+4 推理、条件草案、不可覆盖版本、部分与完整到期复盘已跑通，并保留朴素预测及固定规则对照。真实回放覆盖 2025-03-04/05；110 项完整测试、最后 7 项相关回归、5,317 行检查点重现和 14 条价格指标独立重算通过。当前源数据仅到 2026-09-04，真实前向草案按时点与数据检查阻断；未启用每日调度。模型建议质量没有晋级，后续先接增量输入，再继续有限模型与决策比较。见 [实施与验收](docs/daily-research-loop-20260913.md)。

2026-09-13 模型效果路线首批实施完成：已冻结 G0 协议，通过 G1 三窗口结构审计，并运行 G2 四方案对照，覆盖 2022/2023/2025 三季度、36 日期、180,008 条输入。原始 LightGBM 相对两类简单基线改善约 3.1%–3.2%，但 2022 区间覆盖未过门槛；标准化目标改善覆盖却损伤中位数准确性。两者均未晋级决策研究。独立复核识别 4 对标签重叠并完成敏感性检查，门槛判断仍对日期选择敏感。102 项完整测试及最后 9 项针对性回归通过，551 文件哈希核验、90 行模型重现通过。见 [模型质量第一批结果](docs/model-quality-20260913.md)；本轮已结束，报告交付后移，旧训练队列不恢复。

2026-09-13 目标与路线图更新：最终交付为每日盘前、供人工决策的 T 至 T+4 五日策略。已制定 P0 报告契约 → P1 盘前数据包 → P2 条件策略 → P3 每日报告 → P4 持续复盘路线，详见 [盘前策略路线图](docs/premarket-strategy-roadmap.md)。现有五步预测可映射到该周期；本次仅更新文档，尚未启动这些实施阶段、训练或定时任务。

2026-09-13 后续对照完成：新增 5 日价格分位数 Transformer，4 轮、45,180 参数；前 6 校准日期拟合区间、后 6 日期校准买卖计划。Transformer 全量损失比 LightGBM 高 7.4%，共同小样本略好；全局区间扩宽导致覆盖过高、损失变差，且未改变选单，继续保留原始 LightGBM 研究基线。94 项测试、203 文件哈希及检查点重现通过；最终产物 price-calibration-transformer-20260913-v2。所有进程已退出，留出集封存，计划不可执行。见 [校准与 Transformer 结果](docs/price-calibration-transformer-20260913.md)。

2026-09-13 扩大 5 日模型验证已完成：评估 12 日期、64,092 条输入，Kronos 校准及评估共 828 个窗口。新增 10 个价格评估日期中，LightGBM 全量分位损失比朴素基线低约 2.8%，10/10 日期改善；Kronos 共同样本未显示优势，80% 区间覆盖仅 45.1%。共同校准与固定订单两倍成本压力均已完成。89 项测试及 143 个文件哈希核验通过，原 60 个 LightGBM 模型逐字节不变，最终留出集封存，所有计划不可执行。见 [扩大验证结果](docs/price-stability-20260913.md)。最终报告位于 price-stability-20260913-v1/stability-v2；本轮已结束，旧队列保持停止。

2026-09-12 新目标已完成首轮快速验证：5 日 OHLC 分位数、买入限价/止盈/止损/有效期及观望，LightGBM 使用 121,624 条训练输入，Kronos 使用原始权重零样本预测并保留各条路径。86 项测试及 115 文件哈希核验通过，全部试验进程已退出。最终产物为 price-pilot-20260912-v4，详见 [报告](artifacts/price-pilot-20260912-v4/report.md) 和 [目标、运行与数据边界](docs/price-pilot-20260912.md)。旧 Kronos 排名队列保持停止，旧 LightGBM/Transformer 结果保留。新入口仅输出历史价格与假设成交诊断，训练专用数据不能产生可执行计划或组合净值。

2026-09-11 近似研究处理：按用户允许部分不严谨数据的要求，220 个遗留价格边界全部给出分级方案，研究未处理数为 0。136 个跨供应方容差支持、4 个公告参考公式支持、20 个分配公式容差支持、59 个保留来源并标记证据不足或冲突，1 个估算缺失因子。新候选 717 个分区及全部 11,463,243 行已独立核验；严格待核状态和股东权益标签隔离仍保留，当前训练未切换输入。源码及 198 文件的增量包已存 R2，并从云端恢复的父数据实际重建全部 729 个候选文件，逐字节一致。见 [近似研究版本与逐项处理](docs/research-estimates-20260911.md)。

2026-09-10 异常验收标准更新：允许公告之外的原始行情与跨供应方因子证据。现有证据新增核验 18 个价格边界，待核 238 → 220，累计闭合 2,806 / 3,026；旧结论无退回，股东权益标签仍独立核验。新审计尚未替换运行中的冻结训练快照。见 [按可复核结果验收异常](docs/market-reference-evidence-20260910.md)。

2026-09-10 全市场开发训练已启动：5,901 条候选身份保留，5,812 条区间内历史身份覆盖无未解释缺口；5/20 日各 10,747,771 条评分样本完成独立核验。LightGBM 首轮已完成，Transformer 使用 5,117,608 条训练样本在本地 MPS 上训练。238 个价格边界和 2,578 个股东权益标签屏障继续隔离，最终留出区间封存，正式评价门槛未开放。详情见 [开发训练准入与补证结果](docs/full-market-training-entry-20260910.md)。


2026-09-10 补证第二轮：新增闭合 39 个，累计 2,788/3,026，剩余 238 个价格边界；新增 41 份 PDF 均独立复验。修复股东现金与除息参考金额混用，导出全部 5,901 条历史证券候选、11,459,092 条有效价格并逐列核验。股东权益标签另保留 2,578 个屏障，不以价格连续性代替权益证明。正式训练尚未启动，继续核验日历、历史范围和标签准入。详见 [补证与训练准备](docs/training-readiness-supplement-20260910.md)。

2026-09-10 公司行动补证进展：原 3,026 个待核边界已解决 2,749 个（90.85%），剩余 277 个继续隔离。3,293 份公告 PDF、8,603 个原始文件和全部 11,463,243 行候选均已独立核验；六组因子晚记和 19 个异常水平重置已修复。247 项 Rust 测试通过。源码及完整补证包 R2 恢复验证完成，15,453 个文件独立逐一对比原件一致。最终归档使用八路直连、4 MiB 分块与压缩检查点。正式训练门槛保持关闭。见 [补证结果与剩余清单](docs/corporate-action-supplement-20260910.md)。
2026-09-10T04:15:29.354452Z：公司行动核验第一轮完成。302132 改码日因子 1 已按旧代码已有因子修复为 5.975678；修正每股分红单位和重复计算送股的问题。免费采集 37,721 条分派，34,493 个事件可解释，3,026 个待核事件边界已切断特征序列与跨界收益标签。11,463,243 行原始字段独立核验一致，正式训练门槛保持关闭。234 项 Rust 与 49 项 Python 测试通过。数据与研究工程 2,106 个文件、源码 313 个文件已持久保存到 R2，并在新目录恢复后逐文件独立验证一致。见 [公司行动核验与剩余事项](docs/corporate-action-audit-20260910.md)。

2026-09-10T03:14:02.923091Z：因子锚点与参考缺口修复完成并通过全量独立验证。55 只新股补入上市日初始因子，修复 78 行；三组改码按生效日期去重，4,017 条新代码回填历史不再误报缺锚点，合法旧代码历史保留。有效候选缺锚点为 0；6,312 个原待分类缺口已核为 3,875 个旧代码区间、2,396 个停牌日、41 个摘牌端点，未分类为 0。全部 11,463,243 行源字段与原快照一致。另隔离 302132 改码日异常因子 1 行，各代码序列暂不跨界拼接；正式训练门槛仍关闭。见 [修复与验收记录](docs/quality-repair-20260910.md)。 新候选与证据共 1,399 个文件、源码 298 个文件已存 R2，并在新目录恢复、逐文件独立验证一致。

2026-09-10 00:47 UTC：全量入仓已完成并通过独立数仓对账，11,623 个已发布任务、11,528,589 行，另 1 个已核验历史区间不适用。已进入全量质量验收与训练数据准备：固定快照扫描、717 个分区 Parquet、全部行与哈希独立验证完成；20,087 条旧样本逐字段一致。发现 4,095 条缺因子锚点，98.1% 集中在 302132、001914、001872；剔除已核验不适用区间后有 6,312 个参考日历缺口待分类。ST/未知状态保留，正式训练门槛仍关闭；755 个产物文件与 291 个源码文件均已存 R2 并恢复核验。详见 [全市场验收与候选导出](docs/full-market-quality-20260910.md)。

2026-09-09 10:29 UTC：已部署 v8 公告与历史代码异常修复，本轮 113 个原异常已全部处理：112 个成功重采并入仓，共 681 行；600849 为经核验的历史代码窗口不适用，0 行且无虚假发布回执。110 个因子异常样本、373 次事件通过离线复核，误差阈值未放宽；600656 原异常行保留源字段并明确标为不可训练。全部 11,624 项身份及 9,732 项原发布回执保留，原失败证据和尝试次数未清空。两次东方财富瞬时连接失败均自动重试成功，云端已继续采集正常队列，累计 9,931 项、10,600,994 行入仓，active。源码已备份并恢复验证；本轮异常关闭不代表全市场采集或训练验收完成。详见 [公告与历史代码异常修复](docs/cloud-anomaly-fixes-20260909.md)。

10:31 UTC 最终实际数仓对账：上述 112 项共 681 行与 D1 一致，无重复，600849 无数仓记录；原完整采集计划哈希一致。最终原异常数仓核验与全范围身份核验均通过。

2026-09-09 02:40 UTC：已部署 v7 免费北交所因子链路，新浪因子由东方财富已实施分红送转和已入库日线逐次核验。三个 CF 样本 16 次事件通过，三个正式任务 19 条因子记录已入仓，实际 R2 SQL 核验重复零；346 项原受限任务已转入免费采集队列，单路持续运行。全量 11,624 项身份、8,307 项原采集回执、8,305 项原入仓回执及旧失败证据保留；源码与证据持久化到 R2。旧代码缺失、覆盖不足或因子不符会单独待复核，完整训练门槛仍关闭。详见 [免费因子接入与验收](docs/cloud-free-factors-20260909.md)。

2026-09-08 22:19 UTC：每小时巡检发现 D1 HTTP 429 阻断已提交数据的入仓回执，已部署 v6 请求节流和拒绝重试。通过原始归档与精确 Iceberg 提交核对后，3 项积压共 4,134 行已处理；10 项新采集全部成功。13 项共 14,774 行经实际 R2 SQL 核验、重复零；全部 11,624 项身份、7,114 项原采集回执、7,111 项原发布回执与 349 项既有失败/限制保留。已恢复单路持续采集，源码已备份 R2，完整训练门槛继续关闭。详见 [D1 限流与数仓回执恢复](docs/cloud-d1-rate-limit-20260908.md)。

2026-09-08 13:22 UTC：每小时巡检发现 D1 响应解码故障造成暂停，已修复并部署 v5 控制面结果核对。核验卡住任务没有实际来源尝试、采集原件或数仓记录后，已通过受控恢复释放原认领，并保存原错误和 R2 凭据。10 项云端小批量全部成功，新增 10,096 行经实际 R2 SQL 对账且无重复；全部 11,624 项身份、4,733 项原成功回执和 349 项原失败/限制保留。已恢复单路不限量采集，完整训练门槛继续关闭。详见 [D1 响应故障恢复与验收](docs/cloud-control-recovery-20260908.md)。

2026-09-08 09:33 UTC：Iceberg 临时错误导致整体暂停的问题已修复并部署，恢复单路持续采集。数仓发布按固定批次持久重试，重放前核对已有提交，重试等待期间采集可继续到待入仓上限。此前积压 4 项、5,378 行已补入；新增 10 项小批量全部成功，14 项共 18,833 行通过实际 R2 SQL 对账且无重复。全部原采集身份、4,072 项原采集回执、4,068 项原入仓回执和 349 项失败/权限限制保持完整。后续新任务已继续入仓；源码及恢复证据存于 R2，完整训练门槛仍关闭。详见 [数仓重试与云端验收](docs/cloud-warehouse-retry-20260908.md)。

2026-09-08 02:16 UTC：云端自动恢复与多源补缺已部署，并恢复单路持续采集。瞬时失败按任务延迟重试，连续失败只冷却对应来源接口，由定时小查询探测恢复；已验收的日线备用来源自动补缺。Tushare 沪深、东方财富沪市与北交所备用已启用；东方财富深市因旧代码返回不匹配保持关闭，因子不混用备用日线。20 次云端小批量全部成功，新增 32,621 行通过独立 R2 SQL 对账且无重复；小批量结束时累计 3,158 项、3,716,961 行入仓。16 项已知网络失败已补采，346 项因子权限受限与 2 项待核查失败保留，全部原计划身份及旧失败原件保持可追溯。持续运行中的最新进度以 D1 为准，完整训练门槛仍关闭。详见 [自动恢复、多源资格与验收证据](docs/cloud-resilience-20260908.md)。

2026-09-07。全市场目录已刷新为 5,558 个成员：沪市 2,317、深市 2,899、北交所 342。新主表补上旧目录缺少的 4 个代码，并补齐所有名称。个人自选列表不参与范围定义；停牌、ST 和行情缺失不会删除成员。详见 [完整股票池与覆盖验收](docs/full-market-pool.md)。

全市场历史采集已迁移到 Cloudflare。历史代码候选 5,901 条，其中 5,812 条与研究区间相交，形成 11,624 个行情和因子任务。切换时完整保留 1,481 个成功任务、1,732,701 条原始日线/停牌记录、11,596 条 BaoStock 因子事件；9,796 个待采任务由云端续采，346 个受限接口任务和 1 个失败任务保留。初始研究树 9,668 个文件已上传 R2 并在新目录恢复验收。最新进度读取 D1；代码身份、全量质量及北交所日历/因子尚未验收，默认全市场训练门槛继续关闭。详见 [云端采集与换机恢复](docs/cloud-research.md)。

2026-09-07 08:05 UTC 优化验收：连续采集、独立入库和双路连接复用已部署。40 项单路与 40 项双路实测全部成功；吞吐量从旧流程约 3.97 项/分钟提高到双路 11.39 项/分钟。全部计划与既有状态保留，数仓 1,260 项、1,469,281 行与 D1 对账一致，无重复记录。累计成功采集 1,790 项，346 项接口受限和 2 项既有失败继续保留。正式采用双路持续运行；[实测和最新进度命令](docs/cloud-optimization-20260907.md)。

2026-09-07 10:11 UTC 连接故障核验后恢复为单路持续运行。原 Cloudflare 容器两次低频登录和小查询成功，单路 10 项历史采集验证全部成功，新增 6,959 条记录；13 项失败和 346 项接口受限原件与状态保留。新增受控探测入口及 R2 原件归档，未实现自动故障恢复。此前 14 小时推算不能作为当前单路的完成承诺。详见 [来源连接核验与恢复](docs/cloud-source-recovery-20260907.md)。

10:35 UTC 持续运行复验：累计成功采集 2,779 项，2,777 项已入库，共 3,211,169 行；期间新增 1 个来源响应失败，失败总数 14，采集仍 active。10 项恢复验证任务的 6,959 行全部通过实际 R2 SQL 对账，原有 359 条失败或受限记录逐条保持原状。

| 阶段 | 已完成 | 待完成 |
| --- | --- | --- |
| P0 | 全量上市主表更新、北交所日线可用性确认、Sina 因子与 Tushare 差异留痕，沪深来源核对 | 历史规则、北交所日历/因子口径、改码；完整 ST 时点证据仍是非 ST 回测依赖 |
| P1 | 5,558 个市场成员、5,901 条历史代码候选；11,624 项采集计划；Cloudflare 续采、R2 恢复验收、D1 持久进度；原始证据重放、跨版本复用；按接口冷却、探测自动恢复、任务重试及验收后日线多源补缺 | 历史身份、公司行动、行业和正式训练输入验收；全量采集、固定快照审计与候选分区导出已完成 |
| P2 | 25 个向后特征、60 日序列、5/20 日标签、日期隔离；核对 2,675 天价格；停牌/缺失/退市端点分别留痕，保留测试评分样本 | 复权逐事件和全范围对账、未解决终值与删失偏差验收 |
| P3 | 动量、LightGBM、序列展平对照；现金/费用/手数/限价/停牌/T+1/ST/公司行动账本 | 历史行业、基准、退市结算、复杂公司行动和真实参数验收 |
| P4 | 小型 Transformer、Huber 训练、验证早停、检查点、排名融合；两个周期真实小样本工程运行 | 全市场真实数据的多种子训练与参数比较 |
| P5 | 成本翻倍、延迟一天、成对 Rank IC 区块自助法、保留集隔离代码 | 多年滚动、多种子、市场阶段、保留集和容量验收 |
| P6 | 本地报告、完整实验产物与版本溯源 | 每日预测、幂等模拟状态和仪表盘集成 |
| P7 | 尚未开始 | 至少 40 个真实交易日，不能用合成日期代替 |

## 云端运行验收

2026-09-07 06:54 UTC 独立复验：云端状态为 active；已采集 1,511 项，待采 9,766 项，346 项限额暂停、1 项历史失败保持原状。60 个任务已取得数仓提交回执，合计 67,913 条来源记录。首批 20 个任务的 21,002 行已用 R2 SQL 对账，重复记录检查为零。另一个全新目录成功恢复全部初始文件和 20 个云端新增任务，Rust 重放审计与独立 Python 哈希、查询范围、行数检查通过。

- [云端独立验收快照（06:54 UTC）](artifacts/cloud-migration-20260907-v1/verification-cloud-running-v1.json)
- [收尾实时状态（06:57 UTC）](artifacts/cloud-migration-20260907-v1/final-cloud-status-v1.json)
- [恢复与进度操作说明](docs/cloud-research.md)
- [换机恢复独立复验](artifacts/cloud-migration-20260907-v1/recovery-verification-v1.json)

## 验证

- 云端优化通过 193 项 Rust 测试、6 项独立 SQLite 合约验证、3 项 Worker 调度测试、Clippy、格式、生成绑定与 TypeScript 检查；新增来源样本完成 R2 恢复与 Rust 原件回放，优化源码已单独备份并恢复验证。
- 云端迁移新增代码完成目标 Rust 包 196 项测试，之后的归档与代理调整通过相关回归、Clippy 和构建；研究 Worker 绑定生成和 TypeScript 检查通过。
- Python 47 项测试、Ruff 检查通过；新增全市场目录校验、禁止自选范围配置、缺证券/缺价格仍保留预期成员、默认入口拒绝小样本的回归。
- 此前数据管线 Rust 目标包 158 项单元/集成测试、仓库格式和 Clippy 检查通过；新增全市场任务规划、上市主表适配、进程锁、续采/暂停/重试、Tushare 日线单位及无价格停牌证据回归。此前阶段 Cargo.lock 未改动；云端迁移新增直接引用锁文件中已有的 axum、tar、aws-smithy-http-client，没有升级版本，rsproxy 配置保持不变。
- v2 名单与历史候选均检查唯一性和文件哈希，原 v1 的六只停牌证券仍保留。八股快照相对 v2 预期范围缺少 5,804 条证券记录，默认入口继续拒绝。
- BaoStock Rust 结果与官方 SDK 日线、日历、复权事件逐字段一致；证券主表与原始消息一致，保留 SDK 会删除的名称空格。4,022 个日历日与 Tushare SSE 无差异，2,675 天招商银行 OHLC 完全一致；成交额与因子的小幅差异明确留存。
- 两个真实实验的全部文件哈希、25 个检查点特征及 ST 入样已复验；篡改原始页被拒绝且不创建规范导出，已有导出目录拒绝覆盖。
- 真实训练/验证/测试的来源 ST 标记：5 日为 158/271/58 条，20 日为 143/256/58 条；20 日测试中有 1 条未解决终点，仍保留评分并披露，未填收益。
- 此前合成数据的 5/20 日实验均完成快照、特征、训练、五组预测对照、成本/延迟压力测试、账本和 HTML 报告。
- LightGBM 改为独立进程，解决本机与 PyTorch 共用线程运行时产生的等待问题；测试检查子进程未载入 PyTorch。
- 合成演示在 18 只虚构股票中最多选择 6 只，验证评分能改变持仓。正式配置仍为全部 A 股、最多 50 只、单只 2% 的研究默认值。
- 此前合成 ST 回归：5/20 日均完成两轮训练、五组对照及回测；训练分区各含 10 条 ST、10 条 *ST、30 条未知状态样本，验证和测试分区也覆盖四种状态。ST 标记不进入 25 个模型特征。

## 产物

- [完整选股池：5,558 个成员](data/universes/full-a-share-20260907-v2/selection-members.csv)
- [历史代码候选：5,901 条](data/universes/full-a-share-20260907-v2/historical-candidates.csv)
- [逐证券历史采集计划](data/universes/full-a-share-20260907-v2/acquisition-plan.json)
- [全市场实际采集任务](data/backfills/full-a-share-20260907-v2/plan.json)
- [采集、限额和失败恢复说明](docs/full-market-backfill.md)
- [采集状态与原始文件独立复验](artifacts/full-backfill-verification-20260907-v3.json)
- [全体成员的采集缺口 CSV](artifacts/full-backfill-verification-20260907-v3.csv)
- [切换前的本地采集日志（已停止）](artifacts/full-backfill-runtime-20260907-v1/runner.log)
- [完整股票池的来源和验收说明](docs/full-market-pool.md)
- [已有工程数据相对全池的缺口台账](artifacts/full-pool-vs-engineering-sample-20260907-v2.csv)
- [目录与默认入口拒绝小样本的验证](artifacts/full-pool-verification-20260907-v2.json)
- [本轮逐项解决记录](docs/p1-long-history.md)
- [真实 5 日训练工程报告](artifacts/p1-real-pilot-training-20260907-v2/h5-seed17/report.html)
- [真实 20 日训练工程报告](artifacts/p1-real-pilot-training-20260907-v2/h20-seed17/report.html)
- [长历史跨来源对账](artifacts/p1-long-history-crosscheck-20260907.json)
- [真实实验与 ST 入样复验](artifacts/p1-real-pilot-training-20260907-v2/verification-status.json)
- [原始页篡改与覆盖拒绝验证](artifacts/p1-integrity-negatives-20260907/verification.json)
- [使用说明](README.md)
- [P0 数据与 ST 核验](docs/p0-data-audit.md)
- [5 日工程报告：训练包含 ST](artifacts/st-inclusion-20260907-v2/h5-seed17/report.html)
- [20 日工程报告：训练包含 ST](artifacts/st-inclusion-20260907-v2/h20-seed17/report.html)
- [ST 纳入验证记录](artifacts/st-inclusion-20260907-v2/verification-status.json)

最新真实数据报告来自 p1-real-pilot-training-20260907-v2，8 只股票、单种子、单 epoch，仅训练和排序评估，没有生成模拟净值。真实日线包含 1,126 条来源 ST 标记和 416 条停牌记录；来源日标记没有被伪装成已核验的历史公告时刻。较早 v1 分区没有 ST 标记，v2 为覆盖该情形调整工程日期；原记录保留。

带净值的 st-inclusion-20260907-v2 等较早报告仍为合成数据，不能判断真实 A 股上的有效性。旧实验、失败采集与失败夹具记录均保留，未用新口径改写。

代码尚未提交或推送。已部署独立的 mootdx-cf-research Worker/Container，新增研究专用 D1 表和 R2 归档；现有行情服务与表未改动，没有启动交易。a-stock-data 未改动。本机安装了运行 LightGBM 所需的 libomp；研究虚拟环境、原始探测和实验产物默认不提交。

用户已允许 ST 纳入训练。当前默认训练、验证和排序测试包含 ST、*ST 与未知状态，已有状态只作为审计元数据；不根据今天的名称改写历史。后续工作按完整采集计划补齐全市场数据，再处理北交所及其他改码身份、复权和公司行动对账；不再通过手选股票扩展研究范围。全 A 股、5/20 日目标继续有效。

模拟买入保持已确认的非 ST 规则，未知状态也不买入。训练快照可不带成交规则；这种快照即使在工程模式也禁止进入组合模拟。训练审计不再依赖历史行业与成交费率，但仍保留证券身份、公司行动、历史版本和完整性门槛；正式实验入口未放开。

补充证据：通过 AKShare 使用的深交所上游，已取得 7,472 条带日期简称变更记录，深市 A 股候选 7,322 条；新浪曾用名列表也可访问但不带日期。候选重建与 ST 区间核验保留为辅助任务，沪市、北交所和全市场完整性仍待补齐，详见 P0 审计新增章节。
