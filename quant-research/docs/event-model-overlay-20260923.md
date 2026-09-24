# 事件雷达与五日模型联合排序

日期：2026-09-23

## 实施结果

本轮完成事件雷达与每日 token 模型的第一阶段结合。模型和检查点保持不变，事件信息只在模型完成五日价格路径预测后作为独立覆盖层参与筛选和重排。

系统同时冻结三组结果：

1. control-ranking.csv：完整纯模型对照组，不受事件数据影响。
2. event-overlay-ranking.csv：预测窗口内、通过事件门槛的增强组。
3. confirmed-event-ranking.csv：已经发生并通过首个完整收盘确认的事件组。

另保存 event-diagnostics.csv、metadata.json、report.md 和 manifest.json。所有输入哈希、配置、决策时间、剔除原因和输出文件哈希都可审计。

## 排名方法

事件增强分由三个零到一的分数组成：

    联合分 = 0.70 × 模型收益分位 + 0.20 × 事件分 + 0.10 × 风险质量

风险质量由收益分布下沿和模型一致性各占一半。模型收益分位在完整模型股票池上计算，事件分使用雷达的零到一百分标准分。初始权重保存在 configs/event-overlay-v1.json，只是影子研究起点，不代表已经验证的最优参数。

## 时间与防泄漏规则

- 模型信号日必须早于全部预测交易日。
- 事件覆盖层必须在第一个预测交易日上海时间九点十五分以前冻结。
- 每条事件的 available_at 必须不晚于覆盖层决策时间。
- 未来事件只在其日历日期落入五日预测窗口时参与排名；周末事件映射到其后的第一个预测交易日。
- 已完成事件只在信号日前七个日历日以内保留，而且必须通过事件雷达的事后价格确认。
- cancelled、postponed、avoid、评分不完整、价格数据陈旧、过度延伸、收益风险比不足等情况不能进入事件增强组。
- 事件后确认不能反向用于事件前排名。

## 运行顺序

先按事件雷达文档生成带同名元数据文件的 CSV，再对已经冻结的每日模型运行目录执行：

    uv run --locked quant-research event-overlay \
      artifacts/daily-token-live-v2/runs/YYYY-MM-DD \
      artifacts/event-radar-YYYY-MM-DD.csv \
      --overlay-config configs/event-overlay-v1.json \
      --output artifacts/event-overlay-YYYY-MM-DD-v1

输入要求：

- 模型运行目录包含 run.json 和 ranking.csv。
- 雷达 CSV 旁边存在 event-radar-YYYY-MM-DD.csv.json，其中 as_of 是事件快照冻结时间。
- 模型排名必须包含 instrument_id、expected_net_return、return_q10 和 model_disagreement。

## 成熟后复盘

结果成熟后准备包含 instrument_id 和 realized_net_return 的 CSV，然后运行：

    uv run --locked quant-research event-overlay-review \
      artifacts/event-overlay-YYYY-MM-DD-v1 \
      path/to/realized-returns.csv \
      --output artifacts/event-overlay-review-YYYY-MM-DD-v1.json

复盘分别报告三组的发布数量、Top N 已知结果覆盖、平均与中位实现收益、正收益比例和组内 Rank IC。不同组的候选集合不同，因此差值只是影子组合描述，不是因果收益估计。

## 代码与验证

- src/quant_research/event_overlay.py：时间对齐、门槛、联合排序、不可变输出和成熟后复盘。
- src/quant_research/cli.py：event-overlay 与 event-overlay-review 命令。
- configs/event-overlay-v1.json：冻结的第一版权重和硬阻碍项。
- tests/test_event_overlay.py：预测窗口、硬阻碍、事件后确认、防未来信息、不可变产物和复盘测试。

本阶段不重训 Transformer，不修改现有模型排名，不自动下单，也不把尚未验证的事件权重提升为生产策略。下一阶段应在足够多的成熟日期上做滚动影子比较，再决定是否训练事件感知模型。

