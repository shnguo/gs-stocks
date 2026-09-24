# Kronos 与 LightGBM、Transformer 对照

2026-09-10：加入 Kronos-small 预训练骨干的监督微调实验。原有 Transformer 队列继续运行，Kronos 完整队列等待其完成后使用同一台 Mac 的 MPS。当前仅完成数据与工程验证，尚无 Kronos 的完整开发期比较结果。

官方来源：[Kronos](https://github.com/shiyu-coder/Kronos)、[Kronos-small](https://huggingface.co/NeoQuasar/Kronos-small)。代码固定到 67b630e67f6a18c9e9be918d9b4337c960db1e9a；模型固定到 901c26c1332695a2a8f243eb2f37243a37bea320；Tokenizer-base 固定到 0e0117387f39004a9016484a186a908917e22426。下载的配置、权重和模型说明已通过 Hub 校验；上游 MIT 许可证和逐文件哈希保存在实验目录。

## 对照约定

| 项目 | 本轮设置 |
| --- | --- |
| 数据 | 原有 full-market-training-20260910-v1 的同一验收快照及样本身份 |
| 分区 | fold 14；训练从 2019-03-01 开始，验证从 2024-03-01 开始，测试为 2025-03-01 至 2025-05-31；真实标签终点按原规则清除跨界样本 |
| 周期与种子 | 5 / 20 个交易日；17 / 29 / 43 |
| 标签 | 原有 T+1 开盘至 T+h+1 开盘的股东财富收益横截面排名，保留不可用标签 |
| Kronos 输入 | 60 个历史交易日的 OHLC、股数、人民币成交额；价格使用截至信号日的历史因子归一到信号日锚点 |
| 归一化 | 每个历史窗口独立均值与总体标准差，epsilon 1e-5，截断到 ±5；不使用未来统计量 |
| 停牌 | 仅允许原验收面板明确标记的无价格停牌使用同一价格序列内的此前收盘价；股数和金额记零 |
| 微调 | Tokenizer 冻结；更新预训练骨干和新排名头。原词表输出头不参与排名优化 |
| 优化 | 与原 Transformer 相同的日期等权 Huber 目标，delta 0.1；AdamW，学习率 1e-5，weight decay 0.01，梯度裁剪 1 |
| 批量与早停 | 有效批量 256，微批量 32；最多 30 轮，验证 Rank IC 连续 5 轮不改善则停止 |
| 对照 | 精确匹配每个日期和证券身份，核对标签一致；缺失、额外、重复或非有限评分均拒绝比较 |
| 输出 | 三模型逐日 Rank IC、均值及标准差、配对日 Rank IC 差的 20 日区块自助区间；保留无标签样本和标签覆盖数量 |

这是使用预训练骨干的任务适配，不是上游原生的逐根 K 线生成式微调。Kronos 使用原生 OHLCV 输入，原有模型使用工程特征，因此结果是模型方案比较，不能单独归因于网络结构；60 日工程特征还包含更早的滚动历史。

预训练数据的时间截止与现有测试日期是否重叠尚未核实。现有 2025 年历史测试只能作为回顾性开发诊断，不能称为已排除预训练污染的样本外结果。最终保留集保持封存；未新增回测、组合收益或可实盘结论。

## 运行与日志

实验根目录：

    /Users/guo/Documents/stocks/quant-research/artifacts/kronos-comparison-20260910-v1

独立虚拟环境、固定依赖、官方权重、原始来源、准备好的输入和运行代码副本均保存在此目录。原有运行环境及训练脚本未修改。Kronos 每个周期登记一个新超参数组合，计入现有每周期最多 8 次的试验预算；三个种子不是三个超参数试验。

查看完整队列输出：

    tail -F /Users/guo/Documents/stocks/quant-research/artifacts/kronos-comparison-20260910-v1/queue.log

查看队列当前状态：

    cat /Users/guo/Documents/stocks/quant-research/artifacts/kronos-comparison-20260910-v1/queue.json

训练期间每个优化步骤结束后检查日志间隔，至少每约 15 秒输出一次进度；验证和测试评分也输出进度。每个运行单独保存 progress.json、events.jsonl、training-log.json、latest.pt 和 best.pt。模型更新或文件保存较慢时，实际间隔可超过 15 秒。

每个运行完成后生成 comparison.json 和 comparison-predictions.parquet；队列汇总到 comparison-summary.json。现有六模型基线队列成功完成后才启动 Kronos；基线失败或进程意外退出时停止并记录失败，不修改基线状态。队列锁防止重复启动。没有自动重启或断点恢复；检查点保留用于人工恢复，旧运行目录不覆盖。

本机 32 样本、60 日输入的短稳态探针约 0.208 秒/步，且同时存在原 Transformer 任务。按 512 万训练样本线性估算约 9.2 小时/轮，尚未计验证、检查点和长期运行波动，不能视为完成时间承诺。

## 验证

已验证全部首轮训练、验证、测试样本的输入窗口完整性。官方权重在真实训练分区的 32 个样本上通过 MPS 前向、反向和优化器更新：骨干权重确实变化，Tokenizer 保持冻结，未使用测试或保留集进行该检查。

新增测试覆盖未来数据隔离、价格因子连续性、无标签样本保留、样本/标签/评分失配拒绝、冻结与骨干梯度，以及训练—验证—测试—比较产物的完整顺序。完整测试与检查证据见实验目录下 tests.log、adapter-tests.log、smoke-verification.json、throughput-probe.json 及输入 manifest。

项目范围的 Ruff 检查另发现已有 scripts/verify_cloud_resilience.py 的导入排序问题；本轮新增文件通过 Ruff，该无关文件未修改。
