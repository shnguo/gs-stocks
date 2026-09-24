# 自建 Transformer 改为行情 token 生成

后续状态：已完成 [完整开发窗口容量对照](token-capacity-20260914.md)，22.86 万条训练输入，372 万/28 万参数模型分别训练 9/39 轮，并完成多日生成与独立复核。下文保留最初代码改造及工程验证记录。

2026-09-14。按用户确认，采用 Kronos 的粗、细 token 生成机制，并将自建主体改为 Decoder-only 因果结构。代码、训练入口、路径输出及小批真实数据工程验证已完成；尚未完成新模型的正式效果训练和滚动评价。

## 新任务与旧模型的关系

旧 PlanTransformer 接收过去 60 日的 27 项特征，可叠加信号日 11 项信息，直接输出 12 个固定交易计划的五项指标与价格分位数。新 TokenTransformer 接收过去行情的 token 对和日历信息，学习下一根 K 线的 token 分布。新的训练损失没有买入档位、止盈、止损、成交概率或计划收益标签。

本次输入也随任务变化：历史开、高、低、收、成交量、成交额由冻结 Kronos tokenizer 编码；因子只用于将历史价格锚定到信号日，时间字段只包含日历。原 27 项工程特征与 11 项上下文分支不接入本版本。不能把该实验解释为只改变最后一层的架构消融。

新研究入口是 scripts/token_transformer_run.py。旧 PlanTransformer、历史检查点、冻结产物和现有每日 LightGBM 模型保留；旧计划检查点不能加载为 token 模型。此前停止的优化队列没有恢复。

## 结构与目标

自建主体保留 2 层、48 维、4 头，使用 LayerNorm、GELU 和学习式位置编码。新模型共有 281,936 个可训练参数，包含 token embedding、日历 embedding、因果主体和两个词表分类头。主体权重随机初始化，不加载 Kronos 的预训练预测器。

复用已固定版本并校验 SHA-256 的 Kronos-Tokenizer-base；tokenizer 全程 eval、禁止梯度。每根 K 线编码为两个 token，各有 1,024 种取值。

1. 历史 token 对经过因果自注意力，预测下一根 K 线的粗 token 分布。
2. 以粗 token 的 embedding 为 query，对允许的历史表示做依赖注意力，再预测细 token 分布。
3. 训练时第二步使用同一天的真实粗 token，即 teacher forcing；推理时使用采样出的粗 token。
4. 将采样的粗、细 token 对追加到历史，继续生成下一天。
5. 使用冻结 tokenizer 解码为六项连续行情，反归一化，并保留每条独立路径。

这是与 Kronos 一致的 token 表示及概率分解机制。自建主体的归一化、位置编码和前馈层不与官方逐层相同，也不兼容官方预测器权重。没有直接照搬官方依赖注意力在 eval 模式下的遮罩切换；本版本对整段训练/评价始终采用因果遮罩，逐步采样时最后一个 query 可查看全部已知历史。

## 信息隔离

训练序列是 60 根历史 K 线加未来 5 根标签 K 线。输入取序列的前 64 根，目标错开一位取后 64 根；只对最后 5 个预测位置计算两种 token 的平均交叉熵。所有历史位置用于提供条件，没有额外的历史下一步损失。

原因是归一化统计量由信号日前完整 60 日历史确定；对历史内部位置计算下一步损失，会让其尺度包含相对于该位置更晚的历史信息。只监督信号日之后的预测位置，避免这项问题。

自注意力和粗细依赖注意力都遮住右侧位置。真实未来粗 token 只允许影响对应的细 token 分类，不进入对应粗 token 的预测。官方 tokenizer 的前缀因果性也经过真实检查点核验：编码完整序列与只编码历史，得到的历史 token 完全相同。

未来数据继承已冻结价格数据的缺失、公司行动与序列边界有效性标记，再检查六项行情合法性。首次未知之后的后缀全部遮罩，后面的有效日不能越过缺口重新成为训练目标。未知目标使用占位值完成编码，但不计入损失；原始未知数据继续保留。

训练、选择和评价按原开发日期分区，目标结束日必须早于下一分区。新入口要求信号日不早于 2024-07-01，且严格禁止标签到达 2025-08-07 的封存期。论文预训练截至 2024 年 6 月的声明不能代替下载 tokenizer 的独立训练日期清单；因此仍保留预训练污染未完全排除的限制。

## 训练与输出

配置：configs/token-transformer-v1.json。默认 60 日输入、五日预测、每个股票日期 32 条路径，主体上下文容量为 512。512 是配置容量；当前训练使用 60 日历史，不代表已验证长上下文效果。

训练按日期等权、按每行已知未来日期平均，粗细交叉熵等权。选择集 token 交叉熵用于早停和检查点选择；评价结果不参与选择。保存最佳、最后检查点、逐轮日志、初始化选择集误差与 CPU 重载参考值。达到轮数上限且未满足早停条件时标注预算受限。

推理函数只接受历史行情/token、历史日历、未来日历和采样参数，不接受未来行情值。输出形状为：

    [股票日期数, 采样路径数, 预测交易日数, 6]

字段依次为 open、high、low、close、volume、amount。额外保留生成的两组 token、随机种子和路径合法性标记。负价格、负量额、最高最低价关系错误等路径保留原值并标记，不自动修成合法行情；合法性也不代表满足涨跌停或真实成交条件。

研究评价输出六项行情在每一天的历史尺度归一化中位数 MAE、采样 CRPS、80% 区间覆盖，以及持平预测基线。指标按日期等权，只对标签已知且至少有八条完整合法路径的行计算，同时公布原始覆盖数，不将筛选后的误差当成全体误差。

## 可复现入口

运行目录为 /Users/guo/Documents/stocks/quant-research。tokenizer 依赖使用已存在的隔离 Kronos 环境，基础项目的 uv.lock 没有改动。首次准备：

    artifacts/kronos-comparison-20260910-v1/venv/bin/python scripts/token_transformer_run.py prepare --prior artifacts/plan-value-20260914-v1 --inputs artifacts/kronos-inputs-20260914-v3 --bundle artifacts/kronos-comparison-20260910-v1 --config configs/token-transformer-v1.json --output artifacts/token-transformer-new

准备完成会冻结源码、配置、数据及来源哈希。后续使用冻结入口及冻结模块，依次运行 train、predict、verify：

    PYTHONPATH=/Users/guo/Documents/stocks/quant-research/artifacts/token-transformer-new/code/src artifacts/kronos-comparison-20260910-v1/venv/bin/python artifacts/token-transformer-new/code/scripts/token_transformer_run.py train --output artifacts/token-transformer-new

将上述命令的 train 替换为 predict、verify 即为另外两步。已有目录禁止覆盖；当前不提供中断后精确续训。全量配置只提供入口，本次未启动全量训练。

## 本次验证结果

- 25 项相关测试通过，包括因果性、粗细依赖、teacher forcing 与逐步推理一致性、未知后缀零损失/零梯度影响、归一化、采样种子、上下文滚动、检查点重载，以及原计划/上下文模型回归。
- 新模型专项测试 10 项通过，其中包含真实 Kronos tokenizer 的编码与六项行情解码。
- Ruff 对新增模块、脚本和测试通过。
- 小批真实数据：32 行训练、16 行选择、16 行评价；分别覆盖 4、2、2 个信号日期。评价 80 个未来日中有 75 个已知，未知行保留。
- 两轮工程训练：选择集 token 交叉熵从初始化的 7.116927 降至 7.042422。只用于检查梯度更新和检查点选择；明确标注预算受限。
- 生成形状为 [16, 32, 5, 6]，共 512 条路径，其中 205 条满足六项行情合法性，13 行达到八条合法路径的评价门槛。这种小样本、极短训练的结果不能判断模型质量。
- 最终验证复现 262,144 个 CPU logits、官方 tokenizer 前缀、固定种子采样路径，核验路径解码和合法性，并重算 30 项日/字段指标。

证据：[工程验证结果](../artifacts/token-transformer-smoke-20260914-v1/verification.json)、[训练记录](../artifacts/token-transformer-smoke-20260914-v1/training/summary.json)、[输出覆盖](../artifacts/token-transformer-smoke-20260914-v1/forecast/summary.json)。全部工程进程已完成。代码未提交，未部署，未替换每日模型。
