# 2026-09-14 数据接入与今日策略

已完成本轮目标：行情更新到上一完整交易日 2026-09-11，并输出今天至 9 月 18 日的五日策略。实际发布版本为上海时间 10:31 的盘中补发研究版，不是追溯伪造的盘前报告。

[打开今日完整策略](../artifacts/daily-strategy-20260914-v1/runs/2026-09-14/599f1bf69970bce70ebe9b08/report.md)

## 今日结论

当前模型与固定收益风险门槛没有产生买入候选，今天暂不新开仓。该结论只表示本模型未找到符合条件的机会，不能解释为市场上没有机会。报告保留十只观察股票的研究买价、目标价、失效价和每日价格区间；这些观察价不构成挂单建议。没有用户持仓信息，因此不编造实际持仓的卖出指令。

完整报告覆盖 T=9 月 14 日、T+1=9 月 15 日、T+2=9 月 16 日、T+3=9 月 17 日、T+4=9 月 18 日。当前模型仍未通过正式建议质量验收，盘中实时行情与公告层尚未纳入预测输入。原有固定规则仅保留为研究对照，不作为今日买入名单。

## 实际接入的数据

- 最新上市名单：5,562 只，包含沪深北；其中 9 月 11 日有日线价格的 5,550 只。
- 既有历史身份与新增身份合并后为 5,905 条，退市和历史代码身份继续留痕。
- 完成连续性与特征检查后，5,361 只进入本次推理。
- 接续覆盖台账记录 85 条参考前收变化、7 条缺少连续交易日行情、357 条缺少跨来源接续锚点；其中也包含历史身份与新股，不能将这些数字全部解释为当前股票缺数据。
- 沪深北九月日历按已取到的交易日历及三所正式休市通知交叉核对，证据仅在登记的日期范围内生效。

采集和金融字段解析仍由现有 Rust 工程承担。新增 research-source-normalize 命令重放已归档原文，校验来源状态、参数范围、重复行、价格关系、数量单位和截断风险；Python 只负责研究特征和推理衔接。所有失败响应、成功原文及归一化文件分别保留。

复权因子接口在本账户上返回每小时一次限制，因此本轮没有假定已取得最近一周全部复权因子。改用日线中的除权参考前收价识别新的价格边界：先核对 9 月 4 日新旧来源原始 OHLC，再逐日要求参考前收与上一实际收盘严格一致。只有这条链连续的股票才续用原有因子；参考前收变化或缺行情时停止该股票接续，不估算分红因子或股东权益。该方法是研究价格连续性证据，不是公司行动完整性的证明。

资料：[Tushare 日线字段定义](https://tushare.pro/document/2?doc_id=27)、[上交所 2026 年休市通知](https://www.sse.com.cn/disclosure/announcement/general/c/c_20251222_10802507.shtml)、[深交所通知](https://www.szse.cn/disclosure/notice/t20251222_618087.html)、[北交所公告原文](https://dataclouds.cninfo.com.cn/sjother2/regulatory/2025/20251222/55875a9937374da4ae2d4999dfda2722.pdf)。

## 复用入口

工作目录：/Users/guo/Documents/stocks/quant-research。

    .venv/bin/python scripts/daily_refresh.py --base artifacts/full-market-training-20260910-v1 --workspace artifacts/daily-input-20260914-v1 --rust-repo /Users/guo/github/mootdx-cf --env-file /Users/guo/github/mootdx-cf/.env --package artifacts/daily-loop-20260913-v1/models/01b7ccdceb61dd477994 --store artifacts/daily-strategy-20260914-v1 --date 2026-09-14 --calendar-evidence artifacts/daily-input-20260914-v1/calendar-evidence.json --mode intraday_research --input-version input-v2

该入口依次完成 Rust 采集、原文重放、增量研究快照、特征计算和策略生成。成功采集可复用，失败响应不覆盖，也不绕过供应方配额。原始模型和历史快照未被改写。

下一交易日使用新的日期和工作目录，以当前已验收输入作为 base；未来日历占位不能充当已完成行情。正式盘前运行使用 prospective 模式并在 09:15 前完成，盘中补发必须显式选择 intraday_research。没有发布后的分时证据时，补发计划的日线回放不能追認发布前成交，结果保持未知。

本轮未创建持续调度，也没有训练、部署或自动下单。数据入口与今日输出已完成，接下来主任务仍是改善模型预测及规则的决策价值。

## 验证证据

118 项 Python 完整测试、最终兼容性修正后 15 项相关回归、8 项 Rust 来源解析测试通过；Rust clippy 与变更文件检查通过。

从原始数据独立重算本次全部 5,361 只股票的 27 项输入，共 144,747 个数值，以及 16,083 个研究买卖价，均一致。未来日期的特征仍全为空，没有使用今天未完成的日线作为预测输入。原文、快照、模型输入和报告文件已校验哈希。

[独立核验结果](../artifacts/daily-strategy-20260914-v1/verification.json)；输入与来源链见 artifacts/daily-input-20260914-v1/input-v2。本次工程和算术验证不改变模型质量未过关的结论。
