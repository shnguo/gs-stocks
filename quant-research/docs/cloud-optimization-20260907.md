# 云端采集优化实测

2026-09-07，已部署连续采集、独立入库和双路采集，最终选择双路持续运行。仍使用原来的单个 Cloudflare Container，原始数据保存到 R2，任务和发布回执保存到 D1，来源记录进入 Parquet/Iceberg 数仓。

## 实测结果

| 方式 | 样本 | 采集窗口 | 每分钟任务数 |
| --- | --- | --- | --- |
| 原有串行分批流程 | 暂停前最近 80 个成功任务，79 个完成间隔 | 1,194.36 秒 | 3.97 |
| 连续单路采集、独立入库、连接复用 | 40 个待采任务 | 385.56 秒 | 6.22 |
| 连续双路采集、独立入库、连接复用 | 接下来的 40 个待采任务 | 210.71 秒 | 11.39 |

双路相对旧流程约为 2.87 倍，相对优化后的单路约为 1.83 倍。单路与双路分别取得 50,556 和 51,632 条来源记录，80 个任务全部成功，没有新增失败。两组分别复用了 36 次和 38 次 BaoStock 登录连接。

这是相邻真实任务的现场比较，包含网络波动、历史长度与数据源组合差异，并非随机对照。新流程的窗口从第一项采集开始到最后一项归档回执完成，不含测试结束后当前发布批次的收尾。单路运行整体耗时 398.06 秒，双路 227.10 秒。旧样本包含 16 项 Tushare 日线，单路样本包含 3 项，双路样本全部来自 BaoStock；后续吞吐量仍以实时指标为准。

实测结束时待采 9,486 项，按双路速度线性推算约 13.9 小时。该估算以接口和云端运行稳定为前提，不包括 346 项接口受限与 2 项既有失败的处理，也不代表训练数据质量验收完成。

## 执行与保护

- 固定全市场计划仍为 11,624 项，含 ST，个人自选列表不参与范围定义。
- 采集持续认领 D1 中的待采任务；独立发布流程每批最多入库 20 项，采集不再等待入库或下一分钟的 cron。
- 两路采集共享至少 2 秒的查询起始间隔。双路实测最小间隔为 2.000484 秒，最大同时处理任务数为 2。
- 每路独立复用 BaoStock TCP 会话。每份采集保留实际登录响应、登录时间和复用标记；失败请求保留证据，不隐式重试。
- D1 租约定时续期，认领和结果写回核对租约所有者。暂停后停止认领新任务，已开始的采集和发布完成后释放租约。
- Worker 启动任务前检查实际 Container 运行版本，避免部署配置更新后仍把任务交给旧实例。实际运行版本为 research-cloud-v2-pipeline。
- 正式配置为 capture_concurrency=2、capture_budget=0。每分钟 cron 保持云端任务推进，不依赖本机在线。

## 验收证据

08:05 UTC 的稳定快照中，全部任务身份保持不变；优化前 1,700 项成功原件及其行数、哈希、路径完整保留，346 项接口受限和 2 项既有失败也保持不变。另有 10 项在平台实例切换期间由旧流程完成，未计入两组优化测试，原件和回执均已保存。

此时累计采集成功 1,790 项，已发布 1,260 项。R2 SQL 实际查得 1,469,281 行、1,260 个任务，与 D1 一致；按任务和记录序号分组，未发现重复记录。来源表仍标记 source_verified_canonical_quality_pending，完整训练门槛保持关闭。

4 份新来源样本从 R2 下载恢复，包含新连接日线、复用连接日线、复用连接因子和北交所日线。独立文件哈希、查询范围、行数核验及 Rust 原始报文回放通过。193 项 Rust 测试、6 项独立 SQLite 合约验证、3 项 Worker 调度测试、Clippy、格式和 TypeScript 检查通过。

- [全部任务与两组实测验证](../artifacts/cloud-optimization-20260907-v1/benchmark-verification.json)
- [速度对比与估算](../artifacts/cloud-optimization-20260907-v1/comparison.json)
- [数仓实际行数](../artifacts/cloud-optimization-20260907-v1/warehouse-count.txt)
- [数仓重复检查](../artifacts/cloud-optimization-20260907-v1/warehouse-duplicates.txt)
- [原件恢复复验](../artifacts/cloud-optimization-20260907-v1/capture-recovery-verification.json)
- [代码和部署验收](../artifacts/cloud-optimization-20260907-v1/implementation-validation.json)
- [恢复工具源码云端归档](../artifacts/cloud-optimization-20260907-v1/operator-code-receipt-v1.json)
- [持续双路运行复验](../artifacts/cloud-optimization-20260907-v1/continuous-verification.json)
- [08:11 UTC 持续运行状态](../artifacts/cloud-optimization-20260907-v1/final-cloud-status.json)
- [报告与验收证据云端归档](../artifacts/cloud-optimization-20260907-v1/evidence-archive-receipt-v2.json)

报告与验收证据已另行上传 R2 并下载恢复验证，共 89 个文件，压缩后约 5.12 MB。归档键为 research/v1/archives/de6bc03560db03b2f7057be4cfe24da5e5a7dc38c85f3c09fe50d158af452222.tar.gz。

08:11 UTC 收尾检查：云端 active，双路、无限任务预算，last_error 为空；持续模式已额外完成 54 项，累计成功采集 1,844 项、待采 9,430 项，已入库 1,540 项、1,816,471 行。本机没有运行来源采集进程。进度持续变化，上述数字是带时间的验收快照。

Worker 版本为 5a8b2fa4-910b-4004-8c18-3b0d2070a261；Container 镜像为 sha256:e184851d5a0143f21efc6d0082469e03203c5e361ad8dc2847a0fcf5ddf76a5b。源码归档已下载恢复，离线锁定 Cargo 工作区验证通过。工作区改动尚未提交或推送到 Git。

## 查看最新进度

在 mootdx-cf 仓库目录运行：

    cargo run --locked -- research-cloud status workers/research/operator.env

查看各次运行的完成数量、来源采集与归档耗时、连接复用：

    cargo run --locked -- research-cloud metrics workers/research/operator.env

也可使用经过 Bearer 鉴权的 GET /status。更多操作及换机恢复说明见 [云端采集与恢复](cloud-research.md)。
