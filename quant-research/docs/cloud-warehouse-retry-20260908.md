# 数仓临时故障恢复

2026-09-08 05:20 UTC，Iceberg 请求返回 502，旧版本把发布错误直接传播为整体暂停。此次修复保持全部 11,624 项原采集计划和已有回执，新增数仓发布的持久重试。

## 行为

- 发布批次保持同一组任务及其内容身份。每次重试使用新数仓连接，检查已有 Iceberg 快照；已提交但响应丢失时恢复回执，不追加重复数据。
- 临时错误最多尝试 6 次，重试间隔为 5、30、120、300、900 秒。次数、到期时间和每次结果写入 D1，重启不重置。
- 等待期间采集继续；待入仓和正在采集的任务合计达到 100 项后先处理入仓，再继续采集。
- 权限、格式、完整性、未知错误和 D1 控制面故障仍需审查。持续失败耗尽预算后才整体暂停。
- 现有进度入口新增发布批次与最近发布尝试，采集任务完成量不因重试重复增加。

原始归档和成功回执不被重写，ST 与全市场范围保持原研究计划。完整训练门槛继续关闭。

## 验证

故障注入覆盖提交前 502、服务端已经提交后丢失响应、重启重放：最终只有一个 Iceberg 快照。独立 SQLite 执行生产 SQL，核对延迟重试、次数上限、租约、采集暂缓和旧错误恢复。Rust 相关测试、Worker 鉴权与调度测试、Clippy、格式和 TypeScript 检查通过。

迁移、部署及云端恢复结果保存在 artifacts/cloud-warehouse-retry-20260908-v1。

## 云端验收

2026-09-08 09:30 UTC，小批量 10 次采集全部成功，新增 13,455 行；此前 4 项积压、5,378 行全部入仓。实际 R2 SQL 按 14 个任务逐项核对来源、行数和唯一记录序号，共 18,833 行，重复为零。此时累计采集并入仓 4,082 项、4,871,155 行，发布队列已排空。

09:33 UTC 已恢复单路不限量持续采集，状态 active，随后新任务继续采集并入仓，心跳及运行回执正常。实时进度读取 GET /status；本页数字为固定验收时点快照。

全部 11,624 项任务身份与序号一致；4,072 项原采集回执、4,068 项原发布回执、346 项权限受限与 3 项待核查失败保持原状。原 502 的固定发布批次重放成功，旧错误另存 D1 尝试记录和经过下载核验的 R2 证据。源码也已单独备份 R2，并恢复到新目录逐文件核验。

Worker 版本为 ba8f0fda-a7d6-4bf4-b5a3-533c142a5560，运行标记 research-cloud-v4-warehouse-retry，部署镜像摘要 sha256:20cb205b236162919e667fb63625a6701325a325013b773ead2a87a4cdb27bd4。新增迁移 0028 已远程执行。代码尚未提交 Git；本次没有升级依赖或修改其他行情服务。

- [完整任务基线](../artifacts/cloud-warehouse-retry-20260908-v1/baseline-jobs.json)
- [原件、旧回执和恢复任务验证](../artifacts/cloud-warehouse-retry-20260908-v1/canary-verification.json)
- [实际数仓行数及重复核验](../artifacts/cloud-warehouse-retry-20260908-v1/warehouse-verification.json)
- [源码备份与恢复回执](../artifacts/cloud-warehouse-retry-20260908-v1/operator-code-receipt.json)
- [持续运行验收快照](../artifacts/cloud-warehouse-retry-20260908-v1/continuous-status-1.json)
- [独立恢复验证脚本](../scripts/verify_cloud_publication.py)
- [数仓运维说明](/Users/guo/github/mootdx-cf/docs/research-cloud.md)
