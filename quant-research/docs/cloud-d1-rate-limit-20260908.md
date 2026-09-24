# D1 限流与数仓回执恢复

2026-09-08 21:54 UTC 每小时巡检发现，采集在 21:16 UTC 因 D1 HTTP 429 暂停。错误中的 SQL 哈希对应原批次的任务入仓回执更新。没有新的来源故障；暂停时 7,114 项已采集，7,111 项已登记入仓，3 项共 4,134 行等待核对。

## 核验与修复

实际 R2 SQL 查到原发布批次的 2,675 行，记录序号唯一；另外两项共 1,459 行尚未入仓。Rust 恢复程序下载核验三项原始归档和计划身份，按原始内容计算批次提交身份，并只读查询 Iceberg 快照，确认已存在完全一致的提交 206b25a7abde91009e516d1f5bcc2486016ad1050fb245142f2d0030e4253613。

D1 请求由采集、发布和心跳共享节流器，两次请求开始至少间隔 500 毫秒。收到 429 时共享服务器 Retry-After 等待；缺少该字段时等待 300 秒。每次查询至多追加三次重试，超过 600 秒的等待要求停止并保留审查状态。

只有完整的 429 API 拒绝响应，且没有执行结果，才允许普通写入在等待后重试。网络中断、响应截断和 5xx 等结果不确定的写入保持原有核对要求；权限、数据格式和完整性门槛不放宽。采集并发仍为单路，来源和发布的持久尝试次数不重置。

recover-warehouse 增加受控分支：只接受与唯一待发布批次完全匹配的 429 回执更新语句；必须已停止协调器和来源请求、租约释放、原始归档完整、数仓已有准确内容提交。恢复只准备人工暂停状态，后续通过原发布流程重放回执，并保留原失败尝试。未找到准确快照时拒绝恢复，不追加数据。

参考 [Cloudflare API 限流说明](https://developers.cloudflare.com/fundamentals/api/reference/limits/) 和 [D1 重试建议](https://developers.cloudflare.com/d1/best-practices/retry-queries/)。本次错误没有保存当时完整限流头，因此不能确定触发的是账号共享额度还是某个更细的接口限制。

## 验证进度

13 项 Rust 控制面测试、3 项数仓测试、61 项独立 SQLite 合约执行、9 项 Worker 测试通过；Clippy、TypeScript 与格式检查通过。数仓测试覆盖已有提交与不同内容的只读查找，以及丢失发布回执后的无重复重放。

22:08 UTC 受控恢复准备完成。原错误、原批次和原采集记录已存入 R2，并下载核验：research/v1/warehouse-control-recovery/full-a-share-20260907-v2/587ca70cca593810bb53226a95764a41f2be3f4a2e66ada5021fee0d887aec44.json。

22:11 UTC 新版本部署成功：Worker a6bb229e-511e-4cea-87ca-118fe48434ff，镜像 sha256:76ca41e5ba4c9cf19215811b3096eb344637d2f14415482c60ed4280b7bdd5d1。运行标记 research-cloud-v6-d1-rate-limit。第一次部署在上传前网络超时，核对版本没有变化后重试成功。

22:16 UTC 小批量结束，10 项采集全部成功，新增 10,640 行，没有发布失败。3 项积压均已恢复回执或补入，共 4,134 行；其中原 2,675 行批次复用了已有内容提交，没有追加同批数据。原发布尝试保留为 outcome_unknown，另存 control_failed 错误凭据，原批次尝试次数从 1 增至 2，没有重置。

22:18 UTC 实际 R2 SQL 按 13 项任务核对来源、行数和唯一序号，共 14,774 行、重复零。独立 D1 全表比对确认全部 11,624 项身份与序号、7,114 项原采集回执、7,111 项原发布回执以及 349 项既有失败或权限限制保持完整。

262 个源码文件已备份并下载还原，逐文件哈希及变更文件与本地一致。源码对象为 research/v1/operator-code/47d4ad5230fefc3b109e2eff37b1bc9e82c3cbe76bb739e329e84610caa7cd9e.tar.gz。

完整训练门槛保持关闭；源码未提交或推送。证据目录为 artifacts/cloud-hourly-check-20260908T2154Z。

22:19 UTC 已恢复单路、不限量持续采集，新运行 1788905947708291314-1。Cloudflare 采集不依赖本机保持在线。

- [实际数仓行数与重复核验](../artifacts/cloud-hourly-check-20260908T2154Z/warehouse-verification.json)
- [全体任务身份与旧回执核验](../artifacts/cloud-hourly-check-20260908T2154Z/canary-verification.json)
- [原批次及失败证据保留](../artifacts/cloud-hourly-check-20260908T2154Z/recovered-batch-audit.json)
- [源码备份与恢复回执](../artifacts/cloud-hourly-check-20260908T2154Z/operator-code-receipt.json)

修复验收证据已于 2026-09-08T22:24:38.071491Z 归档至 R2，下载恢复后独立核对全部 51 个文件的哈希及大小一致。

归档位置：mootdx-cf-raw-preview-deployment-01/research/v1/archives/cfad1b8c52cd49cfb05a18efe0ea4de9050a2fde597893070fba9bade848d6f9.tar.gz

连续采集运行快照（2026-09-08T22:20:44Z）：已采集 7,130 个任务，已入仓 7,129 个任务、8,283,373 行；持续采集已恢复。全量数据尚未就绪，每小时巡检继续。
