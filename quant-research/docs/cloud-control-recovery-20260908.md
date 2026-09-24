# D1 响应故障与未开始任务恢复

2026-09-08 12:43 UTC 的每小时巡检发现，采集在 11:51 UTC 暂停，最后错误为 error decoding response body。暂停时 4,733 项已采集并入仓，共 5,681,504 行；没有待发布批次，一项任务停留在 running，另外 6,541 项待采。

## 原因与证据

出错区间是 D1 已保存认领、来源尝试尚未登记之间。该任务为 002513 的 BaoStock 复权因子请求。任务尝试次数为零，D1 没有该任务的来源尝试，R2 采集前缀为空，也没有发布批次。实际 R2 SQL 查不到该任务记录，同批检查的已发布对照任务有 2,337 行且无重复。

原代码直接解析 D1 JSON，解析失败没有保留 HTTP 状态及请求标识，因此不能确定当时的具体 HTTP 状态或上游返回内容。证据支持认领响应或尝试登记阶段的控制面响应故障，不支持推断 BaoStock 封禁。

## 修复

- 只读 D1 查询对临时 HTTP、网络或 JSON 截断进行最多三次追加重试，采用 1、3、10 秒延迟及少量错峰。
- 普通写入保持不自动重放。任务认领与尝试登记遇到响应不确定时，先按固定任务身份、认领时间、尝试 ID 及有效租约核对数据库，再决定是否继续或重试。
- 错误记录增加 HTTP 状态、请求标识和 SQL/响应哈希，不输出响应正文或凭据。
- 新增受控恢复入口 recover-unstarted，只有协调器已停止、没有运行中来源尝试，且确认任务从未开始采集时才释放原认领。其他有原件或尝试记录的任务仍要求独立核对。
- 迁移 0029 新增恢复凭据表。原错误和恢复证据已写入不可变 R2 对象并下载核验，任务身份、原件、成功回执、已有失败及来源尝试预算保持原状。

Rust 目标测试 8 项、SQLite 合约执行 59 项、Worker 测试 9 项通过；Clippy、TypeScript、格式和部署预检通过。SQL 合约执行数包含继承的回归用例。

实现参考 [Cloudflare D1 重试说明](https://developers.cloudflare.com/d1/best-practices/retry-queries/)，写入恢复额外依据本项目的持久身份与租约约束。

## 部署与恢复

Worker 版本为 1be2f368-cf93-4a86-aafd-e8ab94778d0f，镜像摘要 sha256:d398923f4d03266232641c225e512a68e1e0291baae5fa1c678d0cbbab8cab29，运行标记 research-cloud-v5-control-reconcile。

12:53 UTC 已通过 Rust operator 安全释放未开始的认领。恢复对象为 research/v1/control-recovery/full-a-share-20260907-v2/33829cde16d2b5073bf2e5c867ac82876a57ce3fc6273d47e97cad78bc77edd1.json。

260 个源码及配置文件已备份为 research/v1/operator-code/a899e4886f7d8b7ce99ec3119374efd650fe5ab1248a1eb4882f95a054cb2b49.tar.gz，并恢复到新目录验证。构建时 Mac 处于后台唤醒状态，OrbStack 虚拟机仍休眠；短时唤醒后构建恢复，没有重启 Docker 或其他容器，也没有改变长期电源设置。

证据目录为 artifacts/cloud-hourly-check-20260908T1242Z。完整训练数据门槛继续关闭，代码尚未提交或推送。

## 云端验收

13:18 UTC，小批量 10 项全部成功，新增 10,096 行。13:21 UTC 的独立 R2 SQL 对账确认 10 个任务的来源、行数与唯一记录序号一致，重复为零。原卡住任务只发生一次实际来源尝试，已经完成入仓。

独立 D1 全表对比确认 11,624 个原任务身份与序号、4,733 项原采集及发布回执、349 项原失败或权限限制均保持完整。原协调器失败记录和新的控制面恢复凭据同时保留。

13:22 UTC 已恢复单路不限量持续运行，状态 active、当前错误为空。最新持续进度以远端 D1 为准。

- [任务与旧回执保留验证](../artifacts/cloud-hourly-check-20260908T1242Z/canary-verification.json)
- [实际数仓逐项对账](../artifacts/cloud-hourly-check-20260908T1242Z/warehouse-verification.json)
- [恢复凭据与采集尝试验证](../artifacts/cloud-hourly-check-20260908T1242Z/recovery-audit.json)
- [源码备份回执](../artifacts/cloud-hourly-check-20260908T1242Z/operator-code-receipt.json)
- [持续运行状态](../artifacts/cloud-hourly-check-20260908T1242Z/continuous-status-1.json)
