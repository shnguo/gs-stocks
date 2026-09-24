# 云端采集、数仓和换机恢复

全市场历史采集使用独立的 Cloudflare Worker mootdx-cf-research 和 Rust Container。Worker 每分钟检查 D1 中的活动批次，Container 续采固定计划中的待采任务。关闭本机、退出编辑器和更换电脑不会终止云端任务。

股票池仍为完整 A 股目录，包含 ST，个人自选列表不参与任务生成。当前计划是 full-a-share-20260907-v2，包含 11,624 项任务、5,901 条历史代码候选，其中 5,812 条与研究区间相交。

## 数据存放

| 内容 | 云端位置 |
| --- | --- |
| 初始研究数据及原始证据 | R2 mootdx-cf-raw-preview-deployment-01，research/v1/archives |
| 云端新增原始响应及尝试记录 | 同一 R2 桶，research/v1/campaigns/full-a-share-20260907-v2/captures |
| 任务、限额暂停、失败和入库回执 | D1 mootdx-cf-control-preview-deployment-01，research_campaigns、research_jobs、research_publications |
| 可查询的来源数据 | R2 Data Catalog，quant_research_v1.research_source_records，底层格式为 Parquet/Iceberg |
| 恢复工具源码、锁文件和配置模板 | 同一 R2 原始桶，research/v1/operator-code |

R2 对象包含 SHA256。采集完成后必须上传、下载验证，再持久化 D1 回执；来源错误及失败原件保留。入库前固定发布批次，重放已提交的相同内容不会重复增加记录。Container 的本地目录仅用作临时缓存。

当前数仓表是来源数据层，包含任务、股票、来源、原始字段 JSON、原件位置及哈希。质量标记为 source_verified_canonical_quality_pending。这不等于规范价格、复权、公司行动、历史身份或完整训练快照已经通过验收，data_ready 继续为 false。

## 查看最新进度

在 mootdx-cf 仓库目录运行：

    cargo run --locked -- research-cloud status workers/research/operator.env

返回值中，captured 是已采集并验证原件的任务数，pending 是待采任务数，published 是已完成来源层入库的任务数，warehouse_rows 是这些任务的记录数。status 为 active 表示云端继续推进；last_error 和 paused_requires_review 表示需要核验异常。数量随云端批次推进变化，旧本地日志不再代表最新进度。

已配置 RESEARCH_API_TOKEN 的客户端也可访问 https://mootdx-cf-research.tap2rap.workers.dev/status，需要 Bearer 鉴权。POST /pause 停止认领新任务，已认领任务和当前入库批次会先保存结果。来源权限或频率错误不会通过重启自动绕过。

2026-09-07 优化后的执行方式是持续消费待采任务、独立入库，以及最多两路来源采集。两路采集共享至少 2 秒的查询起始间隔；BaoStock 每路复用自己的登录连接，失败原件保留且不自动重试。采集和入库都保留 D1 进度与 R2 原件，不依赖本机。

2026-09-07 连接故障核验后改为单路持续运行。原容器两次低频登录/小查询成功，单路 10 项历史采集验证全部成功；完整计划和既有失败、受限记录保留。新增 GET /probe 查看诊断结果，POST /probe 在故障暂停且已排空时执行固定小查询，间隔至少 180 秒；POST /probe/archive 仅补传原件。详见 [来源核验与恢复记录](cloud-source-recovery-20260907.md)。

查看每次运行的完成数量、采集与归档耗时、连接复用情况：

    cargo run --locked -- research-cloud metrics workers/research/operator.env

GET /status 的 runs 字段也提供最近运行汇总。性能测试可以设置任务预算，完成后进入 paused_benchmark_complete；正式持续运行的 capture_budget 为 0。切换配置前必须暂停并等当前工作完成。

## 换电脑

1. 从代码仓库或 R2 的 research/v1/operator-code 目录取得恢复工具源码。R2 备份为标准 tar.gz，可以直接在 Cloudflare 控制台下载并解包。
2. 单独安全配置云端凭据，设置 workers/research/operator.env 中的 RESEARCH_SECRETS_FILE。备份不包含密钥。当前模板的 RESEARCH_HTTP_PROXY 指向旧电脑的本地代理，新电脑应删除这一项或改成自己的代理。
3. 在 mootdx-cf 源码目录执行，目标目录必须不存在：

    cargo run --locked -- research-cloud restore-campaign workers/research/operator.env /path/to/new-research-directory

恢复会下载并验证初始归档，再根据 D1 补入云端新增采集，保留限额、失败和任务状态。恢复目录的 cloud-managed 标记阻止重复启动本地采集。执行中的云端任务可能晚于恢复快照完成；需要更新副本时恢复到另一个新目录。

初始归档中的文档保留归档时的原文，恢复时不会改写历史文件。恢复目录中的 CLOUD-CAMPAIGN-STATUS.json 是本次恢复的云端状态快照，最新进度以 status 命令为准。

## 初始归档证据

2026-09-07 06:38:33 UTC，初始研究树完成 R2 上传、完整下载与逐文件恢复验证：9,668 个文件，原始内容 605,488,720 字节，压缩归档 194,705,025 字节。

- [初始归档回执](../artifacts/cloud-migration-20260907-v1/archive-receipt-v4.json)
- [恢复工具归档回执](../artifacts/cloud-migration-20260907-v1/operator-code-receipt-v1.json)
- [全部任务与初始状态独立复验](../artifacts/cloud-migration-20260907-v1/verification-before-start-v1.json)
- [首次云端启动回执](../artifacts/cloud-migration-20260907-v1/cloud-start-v1.json)
- [包含 20 个云端新增任务的恢复复验](../artifacts/cloud-migration-20260907-v1/recovery-verification-v1.json)

初始归档键：research/v1/archives/4d5a8845a2bbdcec01477bd219538e207a450937ae58d67cdb6e60133b9c60f2.tar.gz。云端后续采集另存，换机时应使用 restore-campaign 恢复完整状态。

## 数仓与运行验收

2026-09-07 已通过 R2 SQL 查询验证首批 20 个任务的 21,002 条记录，与 D1 发布回执一致，按任务及记录序号分组未发现重复。这一批是已归档历史资料的数仓回填；云端新增采集进入同一发布队列。发布按固定任务顺序推进，采集完成数可以领先于入库任务数。

- [首批 D1 发布回执与预期行数](../artifacts/cloud-migration-20260907-v1/first-publication-v1.json)
- [实际 SQL 行数及任务数](../artifacts/cloud-migration-20260907-v1/warehouse-first-batch-v1.txt)
- [按来源、数据集和质量状态查询](../artifacts/cloud-migration-20260907-v1/warehouse-query-v1.txt)
- [重复记录检查](../artifacts/cloud-migration-20260907-v1/warehouse-duplicates-v1.txt)

首次迁移验收时的云端 Worker 版本为 0ee54d6b-b8a0-4434-a055-eb9fcf6085ec，独立 Container 应用为 a0360d80-8b0d-4969-855e-e42fc60c58d3，运行实例数实查为 1。已验证首批完成后由云端自动进入下一批；本机没有继续执行来源采集。

截至 2026-09-07 06:54 UTC 的独立快照：云端 active，成功采集 1,511 项、待采 9,766 项；数仓提交回执覆盖 60 个任务、67,913 行。346 项限额暂停和 1 项历史失败继续保留。该快照是验收记录，实时进度应读取 D1。[运行中复验](../artifacts/cloud-migration-20260907-v1/verification-cloud-running-v1.json)。

06:57 UTC 收尾检查：云端继续 active、last_error 为空；成功采集 1,530 项，正在采集 1 项，待采 9,746 项，80 项已入库，合计 90,476 行。[实时接口回执](../artifacts/cloud-migration-20260907-v1/final-cloud-status-v1.json)。
