# 采集异常修复，2026-09-09

范围为 full-a-share-20260907-v2 中保留的 113 个异常任务。全 A 股清单、11,624 个任务标识和顺序保持不变。

2026-09-09 10:29 UTC，本轮 113 项异常全部完成处理：112 项成功重采并入仓，共 681 条来源记录；600849 为经核验的历史代码窗口不适用，0 行且无虚假数仓回执。云端已继续处理剩余正常任务。

10:31 UTC 最终实际 R2 SQL 对账通过：112 项、681 行与 D1 一致，逐任务记录下标无重复，600849 为 0 行。最终全范围核验的原计划 SHA256 为 0ba286d43fdf8b6728dcc092c82fd8e05fff829377fcbc6066f25887b1a12e3f，全部原任务与发布回执保留。

## 规则

1. 北交所复权差异：读取公司权益分派实施公告正文，使用总股本口径的现金和送转比例。每次因子变化的相对误差阈值保持 0.00001。
2. 分红接口返回空：必须完整保留原代码、原窗口的公告分页，并核对所有实施公告。空接口响应本身不能证明没有分红事件。
3. 历史代码和长期停牌：保留旧代码身份。泰祥股份只有在停牌公告和完整公告索引均支持时，才能使用最后实际交易日的收盘价；不制造近期价格。
4. 600656：保留源数据中 2016-05-12 的 active 标记，并增加最后交易日公告支持的不可训练注释，原始字段不改写。
5. 600849：保留原任务和有效空响应，将该历史代码窗口标为经复核不适用；不填入 601607 的数据，不伪造数仓发布回执。
6. 000598：原登录失败证据保留，新采集作为独立尝试记录。
7. 龙竹科技一份原公告公式出现内部冲突：仅对正文哈希完全一致的已审阅公告，按明确披露的现金总额和总股本重算现金、按完整现金分派方案确定零送转，并保留公式冲突标记。金额不从复权因子反推。

## 部署前验证

110 个保留的复权异常样本全部通过复核，共检验 373 个窗口内事件；其中此前不一致的事件为 182 个。

158 项源解析测试、15 项云端运行测试、80 项独立数据库契约测试、9 项 Worker 测试通过；类型检查、Clippy 和格式检查通过。

运行版本：research-cloud-v8-reviewed-anomalies。云端修复先在暂停且无运行中任务时执行八个代表样本，确认回执后才重新开放剩余失败任务。每次变化前，旧状态和旧尝试记录先上传 R2，并下载核对；既有自动重试次数不清零。

## 云端执行和验收

修复已部署至 Cloudflare。Worker 版本为 c68acc78-4cc6-43ee-a054-1e8789982453，镜像清单 SHA256 为 3a57b39432c4dd790d6b8b1fee9af787128a27eddd2f799be4df2d1fc1261db9。GET /runtime 返回 research-cloud-v8-reviewed-anomalies。

八个代表任务均完成分类处理：七个任务共 225 条实际数仓记录，逐任务行数与唯一记录下标计数一致；600849 无数仓记录，原有效空响应、重组文件和独立身份复核回执均保留。000598 首次云端重试遇到 TCP 连接超时，第二次成功取得 29 条记录，两次尝试分别留档。

2026-09-09 09:47 UTC，105 个剩余任务完成受控重新开放。独立 D1 比对确认，仅调度状态、下一次调度时间和更新时间变化，其余原字段逐项一致。R2 的恢复前后两份证据也已独立下载核验，包含 105 个任务和 107 条原尝试记录。09:49 UTC 恢复单路、预算 0 的持续采集。

全范围复验确认：11,624 个原任务身份及顺序一致，原来 9,732 个数仓发布回执及原件保持完整，没有重置尝试次数。此前正常采集期间 300708 日线从 BaoStock 自动切换到已验收的 Tushare 备源，原计划身份未变，实际来源与两次尝试单独保留。

10:29:34 UTC，全部原异常的入仓回执已齐全。全批次累计 9,931 个任务、10,600,994 行已入仓，云端状态为 active；后续正常任务继续采集，完整训练门槛仍关闭。两次东方财富临时连接失败均由单任务退避自动恢复，没有放宽数据校验，也没有触发整体暂停。

600656 的异常日期另用实际 R2 SQL 核查：源 tradestatus 仍为 1、成交量和成交额仍为空，lifecycle_annotation.training_eligible 为 false，官方最后交易日和证据 SHA256 一并入仓。

扩展抽查先验证 24 个已发布任务的 271 行无重复。832317、833994、920305、920493、920680 的新增云端归档另经独立下载、逐文件哈希及事件算术复验；920493 的完整公告索引有 660 条，920680 有 513 条，空分红接口未被直接用作零事件依据。107 条原尝试在重采后又经 D1 逐字段比对，完全保留。

## 证据与恢复

- [本地测试和离线事件复核](../artifacts/anomaly-fix-20260909-v1/local-validation-summary.json)
- [代表任务数仓核验](../artifacts/anomaly-fix-20260909-v1/canary-warehouse-verification.json)
- [重新开放的独立字段核验](../artifacts/anomaly-fix-20260909-v1/re_enrollment-independent-verification.json)
- [恢复前后 R2 对象核验](../artifacts/anomaly-fix-20260909-v1/repair-review-r2-verification.json)
- [最终全量范围和原回执核验](../artifacts/anomaly-fix-20260909-v1/scope-final.json)
- [全部原异常的最终数仓对账](../artifacts/anomaly-fix-20260909-v1/final-anomaly-warehouse-verification.json)
- [异常日期的实际数仓记录核验](../artifacts/anomaly-fix-20260909-v1/600656-warehouse-annotation-verification.json)
- [扩展数仓对账](../artifacts/anomaly-fix-20260909-v1/expanded-warehouse-verification.json)
- [原尝试保留的独立复验](../artifacts/anomaly-fix-20260909-v1/prior-attempts-independent-verification.json)
- [修复源码 R2 归档回执](../artifacts/anomaly-fix-20260909-v1/code-archive-receipt-completed.json)

修复源码位于原始桶 research/v1/operator-code/267690c8d11c1a9d4db04f1050c04ded8dc912c2b1278b0a68865daab32c4601.tar.gz，共 287 个文件，上传后下载恢复并核对哈希。恢复前后决策对象在 research/v1/review-evidence/full-a-share-20260907-v2 下，哈希分别为 b520c6d09fc8743948241c14ff4eb7488d2106a7fae96b9aace4783c750c6e56 和 fe60badde0c5ac76f38fc528381034ba2ac25ae779f4ffe0ae92356637b369f7。

云端新增原件和数仓数据持续写入 R2/Iceberg，不依赖当前电脑在线。代码尚未提交或推送；a-stock-data 未改动。本文是带时间戳的验收记录，最新进度仍以 D1 为准。数据可训练状态需完整数据覆盖和训练验收完成后确认。

最终验收包已存入原始桶 research/v1/archives/05ba614bf05c4dab9979fa3c18a96292ea3b44b64849df239019022fdd84525d.tar.gz，55 个文件、7,472,083 压缩字节，上传后完整下载恢复并独立逐文件核验。归档外生成的回执见 [验收包 R2 回执](../artifacts/anomaly-fix-20260909-v1/evidence-archive-receipt-completed.json) 和 [恢复后的独立复验](../artifacts/anomaly-fix-20260909-v1/evidence-completed-independent-verification.json)。
