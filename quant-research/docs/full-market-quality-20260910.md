# 全市场数据验收与候选导出

2026 年 9 月 10 日。本轮已进入全量质量验收与训练数据准备阶段。固定 Iceberg 快照为 3900223016037240817，范围保持为完整 A 股历史目录；ST 与未知状态保留。

## 已完成

- 对全部 11,623 个已发布任务做独立 R2 SQL 对账：11,528,589 行与 D1 一致，重复零。另 1 个 600849 任务经核验为历史区间不适用，保留原任务与证据。
- Rust 按固定快照扫描 11,463,243 条日线/停牌观察和 65,346 条因子记录，逐行核对来源、原始归档、哈希、行号、日期范围和业务主键。来源不符、重复与日期越界均为零。
- 产出 717 个按交易所与年份分区的 Parquet，共约 290 MB。11,459,144 行具备合法价格与当日或更早的因子，仍属于候选集。
- 保留 352,698 条 ST 标记、402,235 条未知 ST 状态和 216,135 条明确停牌观察。3 条停牌无价格及 1 条已核验最后交易日之后残留单独排除，未造价。
- 独立 Python 遍历全部文件核对哈希、数值、OHLC、空值、来源和因子日期；20,087 条既有样本逐字段一致。
- 新增 6 个相关 Rust 测试通过；锁定依赖构建、Clippy、格式与差异检查通过。新增 futures 直接依赖使用已有锁定版本，无版本升级。
- HTML 报告通过 1440px 与 390px、源数据交互和自包含检查。研究 Notebook 和脚本可重复运行。

## 已定位的后续问题

4,095 条观察缺少当日或更早的因子，涉及 58 个代码。其中 302132、001914、001872 占 4,017 条，即 98.1%；其日线覆盖早于当前代码因子起点。优先核验改码与来源历史回溯口径，不直接拼接身份或向前回填未来因子。

相对原计划和通用参考日历共有 8,987 个无来源观察的证券日，其中 2,675 个属于已核验不适用的 600849 历史区间。剩余 6,312 个参考缺口需按旧代码、停牌、交易所日历和真实漏采分类。原全量目录与原缺口台账均保留；该数字不等于确认漏采。

正式训练门槛仍关闭。剩余验收为历史证券身份与市场成员区间、交易所日历和缺失会话、全市场因子/公司行动、历史信息版本。完成后再冻结 60 日序列、5/20 日标签和隔离分区，启动基线及 Transformer 多种子训练。

## 产物

- [验收报告](../artifacts/full-market-quality-20260910-v1/package/artifacts/report/report.html)
- [完整扫描与文件哈希](../artifacts/full-market-quality-20260910-v1/package/artifacts/audit-v1/manifest.json)
- [独立验证](../artifacts/full-market-quality-20260910-v1/package/artifacts/audit-v1/independent-verification.json)
- [问题优先级](../artifacts/full-market-quality-20260910-v1/package/artifacts/audit-v1/review-priorities.json)
- [研究 Notebook](../artifacts/full-market-quality-20260910-v1/package/artifacts/verification.ipynb)
- [运行与恢复说明](../artifacts/full-market-quality-20260910-v1/package/docs/README.md)

本轮验收由本机 Rust 操作器读取 Cloudflare 固定快照，云端采集和数仓未重新启动或改写。源码已上传 R2 并在新目录恢复验证；数据、报告与脚本的最终 R2 凭据见本轮 data-receipt.json。历史快照、原件和任务身份继续保留。代码尚未提交或推送。

快照 manifest 的 plan_sha256 是本地完整计划文件的字节哈希；云端 campaign 的 plan_sha256 是同一计划规范 JSON 的哈希。两种表示在运行前经规范化比对一致，文件格式差异没有改变任何任务身份。

## R2 最终恢复凭据

数据、报告与复验脚本归档：research/v1/archives/c1723e1a09bce0656da476622811745e2c98b22cbc336095a94eb0f7ebdf221a.tar.gz。共 755 个文件，312,369,401 字节；R2 下载恢复及独立 Python 文件哈希核验全部通过。

操作器源码归档：research/v1/operator-code/70ec64413259cf6058f47febf4abf2d6781282771587f707115fb7e6527e5898.tar.gz。共 291 个文件，恢复验证通过。

恢复命令为 Rust 操作器 research-cloud restore，传入既有 operator.env、以上归档键和一个不存在的目标目录。研究原始数据仍在 Cloudflare 数仓与原始 R2 归档中。
