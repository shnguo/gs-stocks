# A股未来事件雷达 MVP

日期：2026-09-22

## 已完成范围

本实现把事件研究接入现有 quant-research 工程，但不改动现有模型、检查点、
每日调度或交易逻辑。

已完成：

1. 事件标准化契约与显式来源等级。
2. SQLite 追加式修订库，保留每个事件的所有版本。
3. 原始发布时间、本地首次观测时间和实际可用时间分离。
4. 上交所定期报告预约页面，以及深交所官网指向的巨潮预约披露结构化数据采集。
5. 事件确定性、财务传导、预期差、资金影响和价格确认评分。
6. 拥挤、跳空和流动性三项扣分。
7. 五个已完成前复权收盘价的正式 MA5、ATR14、成交量和过度延伸门槛。
8. 观察、等待确认和回避三种研究状态。
9. 事件前与事件后的开盘到开盘研究统计。
10. 数据血缘、批次、原始文件哈希和缺失评分审计。
11. 未来60天、未来10天、未来3天和事件后5天四个每日列表。

## 关键文件

- src/quant_research/event_store.py：事件契约、修订库、时间点查询和审计。
- src/quant_research/event_sources.py：上交所和巨潮预约披露采集与原始响应留存。
- src/quant_research/event_radar.py：评分、技术门槛和雷达状态。
- src/quant_research/event_study.py：无未来数据的事件前后研究统计。
- src/quant_research/event_digest.py：每日四列表中文输出。
- tests/test_event_radar.py：防泄漏、成交时点、技术门槛和来源转换测试。
- configs/event-radar-example.json：通用事件导入示例。

## 时间点规则

每条事件同时保存：

- published_at：来源声称的首次发布时间。
- observed_at：本地首次看到这条记录的时间。
- availability_basis：选择哪一个时间作为研究可用时间。
- available_at：最终用于时间点查询和回测的可用时间。

只有来源提供可验证的历史发布时间时，才允许使用
verified_source_timestamp。上交所当前预约接口返回预约日期及变更后的日期，
但不返回每次变更被公开的准确时间。因此采集器使用 local_observation，绝不把
今天抓到的旧预约记录倒灌为历史上已知。

## 事件状态与评分

正向分满分100分：

- 确定性25分。
- 财务传导25分。
- 相对预期差20分。
- 资金或筹码影响15分。
- 价格确认15分。

扣分项分别为拥挤、跳空和流动性，每项最多20分。任一评分缺失时，系统不补默认值，
而是将事件标为研究不完整。

评分不是交易指令。事件前即使分数较高，也只进入观察。事件完成后还需要第一个完整
可交易收盘、正式 MA5、趋势、成交量、不过度延伸和至少2比1的收益风险比。
任何支撑价、目标价或失效价都必须显式声明为前复权口径，否则导入会失败。

## 命令顺序

以下命令在 quant-research 目录执行。

初始化事件库：

    PYTHONPATH=src .venv/bin/python -m quant_research.cli event-store-init data/events.sqlite

抓取一个上交所报告批次，并保留原始响应：

    PYTHONPATH=src .venv/bin/python -m quant_research.cli event-sse-periodic \
      data/events.sqlite artifacts/event-captures/sse-20260922 \
      --report 2026:L012 --observed-at 2026-09-22T09:00:00Z

抓取一个深市报告批次，并保留原始响应：

    PYTHONPATH=src .venv/bin/python -m quant_research.cli event-cninfo-periodic \
      data/events.sqlite artifacts/event-captures/cninfo-sz-20260922 \
      --section 2026-06-30 --market sz --observed-at 2026-09-22T10:00:00Z

导入人工准备或其他来源的标准化 JSON、JSONL 或 CSV：

    PYTHONPATH=src .venv/bin/python -m quant_research.cli event-ingest \
      data/events.sqlite configs/event-radar-example.json

导出某个历史时点可见的事件快照：

    PYTHONPATH=src .venv/bin/python -m quant_research.cli event-snapshot \
      data/events.sqlite --as-of 2026-09-22T09:00:00Z \
      --output artifacts/event-snapshot-20260922.json

构建事件雷达：

    PYTHONPATH=src .venv/bin/python -m quant_research.cli event-radar \
      data/events.sqlite path/to/bars.parquet \
      --as-of 2026-09-22T09:00:00Z --completed-through 2026-09-21 \
      --output artifacts/event-radar-20260922.csv

生成每日四列表：

    PYTHONPATH=src .venv/bin/python -m quant_research.cli event-digest \
      artifacts/event-radar-20260922.csv --radar-date 2026-09-22 \
      --completed-through 2026-09-21 \
      --output artifacts/event-digest-20260922.md

检查来源文件和修订完整性：

    PYTHONPATH=src .venv/bin/python -m quant_research.cli event-audit \
      data/events.sqlite --as-of 2026-09-22T09:00:00Z \
      --output artifacts/event-audit-20260922.json

分别运行事件前和事件后研究：

    PYTHONPATH=src .venv/bin/python -m quant_research.cli event-study \
      data/events.sqlite path/to/bars.parquet --as-of 2026-09-22T09:00:00Z \
      --mode post --horizon-sessions 5 --output artifacts/event-study-post5.csv

## 本轮真实烟雾测试

对上交所2026年半年报批次进行只读采集：

- 规范化事件：2321条。
- 指定实际披露日快照：278条。
- 与研究价格表匹配且具备技术计算历史：1668条。
- 研究价格表最新日期：2026-09-04。
- 事件本地首次观测时间：2026-09-22。

对深交所官网指向的巨潮预约披露页面进行深市2026年半年报只读采集：

- 规范化事件：2901条。
- 已实际披露：2900条。
- 原始响应文件哈希审计通过。
- 全部按本地首次观测时间进入研究库。

系统没有把9月4日以前的收盘当成9月22日观测事件的事后确认，并对陈旧价格、
缺失评分及未定义收益风险比保留了明确阻碍项。

## 验证

- 新增事件雷达测试：15项通过。
- 全项目回归：336项通过。
- Ruff 对 src 和 tests 的静态检查通过。
- 上交所官方接口真实只读采集、快照、雷达、审计和每日摘要命令均完成烟雾测试。

## 尚未完成

- 北交所预约披露自动采集的独立验证；巨潮采集器已支持 bj 参数但本轮未做真实烟雾测试。
- 解禁、回购、股东会和指数调仓的官方自动采集器。
- 对评分输入的独立研究和冻结流程。
- 行业、市值、波动率和动量匹配的对照组事件研究。
- 与 stock-indicator-dashboard 的页面集成。
- 每日调度、告警和任何实盘连接。

这些缺口意味着当前版本是一个可运行、可审计的研究骨架，不是完整生产事件平台。
