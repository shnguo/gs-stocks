# 正式成交快照与逐股票缺口

每日复盘为每个冻结信号生成 execution-data 下的不可变检查包。gap-report.json 提供股票级缺口和首轮前瞻验收状态，gaps.csv 提供完整逐日缺口。研究台显示最近信号的各榜单及其缺口。

这是研究侧的证据导入与验收器。已有行情采集器提供训练行情和证券资料，但尚无经过验证的完整成交证据；本轮没有新增生产采集器，也没有将训练快照升级为正式资料。

## 使用方式

在 quant-research 目录独立复盘，不启动训练或推理：

    PYTHONPATH=src:scripts .venv/bin/python scripts/review_daily_research.py \
      --store artifacts/daily-token-live-v2 \
      --output ../stock-indicator-dashboard/public/research/latest.json

已有完整证据时，追加 --execution-evidence /absolute/path/to/immutable-evidence-bundle。首次导入会登记完整身份，以后自动复用。修订资料应另建版本再显式导入，不能编辑已登记的目录。资料不完整时只生成缺口报告。

也可以检查单个已冻结的决策：

    PYTHONPATH=src:scripts .venv/bin/python scripts/build_execution_snapshot.py \
      --decision /absolute/path/to/decision-bundle \
      --source /absolute/path/to/outcome-input \
      --evidence /absolute/path/to/immutable-evidence-bundle

## 证据包契约

根目录的 execution-evidence.json 必须包含 schema_version（固定 1）、synthetic（显式布尔）、sources（来源 ID 映射）、tables（标准表映射）、coverage（覆盖 CSV 身份）。

每个 sources 项必填 path、sha256、source_url、observed_at、available_at、availability_basis。时间必须带时区，不能晚于实际导入时钟，available_at 不得晚于 observed_at。

- local_observation 表示采集时才可知，available_at 必须等于 observed_at；不能把今天采集的历史资料标成当时可见。
- verified_publication_archive 额外要求 publication_proof_path 和 publication_proof_sha256，保存经过核验的历史公开时间证明。

真实来源必须有 HTTP 或 HTTPS 来源链接。所有文件必须位于包内，哈希必须匹配。哈希只证明完整性；来源语义、资料穷尽性和历史公开时间仍由上游核验流程负责，不能靠填写标记获得认证。

tables 必须包含 instruments、calendar、bars、risk_events、risk_coverage、rules、actions。每项必填 path、sha256、source_ids、normalizer_path、normalizer_sha256；path 指向对应名称的 Parquet 文件。列定义见 src/quant_research/storage.py 中的 TABLES。归一化代码归档但不执行；生产采集和金融资料解析继续由 Rust 数据服务负责。

coverage 项必填 path 和 sha256。CSV 列为 component、instrument_id、start_date、end_date、known_at、source_id、status。

| component | 核验内容 | 所需可见时间 |
| --- | --- | --- |
| universe | 身份、上市、退市和完整股票池 | 原榜单发布时间 |
| industry | 当时的行业归属 | 原榜单发布时间 |
| calendar | 预测交易日历 | 执行日盘前 |
| risk | 风险状态及完整覆盖 | 信号发布时和各执行日 09:25 前 |
| rules | 成交单位、结算、费用与滑点 | 各执行日 09:25 前 |
| actions | 公司行动权利、支付日期及无事件覆盖 | 各执行日 09:25 前 |
| bars | 原始行情、成交金额、复权因子 | 历史输入在发布前，结果在完整收盘后 |
| limits | 确切的上下限价格 | 各执行日 09:25 前 |

status 只能为 verified_pit、reconstructed、unknown。只有 verified_pit 在所需时间前生效才通过。source_id 必须同时绑定相应标准表。较晚的未知修订不能被较早的已验证断言覆盖。

instrument_id 使用标准股票 ID；星号仅表示上游明确核验过整个股票池或板块的完整覆盖，不能表示“未发现异常”。空 actions 表仍需无事件覆盖证明。风险状态必须通过实际事件或当日完整覆盖表可恢复，未知状态始终受阻。

## 验收与范围

股票级检查固定使用各榜单前 20 名。行业权重、上市状态和信号风险检查覆盖完整冻结股票池。定仓所用的发布前 20 日历史必须与原冻结输入一致；不能用随后修订的资料重新定仓。

所有检查通过才生成 snapshot。检查包完整保留来源、覆盖文件、归一化代码及嵌套快照身份，可以独立恢复。execution_scope 将快照限定在该信号、候选与五日窗口，不能作为完整历史训练或回测资料。

迟发报告仍被排除，合成来源不计入真实前瞻样本。成熟且及时发布的完整快照进入首日开盘买入、第五日收盘退出的三档成本模拟；第五日无法退出的持仓留在终值，不延长到第六日。研究台分别显示窗口成熟和实际完成模拟的日期数。

## 发布时间预检

    scripts/run_daily_token.sh --preflight

此命令只读取已缓存、交叉核验的交易日历和耗时日志，不请求行情、不训练、不上传。返回截止时间、下一个发布窗口、新周期测量数、90% 分位耗时及最长阶段。缺少测量时不捏造耗时。

已有测量时，新运行保留 15 分钟发布余量；时间不足则停止启动训练。榜单发布后立即冻结候选与匹配证据，再执行成熟复盘。相同发布和协议的队列不会因复盘代码升级而重建。

本轮已检查四份真实日报并执行离线预检。真实资料仍不足以生成正式快照，首轮前瞻验收仍待及时发布、完整来源证据和第五个预测交易日成熟。
