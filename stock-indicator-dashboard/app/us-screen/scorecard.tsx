"use client";

import { useEffect, useState } from "react";

type BasketOutcome = {
  listed: number; stocks: number; gross: number | null; net: number | null;
  excess_universe: number | null; excess_benchmark: number | null;
};
type Run = {
  as_of: string; status: "mature" | "interim" | "waiting"; timing: "prospective" | "late" | null;
  entry_date: string | null; exit_date: string | null; sessions_held: number; sessions_left: number;
  scored: number; priced: number; price_mismatches: number;
  universe_return: number | null; benchmark_return: number | null; rank_ic: number | null;
  quintiles: (number | null)[] | null; baskets: Record<string, BasketOutcome>;
};
type Stats = {
  runs: number; mean_net: number | null; mean_excess_universe: number | null;
  mean_excess_benchmark: number | null; share_beating_universe: number | null;
};
type RecordGroup = {
  runs: number; independent_runs: number; enough: boolean; mean_rank_ic_independent: number | null;
  baskets: Record<string, { all: Stats; independent: Stats }>;
};
type Basket = {
  key: string; label: string; note: string; primary: boolean;
  backtest: { months: number; mean_monthly_excess: number; monthly_sd: number; hit_rate: number } | null;
};
type Card = {
  schema_version: number; generated_at: string; version: string; last_session: string;
  holding_sessions: number; cost_round_trip: number; benchmark: string; min_independent_runs: number;
  baskets: Basket[]; runs: Run[]; summary: { prospective: RecordGroup; late: RecordGroup };
  source: { config: string; config_sha256: string; outcomes: string };
};

const MISSING = "还没有前瞻记录。在 quant-research 运行 scripts/us_scorecard.py（加 --dashboard）后，这里会显示每一批筛选结果之后的实际表现。";

function valid(value?: number | null): value is number { return typeof value === "number" && Number.isFinite(value); }
function pct(value?: number | null, digits = 2) { return valid(value) ? `${value > 0 ? "+" : ""}${(value * 100).toFixed(digits)}%` : "—"; }
function tone(value?: number | null) { return valid(value) ? (value > 0 ? "up" : value < 0 ? "down" : undefined) : undefined; }
function share(value?: number | null) { return valid(value) ? `${Math.round(value * 100)}%` : "—"; }

function statusLabel(run: Run) {
  if (run.status === "mature") return "已到期";
  if (run.status === "waiting") return "等待买入日开盘";
  return `进行中 · 还差 ${run.sessions_left} 个交易日`;
}

/** Forward record of the frozen screen runs; kept apart from the backtest evidence. */
export function Scorecard({ reload }: { reload: number }) {
  const [card, setCard] = useState<Card | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [chosen, setChosen] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    fetch("/us-screen/scorecard.json", { cache: "no-store", signal: controller.signal })
      .then(async (response) => {
        if (response.status === 404 || !response.headers.get("content-type")?.includes("application/json")) throw new Error(MISSING);
        if (!response.ok) throw new Error("前瞻记录暂时无法读取。");
        const data = await response.json() as Card;
        if (!data || data.schema_version !== 1 || !Array.isArray(data.runs) || !Array.isArray(data.baskets) || !data.summary) throw new Error("前瞻记录格式无法识别，请重新生成。");
        return data;
      })
      .then((data) => { if (!controller.signal.aborted) { setCard(data); setError(null); } })
      .catch((reason: unknown) => { if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : "前瞻记录读取失败。"); });
    return () => controller.abort();
  }, [reload]);

  if (error) return <section className="research-section"><h2>前瞻记录</h2><p className="research-empty" role="status">{error}</p></section>;
  if (!card) return <section className="research-section"><h2>前瞻记录</h2><p className="research-empty" role="status">正在读取前瞻记录…</p></section>;

  const basket = card.baskets.find((item) => item.key === chosen) ?? card.baskets.find((item) => item.primary) ?? card.baskets[0];
  const forward = card.summary.prospective;
  const late = card.summary.late;
  const record = basket ? forward.baskets[basket.key] : undefined;
  const runs = [...card.runs].sort((a, b) => b.as_of.localeCompare(a.as_of));
  const waitingForward = card.runs.filter((run) => run.status !== "mature" && run.timing !== "late");
  const next = waitingForward.length ? waitingForward.reduce((a, b) => (a.sessions_left <= b.sessions_left ? a : b)) : null;
  const cost = `${(card.cost_round_trip * 100).toFixed(2)}%`;
  const unfinished = card.runs.some((run) => run.status !== "mature");
  const groups = basket ? ([
    ["正式前瞻 · 全部到期批次", forward.baskets[basket.key]?.all], ["正式前瞻 · 互不重叠", forward.baskets[basket.key]?.independent],
    ["补记 · 全部到期批次", late.baskets[basket.key]?.all], ["补记 · 互不重叠", late.baskets[basket.key]?.independent],
  ] as [string, Stats | undefined][]).filter((row): row is [string, Stats] => !!row[1] && row[1].runs > 0) : [];

  return <section className="research-section scorecard" aria-labelledby="scorecard-title">
    <div className="research-section-heading"><h2 id="scorecard-title">前瞻记录</h2><span>行情截至 {card.last_session}</span></div>
    <p className="research-note">每一批筛选结果生成后就冻结。这里按固定规则给它记分：信号日收盘出结果，下一个交易日开盘买入，持有 {card.holding_sessions} 个交易日后在开盘卖出，等权，来回扣 {cost} 费用；再和同一批全部打分股票的等权收益、以及 {card.benchmark} 比较。规则在任何一批有结果之前就定好了，到期的结果只算一次、不再改。这是事后的真实表现，和“回测证据”分开记。</p>
    <dl className="research-metrics">
      <div><dt>已到期的正式批次</dt><dd>{forward.runs}</dd><small>其中互不重叠 {forward.independent_runs} 批 · 至少 {card.min_independent_runs} 批才下结论</small></div>
      <div><dt>互不重叠批次平均超额（对股票池）</dt><dd className={tone(record?.independent.mean_excess_universe)}>{pct(record?.independent.mean_excess_universe)}</dd><small>{basket?.backtest ? `回测里平均每月 ${pct(basket.backtest.mean_monthly_excess)}` : "这个篮子没有回测"}</small></div>
      <div><dt>下一批到期</dt><dd>{next ? `${next.sessions_left} 个交易日后` : "—"}</dd><small>{next ? `信号日 ${next.as_of}` : "没有进行中的正式批次"}</small></div>
    </dl>
    <p className="screen-warning">{forward.enough
      ? <>已有 {forward.independent_runs} 批互不重叠的正式批次到期，可以和回测对照了，但样本仍然很小。</>
      : <><strong>现在还不能下结论。</strong>{forward.runs ? `只有 ${forward.independent_runs} 批互不重叠的正式批次到期` : "还没有正式批次到期"}，至少要 {card.min_independent_runs} 批。{unfinished ? "下面“进行中”的数字是按最新收盘价的临时估值，每天会变，不计入记录。" : ""}</>}</p>

    <div className="screen-controls">
      <label>篮子<select value={basket?.key ?? ""} onChange={(event) => setChosen(event.target.value)}>{card.baskets.map((item) => <option key={item.key} value={item.key}>{item.label}</option>)}</select></label>
    </div>
    {basket ? <p className="research-note">{basket.note}{basket.backtest
      ? ` 回测（${basket.backtest.months} 个月）里这个篮子平均每月比股票池${basket.backtest.mean_monthly_excess >= 0 ? "高" : "低"} ${Math.abs(basket.backtest.mean_monthly_excess * 100).toFixed(2)}%，单月上下波动约 ${(basket.backtest.monthly_sd * 100).toFixed(1)}%，跑赢的月份占 ${share(basket.backtest.hit_rate)}。单独一批的结果几乎全是噪音；${card.min_independent_runs} 批互不重叠的批次平均下来，误差仍有约 ±${(basket.backtest.monthly_sd / Math.sqrt(card.min_independent_runs) * 100).toFixed(1)}%，只够发现明显的失效，不足以证明优势存在。`
      : " 这个篮子没有回测，前瞻记录是它唯一的证据。"}</p> : null}

    {groups.length ? <>
      <h3>到期批次汇总</h3>
      <div className="research-table-scroll"><table className="research-table">
        <caption className="sr-only">到期批次的平均表现</caption>
        <thead><tr><th scope="col">范围</th><th scope="col">批次数</th><th scope="col">平均净收益</th><th scope="col">平均超额 · 对股票池</th><th scope="col">平均超额 · 对 {card.benchmark}</th><th scope="col">跑赢股票池的批次</th></tr></thead>
        <tbody>{groups.map(([label, stats]) => <tr key={label}><td>{label}</td><td>{stats.runs}</td><td className={tone(stats.mean_net)}>{pct(stats.mean_net)}</td><td className={tone(stats.mean_excess_universe)}>{pct(stats.mean_excess_universe)}</td><td className={tone(stats.mean_excess_benchmark)}>{pct(stats.mean_excess_benchmark)}</td><td>{share(stats.share_beating_universe)}</td></tr>)}</tbody>
      </table></div>
      <p className="research-note">每天一批的结果互相重叠（持有期有 {card.holding_sessions} 个交易日），不能当成独立样本；“互不重叠”只取前一批卖出之后才买入的批次。补记的批次不计入正式记录。</p>
    </> : null}

    <h3>每一批的表现</h3>
    <div className="research-table-scroll"><table className="research-table screen-table">
      <caption className="sr-only">每一批筛选结果的前瞻表现</caption>
      <thead><tr><th scope="col">信号日</th><th scope="col">类型</th><th scope="col">状态</th><th scope="col">买入 → 卖出</th><th scope="col">篮子股票数</th><th scope="col">篮子净收益</th><th scope="col">股票池等权</th><th scope="col">{card.benchmark}</th><th scope="col">超额 · 对股票池</th><th scope="col">超额 · 对 {card.benchmark}</th><th scope="col">Rank IC</th></tr></thead>
      <tbody>{runs.map((run) => {
        const done = run.status === "mature";
        const result = basket ? run.baskets[basket.key] : undefined;
        const color = (value?: number | null) => (done ? tone(value) : undefined);
        return <tr key={run.as_of} className={done ? undefined : "scorecard-pending"}>
          <td><strong>{run.as_of}</strong></td>
          <td>{run.timing === "prospective" ? <span className="research-status">正式前瞻</span> : run.timing === "late" ? <span className="research-status late">补记 · 不计入</span> : "待定"}</td>
          <td>{statusLabel(run)}{done ? null : run.status === "interim" ? <small>买入后已过 {run.sessions_held} 个交易日 · 按最新收盘价临时估值</small> : null}</td>
          <td>{run.entry_date ?? "—"} → {run.exit_date ?? "未到期"}</td>
          <td>{result ? <>{result.stocks}<small>名单 {result.listed} 只</small></> : "—"}</td>
          <td className={color(result?.net)}>{result ? pct(result.net) : "这一批没有这个篮子"}</td>
          <td className={color(run.universe_return)}>{pct(run.universe_return)}</td>
          <td className={color(run.benchmark_return)}>{pct(run.benchmark_return)}</td>
          <td className={color(result?.excess_universe)}>{pct(result?.excess_universe)}</td>
          <td className={color(result?.excess_benchmark)}>{pct(result?.excess_benchmark)}</td>
          <td>{valid(run.rank_ic) ? run.rank_ic.toFixed(3) : "—"}</td>
        </tr>;
      })}</tbody>
    </table></div>
    <p className="research-note">正式前瞻 = 筛选结果在买入日开盘（纽约 09:30）之前就已生成并冻结。补记 = 结果是事后才生成的，无法证明没有看到后面的行情，只列出、不计入记录。灰色的行还没到期。篮子不足 5 只可买股票时不记分。Rank IC = 漂移排名和之后收益的排序相关，覆盖这一批全部打分股票。</p>

    <h3>按漂移分五组的收益</h3>
    <p className="research-note">每一批全部打分股票按漂移从弱到强分五组，各组等权、未扣费。排名有效时，应当大体从左到右升高；单独一批常常不是这样。</p>
    <div className="research-table-scroll"><table className="research-table">
      <caption className="sr-only">每一批按漂移五分组的收益</caption>
      <thead><tr><th scope="col">信号日</th><th scope="col">状态</th><th scope="col">最弱</th><th scope="col">第 2 组</th><th scope="col">第 3 组</th><th scope="col">第 4 组</th><th scope="col">最强</th><th scope="col">最强 − 最弱</th></tr></thead>
      <tbody>{runs.map((run) => {
        const done = run.status === "mature";
        const groups5 = run.quintiles ?? [null, null, null, null, null];
        const spread = valid(groups5[4]) && valid(groups5[0]) ? groups5[4] - groups5[0] : null;
        return <tr key={run.as_of} className={done ? undefined : "scorecard-pending"}>
          <td><strong>{run.as_of}</strong></td><td>{done ? "已到期" : run.status === "waiting" ? "等待买入" : "进行中"}</td>
          {groups5.map((value, index) => <td key={index} className={done ? tone(value) : undefined}>{pct(value)}</td>)}
          <td className={done ? tone(spread) : undefined}>{pct(spread)}</td>
        </tr>;
      })}</tbody>
    </table></div>

    <details className="scorecard-source"><summary>记录方法和局限</summary>
      <p>规则版本 {card.version}（配置 {card.source.config_sha256.slice(0, 12)}…）。到期结果连同每只股票的收益冻结在 {card.source.outcomes}，带 SHA-256 校验，之后只读取、不重算。</p>
      <p>“正式前瞻”按筛选结果清单里记录的生成时间判断，这个时间是本机自己写的，没有第三方证明。停止交易的股票按最后一个成交收盘价计；买入日没有成交或一字涨停的股票不计入。价格是拆股和分红复权后的价格，所以收益包含持有期内的分红（股票和 {card.benchmark} 都是）。</p>
      <p>研究记录，不是买卖建议。</p>
    </details>
  </section>;
}
