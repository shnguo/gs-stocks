"use client";

import { useEffect, useRef, useState } from "react";
import { PageHeader } from "../components/page-header";
import "./research.css";

type Quality = {
  field: string; label: string; expected: number; known: number; pending: number; missing: number;
  mean_absolute_percentage_error: number | null; mean_bias: number | null;
  baseline_mean_absolute_percentage_error: number | null;
  band_known: number; band_hits: number; band_hit_rate: number | null;
  direction_known: number; direction_hits: number; direction_agreement: number | null;
};
type Point = { date: string; actual: number | null; predicted: number | null; q10: number | null; q90: number | null; status: string };
type Candidate = {
  instrument_id: string; name: string | null; rank: number; board_label: string; industry: string | null;
  expected_net_return: number | null; model_disagreement: number | null;
  volatility_20: number | null; liquidity_20: number | null; early_stage_score: number | null;
  event_type?: string; event_score?: number; risks: string[]; anchor_close: number | null;
  quality: Quality; path: Point[]; trend: { status: string; return_5: number | null; return_20: number | null; ma20_distance: number | null };
};
type Publication = { prospective: boolean; published_at: string; deadline: string };
type Arm = {
  arm: string; label: string; candidate_count: number; ranking_population: number;
  completed_sessions: number; total_sessions: number; publication: Publication;
  quality: Quality; fields: Quality[]; horizons: (Quality & { date: string })[]; candidates: Candidate[];
  strata: Record<string, (Quality & { group: string; candidate_count: number })[]>;
};
type Report = {
  signal_date: string; horizon_dates: string[]; observed_through: string; synthetic: boolean;
  decision_sha256: string; manifest_sha256: string; arms: Arm[];
  state_labels: Record<string, string>; definitions: Record<string, string>;
};
type Overview = {
  scope: string; arm: string; signal_dates: number; known: number;
  mean_absolute_percentage_error: number | null; mean_bias: number | null; direction_agreement: number | null;
  baseline_mean_absolute_percentage_error: number | null;
};
type Summary = {
  schema_version: number; generated_at: string; observed_through: string | null; engineering_only: boolean;
  freshness: { expected_through: string | null; observed_through: string | null; missing_sessions: string[]; status: string };
  analysis_reports: Report[]; analysis_overview: Overview[];
  source_observations?: {
    manifest_sha256: string; sources: { name: string; label: string; source_url: string; symbol?: string;
      status: string; status_label: string; failure_label?: string; rows?: number; observed_at?: string }[];
  } | null;
};
const labels: Record<string, string> = { control: "收益主榜", start_stage: "上涨初期榜", event_overlay: "事件窗口榜", confirmed_event: "事件确认榜" };
const strataLabels: Record<string, string> = { board: "板块", industry: "行业", volatility_band: "历史波动率分组" };
const boards: Record<string, string> = { shanghai_main: "沪市主板", shenzhen_main: "深市主板", star: "科创板", chinext: "创业板", chi_next: "创业板", beijing: "北交所", unknown: "未知", nan: "未知" };
function valid(value?: number | null): value is number { return typeof value === "number" && Number.isFinite(value); }
function percent(value?: number | null) { return valid(value) ? `${(value * 100).toFixed(2)}%` : "—"; }
function price(value?: number | null) { return valid(value) ? value.toFixed(2) : "—"; }
function amount(value?: number | null) { return valid(value) ? `${(value / 100000000).toFixed(2)} 亿` : "—"; }
function timestamp(value: string) {
  return new Intl.DateTimeFormat("zh-CN", { timeZone: "Asia/Shanghai", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hourCycle: "h23" }).format(new Date(value));
}

function PriceChart({ candidate, signal }: { candidate: Candidate; signal: string }) {
  const anchor = candidate.anchor_close;
  if (!valid(anchor) || anchor <= 0) return <p className="research-empty">缺少信号日价格，暂不能绘制价格对照。</p>;
  const points = [{ date: signal, actual: anchor, predicted: anchor, q10: anchor, q90: anchor, status: "comparable" }, ...candidate.path];
  const values = points.flatMap((point) => [point.actual, point.predicted, point.q10, point.q90]).filter(valid).map((value) => value / anchor * 100);
  const lowest = Math.min(...values), highest = Math.max(...values), padding = Math.max((highest - lowest) * .12, 1);
  const minimum = lowest - padding, maximum = highest + padding;
  const x = (index: number) => 82 + index * 508 / Math.max(1, points.length - 1);
  const y = (value: number) => 190 - (value / anchor * 100 - minimum) / (maximum - minimum) * 160;
  const segments = (key: "actual" | "predicted") => {
    const result: string[] = [];
    let current: string[] = [];
    points.forEach((point, index) => {
      const value = point[key];
      if (valid(value)) current.push(`${current.length ? "L" : "M"}${x(index)},${y(value)}`);
      else if (current.length) { result.push(current.join(" ")); current = []; }
    });
    if (current.length) result.push(current.join(" "));
    return result;
  };
  return <div className="analysis-plot">
    <svg viewBox="0 0 640 236" role="img" aria-label={`${candidate.name || candidate.instrument_id}价格预测与后续收盘对照`}>
      <desc>信号日收盘价格归一为100。未取得或不可对照的实测价格留空，不连接缺失区间。</desc>
      {[minimum, (minimum + maximum) / 2, maximum].map((value) => <g key={value}><line className="analysis-grid" x1="82" x2="590" y1={y(value / 100 * anchor)} y2={y(value / 100 * anchor)} /><text x="73" y={y(value / 100 * anchor) + 4} textAnchor="end">{value.toFixed(1)}</text></g>)}
      {points.slice(1).map((point, offset) => {
        const previous = points[offset];
        if (![previous.q10, previous.q90, point.q10, point.q90].every(valid)) return null;
        return <polygon className="analysis-band" key={point.date} points={`${x(offset)},${y(previous.q10!)} ${x(offset + 1)},${y(point.q10!)} ${x(offset + 1)},${y(point.q90!)} ${x(offset)},${y(previous.q90!)}`} />;
      })}
      {segments("predicted").map((path, index) => <path key={index} className="analysis-predicted" d={path} />)}
      {segments("actual").map((path, index) => <path key={index} className="analysis-actual" d={path} />)}
      {points.map((point, index) => <g key={point.date}>{valid(point.actual) ? <circle className="analysis-dot" cx={x(index)} cy={y(point.actual)} r="3"><title>{point.date} 实测收盘 {price(point.actual)} 元</title></circle> : null}<text x={x(index)} y="219" textAnchor="middle">{point.date.slice(5)}</text></g>)}
    </svg>
    <div className="analysis-legend"><span className="analysis-legend-actual">实测收盘</span><span className="analysis-legend-predicted">模型参考收盘</span><span>阴影：模型分位带</span><span>信号日收盘 = 100</span></div>
  </div>;
}

export default function ResearchPage() {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [signal, setSignal] = useState("");
  const [selectedArm, setSelectedArm] = useState("control");
  const [stock, setStock] = useState("");
  const [scope, setScope] = useState("retrospective");
  const [reload, setReload] = useState(0);
  const detailHeading = useRef<HTMLHeadingElement>(null);
  function viewStock(instrument: string) {
    setStock(instrument);
    detailHeading.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    detailHeading.current?.focus({ preventScroll: true });
  }
  useEffect(() => {
    const controller = new AbortController();
    let timedOut = false;
    const timeout = window.setTimeout(() => { timedOut = true; controller.abort(); }, 15000);
    fetch("/research/latest.json", { cache: "no-store", signal: controller.signal })
      .then(async (response) => {
        if (response.status === 404 || !response.headers.get("content-type")?.includes("application/json")) throw new Error("尚未发布分析摘要。完成一次研究复盘后，这里会显示结果。");
        if (!response.ok) throw new Error("分析摘要暂时无法读取。");
        const data = await response.json() as Summary;
        if (!data || data.schema_version !== 1 || !Array.isArray(data.analysis_reports) || !Array.isArray(data.analysis_overview) || !data.freshness) throw new Error("分析摘要格式无法识别，请重新导出。");
        return data;
      })
      .then((data) => { if (!controller.signal.aborted) { setSummary(data); setError(null); } })
      .catch((reason: unknown) => { if (timedOut) setError("分析摘要读取超时，请刷新重试。"); else if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : "分析摘要读取失败。"); })
      .finally(() => window.clearTimeout(timeout));
    return () => { controller.abort(); window.clearTimeout(timeout); };
  }, [reload]);
  const report = summary?.analysis_reports.find((item) => item.signal_date === signal) ?? summary?.analysis_reports.at(-1);
  const arm = report?.arms.find((item) => item.arm === selectedArm) ?? report?.arms[0];
  const candidate = arm?.candidates.find((item) => item.instrument_id === stock) ?? arm?.candidates[0];
  const observations = summary?.source_observations;

  return <main className="page research-shell">
    <PageHeader eyebrow="研究复盘" title="每日研究分析" description="解释候选，核对价格预测，持续积累分析证据。"
      actions={<button type="button" className="btn" onClick={() => setReload((value) => value + 1)}>刷新摘要</button>} />
    <div aria-live="polite">
      {error ? <p className="research-empty" role="status">{error}</p> : null}
      {!summary && !error ? <p className="research-empty" role="status">正在读取分析摘要…</p> : null}
    </div>
    {summary ? <>
      {summary.engineering_only ? <p className="research-empty">当前为合成市场工程验证数据，不计入真实前瞻成绩。</p> : null}
      <div className="research-asof"><span>行情截至 <strong>{summary.observed_through || "未知"}</strong></span><span>摘要生成 {timestamp(summary.generated_at)} · 上海时间</span><a href="/research/latest.json" download="research-analysis.json">下载分析摘要</a></div>
      {summary.freshness.status !== "current" ? <p className="analysis-freshness">行情待更新：最近应已收盘的交易日为 {summary.freshness.expected_through || "未知"}，尚缺 {summary.freshness.missing_sessions.length} 个后续交易日。缺少的结果保留为空。</p> : null}
      {report && arm ? <>
        <section className="research-section" aria-labelledby="candidate-title">
          <div className="research-section-heading"><h2 id="candidate-title">发布时的候选榜</h2><label className="analysis-controls">信号日<select value={report.signal_date} onChange={(event) => { setSignal(event.target.value); setStock(""); }}>{[...summary.analysis_reports].reverse().map((item) => <option key={item.signal_date} value={item.signal_date}>{item.signal_date}</option>)}</select></label></div>
          <p className="research-note">预测交易日：{report.horizon_dates.join(" / ")}</p>
          <div className="research-tabs" role="group" aria-label="选择候选榜">{Object.entries(labels).map(([key, label]) => <button key={key} type="button" aria-pressed={arm.arm === key} disabled={!report.arms.some((item) => item.arm === key)} onClick={() => { setSelectedArm(key); setStock(""); }}>{label}{!report.arms.some((item) => item.arm === key) ? " · 未发布" : ""}</button>)}</div>
          <div className="research-publication"><span className={arm.publication.prospective ? "research-status" : "research-status late"}>{arm.publication.prospective ? "开盘前冻结" : "迟发研究报告"}</span><span>发布 {timestamp(arm.publication.published_at)} · 截止 {timestamp(arm.publication.deadline)} · 上海时间</span>{!arm.publication.prospective ? <span>可回顾价格，不计入前瞻成绩</span> : null}</div>
          <dl className="research-metrics">
            <div><dt>已对照收盘预测</dt><dd>{arm.quality.known} / {arm.quality.expected}<small>{arm.candidate_count} 只候选 × {arm.total_sessions} 个预测日</small></dd></div>
            <div><dt>收盘价格误差</dt><dd>{percent(arm.quality.mean_absolute_percentage_error)}<small>平均绝对百分比误差 · 越低越好</small></dd></div>
            <div><dt>涨跌方向一致</dt><dd>{percent(arm.quality.direction_agreement)}<small>{arm.quality.direction_hits} / {arm.quality.direction_known} 条可对照记录</small></dd></div>
          </dl>
          <p className="research-note">窗口已收盘 {arm.completed_sessions} / {arm.total_sessions} 日；等待收盘 {arm.quality.pending} 条，已收盘但资料待补齐 {arm.quality.missing} 条。价格误差与涨跌方向仅使用可对照记录。</p>
          <p className="research-note">下面保留原榜单的低价至未来高价参考收益，与收盘预测误差分开计算。候选按发布时冻结的排名展示，共 {arm.candidate_count} 只，原榜单 {arm.ranking_population} 只。</p>
          <div className="research-table-scroll"><table className="research-table"><caption className="sr-only">{arm.label}候选解释与分析入口</caption><thead><tr><th scope="col">排名</th><th scope="col">股票</th><th scope="col">预期参考收益</th><th scope="col">模型分歧</th><th scope="col">过去5日涨跌</th><th scope="col">距20日均线</th><th scope="col">收盘对照</th><th scope="col">详情</th></tr></thead><tbody>{arm.candidates.map((row) => <tr key={row.instrument_id} className={candidate?.instrument_id === row.instrument_id ? "analysis-selected" : undefined}><td>{row.rank}</td><td><strong>{row.name || row.instrument_id}</strong><small>{row.board_label} · {row.instrument_id}</small></td><td>{percent(row.expected_net_return)}</td><td>{percent(row.model_disagreement)}</td><td>{percent(row.trend.return_5)}</td><td>{percent(row.trend.ma20_distance)}</td><td>{row.quality.known} / {row.quality.expected}</td><td><button className="analysis-stock-button" type="button" aria-label={`查看${row.name || row.instrument_id}分析`} aria-pressed={candidate?.instrument_id === row.instrument_id} onClick={() => viewStock(row.instrument_id)}>查看</button></td></tr>)}</tbody></table></div>
          {!arm.candidates.length ? <p className="research-empty">本次没有发布候选股票。</p> : null}
        </section>
        {candidate ? <section className="research-section" aria-labelledby="stock-title">
          <div className="research-section-heading"><h2 id="stock-title" ref={detailHeading} tabIndex={-1}>{candidate.name || candidate.instrument_id} · 价格对照</h2><label className="analysis-controls">选择股票<select value={candidate.instrument_id} onChange={(event) => setStock(event.target.value)}>{arm.candidates.map((row) => <option key={row.instrument_id} value={row.instrument_id}>{row.rank}. {row.name || row.instrument_id}</option>)}</select></label></div>
          <p className="research-note">{candidate.event_type ? `${candidate.event_type} · 事件分 ${candidate.event_score ?? "—"}` : arm.arm === "start_stage" ? `上涨初期评分 ${price(candidate.early_stage_score)} · 原榜单第 ${candidate.rank} 名` : `按模型参考收益排序 · 原榜单第 ${candidate.rank} 名`}。模型分歧反映各模型参考收益的差异。</p>
          <dl className="analysis-facts"><div><dt>信号日收盘</dt><dd>{price(candidate.anchor_close)} 元</dd></div><div><dt>过去20日涨跌</dt><dd>{percent(candidate.trend.return_20)}</dd></div><div><dt>20日日收益波动率</dt><dd>{percent(candidate.volatility_20)}</dd></div><div><dt>20日平均成交额</dt><dd>{amount(candidate.liquidity_20)} 元</dd></div></dl>
          {candidate.risks.length ? <ul className="analysis-risks">{candidate.risks.map((risk) => <li key={risk}>{risk}</li>)}</ul> : null}
          <PriceChart candidate={candidate} signal={report.signal_date} />
          <p className="research-note">参考收盘为冻结的各模型中位预测均值。{report.definitions.band}</p><p className="research-note">{report.definitions.boundary}</p>
          <div className="research-table-scroll"><table className="research-table"><caption className="sr-only">{candidate.name}逐日收盘预测与实测价格</caption><thead><tr><th scope="col">预测交易日</th><th scope="col">参考收盘 · 元</th><th scope="col">模型分位带 · 元</th><th scope="col">实测收盘 · 元</th><th scope="col">对照状态</th></tr></thead><tbody>{candidate.path.map((point) => <tr key={point.date}><td>{point.date}</td><td>{price(point.predicted)}</td><td>{price(point.q10)} ～ {price(point.q90)}</td><td>{price(point.actual)}</td><td>{report.state_labels[point.status] || point.status}</td></tr>)}</tbody></table></div>
        </section> : null}
        <section className="research-section" aria-labelledby="quality-title"><h2 id="quality-title">价格预测质量 · {arm.label}</h2><p className="research-note">{report.definitions.price_error}偏差为正表示预测偏高。{report.definitions.direction}</p>
          <div className="research-table-scroll"><table className="research-table"><caption className="sr-only">不同价格字段的预测质量</caption><thead><tr><th scope="col">价格字段</th><th scope="col">可对照 / 总记录</th><th scope="col">模型价格误差</th><th scope="col">朴素基线误差</th><th scope="col">平均预测偏差</th><th scope="col">落入模型分位带</th></tr></thead><tbody>{arm.fields.map((row) => <tr key={row.field}><td>{row.label}</td><td>{row.known} / {row.expected}</td><td>{percent(row.mean_absolute_percentage_error)}</td><td>{percent(row.baseline_mean_absolute_percentage_error)}</td><td>{percent(row.mean_bias)}</td><td>{percent(row.band_hit_rate)}<small>{row.band_hits} / {row.band_known} 条有完整分位带的记录</small></td></tr>)}</tbody></table></div>
          <p className="research-note">{report.definitions.baseline}误差越低越好。</p>
          <h3>逐预测日收盘质量</h3><div className="research-table-scroll"><table className="research-table"><thead><tr><th scope="col">预测日</th><th scope="col">可对照 / 总记录</th><th scope="col">模型价格误差</th><th scope="col">朴素基线误差</th><th scope="col">预测偏差</th><th scope="col">方向一致</th></tr></thead><tbody>{arm.horizons.map((row, index) => <tr key={row.date}><td>T+{index} · {row.date}</td><td>{row.known} / {row.expected}</td><td>{percent(row.mean_absolute_percentage_error)}</td><td>{percent(row.baseline_mean_absolute_percentage_error)}</td><td>{percent(row.mean_bias)}</td><td>{percent(row.direction_agreement)}</td></tr>)}</tbody></table></div>
          <details className="research-strata"><summary>查看板块、行业和波动率分组</summary>{Object.entries(arm.strata).map(([key, values]) => <div key={key}><h3>{strataLabels[key] || key}</h3>{key === "volatility_band" ? <p className="research-note">按信号日有模型排名的股票分为五组，q1为波动率最低组，q5为最高组。</p> : null}<div className="research-table-scroll"><table className="research-table"><thead><tr><th scope="col">分组</th><th scope="col">候选数</th><th scope="col">收盘对照 / 总记录</th><th scope="col">价格误差</th><th scope="col">预测偏差</th></tr></thead><tbody>{values.map((row) => <tr key={row.group}><td>{boards[row.group] || row.group}</td><td>{row.candidate_count}</td><td>{row.known} / {row.expected}</td><td>{percent(row.mean_absolute_percentage_error)}</td><td>{percent(row.mean_bias)}</td></tr>)}</tbody></table></div></div>)}</details>
        </section>
      </> : <p className="research-empty">尚无已发布的候选榜。</p>}
      <section className="research-section" aria-labelledby="overview-title"><h2 id="overview-title">跨信号日分析</h2><div className="research-tabs" role="group" aria-label="选择研究样本口径"><button type="button" aria-pressed={scope === "retrospective"} onClick={() => setScope("retrospective")}>迟发回顾</button><button type="button" aria-pressed={scope === "prospective"} onClick={() => setScope("prospective")}>有效前瞻</button></div><p className="research-note">按可对照价格记录加权汇总。迟发报告与前瞻报告分开统计，合成数据不计入。重叠窗口不是独立样本，不同榜单的差异不能直接归因于筛选方法。</p><div className="research-table-scroll"><table className="research-table"><caption className="sr-only">{scope === "prospective" ? "有效前瞻" : "迟发回顾"}预测质量汇总</caption><thead><tr><th scope="col">候选榜</th><th scope="col">信号日期数</th><th scope="col">可对照收盘记录</th><th scope="col">模型价格误差</th><th scope="col">朴素基线误差</th><th scope="col">预测偏差</th><th scope="col">方向一致</th></tr></thead><tbody>{summary.analysis_overview.filter((row) => row.scope === scope).map((row) => <tr key={row.arm}><td>{labels[row.arm] || row.arm}</td><td>{row.signal_dates}</td><td>{row.known}</td><td>{percent(row.mean_absolute_percentage_error)}</td><td>{percent(row.baseline_mean_absolute_percentage_error)}</td><td>{percent(row.mean_bias)}</td><td>{percent(row.direction_agreement)}</td></tr>)}</tbody></table></div></section>
      <section className="research-section"><h2>历史发布与对照进度</h2><div className="research-table-scroll"><table className="research-table"><thead><tr><th scope="col">信号日</th><th scope="col">候选榜</th><th scope="col">样本口径</th><th scope="col">窗口已收盘</th><th scope="col">收盘对照 / 总记录</th><th scope="col">价格误差</th></tr></thead><tbody>{summary.analysis_reports.flatMap((item) => item.arms.map((group) => <tr key={`${item.signal_date}-${group.arm}`}><td>{item.signal_date}</td><td>{group.label}</td><td>{item.synthetic ? "合成验证" : group.publication.prospective ? "有效前瞻" : "迟发回顾"}</td><td>{group.completed_sessions} / {group.total_sessions}</td><td>{group.quality.known} / {group.quality.expected}</td><td>{percent(group.quality.mean_absolute_percentage_error)}</td></tr>))}</tbody></table></div></section>
      {observations ? <details className="research-section analysis-sources"><summary>辅助资料与来源</summary><p className="research-note">保留实际获取时间。后采集的行业或风险资料不会填回较早的冻结判断。</p><div className="research-table-scroll"><table className="research-table"><thead><tr><th scope="col">来源 / 股票</th><th scope="col">采集状态</th><th scope="col">记录数</th><th scope="col">获取时间 · 上海</th></tr></thead><tbody>{observations.sources.filter((row) => !["skipped_api_failure", "not_requested_budget"].includes(row.status)).map((row) => <tr key={row.name}><td><a href={row.source_url} target="_blank" rel="noreferrer">{row.label}</a><small>{row.symbol || "市场列表"}</small></td><td>{row.failure_label || row.status_label}</td><td>{row.rows ?? "—"}</td><td>{row.observed_at ? timestamp(row.observed_at) : "—"}</td></tr>)}</tbody></table></div><p className="research-note">资料证据：{observations.manifest_sha256}</p></details> : null}
      {report ? <footer className="research-footer"><details><summary>分析证据摘要</summary><p>冻结候选：{report.decision_sha256}</p><p>价格分析：{report.manifest_sha256}</p></details></footer> : null}
    </> : null}
  </main>;
}
