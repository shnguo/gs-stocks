"use client";

import { Fragment, useEffect, useMemo, useState } from "react";
import { PageHeader } from "../components/page-header";
import "../research/research.css";
import "./us-screen.css";
import { Scorecard } from "./scorecard";

type Stock = {
  symbol: string; name: string | null; exchange: string | null; instrument_id: string | null;
  sector: string | null; sic_description?: string | null;
  rank?: number; drift_percentile?: number | null; drift_quintile?: number | null; drift_sector_percentile?: number | null;
  controlled_drift_percentile?: number | null; drift?: number | null; momentum_12_1?: number | null;
  jump_return?: number | null; jump_days?: number | null; return_1m?: number | null; daily_volatility?: number | null;
  avg_traded_value_musd?: number | null; last_close?: number | null;
  in_sp500?: boolean; in_ndx?: boolean; in_r1000?: boolean; extra_only?: boolean;
  ma50_gap?: number | null; return_5d?: number | null; gap_down_1m?: boolean | null; pullback?: boolean | null; pullback_mild?: boolean | null;
  drawdown_12m?: number | null; delta_roa?: number | null; revenue_growth?: number | null; net_income_ttm?: number | null;
  net_income_ttm_prev?: number | null; operating_cash_ttm?: number | null; fscore?: number | null; fscore_signals?: number | null;
  fund_turnaround?: boolean | null; fund_profit_turn?: boolean | null; fund_deteriorating?: boolean | null;
};
type BasketEvidence = {
  key: string; label: string; annual_excess: number | null; volatility: number | null; max_drawdown: number | null;
  hit_rate: number | null; monthly_ci: (number | null)[]; losing_years: number; years: number; median_stocks: number | null; months: number;
};
type ShapeEvidence = {
  key: string; shape: string; shape_label: string; period: string; period_label: string; note: string;
  annual_excess: number | null; volatility: number | null; max_drawdown: number | null;
  losing_years: number; years: number; monthly_ci: (number | null)[]; mean_stocks: number | null;
};
type Payload = {
  schema_version: number; generated_at: string; as_of: string; runs: string[]; membership_date: string; window: string;
  scored: number; unscored: number; jump_sigma: number; top_count: number;
  sector_mix: { sector: string; all: number; top: number }[];
  stocks: Stock[]; shape_matches: Record<string, { symbol: string; score: number | null }[]>;
  evidence: {
    period: string;
    factors: { label: string; ic: number | null; top_excess: number | null; long_short: number | null; ic_ci?: (number | null)[] }[];
    trend_filters: { label: string; top_excess: number | null }[];
    shapes: ShapeEvidence[];
    pullback?: BasketEvidence[];
    fundamental?: BasketEvidence[];
    fundamental_warning?: string;
  };
  source: { run: string; manifest_sha256: string };
};
type SortKey = "drift_percentile" | "drift_sector_percentile" | "controlled_drift_percentile" | "return_1m";
type FavoriteState = { status: "working" | "added" | "error"; message?: string };

const sortLabels: Record<SortKey, string> = {
  drift_percentile: "漂移百分位（主排名）", drift_sector_percentile: "行业内百分位",
  controlled_drift_percentile: "控制后百分位", return_1m: "近 1 个月涨跌",
};
const shapeOrder = ["rising_now", "turning_up", "long_uptrend"];
const periodOrder = ["1m", "3m", "6m", "12m"];

function valid(value?: number | null): value is number { return typeof value === "number" && Number.isFinite(value); }
function pct(value?: number | null, digits = 1) { return valid(value) ? `${value > 0 ? "+" : ""}${(value * 100).toFixed(digits)}%` : "—"; }
function plain(value?: number | null, digits = 2) { return valid(value) ? value.toFixed(digits) : "—"; }
function tone(value?: number | null) { return valid(value) ? (value > 0 ? "up" : value < 0 ? "down" : undefined) : undefined; }
function musd(value?: number | null) { return valid(value) ? `${(value / 1e6).toLocaleString("en-US", { maximumFractionDigits: 0 })} 百万美元` : "—"; }
function timestamp(value: string) {
  return new Intl.DateTimeFormat("zh-CN", { timeZone: "Asia/Shanghai", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hourCycle: "h23" }).format(new Date(value));
}

function StockCell({ stock }: { stock: Stock }) {
  const indexes = [stock.in_sp500 ? "标普500" : null, stock.in_ndx ? "纳指100" : null, stock.in_r1000 ? "罗素1000" : null].filter(Boolean);
  return <td><strong>{stock.symbol}</strong><small>{stock.name || "—"}{indexes.length ? ` · ${indexes.join(" / ")}` : ""}</small></td>;
}

function FavoriteButton({ stock, state, onAdd }: { stock: Stock; state?: FavoriteState; onAdd: (stock: Stock) => void }) {
  const label = state?.status === "added" ? "已加入" : state?.status === "working" ? "收录中…" : state?.status === "error" ? "重试" : "加入自选";
  return <div className="screen-favorite-cell">
    <button type="button" className="screen-favorite" disabled={state?.status === "added" || state?.status === "working"} aria-label={`把 ${stock.symbol} 加入自选`} onClick={() => onAdd(stock)}>{label}</button>
    {state?.message && state.status !== "added" ? <small className={state.status === "error" ? "screen-favorite-error" : undefined}>{state.message}</small> : null}
  </div>;
}

const PENDING_KEY = "stock-indicator-dashboard.us-screen.pending-admissions:v1";
const ADMISSION_WAIT_MS = 30 * 60 * 1000;
type Pending = Record<string, { requestId: string; startedAt: number }>;

function readPending(): Pending {
  try {
    const value: unknown = JSON.parse(window.localStorage.getItem(PENDING_KEY) || "{}");
    return value && typeof value === "object" && !Array.isArray(value) ? value as Pending : {};
  } catch {
    return {};
  }
}

function writePending(update: (pending: Pending) => void) {
  try {
    const pending = readPending();
    update(pending);
    window.localStorage.setItem(PENDING_KEY, JSON.stringify(pending));
  } catch {
    // Without storage the request still completes while this page stays open.
  }
}

function elapsed(startedAt: number) {
  const minutes = Math.floor((Date.now() - startedAt) / 60000);
  return minutes < 1 ? "不到 1 分钟" : `${minutes} 分钟`;
}

type CatalogMatch = { instrumentId: string; code: string; historicalStatus: string };

async function readJson(response: Response): Promise<Record<string, unknown>> {
  const payload: unknown = await response.json().catch(() => null);
  return payload && typeof payload === "object" && !Array.isArray(payload) ? payload as Record<string, unknown> : {};
}

/** The dashboard catalog's own id for a US ticker, or null when it is not listed yet. */
async function findInCatalog(symbol: string): Promise<CatalogMatch | null> {
  const response = await fetch(`/api/stock-search?${new URLSearchParams({ q: symbol, market: "us" })}`, { cache: "no-store" });
  const record = await readJson(response);
  if (!response.ok) throw new Error(typeof record.error === "string" ? record.error : "股票目录查询失败");
  const results = Array.isArray(record.results) ? record.results as CatalogMatch[] : [];
  return results.find((row) => typeof row.code === "string" && row.code.toUpperCase() === symbol.toUpperCase()) ?? null;
}

/** Ask the collector to admit an unlisted ticker (same flow as the main page's 添加股票). */
async function submitAdmission(symbol: string): Promise<string> {
  const response = await fetch("/api/symbol-admissions", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ query: symbol, market: "us" }) });
  const record = await readJson(response);
  if (!response.ok) throw new Error(typeof record.error === "string" ? record.error : "股票收录申请失败");
  const requestId = typeof record.request_id === "string" ? record.request_id : "";
  if (!/^[0-9a-f]{32}$/u.test(requestId)) throw new Error("股票收录申请编号无效");
  return requestId;
}

/** Poll an admission until the instrument is active; returns its catalog id. */
async function waitForAdmission(symbol: string, requestId: string, startedAt: number,
  onProgress: (message: string) => void): Promise<string> {
  for (let attempt = 0; Date.now() - startedAt < ADMISSION_WAIT_MS; attempt += 1) {
    await new Promise((resolve) => window.setTimeout(resolve, attempt === 0 ? 900 : 5000));
    const poll = await fetch(`/api/symbol-admissions/${requestId}`, { cache: "no-store" });
    const state = await readJson(poll);
    if (!poll.ok) throw new Error(typeof state.error === "string" ? state.error : "无法读取股票收录进度");
    const status = typeof state.status === "string" ? state.status : "";
    if (status === "active") {
      const candidate = state.candidate && typeof state.candidate === "object" ? state.candidate as Record<string, unknown> : {};
      const id = typeof candidate.instrument_id === "string" ? candidate.instrument_id : "";
      if (!id) throw new Error("股票已完成收录，但返回的标的信息无效");
      return id;
    }
    if (["rejected", "unsupported", "quota-blocked", "rights-blocked", "failed", "inactive"].includes(status)) {
      const reason = typeof state.reason === "string" ? state.reason : "";
      if (/storage backend error|dispatch failure/u.test(reason)) {
        throw new Error("后台保存历史行情时连接存储失败（收录服务的问题，不是这只股票的问题）。稍后点“重试”会重新收录");
      }
      throw new Error(reason || "该股票暂时无法完成收录");
    }
    onProgress(status === "bootstrapping"
      ? `代码已确认，正在生成历史走势（已 ${elapsed(startedAt)}，通常需要几分钟，失败会在 15 分钟后自动重试）`
      : `收录申请排队中（已 ${elapsed(startedAt)}）`);
  }
  throw new Error("收录超过 30 分钟仍未完成，稍后回到本页会继续等待");
}

export default function UsScreenPage() {
  const [payload, setPayload] = useState<Payload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reload, setReload] = useState(0);
  const [view, setView] = useState<"ranking" | "pullback" | "shapes" | "evidence" | "scorecard">("ranking");
  const [mildOnly, setMildOnly] = useState(true);
  const [sector, setSector] = useState("全部");
  const [sortKey, setSortKey] = useState<SortKey>("drift_percentile");
  const [limit, setLimit] = useState("50");
  const [risingOnly, setRisingOnly] = useState(false);
  const [shapeKey, setShapeKey] = useState("turning_up_3m");
  const [favorites, setFavorites] = useState<Record<string, FavoriteState>>({});  // keyed by ticker
  const [favoriteNote, setFavoriteNote] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    let timedOut = false;
    const timeout = window.setTimeout(() => { timedOut = true; controller.abort(); }, 15000);
    fetch("/us-screen/latest.json", { cache: "no-store", signal: controller.signal })
      .then(async (response) => {
        if (response.status === 404 || !response.headers.get("content-type")?.includes("application/json")) throw new Error("尚未发布美股筛选结果。在 quant-research 运行筛选并导出后，这里会显示结果。");
        if (!response.ok) throw new Error("筛选结果暂时无法读取。");
        const data = await response.json() as Payload;
        if (!data || data.schema_version !== 1 || !Array.isArray(data.stocks) || !data.evidence) throw new Error("筛选结果格式无法识别，请重新导出。");
        return data;
      })
      .then((data) => { if (!controller.signal.aborted) { setPayload(data); setError(null); } })
      .catch((reason: unknown) => { if (timedOut) setError("筛选结果读取超时，请刷新重试。"); else if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : "筛选结果读取失败。"); })
      .finally(() => window.clearTimeout(timeout));
    return () => { controller.abort(); window.clearTimeout(timeout); };
  }, [reload]);

  useEffect(() => {
    const controller = new AbortController();
    void fetch("/api/favorites", { cache: "no-store", signal: controller.signal })
      .then(async (response) => response.ok ? response.json() as Promise<unknown> : null)
      .then((data) => {
        const rows = data && typeof data === "object" && !Array.isArray(data) ? (data as Record<string, unknown>).favorites : null;
        if (!Array.isArray(rows)) return;
        const symbols = rows.flatMap((row) => {
          const record = row && typeof row === "object" ? row as Record<string, unknown> : {};
          return record.market === "us" && typeof record.symbol === "string" ? [record.symbol.toUpperCase()] : [];
        });
        setFavorites((current) => ({ ...Object.fromEntries(symbols.map((symbol) => [symbol, { status: "added" as const }])), ...current }));
      })
      .catch(() => undefined);
    return () => controller.abort();
  }, []);

  async function saveFavorite(symbol: string, instrumentId: string) {
    const response = await fetch("/api/favorites", { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify({ instrument_id: instrumentId }) });
    if (!response.ok) {
      const record = await readJson(response);
      throw new Error(typeof record.error === "string" ? record.error : `加入自选失败（${response.status}）`);
    }
    setFavorites((current) => ({ ...current, [symbol]: { status: "added" } }));
    setFavoriteNote(`${symbol} 已加入自选，回到行情研究台即可查看走势。`);
  }

  async function finishAdmission(symbol: string, requestId: string, startedAt: number) {
    const mark = (state: FavoriteState) => setFavorites((current) => ({ ...current, [symbol]: state }));
    try {
      const instrumentId = await waitForAdmission(symbol, requestId, startedAt,
        (message) => mark({ status: "working", message }));
      writePending((pending) => { delete pending[symbol]; });
      await saveFavorite(symbol, instrumentId);
    } catch (reason: unknown) {
      const message = reason instanceof Error ? reason.message : "收录失败";
      if (!message.includes("30 分钟")) writePending((pending) => { delete pending[symbol]; });
      mark({ status: "error", message });
      setFavoriteNote(`${symbol}：${message}`);
    }
  }

  async function addFavorite(stock: Stock) {
    const symbol = stock.symbol.toUpperCase();
    const mark = (state: FavoriteState) => setFavorites((current) => ({ ...current, [symbol]: state }));
    mark({ status: "working", message: "正在查找…" });
    try {
      const match = await findInCatalog(symbol);
      if (match && match.historicalStatus !== "ready") throw new Error("行情数据还在准备中，请稍后再试");
      if (match) {
        await saveFavorite(symbol, match.instrumentId);
        return;
      }
      const startedAt = Date.now();
      const requestId = await submitAdmission(symbol);
      writePending((pending) => { pending[symbol] = { requestId, startedAt }; });
      mark({ status: "working", message: "尚未收录，已提交收录申请，正在核验代码…" });
      setFavoriteNote(`${symbol} 尚未收录，已提交收录申请。完成后会自动加入自选；离开本页也不会丢失，回来后会继续。`);
      await finishAdmission(symbol, requestId, startedAt);
    } catch (reason: unknown) {
      const message = reason instanceof Error ? reason.message : "加入自选失败";
      mark({ status: "error", message });
      setFavoriteNote(`${symbol}：${message}`);
    }
  }

  // Resume admissions started earlier (page reload or revisit).
  useEffect(() => {
    const timer = window.setTimeout(() => {
      for (const [symbol, item] of Object.entries(readPending())) {
        if (!item || typeof item.requestId !== "string") continue;
        setFavorites((current) => ({ ...current, [symbol]: { status: "working", message: "继续等待收录…" } }));
        void finishAdmission(symbol, item.requestId, Number(item.startedAt) || Date.now());
      }
    }, 0);
    return () => window.clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const bySymbol = useMemo(() => new Map((payload?.stocks ?? []).map((stock) => [stock.symbol, stock])), [payload]);
  const ranked = useMemo(() => (payload?.stocks ?? []).filter((stock) => !stock.extra_only), [payload]);
  const sectors = useMemo(() => ["全部", ...(payload?.sector_mix ?? []).map((row) => row.sector)], [payload]);
  const rows = useMemo(() => {
    let list = ranked.filter((stock) => sector === "全部" || (stock.sector || "未知") === sector);
    if (risingOnly) list = list.filter((stock) => valid(stock.return_1m) && stock.return_1m > 0);
    list = [...list].sort((a, b) => (b[sortKey] ?? -Infinity) - (a[sortKey] ?? -Infinity));
    if (limit === "top") list = list.filter((stock) => stock.drift_quintile === 5);
    else if (limit !== "all") list = list.slice(0, Number(limit));
    return list;
  }, [ranked, sector, risingOnly, sortKey, limit]);
  const shapeEvidence = payload?.evidence.shapes.find((row) => row.key === shapeKey);
  const shapeRows = (payload?.shape_matches[shapeKey] ?? []).map((match) => ({ match, stock: bySymbol.get(match.symbol) })).filter((row): row is { match: { symbol: string; score: number | null }; stock: Stock } => !!row.stock);
  // "Mild" is exactly the backtested 0–10% basket (no moving-average condition).
  const pullbackRows = ranked.filter((stock) => mildOnly ? stock.pullback_mild : stock.pullback || stock.pullback_mild);
  const pullbackAll = ranked.filter((stock) => stock.pullback).length;
  const pullbackMild = ranked.filter((stock) => stock.pullback_mild).length;
  const turnaroundRows = ranked.filter((stock) => stock.fund_turnaround).sort((a, b) => (a.drawdown_12m ?? 0) - (b.drawdown_12m ?? 0));
  const deteriorating = ranked.filter((stock) => stock.fund_deteriorating);
  const pullbackEvidence = new Map((payload?.evidence.pullback ?? []).map((row) => [row.key, row]));
  const topFallen = ranked.filter((stock) => (stock.rank ?? Infinity) <= 50);
  const fallenShare = topFallen.length ? topFallen.filter((stock) => valid(stock.return_1m) && stock.return_1m < 0).length / topFallen.length : null;

  return <main className="page research-shell screen-shell">
    <PageHeader eyebrow="美股筛选" title="美股动量筛选" description="把过去一年的走势拆成“跳动日”和其余日子的“平稳漂移”，按漂移给标普 500、纳斯达克 100、罗素 1000 成分股排名。研究用排名，不是买卖建议。"
      actions={<button type="button" className="btn" onClick={() => setReload((value) => value + 1)}>刷新结果</button>} />
    <div aria-live="polite">
      {error ? <p className="research-empty" role="status">{error}</p> : null}
      {!payload && !error ? <p className="research-empty" role="status">正在读取筛选结果…</p> : null}
      {favoriteNote ? <p className="screen-toast" role="status">{favoriteNote}</p> : null}
    </div>
    {payload ? <>
      <div className="research-asof">
        <span>信号日 <strong>{payload.as_of}</strong></span>
        <span>漂移窗口 {payload.window.replace(" to ", " 至 ")}</span>
        <span>成分股名单 {payload.membership_date}</span>
        <span>打分 {payload.scored} 只 · 未打分 {payload.unscored} 只</span>
        <span>导出 {timestamp(payload.generated_at)} · 上海时间</span>
        <a href="/us-screen/latest.json" download={`us-screen-${payload.as_of}.json`}>下载结果</a>
      </div>
      <div className="research-tabs" role="group" aria-label="选择视图">
        <button type="button" aria-pressed={view === "ranking"} onClick={() => setView("ranking")}>漂移排名</button>
        <button type="button" aria-pressed={view === "pullback"} onClick={() => setView("pullback")}>回调与困境反转</button>
        <button type="button" aria-pressed={view === "shapes"} onClick={() => setView("shapes")}>走势形态 · 12 种</button>
        <button type="button" aria-pressed={view === "evidence"} onClick={() => setView("evidence")}>回测证据</button>
        <button type="button" aria-pressed={view === "scorecard"} onClick={() => setView("scorecard")}>前瞻记录</button>
      </div>

      {view === "ranking" ? <>
        <section className="research-section" aria-labelledby="ranking-title">
          <div className="research-section-heading"><h2 id="ranking-title">漂移排名</h2></div>
          <p className="screen-warning">排名看的是 12 个月前到 1 个月前的走势，<strong>故意不看最近 1 个月</strong>，所以排名靠前的股票可能正在下跌{valid(fallenShare) ? `：当前前 50 名里 ${Math.round(fallenShare * 100)}% 最近 1 个月下跌` : ""}。回测里只保留“现在还在涨”的股票反而更差，详见“回测证据”。</p>
          <div className="screen-controls">
            <label>行业<select value={sector} onChange={(event) => setSector(event.target.value)}>{sectors.map((item) => <option key={item} value={item}>{item}</option>)}</select></label>
            <label>排序<select value={sortKey} onChange={(event) => setSortKey(event.target.value as SortKey)}>{Object.entries(sortLabels).map(([key, label]) => <option key={key} value={key}>{label}</option>)}</select></label>
            <label>显示<select value={limit} onChange={(event) => setLimit(event.target.value)}><option value="50">前 50</option><option value="100">前 100</option><option value="top">漂移最强五分之一</option><option value="all">全部</option></select></label>
            <label className="screen-check"><input type="checkbox" checked={risingOnly} onChange={(event) => setRisingOnly(event.target.checked)} />只看近 1 个月上涨</label>
            <span className="screen-muted">{rows.length} 只</span>
          </div>
          <div className="research-table-scroll"><table className="research-table screen-table">
            <caption className="sr-only">美股漂移排名</caption>
            <thead><tr><th scope="col">排名</th><th scope="col">股票</th><th scope="col">行业</th><th scope="col">漂移百分位</th><th scope="col">行业内</th><th scope="col">控制后</th><th scope="col">漂移涨幅</th><th scope="col">12-1 涨幅</th><th scope="col">跳动部分</th><th scope="col">近 1 月</th><th scope="col">日波动</th><th scope="col">自选</th></tr></thead>
            <tbody>{rows.map((stock) => <tr key={stock.symbol}>
              <td>{stock.rank}</td><StockCell stock={stock} />
              <td>{stock.sector || "未知"}<small>{stock.sic_description || ""}</small></td>
              <td>{plain(stock.drift_percentile)}</td><td>{plain(stock.drift_sector_percentile)}</td><td>{plain(stock.controlled_drift_percentile)}</td>
              <td className={tone(stock.drift)}>{pct(stock.drift, 0)}</td><td className={tone(stock.momentum_12_1)}>{pct(stock.momentum_12_1, 0)}</td>
              <td className={tone(stock.jump_return)}>{pct(stock.jump_return, 0)}<small>{stock.jump_days ?? 0} 天</small></td>
              <td className={tone(stock.return_1m)}>{pct(stock.return_1m)}</td><td>{valid(stock.daily_volatility) ? `${(stock.daily_volatility * 100).toFixed(1)}%` : "—"}</td>
              <td><FavoriteButton stock={stock} state={favorites[stock.symbol.toUpperCase()]} onAdd={(item) => void addFavorite(item)} /></td>
            </tr>)}</tbody>
          </table></div>
          <p className="research-note">漂移 = 12-1 涨幅去掉跳动日（日收益偏离中位数超过 {payload.jump_sigma} 倍稳健标准差）后的部分；跳动部分在回测里倾向回吐。行业内 = 在同一行业大类里的漂移排名；控制后 = 去掉普通动量、波动率和成交额影响后的排名。百分位 1.00 最强。</p>
        </section>
        <section className="research-section" aria-labelledby="mix-title">
          <h2 id="mix-title">行业分布</h2>
          <p className="research-note">回测验证的是持有漂移最强的五分之一（{payload.top_count} 只）。超配倍数 = 该行业在这一组里的占比 ÷ 在全部股票里的占比。不想押注行业时，可按“行业内百分位”排序或按行业筛选。</p>
          <div className="research-table-scroll"><table className="research-table"><thead><tr><th scope="col">行业</th><th scope="col">全部股票</th><th scope="col">最强五分之一</th><th scope="col">超配倍数</th></tr></thead>
            <tbody>{payload.sector_mix.map((row) => { const ratio = row.all ? (row.top / payload.top_count) / (row.all / payload.scored) : 0; return <tr key={row.sector}><td><button type="button" className="screen-link" onClick={() => { setSector(row.sector); setSortKey("drift_sector_percentile"); }}>{row.sector}</button></td><td>{row.all}</td><td>{row.top}</td><td className={ratio >= 1.5 ? "up" : ratio <= 0.5 ? "down" : undefined}>{ratio.toFixed(1)}</td></tr>; })}</tbody>
          </table></div>
        </section>
      </> : null}

      {view === "pullback" ? <section className="research-section" aria-labelledby="pullback-title">
        <h2 id="pullback-title">上升趋势中的回调</h2>
        <p className="research-note">上升趋势 = 漂移排名最强的五分之一；回调 = 收盘价在 50 日均线下方。回测（2017–2026，每月买入、持有一个月）：回调篮子扣费后每年跑赢股票池 {pct(pullbackEvidence.get("pullback_ma50")?.annual_excess)}，浅回调（近 1 个月跌 0–10%）{pct(pullbackEvidence.get("depth_0_10")?.annual_excess)}，对照的整个上升趋势组 {pct(pullbackEvidence.get("trend_all")?.annual_excess)}。</p>
        <p className="screen-warning"><strong>怎么理解：</strong>回调的股票不比没回调的差，但比整个上升趋势组多出来的部分分辨不出来（统计上和零差不多）。跌超 10% 的深回调历史样本很少、结果分散，没有证据支持；等“止跌”（近 5 日上涨）也没有帮助。优势只在买入后第一个月。</p>
        <div className="screen-controls">
          <label className="screen-check"><input type="checkbox" checked={mildOnly} onChange={(event) => setMildOnly(event.target.checked)} />只看浅回调（近 1 个月跌 0–10%，有回测支持）</label>
          <span className="screen-muted">今天在 50 日均线下方 {pullbackAll} 只，浅回调 {pullbackMild} 只</span>
        </div>
        <div className="research-table-scroll"><table className="research-table screen-table">
          <caption className="sr-only">上升趋势中回调的股票</caption>
          <thead><tr><th scope="col">排名</th><th scope="col">股票</th><th scope="col">行业</th><th scope="col">漂移百分位</th><th scope="col">近 1 月</th><th scope="col">离 50 日均线</th><th scope="col">近 5 日</th><th scope="col">回调类型</th><th scope="col">日波动</th><th scope="col">自选</th></tr></thead>
          <tbody>{pullbackRows.map((stock) => { const r = stock.return_1m; const depth = !valid(r) ? "—" : r > 0 ? "近 1 月仍上涨" : r > -0.1 ? "浅回调" : r > -0.2 ? "中等回调 · 无证据" : "深回调 · 无证据"; return <tr key={stock.symbol}>
            <td>{stock.rank}</td><StockCell stock={stock} /><td>{stock.sector || "未知"}</td><td>{plain(stock.drift_percentile)}</td>
            <td className={tone(r)}>{pct(r)}</td><td className={tone(stock.ma50_gap)}>{pct(stock.ma50_gap)}</td>
            <td className={tone(stock.return_5d)}>{pct(stock.return_5d)}</td>
            <td>{depth}<small>{stock.gap_down_1m ? "有跳空大跌日" : "缓慢回调"}</small></td>
            <td>{valid(stock.daily_volatility) ? `${(stock.daily_volatility * 100).toFixed(1)}%` : "—"}</td>
            <td><FavoriteButton stock={stock} state={favorites[stock.symbol.toUpperCase()]} onAdd={(item) => void addFavorite(item)} /></td>
          </tr>; })}</tbody>
        </table></div>
        {!pullbackRows.length ? <p className="research-empty">{payload.evidence.pullback ? "今天没有符合条件的股票。" : "这份结果还没有回调数据，请重新运行筛选并导出。"}</p> : null}
      </section> : null}

      {view === "pullback" && payload.evidence.fundamental ? <section className="research-section" aria-labelledby="fund-title">
        <h2 id="fund-title">困境反转（看财报）· 观察名单</h2>
        <p className="screen-warning">{payload.evidence.fundamental_warning}</p>
        <p className="research-note">条件：股价离一年高点的跌幅在全池最深的三分之一，近 12 个月 ROA（净利润 ÷ 总资产）比一年前改善、收入比一年前增长、经营现金流为正。只用信号日前已提交给 SEC 的财报。按跌幅从深到浅排列。检查表 = Piotroski 9 项里通过的比例（这里缺股本变化一项）。共 {turnaroundRows.length} 只。</p>
        <div className="research-table-scroll"><table className="research-table screen-table">
          <caption className="sr-only">基本面改善但股价深跌的股票</caption>
          <thead><tr><th scope="col">#</th><th scope="col">股票</th><th scope="col">行业</th><th scope="col">离一年高点</th><th scope="col">近 1 月</th><th scope="col">ROA 变化</th><th scope="col">收入增长</th><th scope="col">净利润 近 12 月 / 一年前</th><th scope="col">检查表</th><th scope="col">自选</th></tr></thead>
          <tbody>{turnaroundRows.map((stock, index) => <tr key={stock.symbol}>
            <td>{index + 1}</td><StockCell stock={stock} /><td>{stock.sector || "未知"}</td>
            <td className={tone(stock.drawdown_12m)}>{pct(stock.drawdown_12m, 0)}</td><td className={tone(stock.return_1m)}>{pct(stock.return_1m)}</td>
            <td className={tone(stock.delta_roa)}>{valid(stock.delta_roa) ? `${stock.delta_roa > 0 ? "+" : ""}${(stock.delta_roa * 100).toFixed(1)} 点` : "—"}</td>
            <td className={tone(stock.revenue_growth)}>{pct(stock.revenue_growth, 0)}</td>
            <td>{musd(stock.net_income_ttm)}<small>一年前 {musd(stock.net_income_ttm_prev)}</small></td>
            <td>{plain(stock.fscore)}<small>{valid(stock.fscore_signals) ? `${stock.fscore_signals} 项可判断` : ""}</small></td>
            <td><FavoriteButton stock={stock} state={favorites[stock.symbol.toUpperCase()]} onAdd={(item) => void addFavorite(item)} /></td>
          </tr>)}</tbody>
        </table></div>
        {deteriorating.length ? <p className="research-note">股价深跌且检查表 ≤ 0.3（基本面继续恶化）：{deteriorating.map((stock) => stock.symbol).join("、")}。这一组回测每年比股票池低 {(() => { const v = payload.evidence.fundamental.find((row) => row.key === "beaten_low_f")?.annual_excess; return valid(v) ? `${Math.abs(v * 100).toFixed(1)}%` : "—"; })()}（每月只有几只，区间很宽），更像是要回避的名单。</p> : null}
      </section> : null}

      {view === "shapes" ? <section className="research-section" aria-labelledby="shape-title">
        <h2 id="shape-title">走势形态 · 三种形态 × 四个时间段</h2>
        <p className="research-note">格子里是回测年化超额（每月买入符合条件的股票、等权持有一个月，扣费后相对股票池平均）和今天符合的股票数。对照：漂移主排名 +4.4%。12 种组合同时检验，单个结果要打折扣。</p>
        <div className="shape-grid" role="group" aria-label="选择走势形态和时间段">
          <span />{periodOrder.map((period) => <span key={period} className="shape-axis">{payload.evidence.shapes.find((row) => row.period === period)?.period_label}</span>)}
          {shapeOrder.map((shape) => <Fragment key={shape}>
            <span className="shape-axis shape-row-label">{payload.evidence.shapes.find((row) => row.shape === shape)?.shape_label}</span>
            {periodOrder.map((period) => { const key = `${shape}_${period}`; const ev = payload.evidence.shapes.find((row) => row.key === key); return <button key={key} type="button" aria-pressed={shapeKey === key} className={`shape-cell ${valid(ev?.annual_excess) && ev!.annual_excess! < 0 ? "negative" : "positive"}`} onClick={() => setShapeKey(key)}><strong>{pct(ev?.annual_excess)}</strong><small>今天 {payload.shape_matches[key]?.length ?? 0} 只</small></button>; })}
          </Fragment>)}
        </div>
        {shapeEvidence ? <>
          <h3>{shapeEvidence.shape_label} · {shapeEvidence.period_label}</h3>
          <p className="research-note">条件：{shapeEvidence.note}（时间段 = {shapeEvidence.period_label}）。漂移都去掉了跳动日，所以近期跌幅集中在几天的股票仍可能符合，请看“近 1 月”列。</p>
          {valid(shapeEvidence.annual_excess) && shapeEvidence.annual_excess < 0 ? <p className="screen-warning"><strong>回测跑输：</strong>每年比股票池平均低 {(-shapeEvidence.annual_excess * 100).toFixed(1)}%，{shapeEvidence.years} 年里 {shapeEvidence.losing_years} 年亏损。只适合作为看图的观察名单。</p> : <p className="research-note">回测：每年比股票池平均高 {pct(shapeEvidence.annual_excess)}，{shapeEvidence.years} 年里 {shapeEvidence.losing_years} 年亏损，最大回撤 {pct(shapeEvidence.max_drawdown)}。</p>}
          <div className="research-table-scroll"><table className="research-table screen-table">
            <caption className="sr-only">{shapeEvidence.shape_label}{shapeEvidence.period_label}符合的股票</caption>
            <thead><tr><th scope="col">#</th><th scope="col">股票</th><th scope="col">行业</th><th scope="col">形态得分</th><th scope="col">近 1 月</th><th scope="col">漂移百分位</th><th scope="col">12-1 涨幅</th><th scope="col">日波动</th><th scope="col">自选</th></tr></thead>
            <tbody>{shapeRows.map(({ match, stock }, index) => <tr key={stock.symbol}>
              <td>{index + 1}</td><StockCell stock={stock} /><td>{stock.sector || "未知"}</td><td>{plain(match.score)}</td>
              <td className={tone(stock.return_1m)}>{pct(stock.return_1m)}</td><td>{plain(stock.drift_percentile)}</td>
              <td className={tone(stock.momentum_12_1)}>{pct(stock.momentum_12_1, 0)}</td>
              <td>{valid(stock.daily_volatility) ? `${(stock.daily_volatility * 100).toFixed(1)}%` : "—"}</td>
              <td><FavoriteButton stock={stock} state={favorites[stock.symbol.toUpperCase()]} onAdd={(item) => void addFavorite(item)} /></td>
            </tr>)}</tbody>
          </table></div>
          {!shapeRows.length ? <p className="research-empty">今天没有符合这个形态的股票。</p> : null}
        </> : null}
      </section> : null}

      {view === "evidence" ? <section className="research-section" aria-labelledby="evidence-title">
        <h2 id="evidence-title">回测证据</h2>
        <p className="research-note">{payload.evidence.period}。都是样本内检验，前瞻表现可能不同；筛选结果之后的实际表现记在<button type="button" className="screen-link" onClick={() => setView("scorecard")}>前瞻记录</button>里。</p>
        <h3>排名方法</h3>
        <div className="research-table-scroll"><table className="research-table"><thead><tr><th scope="col">方法</th><th scope="col">Rank IC</th><th scope="col">IC 95% 区间</th><th scope="col">最强五分之一年化超额</th><th scope="col">多空年化</th></tr></thead>
          <tbody>{payload.evidence.factors.map((row) => <tr key={row.label}><td>{row.label}</td><td>{plain(row.ic, 3)}</td><td>{row.ic_ci ? `${plain(row.ic_ci[0], 3)} ～ ${plain(row.ic_ci[1], 3)}` : "—"}</td><td className={tone(row.top_excess)}>{pct(row.top_excess)}</td><td className={tone(row.long_short)}>{pct(row.long_short)}</td></tr>)}</tbody></table></div>
        <h3>要求“现在还在涨”</h3>
        <p className="research-note">在漂移主排名基础上加均线条件后的最强五分之一年化超额。对照主排名 +4.4%。</p>
        <div className="research-table-scroll"><table className="research-table"><thead><tr><th scope="col">版本</th><th scope="col">年化超额</th></tr></thead>
          <tbody>{payload.evidence.trend_filters.map((row) => <tr key={row.label}><td>{row.label}</td><td className={tone(row.top_excess)}>{pct(row.top_excess)}</td></tr>)}</tbody></table></div>
        {payload.evidence.pullback ? <>
          <h3>上升趋势中的回调</h3>
          <p className="research-note">都在漂移最强五分之一里分组。区间 = 月均超额 × 12 的 95% 区间，不含零才算有证据。</p>
          <div className="research-table-scroll"><table className="research-table"><thead><tr><th scope="col">篮子</th><th scope="col">年化超额</th><th scope="col">95% 区间</th><th scope="col">亏损年数</th><th scope="col">每月股票数（中位）</th></tr></thead>
            <tbody>{payload.evidence.pullback.map((row) => <tr key={row.key}><td>{row.label}</td><td className={tone(row.annual_excess)}>{pct(row.annual_excess)}</td><td>{valid(row.monthly_ci[0]) && valid(row.monthly_ci[1]) ? `${pct(row.monthly_ci[0] * 12)} ～ ${pct(row.monthly_ci[1] * 12)}` : "—"}</td><td>{row.losing_years} / {row.years}</td><td>{plain(row.median_stocks, 0)}</td></tr>)}</tbody></table></div>
        </> : null}
        {payload.evidence.fundamental ? <>
          <h3>困境反转（看财报）</h3>
          <p className="research-note">没有一个篮子的区间不含零。基本面改善能抵消“股价深跌”本身的拖累（深跌全部 {pct(payload.evidence.fundamental.find((row) => row.key === "beaten_all")?.annual_excess)}），但不能变成稳定跑赢。</p>
          <div className="research-table-scroll"><table className="research-table"><thead><tr><th scope="col">篮子</th><th scope="col">年化超额</th><th scope="col">95% 区间</th><th scope="col">亏损年数</th><th scope="col">每月股票数（中位）</th></tr></thead>
            <tbody>{payload.evidence.fundamental.map((row) => <tr key={row.key}><td>{row.label}</td><td className={tone(row.annual_excess)}>{pct(row.annual_excess)}</td><td>{valid(row.monthly_ci[0]) && valid(row.monthly_ci[1]) ? `${pct(row.monthly_ci[0] * 12)} ～ ${pct(row.monthly_ci[1] * 12)}` : "—"}</td><td>{row.losing_years} / {row.years}</td><td>{plain(row.median_stocks, 0)}</td></tr>)}</tbody></table></div>
        </> : null}
        <h3>走势形态</h3>
        <div className="research-table-scroll"><table className="research-table"><thead><tr><th scope="col">形态</th><th scope="col">时间段</th><th scope="col">年化超额</th><th scope="col">亏损年数</th><th scope="col">最大回撤</th><th scope="col">每月平均股票数</th></tr></thead>
          <tbody>{payload.evidence.shapes.map((row) => <tr key={row.key}><td>{row.shape_label}</td><td>{row.period_label}</td><td className={tone(row.annual_excess)}>{pct(row.annual_excess)}</td><td>{row.losing_years} / {row.years}</td><td>{pct(row.max_drawdown)}</td><td>{plain(row.mean_stocks, 0)}</td></tr>)}</tbody></table></div>
        <p className="research-note">完整方法见 quant-research/docs/physics-momentum-20261106.md 和 docs/us-screen-20261106.md。</p>
      </section> : null}

      {view === "scorecard" ? <Scorecard reload={reload} /> : null}

      <footer className="research-footer"><details><summary>数据来源</summary><p>筛选运行：{payload.source.run}</p><p>运行清单：{payload.source.manifest_sha256}</p></details></footer>
    </> : null}
  </main>;
}
