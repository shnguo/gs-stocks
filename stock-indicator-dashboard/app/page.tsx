"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { PageHeader } from "./components/page-header";
import { marketSessions, type MarketSession } from "./lib/market-sessions";
import {
  marketInstrumentFromCatalog,
  realtimeMarketInstrument,
  savedMarketInstrument,
  type MarketInstrumentIdentity,
} from "./lib/stock-symbol";
import { prefetchRealtimeQuotes, type RealtimeQuote } from "./use-realtime-quote";
import "./overview.css";

const FAVORITES_STORAGE_KEY = "stock-indicator-dashboard.favorite-stocks:v5";
const WATCHLIST_LIMIT = 8;

type Favorite = MarketInstrumentIdentity & { name: string };
type ResearchQuality = { known: number; expected: number; mean_absolute_percentage_error: number | null; direction_agreement: number | null };
type ResearchSummary = {
  schema_version: number; generated_at: string; observed_through: string | null;
  freshness: { status: string; expected_through: string | null; missing_sessions: string[] };
  analysis_reports: { signal_date: string; arms: { arm: string; label: string; candidate_count: number; quality: ResearchQuality; publication: { prospective: boolean };
    candidates: { instrument_id: string; name: string | null; rank: number; expected_net_return: number | null; trend: { return_5: number | null } }[] }[] }[];
};
type ScreenStock = { symbol: string; name: string | null; sector: string | null; rank?: number; drift?: number | null; return_1m?: number | null; extra_only?: boolean };
type ScreenSummary = {
  schema_version: number; generated_at: string; as_of: string; membership_date: string; scored: number; top_count: number;
  sector_mix: { sector: string; all: number; top: number }[]; stocks: ScreenStock[];
  shape_matches: Record<string, unknown[]>;
  evidence: { factors: { label: string; top_excess: number | null }[] };
};
type Load<T> = { state: "loading" } | { state: "missing"; message: string } | { state: "ready"; data: T };

function valid(value?: number | null): value is number { return typeof value === "number" && Number.isFinite(value); }
function pct(value?: number | null, digits = 1) { return valid(value) ? `${value > 0 ? "+" : ""}${(value * 100).toFixed(digits)}%` : "—"; }
function plainPct(value?: number | null, digits = 1) { return valid(value) ? `${(value * 100).toFixed(digits)}%` : "—"; }
function tone(value?: number | null) { return valid(value) && value !== 0 ? (value > 0 ? "tone-up" : "tone-down") : undefined; }
function stamp(value?: string | null) {
  if (!value) return "—";
  return new Intl.DateTimeFormat("zh-CN", { timeZone: "Asia/Shanghai", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hourCycle: "h23" }).format(new Date(value));
}

function useJson<T>(url: string, check: (data: T) => boolean, missing: string): Load<T> {
  const [load, setLoad] = useState<Load<T>>({ state: "loading" });
  useEffect(() => {
    const controller = new AbortController();
    fetch(url, { cache: "no-store", signal: controller.signal })
      .then(async (response) => {
        if (!response.ok || !response.headers.get("content-type")?.includes("application/json")) throw new Error(missing);
        const data = await response.json() as T;
        if (!check(data)) throw new Error("数据格式无法识别，请重新导出。");
        setLoad({ state: "ready", data });
      })
      .catch((reason: unknown) => { if (!controller.signal.aborted) setLoad({ state: "missing", message: reason instanceof Error ? reason.message : missing }); });
    return () => controller.abort();
  }, [url, check, missing]);
  return load;
}

const checkResearch = (data: ResearchSummary) => data?.schema_version === 1 && Array.isArray(data.analysis_reports);
const checkScreen = (data: ScreenSummary) => data?.schema_version === 1 && Array.isArray(data.stocks);

function readLocalFavorites(): Favorite[] {
  try {
    const value: unknown = JSON.parse(window.localStorage.getItem(FAVORITES_STORAGE_KEY) || "[]");
    if (!Array.isArray(value)) return [];
    return value.flatMap((item) => {
      const identity = savedMarketInstrument(item);
      const name = item && typeof item === "object" && typeof (item as Record<string, unknown>).name === "string" ? String((item as Record<string, unknown>).name) : identity?.code;
      return identity && name ? [{ ...identity, name }] : [];
    });
  } catch {
    return [];
  }
}

type DailyClose = { price: number; change: number | null; date: string };

/** Latest published daily close (and change from the prior close) for one instrument. */
async function latestClose(instrumentId: string, signal: AbortSignal): Promise<DailyClose | null> {
  const response = await fetch(`/api/chart-daily-bars?${new URLSearchParams({ instrument_id: instrumentId, adjustment_basis: "none" })}`, { signal });
  if (!response.ok) return null;
  const payload = await response.json().catch(() => null) as { rows?: { trading_date?: string; close?: string | number }[] } | null;
  const rows = (payload?.rows ?? []).map((row) => ({ date: String(row.trading_date ?? ""), close: Number(row.close) }))
    .filter((row) => row.date && Number.isFinite(row.close) && row.close > 0).sort((a, b) => a.date.localeCompare(b.date));
  const last = rows.at(-1);
  if (!last) return null;
  const previous = rows.at(-2);
  return { price: last.close, change: previous ? last.close / previous.close - 1 : null, date: last.date };
}

function useWatchlist() {
  const [favorites, setFavorites] = useState<Favorite[] | null>(null);
  const [quotes, setQuotes] = useState<Record<string, RealtimeQuote>>({});
  const [closes, setCloses] = useState<Record<string, DailyClose>>({});
  const [quotesAt, setQuotesAt] = useState<string | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    const timer = window.setTimeout(() => setFavorites(readLocalFavorites()), 0);
    fetch("/api/favorites", { cache: "no-store", signal: controller.signal })
      .then((response) => response.ok ? response.json() as Promise<{ favorites?: unknown[] }> : null)
      .then((payload) => {
        const server = (payload?.favorites ?? []).flatMap((row) => {
          const identity = marketInstrumentFromCatalog(row);
          const name = row && typeof row === "object" ? (row as Record<string, unknown>).name : null;
          return identity ? [{ ...identity, name: typeof name === "string" ? name : identity.code }] : [];
        });
        setFavorites((current) => Array.from(new Map([...(current ?? readLocalFavorites()), ...server].map((item) => [item.instrumentId, item])).values()));
      })
      .catch(() => undefined);
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, []);
  const key = (favorites ?? []).slice(0, WATCHLIST_LIMIT).map((item) => item.instrumentId).join(",");
  useEffect(() => {
    if (!favorites?.length) return;
    const controller = new AbortController();
    const instruments = favorites.slice(0, WATCHLIST_LIMIT).map((item) => realtimeMarketInstrument(item));
    prefetchRealtimeQuotes(instruments, controller.signal)
      .catch(() => [] as RealtimeQuote[])
      .then(async (rows) => {
        if (controller.signal.aborted) return;
        setQuotes(Object.fromEntries(rows.map((row) => [row.instrumentId, row])));
        if (rows.length) setQuotesAt(new Date().toISOString());
        // Without a live quote, fall back to the latest published daily close.
        const missing = instruments.filter((item) => !rows.some((row) => row.instrumentId === item.instrumentId));
        await Promise.all(missing.map(async (item) => {
          const close = await latestClose(item.instrumentId, controller.signal).catch(() => null);
          if (close && !controller.signal.aborted) setCloses((current) => ({ ...current, [item.instrumentId]: close }));
        }));
      });
    return () => controller.abort();
    // The key captures the watched instruments.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return { favorites, quotes, closes, quotesAt };
}

function Unavailable({ message }: { message: string }) {
  return <p className="overview-empty">{message}</p>;
}

function SessionStrip() {
  const [sessions, setSessions] = useState<MarketSession[] | null>(null);
  useEffect(() => {
    const update = () => setSessions(marketSessions());
    const first = window.setTimeout(update, 0);
    const timer = window.setInterval(update, 30_000);
    return () => { window.clearTimeout(first); window.clearInterval(timer); };
  }, []);
  return <div className="overview-sessions" aria-label="交易时段">
    {(sessions ?? []).map((item) => <div key={item.market} className={`overview-session session-${item.state}`}>
      <span>{item.label}</span><strong>{item.stateLabel}</strong><small>当地 {item.localTime}</small>
    </div>)}
  </div>;
}

export default function OverviewPage() {
  const { favorites, quotes, closes, quotesAt } = useWatchlist();
  const research = useJson<ResearchSummary>("/research/latest.json", checkResearch, "尚未发布研究摘要。");
  const screen = useJson<ScreenSummary>("/us-screen/latest.json", checkScreen, "尚未发布美股筛选结果。");
  const [today, setToday] = useState("");
  useEffect(() => {
    const timer = window.setTimeout(() => setToday(new Intl.DateTimeFormat("zh-CN", { timeZone: "Asia/Shanghai", year: "numeric", month: "long", day: "numeric", weekday: "long" }).format(new Date())), 0);
    return () => window.clearTimeout(timer);
  }, []);

  const latestReport = research.state === "ready" ? research.data.analysis_reports.at(-1) : undefined;
  const controlArm = latestReport?.arms.find((arm) => arm.arm === "control") ?? latestReport?.arms[0];
  const screenTop = useMemo(() => screen.state === "ready" ? screen.data.stocks.filter((s) => !s.extra_only && (s.rank ?? 99) <= 5) : [], [screen]);
  const overweight = useMemo(() => {
    if (screen.state !== "ready") return null;
    const { sector_mix, top_count, scored } = screen.data;
    return [...sector_mix].filter((row) => row.all > 0 && row.sector !== "未知")
      .map((row) => ({ ...row, ratio: (row.top / top_count) / (row.all / scored) }))
      .sort((a, b) => b.ratio - a.ratio)[0] ?? null;
  }, [screen]);
  const mainEvidence = screen.state === "ready" ? screen.data.evidence.factors.find((row) => row.label.startsWith("漂移（主排名）")) : undefined;

  return <main className="page overview">
    <PageHeader eyebrow="总览" title="今日概览" description="自选股票行情、最新研究复盘和美股筛选结果。点击任一卡片进入对应模块。"
      meta={<><span>{today || " "}</span><span>研究用途，不构成投资建议</span></>} />

    <SessionStrip />

    <div className="overview-grid">
      <section className="card overview-watchlist" aria-labelledby="ov-watch">
        <div className="card-head"><h2 id="ov-watch">自选股票</h2><Link href="/market">进入行情 →</Link></div>
        <div className="card-body">
          {favorites === null ? <Unavailable message="正在读取自选股票…" /> : !favorites.length ? <Unavailable message="还没有自选股票。在行情页左侧添加，或在美股筛选里点“加入自选”。" /> :
            <ul className="overview-quotes">
              {favorites.slice(0, WATCHLIST_LIMIT).map((item) => {
                const quote = quotes[item.instrumentId];
                const close = closes[item.instrumentId];
                const change = quote && quote.previousClose > 0 ? quote.last / quote.previousClose - 1 : close?.change ?? null;
                const price = quote ? quote.last : close?.price;
                return <li key={item.instrumentId}>
                  <Link href={`/market?instrument=${encodeURIComponent(item.instrumentId)}`}>
                    <span className="overview-quote-name"><strong>{item.name}</strong><small>{item.marketGroup} · {item.code}</small></span>
                    <span className="overview-quote-price"><strong>{valid(price) ? price.toFixed(2) : "—"}</strong><small className={tone(change)}>{quote ? `${pct(change, 2)} · 实时` : close ? `${pct(change, 2)} · 收盘 ${close.date.slice(5)}` : "暂无行情"}</small></span>
                  </Link>
                </li>;
              })}
            </ul>}
        </div>
        <div className="card-foot">{favorites && favorites.length > WATCHLIST_LIMIT ? `共 ${favorites.length} 只，显示前 ${WATCHLIST_LIMIT} 只。` : ""}{quotesAt ? `实时报价读取于 ${stamp(quotesAt)}（上海时间）。` : "实时行情未连接时显示最近一个交易日的收盘价。"}</div>
      </section>

      <section className="card" aria-labelledby="ov-research">
        <div className="card-head"><h2 id="ov-research">研究复盘</h2><Link href="/research">查看复盘 →</Link></div>
        <div className="card-body">
          {research.state === "loading" ? <Unavailable message="正在读取研究摘要…" /> : research.state === "missing" ? <Unavailable message={research.message} /> : <>
            <p className="overview-kicker">最新信号日 <strong>{latestReport?.signal_date ?? "—"}</strong>{controlArm ? <> · {controlArm.label} · <span className={controlArm.publication.prospective ? "badge badge-ok" : "badge badge-warn"}>{controlArm.publication.prospective ? "开盘前冻结" : "迟发回顾"}</span></> : null}</p>
            {controlArm ? <dl className="stats overview-stats">
              <div className="stat"><dt>收盘价格误差</dt><dd>{plainPct(controlArm.quality.mean_absolute_percentage_error, 2)}<small>越低越好</small></dd></div>
              <div className="stat"><dt>涨跌方向一致</dt><dd>{plainPct(controlArm.quality.direction_agreement, 0)}<small>{controlArm.quality.known} / {controlArm.quality.expected} 条已对照</small></dd></div>
            </dl> : null}
            {controlArm?.candidates.length ? <ol className="overview-list">
              {controlArm.candidates.slice(0, 5).map((row) => <li key={row.instrument_id}><span>{row.rank}. {row.name || row.instrument_id}</span><span className={tone(row.expected_net_return)}>{pct(row.expected_net_return)}</span></li>)}
            </ol> : null}
          </>}
        </div>
        <div className="card-foot">{research.state === "ready" ? <>行情截至 {research.data.observed_through ?? "未知"} · {research.data.freshness.status === "current" ? "数据已是最新" : `尚缺 ${research.data.freshness.missing_sessions.length} 个交易日`}</> : "候选榜右侧为预期参考收益。"}</div>
      </section>

      <section className="card" aria-labelledby="ov-screen">
        <div className="card-head"><h2 id="ov-screen">美股筛选</h2><Link href="/us-screen">查看排名 →</Link></div>
        <div className="card-body">
          {screen.state === "loading" ? <Unavailable message="正在读取筛选结果…" /> : screen.state === "missing" ? <Unavailable message={screen.message} /> : <>
            <p className="overview-kicker">信号日 <strong>{screen.data.as_of}</strong> · 打分 {screen.data.scored} 只</p>
            <dl className="stats overview-stats">
              <div className="stat"><dt>主排名回测</dt><dd className={tone(mainEvidence?.top_excess)}>{pct(mainEvidence?.top_excess)}<small>最强五分之一年化超额</small></dd></div>
              <div className="stat"><dt>超配最多的行业</dt><dd className="overview-sector">{overweight?.sector ?? "—"}<small>{overweight ? `${overweight.ratio.toFixed(1)} 倍 · 最强组 ${overweight.top} 只` : ""}</small></dd></div>
            </dl>
            <ol className="overview-list">
              {screenTop.map((row) => <li key={row.symbol}><span>{row.rank}. {row.symbol}<small>{row.sector ?? ""}</small></span><span className={tone(row.return_1m)}>近 1 月 {pct(row.return_1m)}</span></li>)}
            </ol>
          </>}
        </div>
        <div className="card-foot">排名不看最近 1 个月，前列股票可能正在下跌。研究用排名，不是买卖建议。</div>
      </section>
    </div>

    <section className="card overview-status" aria-labelledby="ov-status">
      <div className="card-head"><h2 id="ov-status">数据状态</h2><small>各模块读取的数据及更新时间</small></div>
      <div className="research-table-scroll"><table className="research-table">
        <thead><tr><th scope="col">模块</th><th scope="col">数据</th><th scope="col">数据日期</th><th scope="col">更新时间 · 上海</th><th scope="col">状态</th></tr></thead>
        <tbody>
          <tr><td>行情</td><td>自选股票实时报价</td><td>—</td><td>{stamp(quotesAt)}</td><td>{Object.keys(quotes).length ? <span className="badge badge-ok">已连接 · {Object.keys(quotes).length} 只</span> : Object.keys(closes).length ? <span className="badge badge-warn">未连接 · 显示收盘价</span> : <span className="badge badge-muted">未取得报价</span>}</td></tr>
          <tr><td>研究复盘</td><td>每日研究分析摘要</td><td>{research.state === "ready" ? research.data.observed_through ?? "—" : "—"}</td><td>{research.state === "ready" ? stamp(research.data.generated_at) : "—"}</td><td>{research.state !== "ready" ? <span className="badge badge-muted">未发布</span> : research.data.freshness.status === "current" ? <span className="badge badge-ok">最新</span> : <span className="badge badge-warn">待更新</span>}</td></tr>
          <tr><td>美股筛选</td><td>漂移动量排名（成分股名单 {screen.state === "ready" ? screen.data.membership_date : "—"}）</td><td>{screen.state === "ready" ? screen.data.as_of : "—"}</td><td>{screen.state === "ready" ? stamp(screen.data.generated_at) : "—"}</td><td>{screen.state === "ready" ? <span className="badge badge-ok">已导出</span> : <span className="badge badge-muted">未发布</span>}</td></tr>
        </tbody>
      </table></div>
    </section>
  </main>;
}
