"use client";

import { SidebarSimpleIcon } from "@phosphor-icons/react";
import { ChangeEvent, PointerEvent, useEffect, useMemo, useState } from "react";

import {
  useRealtimeQuote,
  type RealtimeConnectionStatus,
  type RealtimeQuote,
} from "./use-realtime-quote";

type Candle = {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  sourceId?: string;
  comparisonStatus?: string;
  realtime?: boolean;
  unconfirmed?: boolean;
};

type IndicatorPoint = {
  score: number;
  raw: number;
  closePosition: number;
  volumeRatio: number;
  confirm: boolean;
};

type StockPreset = {
  code: string;
  name: string;
  market: string;
  marketGroup: MarketGroup;
  instrumentId: string;
};

type MarketGroup = "A股" | "港股" | "美股";

type AdjustmentBasis = "none" | "qfq" | "hfq";

type DataMeta = {
  requestKey: string;
  dataVersion: string | null;
  asOf: string;
  cacheStatus: string;
  sourceId: string;
  comparisonStatus: string;
};

const STOCKS: StockPreset[] = [
  { code: "688008", name: "澜起科技", market: "科创板", marketGroup: "A股", instrumentId: "cn.xshg.688008" },
  { code: "600036", name: "招商银行", market: "沪市", marketGroup: "A股", instrumentId: "cn.xshg.600036" },
  { code: "000333", name: "美的集团", market: "深市", marketGroup: "A股", instrumentId: "cn.xshe.000333" },
  { code: "300059", name: "东方财富", market: "创业板", marketGroup: "A股", instrumentId: "cn.xshe.300059" },
];

const MARKET_TABS: MarketGroup[] = ["A股", "港股", "美股"];

const PERIODS = [
  { label: "60日", value: 60 },
  { label: "120日", value: 120 },
  { label: "250日", value: 250 },
];

const ADJUSTMENT_BASES: { label: string; value: AdjustmentBasis }[] = [
  { label: "不复权", value: "none" },
  { label: "前复权", value: "qfq" },
  { label: "后复权", value: "hfq" },
];

const MOVING_AVERAGE_PERIODS = [5, 10, 20, 30, 60] as const;

const CHART_WIDTH = 1120;
const PRICE_HEIGHT = 410;
const INDICATOR_HEIGHT = 184;
const LEFT = 18;
const RIGHT = 74;
const TOP = 20;
const BOTTOM = 30;

function movingAverage(values: number[], length: number) {
  return values.map((_, index) => {
    const from = Math.max(0, index - length + 1);
    const window = values.slice(from, index + 1);
    return window.reduce((sum, value) => sum + value, 0) / window.length;
  });
}

function fullWindowMovingAverage(values: number[], length: number): (number | null)[] {
  let sum = 0;
  return values.map((value, index) => {
    sum += value;
    if (index >= length) sum -= values[index - length];
    return index >= length - 1 ? sum / length : null;
  });
}

function calculateIndicator(data: Candle[]): IndicatorPoint[] {
  const trueRanges = data.map((item, index) => {
    if (index === 0) return item.high - item.low;
    const previousClose = data[index - 1].close;
    return Math.max(
      item.high - item.low,
      Math.abs(item.high - previousClose),
      Math.abs(item.low - previousClose),
    );
  });
  const atr = movingAverage(trueRanges, 14);
  const volumeAverage = movingAverage(data.map((item) => item.volume), 20);
  const closeAverage = movingAverage(data.map((item) => item.close), 5);
  let smooth = 0;

  return data.map((item, index) => {
    if (index === 0) {
      return { score: 0, raw: 0, closePosition: 0.5, volumeRatio: 1, confirm: false };
    }
    const previousLow = data[index - 1].low;
    const probe = Math.max(previousLow - item.low, 0) / Math.max(atr[index], 0.01) * 100;
    const priorLows = data.slice(Math.max(0, index - 38), index).map((point) => point.low);
    const newLow = item.low <= Math.min(...priorLows);
    const closePosition = (item.close - item.low) / Math.max(item.high - item.low, 0.01);
    const volumeRatio = Math.min(item.volume / Math.max(volumeAverage[index], 1), 2);
    const raw = newLow ? probe * closePosition * volumeRatio : 0;
    smooth = index === 1 ? raw : raw * 0.5 + smooth * 0.5;
    const score = Math.min(smooth, 100);
    const trend = item.close > closeAverage[index] && closeAverage[index] > closeAverage[index - 1];
    const recover = item.close > item.open && item.close > data[index - 1].close && closePosition > 0.65;
    const volumeOk = item.volume > volumeAverage[index] * 1.2;
    return { score, raw, closePosition, volumeRatio, confirm: score > 30 && trend && recover && volumeOk };
  });
}

function formatNumber(value: number, digits = 2) {
  return value.toLocaleString("zh-CN", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

function formatVolume(value: number) {
  if (value >= 100_000_000) return `${formatNumber(value / 100_000_000)}亿`;
  return `${formatNumber(value / 10_000, 0)}万`;
}

function formatDate(date: string) {
  return date.slice(5).replace("-", "/");
}

function axisTicks(min: number, max: number, count: number) {
  return Array.from({ length: count }, (_, index) => min + (max - min) * index / (count - 1));
}

function seriesFingerprint(data: Candle[]) {
  const input = data.map((item) => [
    item.date,
    item.open,
    item.high,
    item.low,
    item.close,
    item.volume,
  ].join("|")).join("\n");
  let hash = 2166136261;
  for (let index = 0; index < input.length; index += 1) {
    hash = Math.imul(hash ^ input.charCodeAt(index), 16777619) >>> 0;
  }
  return hash.toString(16).padStart(8, "0");
}

function mergeRealtimeCandle(history: Candle[], quote: RealtimeQuote | null): Candle[] {
  if (!quote || history.length === 0) return history;
  const latest = history[history.length - 1];
  if (!latest || quote.tradingDate < latest.date) return history;
  const replacingLatest = quote.tradingDate === latest.date;
  const reference = replacingLatest ? history[history.length - 2] : latest;
  const adjustmentFactor =
    reference && quote.previousClose > 0 ? reference.close / quote.previousClose : 1;
  const candle: Candle = {
    date: quote.tradingDate,
    open: quote.open * adjustmentFactor,
    high: quote.high * adjustmentFactor,
    low: quote.low * adjustmentFactor,
    close: quote.last * adjustmentFactor,
    volume: quote.cumulativeVolume,
    sourceId: "tencent-realtime",
    comparisonStatus: "unconfirmed",
    realtime: true,
    unconfirmed: true,
  };
  return replacingLatest ? [...history.slice(0, -1), candle] : [...history, candle];
}

function alignFormalIndicators(displayData: Candle[], completedData: Candle[]): IndicatorPoint[] {
  const formal = calculateIndicator(completedData);
  const byDate = new Map(completedData.map((item, index) => [item.date, formal[index]]));
  const fallback = formal[formal.length - 1] ?? {
    score: 0,
    raw: 0,
    closePosition: 0.5,
    volumeRatio: 1,
    confirm: false,
  };
  return displayData.map((item) => {
    const point = byDate.get(item.date);
    return point ?? { ...fallback, confirm: false };
  });
}

function WatchlistSidebar({
  selectedCode,
  activePrice,
  activeChange,
  onSelect,
}: {
  selectedCode: string | null;
  activePrice?: number;
  activeChange?: number;
  onSelect: (code: string) => void;
}) {
  const [activeMarket, setActiveMarket] = useState<MarketGroup>("A股");
  const visibleStocks = STOCKS.filter((item) => item.marketGroup === activeMarket);

  return (
    <nav className="watchlist-sidebar" aria-label="收藏股票">
      <div className="watchlist-heading">
        <div><h2>自选股票</h2><p>收藏列表</p></div>
        <span>{STOCKS.length} 只</span>
      </div>
      <div className="market-tabs" role="tablist" aria-label="按市场筛选收藏股票">
        {MARKET_TABS.map((market) => {
          const count = STOCKS.filter((item) => item.marketGroup === market).length;
          return (
            <button
              type="button"
              role="tab"
              aria-selected={activeMarket === market}
              aria-controls={`watchlist-panel-${market}`}
              className={activeMarket === market ? "active" : ""}
              key={market}
              onClick={() => setActiveMarket(market)}
            >
              <span>{market}</span><small>{count}</small>
            </button>
          );
        })}
      </div>
      <div className="watchlist-panel" id={`watchlist-panel-${activeMarket}`} role="tabpanel">
        {visibleStocks.length > 0 ? (
          <ul className="watchlist-items">
            {visibleStocks.map((item) => {
              const isActive = selectedCode === item.code;
              return (
                <li key={item.code}>
                  <button
                    type="button"
                    className={isActive ? "active" : ""}
                    aria-current={isActive ? "true" : undefined}
                    onClick={() => onSelect(item.code)}
                  >
                    <span className="watchlist-security"><strong>{item.name}</strong><small>{item.code}</small></span>
                    {isActive && activePrice !== undefined && activeChange !== undefined ? (
                      <span className="watchlist-quote"><strong>{formatNumber(activePrice)}</strong><small className={activeChange >= 0 ? "price-up" : "price-down"}>{activeChange >= 0 ? "+" : ""}{formatNumber(activeChange)}%</small></span>
                    ) : (
                      <span className="watchlist-market">{item.market}</span>
                    )}
                  </button>
                </li>
              );
            })}
          </ul>
        ) : (
          <div className="watchlist-empty"><strong>{activeMarket}暂无收藏</strong><p>收藏后会显示在这里。</p></div>
        )}
      </div>
    </nav>
  );
}

function LeftRail({
  collapsed,
  selectedCode,
  activePrice,
  activeChange,
  onToggle,
  onSelect,
}: {
  collapsed: boolean;
  selectedCode: string | null;
  activePrice?: number;
  activeChange?: number;
  onToggle: () => void;
  onSelect: (code: string) => void;
}) {
  return (
    <aside className={`left-rail ${collapsed ? "collapsed" : ""}`} aria-label="左侧收藏栏">
      <header className="left-rail-topbar">
        <a className="brand" href="#top" aria-label="低位承接研究台首页">
          <span className="brand-mark">承</span>
          <span className="brand-copy"><strong>低位承接研究台</strong><small>Price absorption research</small></span>
        </a>
        <button className="rail-toggle" type="button" aria-expanded={!collapsed} aria-controls="favorite-stock-panel" title={collapsed ? "展开左侧栏" : "收起左侧栏"} onClick={onToggle}>
          <SidebarSimpleIcon aria-hidden="true" size={18} weight="regular" />
          <span className="sr-only">{collapsed ? "展开左侧栏" : "收起左侧栏"}</span>
        </button>
      </header>
      <div id="favorite-stock-panel" hidden={collapsed}>
        <WatchlistSidebar selectedCode={selectedCode} activePrice={activePrice} activeChange={activeChange} onSelect={onSelect} />
      </div>
      {collapsed && <span className="collapsed-rail-label" aria-hidden="true">自选股票</span>}
    </aside>
  );
}

function CenterTopbar({
  sourceLabel,
  realtimeStatus = "disabled",
}: {
  sourceLabel: string;
  realtimeStatus?: RealtimeConnectionStatus;
}) {
  return (
    <header className="topbar" aria-label="行情数据状态">
      <div className="topbar-note" data-realtime-status={realtimeStatus}>
        <span className={`status-dot status-${realtimeStatus}`} aria-hidden="true" />
        {sourceLabel}
      </div>
    </header>
  );
}

function ReservedRightRail({ collapsed, onToggle }: { collapsed: boolean; onToggle: () => void }) {
  return (
    <aside className={`right-rail ${collapsed ? "collapsed" : ""}`} aria-label="右侧预留区域">
      <header className="right-rail-topbar">
        <span className="right-rail-title"><strong>信息面板</strong><small>Reserved panel</small></span>
        <button className="rail-toggle" type="button" aria-expanded={!collapsed} aria-controls="reserved-right-panel" title={collapsed ? "展开右侧栏" : "收起右侧栏"} onClick={onToggle}>
          <SidebarSimpleIcon aria-hidden="true" mirrored size={18} weight="regular" />
          <span className="sr-only">{collapsed ? "展开右侧栏" : "收起右侧栏"}</span>
        </button>
      </header>
      <div id="reserved-right-panel" className="right-rail-placeholder" hidden={collapsed}><span>右侧区域已预留</span></div>
    </aside>
  );
}

export default function Home() {
  const [stockCode, setStockCode] = useState(STOCKS[0].code);
  const [period, setPeriod] = useState(120);
  const [adjustmentBasis, setAdjustmentBasis] = useState<AdjustmentBasis>("qfq");
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const [importedData, setImportedData] = useState<Candle[] | null>(null);
  const [importName, setImportName] = useState("");
  const [remoteData, setRemoteData] = useState<Candle[]>([]);
  const [dataMeta, setDataMeta] = useState<DataMeta | null>(null);
  const [loadError, setLoadError] = useState<{ requestKey: string; message: string } | null>(null);
  const [retryNonce, setRetryNonce] = useState(0);
  const [leftRailCollapsed, setLeftRailCollapsed] = useState(false);
  const [rightRailCollapsed, setRightRailCollapsed] = useState(false);

  const stock = STOCKS.find((item) => item.code === stockCode) ?? STOCKS[0];
  const requestKey = `${stock.instrumentId}:${adjustmentBasis}:${retryNonce}`;

  useEffect(() => {
    const controller = new AbortController();
    const query = new URLSearchParams({
      instrument_id: stock.instrumentId,
      adjustment_basis: adjustmentBasis,
    });
    fetch(`/api/chart-daily-bars?${query}`, { signal: controller.signal })
      .then(async (response) => {
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "真实行情加载失败");
        return payload;
      })
      .then((payload) => {
        const rows = Array.isArray(payload.rows) ? payload.rows : [];
        const parsed = rows.map((row: Record<string, unknown>) => ({
          date: String(row.trading_date ?? ""),
          open: Number(row.open),
          high: Number(row.high),
          low: Number(row.low),
          close: Number(row.close),
          volume: Number(row.volume),
          sourceId: String(row.source_id ?? "unknown"),
          comparisonStatus: String(row.comparison_status ?? "unknown"),
        })).filter((item: Candle) =>
          item.date &&
          [item.open, item.high, item.low, item.close, item.volume].every(Number.isFinite),
        ).sort((a: Candle, b: Candle) => a.date.localeCompare(b.date));
        if (parsed.length < 20) throw new Error("真实行情不足 20 个交易日");
        setRemoteData(parsed);
        setLoadError(null);
        setDataMeta({
          requestKey,
          dataVersion: typeof payload.data_version === "string" ? payload.data_version : null,
          asOf: String(payload.as_of ?? ""),
          cacheStatus: String(payload.cache_status ?? "unknown"),
          sourceId: parsed[parsed.length - 1].sourceId ?? "unknown",
          comparisonStatus: parsed[parsed.length - 1].comparisonStatus ?? "unknown",
        });
      })
      .catch((error: unknown) => {
        if (error instanceof Error && error.name === "AbortError") return;
        setLoadError({
          requestKey,
          message: error instanceof Error ? error.message : "真实行情加载失败",
        });
      });
    return () => controller.abort();
  }, [adjustmentBasis, requestKey, stock.instrumentId]);

  const currentRemoteData = dataMeta?.requestKey === requestKey ? remoteData : [];
  const currentError = loadError?.requestKey === requestKey ? loadError.message : "";
  const realtime = useRealtimeQuote(
    stock.instrumentId,
    !importedData && currentRemoteData.length >= 2,
  );
  const completedData = importedData ?? currentRemoteData;
  const allData = useMemo(
    () => importedData ?? mergeRealtimeCandle(currentRemoteData, realtime.quote),
    [currentRemoteData, importedData, realtime.quote],
  );
  const allIndicator = useMemo(
    () => alignFormalIndicators(allData, completedData),
    [allData, completedData],
  );

  function selectStock(code: string) {
    setStockCode(code);
    setImportedData(null);
    setImportName("");
    setHoverIndex(null);
  }

  const layoutClassName = [
    "app-layout",
    leftRailCollapsed ? "left-collapsed" : "",
    rightRailCollapsed ? "right-collapsed" : "",
  ].filter(Boolean).join(" ");

  if (!importedData && allData.length < 2) {
    return (
      <main className="page-shell">
        <div className={layoutClassName}>
          <LeftRail collapsed={leftRailCollapsed} selectedCode={stockCode} onToggle={() => setLeftRailCollapsed((value) => !value)} onSelect={selectStock} />
          <div className="content-column">
            <CenterTopbar sourceLabel="mootdx-cf 真实日线" realtimeStatus="connecting" />
            <section className="workspace data-state" id="top" aria-live="polite">
              {!currentError ? (
                <><div className="data-state-line" /><div className="data-state-line short" /><p>正在读取真实日线，R2 SQL 冷查询可能需要数秒。</p></>
              ) : (
                <><h2>暂时无法显示真实行情</h2><p>{currentError}</p><button type="button" onClick={() => setRetryNonce((value) => value + 1)}>重新读取</button></>
              )}
            </section>
          </div>
          <ReservedRightRail collapsed={rightRailCollapsed} onToggle={() => setRightRailCollapsed((value) => !value)} />
        </div>
      </main>
    );
  }

  const from = Math.max(0, allData.length - period);
  const data = allData.slice(from);
  const indicator = allIndicator.slice(from);
  const completedCloses = completedData.map((item) => item.close);
  const movingAverages = MOVING_AVERAGE_PERIODS.map((movingAveragePeriod) => ({
    period: movingAveragePeriod,
    values: (() => {
      const formalValues = fullWindowMovingAverage(completedCloses, movingAveragePeriod);
      const byDate = new Map(
        completedData.map((item, index) => [item.date, formalValues[index] ?? null]),
      );
      return data.map((item) => byDate.get(item.date) ?? null);
    })(),
  }));
  const activeIndex = hoverIndex ?? data.length - 1;
  const active = data[activeIndex];
  const activeIndicator = indicator[activeIndex];
  const latest = data[data.length - 1];
  const previous = data[data.length - 2] ?? latest;
  const activeRealtimeQuote = importedData ? null : realtime.quote;
  const displayPrice = activeRealtimeQuote?.last ?? latest.close;
  const change = activeRealtimeQuote && activeRealtimeQuote.previousClose > 0
    ? (activeRealtimeQuote.last / activeRealtimeQuote.previousClose - 1) * 100
    : (latest.close / previous.close - 1) * 100;
  const latestSignal = [...indicator].reverse().findIndex((point) => point.confirm);
  const signalIndex = latestSignal < 0 ? -1 : indicator.length - 1 - latestSignal;

  const plotWidth = CHART_WIDTH - LEFT - RIGHT;
  const pricePlotHeight = PRICE_HEIGHT - TOP - BOTTOM;
  const lows = data.map((item) => item.low);
  const highs = data.map((item) => item.high);
  const visibleMovingAverageValues = movingAverages.flatMap(({ values }) =>
    values.filter((value): value is number => value !== null),
  );
  const rawMin = Math.min(...lows, ...visibleMovingAverageValues);
  const rawMax = Math.max(...highs, ...visibleMovingAverageValues);
  const margin = (rawMax - rawMin) * 0.08;
  const priceMin = rawMin - margin;
  const priceMax = rawMax + margin;
  const xAt = (index: number) => LEFT + (index + 0.5) * plotWidth / data.length;
  const yPrice = (value: number) => TOP + (priceMax - value) / (priceMax - priceMin) * pricePlotHeight;
  const candleWidth = Math.max(2, Math.min(8, plotWidth / data.length * 0.62));
  const dateTickIndexes = Array.from(new Set([0, 1, 2, 3, 4, 5].map((step) => Math.round(step * (data.length - 1) / 5))));
  const priceTicks = axisTicks(priceMin, priceMax, 5);
  const yScore = (value: number) => TOP + (100 - value) / 100 * (INDICATOR_HEIGHT - TOP - BOTTOM);

  function onPointerMove(event: PointerEvent<SVGSVGElement>) {
    const box = event.currentTarget.getBoundingClientRect();
    const pointerX = (event.clientX - box.left) / box.width * CHART_WIDTH;
    const nextIndex = Math.round((pointerX - LEFT) / plotWidth * data.length - 0.5);
    setHoverIndex(Math.max(0, Math.min(data.length - 1, nextIndex)));
  }

  async function handleCsv(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    const text = await file.text();
    const rows = text.trim().split(/\r?\n/).slice(1);
    const parsed = rows.map((row) => {
      const [date, open, high, low, close, volume] = row.split(",").map((value) => value.trim());
      return { date, open: Number(open), high: Number(high), low: Number(low), close: Number(close), volume: Number(volume) };
    }).filter((item) => item.date && [item.open, item.high, item.low, item.close, item.volume].every(Number.isFinite));
    if (parsed.length >= 20) {
      setImportedData(parsed.sort((a, b) => a.date.localeCompare(b.date)));
      setImportName(file.name);
      setPeriod(Math.min(120, parsed.length));
      setHoverIndex(null);
    }
    event.target.value = "";
  }

  return (
    <main className="page-shell">
      <div className={layoutClassName}>
        <LeftRail
          collapsed={leftRailCollapsed}
          selectedCode={importedData ? null : stockCode}
          activePrice={importedData ? undefined : displayPrice}
          activeChange={importedData ? undefined : change}
          onToggle={() => setLeftRailCollapsed((value) => !value)}
          onSelect={selectStock}
        />
        <div className="content-column">
          <CenterTopbar
            sourceLabel={importedData ? "本地 CSV" : realtime.message}
            realtimeStatus={importedData ? "disabled" : realtime.status}
          />

      <section
        className="workspace"
        id="top"
        aria-label="股票走势图和指标图"
        data-series-fingerprint={seriesFingerprint(allData)}
        data-series-size={allData.length}
        data-realtime-status={importedData ? "disabled" : realtime.status}
        data-realtime-event-id={activeRealtimeQuote?.eventId ?? ""}
        data-realtime-observed-at={activeRealtimeQuote?.observedAt ?? ""}
      >
        <div className="toolbar">
          <div className="security-title">
            <span className="market-tag">{importedData ? "CSV" : stock.market}</span>
            <div><h2>{importedData ? importName.replace(/\.csv$/i, "") : stock.name}</h2><span>{importedData ? "本地导入数据" : stock.code}</span></div>
          </div>
          <div className="price-summary">
            <strong className={change >= 0 ? "price-up" : "price-down"}>{formatNumber(displayPrice)}</strong>
            <span className={change >= 0 ? "price-up" : "price-down"}>{change >= 0 ? "+" : ""}{formatNumber(change)}%</span>
          </div>
          <div className="toolbar-actions">
            <div className="basis-switch" aria-label="选择复权口径">
              {ADJUSTMENT_BASES.map((item) => <button type="button" disabled={Boolean(importedData)} className={adjustmentBasis === item.value ? "active" : ""} key={item.value} onClick={() => { setAdjustmentBasis(item.value); setHoverIndex(null); }}>{item.label}</button>)}
            </div>
            <div className="period-switch" aria-label="选择图表周期">
              {PERIODS.map((item) => <button className={period === item.value ? "active" : ""} key={item.value} onClick={() => { setPeriod(Math.min(item.value, allData.length)); setHoverIndex(null); }}>{item.label}</button>)}
            </div>
            <label className="upload-button">导入 CSV<input type="file" accept=".csv,text/csv" onChange={handleCsv} /></label>
          </div>
        </div>

        {!importedData && dataMeta && (
          <div className="provenance-strip" aria-label="行情数据血缘">
            <span>正式投影</span>
            <span>源 {dataMeta.sourceId}</span>
            <span>对照 {dataMeta.comparisonStatus}</span>
            <span>版本 {dataMeta.dataVersion?.slice(0, 12) ?? "未提供"}</span>
            <span>缓存 {dataMeta.cacheStatus}</span>
            <span>截面 {dataMeta.asOf.slice(0, 19).replace("T", " ")} UTC</span>
            {activeRealtimeQuote && (
              <>
                <span>实时源 tencent</span>
                <span>状态 {activeRealtimeQuote.marketSession}</span>
                <span>盘中蜡烛未确认</span>
              </>
            )}
          </div>
        )}

        <div className="quote-strip" aria-live="polite">
          <span>{active.date}</span><span>开 <b>{formatNumber(active.open)}</b></span><span>高 <b>{formatNumber(active.high)}</b></span><span>低 <b>{formatNumber(active.low)}</b></span><span>收 <b>{formatNumber(active.close)}</b></span><span>量 <b>{formatVolume(active.volume)}</b></span>
          {active.unconfirmed && <span className="unconfirmed-label">盘中未确认</span>}
          {movingAverages.map(({ period: movingAveragePeriod, values }) => (
            <span className={`ma-label ma-label-${movingAveragePeriod}`} key={movingAveragePeriod}>
              MA{movingAveragePeriod} <b>{values[activeIndex] === null ? "—" : formatNumber(values[activeIndex])}</b>
            </span>
          ))}
        </div>

          <div className="chart-stack">
          <div className="chart-heading">
            <div><h3>每日价格走势</h3>{latest.unconfirmed && <small className="intraday-note">末根为盘中实时蜡烛</small>}</div>
            <div className="legend" aria-label="价格图例">
              <span className="legend-item"><span className="legend-swatch legend-up" />上涨</span>
              <span className="legend-item"><span className="legend-swatch legend-down" />下跌</span>
              {movingAverages.map(({ period: movingAveragePeriod, values }) => (
                <span className={`legend-item legend-value legend-value-${movingAveragePeriod}`} key={movingAveragePeriod}>
                  <span className={`legend-swatch legend-ma-${movingAveragePeriod}`} />
                  MA{movingAveragePeriod}
                  <b>{values[activeIndex] === null ? "—" : formatNumber(values[activeIndex])}</b>
                </span>
              ))}
            </div>
          </div>

          <svg className="chart price-chart" viewBox={`0 0 ${CHART_WIDTH} ${PRICE_HEIGHT}`} role="img" aria-label={`${stock.name}日线蜡烛图，包含MA5、MA10、MA20、MA30和MA60均线`} onPointerMove={onPointerMove} onPointerLeave={() => setHoverIndex(null)}>
            {priceTicks.map((tick) => { const y = yPrice(tick); return <g key={tick}><line className="grid-line" x1={LEFT} x2={CHART_WIDTH - RIGHT} y1={y} y2={y} /><text className="axis-label" x={CHART_WIDTH - RIGHT + 12} y={y + 4}>{formatNumber(tick)}</text></g>; })}
            {dateTickIndexes.map((index) => <g key={index}><line className="grid-line vertical" x1={xAt(index)} x2={xAt(index)} y1={TOP} y2={PRICE_HEIGHT - BOTTOM} /><text className="date-label" x={xAt(index)} y={PRICE_HEIGHT - 7} textAnchor="middle">{formatDate(data[index].date)}</text></g>)}
            {data.map((item, index) => {
              const up = item.close >= item.open;
              const x = xAt(index);
              const bodyTop = yPrice(Math.max(item.open, item.close));
              const bodyBottom = yPrice(Math.min(item.open, item.close));
              const className = ["candle", up ? "up" : "down", item.realtime ? "realtime-candle" : ""].filter(Boolean).join(" ");
              return <g className={className} data-date={item.date} data-open={item.open} data-high={item.high} data-low={item.low} data-close={item.close} data-volume={item.volume} data-unconfirmed={item.unconfirmed ? "true" : "false"} key={item.date}><line x1={x} x2={x} y1={yPrice(item.high)} y2={yPrice(item.low)} /><rect x={x - candleWidth / 2} y={bodyTop} width={candleWidth} height={Math.max(1.5, bodyBottom - bodyTop)} /></g>;
            })}
            {movingAverages.map(({ period: movingAveragePeriod, values }) => (
              <polyline
                className={`ma-line ma-line-${movingAveragePeriod}`}
                key={movingAveragePeriod}
                points={values.flatMap((value, index) => value === null ? [] : `${xAt(index)},${yPrice(value)}`).join(" ")}
              />
            ))}
            {hoverIndex !== null && <line className="crosshair" x1={xAt(hoverIndex)} x2={xAt(hoverIndex)} y1={TOP} y2={PRICE_HEIGHT - BOTTOM} />}
            <rect className="pointer-layer" x={LEFT} y={TOP} width={plotWidth} height={pricePlotHeight} />
          </svg>

          <div className="chart-divider" />
          <div className="chart-heading indicator-heading">
            <div><h3>低位承接强度</h3></div>
            <div className="indicator-readout"><span>当前</span><strong>{formatNumber(activeIndicator.score, 1)}</strong><small>{activeIndicator.score >= 60 ? "强承接" : activeIndicator.score >= 30 ? "观察" : "中性"}</small></div>
          </div>

          <svg className="chart indicator-chart" viewBox={`0 0 ${CHART_WIDTH} ${INDICATOR_HEIGHT}`} role="img" aria-label="低位承接强度指标图" onPointerMove={onPointerMove} onPointerLeave={() => setHoverIndex(null)}>
            {[0, 30, 60, 100].map((tick) => <g key={tick}><line className={`score-grid score-${tick}`} x1={LEFT} x2={CHART_WIDTH - RIGHT} y1={yScore(tick)} y2={yScore(tick)} /><text className="axis-label" x={CHART_WIDTH - RIGHT + 12} y={yScore(tick) + 4}>{tick}</text></g>)}
            {indicator.map((point, index) => { const y = yScore(point.score); return <rect className={point.score >= 60 ? "score-bar strong" : "score-bar"} key={data[index].date} x={xAt(index) - candleWidth / 2} y={y} width={candleWidth} height={yScore(0) - y} rx={Math.min(2, candleWidth / 3)} />; })}
            <polyline className="score-line" points={indicator.map((point, index) => `${xAt(index)},${yScore(point.score)}`).join(" ")} />
            {indicator.map((point, index) => point.confirm ? <g className="confirm-marker" key={`confirm-${data[index].date}`}><circle cx={xAt(index)} cy={yScore(Math.min(point.score + 9, 96))} r="5" /><path d={`M ${xAt(index) - 2.5} ${yScore(Math.min(point.score + 9, 96))} l 2 2.5 l 4 -5`} /></g> : null)}
            {hoverIndex !== null && <line className="crosshair" x1={xAt(hoverIndex)} x2={xAt(hoverIndex)} y1={TOP} y2={INDICATOR_HEIGHT - BOTTOM} />}
            <rect className="pointer-layer" x={LEFT} y={TOP} width={plotWidth} height={INDICATOR_HEIGHT - TOP - BOTTOM} />
          </svg>
        </div>
      </section>

      <section className="factor-card" aria-label="光标日指标拆解">
          <div className="factor-card-head"><div><h2>指标拆解</h2><span>{active.date}</span></div>{activeIndicator.confirm && <span className="confirm-pill">确认信号</span>}</div>
          <div className="factor-grid">
          <div className="factor-row"><span>收盘位置</span><div><i style={{ width: `${Math.min(activeIndicator.closePosition * 100, 100)}%` }} /></div><b>{formatNumber(activeIndicator.closePosition * 100, 0)}%</b></div>
          <div className="factor-row"><span>相对量能</span><div><i style={{ width: `${Math.min(activeIndicator.volumeRatio / 2 * 100, 100)}%` }} /></div><b>{formatNumber(activeIndicator.volumeRatio, 2)}×</b></div>
          <div className="factor-row"><span>承接强度</span><div><i style={{ width: `${activeIndicator.score}%` }} /></div><b>{formatNumber(activeIndicator.score, 1)}</b></div>
          </div>
          <p className="signal-note">{active.unconfirmed ? "盘中实时蜡烛仅用于观察，正式指标继续使用已完成日线。" : signalIndex >= 0 ? `窗口内最近一次确认出现在 ${data[signalIndex].date}。` : "当前窗口没有满足全部确认条件的日期。"}</p>
      </section>
        </div>
        <ReservedRightRail collapsed={rightRailCollapsed} onToggle={() => setRightRailCollapsed((value) => !value)} />
      </div>
    </main>
  );
}
