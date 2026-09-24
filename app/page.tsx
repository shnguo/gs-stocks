"use client";

import {
  CaretDownIcon,
  MagnifyingGlassIcon,
  PlusIcon,
  SidebarSimpleIcon,
  XIcon,
} from "@phosphor-icons/react";
import {
  ChangeEvent,
  KeyboardEvent,
  MouseEvent,
  PointerEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import {
  prefetchRealtimeQuotes,
  useRealtimeQuote,
  type RealtimeConnectionStatus,
  type RealtimeQuote,
} from "./use-realtime-quote";
import {
  inferredProvider,
  useRealtimeProvider,
  type RealtimeProviderInfo,
} from "./use-realtime-provider";
import {
  closedBarRevalidationDelayMs,
  completedQuoteNeedsHistoricalRevalidation,
  displayMovingAverageValues,
  hasHistoricalSeriesGap,
  historicalSeriesGapDates,
  instrumentReadinessRetryDelayMs,
  isCompletedRealtimeQuote,
  selectCompletedSeries,
} from "./market-series";
import { readChartCache, writeChartCache } from "./chart-cache";
import {
  chartDataIndexAt,
  chartHorizontalLayout,
  chartXAt,
} from "./chart-layout";
import {
  isPlaceholderFavoriteName,
  stockSearchReadinessLabel,
  stockSearchResultIsAddable,
  type StockSearchHistoricalStatus,
} from "./lib/favorite-stock";
import { realtimeMiddayState } from "./lib/realtime-market-schedule";
import {
  isSupportedAShareInstrumentId,
  marketInstrumentFromCatalog,
  providerSupportsRealtimeInstrument,
  realtimeMarketInstrument,
  savedMarketInstrument,
  type AdjustmentBasis,
  type MarketGroup,
  type MarketInstrumentIdentity,
} from "./lib/stock-symbol";

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

type StockPreset = MarketInstrumentIdentity & {
  name: string;
};

type DataMeta = {
  requestKey: string;
  dataVersion: string | null;
  asOf: string;
  cacheStatus: string;
  sourceId: string;
  comparisonStatus: string;
  generationId: string | null;
  businessDate: string | null;
  stockReadiness: string | null;
  marketReadiness: string | null;
  availableAdjustmentBases: AdjustmentBasis[];
};

type ParsedChartData = {
  rows: Candle[];
  meta: DataMeta;
};

type MarketDayContinuity = {
  requestKey: string;
  status: "loading" | "ready";
  closedDates: string[];
};

type HistoricalGapRepairState = {
  requestKey: string;
  status: "repairing" | "failed";
};

type InstrumentDayReadiness = {
  requestKey: string;
  readiness: string;
  chartReady: boolean;
  marketComplete: boolean;
};

type WatchlistMenuState = {
  instrumentId: string;
  x: number;
  y: number;
};

type StockSearchResult = StockPreset & {
  pinyinInitials: string;
  historicalStatus: StockSearchHistoricalStatus;
};

type StockSearchStatus = "idle" | "loading" | "success" | "error";
type AdmissionSubmitStatus = "idle" | "sending" | "submitted" | "error";

const STOCKS: StockPreset[] = [
  { code: "688008", name: "澜起科技", exchangeId: "xshg", market: "科创板", marketGroup: "A股", marketRegion: "cn", currency: "CNY", availableAdjustmentBases: ["none", "qfq", "hfq"], instrumentId: "cn.xshg.688008" },
  { code: "600036", name: "招商银行", exchangeId: "xshg", market: "沪市", marketGroup: "A股", marketRegion: "cn", currency: "CNY", availableAdjustmentBases: ["none", "qfq", "hfq"], instrumentId: "cn.xshg.600036" },
  { code: "000333", name: "美的集团", exchangeId: "xshe", market: "深市", marketGroup: "A股", marketRegion: "cn", currency: "CNY", availableAdjustmentBases: ["none", "qfq", "hfq"], instrumentId: "cn.xshe.000333" },
  { code: "300059", name: "东方财富", exchangeId: "xshe", market: "创业板", marketGroup: "A股", marketRegion: "cn", currency: "CNY", availableAdjustmentBases: ["none", "qfq", "hfq"], instrumentId: "cn.xshe.300059" },
];

const FAVORITES_STORAGE_KEY = "stock-indicator-dashboard.favorite-stocks:v5";
const PREVIOUS_FAVORITES_STORAGE_KEY = "stock-indicator-dashboard.favorite-stocks:v4";
const LEGACY_FAVORITES_STORAGE_KEY = "stock-indicator-dashboard.favorite-stock-codes:v1";

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
const MAX_CLOSED_BAR_REVALIDATION_ATTEMPTS = 64;
const REALTIME_COHERENCE_GRACE_MS = 2_000;
const WATCHLIST_REFRESH_INTERVAL_MS = 15_000;

function savedStock(value: unknown): StockPreset | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const record = value as Record<string, unknown>;
  if (typeof record.code !== "string" || typeof record.name !== "string") return null;
  const identity = savedMarketInstrument(record);
  const name = record.name.trim().slice(0, 30);
  if (!identity || !name) return null;
  return { ...identity, name };
}

function catalogStock(value: unknown): StockPreset | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const record = value as Record<string, unknown>;
  const identity = marketInstrumentFromCatalog(record);
  const name = typeof record.name === "string" ? record.name.trim().slice(0, 30) : "";
  return identity && name ? { ...identity, name } : null;
}

function admittedStock(value: unknown): StockPreset | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const record = value as Record<string, unknown>;
  const candidate = record.candidate;
  if (!candidate || typeof candidate !== "object" || Array.isArray(candidate)) return null;
  const candidateRecord = candidate as Record<string, unknown>;
  return catalogStock({
    instrument_id: candidateRecord.instrument_id,
    market: candidateRecord.market,
    exchange_id: candidateRecord.exchange_id,
    symbol: candidateRecord.listing_symbol,
    name: candidateRecord.display_name,
    currency: candidateRecord.currency,
    available_adjustment_bases: candidateRecord.available_adjustment_bases,
  });
}

function savedStockList(value: unknown): StockPreset[] {
  if (!Array.isArray(value)) return [];
  const stocks = value.map((item) =>
    typeof item === "string"
      ? STOCKS.find((stock) => stock.code === item) ?? null
      : savedStock(item)
  ).filter((item): item is StockPreset => item !== null);
  return Array.from(new Map(stocks.map((item) => [item.instrumentId, item])).values());
}

async function resolveCanonicalFavorite(
  stock: StockPreset,
  signal: AbortSignal,
): Promise<StockPreset | null> {
  try {
    const response = await fetch(`/api/stock-search?q=${encodeURIComponent(stock.code)}`, { signal });
    if (!response.ok) return null;
    const payload = await response.json() as { results?: unknown[] };
    const candidates = Array.isArray(payload.results)
      ? payload.results.map(savedStock).filter((item): item is StockPreset => item !== null)
      : [];
    return candidates.find((item) => item.instrumentId === stock.instrumentId) ?? null;
  } catch (error) {
    if (signal.aborted || (error instanceof Error && error.name === "AbortError")) return null;
    return null;
  }
}

function parseChartPayload(
  payload: unknown,
  requestKey: string,
  cacheStatus?: string,
): ParsedChartData {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw new Error("真实行情响应格式无效");
  }
  const record = payload as Record<string, unknown>;
  const sourceRows = Array.isArray(record.rows) ? record.rows : [];
  const rows = sourceRows.map((row): Candle | null => {
    const value = row && typeof row === "object" && !Array.isArray(row)
      ? row as Record<string, unknown>
      : {};
    const date = String(value.trading_date ?? "");
    const open = finiteMarketNumber(value.open);
    const high = finiteMarketNumber(value.high);
    const low = finiteMarketNumber(value.low);
    const close = finiteMarketNumber(value.close);
    const volume = finiteMarketNumber(value.volume);
    if (!date || open === null || high === null || low === null || close === null || volume === null) {
      return null;
    }
    return {
      date,
      open,
      high,
      low,
      close,
      volume,
      sourceId: String(value.source_id ?? "unknown"),
      comparisonStatus: String(value.comparison_status ?? "unknown"),
    };
  }).filter((item): item is Candle => item !== null)
    .sort((a, b) => a.date.localeCompare(b.date));
  if (rows.length < 2) throw new Error("该股票暂无足够的已发布日线");
  const availableAdjustmentBases = Array.isArray(record.available_adjustment_bases)
    ? record.available_adjustment_bases.filter(isAdjustmentBasis)
    : ADJUSTMENT_BASES.map((item) => item.value);
  return {
    rows,
    meta: {
      requestKey,
      dataVersion: typeof record.data_version === "string" ? record.data_version : null,
      asOf: String(record.as_of ?? ""),
      cacheStatus: cacheStatus ?? String(record.cache_status ?? "unknown"),
      sourceId: rows[rows.length - 1]?.sourceId ?? "unknown",
      comparisonStatus: rows[rows.length - 1]?.comparisonStatus ?? "unknown",
      generationId: typeof record.generation_id === "string" ? record.generation_id : null,
      businessDate: typeof record.business_date === "string" ? record.business_date : null,
      stockReadiness: typeof record.stock_readiness === "string" ? record.stock_readiness : null,
      marketReadiness: typeof record.market_readiness === "string" ? record.market_readiness : null,
      availableAdjustmentBases,
    },
  };
}

function isAdjustmentBasis(value: unknown): value is AdjustmentBasis {
  return value === "none" || value === "qfq" || value === "hfq";
}

function finiteMarketNumber(value: unknown): number | null {
  if (typeof value !== "number" && typeof value !== "string") return null;
  if (typeof value === "string" && !value.trim()) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function parseHistoricalGapRepairRows(
  payload: unknown,
  startDate: string,
  endDate: string,
  adjustmentBasis: AdjustmentBasis,
): Candle[] {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) return [];
  const snapshot = (payload as Record<string, unknown>).snapshot;
  if (!snapshot || typeof snapshot !== "object" || Array.isArray(snapshot)) return [];
  const sourceRows = (snapshot as Record<string, unknown>).rows;
  if (!Array.isArray(sourceRows)) return [];
  return sourceRows.flatMap((row): Candle[] => {
    if (!row || typeof row !== "object" || Array.isArray(row)) return [];
    const value = row as Record<string, unknown>;
    const date = typeof value.trading_date === "string" ? value.trading_date : "";
    const open = finiteMarketNumber(value.open);
    const high = finiteMarketNumber(value.high);
    const low = finiteMarketNumber(value.low);
    const close = finiteMarketNumber(value.close);
    const volume = finiteMarketNumber(value.volume);
    if (
      date < startDate ||
      date > endDate ||
      value.adjustment_basis !== adjustmentBasis ||
      open === null ||
      high === null ||
      low === null ||
      close === null ||
      volume === null
    ) return [];
    return [{
      date,
      open,
      high,
      low,
      close,
      volume,
      sourceId: String(value.source_id ?? "unknown"),
      comparisonStatus: String(value.comparison_status ?? "single_source_unverified"),
    }];
  }).sort((left, right) => left.date.localeCompare(right.date));
}

function mergeHistoricalGapRows(history: Candle[], recovered: Candle[]): Candle[] {
  const byDate = new Map(history.map((row) => [row.date, row]));
  recovered.forEach((row) => byDate.set(row.date, row));
  return Array.from(byDate.values()).sort((left, right) => left.date.localeCompare(right.date));
}

class ChartRequestError extends Error {
  constructor(message: string, readonly fallbackBasis: AdjustmentBasis | null = null) {
    super(message);
    this.name = "ChartRequestError";
  }
}

async function fetchChartPayload(
  query: URLSearchParams,
  signal: AbortSignal,
  cache: RequestCache = "default",
  maximumAttempts = 3,
): Promise<unknown> {
  for (let attempt = 0; attempt < maximumAttempts; attempt += 1) {
    const response = await fetch(`/api/chart-daily-bars?${query}`, { signal, cache });
    const payload: unknown = await response.json().catch(() => null);
    if (response.ok) return payload;
    const record = payload && typeof payload === "object" && !Array.isArray(payload)
      ? payload as Record<string, unknown>
      : {};
    const message = typeof record.error === "string" ? record.error : "真实行情加载失败";
    const fallbackBasis = record.fallback_adjustment_basis === "none" ? "none" : null;
    if (fallbackBasis) throw new ChartRequestError(message, fallbackBasis);
    if (record.retryable === true && attempt + 1 < maximumAttempts) {
      const requestedDelay = Number(record.retry_after_seconds ?? response.headers.get("retry-after"));
      const delaySeconds = Number.isFinite(requestedDelay)
        ? Math.min(5, Math.max(1, requestedDelay))
        : 2;
      await waitForChartRetry(delaySeconds * 1_000, signal);
      continue;
    }
    throw new ChartRequestError(message);
  }
  throw new ChartRequestError("日线快照生成超时，请重试");
}

async function fetchVerifiedClosedMarketDays(
  startDate: string,
  endDate: string,
  signal: AbortSignal,
): Promise<string[]> {
  const query = new URLSearchParams({ start_date: startDate, end_date: endDate });
  const response = await fetch(`/api/market-days?${query}`, {
    signal,
    cache: "no-store",
  });
  const payload: unknown = await response.json().catch(() => null);
  if (!response.ok || !payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw new Error("无法核对交易日连续性");
  }
  const rows = Array.isArray((payload as Record<string, unknown>).rows)
    ? (payload as Record<string, unknown>).rows as unknown[]
    : [];
  return rows.flatMap((row) => {
    if (!row || typeof row !== "object" || Array.isArray(row)) return [];
    const record = row as Record<string, unknown>;
    return record.status === "closed" && typeof record.business_date === "string"
      ? [record.business_date]
      : [];
  });
}

async function fetchInstrumentDayReadiness(
  businessDate: string,
  instrumentId: string,
  signal: AbortSignal,
): Promise<InstrumentDayReadiness | null> {
  const query = new URLSearchParams({
    business_date: businessDate,
    instrument_id: instrumentId,
  });
  const response = await fetch(`/api/market-day-readiness?${query}`, {
    signal,
    cache: "no-store",
  });
  const payload: unknown = await response.json().catch(() => null);
  if (!response.ok || !payload || typeof payload !== "object" || Array.isArray(payload)) {
    return null;
  }
  const record = payload as Record<string, unknown>;
  if (
    typeof record.readiness !== "string" ||
    typeof record.chart_ready !== "boolean" ||
    typeof record.market_complete !== "boolean"
  ) {
    return null;
  }
  return {
    requestKey: `${instrumentId}:${businessDate}`,
    readiness: record.readiness,
    chartReady: record.chart_ready,
    marketComplete: record.market_complete,
  };
}

function waitForChartRetry(milliseconds: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const onAbort = () => {
      window.clearTimeout(timer);
      reject(new DOMException("The operation was aborted", "AbortError"));
    };
    const timer = window.setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve();
    }, milliseconds);
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

function movingAverage(values: number[], length: number) {
  return values.map((_, index) => {
    const from = Math.max(0, index - length + 1);
    const window = values.slice(from, index + 1);
    return window.reduce((sum, value) => sum + value, 0) / window.length;
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

function mergeRealtimeCandle(
  history: Candle[],
  quote: RealtimeQuote | null,
  verifiedClosedDates: ReadonlySet<string> = new Set(),
  allowHistoricalGap = false,
): Candle[] {
  if (!quote || history.length === 0) return history;
  const historicalGap = hasHistoricalSeriesGap(history, quote, verifiedClosedDates);
  if (!allowHistoricalGap && historicalGap) {
    return history;
  }
  const latest = history[history.length - 1];
  if (!latest || quote.tradingDate < latest.date) return history;
  const replacingLatest = quote.tradingDate === latest.date;
  const reference = replacingLatest ? history[history.length - 2] : latest;
  const adjustmentFactor = historicalGap
    ? 1
    : reference && quote.previousClose > 0
      ? reference.close / quote.previousClose
      : 1;
  const quoteIsComplete = isCompletedRealtimeQuote(quote);
  if (replacingLatest && quoteIsComplete) return history;
  const candle: Candle = {
    date: quote.tradingDate,
    open: quote.open * adjustmentFactor,
    high: quote.high * adjustmentFactor,
    low: quote.low * adjustmentFactor,
    close: quote.last * adjustmentFactor,
    volume: quote.cumulativeVolume,
    sourceId: quote.provider,
    comparisonStatus: quoteIsComplete ? "complete" : "unconfirmed",
    realtime: true,
    unconfirmed: !quoteIsComplete,
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
  favoriteStocks,
  selectedCode,
  quotesByInstrumentId,
  activePrice,
  activeChange,
  onAdd,
  onRemove,
  onSelect,
}: {
  favoriteStocks: StockPreset[];
  selectedCode: string | null;
  quotesByInstrumentId: Readonly<Record<string, RealtimeQuote>>;
  activePrice?: number;
  activeChange?: number;
  onAdd: (stock: StockPreset) => void;
  onRemove: (code: string) => void;
  onSelect: (code: string) => void;
}) {
  const [activeMarket, setActiveMarket] = useState<MarketGroup>("A股");
  const [managerOpen, setManagerOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<StockSearchResult[]>([]);
  const [searchStatus, setSearchStatus] = useState<StockSearchStatus>("idle");
  const [searchError, setSearchError] = useState("");
  const [admissionStatus, setAdmissionStatus] = useState<AdmissionSubmitStatus>("idle");
  const [admissionMessage, setAdmissionMessage] = useState("");
  const [admissionRequestId, setAdmissionRequestId] = useState("");
  const [admissionPollNonce, setAdmissionPollNonce] = useState(0);
  const [highlightedResult, setHighlightedResult] = useState(0);
  const [stockMenu, setStockMenu] = useState<WatchlistMenuState | null>(null);
  const stockMenuRef = useRef<HTMLDivElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const favoriteInstrumentSet = new Set(favoriteStocks.map((item) => item.instrumentId));
  const visibleStocks = favoriteStocks.filter((item) => item.marketGroup === activeMarket);
  const menuStock = favoriteStocks.find(
    (item) => item.instrumentId === stockMenu?.instrumentId,
  ) ?? null;

  function closeFavoriteManager() {
    setManagerOpen(false);
    setSearchQuery("");
    setSearchResults([]);
    setSearchStatus("idle");
    setSearchError("");
    setAdmissionStatus("idle");
    setAdmissionMessage("");
    setAdmissionRequestId("");
    setAdmissionPollNonce(0);
    setHighlightedResult(0);
  }

  function addSearchResult(stock: StockSearchResult) {
    if (
      favoriteInstrumentSet.has(stock.instrumentId) ||
      !stockSearchResultIsAddable(stock.historicalStatus)
    ) return;
    onAdd(stock);
    setActiveMarket(stock.marketGroup);
    setSearchQuery("");
    setSearchResults([]);
    setSearchStatus("idle");
    setSearchError("");
    setHighlightedResult(0);
    setManagerOpen(false);
  }

  async function requestCustomSymbol() {
    const query = searchQuery.trim();
    const market = activeMarket === "港股" ? "hk" : activeMarket === "美股" ? "us" : null;
    if (!query || !market || admissionStatus === "sending") return;
    setAdmissionStatus("sending");
    setAdmissionMessage("正在提交收录申请…");
    try {
      const response = await fetch("/api/symbol-admissions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ query, market }),
      });
      const payload: unknown = await response.json().catch(() => null);
      const record = payload && typeof payload === "object" && !Array.isArray(payload)
        ? payload as Record<string, unknown>
        : {};
      if (!response.ok) throw new Error(
        typeof record.error === "string" ? record.error : "股票收录申请失败",
      );
      const requestId = typeof record.request_id === "string" ? record.request_id : "";
      if (!/^[0-9a-f]{32}$/u.test(requestId)) throw new Error("股票收录申请编号无效");
      setAdmissionStatus("submitted");
      setAdmissionRequestId(requestId);
      setAdmissionMessage("申请已提交。系统会先核验代码、数据权限和历史覆盖，再加入自选列表。");
    } catch (error) {
      setAdmissionStatus("error");
      setAdmissionMessage(error instanceof Error ? error.message : "股票收录申请失败");
    }
  }

  function handleSearchKey(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === "Escape") {
      if (searchQuery) {
        event.preventDefault();
        setSearchQuery("");
        setSearchResults([]);
        setSearchStatus("idle");
        setSearchError("");
        setHighlightedResult(0);
      } else {
        closeFavoriteManager();
      }
      return;
    }
    if (!searchResults.length) return;
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      const direction = event.key === "ArrowDown" ? 1 : -1;
      setHighlightedResult((current) =>
        (current + direction + searchResults.length) % searchResults.length
      );
      return;
    }
    if (event.key === "Enter") {
      event.preventDefault();
      const result = searchResults[highlightedResult];
      if (result) addSearchResult(result);
    }
  }

  function openStockMenu(instrumentId: string, clientX: number, clientY: number) {
    const menuWidth = 168;
    const menuHeight = 92;
    const viewportMargin = 8;
    setStockMenu({
      instrumentId,
      x: Math.max(viewportMargin, Math.min(clientX, window.innerWidth - menuWidth - viewportMargin)),
      y: Math.max(viewportMargin, Math.min(clientY, window.innerHeight - menuHeight - viewportMargin)),
    });
  }

  function handleStockContextMenu(
    event: MouseEvent<HTMLButtonElement>,
    instrumentId: string,
  ) {
    event.preventDefault();
    openStockMenu(instrumentId, event.clientX, event.clientY);
  }

  function handleStockMenuKey(
    event: KeyboardEvent<HTMLButtonElement>,
    instrumentId: string,
  ) {
    if (event.key !== "ContextMenu" && !(event.shiftKey && event.key === "F10")) return;
    event.preventDefault();
    const bounds = event.currentTarget.getBoundingClientRect();
    openStockMenu(instrumentId, bounds.left + 14, bounds.top + bounds.height - 4);
  }

  function handleMenuKey(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === "Escape") {
      event.preventDefault();
      const instrumentId = stockMenu?.instrumentId;
      setStockMenu(null);
      window.requestAnimationFrame(() => {
        document.querySelector<HTMLButtonElement>(
          `[data-stock-code="${instrumentId}"]`,
        )?.focus();
      });
      return;
    }
    if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
    event.preventDefault();
    const menuItems = Array.from(event.currentTarget.querySelectorAll<HTMLButtonElement>("[role='menuitem']"));
    const activeIndex = menuItems.indexOf(document.activeElement as HTMLButtonElement);
    const direction = event.key === "ArrowDown" ? 1 : -1;
    const nextIndex = (activeIndex + direction + menuItems.length) % menuItems.length;
    menuItems[nextIndex]?.focus();
  }

  useEffect(() => {
    if (!stockMenu) return;
    const focusFrame = window.requestAnimationFrame(() => {
      stockMenuRef.current?.querySelector<HTMLButtonElement>("[role='menuitem']")?.focus();
    });
    const closeMenu = () => setStockMenu(null);
    window.addEventListener("pointerdown", closeMenu);
    window.addEventListener("resize", closeMenu);
    window.addEventListener("scroll", closeMenu, true);
    return () => {
      window.cancelAnimationFrame(focusFrame);
      window.removeEventListener("pointerdown", closeMenu);
      window.removeEventListener("resize", closeMenu);
      window.removeEventListener("scroll", closeMenu, true);
    };
  }, [stockMenu]);

  useEffect(() => {
    if (!managerOpen) return;
    const focusFrame = window.requestAnimationFrame(() => searchInputRef.current?.focus());
    return () => window.cancelAnimationFrame(focusFrame);
  }, [managerOpen]);

  useEffect(() => {
    if (!managerOpen) return;
    const query = searchQuery.trim();
    if (!query) return;

    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      try {
        const market = activeMarket === "A股" ? "cn" : activeMarket === "港股" ? "hk" : "us";
        const search = new URLSearchParams({ q: query, market });
        const response = await fetch(`/api/stock-search?${search}`, {
          signal: controller.signal,
        });
        const payload = await response.json() as { results?: StockSearchResult[]; error?: string };
        if (!response.ok) throw new Error(payload.error || "股票搜索失败");
        const nextResults = Array.isArray(payload.results) ? payload.results : [];
        setSearchResults(nextResults);
        if (nextResults.length > 0) {
          setAdmissionStatus("idle");
          setAdmissionMessage("");
          setAdmissionRequestId("");
          setAdmissionPollNonce(0);
        }
        setSearchStatus("success");
      } catch (error) {
        if (controller.signal.aborted) return;
        setSearchResults([]);
        setSearchError(error instanceof Error ? error.message : "股票搜索失败，请重试");
        setSearchStatus("error");
      }
    }, 180);

    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [activeMarket, managerOpen, searchQuery]);

  useEffect(() => {
    if (!managerOpen || admissionStatus !== "submitted" || !admissionRequestId) return;
    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      try {
        const response = await fetch(`/api/symbol-admissions/${admissionRequestId}`, {
          cache: "no-store",
          signal: controller.signal,
        });
        const payload: unknown = await response.json().catch(() => null);
        const record = payload && typeof payload === "object" && !Array.isArray(payload)
          ? payload as Record<string, unknown>
          : {};
        if (!response.ok) throw new Error(
          typeof record.error === "string" ? record.error : "无法读取股票收录进度",
        );
        const status = typeof record.status === "string" ? record.status : "";
        if (status === "active") {
          const admitted = admittedStock(record);
          if (!admitted) throw new Error("股票已完成收录，但返回的标的信息无效");
          onAdd(admitted);
          setActiveMarket(admitted.marketGroup);
          setAdmissionStatus("idle");
          setAdmissionMessage("");
          setAdmissionRequestId("");
          setSearchQuery("");
          setSearchResults([]);
          setSearchStatus("idle");
          setManagerOpen(false);
          return;
        }
        if (["rejected", "unsupported", "quota-blocked", "rights-blocked", "failed", "inactive"].includes(status)) {
          setAdmissionStatus("error");
          setAdmissionMessage(
            typeof record.reason === "string" && record.reason
              ? record.reason
              : "该股票暂时无法完成收录，请稍后重试。",
          );
          return;
        }
        const progress = status === "resolving"
          ? "正在核验交易所、代码和数据源映射…"
          : status === "bootstrapping"
            ? "代码已确认，正在生成并校验历史走势图…"
            : "申请已排队，等待核验…";
        setAdmissionMessage(progress);
        setAdmissionPollNonce((value) => value + 1);
      } catch (error) {
        if (controller.signal.aborted) return;
        setAdmissionStatus("error");
        setAdmissionMessage(error instanceof Error ? error.message : "无法读取股票收录进度");
      }
    }, admissionPollNonce === 0 ? 900 : 2_500);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [admissionPollNonce, admissionRequestId, admissionStatus, managerOpen, onAdd]);

  return (
    <nav className="watchlist-sidebar" aria-label="收藏股票">
      <div className="watchlist-heading">
        <div><h2>自选股票</h2><p>右键管理</p></div>
        <div className="watchlist-heading-actions">
          <span aria-live="polite">{favoriteStocks.length} 只</span>
          <button
            className="watchlist-manage-button"
            type="button"
            aria-expanded={managerOpen}
            aria-controls="favorite-stock-manager"
            title="添加自选股票"
            onClick={() => managerOpen ? closeFavoriteManager() : setManagerOpen(true)}
          >
            <PlusIcon aria-hidden="true" size={16} weight="regular" />
            <span className="sr-only">添加自选股票</span>
          </button>
        </div>
      </div>
      {managerOpen && (
        <section className="favorites-manager" id="favorite-stock-manager" aria-label="添加自选股票">
          <div className="favorites-manager-heading">
            <div><strong>添加股票</strong><small>查找 {activeMarket}，未收录代码可提交核验</small></div>
            <button type="button" title="关闭添加面板" onClick={closeFavoriteManager}>
              <XIcon aria-hidden="true" size={15} weight="regular" />
              <span className="sr-only">关闭添加面板</span>
            </button>
          </div>
          <div className="favorites-search">
            <label className="sr-only" htmlFor="favorite-stock-search">搜索股票</label>
            <div className="favorites-search-field">
              <MagnifyingGlassIcon aria-hidden="true" size={15} weight="regular" />
              <input
                id="favorite-stock-search"
                ref={searchInputRef}
                type="search"
                role="combobox"
                value={searchQuery}
                placeholder="代码、名称或拼音"
                autoComplete="off"
                maxLength={40}
                aria-autocomplete="list"
                aria-expanded={searchStatus !== "idle"}
                aria-controls="favorite-stock-results"
                aria-activedescendant={searchResults[highlightedResult]
                  ? `favorite-stock-result-${highlightedResult}`
                  : undefined}
                aria-describedby="favorite-stock-feedback"
                onChange={(event) => {
                  const value = event.target.value;
                  setSearchQuery(value);
                  setSearchError("");
                  setAdmissionStatus("idle");
                  setAdmissionMessage("");
                  setAdmissionRequestId("");
                  setAdmissionPollNonce(0);
                  setSearchResults([]);
                  setHighlightedResult(0);
                  setSearchStatus(value.trim() ? "loading" : "idle");
                }}
                onKeyDown={handleSearchKey}
              />
            </div>
            <p
              className={searchStatus === "error" ? "favorites-search-feedback error" : "favorites-search-feedback"}
              id="favorite-stock-feedback"
              role={searchStatus === "error" ? "alert" : "status"}
            >
              {searchStatus === "loading"
                ? "正在搜索…"
                : searchStatus === "error"
                  ? searchError
                  : searchStatus === "success" && searchResults.length === 0
                    ? `没有匹配的${activeMarket}，可换个关键词${activeMarket === "A股" ? "。" : "或提交收录核验。"}`
                    : searchStatus === "success" && searchResults.every(
                      (stock) => !stockSearchResultIsAddable(stock.historicalStatus)
                    )
                      ? "已找到默认收录股票，历史数据准备完成后可加入自选。"
                      : "支持代码、中文、字母、全拼和拼音首字母。"}
            </p>
            {searchStatus === "success" && searchResults.length === 0 && activeMarket !== "A股" && (
              <div className="favorites-admission" aria-live="polite">
                <button
                  type="button"
                  disabled={admissionStatus === "sending" || admissionStatus === "submitted"}
                  onClick={() => void requestCustomSymbol()}
                >
                  {admissionStatus === "sending"
                    ? "正在提交…"
                    : admissionStatus === "submitted"
                      ? "已提交核验"
                      : `申请收录 ${searchQuery.trim()}`}
                </button>
                {admissionMessage && (
                  <p className={admissionStatus === "error" ? "error" : ""}>{admissionMessage}</p>
                )}
              </div>
            )}
            {searchStatus !== "idle" && (
              <div className="favorites-search-results" id="favorite-stock-results" role="listbox" aria-label="股票搜索结果">
                {searchStatus === "loading" ? (
                  <div className="favorites-search-state" aria-hidden="true"><span /><span /><span /></div>
                ) : searchResults.map((stock, index) => {
                  const alreadyFavorite = favoriteInstrumentSet.has(stock.instrumentId);
                  const addable = stockSearchResultIsAddable(stock.historicalStatus);
                  const readinessLabel = stockSearchReadinessLabel(stock.historicalStatus);
                  return (
                    <button
                      type="button"
                      role="option"
                      id={`favorite-stock-result-${index}`}
                      key={stock.instrumentId}
                      className={index === highlightedResult ? "active" : ""}
                      aria-selected={index === highlightedResult}
                      aria-disabled={alreadyFavorite || !addable}
                      onMouseDown={(event) => event.preventDefault()}
                      onMouseEnter={() => setHighlightedResult(index)}
                      onClick={() => addSearchResult(stock)}
                    >
                      <span><strong>{stock.name}</strong><small>{stock.code} · {stock.market}</small></span>
                      <span className={alreadyFavorite ? "already-added" : addable ? "result-add" : "result-readiness"}>
                        {alreadyFavorite
                          ? "已添加"
                          : addable
                            ? <PlusIcon aria-hidden="true" size={14} weight="regular" />
                            : readinessLabel}
                      </span>
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        </section>
      )}
      <div className="market-tabs" role="tablist" aria-label="按市场筛选收藏股票">
        {MARKET_TABS.map((market) => {
          const count = favoriteStocks.filter((item) => item.marketGroup === market).length;
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
              const isActive = selectedCode === item.instrumentId;
              const quote = quotesByInstrumentId[item.instrumentId];
              const itemPrice = isActive && activePrice !== undefined
                ? activePrice
                : quote?.last;
              const itemChange = isActive && activeChange !== undefined
                ? activeChange
                : quote && quote.previousClose > 0
                  ? (quote.last / quote.previousClose - 1) * 100
                  : undefined;
              return (
                <li key={item.instrumentId}>
                  <button
                    type="button"
                    className={`watchlist-select ${isActive ? "active" : ""}`}
                    aria-current={isActive ? "true" : undefined}
                    aria-haspopup="menu"
                    aria-expanded={stockMenu?.instrumentId === item.instrumentId}
                    data-stock-code={item.instrumentId}
                    title="打开股票，右键查看更多操作"
                    onClick={() => { setStockMenu(null); onSelect(item.instrumentId); }}
                    onContextMenu={(event) => handleStockContextMenu(event, item.instrumentId)}
                    onKeyDown={(event) => handleStockMenuKey(event, item.instrumentId)}
                  >
                    <span className="watchlist-security"><strong>{item.name}</strong><small>{item.code}</small></span>
                    {itemPrice !== undefined && itemChange !== undefined ? (
                      <span className="watchlist-quote"><strong>{formatNumber(itemPrice)}</strong><small className={itemChange >= 0 ? "price-up" : "price-down"}>{itemChange >= 0 ? "+" : ""}{formatNumber(itemChange)}%</small></span>
                    ) : (
                      <span className="watchlist-quote watchlist-quote-unavailable"><strong>—</strong><small>暂缺</small></span>
                    )}
                  </button>
                </li>
              );
            })}
          </ul>
        ) : (
          <div className="watchlist-empty">
            <strong>{activeMarket}暂无收藏</strong>
            <p>添加后会显示在这里。</p>
            <button type="button" onClick={() => setManagerOpen(true)}>添加股票</button>
          </div>
        )}
      </div>
      {menuStock && stockMenu && (
        <div
          className="watchlist-context-menu"
          ref={stockMenuRef}
          role="menu"
          tabIndex={-1}
          aria-label={`${menuStock.name}操作`}
          style={{ left: stockMenu.x, top: stockMenu.y }}
          onBlur={(event) => {
            if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setStockMenu(null);
          }}
          onContextMenu={(event) => event.preventDefault()}
          onKeyDown={handleMenuKey}
          onPointerDown={(event) => event.stopPropagation()}
        >
          <span>{menuStock.name}</span>
          <button type="button" role="menuitem" onClick={() => { setStockMenu(null); onSelect(menuStock.instrumentId); }}>打开走势图</button>
          <button className="danger" type="button" role="menuitem" onClick={() => { setStockMenu(null); onRemove(menuStock.instrumentId); }}>从自选删除</button>
        </div>
      )}
    </nav>
  );
}

function LeftRail({
  collapsed,
  favoriteStocks,
  selectedCode,
  quotesByInstrumentId,
  activePrice,
  activeChange,
  onAdd,
  onRemove,
  onToggle,
  onSelect,
}: {
  collapsed: boolean;
  favoriteStocks: StockPreset[];
  selectedCode: string | null;
  quotesByInstrumentId: Readonly<Record<string, RealtimeQuote>>;
  activePrice?: number;
  activeChange?: number;
  onAdd: (stock: StockPreset) => void;
  onRemove: (code: string) => void;
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
        <WatchlistSidebar favoriteStocks={favoriteStocks} selectedCode={selectedCode} quotesByInstrumentId={quotesByInstrumentId} activePrice={activePrice} activeChange={activeChange} onAdd={onAdd} onRemove={onRemove} onSelect={onSelect} />
      </div>
      {collapsed && <span className="collapsed-rail-label" aria-hidden="true">自选股票</span>}
    </aside>
  );
}

function CenterTopbar({
  sourceLabel,
  provider,
  providerStatusAvailable,
  realtimeStatus = "disabled",
  rightRailCollapsed,
  onRightRailToggle,
}: {
  sourceLabel: string;
  provider?: RealtimeProviderInfo;
  providerStatusAvailable: boolean;
  realtimeStatus?: RealtimeConnectionStatus;
  rightRailCollapsed: boolean;
  onRightRailToggle: () => void;
}) {
  const resolvedProvider = provider ?? inferredProvider(null);

  return (
    <header className="topbar" aria-label="行情数据状态">
      <div className="topbar-market-status">
        <div className="topbar-note" data-realtime-status={realtimeStatus} aria-live="polite">
          <span className={`status-dot status-${realtimeStatus}`} aria-hidden="true" />
          {sourceLabel}
        </div>
        <details className="provider-details">
          <summary aria-label={`查看行情来源：${resolvedProvider.displayName}`}>
            <span>{providerLabel(resolvedProvider)}</span>
            <CaretDownIcon aria-hidden="true" size={12} weight="regular" />
          </summary>
          <div className="provider-panel">
            <strong>{resolvedProvider.displayName}</strong>
            <dl>
              <div><dt>连接</dt><dd>{connectionLabel(resolvedProvider, realtimeStatus)}</dd></div>
              <div><dt>账号范围</dt><dd>{accountScopeLabel(resolvedProvider.accountScope)}</dd></div>
              {resolvedProvider.maximumSubscribedSymbols && (
                <div><dt>订阅上限</dt><dd>{resolvedProvider.maximumSubscribedSymbols} 标的</dd></div>
              )}
            </dl>
            <p>{providerNote(resolvedProvider, providerStatusAvailable)}</p>
          </div>
        </details>
      </div>
      {rightRailCollapsed && (
        <button className="rail-toggle topbar-right-rail-toggle" type="button" aria-expanded={false} aria-controls="reserved-right-panel" title="展开右侧栏" onClick={onRightRailToggle}>
          <SidebarSimpleIcon aria-hidden="true" mirrored size={18} weight="regular" />
          <span className="sr-only">展开右侧栏</span>
        </button>
      )}
    </header>
  );
}

function providerLabel(provider: RealtimeProviderInfo): string {
  if (provider.provider === "longbridge") return "长桥 · 个人授权";
  if (provider.provider === "local-csv") return "本地数据";
  return "腾讯 · 开发源";
}

function connectionLabel(
  provider: RealtimeProviderInfo,
  realtimeStatus: RealtimeConnectionStatus,
): string {
  if (provider.provider === "local-csv") return "本地读取";
  if (realtimeStatus === "polling") return "两秒轮询";
  if (["live", "recess", "closed", "stale"].includes(realtimeStatus)) {
    return "WebSocket · 1 秒";
  }
  if (["connecting", "reconnecting"].includes(realtimeStatus)) return "连接中";
  return "暂时不可用";
}

function accountScopeLabel(scope: string): string {
  if (scope === "single_user") return "仅当前用户";
  if (scope === "local_user") return "仅本机";
  return "受控预览";
}

function providerNote(provider: RealtimeProviderInfo, statusAvailable: boolean): string {
  if (provider.provider === "longbridge") {
    return "个人 OAuth 试用，凭证保存在本机长桥 SDK 目录，不与其他用户共享。";
  }
  if (provider.provider === "local-csv") return "导入数据只在当前浏览器会话中使用。";
  if (!statusAvailable) {
    return "实时事件显示为腾讯来源；服务端来源状态接口尚未更新。";
  }
  return "当前是开发回退源，不代表长桥账号已经连接。";
}

function ReservedRightRail({ collapsed, onToggle }: { collapsed: boolean; onToggle: () => void }) {
  return (
    <aside className="right-rail" aria-label="右侧信息面板" hidden={collapsed}>
      <header className="right-rail-topbar">
        <span className="right-rail-title"><strong>信息面板</strong><small>Reserved panel</small></span>
        <button className="rail-toggle" type="button" aria-expanded={true} aria-controls="reserved-right-panel" title="收起右侧栏" onClick={onToggle}>
          <SidebarSimpleIcon aria-hidden="true" mirrored size={18} weight="regular" />
          <span className="sr-only">收起右侧栏</span>
        </button>
      </header>
      <div id="reserved-right-panel" className="right-rail-placeholder"><span>右侧区域已预留</span></div>
    </aside>
  );
}

export default function Home() {
  const [stock, setStock] = useState<StockPreset>(() => STOCKS[0]);
  const [favoriteStocks, setFavoriteStocks] = useState<StockPreset[]>(() => STOCKS);
  const [watchlistSnapshots, setWatchlistSnapshots] = useState<Record<string, RealtimeQuote>>({});
  const [favoritesReady, setFavoritesReady] = useState(false);
  const restoredFavoritesKey = useRef<string | null>(null);
  const loadedServerFavorites = useRef(false);
  const [period, setPeriod] = useState(120);
  const [adjustmentBasis, setAdjustmentBasis] = useState<AdjustmentBasis>("none");
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const [importedData, setImportedData] = useState<Candle[] | null>(null);
  const [importName, setImportName] = useState("");
  const [remoteData, setRemoteData] = useState<Candle[]>([]);
  const [dataMeta, setDataMeta] = useState<DataMeta | null>(null);
  const [loadError, setLoadError] = useState<{ requestKey: string; message: string } | null>(null);
  const [marketDayContinuity, setMarketDayContinuity] = useState<MarketDayContinuity | null>(null);
  const [historicalGapRepairState, setHistoricalGapRepairState] =
    useState<HistoricalGapRepairState | null>(null);
  const [instrumentDayReadiness, setInstrumentDayReadiness] =
    useState<InstrumentDayReadiness | null>(null);
  const [retryNonce, setRetryNonce] = useState(0);
  const [realtimeGraceExpiredKey, setRealtimeGraceExpiredKey] = useState<string | null>(null);
  const [leftRailCollapsed, setLeftRailCollapsed] = useState(false);
  const [rightRailCollapsed, setRightRailCollapsed] = useState(true);

  const dataKey = `${stock.instrumentId}:${adjustmentBasis}`;
  const requestKey = `${dataKey}:${retryNonce}`;

  useEffect(() => {
    const restoreTimer = window.setTimeout(() => {
      try {
        const savedValue = window.localStorage.getItem(FAVORITES_STORAGE_KEY);
        if (savedValue !== null) {
          setFavoriteStocks(savedStockList(JSON.parse(savedValue)));
        } else {
          const previousValue = window.localStorage.getItem(PREVIOUS_FAVORITES_STORAGE_KEY);
          if (previousValue !== null) {
            setFavoriteStocks(savedStockList(JSON.parse(previousValue)));
          } else {
            const legacyValue = window.localStorage.getItem(LEGACY_FAVORITES_STORAGE_KEY);
            if (legacyValue !== null) {
              const legacyCodes: unknown = JSON.parse(legacyValue);
              if (Array.isArray(legacyCodes)) {
                const legacyStocks = STOCKS.filter((item) => legacyCodes.includes(item.code));
                setFavoriteStocks(legacyStocks);
              }
            }
          }
        }
      } catch {
        // Keep the default list when browser storage is unavailable or malformed.
      } finally {
        restoredFavoritesKey.current = FAVORITES_STORAGE_KEY;
        setFavoritesReady(true);
      }
    }, 0);
    return () => window.clearTimeout(restoreTimer);
  }, []);

  useEffect(() => {
    if (!favoritesReady || restoredFavoritesKey.current !== FAVORITES_STORAGE_KEY) return;
    try {
      window.localStorage.setItem(FAVORITES_STORAGE_KEY, JSON.stringify(favoriteStocks));
    } catch {
      // Favorites still work for the current session when storage is unavailable.
    }
  }, [favoriteStocks, favoritesReady]);

  useEffect(() => {
    if (!favoritesReady || loadedServerFavorites.current) return;
    loadedServerFavorites.current = true;
    const controller = new AbortController();
    void fetch("/api/favorites", { cache: "no-store", signal: controller.signal })
      .then(async (response) => response.ok ? response.json() as Promise<unknown> : null)
      .then((payload) => {
        if (controller.signal.aborted || !payload || typeof payload !== "object" || Array.isArray(payload)) return;
        const rows = (payload as Record<string, unknown>).favorites;
        if (!Array.isArray(rows)) return;
        const serverStocks = rows.map(catalogStock).filter((item): item is StockPreset => item !== null);
        if (serverStocks.length === 0) return;
        setFavoriteStocks((current) => Array.from(new Map(
          [...current, ...serverStocks].map((item) => [item.instrumentId, item]),
        ).values()));
      })
      .catch(() => undefined);
    return () => controller.abort();
  }, [favoritesReady]);

  useEffect(() => {
    if (!favoritesReady) return;
    const placeholders = favoriteStocks.filter((item) =>
      isPlaceholderFavoriteName(item.name, item.code)
    );
    if (placeholders.length === 0) return;

    const controller = new AbortController();
    void Promise.all(
      placeholders.map((item) => resolveCanonicalFavorite(item, controller.signal)),
    ).then((resolved) => {
      if (controller.signal.aborted) return;
      const canonicalByCode = new Map(
        resolved.filter((item): item is StockPreset => item !== null)
          .map((item) => [item.code, item]),
      );
      if (canonicalByCode.size === 0) return;

      setFavoriteStocks((current) => {
        let changed = false;
        const next = current.map((item) => {
          const canonical = canonicalByCode.get(item.code);
          if (!canonical || !isPlaceholderFavoriteName(item.name, item.code)) return item;
          changed = true;
          return canonical;
        });
        return changed ? next : current;
      });
      setStock((current) => {
        const canonical = canonicalByCode.get(current.code);
        return canonical && isPlaceholderFavoriteName(current.name, current.code)
          ? canonical
          : current;
      });
    });

    return () => controller.abort();
  }, [favoriteStocks, favoritesReady]);

  useEffect(() => {
    const controller = new AbortController();
    let stopped = false;
    const query = new URLSearchParams({
      instrument_id: stock.instrumentId,
      adjustment_basis: adjustmentBasis,
    });
    const remotePayload = fetchChartPayload(query, controller.signal, "no-store")
      .then(
        (payload) => ({ ok: true as const, payload }),
        (error: unknown) => ({ ok: false as const, error }),
      );
    void (async () => {
      let hasCachedData = false;
      try {
        const cachedPayload = await readChartCache(dataKey);
        if (cachedPayload !== null && !stopped) {
          const cached = parseChartPayload(cachedPayload, requestKey, "browser-cache");
          hasCachedData = true;
          setRemoteData(cached.rows);
          setDataMeta(cached.meta);
          setLoadError(null);
        }
      } catch {
        // IndexedDB is an optional acceleration layer.
      }

      try {
        const remote = await remotePayload;
        if (!remote.ok) throw remote.error;
        const payload = remote.payload;
        if (stopped) return;
        const current = parseChartPayload(payload, requestKey);
        setRemoteData(current.rows);
        setDataMeta(current.meta);
        setLoadError(null);
        void writeChartCache(dataKey, payload).catch(() => undefined);
      } catch (error) {
        if (stopped || (error instanceof Error && error.name === "AbortError")) return;
        if (error instanceof ChartRequestError && error.fallbackBasis) {
          setAdjustmentBasis(error.fallbackBasis);
          setHoverIndex(null);
          return;
        }
        if (hasCachedData) {
          setDataMeta((current) =>
            current?.requestKey === requestKey
              ? { ...current, cacheStatus: "browser-stale" }
              : current
          );
          return;
        }
        setLoadError({
          requestKey,
          message: error instanceof Error ? error.message : "真实行情加载失败",
        });
      }
    })();
    return () => {
      stopped = true;
      controller.abort();
    };
  }, [adjustmentBasis, dataKey, requestKey, stock.instrumentId]);

  const currentRemoteData = useMemo(
    () => dataMeta?.requestKey === requestKey ? remoteData : [],
    [dataMeta?.requestKey, remoteData, requestKey],
  );
  const currentDataMeta = dataMeta?.requestKey === requestKey ? dataMeta : null;
  const chartReady = Boolean(importedData || currentRemoteData.length >= 2);
  const currentError = loadError?.requestKey === requestKey ? loadError.message : "";
  const realtimeProvider = useRealtimeProvider(!importedData && chartReady);
  const realtimeProviderName = realtimeProvider.info?.provider ?? "";
  const selectedRealtimeInstrument = useMemo(
    () => realtimeMarketInstrument(stock),
    [stock],
  );

  useEffect(() => {
    if (
      !favoritesReady ||
      importedData ||
      !chartReady ||
      realtimeProvider.status !== "ready" ||
      !realtimeProviderName
    ) return;
    const instruments = Array.from(new Map(
      favoriteStocks
        .filter((item) =>
          item.instrumentId !== stock.instrumentId &&
          providerSupportsRealtimeInstrument(realtimeProviderName, item)
        )
        .map((item) => [item.instrumentId, realtimeMarketInstrument(item)]),
    ).values());
    if (instruments.length === 0) return;

    const controller = new AbortController();
    let stopped = false;
    let idleHandle: number | null = null;
    let fallbackTimer: number | null = null;
    let refreshTimer: number | null = null;
    const prefetch = async () => {
      const now = new Date();
      const activeInstruments = instruments.filter((instrument) =>
        !realtimeMiddayState(instrument.market, now).paused
      );
      if (activeInstruments.length > 0) {
        await prefetchRealtimeQuotes(activeInstruments, controller.signal).then((quotes) => {
          if (stopped || quotes.length === 0) return;
          setWatchlistSnapshots((current) => {
            const next = { ...current };
            quotes.forEach((quote) => {
              next[quote.instrumentId] = quote;
            });
            return next;
          });
        });
      }
      if (!stopped) {
        const transitionDelays = instruments.flatMap((instrument) => {
          const delay = realtimeMiddayState(instrument.market).transitionInMs;
          return delay === null ? [] : [delay];
        });
        const nextTransition = transitionDelays.length > 0
          ? Math.min(...transitionDelays)
          : Number.POSITIVE_INFINITY;
        const delay = activeInstruments.length > 0
          ? Math.min(WATCHLIST_REFRESH_INTERVAL_MS, nextTransition)
          : nextTransition;
        refreshTimer = window.setTimeout(
          prefetch,
          Math.max(50, Number.isFinite(delay) ? delay + 25 : WATCHLIST_REFRESH_INTERVAL_MS),
        );
      }
    };
    if ("requestIdleCallback" in window) {
      idleHandle = window.requestIdleCallback(() => void prefetch(), { timeout: 1_500 });
    } else {
      fallbackTimer = window.setTimeout(() => void prefetch(), 250);
    }
    return () => {
      stopped = true;
      controller.abort();
      if (idleHandle !== null) window.cancelIdleCallback(idleHandle);
      if (fallbackTimer !== null) window.clearTimeout(fallbackTimer);
      if (refreshTimer !== null) window.clearTimeout(refreshTimer);
    };
  }, [
    chartReady,
    favoriteStocks,
    favoritesReady,
    importedData,
    realtimeProvider.status,
    realtimeProviderName,
    stock.instrumentId,
  ]);

  const realtimeEnabled = !importedData &&
    realtimeProvider.status === "ready" &&
    providerSupportsRealtimeInstrument(realtimeProviderName, stock);
  const realtime = useRealtimeQuote(selectedRealtimeInstrument, realtimeEnabled);
  useEffect(() => {
    if (
      !realtimeEnabled ||
      !chartReady ||
      realtime.quote ||
      realtimeGraceExpiredKey === dataKey
    ) {
      return;
    }
    const timer = window.setTimeout(() => {
      setRealtimeGraceExpiredKey(dataKey);
    }, REALTIME_COHERENCE_GRACE_MS);
    return () => window.clearTimeout(timer);
  }, [chartReady, dataKey, realtime.quote, realtimeEnabled, realtimeGraceExpiredKey]);
  const waitingForCoherentRealtime = realtimeEnabled &&
    chartReady &&
    !realtime.quote &&
    realtimeGraceExpiredKey !== dataKey;
  const quotesByInstrumentId = useMemo(
    () => realtime.quote
      ? { ...watchlistSnapshots, [realtime.quote.instrumentId]: realtime.quote }
      : watchlistSnapshots,
    [realtime.quote, watchlistSnapshots],
  );
  const completedRealtimeDate = realtime.quote && isCompletedRealtimeQuote(realtime.quote)
    ? realtime.quote.tradingDate
    : null;
  const needsHistoricalRevalidation = stock.marketRegion === "cn" &&
    completedQuoteNeedsHistoricalRevalidation(currentRemoteData, realtime.quote);
  const suspectedGapDates = historicalSeriesGapDates(currentRemoteData, realtime.quote);
  const suspectedGapStart = suspectedGapDates[0] ?? "";
  const suspectedGapEnd = suspectedGapDates[suspectedGapDates.length - 1] ?? "";
  const marketDayContinuityKey = stock.marketRegion === "cn" && suspectedGapDates.length > 0
    ? `${suspectedGapStart}:${suspectedGapEnd}`
    : "";
  const matchingMarketDayContinuity = marketDayContinuity?.requestKey === marketDayContinuityKey
    ? marketDayContinuity
    : null;
  const verifiedClosedDates = useMemo(
    () => new Set(
      matchingMarketDayContinuity?.status === "ready"
        ? matchingMarketDayContinuity.closedDates
        : [],
    ),
    [matchingMarketDayContinuity],
  );
  const calendarContinuityChecking = stock.marketRegion === "cn" &&
    suspectedGapDates.length > 0 &&
    matchingMarketDayContinuity?.status !== "ready";
  const historicalGap = suspectedGapDates.length > 0 &&
    (stock.marketRegion !== "cn" || (
      matchingMarketDayContinuity?.status === "ready" &&
      hasHistoricalSeriesGap(currentRemoteData, realtime.quote, verifiedClosedDates)
    ));
  const cachedAShareHistoryRevalidating = stock.marketRegion === "cn" &&
    suspectedGapDates.length > 0 &&
    currentDataMeta?.cacheStatus === "browser-cache";
  const remoteHistoryRevalidating = stock.marketRegion !== "cn" && historicalGap &&
    ["browser-cache", "browser-stale"].includes(currentDataMeta?.cacheStatus ?? "");
  const historicalGapRepairKey = stock.marketRegion !== "cn" && historicalGap &&
    !remoteHistoryRevalidating
    ? `${requestKey}:${suspectedGapStart}:${suspectedGapEnd}:${seriesFingerprint(currentRemoteData)}`
    : "";
  const matchingHistoricalGapRepair = historicalGapRepairState?.requestKey === historicalGapRepairKey
    ? historicalGapRepairState
    : null;
  const historicalGapRepairing = remoteHistoryRevalidating || (
    Boolean(historicalGapRepairKey) && adjustmentBasis !== "hfq" &&
    matchingHistoricalGapRepair?.status !== "failed"
  );
  const visibleHistoricalGap = historicalGap &&
    !historicalGapRepairing &&
    !cachedAShareHistoryRevalidating;
  const waitingForCachedHistoryContinuity = chartReady &&
    ["browser-cache", "browser-stale"].includes(currentDataMeta?.cacheStatus ?? "") &&
    calendarContinuityChecking;
  const historicalRevalidationDate = cachedAShareHistoryRevalidating ||
      stock.marketRegion !== "cn" || calendarContinuityChecking
    ? null
    : historicalGap
      ? suspectedGapEnd || null
      : needsHistoricalRevalidation
        ? realtime.quote?.tradingDate ?? null
        : null;
  const instrumentReadinessKey = historicalRevalidationDate
    ? `${stock.instrumentId}:${historicalRevalidationDate}`
    : "";
  const currentInstrumentDayReadiness = instrumentDayReadiness?.requestKey === instrumentReadinessKey
    ? instrumentDayReadiness
    : null;

  useEffect(() => {
    if (
      importedData ||
      !historicalGapRepairKey ||
      !suspectedGapStart ||
      !suspectedGapEnd ||
      adjustmentBasis === "hfq" ||
      (stock.marketRegion !== "us" && stock.marketRegion !== "hk")
    ) return;
    const controller = new AbortController();
    let active = true;
    const failRepair = () => {
      if (active) {
        setHistoricalGapRepairState({ requestKey: historicalGapRepairKey, status: "failed" });
      }
    };
    const timeout = window.setTimeout(() => {
      controller.abort();
      failRepair();
    }, 15_000);
    void fetch("/api/historical-gap-repair", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        instrument_id: stock.instrumentId,
        market: stock.marketRegion,
        exchange_id: stock.exchangeId,
        provider_symbol: selectedRealtimeInstrument.providerSymbol,
        currency: stock.currency,
        adjustment_basis: adjustmentBasis,
        start_date: suspectedGapStart,
        end_date: suspectedGapEnd,
      }),
      cache: "no-store",
      signal: controller.signal,
    }).then(async (response) => {
      const payload: unknown = await response.json().catch(() => null);
      if (!response.ok || controller.signal.aborted) {
        failRepair();
        return;
      }
      const recovered = parseHistoricalGapRepairRows(
        payload,
        suspectedGapStart,
        suspectedGapEnd,
        adjustmentBasis,
      );
      if (recovered.length === 0 || controller.signal.aborted) {
        failRepair();
        return;
      }
      setRemoteData((current) => mergeHistoricalGapRows(current, recovered));
    }).catch(() => {
      if (!controller.signal.aborted) failRepair();
    }).finally(() => window.clearTimeout(timeout));
    return () => {
      active = false;
      window.clearTimeout(timeout);
      controller.abort();
    };
  }, [
    adjustmentBasis,
    historicalGapRepairKey,
    importedData,
    selectedRealtimeInstrument.providerSymbol,
    stock.currency,
    stock.exchangeId,
    stock.instrumentId,
    stock.marketRegion,
    suspectedGapEnd,
    suspectedGapStart,
  ]);

  useEffect(() => {
    if (importedData || !marketDayContinuityKey || !suspectedGapStart) return;
    const controller = new AbortController();
    void fetchVerifiedClosedMarketDays(
      suspectedGapStart,
      suspectedGapEnd,
      controller.signal,
    ).then(
      (closedDates) => {
        if (!controller.signal.aborted) {
          setMarketDayContinuity({
            requestKey: marketDayContinuityKey,
            status: "ready",
            closedDates,
          });
        }
      },
      () => {
        if (!controller.signal.aborted) {
          setMarketDayContinuity({
            requestKey: marketDayContinuityKey,
            status: "ready",
            closedDates: [],
          });
        }
      },
    );
    return () => controller.abort();
  }, [
    importedData,
    marketDayContinuityKey,
    suspectedGapEnd,
    suspectedGapStart,
  ]);

  useEffect(() => {
    if (importedData || !historicalRevalidationDate) return;
    const controller = new AbortController();
    let stopped = false;
    const query = new URLSearchParams({
      instrument_id: stock.instrumentId,
      adjustment_basis: adjustmentBasis,
    });

    void (async () => {
      for (
        let attempt = 0;
        attempt < MAX_CLOSED_BAR_REVALIDATION_ATTEMPTS && !stopped;
        attempt += 1
      ) {
        try {
          const readiness = await fetchInstrumentDayReadiness(
            historicalRevalidationDate,
            stock.instrumentId,
            controller.signal,
          ).catch(() => null);
          if (readiness && !stopped) {
            setInstrumentDayReadiness(readiness);
          }
          if (readiness && !readiness.chartReady && readiness.readiness !== "closed") {
            await waitForChartRetry(
              instrumentReadinessRetryDelayMs(readiness.readiness, attempt),
              controller.signal,
            );
            continue;
          }
          const payload = await fetchChartPayload(query, controller.signal, "no-store", 1);
          const current = parseChartPayload(payload, requestKey);
          const latestHistoricalDate = current.rows[current.rows.length - 1]?.date ?? "";
          if (latestHistoricalDate < historicalRevalidationDate) {
            await waitForChartRetry(closedBarRevalidationDelayMs(attempt), controller.signal);
            continue;
          }
          if (stopped) return;
          setRemoteData(current.rows);
          setDataMeta(current.meta);
          setLoadError(null);
          void writeChartCache(dataKey, payload).catch(() => undefined);
          return;
        } catch (error) {
          if (stopped || (error instanceof Error && error.name === "AbortError")) return;
          if (error instanceof ChartRequestError && error.fallbackBasis) {
            setAdjustmentBasis(error.fallbackBasis);
            return;
          }
          try {
            await waitForChartRetry(closedBarRevalidationDelayMs(attempt), controller.signal);
          } catch {
            return;
          }
        }
      }
    })();

    return () => {
      stopped = true;
      controller.abort();
    };
  }, [
    adjustmentBasis,
    dataKey,
    historicalRevalidationDate,
    importedData,
    requestKey,
    stock.instrumentId,
  ]);

  const mergedRemoteData = useMemo(
    () => mergeRealtimeCandle(
      currentRemoteData,
      realtime.quote,
      verifiedClosedDates,
      stock.marketRegion !== "cn",
    ),
    [currentRemoteData, realtime.quote, stock.marketRegion, verifiedClosedDates],
  );
  const allData = importedData ?? mergedRemoteData;
  const completedData = importedData ?? selectCompletedSeries(currentRemoteData, mergedRemoteData);
  const allIndicator = useMemo(
    () => alignFormalIndicators(allData, completedData),
    [allData, completedData],
  );

  function selectStock(instrumentId: string) {
    const nextStock = favoriteStocks.find((item) => item.instrumentId === instrumentId) ??
      STOCKS.find((item) => item.instrumentId === instrumentId);
    if (!nextStock) return;
    setStock(nextStock);
    setAdjustmentBasis((current) =>
      nextStock.availableAdjustmentBases.includes(current) ? current : "none"
    );
    setImportedData(null);
    setImportName("");
    setHoverIndex(null);
  }

  function addFavorite(nextStock: StockPreset) {
    setFavoriteStocks((stocks) => {
      if (stocks.some((item) => item.instrumentId === nextStock.instrumentId)) return stocks;
      const nextStocks = [...stocks, nextStock];
      persistFavoriteStocks(nextStocks);
      return nextStocks;
    });
    if (nextStock.marketRegion !== "cn") {
      void fetch("/api/favorites", {
        method: "PUT",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ instrument_id: nextStock.instrumentId }),
      }).catch(() => undefined);
    }
  }

  function removeFavorite(instrumentId: string) {
    setFavoriteStocks((stocks) => {
      const nextStocks = stocks.filter((item) => item.instrumentId !== instrumentId);
      persistFavoriteStocks(nextStocks);
      return nextStocks;
    });
    if (!isSupportedAShareInstrumentId(instrumentId)) {
      void fetch(`/api/favorites?instrument_id=${encodeURIComponent(instrumentId)}`, {
        method: "DELETE",
      }).catch(() => undefined);
    }
  }

  function persistFavoriteStocks(stocks: StockPreset[]) {
    try {
      window.localStorage.setItem(FAVORITES_STORAGE_KEY, JSON.stringify(stocks));
    } catch {
      // Favorites still work for the current session when storage is unavailable.
    }
  }

  function toggleRightRail() {
    setRightRailCollapsed((value) => !value);
  }

  const layoutClassName = [
    "app-layout",
    leftRailCollapsed ? "left-collapsed" : "",
    rightRailCollapsed ? "right-collapsed" : "",
  ].filter(Boolean).join(" ");

  if (
    !importedData &&
    (allData.length < 2 || waitingForCoherentRealtime || waitingForCachedHistoryContinuity)
  ) {
    return (
      <main className="page-shell">
        <div className={layoutClassName}>
          <LeftRail collapsed={leftRailCollapsed} favoriteStocks={favoriteStocks} selectedCode={stock.instrumentId} quotesByInstrumentId={quotesByInstrumentId} onAdd={addFavorite} onRemove={removeFavorite} onToggle={() => setLeftRailCollapsed((value) => !value)} onSelect={selectStock} />
          <div className="content-column">
            <CenterTopbar sourceLabel={stock.marketRegion === "cn" ? "mootdx-cf 真实日线" : `${stock.marketGroup}收盘日线`} provider={realtimeProvider.info ?? inferredProvider(null)} providerStatusAvailable={realtimeProvider.status === "ready"} realtimeStatus={realtimeEnabled ? "connecting" : "disabled"} rightRailCollapsed={rightRailCollapsed} onRightRailToggle={toggleRightRail} />
            <section className="workspace data-state" id="top" aria-live="polite">
              {!currentError ? (
                <><div className="data-state-line" /><div className="data-state-line short" /><p>正在读取已发布的日线快照。</p></>
              ) : (
                <><h2>暂时无法显示真实行情</h2><p>{currentError}</p><button type="button" onClick={() => setRetryNonce((value) => value + 1)}>重新读取</button></>
              )}
            </section>
          </div>
          <ReservedRightRail collapsed={rightRailCollapsed} onToggle={toggleRightRail} />
        </div>
      </main>
    );
  }

  const from = Math.max(0, allData.length - period);
  const data = allData.slice(from);
  const indicator = allIndicator.slice(from);
  const movingAverages = MOVING_AVERAGE_PERIODS.map((movingAveragePeriod) => ({
    period: movingAveragePeriod,
    values: displayMovingAverageValues(allData, movingAveragePeriod).slice(from),
  }));
  const activeIndex = hoverIndex ?? data.length - 1;
  const active = data[activeIndex];
  const activeIndicator = indicator[activeIndex];
  const latest = data[data.length - 1];
  const previous = data[data.length - 2] ?? latest;
  const latestCandleNote = latest.unconfirmed
    ? "盘中均线按实时价更新 · 收盘后固定"
    : latest.realtime && latest.unconfirmed === false
      ? "收盘待历史确认"
      : null;
  const completedDateSet = new Set(completedData.map((item) => item.date));
  const formalIndicatorIndexes = data.flatMap((item, index) =>
    completedDateSet.has(item.date) ? [index] : [],
  );
  const activeIndicatorIsFormal = completedDateSet.has(active.date);
  const activeRealtimeQuote = importedData ? null : realtime.quote;
  const finalizationStatus = importedData
    ? "disabled"
    : calendarContinuityChecking
      ? "checking-calendar"
      : cachedAShareHistoryRevalidating
        ? "revalidating-cache"
        : visibleHistoricalGap
          ? "historical-gap"
          : historicalGapRepairing
            ? "repairing-history"
            : needsHistoricalRevalidation
              ? "awaiting-history"
              : completedRealtimeDate
                ? "historical-confirmed"
                : realtime.quote
                  ? "intraday"
                  : "historical";
  const activeProvider = importedData
    ? realtimeProvider.info ?? inferredProvider("local-csv")
    : realtimeProvider.info ?? inferredProvider(activeRealtimeQuote?.provider ?? null);
  const marketDataStatus = importedData
    ? "本地 CSV"
    : activeRealtimeQuote
      ? realtime.message
      : stock.marketRegion !== "cn" && currentDataMeta
        ? `${currentDataMeta.sourceId === "longbridge" ? "长桥" : currentDataMeta.sourceId} · 收盘日线 · ${stock.currency}`
        : stock.marketRegion === "cn"
          ? realtime.message
          : `${stock.marketGroup}收盘日线`;
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
  const hasRealtimeCandle = Boolean(latest.realtime && latest.unconfirmed);
  const horizontalLayout = chartHorizontalLayout(data.length, plotWidth);
  const xAt = (index: number) => chartXAt(index, LEFT, horizontalLayout);
  const yPrice = (value: number) => TOP + (priceMax - value) / (priceMax - priceMin) * pricePlotHeight;
  const candleWidth = horizontalLayout.candleWidth;
  const currentPriceY = hasRealtimeCandle ? yPrice(latest.close) : null;
  const dateTickIndexes = Array.from(new Set([0, 1, 2, 3, 4, 5].map((step) => Math.round(step * (data.length - 1) / 5))));
  const priceTicks = axisTicks(priceMin, priceMax, 5);
  const yScore = (value: number) => TOP + (100 - value) / 100 * (INDICATOR_HEIGHT - TOP - BOTTOM);

  function onPointerMove(event: PointerEvent<SVGSVGElement>) {
    const box = event.currentTarget.getBoundingClientRect();
    const pointerX = (event.clientX - box.left) / box.width * CHART_WIDTH;
    setHoverIndex(chartDataIndexAt(pointerX, LEFT, data.length, horizontalLayout));
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
          favoriteStocks={favoriteStocks}
          selectedCode={importedData ? null : stock.instrumentId}
          quotesByInstrumentId={quotesByInstrumentId}
          activePrice={importedData ? undefined : displayPrice}
          activeChange={importedData ? undefined : change}
          onAdd={addFavorite}
          onRemove={removeFavorite}
          onToggle={() => setLeftRailCollapsed((value) => !value)}
          onSelect={selectStock}
        />
        <div className="content-column">
          <CenterTopbar
            sourceLabel={marketDataStatus}
            provider={activeProvider}
            providerStatusAvailable={realtimeProvider.status === "ready"}
            realtimeStatus={importedData ? "disabled" : realtime.status}
            rightRailCollapsed={rightRailCollapsed}
            onRightRailToggle={toggleRightRail}
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
        data-cache-status={currentDataMeta?.cacheStatus ?? "unknown"}
        data-chart-generation={currentDataMeta?.generationId ?? ""}
        data-chart-business-date={currentDataMeta?.businessDate ?? ""}
        data-stock-readiness={currentInstrumentDayReadiness?.readiness ?? currentDataMeta?.stockReadiness ?? "unknown"}
        data-market-readiness={currentInstrumentDayReadiness
          ? currentInstrumentDayReadiness.marketComplete ? "complete" : "building"
          : currentDataMeta?.marketReadiness ?? "unknown"}
        data-historical-gap={visibleHistoricalGap ? "true" : "false"}
        data-calendar-continuity={calendarContinuityChecking ? "checking" : visibleHistoricalGap ? "gap" : "verified"}
        data-finalization-status={finalizationStatus}
      >
        <div className="toolbar">
          <div className="security-title">
            <span className="market-tag">{importedData ? "CSV" : stock.market}</span>
            <div><h2>{importedData ? importName.replace(/\.csv$/i, "") : stock.name}</h2><span>{importedData ? "本地导入数据" : stock.code}</span></div>
          </div>
          <div className="price-summary">
            <strong className={change >= 0 ? "price-up" : "price-down"}>{formatNumber(displayPrice)}</strong>
            <span className={change >= 0 ? "price-up" : "price-down"}>{change >= 0 ? "+" : ""}{formatNumber(change)}%</span>
            {!importedData && <small>{stock.currency}</small>}
          </div>
          <div className="toolbar-actions">
            <div className="basis-switch" aria-label="选择复权口径">
              {ADJUSTMENT_BASES.map((item) => {
                const unavailable = Boolean(currentDataMeta) &&
                  !currentDataMeta.availableAdjustmentBases.includes(item.value);
                return <button type="button" disabled={Boolean(importedData) || unavailable} title={unavailable ? "当前市场快照尚未发布该复权口径" : undefined} className={adjustmentBasis === item.value ? "active" : ""} key={item.value} onClick={() => { setAdjustmentBasis(item.value); setHoverIndex(null); }}>{item.label}</button>;
              })}
            </div>
            {!importedData && currentDataMeta && currentDataMeta.availableAdjustmentBases.length < ADJUSTMENT_BASES.length && <span className="adjustment-availability-note" role="status">仅显示当前市场快照已发布的复权口径</span>}
            <div className="period-switch" aria-label="选择图表周期">
              {PERIODS.map((item) => <button className={period === item.value ? "active" : ""} key={item.value} onClick={() => { setPeriod(Math.min(item.value, allData.length)); setHoverIndex(null); }}>{item.label}</button>)}
            </div>
            <label className="upload-button">导入 CSV<input type="file" accept=".csv,text/csv" onChange={handleCsv} /></label>
          </div>
        </div>

        {cachedAShareHistoryRevalidating && realtime.quote && (
          <div className="historical-gap-notice snapshot-refresh-notice" role="status">
            <strong>正在刷新日线快照</strong>
            <span>当前先显示本机缓存，服务器最新快照返回后会自动替换。</span>
          </div>
        )}

        {!cachedAShareHistoryRevalidating && calendarContinuityChecking && realtime.quote && (
          <div className="historical-gap-notice" role="status">
            <strong>正在核对休市日</strong>
            <span>
              历史快照停留在 {currentRemoteData[currentRemoteData.length - 1]?.date}，
              正在核对到 {realtime.quote.tradingDate} 之间的交易日。
            </span>
          </div>
        )}

        {visibleHistoricalGap && realtime.quote && (
          <div
            className="historical-gap-notice"
            role="alert"
            data-stock-readiness={currentInstrumentDayReadiness?.readiness ?? "checking"}
            data-market-readiness={currentInstrumentDayReadiness?.marketComplete ? "complete" : "building"}
          >
            <strong>日线数据存在缺口</strong>
            <span>
              历史快照停留在 {currentRemoteData[currentRemoteData.length - 1]?.date}，
              实时行情已到 {realtime.quote.tradingDate}。
              {stock.marketRegion !== "cn"
                ? "当前实时 K 线已单独显示；缺失的已收盘日线仍需后台补齐。"
                : currentInstrumentDayReadiness?.chartReady
                ? "该股票日线已经发布，正在刷新图表；全市场快照继续在后台完成。"
                : currentInstrumentDayReadiness?.readiness === "processing"
                  ? "该股票日线正在处理，无需等待全市场完成。"
                  : "图表不会跨日拼接，正在自动补齐。"}
            </span>
          </div>
        )}

        <div className="quote-strip" aria-live="polite">
          <span>{active.date}</span><span>开 <b>{formatNumber(active.open)}</b></span><span>高 <b>{formatNumber(active.high)}</b></span><span>低 <b>{formatNumber(active.low)}</b></span><span>收 <b>{formatNumber(active.close)}</b></span><span>量 <b>{formatVolume(active.volume)}</b></span>
          {movingAverages.map(({ period: movingAveragePeriod, values }) => (
            <span className={`ma-label ma-label-${movingAveragePeriod}`} key={movingAveragePeriod}>
              MA{movingAveragePeriod} <b>{values[activeIndex] === null ? "—" : formatNumber(values[activeIndex])}</b>
            </span>
          ))}
        </div>

          <div className="chart-stack">
          <div className="chart-heading">
            <div><h3>每日价格走势</h3>{latestCandleNote && <small className="intraday-note">{latestCandleNote}</small>}</div>
            <div className="legend" aria-label="价格图例">
              <span className="legend-item"><span className="legend-swatch legend-up" />上涨</span>
              <span className="legend-item"><span className="legend-swatch legend-down" />下跌</span>
              {movingAverages.map(({ period: movingAveragePeriod }) => (
                <span className="legend-item" key={movingAveragePeriod}>
                  <span className={`legend-swatch legend-ma-${movingAveragePeriod}`} />
                  MA{movingAveragePeriod}
                </span>
              ))}
            </div>
          </div>

          <svg className="chart price-chart" viewBox={`0 0 ${CHART_WIDTH} ${PRICE_HEIGHT}`} role="img" aria-label={`${stock.name}日线蜡烛图，包含MA5、MA10、MA20、MA30和MA60均线${hasRealtimeCandle ? "，最后一根为盘中实时未完成蜡烛" : ""}`} onPointerMove={onPointerMove} onPointerLeave={() => setHoverIndex(null)}>
            {priceTicks.map((tick) => { const y = yPrice(tick); return <g key={tick}><line className="grid-line" x1={LEFT} x2={CHART_WIDTH - RIGHT} y1={y} y2={y} /><text className="axis-label" x={CHART_WIDTH - RIGHT + 12} y={y + 4}>{formatNumber(tick)}</text></g>; })}
            {dateTickIndexes.map((index) => <g key={index}><line className="grid-line vertical" x1={xAt(index)} x2={xAt(index)} y1={TOP} y2={PRICE_HEIGHT - BOTTOM} /><text className="date-label" x={xAt(index)} y={PRICE_HEIGHT - 7} textAnchor="middle">{formatDate(data[index].date)}</text></g>)}
            {data.map((item, index) => {
              const up = item.close >= item.open;
              const x = xAt(index);
              const bodyTop = yPrice(Math.max(item.open, item.close));
              const bodyBottom = yPrice(Math.min(item.open, item.close));
              const className = ["candle", up ? "up" : "down", item.realtime ? "realtime-candle" : ""].filter(Boolean).join(" ");
              return (
                <g className={className} data-date={item.date} data-open={item.open} data-high={item.high} data-low={item.low} data-close={item.close} data-volume={item.volume} data-unconfirmed={item.unconfirmed ? "true" : "false"} key={item.date}>
                  <line x1={x} x2={x} y1={yPrice(item.high)} y2={yPrice(item.low)} />
                  <rect x={x - candleWidth / 2} y={bodyTop} width={candleWidth} height={Math.max(1.5, bodyBottom - bodyTop)} />
                </g>
              );
            })}
            {movingAverages.map(({ period: movingAveragePeriod, values }) => (
              <polyline
                className={`ma-line ma-line-${movingAveragePeriod}`}
                key={movingAveragePeriod}
                points={values.flatMap((value, index) => value === null ? [] : `${xAt(index)},${yPrice(value)}`).join(" ")}
              />
            ))}
            {currentPriceY !== null && <line className="current-price-line" x1={LEFT} x2={CHART_WIDTH - RIGHT} y1={currentPriceY} y2={currentPriceY} />}
            {hoverIndex !== null && <line className="crosshair" x1={xAt(hoverIndex)} x2={xAt(hoverIndex)} y1={TOP} y2={PRICE_HEIGHT - BOTTOM} />}
            <rect className="pointer-layer" x={LEFT} y={TOP} width={plotWidth} height={pricePlotHeight} />
          </svg>

          <div className="chart-divider" />
          <div className="chart-heading indicator-heading">
            <div><h3>低位承接强度</h3></div>
            <div className="indicator-readout"><span>{activeIndicatorIsFormal ? "当前" : "正式指标"}</span><strong>{activeIndicatorIsFormal ? formatNumber(activeIndicator.score, 1) : "—"}</strong><small>{activeIndicatorIsFormal ? activeIndicator.score >= 60 ? "强承接" : activeIndicator.score >= 30 ? "观察" : "中性" : "待收盘"}</small></div>
          </div>

          <svg className="chart indicator-chart" viewBox={`0 0 ${CHART_WIDTH} ${INDICATOR_HEIGHT}`} role="img" aria-label="低位承接强度指标图" onPointerMove={onPointerMove} onPointerLeave={() => setHoverIndex(null)}>
            {[0, 30, 60, 100].map((tick) => <g key={tick}><line className={`score-grid score-${tick}`} x1={LEFT} x2={CHART_WIDTH - RIGHT} y1={yScore(tick)} y2={yScore(tick)} /><text className="axis-label" x={CHART_WIDTH - RIGHT + 12} y={yScore(tick) + 4}>{tick}</text></g>)}
            {formalIndicatorIndexes.map((index) => { const point = indicator[index]; const y = yScore(point.score); return <rect className={point.score >= 60 ? "score-bar strong" : "score-bar"} key={data[index].date} x={xAt(index) - candleWidth / 2} y={y} width={candleWidth} height={yScore(0) - y} rx={Math.min(2, candleWidth / 3)} />; })}
            <polyline className="score-line" points={formalIndicatorIndexes.map((index) => `${xAt(index)},${yScore(indicator[index].score)}`).join(" ")} />
            {formalIndicatorIndexes.map((index) => { const point = indicator[index]; return point.confirm ? <g className="confirm-marker" key={`confirm-${data[index].date}`}><circle cx={xAt(index)} cy={yScore(Math.min(point.score + 9, 96))} r="5" /><path d={`M ${xAt(index) - 2.5} ${yScore(Math.min(point.score + 9, 96))} l 2 2.5 l 4 -5`} /></g> : null; })}
            {hoverIndex !== null && <line className="crosshair" x1={xAt(hoverIndex)} x2={xAt(hoverIndex)} y1={TOP} y2={INDICATOR_HEIGHT - BOTTOM} />}
            <rect className="pointer-layer" x={LEFT} y={TOP} width={plotWidth} height={INDICATOR_HEIGHT - TOP - BOTTOM} />
          </svg>
        </div>
      </section>

      <section className="factor-card" aria-label="光标日指标拆解">
          <div className="factor-card-head"><div><h2>指标拆解</h2><span>{active.date}</span></div>{activeIndicator.confirm && <span className="confirm-pill">确认信号</span>}</div>
          <div className="factor-grid">
          <div className="factor-row"><span>收盘位置</span><div><i style={{ width: activeIndicatorIsFormal ? `${Math.min(activeIndicator.closePosition * 100, 100)}%` : "0%" }} /></div><b>{activeIndicatorIsFormal ? `${formatNumber(activeIndicator.closePosition * 100, 0)}%` : "—"}</b></div>
          <div className="factor-row"><span>相对量能</span><div><i style={{ width: activeIndicatorIsFormal ? `${Math.min(activeIndicator.volumeRatio / 2 * 100, 100)}%` : "0%" }} /></div><b>{activeIndicatorIsFormal ? `${formatNumber(activeIndicator.volumeRatio, 2)}×` : "—"}</b></div>
          <div className="factor-row"><span>承接强度</span><div><i style={{ width: activeIndicatorIsFormal ? `${activeIndicator.score}%` : "0%" }} /></div><b>{activeIndicatorIsFormal ? formatNumber(activeIndicator.score, 1) : "—"}</b></div>
          </div>
          <p className="signal-note">{active.unconfirmed ? "盘中实时蜡烛仅用于观察，正式指标继续使用已完成日线。" : signalIndex >= 0 ? `窗口内最近一次确认出现在 ${data[signalIndex].date}。` : "当前窗口没有满足全部确认条件的日期。"}</p>
      </section>
        </div>
        <ReservedRightRail collapsed={rightRailCollapsed} onToggle={toggleRightRail} />
      </div>
    </main>
  );
}
