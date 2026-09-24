"use client";

import { useEffect, useState } from "react";
import { selectProgressiveRealtimeQuote } from "./market-series";
import { realtimeMiddayState } from "./lib/realtime-market-schedule";
import type { RealtimeMarketInstrument } from "./lib/stock-symbol";

export type RealtimeConnectionStatus =
  | "disabled"
  | "connecting"
  | "live"
  | "recess"
  | "closed"
  | "stale"
  | "polling"
  | "reconnecting";

export type RealtimeQuote = {
  eventId: string;
  provider: string;
  instrumentId: string;
  marketSession: string;
  tradingDate: string;
  observedAt: string;
  receivedAt: string;
  completeness: string;
  currency: string;
  open: number;
  high: number;
  low: number;
  last: number;
  previousClose: number;
  cumulativeVolume: number;
  cumulativeAmount: number;
  tradingStatus: string;
};

type RealtimeState = {
  status: RealtimeConnectionStatus;
  quote: RealtimeQuote | null;
  message: string;
};

const INITIAL_STATE: RealtimeState = {
  status: "disabled",
  quote: null,
  message: "实时行情未启用",
};
const REALTIME_QUOTE_CACHE_PREFIX = "stock-indicator-dashboard.realtime-quote:v1:";
const REALTIME_QUOTE_CACHE_MAX_AGE_MS = 5 * 60 * 1_000;
const MAX_PREFETCH_CONCURRENCY = 3;
const MAX_BATCH_INSTRUMENTS = 500;
const inflightRealtimeSnapshots = new Map<string, Promise<RealtimeQuote>>();

export async function prefetchRealtimeQuotes(
  instruments: readonly RealtimeMarketInstrument[],
  signal?: AbortSignal,
): Promise<RealtimeQuote[]> {
  const pending = Array.from(new Map(
    instruments.map((instrument) => [instrument.instrumentId, instrument]),
  ).values());
  const snapshots = new Map<string, RealtimeQuote>();
  for (let offset = 0; offset < pending.length; offset += MAX_BATCH_INSTRUMENTS) {
    if (signal?.aborted) break;
    const batch = pending.slice(offset, offset + MAX_BATCH_INSTRUMENTS);
    const quotes = await requestRealtimeBatch(batch, signal).catch(() =>
      requestRealtimeFallback(batch, signal)
    );
    quotes.forEach((quote) => {
      snapshots.set(quote.instrumentId, quote);
      writeCachedRealtimeQuote(quote);
    });
  }
  return pending.flatMap((instrument) => {
    const quote = snapshots.get(instrument.instrumentId) ??
      readCachedRealtimeQuote(instrument.instrumentId);
    return quote ? [quote] : [];
  });
}

export function useRealtimeQuote(
  instrument: RealtimeMarketInstrument,
  enabled: boolean,
): RealtimeState {
  const [state, setState] = useState<RealtimeState>(INITIAL_STATE);
  const instrumentId = instrument.instrumentId;

  useEffect(() => {
    if (!enabled) {
      setState(INITIAL_STATE);
      return;
    }
    let stopped = false;
    let socket: WebSocket | null = null;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;
    let pollingTimer: ReturnType<typeof setTimeout> | null = null;
    let marketTransitionTimer: ReturnType<typeof setTimeout> | null = null;
    let ticketController: AbortController | null = null;
    let quoteController: AbortController | null = null;
    let retryAttempt = 0;
    let pausedForMidday = false;

    function scheduleMarketTransition() {
      if (stopped) return;
      if (marketTransitionTimer) clearTimeout(marketTransitionTimer);
      const midday = realtimeMiddayState(instrument.market);
      if (midday.transitionInMs === null) return;
      marketTransitionTimer = setTimeout(() => {
        marketTransitionTimer = null;
        if (realtimeMiddayState(instrument.market).paused) {
          enterMiddayPause();
        } else {
          resumeAfterMiddayPause();
        }
      }, Math.max(50, midday.transitionInMs + 25));
    }

    function enterMiddayPause() {
      if (stopped) return;
      pausedForMidday = true;
      ticketController?.abort();
      ticketController = null;
      quoteController?.abort();
      quoteController = null;
      if (retryTimer) clearTimeout(retryTimer);
      retryTimer = null;
      if (pollingTimer) clearTimeout(pollingTimer);
      pollingTimer = null;
      const activeSocket = socket;
      socket = null;
      activeSocket?.close(1000, "midday recess");
      setState((current) => ({
        ...current,
        status: "recess",
        message: "午间休市 · 13:00 自动恢复",
      }));
      scheduleMarketTransition();
    }

    function resumeAfterMiddayPause() {
      if (stopped) return;
      pausedForMidday = false;
      scheduleMarketTransition();
      setState((current) => ({
        ...current,
        status: "connecting",
        message: "午休结束，正在恢复实时行情",
      }));
      void primeSnapshot();
      void connect();
    }

    const scheduleReconnect = () => {
      if (stopped || retryTimer) return;
      if (realtimeMiddayState(instrument.market).paused) {
        enterMiddayPause();
        return;
      }
      retryAttempt += 1;
      const ceiling = Math.min(30_000, 500 * 2 ** Math.min(retryAttempt, 6));
      const delay = Math.round(ceiling * (0.5 + secureRandomFraction() * 0.5));
      setState((current) => ({
        ...current,
        status: "reconnecting",
        message: `连接中断，${Math.max(1, Math.ceil(delay / 1_000))} 秒后重连`,
      }));
      retryTimer = setTimeout(() => {
        retryTimer = null;
        void connect();
      }, delay);
    };

    const schedulePoll = () => {
      if (stopped || pollingTimer) return;
      pollingTimer = setTimeout(() => {
        pollingTimer = null;
        void pollOnce();
      }, 2_000);
    };

    const fetchQuoteSnapshot = async (): Promise<RealtimeQuote> => {
      quoteController?.abort();
      const controller = new AbortController();
      quoteController = controller;
      try {
        const response = await fetch(
          realtimeQuoteRequestUrl(instrument),
          { cache: "no-store", signal: controller.signal },
        );
        const payload: unknown = await response.json().catch(() => null);
        const quote = response.ok ? quoteFromStreamMessage(payload, instrumentId) : null;
        if (!quote) {
          throw new Error(
            isRecord(payload) && typeof payload.error === "string"
              ? payload.error
              : "实时行情轮询失败",
          );
        }
        return quote;
      } finally {
        if (quoteController === controller) quoteController = null;
      }
    };

    const primeSnapshot = async () => {
      if (stopped) return;
      try {
        const quote = await requestRealtimeSnapshot(instrument);
        if (stopped) return;
        setState((current) => {
          const selected = selectProgressiveRealtimeQuote(current.quote, quote);
          return selected === current.quote
            ? current
            : {
              status: statusForQuote(selected),
              quote: selected,
              message: labelForQuote(selected),
            };
        });
      } catch (error) {
        if (stopped || (error instanceof Error && error.name === "AbortError")) return;
        // The WebSocket connection remains authoritative when the parallel snapshot misses.
      }
    };

    const pollOnce = async () => {
      if (stopped) return;
      if (realtimeMiddayState(instrument.market).paused) {
        enterMiddayPause();
        return;
      }
      try {
        const quote = await fetchQuoteSnapshot();
        if (stopped) return;
        retryAttempt = 0;
        writeCachedRealtimeQuote(quote);
        setState((current) => {
          const selected = selectProgressiveRealtimeQuote(current.quote, quote);
          return selected === current.quote
            ? current
            : {
              status: selected.completeness === "stale" ? "stale" : "polling",
              quote: selected,
              message: pollingLabelForQuote(selected),
            };
        });
        schedulePoll();
      } catch (error) {
        if (stopped || (error instanceof Error && error.name === "AbortError")) return;
        setState((current) => ({
          ...current,
          status: "stale",
          message: error instanceof Error ? error.message : "实时行情轮询失败",
        }));
        scheduleReconnect();
      }
    };

    const startPolling = () => {
      if (stopped) return;
      if (realtimeMiddayState(instrument.market).paused) {
        enterMiddayPause();
        return;
      }
      setState((current) => ({
        ...current,
        status: "polling",
        message: "实时连接降级为两秒轮询",
      }));
      void pollOnce();
    };

    async function connect() {
      if (stopped) return;
      if (realtimeMiddayState(instrument.market).paused) {
        enterMiddayPause();
        return;
      }
      setState((current) => ({
        ...current,
        status: retryAttempt === 0 ? "connecting" : "reconnecting",
        message: retryAttempt === 0 ? "正在连接实时行情" : "正在重新连接实时行情",
      }));
      ticketController = new AbortController();
      try {
        const response = await fetch(
          `/api/realtime-ticket?${new URLSearchParams({
            instrument_id: instrumentId,
            market: instrument.market,
            provider_symbol: instrument.providerSymbol,
            currency: instrument.currency,
          })}`,
          {
            method: "POST",
            cache: "no-store",
            signal: ticketController.signal,
          },
        );
        const payload: unknown = await response.json().catch(() => null);
        if (!response.ok || !isRecord(payload) || typeof payload.websocket_url !== "string") {
          throw new Error(
            isRecord(payload) && typeof payload.error === "string"
              ? payload.error
              : "实时连接凭证申请失败",
          );
        }
        if (stopped) return;
        const connectedSocket = new WebSocket(payload.websocket_url);
        socket = connectedSocket;
        connectedSocket.addEventListener("open", () => {
          if (stopped || socket !== connectedSocket) return;
          connectedSocket.send(JSON.stringify({
            contract_version: 1,
            type: "subscribe",
            request_id: crypto.randomUUID(),
            instrument_ids: [instrumentId],
          }));
        });
        connectedSocket.addEventListener("message", (event) => {
          if (stopped || typeof event.data !== "string") return;
          const message = parseJson(event.data);
          const quote = quoteFromStreamMessage(message, instrumentId);
          if (quote) {
            retryAttempt = 0;
            writeCachedRealtimeQuote(quote);
            setState((current) => {
              const selected = selectProgressiveRealtimeQuote(current.quote, quote);
              return selected === current.quote
                ? current
                : {
                  status: statusForQuote(selected),
                  quote: selected,
                  message: labelForQuote(selected),
                };
            });
            return;
          }
          if (isRecord(message) && message.type === "error") {
            setState((current) => ({
              ...current,
              status: "stale",
              message: typeof message.message === "string" ? message.message : "实时行情暂时延迟",
            }));
          }
        });
        connectedSocket.addEventListener("close", () => {
          if (socket === connectedSocket) socket = null;
          if (pausedForMidday || realtimeMiddayState(instrument.market).paused) {
            enterMiddayPause();
            return;
          }
          startPolling();
        });
        connectedSocket.addEventListener("error", () => {
          setState((current) => ({
            ...current,
            status: "stale",
            message: "实时连接发生错误",
          }));
        });
      } catch (error) {
        if (stopped || (error instanceof Error && error.name === "AbortError")) return;
        setState((current) => ({
          ...current,
          status: "stale",
          message: error instanceof Error ? error.message : "实时行情连接失败",
        }));
        startPolling();
      }
    }

    const cachedQuote = readCachedRealtimeQuote(instrumentId);
    setState(cachedQuote
      ? {
        status: "stale",
        quote: cachedQuote,
        message: `缓存行情 · ${cachedQuote.observedAt.slice(11, 19)} UTC`,
      }
      : { status: "connecting", quote: null, message: "正在连接实时行情" });
    if (realtimeMiddayState(instrument.market).paused) {
      enterMiddayPause();
    } else {
      scheduleMarketTransition();
      void primeSnapshot();
      void connect();
    }
    return () => {
      stopped = true;
      ticketController?.abort();
      quoteController?.abort();
      if (retryTimer) clearTimeout(retryTimer);
      if (pollingTimer) clearTimeout(pollingTimer);
      if (marketTransitionTimer) clearTimeout(marketTransitionTimer);
      socket?.close(1000, "instrument changed");
    };
  }, [
    enabled,
    instrument,
    instrument.currency,
    instrumentId,
    instrument.instrumentId,
    instrument.market,
    instrument.providerSymbol,
  ]);

  return state;
}

async function requestRealtimeBatch(
  instruments: readonly RealtimeMarketInstrument[],
  signal?: AbortSignal,
): Promise<RealtimeQuote[]> {
  const response = await fetch("/api/realtime-quotes", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      instruments: instruments.map((instrument) => ({
        instrument_id: instrument.instrumentId,
        market: instrument.market,
        provider_symbol: instrument.providerSymbol,
        currency: instrument.currency,
      })),
    }),
    cache: "no-store",
    signal,
  });
  const payload: unknown = await response.json().catch(() => null);
  if (
    !response.ok ||
    !isRecord(payload) ||
    payload.event_type !== "quote_snapshot_batch" ||
    !Array.isArray(payload.events)
  ) {
    throw new Error(
      isRecord(payload) && typeof payload.error === "string"
        ? payload.error
        : "自选行情批量读取失败",
    );
  }
  const requested = new Set(instruments.map((instrument) => instrument.instrumentId));
  return payload.events.flatMap((event) => {
    if (!isRecord(event) || typeof event.instrument_id !== "string") return [];
    if (!requested.has(event.instrument_id)) return [];
    const quote = quoteFromStreamMessage(event, event.instrument_id);
    return quote ? [quote] : [];
  });
}

async function requestRealtimeFallback(
  instruments: readonly RealtimeMarketInstrument[],
  signal?: AbortSignal,
): Promise<RealtimeQuote[]> {
  const snapshots = new Map<string, RealtimeQuote>();
  let cursor = 0;
  const workers = Array.from(
    { length: Math.min(MAX_PREFETCH_CONCURRENCY, instruments.length) },
    async () => {
      while (cursor < instruments.length) {
        if (signal?.aborted) return;
        const instrument = instruments[cursor];
        cursor += 1;
        if (!instrument) continue;
        const quote = await requestRealtimeSnapshot(instrument, signal).catch(() => null);
        if (quote) snapshots.set(instrument.instrumentId, quote);
      }
    },
  );
  await Promise.all(workers);
  return instruments.flatMap((instrument) => {
    const quote = snapshots.get(instrument.instrumentId);
    return quote ? [quote] : [];
  });
}

function realtimeQuoteRequestUrl(instrument: RealtimeMarketInstrument): string {
  const query = new URLSearchParams({
    instrument_id: instrument.instrumentId,
    market: instrument.market,
    provider_symbol: instrument.providerSymbol,
    currency: instrument.currency,
  });
  return `/api/realtime-quote?${query}`;
}

function requestRealtimeSnapshot(
  instrument: RealtimeMarketInstrument,
  signal?: AbortSignal,
): Promise<RealtimeQuote> {
  const instrumentId = instrument.instrumentId;
  const existing = inflightRealtimeSnapshots.get(instrumentId);
  if (existing) return existing;
  const request = fetch(
    realtimeQuoteRequestUrl(instrument),
    { cache: "default", signal },
  ).then(async (response) => {
    const payload: unknown = await response.json().catch(() => null);
    const quote = response.ok ? quoteFromStreamMessage(payload, instrumentId) : null;
    if (!quote) {
      throw new Error(
        isRecord(payload) && typeof payload.error === "string"
          ? payload.error
          : "实时行情快照失败",
      );
    }
    writeCachedRealtimeQuote(quote);
    return quote;
  }).finally(() => {
    if (inflightRealtimeSnapshots.get(instrumentId) === request) {
      inflightRealtimeSnapshots.delete(instrumentId);
    }
  });
  inflightRealtimeSnapshots.set(instrumentId, request);
  return request;
}

function writeCachedRealtimeQuote(quote: RealtimeQuote): void {
  try {
    const cacheKey = `${REALTIME_QUOTE_CACHE_PREFIX}${quote.instrumentId}`;
    const serialized = window.sessionStorage.getItem(cacheKey);
    if (serialized) {
      const cached: unknown = JSON.parse(serialized);
      if (isRecord(cached) && isCachedRealtimeQuote(cached.quote, quote.instrumentId)) {
        const selected = selectProgressiveRealtimeQuote(cached.quote, quote);
        if (selected === cached.quote) return;
      }
    }
    window.sessionStorage.setItem(
      cacheKey,
      JSON.stringify({ cachedAt: Date.now(), quote }),
    );
  } catch {
    // Realtime rendering remains available when session storage is blocked.
  }
}

function readCachedRealtimeQuote(instrumentId: string): RealtimeQuote | null {
  try {
    const serialized = window.sessionStorage.getItem(
      `${REALTIME_QUOTE_CACHE_PREFIX}${instrumentId}`,
    );
    if (!serialized) return null;
    const cached: unknown = JSON.parse(serialized);
    if (!isRecord(cached) || typeof cached.cachedAt !== "number") return null;
    if (Date.now() - cached.cachedAt > REALTIME_QUOTE_CACHE_MAX_AGE_MS) return null;
    const quote = cached.quote;
    if (!isCachedRealtimeQuote(quote, instrumentId)) return null;
    return { ...quote, completeness: "stale" };
  } catch {
    return null;
  }
}

function isCachedRealtimeQuote(value: unknown, instrumentId: string): value is RealtimeQuote {
  if (!isRecord(value) || value.instrumentId !== instrumentId) return false;
  const stringFields = [
    "eventId",
    "provider",
    "marketSession",
    "tradingDate",
    "observedAt",
    "receivedAt",
    "completeness",
    "currency",
    "tradingStatus",
  ];
  const numberFields = [
    "open",
    "high",
    "low",
    "last",
    "previousClose",
    "cumulativeVolume",
    "cumulativeAmount",
  ];
  return stringFields.every((field) => typeof value[field] === "string") &&
    numberFields.every((field) =>
      typeof value[field] === "number" && Number.isFinite(value[field]) && value[field] >= 0
    );
}

function quoteFromStreamMessage(value: unknown, instrumentId: string): RealtimeQuote | null {
  if (!isRecord(value)) return null;
  let event: unknown = null;
  if (value.event_type === "quote_state") event = value;
  if (value.type === "update") event = value.event;
  if (value.type === "snapshot" && Array.isArray(value.events)) event = value.events[0];
  if (!isRecord(event) || event.event_type !== "quote_state" || event.instrument_id !== instrumentId) {
    return null;
  }
  const payload = event.payload;
  if (
    !isRecord(payload) ||
    payload.kind !== "quote_state" ||
    typeof event.event_id !== "string" ||
    typeof event.provider !== "string" ||
    typeof event.market_session !== "string" ||
    typeof event.trading_date !== "string" ||
    typeof event.observed_at !== "string" ||
    typeof event.received_at !== "string" ||
    typeof event.completeness !== "string" ||
    typeof payload.currency !== "string" ||
    typeof payload.trading_status !== "string"
  ) {
    return null;
  }
  const numbers = [
    payload.open,
    payload.high,
    payload.low,
    payload.last,
    payload.previous_close,
    payload.cumulative_volume,
    payload.cumulative_amount,
  ].map(Number);
  if (numbers.some((number) => !Number.isFinite(number) || number < 0)) return null;
  const [open, high, low, last, previousClose, cumulativeVolume, cumulativeAmount] = numbers;
  if (
    open === undefined ||
    high === undefined ||
    low === undefined ||
    last === undefined ||
    previousClose === undefined ||
    cumulativeVolume === undefined ||
    cumulativeAmount === undefined
  ) {
    return null;
  }
  return {
    eventId: event.event_id,
    provider: event.provider,
    instrumentId,
    marketSession: event.market_session,
    tradingDate: event.trading_date,
    observedAt: event.observed_at,
    receivedAt: event.received_at,
    completeness: event.completeness,
    currency: payload.currency,
    open,
    high,
    low,
    last,
    previousClose,
    cumulativeVolume,
    cumulativeAmount,
    tradingStatus: payload.trading_status,
  };
}

function pollingLabelForQuote(quote: RealtimeQuote): string {
  const time = quote.observedAt.slice(11, 19);
  if (quote.completeness === "stale") return `轮询行情延迟 · ${time} UTC`;
  if (quote.marketSession === "pre_market") return `盘前两秒轮询 · ${time} UTC`;
  if (quote.marketSession === "after_hours") return `盘后两秒轮询 · ${time} UTC`;
  if (quote.marketSession === "overnight") return `夜盘两秒轮询 · ${time} UTC`;
  return `两秒轮询 · ${time} UTC`;
}

function statusForQuote(quote: RealtimeQuote): RealtimeConnectionStatus {
  if (quote.completeness === "stale") return "stale";
  if (quote.marketSession === "recess") return "recess";
  if (["pre_open", "closed"].includes(quote.marketSession)) return "closed";
  return "live";
}

function labelForQuote(quote: RealtimeQuote): string {
  const time = quote.observedAt.slice(11, 19);
  if (quote.completeness === "stale") return `行情延迟 · ${time} UTC`;
  if (quote.marketSession === "recess") return `午间休市 · ${time} UTC`;
  if (quote.marketSession === "pre_open") return `开盘前 · ${time} UTC`;
  if (quote.marketSession === "pre_market") return `盘前一秒推送 · ${time} UTC`;
  if (quote.marketSession === "after_hours") return `盘后一秒推送 · ${time} UTC`;
  if (quote.marketSession === "overnight") return `夜盘一秒推送 · ${time} UTC`;
  if (quote.marketSession === "closed") return `已收盘 · ${time} UTC`;
  if (["opening_auction", "closing_auction"].includes(quote.marketSession)) {
    return `集合竞价一秒推送 · ${time} UTC`;
  }
  return `一秒推送 · ${time} UTC`;
}

function parseJson(value: string): unknown {
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}

function secureRandomFraction(): number {
  const value = new Uint32Array(1);
  crypto.getRandomValues(value);
  return (value[0] ?? 0) / 0xffff_ffff;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
