"use client";

import { useEffect, useState } from "react";

export type RealtimeConnectionStatus =
  | "disabled"
  | "connecting"
  | "live"
  | "recess"
  | "closed"
  | "stale"
  | "reconnecting";

export type RealtimeQuote = {
  eventId: string;
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

export function useRealtimeQuote(instrumentId: string, enabled: boolean): RealtimeState {
  const [state, setState] = useState<RealtimeState>(INITIAL_STATE);

  useEffect(() => {
    if (!enabled) {
      setState(INITIAL_STATE);
      return;
    }
    let stopped = false;
    let socket: WebSocket | null = null;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;
    let ticketController: AbortController | null = null;
    let retryAttempt = 0;

    const scheduleReconnect = () => {
      if (stopped || retryTimer) return;
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

    const connect = async () => {
      if (stopped) return;
      setState((current) => ({
        ...current,
        status: retryAttempt === 0 ? "connecting" : "reconnecting",
        message: retryAttempt === 0 ? "正在连接实时行情" : "正在重新连接实时行情",
      }));
      ticketController = new AbortController();
      try {
        const response = await fetch(
          `/api/realtime-ticket?instrument_id=${encodeURIComponent(instrumentId)}`,
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
        socket = new WebSocket(payload.websocket_url);
        socket.addEventListener("open", () => {
          if (stopped || !socket) return;
          socket.send(JSON.stringify({
            contract_version: 1,
            type: "subscribe",
            request_id: crypto.randomUUID(),
            instrument_ids: [instrumentId],
          }));
        });
        socket.addEventListener("message", (event) => {
          if (stopped || typeof event.data !== "string") return;
          const message = parseJson(event.data);
          const quote = quoteFromStreamMessage(message, instrumentId);
          if (quote) {
            retryAttempt = 0;
            setState({
              status: statusForQuote(quote),
              quote,
              message: labelForQuote(quote),
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
        socket.addEventListener("close", () => {
          socket = null;
          scheduleReconnect();
        });
        socket.addEventListener("error", () => {
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
        scheduleReconnect();
      }
    };

    setState({ status: "connecting", quote: null, message: "正在连接实时行情" });
    void connect();
    return () => {
      stopped = true;
      ticketController?.abort();
      if (retryTimer) clearTimeout(retryTimer);
      socket?.close(1000, "instrument changed");
    };
  }, [enabled, instrumentId]);

  return state;
}

function quoteFromStreamMessage(value: unknown, instrumentId: string): RealtimeQuote | null {
  if (!isRecord(value)) return null;
  let event: unknown = null;
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

function statusForQuote(quote: RealtimeQuote): RealtimeConnectionStatus {
  if (quote.completeness === "stale") return "stale";
  if (quote.marketSession === "recess") return "recess";
  if (["pre_open", "after_hours", "closed"].includes(quote.marketSession)) return "closed";
  return "live";
}

function labelForQuote(quote: RealtimeQuote): string {
  const time = quote.observedAt.slice(11, 19);
  if (quote.completeness === "stale") return `行情延迟 · ${time} UTC`;
  if (quote.marketSession === "recess") return `午间休市 · ${time} UTC`;
  if (quote.marketSession === "pre_open") return `开盘前 · ${time} UTC`;
  if (["after_hours", "closed"].includes(quote.marketSession)) return `已收盘 · ${time} UTC`;
  if (["opening_auction", "closing_auction"].includes(quote.marketSession)) {
    return `集合竞价实时行情 · ${time} UTC`;
  }
  return `实时行情 · ${time} UTC`;
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
