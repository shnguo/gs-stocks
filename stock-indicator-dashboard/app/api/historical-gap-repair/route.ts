import { NextRequest, NextResponse } from "next/server";

import {
  marketDataApiBearerToken,
  marketDataApiIsLoopback,
  marketDataRealtimeApiBaseUrl,
} from "../../lib/market-data-env";
import { isSupportedMarketInstrumentId } from "../../lib/stock-symbol";

type HistoricalGapRepairRequest = {
  instrument_id: string;
  market: "hk" | "us";
  exchange_id: string;
  provider_symbol: string;
  currency: "HKD" | "USD";
  adjustment_basis: "none" | "qfq";
  start_date: string;
  end_date: string;
};

export async function POST(request: NextRequest) {
  const origin = request.headers.get("origin");
  if (origin && origin !== request.nextUrl.origin) {
    return NextResponse.json({ error: "不允许跨站补齐历史行情" }, { status: 403 });
  }
  const input = parseRequest(await request.json().catch(() => null));
  if (!input) {
    return NextResponse.json({ error: "历史行情补齐参数无效" }, { status: 400 });
  }
  const baseUrl = marketDataRealtimeApiBaseUrl();
  const token = marketDataApiBearerToken();
  if (!token && !marketDataApiIsLoopback(baseUrl)) {
    return NextResponse.json(
      { error: "历史行情补齐服务尚未配置访问凭证" },
      { status: 503 },
    );
  }

  try {
    const upstream = await fetch(`${baseUrl}/v1/multimarket/history/repair-gap`, {
      method: "POST",
      headers: {
        accept: "application/json",
        "content-type": "application/json",
        ...(token ? { authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify({
        request_id: `gap-${crypto.randomUUID().replaceAll("-", "")}`,
        ...input,
      }),
      cache: "no-store",
      signal: AbortSignal.timeout(20_000),
    });
    const payload: unknown = await upstream.json().catch(() => null);
    if (!upstream.ok || !validResponse(payload, input)) {
      const upstreamMessage = isRecord(payload) && typeof payload.error === "string"
        ? payload.error
        : null;
      return NextResponse.json(
        {
          error: upstreamMessage ?? "缺失日线暂时无法补齐",
          upstream_status: upstream.status,
        },
        { status: upstream.status === 429 ? 429 : 502 },
      );
    }
    return NextResponse.json(payload, {
      headers: {
        "cache-control": "private, no-store, max-age=0",
        "x-content-type-options": "nosniff",
      },
    });
  } catch (error) {
    const timedOut = error instanceof Error && error.name === "TimeoutError";
    return NextResponse.json(
      { error: timedOut ? "历史行情补齐超时" : "无法连接历史行情补齐服务" },
      { status: 502 },
    );
  }
}

function parseRequest(value: unknown): HistoricalGapRepairRequest | null {
  if (!isRecord(value)) return null;
  const instrumentId = value.instrument_id;
  const market = value.market;
  const exchangeId = value.exchange_id;
  const providerSymbol = value.provider_symbol;
  const currency = value.currency;
  const adjustmentBasis = value.adjustment_basis;
  const startDate = value.start_date;
  const endDate = value.end_date;
  if (
    typeof instrumentId !== "string" ||
    !isSupportedMarketInstrumentId(instrumentId) ||
    (market !== "hk" && market !== "us") ||
    typeof exchangeId !== "string" ||
    !/^(?:xhkg|xarc|xnas|xnys)$/u.test(exchangeId) ||
    typeof providerSymbol !== "string" ||
    !/^[A-Z0-9._-]{2,64}$/u.test(providerSymbol) ||
    (currency !== "HKD" && currency !== "USD") ||
    (market === "hk" && (exchangeId !== "xhkg" || currency !== "HKD")) ||
    (market === "us" && (exchangeId === "xhkg" || currency !== "USD")) ||
    (adjustmentBasis !== "none" && adjustmentBasis !== "qfq") ||
    typeof startDate !== "string" ||
    typeof endDate !== "string" ||
    !validDateRange(startDate, endDate)
  ) return null;
  return {
    instrument_id: instrumentId,
    market,
    exchange_id: exchangeId,
    provider_symbol: providerSymbol,
    currency,
    adjustment_basis: adjustmentBasis,
    start_date: startDate,
    end_date: endDate,
  };
}

function validDateRange(startDate: string, endDate: string): boolean {
  if (!/^\d{4}-\d{2}-\d{2}$/u.test(startDate) || !/^\d{4}-\d{2}-\d{2}$/u.test(endDate)) {
    return false;
  }
  const start = Date.parse(`${startDate}T00:00:00Z`);
  const end = Date.parse(`${endDate}T00:00:00Z`);
  return Number.isFinite(start) && Number.isFinite(end) && end >= start && end - start <= 31 * 86_400_000;
}

function validResponse(value: unknown, input: HistoricalGapRepairRequest): boolean {
  if (!isRecord(value) || value.instrument_id !== input.instrument_id || !isRecord(value.snapshot)) {
    return false;
  }
  const rows = value.snapshot.rows;
  return Array.isArray(rows) && rows.length > 0 && rows.every((row) => {
    if (!isRecord(row)) return false;
    return typeof row.trading_date === "string" &&
      row.trading_date >= input.start_date &&
      row.trading_date <= input.end_date &&
      row.adjustment_basis === input.adjustment_basis;
  });
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
