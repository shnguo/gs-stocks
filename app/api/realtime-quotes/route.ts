import { NextRequest, NextResponse } from "next/server";

import {
  marketDataApiBearerToken,
  marketDataApiIsLoopback,
  marketDataRealtimeApiBaseUrl,
} from "../../lib/market-data-env";
import { isSupportedMarketInstrumentId } from "../../lib/stock-symbol";

const MAX_BATCH_INSTRUMENTS = 500;

type RealtimeInstrumentRequest = {
  instrument_id: string;
  market: "cn" | "hk" | "us";
  provider_symbol: string;
  currency: "CNY" | "HKD" | "USD";
};

export async function POST(request: NextRequest) {
  const origin = request.headers.get("origin");
  if (origin && origin !== request.nextUrl.origin) {
    return NextResponse.json({ error: "不允许跨站读取实时行情" }, { status: 403 });
  }
  const payload: unknown = await request.json().catch(() => null);
  const instruments = parseInstruments(payload);
  if (!instruments) {
    return NextResponse.json(
      { error: `每批需要 1 至 ${MAX_BATCH_INSTRUMENTS} 只有效股票` },
      { status: 400 },
    );
  }

  const baseUrl = marketDataRealtimeApiBaseUrl();
  const token = marketDataApiBearerToken();
  if (!token && !marketDataApiIsLoopback(baseUrl)) {
    return NextResponse.json(
      { error: "实时行情服务尚未配置访问凭证" },
      { status: 503 },
    );
  }
  try {
    const upstream = await fetch(`${baseUrl}/v1/realtime/quotes`, {
      method: "POST",
      headers: {
        accept: "application/json",
        "content-type": "application/json",
        ...(token ? { authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify({ instruments }),
      cache: "no-store",
      signal: AbortSignal.timeout(15_000),
    });
    const result: unknown = await upstream.json().catch(() => null);
    if (
      !upstream.ok ||
      !isRecord(result) ||
      result.event_type !== "quote_snapshot_batch" ||
      !Array.isArray(result.events)
    ) {
      return NextResponse.json(
        { error: "自选行情批量读取暂时不可用", upstream_status: upstream.status },
        { status: upstream.status === 429 ? 429 : 502 },
      );
    }
    return NextResponse.json(result, {
      headers: {
        "cache-control": "private, no-store, max-age=0",
        "x-content-type-options": "nosniff",
      },
    });
  } catch (error) {
    const timedOut = error instanceof Error && error.name === "TimeoutError";
    return NextResponse.json(
      { error: timedOut ? "自选行情批量读取超时" : "无法连接实时行情服务" },
      { status: 502 },
    );
  }
}

function parseInstruments(value: unknown): RealtimeInstrumentRequest[] | null {
  if (!isRecord(value) || !Array.isArray(value.instruments)) return null;
  if (value.instruments.length === 0 || value.instruments.length > MAX_BATCH_INSTRUMENTS) {
    return null;
  }
  const seen = new Set<string>();
  const instruments: RealtimeInstrumentRequest[] = [];
  for (const item of value.instruments) {
    if (!isRecord(item)) return null;
    const instrumentId = item.instrument_id;
    const market = item.market;
    const providerSymbol = item.provider_symbol;
    const currency = item.currency;
    if (
      typeof instrumentId !== "string" ||
      !isSupportedMarketInstrumentId(instrumentId) ||
      seen.has(instrumentId) ||
      (market !== "cn" && market !== "hk" && market !== "us") ||
      typeof providerSymbol !== "string" ||
      !/^[A-Z0-9._-]{2,64}$/u.test(providerSymbol) ||
      (currency !== "CNY" && currency !== "HKD" && currency !== "USD") ||
      (market === "cn" && currency !== "CNY") ||
      (market === "hk" && currency !== "HKD") ||
      (market === "us" && currency !== "USD")
    ) return null;
    seen.add(instrumentId);
    instruments.push({
      instrument_id: instrumentId,
      market,
      provider_symbol: providerSymbol,
      currency,
    });
  }
  return instruments;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
