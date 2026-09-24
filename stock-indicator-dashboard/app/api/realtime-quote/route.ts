import { NextRequest, NextResponse } from "next/server";

import {
  marketDataApiBearerToken,
  marketDataApiIsLoopback,
  marketDataRealtimeApiBaseUrl,
} from "../../lib/market-data-env";
import {
  isSupportedAShareInstrumentId,
  isSupportedMarketInstrumentId,
} from "../../lib/stock-symbol";

export async function GET(request: NextRequest) {
  const instrumentId = request.nextUrl.searchParams.get("instrument_id") ?? "";
  if (!isSupportedMarketInstrumentId(instrumentId)) {
    return NextResponse.json({ error: "该股票尚未开放实时行情" }, { status: 400 });
  }
  const market = request.nextUrl.searchParams.get("market");
  const providerSymbol = request.nextUrl.searchParams.get("provider_symbol");
  const currency = request.nextUrl.searchParams.get("currency");
  const hasMapping = validMapping(market, providerSymbol, currency);
  if (!isSupportedAShareInstrumentId(instrumentId) && !hasMapping) {
    return NextResponse.json({ error: "该股票缺少实时行情代码映射" }, { status: 400 });
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
    const upstreamUrl = new URL(
      `${baseUrl}/v1/realtime/quotes/${encodeURIComponent(instrumentId)}`,
    );
    if (hasMapping) {
      upstreamUrl.searchParams.set("market", market);
      upstreamUrl.searchParams.set("provider_symbol", providerSymbol!);
      upstreamUrl.searchParams.set("currency", currency!);
    }
    const upstream = await fetch(
      upstreamUrl,
      {
        headers: {
          accept: "application/json",
          ...(token ? { authorization: `Bearer ${token}` } : {}),
        },
        cache: "no-store",
        signal: AbortSignal.timeout(10_000),
      },
    );
    const payload: unknown = await upstream.json().catch(() => null);
    if (!upstream.ok || !isRecord(payload) || payload.event_type !== "quote_state") {
      return NextResponse.json(
        { error: "实时行情轮询暂时不可用", upstream_status: upstream.status },
        { status: upstream.status === 429 ? 429 : 502 },
      );
    }
    return NextResponse.json(payload, {
      headers: {
        "cache-control": "private, max-age=2, must-revalidate",
        "x-content-type-options": "nosniff",
      },
    });
  } catch (error) {
    const timedOut = error instanceof Error && error.name === "TimeoutError";
    return NextResponse.json(
      { error: timedOut ? "实时行情轮询超时" : "无法连接实时行情服务" },
      { status: 502 },
    );
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function validMapping(
  market: string | null,
  providerSymbol: string | null,
  currency: string | null,
): market is "cn" | "hk" | "us" {
  if (
    (market !== "cn" && market !== "hk" && market !== "us") ||
    typeof providerSymbol !== "string" ||
    !/^[A-Z0-9._-]{2,64}$/u.test(providerSymbol)
  ) return false;
  return (market === "cn" && currency === "CNY") ||
    (market === "hk" && currency === "HKD") ||
    (market === "us" && currency === "USD");
}
