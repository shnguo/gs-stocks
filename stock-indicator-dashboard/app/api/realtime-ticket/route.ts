import { NextRequest, NextResponse } from "next/server";

import {
  marketDataApiBearerToken,
  marketDataApiIsLoopback,
  marketDataRealtimeApiBaseUrl,
  marketDataRealtimeTransport,
} from "../../lib/market-data-env";
import { isSupportedMarketInstrumentId } from "../../lib/stock-symbol";

const SESSION_COOKIE = "mootdx_rt_sid";

export async function POST(request: NextRequest) {
  const origin = request.headers.get("origin");
  if (origin && origin !== request.nextUrl.origin) {
    return NextResponse.json({ error: "不允许跨站申请实时连接" }, { status: 403 });
  }
  const instrumentId = request.nextUrl.searchParams.get("instrument_id") ?? "";
  if (!isSupportedMarketInstrumentId(instrumentId)) {
    return NextResponse.json({ error: "该股票尚未开放实时行情" }, { status: 400 });
  }
  const baseUrl = marketDataRealtimeApiBaseUrl();
  const token = marketDataApiBearerToken();
  const market = request.nextUrl.searchParams.get("market");
  const providerSymbol = request.nextUrl.searchParams.get("provider_symbol");
  const currency = request.nextUrl.searchParams.get("currency");
  if (!validMapping(market, providerSymbol, currency)) {
    return NextResponse.json({ error: "该股票缺少实时行情代码映射" }, { status: 400 });
  }
  if (marketDataRealtimeTransport() === "polling") {
    return NextResponse.json(
      { error: "本地 Rust 服务使用轮询模式", fallback: "polling" },
      { status: 409 },
    );
  }
  if (!token && !marketDataApiIsLoopback(baseUrl)) {
    return NextResponse.json({ error: "实时行情服务尚未配置访问凭证" }, { status: 503 });
  }
  const existingSessionId = request.cookies.get(SESSION_COOKIE)?.value;
  const sessionId = existingSessionId && /^[0-9a-f-]{36}$/u.test(existingSessionId)
    ? existingSessionId
    : crypto.randomUUID();
  try {
    const upstreamUrl = new URL(
      `${baseUrl}/v1/realtime/tickets/${encodeURIComponent(instrumentId)}`,
    );
    upstreamUrl.searchParams.set("market", market);
    upstreamUrl.searchParams.set("provider_symbol", providerSymbol);
    upstreamUrl.searchParams.set("currency", currency);
    const upstream = await fetch(
      upstreamUrl,
      {
        method: "POST",
        headers: {
          accept: "application/json",
          authorization: `Bearer ${token}`,
          "x-mootdx-realtime-subject": `anonymous:${sessionId}`,
        },
        signal: AbortSignal.timeout(10_000),
      },
    );
    const payload: unknown = await upstream.json().catch(() => null);
    if (!upstream.ok || !isRecord(payload) || typeof payload.websocket_url !== "string") {
      return NextResponse.json(
        { error: "暂时无法建立实时行情连接", upstream_status: upstream.status },
        { status: upstream.status === 429 ? 429 : 502 },
      );
    }
    const websocketUrl = new URL(payload.websocket_url);
    const expectedHost = new URL(baseUrl).host;
    if (
      !["ws:", "wss:"].includes(websocketUrl.protocol) ||
      websocketUrl.host !== expectedHost ||
      !websocketUrl.pathname.startsWith("/v1/realtime/connect/")
    ) {
      return NextResponse.json({ error: "实时行情连接地址无效" }, { status: 502 });
    }
    const response = NextResponse.json(
      {
        contract_version: 1,
        instrument_id: instrumentId,
        expires_at: typeof payload.expires_at === "string" ? payload.expires_at : null,
        websocket_url: websocketUrl.toString(),
      },
      {
        headers: {
          "cache-control": "private, no-store, max-age=0",
          "referrer-policy": "no-referrer",
          "x-content-type-options": "nosniff",
        },
      },
    );
    if (!existingSessionId) {
      response.cookies.set(SESSION_COOKIE, sessionId, {
        httpOnly: true,
        sameSite: "strict",
        secure: request.nextUrl.protocol === "https:",
        path: "/",
        maxAge: 31_536_000,
      });
    }
    return response;
  } catch (error) {
    const timedOut = error instanceof Error && error.name === "TimeoutError";
    return NextResponse.json(
      { error: timedOut ? "实时行情连接请求超时" : "无法连接实时行情服务" },
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
