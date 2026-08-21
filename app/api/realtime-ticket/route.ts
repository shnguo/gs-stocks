import { NextRequest, NextResponse } from "next/server";

import {
  marketDataApiBaseUrl,
  marketDataApiBearerToken,
} from "../../lib/market-data-env";

const ALLOWED_INSTRUMENTS = new Set([
  "cn.xshg.600036",
  "cn.xshg.688008",
  "cn.xshe.000333",
  "cn.xshe.300059",
]);

const SESSION_COOKIE = "mootdx_rt_sid";

export async function POST(request: NextRequest) {
  const origin = request.headers.get("origin");
  if (origin && origin !== request.nextUrl.origin) {
    return NextResponse.json({ error: "不允许跨站申请实时连接" }, { status: 403 });
  }
  const instrumentId = request.nextUrl.searchParams.get("instrument_id") ?? "";
  if (!ALLOWED_INSTRUMENTS.has(instrumentId)) {
    return NextResponse.json({ error: "该股票尚未开放实时行情" }, { status: 400 });
  }
  const token = marketDataApiBearerToken();
  if (!token) {
    return NextResponse.json({ error: "实时行情服务尚未配置访问凭证" }, { status: 503 });
  }
  const existingSessionId = request.cookies.get(SESSION_COOKIE)?.value;
  const sessionId = existingSessionId && /^[0-9a-f-]{36}$/u.test(existingSessionId)
    ? existingSessionId
    : crypto.randomUUID();
  const baseUrl = marketDataApiBaseUrl();
  try {
    const upstream = await fetch(
      `${baseUrl}/v1/realtime/tickets/${encodeURIComponent(instrumentId)}`,
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
