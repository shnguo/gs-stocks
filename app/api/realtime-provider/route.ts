import { NextResponse } from "next/server";

import {
  marketDataApiBearerToken,
  marketDataApiIsLoopback,
  marketDataRealtimeApiBaseUrl,
} from "../../lib/market-data-env";

export async function GET() {
  const baseUrl = marketDataRealtimeApiBaseUrl();
  const token = marketDataApiBearerToken();
  if (!token && !marketDataApiIsLoopback(baseUrl)) {
    return NextResponse.json(
      { error: "实时行情服务尚未配置访问凭证" },
      { status: 503 },
    );
  }
  try {
    const upstream = await fetch(`${baseUrl}/v1/realtime/provider`, {
      headers: {
        accept: "application/json",
        ...(token ? { authorization: `Bearer ${token}` } : {}),
      },
      cache: "no-store",
      signal: AbortSignal.timeout(5_000),
    });
    const payload: unknown = await upstream.json().catch(() => null);
    if (!upstream.ok || !isRecord(payload) || typeof payload.provider !== "string") {
      return NextResponse.json(
        { error: "行情来源状态暂时不可用", upstream_status: upstream.status },
        { status: 502 },
      );
    }
    return NextResponse.json(payload, {
      headers: {
        "cache-control": "private, no-store, max-age=0",
        "x-content-type-options": "nosniff",
      },
    });
  } catch {
    return NextResponse.json({ error: "无法连接行情来源状态服务" }, { status: 502 });
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
