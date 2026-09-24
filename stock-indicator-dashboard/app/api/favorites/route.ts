import { NextRequest, NextResponse } from "next/server";

import {
  marketDataApiBaseUrl,
  marketDataApiBearerToken,
  marketDataApiIsLoopback,
} from "../../lib/market-data-env";
import { marketDataSubject } from "../../lib/market-data-subject";
import { isSupportedMarketInstrumentId } from "../../lib/stock-symbol";

export async function GET() {
  return proxyFavorites("GET", "");
}

export async function PUT(request: NextRequest) {
  return mutateFavorite(request, "PUT");
}

export async function DELETE(request: NextRequest) {
  return mutateFavorite(request, "DELETE");
}

async function mutateFavorite(request: NextRequest, method: "PUT" | "DELETE") {
  const origin = request.headers.get("origin");
  if (origin && origin !== request.nextUrl.origin) {
    return NextResponse.json({ error: "不允许跨站修改自选股" }, { status: 403 });
  }
  const instrumentId = method === "PUT"
    ? await request.json().then((value: unknown) =>
      value && typeof value === "object" && !Array.isArray(value)
        ? String((value as Record<string, unknown>).instrument_id ?? "")
        : ""
    ).catch(() => "")
    : request.nextUrl.searchParams.get("instrument_id") ?? "";
  if (!isSupportedMarketInstrumentId(instrumentId)) {
    return NextResponse.json({ error: "自选股票编号无效" }, { status: 400 });
  }
  return proxyFavorites(
    method,
    method === "PUT"
      ? ""
      : `?instrument_id=${encodeURIComponent(instrumentId)}`,
    method === "PUT" ? JSON.stringify({ instrument_id: instrumentId }) : undefined,
  );
}

async function proxyFavorites(
  method: "GET" | "PUT" | "DELETE",
  suffix: string,
  body?: string,
) {
  const baseUrl = marketDataApiBaseUrl();
  const token = marketDataApiBearerToken();
  if (!token && !marketDataApiIsLoopback(baseUrl)) {
    return NextResponse.json({ error: "自选股服务尚未配置访问凭证" }, { status: 503 });
  }
  try {
    const response = await fetch(`${baseUrl}/v1/favorites${suffix}`, {
      method,
      headers: {
        accept: "application/json",
        ...(body ? { "content-type": "application/json" } : {}),
        ...(token ? { authorization: `Bearer ${token}` } : {}),
        "x-mootdx-subject": await marketDataSubject(),
      },
      body,
      cache: "no-store",
      signal: AbortSignal.timeout(5_000),
    });
    const payload: unknown = await response.json().catch(() => ({ error: "自选股响应无效" }));
    return NextResponse.json(payload, {
      status: response.status,
      headers: { "cache-control": "private, no-store, max-age=0" },
    });
  } catch {
    return NextResponse.json({ error: "无法连接自选股服务" }, { status: 502 });
  }
}
