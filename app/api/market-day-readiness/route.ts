import { NextRequest, NextResponse } from "next/server";

import {
  marketDataApiBaseUrl,
  marketDataApiBearerToken,
  marketDataApiIsLoopback,
} from "../../lib/market-data-env";
import { isSupportedAShareInstrumentId } from "../../lib/stock-symbol";

const ISO_DATE = /^\d{4}-\d{2}-\d{2}$/;

export async function GET(request: NextRequest) {
  const businessDate = request.nextUrl.searchParams.get("business_date") ?? "";
  const instrumentId = request.nextUrl.searchParams.get("instrument_id") ?? "";
  if (!ISO_DATE.test(businessDate) || !isSupportedAShareInstrumentId(instrumentId)) {
    return NextResponse.json(
      { error: "股票日线就绪状态查询无效" },
      { status: 400, headers: noStoreHeaders() },
    );
  }

  const baseUrl = marketDataApiBaseUrl();
  const token = marketDataApiBearerToken();
  if (!token && !marketDataApiIsLoopback(baseUrl)) {
    return NextResponse.json(
      { error: "真实行情服务尚未配置访问凭证" },
      { status: 503, headers: noStoreHeaders() },
    );
  }

  try {
    const headers = new Headers({ accept: "application/json" });
    if (token) headers.set("authorization", `Bearer ${token}`);
    const response = await fetch(
      `${baseUrl}/v1/ashare/market-days/${businessDate}/instruments/${instrumentId}`,
      {
        headers,
        cache: "no-store",
        signal: AbortSignal.timeout(10_000),
      },
    );
    const payload: unknown = await response.json().catch(() => null);
    if (!response.ok || !isReadinessPayload(payload)) {
      return NextResponse.json(
        { error: "无法读取股票日线就绪状态" },
        { status: 502, headers: noStoreHeaders() },
      );
    }
    return NextResponse.json(payload, { headers: noStoreHeaders() });
  } catch {
    return NextResponse.json(
      { error: "无法连接股票日线就绪服务" },
      { status: 502, headers: noStoreHeaders() },
    );
  }
}

function isReadinessPayload(value: unknown): value is Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const record = value as Record<string, unknown>;
  return ISO_DATE.test(String(record.business_date ?? "")) &&
    isSupportedAShareInstrumentId(String(record.instrument_id ?? "")) &&
    typeof record.readiness === "string" &&
    typeof record.stock_accounted === "boolean" &&
    typeof record.chart_ready === "boolean" &&
    typeof record.market_complete === "boolean";
}

function noStoreHeaders(): HeadersInit {
  return { "cache-control": "private, no-store, max-age=0" };
}
