import { NextRequest, NextResponse } from "next/server";

import {
  marketDataApiBaseUrl,
  marketDataApiBearerToken,
  marketDataApiIsLoopback,
} from "../../lib/market-data-env";

const ISO_DATE = /^\d{4}-\d{2}-\d{2}$/;
const MAX_RANGE_DAYS = 100;

type MarketDayRecord = {
  business_date: string;
  status: string;
  observed_at: string;
};

export async function GET(request: NextRequest) {
  const startDate = request.nextUrl.searchParams.get("start_date") ?? "";
  const endDate = request.nextUrl.searchParams.get("end_date") ?? "";
  if (!validDateRange(startDate, endDate)) {
    return NextResponse.json(
      { error: "休市日查询范围无效" },
      { status: 400, headers: { "cache-control": "private, no-store, max-age=0" } },
    );
  }

  const baseUrl = marketDataApiBaseUrl();
  const token = marketDataApiBearerToken();
  if (!token && !marketDataApiIsLoopback(baseUrl)) {
    return NextResponse.json(
      { error: "真实行情服务尚未配置访问凭证" },
      { status: 503, headers: { "cache-control": "private, no-store, max-age=0" } },
    );
  }

  const query = new URLSearchParams({
    start_date: startDate,
    end_date: endDate,
    limit: String(MAX_RANGE_DAYS),
  });
  try {
    const headers = new Headers({ accept: "application/json" });
    if (token) headers.set("authorization", `Bearer ${token}`);
    const response = await fetch(`${baseUrl}/v1/ashare/market-days?${query}`, {
      headers,
      cache: "no-store",
      signal: AbortSignal.timeout(10_000),
    });
    const payload: unknown = await response.json().catch(() => null);
    if (!response.ok || !Array.isArray(payload)) {
      return NextResponse.json(
        { error: "无法核对交易日连续性" },
        { status: 502, headers: { "cache-control": "private, no-store, max-age=0" } },
      );
    }
    const rows = payload.map(marketDayRecord).filter((row): row is MarketDayRecord => row !== null);
    return NextResponse.json(
      { rows, start_date: startDate, end_date: endDate },
      { headers: { "cache-control": "private, no-store, max-age=0" } },
    );
  } catch {
    return NextResponse.json(
      { error: "无法连接交易日连续性服务" },
      { status: 502, headers: { "cache-control": "private, no-store, max-age=0" } },
    );
  }
}

function validDateRange(startDate: string, endDate: string): boolean {
  if (!ISO_DATE.test(startDate) || !ISO_DATE.test(endDate)) return false;
  const start = Date.parse(`${startDate}T00:00:00Z`);
  const end = Date.parse(`${endDate}T00:00:00Z`);
  return Number.isFinite(start) && Number.isFinite(end) && end >= start &&
    (end - start) / 86_400_000 < MAX_RANGE_DAYS;
}

function marketDayRecord(value: unknown): MarketDayRecord | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const record = value as Record<string, unknown>;
  const businessDate = typeof record.business_date === "string" ? record.business_date : "";
  const status = typeof record.status === "string" ? record.status : "";
  const observedAt = typeof record.observed_at === "string" ? record.observed_at : "";
  if (!ISO_DATE.test(businessDate) || !status || !observedAt) return null;
  return { business_date: businessDate, status, observed_at: observedAt };
}
