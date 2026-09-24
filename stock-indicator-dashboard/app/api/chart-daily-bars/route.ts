import { NextRequest, NextResponse } from "next/server";

import {
  marketDataApiBaseUrl,
  marketDataApiBearerToken,
  marketDataApiIsLoopback,
} from "../../lib/market-data-env";
import { isSupportedMarketInstrumentId } from "../../lib/stock-symbol";

const ALLOWED_BASES = new Set(["none", "qfq", "hfq"]);
function rowArray(payload: unknown): unknown[] | null {
  if (Array.isArray(payload)) return payload;
  if (!payload || typeof payload !== "object") return null;
  const record = payload as Record<string, unknown>;
  for (const key of ["rows", "data", "result"]) {
    const candidate = record[key];
    const rows = rowArray(candidate);
    if (rows) return rows;
  }
  return null;
}

export async function GET(request: NextRequest) {
  const instrumentId = request.nextUrl.searchParams.get("instrument_id") ?? "";
  const adjustmentBasis = request.nextUrl.searchParams.get("adjustment_basis") ?? "qfq";

  if (!isSupportedMarketInstrumentId(instrumentId) || !ALLOWED_BASES.has(adjustmentBasis)) {
    return NextResponse.json({ error: "不支持的股票或复权口径" }, { status: 400 });
  }

  const baseUrl = marketDataApiBaseUrl();
  const token = marketDataApiBearerToken();
  if (!token && !marketDataApiIsLoopback(baseUrl)) {
    return NextResponse.json(
      { error: "真实行情服务尚未配置访问凭证" },
      { status: 503 },
    );
  }

  const asOf = new Date();
  const query = new URLSearchParams({
    window_days: "400",
    adjustment_basis: adjustmentBasis,
    limit: "250",
    mode: "latest",
    projection: "chart-lite-v1",
  });
  const upstreamUrl = `${baseUrl}/v1/chart/daily-bars/${instrumentId}?${query}`;

  try {
    const headers = new Headers({ accept: "application/json" });
    if (token) headers.set("authorization", `Bearer ${token}`);
    const ifNoneMatch = request.headers.get("if-none-match");
    if (ifNoneMatch) headers.set("if-none-match", ifNoneMatch);
    const response = await fetch(upstreamUrl, {
      headers,
      signal: AbortSignal.timeout(5_000),
    });
    const responseHeaders = marketDataResponseHeaders(response);
    if (response.status === 304) {
      return new NextResponse(null, { status: 304, headers: responseHeaders });
    }
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      const cacheStatus = response.headers.get("x-mootdx-cache");
      const warming = response.status === 503 && cacheStatus === "snapshot-miss";
      const adjustmentUnavailable = response.status === 409 &&
        cacheStatus === "adjustment-unavailable";
      responseHeaders.set("cache-control", "private, no-store, max-age=0");
      const retryAfter = response.headers.get("retry-after");
      if (retryAfter) responseHeaders.set("retry-after", retryAfter);
      return NextResponse.json(
        {
          error: warming
            ? "日线快照正在后台生成"
            : adjustmentUnavailable
              ? "该股票的复权因子尚未覆盖，已切换为不复权"
              : "真实行情服务暂时不可用",
          upstream_status: response.status,
          retryable: warming,
          retry_after_seconds: retryAfter ? Number(retryAfter) : null,
          fallback_adjustment_basis: adjustmentUnavailable ? "none" : null,
        },
        {
          status: warming ? 503 : adjustmentUnavailable ? 409 : 502,
          headers: responseHeaders,
        },
      );
    }
    const rows = rowArray(payload);
    if (!rows) {
      return NextResponse.json({ error: "真实行情响应缺少日线数组" }, { status: 502 });
    }
    const dataVersion =
      payload && typeof payload === "object" && !Array.isArray(payload)
        ? (payload as Record<string, unknown>).data_version
        : null;
    const payloadAsOf =
      payload && typeof payload === "object" && !Array.isArray(payload)
        ? (payload as Record<string, unknown>).as_of
        : null;
    const etag = response.headers.get("etag") ??
      (typeof dataVersion === "string" ? `"${dataVersion}"` : null);
    if (etag) responseHeaders.set("etag", etag);
    if (ifNoneMatch && etag === ifNoneMatch) {
      return new NextResponse(null, { status: 304, headers: responseHeaders });
    }
    return NextResponse.json({
      rows,
      data_version: typeof dataVersion === "string" ? dataVersion : null,
      as_of: typeof payloadAsOf === "string"
        ? payloadAsOf
        : response.headers.get("x-mootdx-as-of") ?? asOf.toISOString(),
      cache_status: response.headers.get("x-mootdx-cache") ?? "unknown",
      adjustment_basis: adjustmentBasis,
      generation_id: response.headers.get("x-mootdx-chart-generation"),
      business_date: response.headers.get("x-mootdx-chart-business-date"),
      stock_readiness: response.headers.get("x-mootdx-stock-readiness"),
      market_readiness: response.headers.get("x-mootdx-market-readiness"),
      available_adjustment_bases: adjustmentBases(
        response.headers.get("x-mootdx-adjustment-bases"),
      ),
    }, { headers: responseHeaders });
  } catch (error) {
    const timedOut = error instanceof Error &&
      ["TimeoutError", "AbortError"].includes(error.name);
    return NextResponse.json(
      { error: timedOut ? "真实行情查询超时，请重试" : "无法连接真实行情服务" },
      {
        status: 502,
        headers: { "cache-control": "private, no-store, max-age=0" },
      },
    );
  }
}

function marketDataResponseHeaders(response: Response): Headers {
  const headers = new Headers({
    "cache-control": "private, max-age=60, stale-while-revalidate=300",
    "content-type": "application/json; charset=utf-8",
    "x-content-type-options": "nosniff",
  });
  for (const name of [
    "etag",
    "server-timing",
    "x-mootdx-as-of",
    "x-mootdx-cache",
    "x-mootdx-snapshot-refreshed-at",
    "x-mootdx-chart-generation",
    "x-mootdx-chart-business-date",
    "x-mootdx-adjustment-bases",
    "x-mootdx-stock-readiness",
    "x-mootdx-market-readiness",
  ]) {
    const value = response.headers.get(name);
    if (value) headers.set(name, value);
  }
  return headers;
}

function adjustmentBases(value: string | null): string[] {
  if (!value) return ["none", "qfq", "hfq"];
  return value.split(",").map((item) => item.trim()).filter((item) => ALLOWED_BASES.has(item));
}
