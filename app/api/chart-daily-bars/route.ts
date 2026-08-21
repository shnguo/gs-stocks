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

  if (!ALLOWED_INSTRUMENTS.has(instrumentId) || !ALLOWED_BASES.has(adjustmentBasis)) {
    return NextResponse.json({ error: "不支持的股票或复权口径" }, { status: 400 });
  }

  const token = marketDataApiBearerToken();
  if (!token) {
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
  });
  const baseUrl = marketDataApiBaseUrl();
  const upstreamUrl = `${baseUrl}/v1/chart/daily-bars/${instrumentId}?${query}`;

  try {
    const headers = new Headers({
      accept: "application/json",
      authorization: `Bearer ${token}`,
    });
    const ifNoneMatch = request.headers.get("if-none-match");
    if (ifNoneMatch) headers.set("if-none-match", ifNoneMatch);
    const response = await fetch(upstreamUrl, {
      headers,
      signal: AbortSignal.timeout(35_000),
    });
    const responseHeaders = marketDataResponseHeaders(response);
    if (response.status === 304) {
      return new NextResponse(null, { status: 304, headers: responseHeaders });
    }
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      return NextResponse.json(
        { error: "真实行情服务暂时不可用", upstream_status: response.status },
        { status: 502 },
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
    }, { headers: responseHeaders });
  } catch (error) {
    const timedOut = error instanceof Error && error.name === "TimeoutError";
    return NextResponse.json(
      { error: timedOut ? "真实行情查询超时，请重试" : "无法连接真实行情服务" },
      { status: 502 },
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
  ]) {
    const value = response.headers.get(name);
    if (value) headers.set(name, value);
  }
  return headers;
}
