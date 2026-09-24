import { NextRequest, NextResponse } from "next/server";

import {
  marketDataApiBaseUrl,
  marketDataApiBearerToken,
  marketDataApiIsLoopback,
} from "../../lib/market-data-env";
import { marketDataSubject } from "../../lib/market-data-subject";

export async function GET() {
  return proxyAdmission("GET");
}

export async function POST(request: NextRequest) {
  const origin = request.headers.get("origin");
  if (origin && origin !== request.nextUrl.origin) {
    return NextResponse.json({ error: "不允许跨站提交股票申请" }, { status: 403 });
  }
  const text = await request.text();
  if (!text || new TextEncoder().encode(text).byteLength > 4_096) {
    return NextResponse.json({ error: "股票申请内容无效" }, { status: 400 });
  }
  return proxyAdmission("POST", text);
}

async function proxyAdmission(method: "GET" | "POST", body?: string) {
  const baseUrl = marketDataApiBaseUrl();
  const token = marketDataApiBearerToken();
  if (!token && !marketDataApiIsLoopback(baseUrl)) {
    return NextResponse.json({ error: "股票申请服务尚未配置访问凭证" }, { status: 503 });
  }
  try {
    const response = await fetch(`${baseUrl}/v1/symbol-admissions`, {
      method,
      headers: {
        accept: "application/json",
        ...(body ? { "content-type": "application/json" } : {}),
        ...(token ? { authorization: `Bearer ${token}` } : {}),
        "x-mootdx-subject": await marketDataSubject(),
      },
      body,
      cache: "no-store",
      signal: AbortSignal.timeout(8_000),
    });
    const payload: unknown = await response.json().catch(() => ({ error: "股票申请响应无效" }));
    return NextResponse.json(payload, {
      status: response.status,
      headers: {
        "cache-control": "private, no-store, max-age=0",
        "x-content-type-options": "nosniff",
        ...(response.headers.get("retry-after")
          ? { "retry-after": response.headers.get("retry-after") as string }
          : {}),
      },
    });
  } catch (error) {
    const timedOut = error instanceof Error && error.name === "TimeoutError";
    return NextResponse.json(
      { error: timedOut ? "股票申请服务响应超时" : "无法连接股票申请服务" },
      { status: 502 },
    );
  }
}
