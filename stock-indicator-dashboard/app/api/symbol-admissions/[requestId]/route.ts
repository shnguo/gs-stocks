import { NextRequest, NextResponse } from "next/server";

import {
  marketDataApiBaseUrl,
  marketDataApiBearerToken,
  marketDataApiIsLoopback,
} from "../../../lib/market-data-env";
import { marketDataSubject } from "../../../lib/market-data-subject";

export async function GET(
  _request: NextRequest,
  context: { params: Promise<{ requestId: string }> },
) {
  const { requestId } = await context.params;
  if (!/^[0-9a-f]{32}$/u.test(requestId)) {
    return NextResponse.json({ error: "股票申请编号无效" }, { status: 400 });
  }
  const baseUrl = marketDataApiBaseUrl();
  const token = marketDataApiBearerToken();
  if (!token && !marketDataApiIsLoopback(baseUrl)) {
    return NextResponse.json({ error: "股票申请服务尚未配置访问凭证" }, { status: 503 });
  }
  try {
    const response = await fetch(`${baseUrl}/v1/symbol-admissions/${requestId}`, {
      headers: {
        accept: "application/json",
        ...(token ? { authorization: `Bearer ${token}` } : {}),
        "x-mootdx-subject": await marketDataSubject(),
      },
      cache: "no-store",
      signal: AbortSignal.timeout(5_000),
    });
    const payload: unknown = await response.json().catch(() => ({ error: "股票申请响应无效" }));
    return NextResponse.json(payload, {
      status: response.status,
      headers: { "cache-control": "private, no-store, max-age=0" },
    });
  } catch {
    return NextResponse.json({ error: "无法读取股票申请进度" }, { status: 502 });
  }
}
