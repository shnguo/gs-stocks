import { NextRequest, NextResponse } from "next/server";

import { derivePinyinInitials } from "../../lib/pinyin-query";
import { normalizeStockSearchHistoricalStatus } from "../../lib/favorite-stock";
import {
  marketDataApiBaseUrl,
  marketDataApiBearerToken,
  marketDataApiIsLoopback,
} from "../../lib/market-data-env";
import { marketInstrumentFromCatalog, resolveAShareSymbol } from "../../lib/stock-symbol";

const SEARCH_ENDPOINT = "https://searchapi.eastmoney.com/api/suggest/get";
const SEARCH_TOKEN = "D43BF722C8E33BDC906FB84D85E326E8";
const ALLOWED_QUERY = /^[0-9A-Za-z\u3400-\u9fff·._\-\s]+$/u;

type SuggestionRecord = Record<string, unknown>;

function suggestionRows(payload: unknown): SuggestionRecord[] {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) return [];
  const table = (payload as Record<string, unknown>).QuotationCodeTable;
  if (!table || typeof table !== "object" || Array.isArray(table)) return [];
  const data = (table as Record<string, unknown>).Data;
  return Array.isArray(data)
    ? data.filter((item): item is SuggestionRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item))
    : [];
}

async function fetchSuggestions(query: string): Promise<SuggestionRecord[]> {
  const search = new URLSearchParams({
    input: query,
    type: "14",
    token: SEARCH_TOKEN,
    count: "12",
  });
  const response = await fetch(`${SEARCH_ENDPOINT}?${search}`, {
    headers: {
      accept: "application/json, text/plain, */*",
      referer: "https://quote.eastmoney.com/",
    },
    signal: AbortSignal.timeout(4_000),
  });
  if (!response.ok) throw new Error(`stock search upstream returned ${response.status}`);
  return suggestionRows(await response.json());
}

async function fetchCatalogSuggestions(query: string, market: string): Promise<SuggestionRecord[]> {
  const baseUrl = marketDataApiBaseUrl();
  const token = marketDataApiBearerToken();
  if (!token && !marketDataApiIsLoopback(baseUrl)) return [];
  const search = new URLSearchParams({ q: query });
  if (market === "hk" || market === "us") search.set("market", market);
  search.set("include_inactive", "1");
  const response = await fetch(`${baseUrl}/v1/market-catalog?${search}`, {
    headers: {
      accept: "application/json",
      ...(token ? { authorization: `Bearer ${token}` } : {}),
    },
    signal: AbortSignal.timeout(4_000),
  });
  if (!response.ok) throw new Error(`market catalog returned ${response.status}`);
  const payload: unknown = await response.json();
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) return [];
  const results = (payload as Record<string, unknown>).results;
  return Array.isArray(results)
    ? results.filter((item): item is SuggestionRecord =>
      Boolean(item) && typeof item === "object" && !Array.isArray(item)
    )
    : [];
}

export async function GET(request: NextRequest) {
  const query = (request.nextUrl.searchParams.get("q") ?? "").trim();
  const market = (request.nextUrl.searchParams.get("market") ?? "").trim().toLowerCase();
  if (!query || query.length > 40 || !ALLOWED_QUERY.test(query)) {
    return NextResponse.json(
      { error: "请输入股票代码、中文名称、字母或拼音" },
      { status: 400 },
    );
  }
  if (market && !["cn", "hk", "us"].includes(market)) {
    return NextResponse.json({ error: "不支持的市场筛选" }, { status: 400 });
  }

  const pinyinInitials = derivePinyinInitials(query);
  const queries = pinyinInitials && pinyinInitials !== query.toLowerCase()
    ? [query, pinyinInitials]
    : [query];

  try {
    const tasks: Promise<{ kind: "ashare" | "catalog"; rows: SuggestionRecord[] }>[] = [];
    if (!market || market === "cn") {
      tasks.push(...queries.map(async (searchQuery) => ({
        kind: "ashare" as const,
        rows: await fetchSuggestions(searchQuery),
      })));
    }
    if (!market || market === "hk" || market === "us") {
      tasks.push((async () => ({
        kind: "catalog" as const,
        rows: await fetchCatalogSuggestions(query, market),
      }))());
    }
    const settled = await Promise.allSettled(tasks);
    const successful = settled.flatMap((result) => result.status === "fulfilled" ? [result.value] : []);
    if (successful.length === 0) throw new Error("all stock search sources failed");
    const aShareResults = successful
      .filter((result) => result.kind === "ashare")
      .flatMap((result) => result.rows)
      .flatMap((row) => {
      if (row.Classify !== "AStock") return [];
      const code = typeof row.Code === "string" ? row.Code : "";
      const name = typeof row.Name === "string" ? row.Name.trim() : "";
      const identity = resolveAShareSymbol(code);
      if (!identity || !name) return [];
      return [{
        ...identity,
        name,
        pinyinInitials: typeof row.PinYin === "string" ? row.PinYin.toLowerCase() : "",
        historicalStatus: "ready" as const,
      }];
    });
    const catalogResults = successful
      .filter((result) => result.kind === "catalog")
      .flatMap((result) => result.rows)
      .flatMap((row) => {
        const identity = marketInstrumentFromCatalog(row);
        const name = typeof row.name === "string" ? row.name.trim() : "";
        return identity && name
          ? [{
            ...identity,
            name,
            pinyinInitials: "",
            historicalStatus: normalizeStockSearchHistoricalStatus(row.historical_status),
          }]
          : [];
      });
    const results = [...aShareResults, ...catalogResults];
    const uniqueResults = Array.from(
      new Map(results.map((item) => [item.instrumentId, item])).values(),
    );
    return NextResponse.json(
      { results: uniqueResults.slice(0, 12) },
      {
        headers: {
          "cache-control": "private, no-store, max-age=0",
          "x-content-type-options": "nosniff",
        },
      },
    );
  } catch (error) {
    const timedOut = error instanceof Error && ["TimeoutError", "AbortError"].includes(error.name);
    return NextResponse.json(
      { error: timedOut ? "股票搜索超时，请稍后重试" : "股票搜索服务暂时不可用，请稍后重试" },
      {
        status: 502,
        headers: { "cache-control": "private, no-store, max-age=0" },
      },
    );
  }
}
