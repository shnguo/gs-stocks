import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("renders the stock research dashboard", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = await response.text();
  assert.match(html, /低位承接研究台/);
  assert.match(html, /mootdx-cf 真实日线/);
  assert.match(html, /正在读取已发布的日线快照/);
  assert.doesNotMatch(html, /演示行情为模拟数据|腾讯控股/);
  assert.doesNotMatch(html, /codex-preview|Building your site|SkeletonPreview/);
});

test("emits finished site metadata", async () => {
  const response = await render();
  const html = await response.text();
  assert.match(html, /<html[^>]+lang="zh-CN"/i);
  assert.match(html, /<title>低位承接研究台｜蜡烛图与承接强度<\/title>/i);
  assert.match(html, /日线蜡烛图与低位承接强度指标/);
});

test("renders the favorite-stock sidebar with market tabs", async () => {
  const response = await render();
  const html = await response.text();
  assert.match(html, /自选股票/);
  assert.match(html, /按市场筛选收藏股票/);
  assert.match(html, /A股/);
  assert.match(html, /港股/);
  assert.match(html, /美股/);
  assert.match(html, /澜起科技/);
  assert.match(html, /左侧收藏栏/);
  assert.match(html, /右侧信息面板/);
  assert.match(html, /右侧区域已预留/);
  assert.match(html, /收起左侧栏/);
  assert.match(html, /收起右侧栏/);
  assert.match(html, /添加自选股票/);
  assert.match(html, /右键管理/);
  assert.match(html, /aria-haspopup="menu"/);
  assert.doesNotMatch(html, /从自选删除澜起科技/);
});

test("supports persistent add and remove actions for favorite stocks", async () => {
  const clientSource = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  const styles = await readFile(new URL("../app/globals.css", import.meta.url), "utf8");
  assert.match(clientSource, /FAVORITES_STORAGE_KEY/);
  assert.match(clientSource, /localStorage\.getItem\(FAVORITES_STORAGE_KEY\)/);
  assert.match(clientSource, /localStorage\.setItem\(FAVORITES_STORAGE_KEY/);
  assert.match(clientSource, /function addFavorite\(nextStock: StockPreset\)/);
  assert.match(clientSource, /function removeFavorite\(instrumentId: string\)/);
  assert.match(clientSource, /resolveCanonicalFavorite/);
  assert.match(clientSource, /isPlaceholderFavoriteName/);
  assert.match(clientSource, /aria-controls="favorite-stock-manager"/);
  assert.match(clientSource, /function addSearchResult/);
  assert.match(clientSource, /\/api\/stock-search\?q=/);
  assert.match(clientSource, /role="combobox"/);
  assert.match(clientSource, /role="listbox"/);
  assert.match(clientSource, /代码、名称或拼音/);
  assert.match(clientSource, /支持代码、中文、字母、全拼和拼音首字母/);
  assert.doesNotMatch(clientSource, /searchResults\s*=\s*STOCKS\.filter/);
  assert.doesNotMatch(clientSource, /没有匹配的受支持股票/);
  assert.match(clientSource, /onContextMenu=\{\(event\) => handleStockContextMenu\(event, item\.instrumentId\)\}/);
  assert.match(clientSource, /event\.key !== "ContextMenu"/);
  assert.match(clientSource, /role="menu"/);
  assert.match(clientSource, /从自选删除/);
  assert.doesNotMatch(clientSource, /className="watchlist-remove"/);
  assert.match(styles, /\.favorites-manager/);
  assert.match(styles, /\.favorites-search-field/);
  assert.match(styles, /\.favorites-search-results/);
  assert.match(styles, /\.watchlist-items button:focus-visible/);
  assert.match(styles, /\.watchlist-select\s*\{[^}]*min-height:\s*46px/);
  assert.match(clientSource, /const quote = quotesByInstrumentId\[item\.instrumentId\]/);
  assert.match(clientSource, /watchlist-quote watchlist-quote-unavailable/);
  assert.match(styles, /\.watchlist-quote-unavailable/);
  assert.doesNotMatch(styles, /\.watchlist-remove/);
});

test("recognizes legacy placeholder favorite names for canonical repair", async () => {
  const {
    isPlaceholderFavoriteName,
    normalizeStockSearchHistoricalStatus,
    stockSearchReadinessLabel,
    stockSearchResultIsAddable,
  } = await import("../app/lib/favorite-stock.ts");
  assert.equal(isPlaceholderFavoriteName("A股 601318", "601318"), true);
  assert.equal(isPlaceholderFavoriteName("股票601318", "601318"), true);
  assert.equal(isPlaceholderFavoriteName("601318", "601318"), true);
  assert.equal(isPlaceholderFavoriteName("中国平安", "601318"), false);
  assert.equal(normalizeStockSearchHistoricalStatus("ready"), "ready");
  assert.equal(normalizeStockSearchHistoricalStatus("unexpected"), "inactive");
  assert.equal(stockSearchResultIsAddable("ready"), true);
  assert.equal(stockSearchResultIsAddable("inactive"), false);
  assert.equal(stockSearchReadinessLabel("inactive"), "数据待接入");
  assert.equal(stockSearchReadinessLabel("bootstrapping"), "数据准备中");
});

test("accepts canonical custom Hong Kong and US instrument ids", async () => {
  const {
    isSupportedMarketInstrumentId,
    marketInstrumentFromCatalog,
    providerSupportsRealtimeInstrument,
    realtimeMarketInstrument,
  } = await import("../app/lib/stock-symbol.ts");
  assert.equal(isSupportedMarketInstrumentId("us.xnas.skhy"), true);
  assert.equal(isSupportedMarketInstrumentId("hk.xhkg.00700"), true);
  assert.equal(isSupportedMarketInstrumentId("us.otc.skhy"), false);
  assert.equal(isSupportedMarketInstrumentId("us.xnas.bad/symbol"), false);
  assert.deepEqual(
    marketInstrumentFromCatalog({
      market: "us",
      exchange_id: "xnas",
      instrument_id: "us.xnas.skhy",
      symbol: "SKHY",
      currency: "USD",
      available_adjustment_bases: ["none"],
    }),
    {
      code: "SKHY",
      exchangeId: "xnas",
      instrumentId: "us.xnas.skhy",
      market: "纳斯达克",
      marketGroup: "美股",
      marketRegion: "us",
      currency: "USD",
      availableAdjustmentBases: ["none"],
    },
  );
  const apple = marketInstrumentFromCatalog({
    market: "us",
    exchange_id: "xnas",
    instrument_id: "instrument.apple",
    symbol: "AAPL",
    currency: "USD",
    available_adjustment_bases: ["none"],
  });
  assert.deepEqual(realtimeMarketInstrument(apple), {
    instrumentId: "instrument.apple",
    market: "us",
    providerSymbol: "AAPL.US",
    currency: "USD",
  });
  assert.equal(providerSupportsRealtimeInstrument("longbridge", apple), true);
  assert.equal(providerSupportsRealtimeInstrument("tencent", apple), false);
});

test("repairs completed US history gaps before joining the live candle", async () => {
  const clientSource = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  const routeSource = await readFile(
    new URL("../app/api/historical-gap-repair/route.ts", import.meta.url),
    "utf8",
  );
  assert.match(clientSource, /fetch\("\/api\/historical-gap-repair"/);
  assert.match(clientSource, /mergeHistoricalGapRows/);
  assert.match(clientSource, /parseHistoricalGapRepairRows/);
  assert.match(clientSource, /adjustmentBasis === "hfq"/);
  assert.match(clientSource, /remoteHistoryRevalidating/);
  assert.match(clientSource, /visibleHistoricalGap && realtime\.quote/);
  assert.match(clientSource, /data-historical-gap=\{visibleHistoricalGap/);
  assert.match(routeSource, /\/v1\/multimarket\/history\/repair-gap/);
  assert.match(routeSource, /end - start <= 31 \* 86_400_000/);
  assert.match(routeSource, /origin !== request\.nextUrl\.origin/);
  assert.match(routeSource, /marketDataRealtimeApiBaseUrl/);
});

test("keeps sparse chart histories dense and left aligned", async () => {
  const {
    MIN_VISIBLE_CHART_SLOTS,
    chartDataIndexAt,
    chartHorizontalLayout,
    chartXAt,
  } = await import("../app/chart-layout.ts");
  const layout = chartHorizontalLayout(37, 1028);
  assert.equal(MIN_VISIBLE_CHART_SLOTS, 60);
  assert.equal(layout.slotCount, 60);
  assert.equal(layout.leadingSlots, 0);
  assert.equal(layout.candleWidth, 8);
  assert.ok(Math.abs(chartXAt(0, 18, layout) - 26.566666666666666) < 1e-9);
  assert.equal(chartDataIndexAt(chartXAt(0, 18, layout), 18, 37, layout), 0);
  assert.equal(chartDataIndexAt(chartXAt(36, 18, layout), 18, 37, layout), 36);
  assert.equal(chartDataIndexAt(1040, 18, 37, layout), null);

  const fullLayout = chartHorizontalLayout(120, 1028);
  assert.equal(fullLayout.slotCount, 120);
  assert.equal(fullLayout.leadingSlots, 0);
});

test("turns full and partial pinyin into stock-name initials", async () => {
  const { derivePinyinInitials } = await import("../app/lib/pinyin-query.ts");
  assert.equal(derivePinyinInitials("zhongguopingan"), "zgpa");
  assert.equal(derivePinyinInitials("guizhoumaotai"), "gzmt");
  assert.equal(derivePinyinInitials("ningdeshidai"), "ndsd");
  assert.equal(derivePinyinInitials("lanqikeji"), "lqkj");
  assert.equal(derivePinyinInitials("zhongguop"), "zgp");
  assert.equal(derivePinyinInitials("zgpa"), null);
});

test("keeps fuzzy stock search on the server and filters to A shares", async () => {
  const routeSource = await readFile(new URL("../app/api/stock-search/route.ts", import.meta.url), "utf8");
  assert.match(routeSource, /searchapi\.eastmoney\.com\/api\/suggest\/get/);
  assert.match(routeSource, /derivePinyinInitials/);
  assert.match(routeSource, /row\.Classify !== "AStock"/);
  assert.match(routeSource, /resolveAShareSymbol\(code\)/);
  assert.match(routeSource, /AbortSignal\.timeout\(4_000\)/);
  assert.match(routeSource, /search\.set\("include_inactive", "1"\)/);
  assert.match(routeSource, /normalizeStockSearchHistoricalStatus/);
});

test("resolves new A-share symbols without a fixed four-stock allowlist", async () => {
  const { resolveAShareSymbol, isSupportedAShareInstrumentId } = await import(
    "../app/lib/stock-symbol.ts"
  );
  assert.deepEqual(resolveAShareSymbol("600519"), {
    code: "600519",
    exchangeId: "xshg",
    instrumentId: "cn.xshg.600519",
    market: "沪市",
    marketGroup: "A股",
    marketRegion: "cn",
    currency: "CNY",
    availableAdjustmentBases: ["none", "qfq", "hfq"],
  });
  assert.equal(resolveAShareSymbol("300750")?.instrumentId, "cn.xshe.300750");
  assert.equal(resolveAShareSymbol("830799")?.instrumentId, "cn.xbse.830799");
  assert.equal(resolveAShareSymbol("920001")?.instrumentId, "cn.xbse.920001");
  assert.equal(resolveAShareSymbol("P196"), null);
  assert.equal(isSupportedAShareInstrumentId("cn.xshg.600519"), true);
  assert.equal(isSupportedAShareInstrumentId("us.xnas.aapl"), false);
});

test("keeps the market data credential on the server", async () => {
  const clientSource = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  const routeSource = await readFile(new URL("../app/api/chart-daily-bars/route.ts", import.meta.url), "utf8");
  const marketDaysRouteSource = await readFile(new URL("../app/api/market-days/route.ts", import.meta.url), "utf8");
  const readinessRouteSource = await readFile(
    new URL("../app/api/market-day-readiness/route.ts", import.meta.url),
    "utf8",
  );
  const ticketRouteSource = await readFile(new URL("../app/api/realtime-ticket/route.ts", import.meta.url), "utf8");
  const quoteRouteSource = await readFile(new URL("../app/api/realtime-quote/route.ts", import.meta.url), "utf8");
  const quotesRouteSource = await readFile(new URL("../app/api/realtime-quotes/route.ts", import.meta.url), "utf8");
  const providerRouteSource = await readFile(new URL("../app/api/realtime-provider/route.ts", import.meta.url), "utf8");
  const environmentSource = await readFile(new URL("../app/lib/market-data-env.ts", import.meta.url), "utf8");
  const previewProxySource = await readFile(new URL("../scripts/mootdx-preview-proxy.mjs", import.meta.url), "utf8");
  const previewProxyResponseSource = await readFile(
    new URL("../scripts/preview-proxy-response.mjs", import.meta.url),
    "utf8",
  );
  assert.doesNotMatch(clientSource, /MOOTDX_DATA_API_BEARER_TOKEN|authorization:\s*`Bearer/);
  assert.doesNotMatch(ticketRouteSource, /MOOTDX_DATA_API_BEARER_TOKEN/);
  assert.doesNotMatch(quoteRouteSource, /MOOTDX_DATA_API_BEARER_TOKEN/);
  assert.doesNotMatch(quotesRouteSource, /MOOTDX_DATA_API_BEARER_TOKEN/);
  assert.doesNotMatch(providerRouteSource, /MOOTDX_DATA_API_BEARER_TOKEN/);
  assert.doesNotMatch(marketDaysRouteSource, /NEXT_PUBLIC_/);
  assert.match(environmentSource, /MOOTDX_DATA_API_BEARER_TOKEN/);
  assert.match(environmentSource, /marketDataApiIsLoopback/);
  assert.match(routeSource, /headers\.set\("authorization",\s*`Bearer/);
  assert.match(marketDaysRouteSource, /headers\.set\("authorization",\s*`Bearer/);
  assert.match(marketDaysRouteSource, /start_date/);
  assert.match(marketDaysRouteSource, /end_date/);
  assert.match(marketDaysRouteSource, /cache:\s*"no-store"/);
  assert.match(readinessRouteSource, /\/v1\/ashare\/market-days\/\$\{businessDate\}\/instruments\/\$\{instrumentId\}/);
  assert.match(readinessRouteSource, /cache:\s*"no-store"/);
  assert.match(readinessRouteSource, /isSupportedAShareInstrumentId/);
  assert.match(previewProxySource, /previewProxyRequestKind/);
  assert.match(previewProxyResponseSource, /\/v1\/ashare\/market-days/);
  assert.match(ticketRouteSource, /authorization:\s*`Bearer/);
  assert.match(quoteRouteSource, /authorization:\s*`Bearer/);
  assert.match(quotesRouteSource, /authorization:\s*`Bearer/);
  assert.match(providerRouteSource, /authorization:\s*`Bearer/);
  assert.match(routeSource, /isSupportedMarketInstrumentId/);
  assert.doesNotMatch(routeSource, /ALLOWED_INSTRUMENTS/);
  assert.match(routeSource, /mode:\s*"latest"/);
  assert.match(routeSource, /window_days:\s*"400"/);
  assert.doesNotMatch(routeSource, /start_date:/);
  assert.doesNotMatch(routeSource, /end_date:/);
  assert.match(routeSource, /if-none-match/);
  assert.doesNotMatch(routeSource, /cache:\s*"no-store"/);
});

test("loads compact snapshots early and restores the last verified browser copy", async () => {
  const layoutSource = await readFile(new URL("../app/layout.tsx", import.meta.url), "utf8");
  const clientSource = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  const cacheSource = await readFile(new URL("../app/chart-cache.ts", import.meta.url), "utf8");
  const routeSource = await readFile(
    new URL("../app/api/chart-daily-bars/route.ts", import.meta.url),
    "utf8",
  );

  assert.match(layoutSource, /rel="preload"[\s\S]{0,180}as="fetch"/);
  assert.match(layoutSource, /instrument_id=cn\.xshg\.688008&adjustment_basis=none/);
  assert.match(routeSource, /projection:\s*"chart-lite-v1"/);
  assert.match(routeSource, /AbortSignal\.timeout\(5_000\)/);
  assert.match(routeSource, /adjustment-unavailable/);
  assert.match(routeSource, /retry_after_seconds/);
  assert.match(routeSource, /x-mootdx-chart-generation/);
  assert.match(routeSource, /available_adjustment_bases/);
  assert.match(clientSource, /fetchChartPayload/);
  assert.match(clientSource, /record\.retryable === true/);
  assert.match(clientSource, /maximumAttempts = 3/);
  assert.match(clientSource, /availableAdjustmentBases/);
  assert.doesNotMatch(clientSource, /ADJUSTED_INSTRUMENT_IDS/);
  assert.match(clientSource, /readChartCache\(dataKey\)/);
  assert.match(clientSource, /fetchChartPayload\(query, controller\.signal, "no-store"\)/);
  assert.match(clientSource, /writeChartCache\(dataKey, payload\)/);
  assert.match(clientSource, /cacheStatus:\s*"browser-stale"/);
  assert.match(cacheSource, /indexedDB\.open/);
  assert.match(cacheSource, /chart-payloads/);
  assert.doesNotMatch(clientSource, /R2 SQL 冷查询/);
});

test("renders the requested moving-average set on the price chart", async () => {
  const clientSource = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  assert.match(clientSource, /MOVING_AVERAGE_PERIODS\s*=\s*\[5, 10, 20, 30, 60\]/);
  assert.match(clientSource, /displayMovingAverageValues/);
  assert.match(clientSource, /MA5、MA10、MA20、MA30和MA60均线/);
  assert.match(clientSource, /ma-line-\$\{movingAveragePeriod\}/);
  assert.match(clientSource, /formatNumber\(values\[activeIndex\]\)/);
  assert.doesNotMatch(clientSource, /legend-value-\$\{movingAveragePeriod\}/);
});

test("promotes only closed complete realtime quotes into formal calculations", async () => {
  const {
    closedBarRevalidationDelayMs,
    completedQuoteNeedsHistoricalRevalidation,
    hasHistoricalSeriesGap,
    historicalSeriesGapDates,
    instrumentReadinessRetryDelayMs,
    isCompletedRealtimeQuote,
    selectProgressiveRealtimeQuote,
    selectCompletedSeries,
  } = await import("../app/market-series.ts");
  const history = [{ date: "2026-08-20" }];
  const completedQuote = [...history, { date: "2026-08-21", realtime: true, unconfirmed: false }];
  const liveQuote = [...history, { date: "2026-08-21", realtime: true, unconfirmed: true }];

  assert.equal(isCompletedRealtimeQuote({
    completeness: "complete",
    marketSession: "closed",
    tradingStatus: "closed",
  }), true);
  assert.equal(isCompletedRealtimeQuote({
    completeness: "complete",
    marketSession: "continuous",
    tradingStatus: "trading",
  }), false);
  const currentLiveQuote = {
    completeness: "partial",
    marketSession: "continuous",
    tradingStatus: "trading",
    tradingDate: "2026-08-21",
    observedAt: "2026-08-21T06:30:00Z",
    last: 10,
  };
  const newerLiveQuote = {
    ...currentLiveQuote,
    observedAt: "2026-08-21T06:31:00Z",
    last: 11,
  };
  const closingQuote = {
    ...newerLiveQuote,
    completeness: "complete",
    marketSession: "closed",
    tradingStatus: "closed",
    observedAt: "2026-08-21T07:00:00Z",
    last: 12,
  };
  const lateIntradayQuote = {
    ...newerLiveQuote,
    observedAt: "2026-08-21T07:01:00Z",
    last: 13,
  };
  const nextDayLiveQuote = {
    ...newerLiveQuote,
    tradingDate: "2026-08-22",
    observedAt: "2026-08-22T01:31:00Z",
    last: 14,
  };
  assert.equal(selectProgressiveRealtimeQuote(currentLiveQuote, newerLiveQuote), newerLiveQuote);
  assert.equal(selectProgressiveRealtimeQuote(newerLiveQuote, closingQuote), closingQuote);
  assert.equal(selectProgressiveRealtimeQuote(closingQuote, lateIntradayQuote), closingQuote);
  assert.equal(selectProgressiveRealtimeQuote(closingQuote, nextDayLiveQuote), nextDayLiveQuote);
  assert.equal(selectCompletedSeries(history, completedQuote), completedQuote);
  assert.equal(selectCompletedSeries(history, liveQuote), history);
  assert.equal(completedQuoteNeedsHistoricalRevalidation(history, {
    completeness: "complete",
    marketSession: "closed",
    tradingStatus: "closed",
    tradingDate: "2026-08-21",
  }), true);
  assert.equal(completedQuoteNeedsHistoricalRevalidation(
    [...history, { date: "2026-08-21" }],
    {
      completeness: "complete",
      marketSession: "closed",
      tradingStatus: "closed",
      tradingDate: "2026-08-21",
    },
  ), false);
  assert.equal(closedBarRevalidationDelayMs(0), 5_000);
  assert.equal(closedBarRevalidationDelayMs(99), 300_000);
  assert.equal(instrumentReadinessRetryDelayMs("awaiting-universe", 0), 60_000);
  assert.equal(instrumentReadinessRetryDelayMs("processing", 0), 15_000);
  assert.equal(instrumentReadinessRetryDelayMs("published", 0), 5_000);
  assert.equal(hasHistoricalSeriesGap(history, { tradingDate: "2026-08-21" }), false);
  assert.equal(hasHistoricalSeriesGap(
    [{ date: "2026-08-21" }],
    { tradingDate: "2026-08-25" },
  ), true);
  assert.equal(hasHistoricalSeriesGap(
    [{ date: "2026-08-21" }],
    { tradingDate: "2026-08-24" },
  ), false);
  assert.deepEqual(
    historicalSeriesGapDates(
      [{ date: "2026-09-30" }],
      { tradingDate: "2026-10-09" },
    ),
    ["2026-10-01", "2026-10-02", "2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08"],
  );
  const verifiedHolidayDates = new Set([
    "2026-10-01",
    "2026-10-02",
    "2026-10-05",
    "2026-10-06",
    "2026-10-07",
    "2026-10-08",
  ]);
  assert.equal(hasHistoricalSeriesGap(
    [{ date: "2026-09-30" }],
    { tradingDate: "2026-10-09" },
    verifiedHolidayDates,
  ), false);
  verifiedHolidayDates.delete("2026-10-08");
  assert.equal(hasHistoricalSeriesGap(
    [{ date: "2026-09-30" }],
    { tradingDate: "2026-10-09" },
    verifiedHolidayDates,
  ), true);
});

test("revalidates a completed realtime candle until canonical history publishes it", async () => {
  const clientSource = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  assert.match(clientSource, /completedQuoteNeedsHistoricalRevalidation/);
  assert.match(clientSource, /closedBarRevalidationDelayMs/);
  assert.match(clientSource, /MAX_CLOSED_BAR_REVALIDATION_ATTEMPTS/);
  assert.match(clientSource, /data-finalization-status/);
  assert.match(clientSource, /data-historical-gap/);
  assert.match(clientSource, /data-calendar-continuity/);
  assert.match(clientSource, /fetchVerifiedClosedMarketDays/);
  assert.match(clientSource, /fetchInstrumentDayReadiness/);
  assert.match(clientSource, /const historicalRevalidationDate = cachedAShareHistoryRevalidating[\s\S]{0,260}suspectedGapEnd/);
  assert.match(clientSource, /readiness && !readiness\.chartReady/);
  assert.match(clientSource, /data-stock-readiness/);
  assert.match(clientSource, /data-market-readiness/);
  assert.match(clientSource, /const cachedAShareHistoryRevalidating = stock\.marketRegion === "cn"[\s\S]{0,160}cacheStatus === "browser-cache"/);
  assert.match(clientSource, /const visibleHistoricalGap = historicalGap[\s\S]{0,100}!cachedAShareHistoryRevalidating/);
  assert.match(clientSource, /const historicalRevalidationDate = cachedAShareHistoryRevalidating/);
  assert.match(clientSource, /revalidating-cache/);
  assert.match(clientSource, /正在刷新日线快照/);
  assert.match(clientSource, /当前先显示本机缓存，服务器最新快照返回后会自动替换/);
  assert.match(clientSource, /!cachedAShareHistoryRevalidating && calendarContinuityChecking/);
  assert.match(clientSource, /正在核对休市日/);
  assert.match(clientSource, /日线数据存在缺口/);
  assert.match(clientSource, /图表不会跨日拼接，正在自动补齐/);
  assert.match(clientSource, /当前实时 K 线已单独显示；缺失的已收盘日线仍需后台补齐/);
  assert.match(clientSource, /stock\.marketRegion !== "cn"/);
  assert.match(clientSource, /const adjustmentFactor = historicalGap[\s\S]{0,30}\? 1/);
  assert.match(clientSource, /const latestCandleNote = latest\.unconfirmed/);
  assert.match(clientSource, /盘中均线按实时价更新 · 收盘后固定/);
  assert.match(clientSource, /收盘待历史确认/);
  assert.match(clientSource, /className="chart-heading"[\s\S]{0,200}\{latestCandleNote/);
  assert.doesNotMatch(clientSource, /className="unconfirmed-label"/);
});

test("keeps the page focused on live market-viewing content", async () => {
  const clientSource = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  assert.doesNotMatch(clientSource, /把“主力吸货”拆成|日线级别 · 技术研究|一次可信的承接|研究边界|Daily research view/);
  assert.doesNotMatch(clientSource, /provenance-strip|行情数据血缘|正式投影/);
  assert.doesNotMatch(clientSource, /className="hero|className="method-card|className="disclaimer/);
  assert.match(clientSource, /className="workspace"[\s\S]{0,160}id="top"/);
  assert.match(clientSource, /className="factor-grid"/);
});

test("connects realtime quotes without treating the intraday candle as confirmed", async () => {
  const clientSource = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  const styles = await readFile(new URL("../app/globals.css", import.meta.url), "utf8");
  const realtimeSource = await readFile(new URL("../app/use-realtime-quote.ts", import.meta.url), "utf8");
  assert.match(clientSource, /useRealtimeQuote/);
  assert.match(clientSource, /data-realtime-event-id/);
  assert.match(clientSource, /realtime-candle/);
  assert.match(clientSource, /open:\s*quote\.open/);
  assert.match(clientSource, /high:\s*quote\.high/);
  assert.match(clientSource, /low:\s*quote\.low/);
  assert.match(clientSource, /close:\s*quote\.last/);
  assert.match(clientSource, /sourceId:\s*quote\.provider/);
  assert.match(clientSource, /currentPriceY\s*=\s*hasRealtimeCandle\s*\?\s*yPrice\(latest\.close\)/);
  assert.match(clientSource, /className="current-price-line"/);
  assert.match(styles, /\.current-price-line\s*\{[^}]*stroke-dasharray:/);
  assert.doesNotMatch(clientSource, /renderedCandleWidth|realtime-candle-label/);
  assert.doesNotMatch(styles, /realtime-session-band|current-price-marker|realtime-candle-label|candle\.realtime-candle/);
  assert.match(clientSource, /formalIndicatorIndexes/);
  assert.match(clientSource, /正式指标/);
  assert.match(clientSource, /盘中实时蜡烛仅用于观察，正式指标继续使用已完成日线/);
  assert.match(realtimeSource, /type:\s*"subscribe"/);
  assert.match(realtimeSource, /cache:\s*"no-store"/);
  assert.match(realtimeSource, /scheduleReconnect/);
  assert.match(realtimeSource, /startPolling/);
  assert.match(realtimeSource, /enterMiddayPause/);
  assert.match(realtimeSource, /午间休市 · 13:00 自动恢复/);
  assert.match(realtimeSource, /\/api\/realtime-quote/);
  assert.match(realtimeSource, /status:\s*"polling"/);
  assert.match(realtimeSource, /夜盘一秒推送/);
  assert.doesNotMatch(realtimeSource, /Math\.random/);
});

test("starts active realtime in parallel and renders a coherent last candle", async () => {
  const layoutSource = await readFile(new URL("../app/layout.tsx", import.meta.url), "utf8");
  const clientSource = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  const realtimeSource = await readFile(
    new URL("../app/use-realtime-quote.ts", import.meta.url),
    "utf8",
  );

  assert.match(
    clientSource,
    /providerSupportsRealtimeInstrument\(realtimeProviderName, stock\)/,
  );
  assert.match(clientSource, /useRealtimeQuote\(selectedRealtimeInstrument, realtimeEnabled\)/);
  assert.match(realtimeSource, /const primeSnapshot = async \(\)/);
  assert.match(realtimeSource, /void primeSnapshot\(\);\s*void connect\(\);/);
  assert.match(realtimeSource, /statusForQuote\(selected\)/);
  assert.match(realtimeSource, /sessionStorage\.getItem/);
  assert.match(realtimeSource, /sessionStorage\.setItem/);
  assert.match(realtimeSource, /REALTIME_QUOTE_CACHE_MAX_AGE_MS\s*=\s*5 \* 60 \* 1_000/);
  assert.match(realtimeSource, /return \{ \.\.\.quote, completeness: "stale" \}/);
  assert.match(layoutSource, /\/api\/realtime-quote\?instrument_id=cn\.xshg\.688008/);
  assert.match(clientSource, /prefetchRealtimeQuotes/);
  assert.match(clientSource, /requestIdleCallback/);
  assert.doesNotMatch(clientSource, /\.slice\(0, 8\)/);
  assert.match(clientSource, /WATCHLIST_REFRESH_INTERVAL_MS\s*=\s*15_000/);
  assert.match(clientSource, /realtimeMiddayState\(instrument\.market, now\)\.paused/);
  assert.match(clientSource, /controller\.abort\(\)/);
  assert.match(clientSource, /REALTIME_COHERENCE_GRACE_MS\s*=\s*2_000/);
  assert.match(clientSource, /waitingForCoherentRealtime/);
  assert.match(clientSource, /waitingForCachedHistoryContinuity/);
  assert.match(realtimeSource, /export async function prefetchRealtimeQuotes/);
  assert.match(realtimeSource, /Promise<RealtimeQuote\[\]>/);
  assert.match(realtimeSource, /requestRealtimeBatch/);
  assert.match(realtimeSource, /\/api\/realtime-quotes/);
  assert.match(realtimeSource, /MAX_BATCH_INSTRUMENTS\s*=\s*500/);
  assert.match(realtimeSource, /MAX_PREFETCH_CONCURRENCY\s*=\s*3/);
  assert.match(realtimeSource, /signal\?\.aborted/);
  assert.match(realtimeSource, /inflightRealtimeSnapshots/);
});

test("renders provisional moving averages with the live candle while formal indicators stay closed-only", async () => {
  const clientSource = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  const {
    displayMovingAverageValues,
    selectCompletedSeries,
  } = await import("../app/market-series.ts");
  const history = [1, 2, 3, 4, 5].map((close, index) => ({
    date: `2026-08-${String(index + 10).padStart(2, "0")}`,
    close,
  }));
  const live = {
    date: "2026-08-15",
    close: 10,
    realtime: true,
    unconfirmed: true,
  };
  const display = [...history, live];

  assert.deepEqual(displayMovingAverageValues(display, 5), [null, null, null, null, 3, 4.8]);
  assert.equal(selectCompletedSeries(history, display), history);
  assert.match(clientSource, /displayMovingAverageValues\(allData, movingAveragePeriod\)/);
  assert.match(clientSource, /盘中均线按实时价更新 · 收盘后固定/);
});

test("shows the active market-data provider without restoring the removed provenance strip", async () => {
  const clientSource = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  const providerSource = await readFile(new URL("../app/use-realtime-provider.ts", import.meta.url), "utf8");
  const styles = await readFile(new URL("../app/globals.css", import.meta.url), "utf8");
  assert.match(clientSource, /className="provider-details"/);
  assert.match(clientSource, /腾讯 · 开发源/);
  assert.match(clientSource, /长桥 · 个人授权/);
  assert.match(clientSource, /provider \?\? inferredProvider\(null\)/);
  assert.match(clientSource, /不代表长桥账号已经连接/);
  assert.match(providerSource, /personal_oauth_pilot/);
  assert.match(providerSource, /maximumSubscribedSymbols:\s*500/);
  assert.match(styles, /\.provider-panel/);
  assert.doesNotMatch(clientSource, /provenance-strip|行情数据血缘/);
});

test("removes the right rail from layout when collapsed", async () => {
  const clientSource = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  const styles = await readFile(new URL("../app/globals.css", import.meta.url), "utf8");
  assert.match(clientSource, /\[rightRailCollapsed, setRightRailCollapsed\]\s*=\s*useState\(true\)/);
  assert.match(clientSource, /className="rail-toggle topbar-right-rail-toggle"/);
  assert.match(clientSource, /<aside className="right-rail"[^>]*hidden=\{collapsed\}/);
  assert.doesNotMatch(clientSource, /rightRailScrollHidden|scrollHidden|directionDistance|requestAnimationFrame\(updateVisibility\)/);
  assert.match(styles, /\.app-layout\.right-collapsed\s*\{[^}]*grid-template-columns:\s*240px minmax\(0, 1fr\) 0/);
  assert.match(styles, /\.app-layout\.left-collapsed\.right-collapsed\s*\{[^}]*grid-template-columns:\s*68px minmax\(0, 1fr\) 0/);
  assert.doesNotMatch(styles, /\.right-rail\.collapsed|scroll-hidden/);
});
