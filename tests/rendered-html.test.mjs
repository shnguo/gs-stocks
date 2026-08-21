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
  assert.match(html, /正在读取真实日线/);
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
  assert.match(html, /右侧预留区域/);
  assert.match(html, /右侧区域已预留/);
  assert.match(html, /收起左侧栏/);
  assert.match(html, /收起右侧栏/);
});

test("keeps the market data credential on the server", async () => {
  const clientSource = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  const routeSource = await readFile(new URL("../app/api/chart-daily-bars/route.ts", import.meta.url), "utf8");
  const ticketRouteSource = await readFile(new URL("../app/api/realtime-ticket/route.ts", import.meta.url), "utf8");
  const environmentSource = await readFile(new URL("../app/lib/market-data-env.ts", import.meta.url), "utf8");
  assert.doesNotMatch(clientSource, /MOOTDX_DATA_API_BEARER_TOKEN|authorization:\s*`Bearer/);
  assert.doesNotMatch(ticketRouteSource, /MOOTDX_DATA_API_BEARER_TOKEN/);
  assert.match(environmentSource, /MOOTDX_DATA_API_BEARER_TOKEN/);
  assert.match(routeSource, /authorization:\s*`Bearer/);
  assert.match(ticketRouteSource, /authorization:\s*`Bearer/);
  assert.match(routeSource, /mode:\s*"latest"/);
  assert.match(routeSource, /window_days:\s*"400"/);
  assert.doesNotMatch(routeSource, /start_date:/);
  assert.doesNotMatch(routeSource, /end_date:/);
  assert.match(routeSource, /if-none-match/);
  assert.doesNotMatch(routeSource, /cache:\s*"no-store"/);
});

test("renders the requested moving-average set on the price chart", async () => {
  const clientSource = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  assert.match(clientSource, /MOVING_AVERAGE_PERIODS\s*=\s*\[5, 10, 20, 30, 60\]/);
  assert.match(clientSource, /fullWindowMovingAverage/);
  assert.match(clientSource, /MA5、MA10、MA20、MA30和MA60均线/);
  assert.match(clientSource, /ma-line-\$\{movingAveragePeriod\}/);
  assert.match(clientSource, /legend-value-\$\{movingAveragePeriod\}/);
  assert.match(clientSource, /formatNumber\(values\[activeIndex\]\)/);
});

test("keeps the page focused on live market-viewing content", async () => {
  const clientSource = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  assert.doesNotMatch(clientSource, /把“主力吸货”拆成|日线级别 · 技术研究|一次可信的承接|研究边界|Daily research view/);
  assert.doesNotMatch(clientSource, /className="hero|className="method-card|className="disclaimer/);
  assert.match(clientSource, /className="workspace"[\s\S]{0,160}id="top"/);
  assert.match(clientSource, /className="factor-grid"/);
});

test("connects realtime quotes without treating the intraday candle as confirmed", async () => {
  const clientSource = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  const realtimeSource = await readFile(new URL("../app/use-realtime-quote.ts", import.meta.url), "utf8");
  assert.match(clientSource, /useRealtimeQuote/);
  assert.match(clientSource, /data-realtime-event-id/);
  assert.match(clientSource, /realtime-candle/);
  assert.match(clientSource, /盘中实时蜡烛仅用于观察，正式指标继续使用已完成日线/);
  assert.match(realtimeSource, /type:\s*"subscribe"/);
  assert.match(realtimeSource, /cache:\s*"no-store"/);
  assert.match(realtimeSource, /scheduleReconnect/);
  assert.doesNotMatch(realtimeSource, /Math\.random/);
});
