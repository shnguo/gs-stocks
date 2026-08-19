import assert from "node:assert/strict";
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
  assert.match(html, /每日价格走势/);
  assert.match(html, /低位承接强度/);
  assert.match(html, /导入 CSV/);
  assert.doesNotMatch(html, /codex-preview|Building your site|SkeletonPreview/);
});

test("emits finished site metadata", async () => {
  const response = await render();
  const html = await response.text();
  assert.match(html, /<html[^>]+lang="zh-CN"/i);
  assert.match(html, /<title>低位承接研究台｜蜡烛图与承接强度<\/title>/i);
  assert.match(html, /日线蜡烛图与低位承接强度指标/);
});
