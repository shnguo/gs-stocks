import assert from "node:assert/strict";
import test from "node:test";

import {
  CURL_METADATA_PREFIX,
  curlResponseWriteOut,
  parseCurlResponse,
  previewProxyRequestHasBody,
  previewProxyRequestKind,
} from "../scripts/preview-proxy-response.mjs";

test("preview proxy admits only the dashboard market-data methods", () => {
  assert.equal(previewProxyRequestKind("GET", "/v1/market-catalog"), "catalog");
  assert.equal(
    previewProxyRequestKind("GET", "/v1/acquisition-universes"),
    "acquisition-universes",
  );
  assert.equal(
    previewProxyRequestKind("GET", "/v1/acquisition-universes/us.sp500/members"),
    "acquisition-universes",
  );
  assert.equal(
    previewProxyRequestKind("POST", "/v1/acquisition-universes/us.sp500/members"),
    null,
  );
  assert.equal(previewProxyRequestKind("POST", "/v1/symbol-admissions"), "admission");
  assert.equal(previewProxyRequestKind("POST", "/v1/realtime/quotes"), "realtime-batch");
  assert.equal(
    previewProxyRequestKind("GET", "/v1/symbol-admissions/0123456789abcdef0123456789abcdef"),
    "admission",
  );
  assert.equal(previewProxyRequestKind("PUT", "/v1/favorites"), "favorites");
  assert.equal(previewProxyRequestKind("DELETE", "/v1/favorites"), "favorites");
  assert.equal(previewProxyRequestKind("POST", "/v1/internal/symbol-admission-outbox/lease"), null);
  assert.equal(previewProxyRequestKind("POST", "/v1/market-catalog"), null);
  assert.equal(previewProxyRequestHasBody("admission", "POST"), true);
  assert.equal(previewProxyRequestHasBody("realtime-batch", "POST"), true);
  assert.equal(previewProxyRequestHasBody("favorites", "PUT"), true);
  assert.equal(previewProxyRequestHasBody("favorites", "DELETE"), false);
});

test("curl response metadata preserves snapshot repair headers", () => {
  const rendered = [
    JSON.stringify({ error: "chart snapshot is warming", retryable: true }),
    `${CURL_METADATA_PREFIX}503\tno-store\tapplication/json\t\t2\tmootdx;dur=1.2\t\tsnapshot-miss\t\t\t\t\t\t`,
  ].join("\n");
  assert.deepEqual(parseCurlResponse(rendered), {
    body: JSON.stringify({ error: "chart snapshot is warming", retryable: true }),
    status: 503,
    headers: {
      "cache-control": "no-store",
      "content-type": "application/json",
      "retry-after": "2",
      "server-timing": "mootdx;dur=1.2",
      "x-mootdx-cache": "snapshot-miss",
    },
  });
});

test("curl response metadata preserves immutable generation headers", () => {
  const rendered = [
    JSON.stringify({ rows: [] }),
    `${CURL_METADATA_PREFIX}200\tprivate, max-age=60\tapplication/json\t\t\tmootdx;dur=42\t2026-08-24T07:58:14Z\tgeneration-hit\t\tb433b27649e3285f8bf3eb27d4ee4f29\t2026-08-21\tnone\tready\tcomplete`,
  ].join("\n");
  assert.deepEqual(parseCurlResponse(rendered), {
    body: JSON.stringify({ rows: [] }),
    status: 200,
    headers: {
      "cache-control": "private, max-age=60",
      "content-type": "application/json",
      "server-timing": "mootdx;dur=42",
      "x-mootdx-as-of": "2026-08-24T07:58:14Z",
      "x-mootdx-cache": "generation-hit",
      "x-mootdx-chart-generation": "b433b27649e3285f8bf3eb27d4ee4f29",
      "x-mootdx-chart-business-date": "2026-08-21",
      "x-mootdx-adjustment-bases": "none",
      "x-mootdx-stock-readiness": "ready",
      "x-mootdx-market-readiness": "complete",
    },
  });
});

test("curl response metadata rejects missing or invalid status", () => {
  assert.equal(parseCurlResponse("{}\n200"), null);
  assert.equal(parseCurlResponse(`{}\n${CURL_METADATA_PREFIX}999`), null);
  assert.match(curlResponseWriteOut(), /%header\{x-mootdx-cache\}/u);
  assert.match(curlResponseWriteOut(), /%header\{x-mootdx-chart-generation\}/u);
  assert.match(curlResponseWriteOut(), /%header\{x-mootdx-stock-readiness\}/u);
  assert.match(curlResponseWriteOut(), /%header\{x-mootdx-market-readiness\}/u);
});
