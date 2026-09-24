export const CURL_METADATA_PREFIX = "__MOOTDX_RESPONSE_META__";

export function previewProxyRequestKind(method, pathname) {
  if (method === "GET" && pathname.startsWith("/v1/chart/daily-bars/")) return "chart";
  if (method === "POST" && pathname.startsWith("/v1/realtime/tickets/")) return "ticket";
  if (method === "POST" && pathname === "/v1/realtime/quotes") return "realtime-batch";
  if (method === "GET" && pathname.startsWith("/v1/realtime/quotes/")) return "realtime";
  if (method === "GET" && pathname === "/v1/realtime/provider") return "provider";
  if (method === "GET" && pathname === "/v1/ashare/market-days") return "market-days";
  if (
    method === "GET" &&
    /^\/v1\/ashare\/market-days\/\d{4}-\d{2}-\d{2}\/instruments\/cn\.(?:xshg|xshe|xbse)\.\d{6}$/u.test(
      pathname,
    )
  ) return "market-day-readiness";
  if (method === "GET" && pathname === "/v1/market-catalog") return "catalog";
  if (
    method === "GET" &&
    (
      pathname === "/v1/acquisition-universes" ||
      /^\/v1\/acquisition-universes\/(?:cn|hk|us)\.[a-z0-9][a-z0-9-]{1,63}\/members$/u.test(
        pathname,
      )
    )
  ) return "acquisition-universes";
  if (
    (method === "GET" || method === "POST") &&
    pathname === "/v1/symbol-admissions"
  ) return "admission";
  if (
    method === "GET" &&
    /^\/v1\/symbol-admissions\/[0-9a-f]{32}$/u.test(pathname)
  ) return "admission";
  if (
    (method === "GET" || method === "PUT" || method === "DELETE") &&
    pathname === "/v1/favorites"
  ) return "favorites";
  return null;
}

export function previewProxyRequestHasBody(kind, method) {
  return (kind === "realtime-batch" && method === "POST") ||
    (kind === "admission" && method === "POST") ||
    (kind === "favorites" && method === "PUT");
}

const FORWARDED_HEADERS = [
  "cache-control",
  "content-type",
  "etag",
  "retry-after",
  "server-timing",
  "x-mootdx-as-of",
  "x-mootdx-cache",
  "x-mootdx-snapshot-refreshed-at",
  "x-mootdx-chart-generation",
  "x-mootdx-chart-business-date",
  "x-mootdx-adjustment-bases",
  "x-mootdx-stock-readiness",
  "x-mootdx-market-readiness",
];

export function curlResponseWriteOut() {
  const values = [
    "%{response_code}",
    ...FORWARDED_HEADERS.map((name) => `%header{${name}}`),
  ];
  return `\n${CURL_METADATA_PREFIX}${values.join("\t")}`;
}

export function parseCurlResponse(rendered) {
  const marker = `\n${CURL_METADATA_PREFIX}`;
  const splitAt = rendered.lastIndexOf(marker);
  if (splitAt < 0) return null;
  const body = rendered.slice(0, splitAt);
  const values = rendered.slice(splitAt + marker.length).split("\t");
  const status = Number(values.shift());
  if (!Number.isInteger(status) || status < 100 || status > 599) return null;
  const headers = Object.fromEntries(
    FORWARDED_HEADERS.flatMap((name, index) => {
      const value = safeHeaderValue(values[index] ?? "");
      return value ? [[name, value]] : [];
    }),
  );
  return { body, status, headers };
}

function safeHeaderValue(value) {
  return value.replace(/[\r\n]/gu, "").trim().slice(0, 2048);
}
