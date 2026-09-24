import { spawn } from "node:child_process";
import http from "node:http";
import { SocksProxyAgent } from "socks-proxy-agent";
import { WebSocket, WebSocketServer } from "ws";

import { closeWebSocketSafely } from "./websocket-close.mjs";
import {
  localRealtimePollMilliseconds,
  parseLocalRealtimeTicketRequest,
  serveLocalRealtimeWebSocket,
} from "./local-realtime-websocket.mjs";
import {
  curlResponseWriteOut,
  parseCurlResponse,
  previewProxyRequestHasBody,
  previewProxyRequestKind,
} from "./preview-proxy-response.mjs";

const token = process.env.MOOTDX_DATA_API_BEARER_TOKEN?.trim();
const upstreamBase = (
  process.env.MOOTDX_PROXY_UPSTREAM_URL ??
  "https://mootdx-cf-collector-preview-deployment-01.tap2rap.workers.dev"
).replace(/\/$/, "");
const socks5Proxy = process.env.MOOTDX_SOCKS5_PROXY ?? "127.0.0.1:1080";
const port = Number(process.env.MOOTDX_LOCAL_PROXY_PORT ?? 3101);
const browserOrigin = process.env.MOOTDX_BROWSER_ORIGIN ?? "http://localhost:3000";
const localRealtimeUpstream = process.env.MOOTDX_LOCAL_REALTIME_UPSTREAM_URL
  ?.trim().replace(/\/$/u, "") || null;
const localRealtimePollMs = localRealtimePollMilliseconds(
  process.env.MOOTDX_LOCAL_REALTIME_POLL_INTERVAL_MS,
);
const startedAt = new Date().toISOString();
const localRealtimeTickets = new Map();

if (!token || token.length < 32) {
  throw new Error("MOOTDX_DATA_API_BEARER_TOKEN is required");
}

const server = http.createServer((request, response) => {
  const requestUrl = new URL(request.url ?? "/", "http://127.0.0.1");
  if (request.method === "GET" && requestUrl.pathname === "/healthz") {
    response.writeHead(200, {
      "content-type": "application/json",
      "cache-control": "no-store",
    });
    response.end(JSON.stringify({
      status: "ok",
      started_at: startedAt,
      upstream_origin: new URL(upstreamBase).origin,
      websocket_bridge: "enabled",
      local_realtime_websocket: localRealtimeUpstream ? "enabled" : "disabled",
      local_realtime_poll_ms: localRealtimeUpstream ? localRealtimePollMs : null,
    }));
    return;
  }
  const authorization = request.headers.authorization ?? "";
  const requestMethod = request.method ?? "GET";
  const requestKind = previewProxyRequestKind(requestMethod, requestUrl.pathname);
  if (
    requestKind === null ||
    authorization !== `Bearer ${token}`
  ) {
    response.writeHead(404, { "content-type": "application/json" });
    response.end(JSON.stringify({ error: "not found" }));
    return;
  }

  if (requestKind === "ticket" && localRealtimeUpstream) {
    const encodedInstrumentId = requestUrl.pathname.slice("/v1/realtime/tickets/".length);
    let instrumentId = "";
    try {
      instrumentId = decodeURIComponent(encodedInstrumentId);
    } catch {
      instrumentId = "";
    }
    const mapping = parseLocalRealtimeTicketRequest(requestUrl, instrumentId);
    if (!mapping) {
      response.writeHead(400, { "content-type": "application/json" });
      response.end(JSON.stringify({ error: "invalid local realtime instrument mapping" }));
      return;
    }
    const now = Date.now();
    for (const [ticket, value] of localRealtimeTickets) {
      if (value.expiresAt <= now) localRealtimeTickets.delete(ticket);
    }
    const ticket = crypto.randomUUID();
    const expiresAt = now + 30_000;
    localRealtimeTickets.set(ticket, { ...mapping, expiresAt });
    const websocketUrl = new URL(
      `/v1/realtime/connect/${encodeURIComponent(instrumentId)}`,
      `ws://127.0.0.1:${port}`,
    );
    websocketUrl.searchParams.set("ticket", ticket);
    response.writeHead(200, {
      "content-type": "application/json",
      "cache-control": "private, no-store, max-age=0",
    });
    response.end(JSON.stringify({
      contract_version: 1,
      instrument_id: instrumentId,
      expires_at: new Date(expiresAt).toISOString(),
      websocket_url: websocketUrl.toString(),
    }));
    return;
  }

  const useLocalRealtime = Boolean(localRealtimeUpstream) &&
    ["provider", "realtime", "realtime-batch"].includes(requestKind);
  const selectedUpstreamBase = useLocalRealtime ? localRealtimeUpstream : upstreamBase;
  const upstreamUrl = `${selectedUpstreamBase}${requestUrl.pathname}${requestUrl.search}`;
  const curlArguments = [
    "--silent",
    "--show-error",
    "--max-time",
    "33",
    "--config",
    "/dev/fd/3",
    "--write-out",
    curlResponseWriteOut(),
  ];
  if (!useLocalRealtime) {
    curlArguments.splice(4, 0, "--proxy", `socks5h://${socks5Proxy}`);
  }
  curlArguments.push("--request", requestMethod);
  const forwardsBody = previewProxyRequestHasBody(requestKind, requestMethod);
  if (forwardsBody) curlArguments.push("--data-binary", "@-");
  curlArguments.push(upstreamUrl);
  const child = spawn("curl", curlArguments, { stdio: ["pipe", "pipe", "pipe", "pipe"] });
  const realtimeSubject = request.headers["x-mootdx-realtime-subject"];
  const safeRealtimeSubject = typeof realtimeSubject === "string" &&
      /^[A-Za-z0-9._:-]{1,128}$/u.test(realtimeSubject)
    ? realtimeSubject
    : null;
  const delegatedSubject = request.headers["x-mootdx-subject"];
  const safeDelegatedSubject = typeof delegatedSubject === "string" &&
      /^[A-Za-z0-9._:-]{1,128}$/u.test(delegatedSubject)
    ? delegatedSubject
    : null;
  const headers = [
    `header = "authorization: Bearer ${token}"`,
    "header = \"accept: application/json\"",
  ];
  if (requestKind === "ticket" && safeRealtimeSubject) {
    headers.push(`header = "x-mootdx-realtime-subject: ${safeRealtimeSubject}"`);
  }
  if ((requestKind === "admission" || requestKind === "favorites") && safeDelegatedSubject) {
    headers.push(`header = "x-mootdx-subject: ${safeDelegatedSubject}"`);
  }
  if (forwardsBody) {
    headers.push("header = \"content-type: application/json\"");
  }
  child.stdio[3].end(`${headers.join("\n")}\n`);

  if (forwardsBody) {
    let requestBytes = 0;
    const maximumRequestBytes = requestKind === "realtime-batch" ? 131_072 : 8_192;
    request.on("data", (chunk) => {
      requestBytes += chunk.length;
      if (requestBytes > maximumRequestBytes) {
        child.kill("SIGTERM");
        if (!response.headersSent) {
          response.writeHead(413, { "content-type": "application/json" });
          response.end(JSON.stringify({ error: "request body is too large" }));
        }
        return;
      }
      if (!child.stdin.destroyed) child.stdin.write(chunk);
    });
    request.on("end", () => {
      if (!child.stdin.destroyed) child.stdin.end();
    });
  } else {
    child.stdin.end();
  }

  const stdout = [];
  const stderr = [];
  let bytes = 0;
  child.stdout.on("data", (chunk) => {
    bytes += chunk.length;
    if (bytes <= 8_000_000) stdout.push(chunk);
    else child.kill("SIGTERM");
  });
  child.stderr.on("data", (chunk) => stderr.push(chunk));
  child.on("close", (code) => {
    if (response.writableEnded) return;
    const rendered = Buffer.concat(stdout).toString("utf8");
    const parsed = parseCurlResponse(rendered);
    if (code !== 0 || parsed === null) {
      logEvent("preview-proxy-request-failed", {
        pathname: requestUrl.pathname,
        exit_code: code,
        detail: Buffer.concat(stderr).toString("utf8").slice(0, 240),
      });
      response.writeHead(502, { "content-type": "application/json" });
      response.end(JSON.stringify({
        error: "preview proxy request failed",
        detail: Buffer.concat(stderr).toString("utf8").slice(0, 240),
      }));
      return;
    }
    let clientBody = parsed.body;
    if (requestKind === "ticket" && parsed.status >= 200 && parsed.status < 300) {
      try {
        const ticketPayload = JSON.parse(parsed.body);
        const websocketUrl = new URL(ticketPayload.websocket_url);
        websocketUrl.protocol = "ws:";
        websocketUrl.hostname = "127.0.0.1";
        websocketUrl.port = String(port);
        ticketPayload.websocket_url = websocketUrl.toString();
        clientBody = JSON.stringify(ticketPayload);
      } catch {
        response.writeHead(502, { "content-type": "application/json" });
        response.end(JSON.stringify({ error: "preview proxy received an invalid ticket response" }));
        return;
      }
    }
    response.writeHead(parsed.status, {
      ...parsed.headers,
      "content-type": parsed.headers["content-type"] ?? "application/json",
      "cache-control": parsed.headers["cache-control"] ?? "no-store",
    });
    response.end(clientBody);
  });
  request.on("aborted", () => {
    if (!child.killed && child.exitCode === null) child.kill("SIGTERM");
  });
  response.on("close", () => {
    if (!response.writableEnded && !child.killed && child.exitCode === null) child.kill("SIGTERM");
  });
});

const websocketServer = new WebSocketServer({ noServer: true });
const socksAgent = new SocksProxyAgent(`socks5h://${socks5Proxy}`);

server.on("upgrade", (request, socket, head) => {
  const requestUrl = new URL(request.url ?? "/", `http://127.0.0.1:${port}`);
  if (
    !requestUrl.pathname.startsWith("/v1/realtime/connect/") ||
    !requestUrl.searchParams.get("ticket")
  ) {
    socket.write("HTTP/1.1 404 Not Found\r\nConnection: close\r\n\r\n");
    socket.destroy();
    return;
  }
  if (localRealtimeUpstream) {
    const ticketId = requestUrl.searchParams.get("ticket");
    const ticket = ticketId ? localRealtimeTickets.get(ticketId) : null;
    let instrumentId = "";
    try {
      instrumentId = decodeURIComponent(
        requestUrl.pathname.slice("/v1/realtime/connect/".length),
      );
    } catch {
      instrumentId = "";
    }
    if (
      !ticketId || !ticket || ticket.expiresAt <= Date.now() ||
      ticket.instrumentId !== instrumentId || request.headers.origin !== browserOrigin
    ) {
      if (ticketId) localRealtimeTickets.delete(ticketId);
      socket.write("HTTP/1.1 401 Unauthorized\r\nConnection: close\r\n\r\n");
      socket.destroy();
      return;
    }
    localRealtimeTickets.delete(ticketId);
    websocketServer.handleUpgrade(request, socket, head, (client) => {
      serveLocalRealtimeWebSocket(client, ticket, {
        upstreamBase: localRealtimeUpstream,
        pollMilliseconds: localRealtimePollMs,
        log: logEvent,
      });
    });
    return;
  }
  const upstreamWebsocketUrl = new URL(`${upstreamBase}${requestUrl.pathname}${requestUrl.search}`);
  upstreamWebsocketUrl.protocol = "wss:";
  const upstream = new WebSocket(upstreamWebsocketUrl, {
    agent: socksAgent,
    origin: browserOrigin,
  });
  const failUpgrade = () => {
    if (!socket.destroyed) {
      socket.write("HTTP/1.1 502 Bad Gateway\r\nConnection: close\r\n\r\n");
      socket.destroy();
    }
  };
  upstream.once("error", failUpgrade);
  upstream.once("open", () => {
    upstream.off("error", failUpgrade);
    websocketServer.handleUpgrade(request, socket, head, (client) => {
      bridgeWebSockets(client, upstream);
    });
  });
});

function bridgeWebSockets(client, upstream) {
  client.on("message", (data, isBinary) => {
    if (upstream.readyState === WebSocket.OPEN) upstream.send(data, { binary: isBinary });
  });
  upstream.on("message", (data, isBinary) => {
    if (client.readyState === WebSocket.OPEN) client.send(data, { binary: isBinary });
  });
  client.on("close", (code, reason) => {
    closeWebSocketSafely(upstream, code, reason, (error) => {
      logEvent("preview-proxy-close-forward-failed", {
        direction: "client-to-upstream",
        error: error instanceof Error ? error.message : String(error),
      });
    });
  });
  upstream.on("close", (code, reason) => {
    closeWebSocketSafely(client, code, reason, (error) => {
      logEvent("preview-proxy-close-forward-failed", {
        direction: "upstream-to-client",
        error: error instanceof Error ? error.message : String(error),
      });
    });
  });
  client.on("error", () => closeWebSocketSafely(upstream, 1011, "local bridge error"));
  upstream.on("error", () => closeWebSocketSafely(client, 1011, "upstream bridge error"));
}

function logEvent(event, fields = {}) {
  process.stderr.write(`${JSON.stringify({ event, ...fields })}\n`);
}

server.listen(port, "127.0.0.1", () => {
  process.stdout.write(`${JSON.stringify({
    event: "preview-proxy-started",
    listen: `http://127.0.0.1:${port}`,
    health: `http://127.0.0.1:${port}/healthz`,
    started_at: startedAt,
  })}\n`);
});
