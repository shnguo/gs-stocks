import { spawn } from "node:child_process";
import http from "node:http";
import { SocksProxyAgent } from "socks-proxy-agent";
import { WebSocket, WebSocketServer } from "ws";

const token = process.env.MOOTDX_DATA_API_BEARER_TOKEN?.trim();
const upstreamBase = (
  process.env.MOOTDX_PROXY_UPSTREAM_URL ??
  "https://mootdx-cf-collector-preview-deployment-01.tap2rap.workers.dev"
).replace(/\/$/, "");
const socks5Proxy = process.env.MOOTDX_SOCKS5_PROXY ?? "127.0.0.1:1080";
const port = Number(process.env.MOOTDX_LOCAL_PROXY_PORT ?? 3101);
const browserOrigin = process.env.MOOTDX_BROWSER_ORIGIN ?? "http://localhost:3000";

if (!token || token.length < 32) {
  throw new Error("MOOTDX_DATA_API_BEARER_TOKEN is required");
}

const server = http.createServer((request, response) => {
  const requestUrl = new URL(request.url ?? "/", "http://127.0.0.1");
  const authorization = request.headers.authorization ?? "";
  const isChartRequest =
    request.method === "GET" && requestUrl.pathname.startsWith("/v1/chart/daily-bars/");
  const isTicketRequest =
    request.method === "POST" && requestUrl.pathname.startsWith("/v1/realtime/tickets/");
  if (
    (!isChartRequest && !isTicketRequest) ||
    authorization !== `Bearer ${token}`
  ) {
    response.writeHead(404, { "content-type": "application/json" });
    response.end(JSON.stringify({ error: "not found" }));
    return;
  }

  const upstreamUrl = `${upstreamBase}${requestUrl.pathname}${requestUrl.search}`;
  const curlArguments = [
    "--silent",
    "--show-error",
    "--max-time",
    "33",
    "--proxy",
    `socks5h://${socks5Proxy}`,
    "--config",
    "-",
    "--write-out",
    "\n%{http_code}",
  ];
  if (isTicketRequest) curlArguments.push("--request", "POST");
  curlArguments.push(upstreamUrl);
  const child = spawn("curl", curlArguments, { stdio: ["pipe", "pipe", "pipe"] });
  const subject = request.headers["x-mootdx-realtime-subject"];
  const safeSubject = typeof subject === "string" && /^[A-Za-z0-9._:-]{1,128}$/u.test(subject)
    ? subject
    : null;
  const headers = [
    `header = "authorization: Bearer ${token}"`,
    "header = \"accept: application/json\"",
  ];
  if (isTicketRequest && safeSubject) {
    headers.push(`header = "x-mootdx-realtime-subject: ${safeSubject}"`);
  }
  child.stdin.end(`${headers.join("\n")}\n`);

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
    const splitAt = rendered.lastIndexOf("\n");
    const body = splitAt >= 0 ? rendered.slice(0, splitAt) : rendered;
    const status = splitAt >= 0 ? Number(rendered.slice(splitAt + 1)) : 0;
    if (code !== 0 || !Number.isInteger(status) || status < 100) {
      response.writeHead(502, { "content-type": "application/json" });
      response.end(JSON.stringify({
        error: "preview proxy request failed",
        detail: Buffer.concat(stderr).toString("utf8").slice(0, 240),
      }));
      return;
    }
    let clientBody = body;
    if (isTicketRequest && status >= 200 && status < 300) {
      try {
        const ticketPayload = JSON.parse(body);
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
    response.writeHead(status, {
      "content-type": "application/json",
      "cache-control": "no-store",
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
    if (upstream.readyState < WebSocket.CLOSING) upstream.close(code, reason.toString());
  });
  upstream.on("close", (code, reason) => {
    if (client.readyState < WebSocket.CLOSING) client.close(code, reason.toString());
  });
  client.on("error", () => upstream.close(1011, "local bridge error"));
  upstream.on("error", () => client.close(1011, "upstream bridge error"));
}

server.listen(port, "127.0.0.1", () => {
  process.stdout.write(`mootdx preview proxy listening on http://127.0.0.1:${port}\n`);
});
