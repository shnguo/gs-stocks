import { env } from "cloudflare:workers";

export const DEFAULT_MARKET_DATA_API_BASE_URL =
  "https://mootdx-cf-collector-preview-deployment-01.tap2rap.workers.dev";

export function marketDataApiBaseUrl(): string {
  return (
    environmentString("MOOTDX_DATA_API_BASE_URL") ?? DEFAULT_MARKET_DATA_API_BASE_URL
  ).replace(/\/$/u, "");
}

export function marketDataApiBearerToken(): string | null {
  return environmentString("MOOTDX_DATA_API_BEARER_TOKEN");
}

export function marketDataRealtimeApiBaseUrl(): string {
  return (
    environmentString("MOOTDX_REALTIME_API_BASE_URL") ?? marketDataApiBaseUrl()
  ).replace(/\/$/u, "");
}

export function marketDataRealtimeTransport(): "websocket" | "polling" {
  return environmentString("MOOTDX_REALTIME_TRANSPORT") === "polling"
    ? "polling"
    : "websocket";
}

export function marketDataApiIsLoopback(baseUrl = marketDataApiBaseUrl()): boolean {
  try {
    const hostname = new URL(baseUrl).hostname;
    return hostname === "127.0.0.1" || hostname === "localhost" || hostname === "::1";
  } catch {
    return false;
  }
}

function environmentString(name: string): string | null {
  const value = Reflect.get(env, name);
  return typeof value === "string" && value.trim() ? value.trim() : null;
}
