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

function environmentString(name: string): string | null {
  const value = Reflect.get(env, name);
  return typeof value === "string" && value.trim() ? value.trim() : null;
}
