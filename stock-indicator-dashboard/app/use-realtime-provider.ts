"use client";

import { useEffect, useState } from "react";

export type RealtimeProviderInfo = {
  provider: string;
  displayName: string;
  connectionStatus: string;
  role: string;
  authentication: string;
  accountScope: string;
  credentialPersistence: string;
  maximumSubscribedSymbols: number | null;
  productionEligible: boolean;
  redistributionStatus: string;
};

type ProviderState = {
  status: "loading" | "ready" | "unavailable";
  info: RealtimeProviderInfo | null;
};

export function useRealtimeProvider(enabled: boolean): ProviderState {
  const [state, setState] = useState<ProviderState>({ status: "loading", info: null });

  useEffect(() => {
    if (!enabled) return;
    const controller = new AbortController();
    fetch("/api/realtime-provider", {
      cache: "no-store",
      signal: controller.signal,
    })
      .then(async (response) => {
        const payload: unknown = await response.json().catch(() => null);
        if (!response.ok || !isRecord(payload)) throw new Error("provider status unavailable");
        const info = parseProvider(payload);
        if (!info) throw new Error("invalid provider status");
        setState({ status: "ready", info });
      })
      .catch((error: unknown) => {
        if (error instanceof Error && error.name === "AbortError") return;
        setState({ status: "unavailable", info: null });
      });
    return () => controller.abort();
  }, [enabled]);

  return enabled ? state : { status: "ready", info: localCsvProvider() };
}

export function inferredProvider(provider: string | null): RealtimeProviderInfo {
  if (provider === "longbridge") {
    return {
      provider,
      displayName: "长桥 OpenAPI",
      connectionStatus: "ready",
      role: "personal_oauth_pilot",
      authentication: "oauth2_pkce",
      accountScope: "single_user",
      credentialPersistence: "longbridge_sdk_local_file",
      maximumSubscribedSymbols: 500,
      productionEligible: false,
      redistributionStatus: "personal_use_only_pending_written_approval",
    };
  }
  return {
    provider: provider ?? "tencent",
    displayName: provider ? provider : "腾讯行情",
    connectionStatus: "ready",
    role: "development_fallback",
    authentication: "none",
    accountScope: "shared_preview",
    credentialPersistence: "none",
    maximumSubscribedSymbols: null,
    productionEligible: false,
    redistributionStatus: "restricted_pending_review",
  };
}

function localCsvProvider(): RealtimeProviderInfo {
  return {
    provider: "local-csv",
    displayName: "本地 CSV",
    connectionStatus: "ready",
    role: "local_import",
    authentication: "none",
    accountScope: "local_user",
    credentialPersistence: "none",
    maximumSubscribedSymbols: null,
    productionEligible: false,
    redistributionStatus: "user_supplied",
  };
}

function parseProvider(value: Record<string, unknown>): RealtimeProviderInfo | null {
  if (
    typeof value.provider !== "string" ||
    typeof value.display_name !== "string" ||
    typeof value.connection_status !== "string" ||
    typeof value.role !== "string" ||
    typeof value.authentication !== "string" ||
    typeof value.account_scope !== "string" ||
    typeof value.credential_persistence !== "string" ||
    typeof value.production_eligible !== "boolean" ||
    typeof value.redistribution_status !== "string"
  ) {
    return null;
  }
  const maximum = value.maximum_subscribed_symbols;
  return {
    provider: value.provider,
    displayName: value.display_name,
    connectionStatus: value.connection_status,
    role: value.role,
    authentication: value.authentication,
    accountScope: value.account_scope,
    credentialPersistence: value.credential_persistence,
    maximumSubscribedSymbols: typeof maximum === "number" ? maximum : null,
    productionEligible: value.production_eligible,
    redistributionStatus: value.redistribution_status,
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
