export type AShareExchange = "xshg" | "xshe" | "xbse";
export type HongKongExchange = "xhkg";
export type UnitedStatesExchange = "xarc" | "xnas" | "xnys";
export type SupportedExchange = AShareExchange | HongKongExchange | UnitedStatesExchange;
export type MarketRegion = "cn" | "hk" | "us";
export type MarketGroup = "A股" | "港股" | "美股";
export type AdjustmentBasis = "none" | "qfq" | "hfq";

export type MarketInstrumentIdentity = {
  code: string;
  exchangeId: SupportedExchange;
  instrumentId: string;
  market: string;
  marketGroup: MarketGroup;
  marketRegion: MarketRegion;
  currency: "CNY" | "HKD" | "USD";
  availableAdjustmentBases: AdjustmentBasis[];
};

export type RealtimeMarketInstrument = {
  instrumentId: string;
  market: MarketRegion;
  providerSymbol: string;
  currency: "CNY" | "HKD" | "USD";
};

export type AShareIdentity = MarketInstrumentIdentity & {
  exchangeId: AShareExchange;
  marketGroup: "A股";
  marketRegion: "cn";
  currency: "CNY";
};

export function resolveAShareSymbol(rawSymbol: string): AShareIdentity | null {
  const code = rawSymbol.trim();
  if (!/^\d{6}$/u.test(code)) return null;

  const exchangeId: AShareExchange = code.startsWith("920") || /^[48]/u.test(code)
    ? "xbse"
    : /^[569]/u.test(code)
      ? "xshg"
      : "xshe";

  return {
    code,
    exchangeId,
    instrumentId: `cn.${exchangeId}.${code}`,
    market: marketLabel(exchangeId, code),
    marketGroup: "A股",
    marketRegion: "cn",
    currency: "CNY",
    availableAdjustmentBases: ["none", "qfq", "hfq"],
  };
}

export function marketInstrumentFromCatalog(value: unknown): MarketInstrumentIdentity | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const record = value as Record<string, unknown>;
  const marketRegion = record.market;
  const exchangeId = record.exchange_id;
  const instrumentId = record.instrument_id;
  const rawSymbol = record.symbol;
  const currency = record.currency;
  if (
    (marketRegion !== "hk" && marketRegion !== "us") ||
    typeof exchangeId !== "string" ||
    typeof instrumentId !== "string" ||
    typeof rawSymbol !== "string" ||
    !isSupportedMarketInstrumentId(instrumentId)
  ) return null;

  const code = rawSymbol.trim().toUpperCase();
  if (marketRegion === "hk") {
    if (exchangeId !== "xhkg" || currency !== "HKD" || !/^\d{5}$/u.test(code)) return null;
  } else if (
    !["xarc", "xnas", "xnys"].includes(exchangeId) ||
    currency !== "USD" ||
    !/^[A-Z][A-Z0-9._-]{0,15}$/u.test(code)
  ) return null;

  const availableAdjustmentBases: AdjustmentBasis[] =
    Array.isArray(record.available_adjustment_bases)
      ? record.available_adjustment_bases.filter(isAdjustmentBasis)
      : ["none"];
  if (!availableAdjustmentBases.includes("none")) availableAdjustmentBases.unshift("none");
  return {
    code,
    exchangeId: exchangeId as SupportedExchange,
    instrumentId,
    market: marketLabel(exchangeId as SupportedExchange, code),
    marketGroup: marketRegion === "hk" ? "港股" : "美股",
    marketRegion,
    currency,
    availableAdjustmentBases,
  };
}

export function savedMarketInstrument(value: unknown): MarketInstrumentIdentity | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const record = value as Record<string, unknown>;
  if (
    typeof record.code === "string" &&
    (record.marketRegion === "cn" || record.marketGroup === "A股" ||
      (typeof record.instrumentId === "string" && record.instrumentId.startsWith("cn.")))
  ) {
    const identity = resolveAShareSymbol(record.code);
    return identity && record.instrumentId === identity.instrumentId ? identity : null;
  }
  return marketInstrumentFromCatalog({
    market: record.marketRegion,
    exchange_id: record.exchangeId,
    instrument_id: record.instrumentId,
    symbol: record.code,
    currency: record.currency,
    available_adjustment_bases: record.availableAdjustmentBases,
  });
}

export function isSupportedAShareInstrumentId(instrumentId: string): boolean {
  const match = /^cn\.(xshg|xshe|xbse)\.(\d{6})$/u.exec(instrumentId);
  if (!match) return false;
  return resolveAShareSymbol(match[2])?.instrumentId === instrumentId;
}

export function isSupportedMarketInstrumentId(instrumentId: string): boolean {
  return isSupportedAShareInstrumentId(instrumentId) ||
    /^hk\.xhkg\.\d{5}$/u.test(instrumentId) ||
    /^us\.(?:xarc|xnas|xnys)\.[a-z0-9][a-z0-9._-]{0,31}$/u.test(instrumentId) ||
    /^instrument\.[a-z0-9][a-z0-9._-]{1,117}$/u.test(instrumentId);
}

export function realtimeMarketInstrument(
  identity: MarketInstrumentIdentity,
): RealtimeMarketInstrument {
  const providerSymbol = identity.marketRegion === "us"
    ? `${identity.code}.US`
    : identity.marketRegion === "hk"
      ? `${identity.code.replace(/^0+/u, "") || "0"}.HK`
      : identity.exchangeId === "xshg"
        ? `${identity.code}.SH`
        : identity.exchangeId === "xshe"
          ? `${identity.code}.SZ`
          : `${identity.code}.BJ`;
  return {
    instrumentId: identity.instrumentId,
    market: identity.marketRegion,
    providerSymbol,
    currency: identity.currency,
  };
}

export function providerSupportsRealtimeInstrument(
  provider: string,
  identity: MarketInstrumentIdentity,
): boolean {
  if (provider === "longbridge") {
    return identity.marketRegion !== "cn" || identity.exchangeId !== "xbse";
  }
  if (provider === "tencent") return identity.marketRegion === "cn";
  return false;
}

function marketLabel(exchangeId: SupportedExchange, code: string): string {
  if (exchangeId === "xhkg") return "港交所";
  if (exchangeId === "xarc") return "NYSE Arca";
  if (exchangeId === "xnas") return "纳斯达克";
  if (exchangeId === "xnys") return "纽交所";
  if (exchangeId === "xbse") return "北交所";
  if (exchangeId === "xshg") return code.startsWith("688") ? "科创板" : "沪市";
  return /^30[01]/u.test(code) ? "创业板" : "深市";
}

function isAdjustmentBasis(value: unknown): value is AdjustmentBasis {
  return value === "none" || value === "qfq" || value === "hfq";
}
