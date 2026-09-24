export function isPlaceholderFavoriteName(name: string, code: string): boolean {
  const normalizedName = name.trim().replace(/\s+/gu, "");
  return normalizedName === code
    || normalizedName === `A股${code}`
    || normalizedName === `股票${code}`;
}

export type StockSearchHistoricalStatus = "inactive" | "bootstrapping" | "ready" | "blocked" | "failed";

export function normalizeStockSearchHistoricalStatus(value: unknown): StockSearchHistoricalStatus {
  return value === "bootstrapping" || value === "ready" || value === "blocked" || value === "failed"
    ? value
    : "inactive";
}

export function stockSearchResultIsAddable(status: StockSearchHistoricalStatus): boolean {
  return status === "ready";
}

export function stockSearchReadinessLabel(status: StockSearchHistoricalStatus): string {
  if (status === "bootstrapping") return "数据准备中";
  if (status === "blocked") return "权限待确认";
  if (status === "failed") return "暂不可用";
  if (status === "inactive") return "数据待接入";
  return "";
}
