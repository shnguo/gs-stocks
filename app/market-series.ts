type CompletionAwareQuote = {
  completeness: string;
  marketSession: string;
  tradingStatus: string;
};

type DatedCompletionAwareQuote = CompletionAwareQuote & {
  tradingDate: string;
};

type ProgressiveRealtimeQuote = DatedCompletionAwareQuote & {
  observedAt: string;
};

type DatedQuote = {
  tradingDate: string;
};

type DatedCandle = {
  date: string;
};

type ClosingCandle = DatedCandle & {
  close: number;
};

type CompletionAwareCandle = {
  realtime?: boolean;
  unconfirmed?: boolean;
};

export function isCompletedRealtimeQuote(quote: CompletionAwareQuote): boolean {
  return quote.completeness === "complete"
    && ["after_hours", "closed"].includes(quote.marketSession)
    && quote.tradingStatus === "closed";
}

export function selectProgressiveRealtimeQuote<T extends ProgressiveRealtimeQuote>(
  current: T | null,
  incoming: T,
): T {
  if (!current) return incoming;
  if (incoming.tradingDate !== current.tradingDate) {
    return incoming.tradingDate > current.tradingDate ? incoming : current;
  }
  if (isCompletedRealtimeQuote(current)) return current;
  if (isCompletedRealtimeQuote(incoming)) return incoming;
  return incoming.observedAt >= current.observedAt ? incoming : current;
}

export function selectCompletedSeries<T extends CompletionAwareCandle>(
  history: T[],
  display: T[],
): T[] {
  const latest = display[display.length - 1];
  return latest?.realtime && latest.unconfirmed === false ? display : history;
}

export function displayMovingAverageValues(
  display: readonly ClosingCandle[],
  period: number,
): (number | null)[] {
  if (!Number.isInteger(period) || period <= 0) {
    return display.map(() => null);
  }
  let sum = 0;
  return display.map((candle, index) => {
    sum += candle.close;
    if (index >= period) sum -= display[index - period]?.close ?? 0;
    return index >= period - 1 ? sum / period : null;
  });
}

export function completedQuoteNeedsHistoricalRevalidation(
  history: readonly DatedCandle[],
  quote: DatedCompletionAwareQuote | null,
): boolean {
  if (!quote || !isCompletedRealtimeQuote(quote)) return false;
  const latestHistoricalDate = history[history.length - 1]?.date ?? "";
  return latestHistoricalDate < quote.tradingDate;
}

export function hasHistoricalSeriesGap(
  history: readonly DatedCandle[],
  quote: DatedQuote | null,
  verifiedClosedDates: ReadonlySet<string> = new Set(),
): boolean {
  return historicalSeriesGapDates(history, quote)
    .some((date) => !verifiedClosedDates.has(date));
}

export function historicalSeriesGapDates(
  history: readonly DatedCandle[],
  quote: DatedQuote | null,
): string[] {
  const latestHistoricalDate = history[history.length - 1]?.date;
  if (!latestHistoricalDate || !quote || quote.tradingDate <= latestHistoricalDate) return [];
  const latest = Date.parse(`${latestHistoricalDate}T00:00:00Z`);
  const realtime = Date.parse(`${quote.tradingDate}T00:00:00Z`);
  if (!Number.isFinite(latest) || !Number.isFinite(realtime)) return [];
  const dates: string[] = [];
  for (let day = latest + 86_400_000; day < realtime; day += 86_400_000) {
    const weekday = new Date(day).getUTCDay();
    if (weekday >= 1 && weekday <= 5) {
      dates.push(new Date(day).toISOString().slice(0, 10));
    }
  }
  return dates;
}

export function closedBarRevalidationDelayMs(attempt: number): number {
  const normalizedAttempt = Number.isInteger(attempt) && attempt > 0 ? attempt : 0;
  const schedule = [5_000, 15_000, 30_000, 60_000, 120_000, 300_000] as const;
  return schedule[Math.min(normalizedAttempt, schedule.length - 1)] ?? 300_000;
}

export function instrumentReadinessRetryDelayMs(readiness: string, attempt: number): number {
  const baseDelay = closedBarRevalidationDelayMs(attempt);
  if (["awaiting-universe", "awaiting-campaign", "failed"].includes(readiness)) {
    return Math.max(60_000, baseDelay);
  }
  if (readiness === "processing") return Math.max(15_000, baseDelay);
  return baseDelay;
}
