import type { MarketRegion } from "./stock-symbol";

const UTC_EIGHT_OFFSET_MS = 8 * 60 * 60 * 1_000;
const MIDDAY_RESUME_MS = 13 * 60 * 60 * 1_000;

export type RealtimeMiddayState = {
  paused: boolean;
  transitionInMs: number | null;
};

export function realtimeMiddayState(
  market: MarketRegion,
  now: Date = new Date(),
): RealtimeMiddayState {
  const pauseStartMinutes = market === "cn" ? 11 * 60 + 30 : market === "hk" ? 12 * 60 : null;
  if (pauseStartMinutes === null) return { paused: false, transitionInMs: null };

  const local = new Date(now.getTime() + UTC_EIGHT_OFFSET_MS);
  const weekday = local.getUTCDay();
  const localTimeMs =
    ((local.getUTCHours() * 60 + local.getUTCMinutes()) * 60 + local.getUTCSeconds()) * 1_000 +
    local.getUTCMilliseconds();
  const pauseStartMs = pauseStartMinutes * 60 * 1_000;
  const isWeekday = weekday >= 1 && weekday <= 5;

  if (isWeekday && localTimeMs < pauseStartMs) {
    return { paused: false, transitionInMs: pauseStartMs - localTimeMs };
  }
  if (isWeekday && localTimeMs < MIDDAY_RESUME_MS) {
    return { paused: true, transitionInMs: MIDDAY_RESUME_MS - localTimeMs };
  }

  for (let daysAhead = 1; daysAhead <= 7; daysAhead += 1) {
    const nextWeekday = (weekday + daysAhead) % 7;
    if (nextWeekday < 1 || nextWeekday > 5) continue;
    return {
      paused: false,
      transitionInMs: daysAhead * 24 * 60 * 60 * 1_000 - localTimeMs + pauseStartMs,
    };
  }
  return { paused: false, transitionInMs: null };
}
