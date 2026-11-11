// Trading-session status from exchange clocks. Weekends are closed; exchange holidays are
// not known here, so the status is an estimate and the UI labels it that way.

export type SessionState = "open" | "break" | "pre" | "post" | "closed";
export type MarketSession = { market: "cn" | "hk" | "us"; label: string; state: SessionState; stateLabel: string; localTime: string };

type Window = { start: number; end: number; state: SessionState };

const MARKETS: { market: MarketSession["market"]; label: string; timeZone: string; windows: Window[] }[] = [
  { market: "cn", label: "A股", timeZone: "Asia/Shanghai", windows: [
    { start: 9 * 60 + 15, end: 9 * 60 + 30, state: "pre" },
    { start: 9 * 60 + 30, end: 11 * 60 + 30, state: "open" },
    { start: 11 * 60 + 30, end: 13 * 60, state: "break" },
    { start: 13 * 60, end: 15 * 60, state: "open" },
  ] },
  { market: "hk", label: "港股", timeZone: "Asia/Hong_Kong", windows: [
    { start: 9 * 60, end: 9 * 60 + 30, state: "pre" },
    { start: 9 * 60 + 30, end: 12 * 60, state: "open" },
    { start: 12 * 60, end: 13 * 60, state: "break" },
    { start: 13 * 60, end: 16 * 60, state: "open" },
  ] },
  { market: "us", label: "美股", timeZone: "America/New_York", windows: [
    { start: 4 * 60, end: 9 * 60 + 30, state: "pre" },
    { start: 9 * 60 + 30, end: 16 * 60, state: "open" },
    { start: 16 * 60, end: 20 * 60, state: "post" },
  ] },
];

const STATE_LABELS: Record<SessionState, string> = { open: "交易中", break: "午间休市", pre: "盘前", post: "盘后", closed: "休市" };

function clock(now: Date, timeZone: string) {
  const parts = new Intl.DateTimeFormat("en-US", { timeZone, weekday: "short", hour: "2-digit", minute: "2-digit", hourCycle: "h23" }).formatToParts(now);
  const get = (type: string) => parts.find((part) => part.type === type)?.value ?? "";
  return { weekday: get("weekday"), minutes: Number(get("hour")) * 60 + Number(get("minute")), text: `${get("hour")}:${get("minute")}` };
}

export function marketSessions(now: Date = new Date()): MarketSession[] {
  return MARKETS.map(({ market, label, timeZone, windows }) => {
    const local = clock(now, timeZone);
    const weekend = local.weekday === "Sat" || local.weekday === "Sun";
    const state = weekend ? "closed" : windows.find((w) => local.minutes >= w.start && local.minutes < w.end)?.state ?? "closed";
    return { market, label, state, stateLabel: STATE_LABELS[state], localTime: local.text };
  });
}
