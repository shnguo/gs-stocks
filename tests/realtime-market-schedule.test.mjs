import assert from "node:assert/strict";
import test from "node:test";

import { realtimeMiddayState } from "../app/lib/realtime-market-schedule.ts";

test("A-share realtime pauses from 11:30 through 13:00 Shanghai time", () => {
  assert.deepEqual(
    realtimeMiddayState("cn", new Date("2026-09-02T03:29:59.000Z")),
    { paused: false, transitionInMs: 1_000 },
  );
  assert.deepEqual(
    realtimeMiddayState("cn", new Date("2026-09-02T03:30:00.000Z")),
    { paused: true, transitionInMs: 90 * 60 * 1_000 },
  );
  assert.deepEqual(
    realtimeMiddayState("cn", new Date("2026-09-02T04:59:59.000Z")),
    { paused: true, transitionInMs: 1_000 },
  );
  assert.equal(
    realtimeMiddayState("cn", new Date("2026-09-02T05:00:00.000Z")).paused,
    false,
  );
});

test("Hong Kong uses its 12:00 lunch boundary and US trading is unaffected", () => {
  assert.equal(
    realtimeMiddayState("hk", new Date("2026-09-02T03:59:59.000Z")).paused,
    false,
  );
  assert.deepEqual(
    realtimeMiddayState("hk", new Date("2026-09-02T04:00:00.000Z")),
    { paused: true, transitionInMs: 60 * 60 * 1_000 },
  );
  assert.deepEqual(
    realtimeMiddayState("us", new Date("2026-09-02T04:00:00.000Z")),
    { paused: false, transitionInMs: null },
  );
});

test("weekends schedule the next weekday pause without entering recess", () => {
  const saturday = realtimeMiddayState("cn", new Date("2026-09-05T04:00:00.000Z"));
  assert.equal(saturday.paused, false);
  assert.equal(saturday.transitionInMs, 47.5 * 60 * 60 * 1_000);
});
