import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import test from "node:test";

import {
  localRealtimeMiddayPauseMilliseconds,
  localRealtimePollMilliseconds,
  localRealtimeQuoteUrl,
  parseLocalRealtimeTicketRequest,
  serveLocalRealtimeWebSocket,
} from "../scripts/local-realtime-websocket.mjs";

test("local realtime pauses cache reads during China and Hong Kong lunch", () => {
  assert.equal(
    localRealtimeMiddayPauseMilliseconds("cn", Date.parse("2026-09-02T03:30:00.000Z")),
    90 * 60 * 1_000,
  );
  assert.equal(
    localRealtimeMiddayPauseMilliseconds("hk", Date.parse("2026-09-02T04:00:00.000Z")),
    60 * 60 * 1_000,
  );
  assert.equal(
    localRealtimeMiddayPauseMilliseconds("us", Date.parse("2026-09-02T04:00:00.000Z")),
    0,
  );
});

test("local realtime websocket enforces the one-second minimum", () => {
  assert.equal(localRealtimePollMilliseconds("500"), 1_000);
  assert.equal(localRealtimePollMilliseconds("1000"), 1_000);
  assert.equal(localRealtimePollMilliseconds("2500"), 2_500);
  assert.equal(localRealtimePollMilliseconds("30001"), 1_000);
});

test("local realtime websocket requires a complete provider mapping", () => {
  const url = new URL(
    "http://127.0.0.1/v1/realtime/tickets/instrument.apple?market=us&provider_symbol=AAPL.US&currency=USD",
  );
  assert.deepEqual(parseLocalRealtimeTicketRequest(url, "instrument.apple"), {
    instrumentId: "instrument.apple",
    market: "us",
    providerSymbol: "AAPL.US",
    currency: "USD",
  });
  url.searchParams.delete("currency");
  assert.equal(parseLocalRealtimeTicketRequest(url, "instrument.apple"), null);
});

test("local realtime websocket builds a bounded mapped quote request", () => {
  assert.equal(
    localRealtimeQuoteUrl("http://127.0.0.1:18080", {
      instrumentId: "cn.xshg.688008",
      market: "cn",
      providerSymbol: "688008.SH",
      currency: "CNY",
    }).toString(),
    "http://127.0.0.1:18080/v1/realtime/quotes/cn.xshg.688008?market=cn&provider_symbol=688008.SH&currency=CNY&stream=true",
  );
});

test("local realtime websocket checks the native push cache every second", async () => {
  const socket = new EventEmitter();
  socket.readyState = 1;
  socket.sent = [];
  socket.send = (message) => socket.sent.push(JSON.parse(message));
  let requests = 0;
  const event = {
    event_type: "quote_state",
    event_id: "longbridge-event-1",
    instrument_id: "instrument.apple",
  };
  const service = serveLocalRealtimeWebSocket(
    socket,
    {
      instrumentId: "instrument.apple",
      market: "us",
      providerSymbol: "AAPL.US",
      currency: "USD",
    },
    {
      upstreamBase: "http://127.0.0.1:18080",
      pollMilliseconds: 1_000,
      fetchImpl: async () => {
        requests += 1;
        return new Response(JSON.stringify(event), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      },
    },
  );

  socket.emit(
    "message",
    Buffer.from(JSON.stringify({
      contract_version: 1,
      type: "subscribe",
      request_id: "one-second-check",
      instrument_ids: ["instrument.apple"],
    })),
    false,
  );
  await new Promise((resolve) => setTimeout(resolve, 2_150));
  service.stop();

  assert.equal(requests, 3);
  assert.equal(socket.sent.filter((message) => message.type === "snapshot").length, 1);
  assert.equal(socket.sent.filter((message) => message.type === "update").length, 0);
});

test("local realtime websocket does not read the cache during midday recess", async () => {
  const socket = new EventEmitter();
  socket.readyState = 1;
  socket.sent = [];
  socket.send = (message) => socket.sent.push(JSON.parse(message));
  let requests = 0;
  const service = serveLocalRealtimeWebSocket(
    socket,
    {
      instrumentId: "cn.xshg.600487",
      market: "cn",
      providerSymbol: "600487.SH",
      currency: "CNY",
    },
    {
      upstreamBase: "http://127.0.0.1:18080",
      pollMilliseconds: 1_000,
      fetchImpl: async () => {
        requests += 1;
        return new Response(null, { status: 500 });
      },
      now: () => Date.parse("2026-09-02T03:40:00.000Z"),
    },
  );
  socket.emit(
    "message",
    Buffer.from(JSON.stringify({
      contract_version: 1,
      type: "subscribe",
      request_id: "midday-check",
      instrument_ids: ["cn.xshg.600487"],
    })),
    false,
  );
  await new Promise((resolve) => setTimeout(resolve, 25));
  service.stop();

  assert.equal(requests, 0);
  assert.equal(socket.sent.some((message) => message.type === "subscribed"), true);
});
