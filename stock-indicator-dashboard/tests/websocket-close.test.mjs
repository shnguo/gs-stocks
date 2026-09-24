import assert from "node:assert/strict";
import test from "node:test";

import {
  closeWebSocketSafely,
  forwardedClose,
} from "../scripts/websocket-close.mjs";

test("forwards valid websocket close codes", () => {
  assert.deepEqual(forwardedClose(1000, "done"), { code: 1000, reason: "done" });
  assert.deepEqual(forwardedClose(4001, "application"), {
    code: 4001,
    reason: "application",
  });
});

test("does not forward reserved websocket close codes", () => {
  assert.deepEqual(forwardedClose(1005, "missing"), {
    code: undefined,
    reason: undefined,
  });
  assert.deepEqual(forwardedClose(1006, "abnormal"), {
    code: 1011,
    reason: "peer connection ended unexpectedly",
  });
  assert.deepEqual(forwardedClose(1015, "tls"), {
    code: 1011,
    reason: "peer connection ended unexpectedly",
  });
});

test("truncates websocket close reasons to the protocol limit", () => {
  const forwarded = forwardedClose(1000, "汉".repeat(100));
  assert.equal(forwarded.code, 1000);
  assert.ok(Buffer.byteLength(forwarded.reason, "utf8") <= 123);
});

test("terminates a peer if close still throws", () => {
  const failures = [];
  const peer = {
    readyState: 1,
    close() {
      throw new TypeError("invalid close");
    },
    terminateCalled: false,
    terminate() {
      this.terminateCalled = true;
    },
  };
  closeWebSocketSafely(peer, 1006, "abnormal", (error) => failures.push(error));
  assert.equal(peer.terminateCalled, true);
  assert.equal(failures.length, 1);
});
