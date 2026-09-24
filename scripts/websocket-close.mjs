const VALID_STANDARD_CLOSE_CODES = new Set([
  1000,
  1001,
  1002,
  1003,
  1007,
  1008,
  1009,
  1010,
  1011,
  1012,
  1013,
  1014,
]);

export function forwardedClose(code, reason) {
  if (code === 1005) {
    return { code: undefined, reason: undefined };
  }
  if (VALID_STANDARD_CLOSE_CODES.has(code) || (code >= 3000 && code <= 4999)) {
    return { code, reason: truncateUtf8(String(reason ?? ""), 123) };
  }
  return { code: 1011, reason: "peer connection ended unexpectedly" };
}

export function closeWebSocketSafely(target, code, reason, logFailure = () => {}) {
  if (!target || target.readyState >= 2) return;
  const forwarded = forwardedClose(code, reason);
  try {
    if (forwarded.code === undefined) target.close();
    else target.close(forwarded.code, forwarded.reason);
  } catch (error) {
    logFailure(error);
    try {
      target.terminate();
    } catch (terminateError) {
      logFailure(terminateError);
    }
  }
}

function truncateUtf8(value, maximumBytes) {
  const bytes = Buffer.from(value, "utf8");
  if (bytes.length <= maximumBytes) return value;
  let end = maximumBytes;
  while (end > 0 && (bytes[end] & 0b1100_0000) === 0b1000_0000) end -= 1;
  return bytes.subarray(0, end).toString("utf8");
}
