const INSTRUMENT_ID = /^(?:cn\.(?:xshg|xshe|xbse)\.\d{6}|hk\.xhkg\.\d{5}|us\.(?:xarc|xnas|xnys)\.[a-z0-9][a-z0-9._-]{0,31}|instrument\.[a-z0-9][a-z0-9._-]{1,117})$/u;
const PROVIDER_SYMBOL = /^[A-Z0-9._-]{2,64}$/u;
const MAX_MESSAGE_BYTES = 4_096;

export function localRealtimePollMilliseconds(value) {
  const parsed = Number.parseInt(value ?? "", 10);
  return Number.isFinite(parsed) && parsed >= 1_000 && parsed <= 30_000
    ? parsed
    : 1_000;
}

export function localRealtimeMiddayPauseMilliseconds(market, nowMs = Date.now()) {
  const pauseStartMinutes = market === "cn" ? 11 * 60 + 30 : market === "hk" ? 12 * 60 : null;
  if (pauseStartMinutes === null) return 0;
  const local = new Date(nowMs + 8 * 60 * 60 * 1_000);
  const weekday = local.getUTCDay();
  if (weekday < 1 || weekday > 5) return 0;
  const localTimeMs =
    ((local.getUTCHours() * 60 + local.getUTCMinutes()) * 60 + local.getUTCSeconds()) * 1_000 +
    local.getUTCMilliseconds();
  const pauseStartMs = pauseStartMinutes * 60 * 1_000;
  const resumeMs = 13 * 60 * 60 * 1_000;
  return localTimeMs >= pauseStartMs && localTimeMs < resumeMs
    ? resumeMs - localTimeMs
    : 0;
}

export function parseLocalRealtimeTicketRequest(requestUrl, instrumentId) {
  if (!INSTRUMENT_ID.test(instrumentId)) return null;
  const market = requestUrl.searchParams.get("market");
  const providerSymbol = requestUrl.searchParams.get("provider_symbol");
  const currency = requestUrl.searchParams.get("currency");
  if (
    !PROVIDER_SYMBOL.test(providerSymbol ?? "") ||
    !(
      (market === "cn" && currency === "CNY") ||
      (market === "hk" && currency === "HKD") ||
      (market === "us" && currency === "USD")
    )
  ) return null;
  return { instrumentId, market, providerSymbol, currency };
}

export function localRealtimeQuoteUrl(upstreamBase, ticket) {
  const url = new URL(
    `/v1/realtime/quotes/${encodeURIComponent(ticket.instrumentId)}`,
    `${upstreamBase.replace(/\/$/u, "")}/`,
  );
  url.searchParams.set("market", ticket.market);
  url.searchParams.set("provider_symbol", ticket.providerSymbol);
  url.searchParams.set("currency", ticket.currency);
  url.searchParams.set("stream", "true");
  return url;
}

export function serveLocalRealtimeWebSocket(
  socket,
  ticket,
  { upstreamBase, pollMilliseconds, fetchImpl = fetch, log = () => {}, now = Date.now },
) {
  const resumeToken = crypto.randomUUID();
  let pollingTimer = null;
  let quoteController = null;
  let stopped = false;
  let subscribed = false;
  let lastEventId = null;

  const send = (value) => {
    if (socket.readyState === 1) socket.send(JSON.stringify(value));
  };
  const schedule = (delay) => {
    if (stopped || pollingTimer !== null) return;
    pollingTimer = setTimeout(() => {
      pollingTimer = null;
      void pollOnce();
    }, delay);
  };
  const pollOnce = async () => {
    if (stopped || !subscribed) return;
    const middayPauseMilliseconds = localRealtimeMiddayPauseMilliseconds(ticket.market, now());
    if (middayPauseMilliseconds > 0) {
      schedule(middayPauseMilliseconds + 25);
      return;
    }
    const controller = new AbortController();
    quoteController = controller;
    try {
      const response = await fetchImpl(localRealtimeQuoteUrl(upstreamBase, ticket), {
        headers: { accept: "application/json" },
        cache: "no-store",
        signal: AbortSignal.any([
          controller.signal,
          AbortSignal.timeout(8_000),
        ]),
      });
      const event = await response.json().catch(() => null);
      if (
        !response.ok ||
        !event || typeof event !== "object" || Array.isArray(event) ||
        event.event_type !== "quote_state" ||
        event.instrument_id !== ticket.instrumentId ||
        typeof event.event_id !== "string"
      ) {
        throw new Error(`local realtime source returned ${response.status}`);
      }
      if (event.event_id !== lastEventId) {
        send(lastEventId === null
          ? {
              contract_version: 1,
              type: "snapshot",
              resume_token: resumeToken,
              events: [event],
            }
          : {
              contract_version: 1,
              type: "update",
              resume_token: resumeToken,
              event,
            });
        lastEventId = event.event_id;
      }
      schedule(pollMilliseconds);
    } catch (error) {
      if (stopped || (error instanceof Error && error.name === "AbortError")) return;
      log("local-realtime-poll-failed", {
        instrument_id: ticket.instrumentId,
        error: error instanceof Error ? error.message : String(error),
      });
      send({
        contract_version: 1,
        type: "error",
        request_id: null,
        code: "market_unavailable",
        message: "realtime source is temporarily unavailable",
        retryable: true,
        retry_after_ms: 5_000,
      });
      schedule(5_000);
    } finally {
      if (quoteController === controller) quoteController = null;
    }
  };
  const stop = () => {
    stopped = true;
    quoteController?.abort();
    if (pollingTimer !== null) clearTimeout(pollingTimer);
    pollingTimer = null;
  };

  socket.on("message", (data, isBinary) => {
    if (isBinary || data.byteLength > MAX_MESSAGE_BYTES) {
      send({
        contract_version: 1,
        type: "error",
        request_id: null,
        code: "invalid_message",
        message: "message is too large or is not text",
        retryable: false,
        retry_after_ms: null,
      });
      return;
    }
    let request;
    try {
      request = JSON.parse(data.toString("utf8"));
    } catch {
      request = null;
    }
    if (
      !request || request.contract_version !== 1 ||
      typeof request.request_id !== "string"
    ) {
      send({
        contract_version: 1,
        type: "error",
        request_id: null,
        code: "invalid_message",
        message: "message does not match stream contract v1",
        retryable: false,
        retry_after_ms: null,
      });
      return;
    }
    if (request.type === "resume") {
      send({
        contract_version: 1,
        type: "resume_required",
        reason: "history_unavailable",
        retry_after_ms: 0,
      });
      return;
    }
    if (
      request.type !== "subscribe" ||
      !Array.isArray(request.instrument_ids) ||
      request.instrument_ids.length !== 1 ||
      request.instrument_ids[0] !== ticket.instrumentId
    ) {
      send({
        contract_version: 1,
        type: "error",
        request_id: request.request_id,
        code: "forbidden",
        message: "ticket permits only its assigned instrument",
        retryable: false,
        retry_after_ms: null,
      });
      return;
    }
    subscribed = true;
    send({
      contract_version: 1,
      type: "subscribed",
      request_id: request.request_id,
      instrument_ids: [ticket.instrumentId],
      resume_token: resumeToken,
    });
    void pollOnce();
  });
  socket.on("close", stop);
  socket.on("error", stop);
  return { stop };
}
