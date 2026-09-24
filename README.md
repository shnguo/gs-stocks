# 低位承接研究台

使用同一组真实日线绘制蜡烛图、MA5 和低位承接强度指标。网页通过服务端路由读取 mootdx-cf 的正式日线投影，浏览器不会收到数据 API 凭证。

## Prerequisites

- Node.js `>=22.13.0`

## Quick Start

```bash
npm install
npm run dev
npm run build
```

This starter does not use `wrangler.jsonc`.

## Real market data

The default watchlist contains the four instruments that passed the controlled 250-trading-day preview acceptance: 600036, 688008, 000333, and 300059. The page supports none, qfq, and hfq. It displays the selected source, comparison state, point-in-time boundary, data version, and serving-cache state instead of presenting a single source as independently verified.

The server route requests the Collector's explicit latest mode with a stable 400-calendar-day rolling window, a 250-row limit, and the compact chart-lite-v1 projection. The key does not move at midnight; concrete query dates are derived only when the serving snapshot refreshes. Ordinary reads use only versioned R2 chart snapshots and edge cache. Missing snapshots schedule Queue repair instead of starting the analytical container in the user request. The bearer token remains server-only, and exact historical point-in-time requests remain separate from this fast path.

The default unadjusted chart request is preloaded from the initial HTML because that basis is present in every activated market generation. This avoids a failed adjusted-basis negotiation on first paint; qfq and hfq become selectable only when the generation advertises them. The browser also keeps each successfully verified instrument-and-basis payload in IndexedDB, renders it immediately on a later visit, and revalidates in the background. If serving repair is temporarily in progress, the UI retains that local copy and marks it stale rather than replacing the chart with a long loading state.

Normal chart reads do not query the control plane. When the latest historical row and realtime trading date have intervening weekdays, the page makes one bounded request to /api/market-days. The upstream Worker serves that range and /api/market-day-readiness from a direct read-only D1 binding, so ingestion load in the Rust container cannot delay the page. It may bridge realtime across those dates only when every intervening weekday has durable closed status. Missing records and awaiting, running, failed, or open-day records remain visible as a historical gap. During that exceptional recovery state, /api/market-day-readiness checks the selected stock before retrying the chart, so a published stock can render from its compact daily delta without waiting for the complete market generation. This handles exchange holidays without turning a provider outage or a lost daily bar into a fake closure.

For local acceptance, run npm test, start npm run dev:local, and request /api/market-days with a bounded start_date and end_date. For an open gap date, also request /api/market-day-readiness with business_date and instrument_id. In the browser, section#top exposes data-calendar-continuity, data-historical-gap, data-finalization-status, data-stock-readiness, and data-market-readiness. A real missing open date must settle on gap and must not append the realtime date; a fully closed holiday interval must settle on verified and may append the realtime candle. When stock readiness becomes published, the next chart revalidation must include that date even if market readiness is still building.

The stable rolling-window acceptance returned the three 250-row chart bases from R2 in 255 ms, 122 ms, and 180 ms at a cold edge location, then in 39 ms, 36 ms, and 28 ms from that location's edge cache. The page exposes the current cache state so a slow origin fill, R2 snapshot read, stale response, and edge hit are distinguishable during diagnosis.

Configure the server-only values shown in `.env.example`. Never rename the bearer token with a `NEXT_PUBLIC_` prefix.

### Hundreds of US watchlist symbols

The watchlist has no ten-symbol application limit. It refreshes supported favorites through POST /api/realtime-quotes in batches of at most 500 and repeats the snapshot refresh every 15 seconds. If a user stores more than 500 favorites, the browser sends additional batches. Only the selected chart opens the continuous realtime connection, so inactive rows do not consume one upstream streaming subscription each.

For the local Longbridge pilot, start the Rust collector with REALTIME_PROVIDER=longbridge and its OAuth client ID, then add these server-only values to `.dev.vars`:

```text
MOOTDX_REALTIME_API_BASE_URL=http://127.0.0.1:3101
MOOTDX_REALTIME_TRANSPORT=websocket
MOOTDX_LOCAL_REALTIME_UPSTREAM_URL=http://127.0.0.1:8080
MOOTDX_LOCAL_REALTIME_POLL_INTERVAL_MS=1000
```

The Longbridge adapter enables overnight quotes and chooses the newest available US pre-market, regular, after-hours, or overnight session payload. The dashboard sends verified provider-symbol mappings with both selected-stock and batch snapshot requests. The personal OAuth path remains local-only and is not a public redistribution configuration.

When started with npm run dev:local, the loopback bridge upgrades this local Longbridge snapshot source into the dashboard WebSocket protocol. It samples only the selected chart once per second, pushes changed events to the browser, uses one-time thirty-second connection tickets, and keeps two-second browser polling as the degraded fallback. A-share refreshes pause from 11:30 through 13:00 China time, Hong Kong refreshes pause from 12:00 through 13:00 Hong Kong time, and both resume automatically. Mixed-market watchlists continue refreshing active US instruments without polling the markets that are at lunch. A legacy local configuration that still points the realtime API directly at the Rust service in polling mode is promoted automatically by dev:local. Direct npm run dev requires the bridge to be started separately when the realtime API points at port 3101.

When the local machine cannot resolve or reach workers.dev reliably, place the values from `.env.example` in `.dev.vars`, then start the site and its loopback-only SOCKS5 bridge together:

```bash
npm run dev:local
```

On macOS, `dev:local` first synchronizes the deployment-01 Collector Token from the `mootdx-cf-preview-collector-api-token` Keychain item into the ignored, mode-600 `.dev.vars` file. An explicitly supplied `MOOTDX_DATA_API_BEARER_TOKEN` takes precedence. This keeps the Vinext Worker environment and the loopback bridge on the same credential after token rotation.

The bridge health endpoint is `http://127.0.0.1:3101/healthz`. For separate terminals, start the bridge first:

```bash
export MOOTDX_DATA_API_BEARER_TOKEN=replace-with-preview-api-token
npm run dev:market-proxy
```

Then start the site in a second terminal:

```bash
export MOOTDX_DATA_API_BEARER_TOKEN=replace-with-preview-api-token
export MOOTDX_DATA_API_BASE_URL=http://127.0.0.1:3101
npm run dev
```

The bridge listens only on 127.0.0.1, verifies the bearer token, and sends the upstream credential to curl through standard input. It also forwards short-lived realtime WebSocket connections through the same SOCKS5 endpoint. Start the site on port 3000, or set MOOTDX_BROWSER_ORIGIN to its exact local origin before starting the bridge. Production uses a Cloudflare Secret named `MOOTDX_DATA_API_BEARER_TOKEN` and connects to the Collector directly; it does not use the local bridge.

## Included Shape

- edit site code under `app/`
- `.openai/hosting.json` declares optional Sites D1 and R2 bindings
- `vite.config.ts` simulates declared bindings for local development
- `db/schema.ts` starts intentionally empty
- `examples/d1/` contains an optional D1 example surface
- `drizzle.config.ts` supports local migration generation when needed

## Workspace Auth Headers

Signed-in visitors receive both `oai-authenticated-user-id` and `oai-authenticated-user-email`. Private Sites require every visitor to sign in; public Sites may also have anonymous visitors, for whom neither header is present.

The user ID is stable for the same user on the same Site and different across Sites. Email and name are intended for display or contact purposes.

SIWC-authenticated workspace sites may also receive
`oai-authenticated-user-full-name` when the user's SIWC profile has a non-empty
`name` claim. The full-name value is percent-encoded UTF-8 and is accompanied by
`oai-authenticated-user-full-name-encoding: percent-encoded-utf-8`.

Treat the full name as optional and fall back to email when it is absent:

```tsx
import { headers } from "next/headers";

export default async function Home() {
  const requestHeaders = await headers();
  const userId = requestHeaders.get("oai-authenticated-user-id");
  const email = requestHeaders.get("oai-authenticated-user-email");
  const encodedFullName = requestHeaders.get("oai-authenticated-user-full-name");
  const fullName =
    encodedFullName &&
    requestHeaders.get("oai-authenticated-user-full-name-encoding") ===
      "percent-encoded-utf-8"
      ? decodeURIComponent(encodedFullName)
      : null;

  const displayName = fullName ?? email;
  // ...
}
```

## Optional Dispatch-Owned ChatGPT Sign-In

Import the ready-to-use helpers from `app/chatgpt-auth.ts` when the site needs
optional or required ChatGPT sign-in:

- Use `getChatGPTUser()` for optional signed-in UI.
- Use `requireChatGPTUser(returnTo)` for server-rendered pages that should send
  anonymous visitors through Sign in with ChatGPT.
- Use `chatGPTSignInPath(returnTo)` and `chatGPTSignOutPath(returnTo)` for
  browser links or actions.
- Pass a same-origin relative `returnTo` path for the destination after sign-in
  or sign-out. The helper validates and safely encodes it.
- Mark protected pages with `export const dynamic = "force-dynamic"` because
  they depend on per-request identity headers.

Dispatch owns `/signin-with-chatgpt`, `/signout-with-chatgpt`, `/callback`, the
OAuth cookies, and identity header injection. Do not implement app routes for
those reserved paths. Routes that do not import and call the helper remain
anonymous-compatible.

SIWC establishes identity only; it does not prove workspace membership. Use the
Sites hosting platform's access policy controls for workspace-wide restrictions,
or enforce explicit server-side membership or allowlist checks.

Use SIWC for account pages, user-specific dashboards, saved records, and write
actions tied to the current ChatGPT user. Leave public content anonymous.

## Useful Commands

- `npm run dev`: start local development
- `npm run dev:local`: start local development and the realtime preview bridge together
- `npm run dev:market-proxy`: start the loopback-only preview bridge for unreliable local workers.dev routing
- `npm run build`: verify the vinext build output
- `npm test`: build the starter and verify its rendered loading skeleton
- `npm run db:generate`: generate Drizzle migrations after schema changes

## Learn More

- [vinext Documentation](https://github.com/cloudflare/vinext)
- [Drizzle D1 Guide](https://orm.drizzle.team/docs/get-started/d1-new)
