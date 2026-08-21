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

The server route requests the Collector's explicit latest mode with a stable 400-calendar-day rolling window and a 250-row limit. The key does not move at midnight; concrete query dates are derived only when the serving snapshot refreshes. That mode uses versioned R2 chart snapshots, conditional ETags, and background refresh. The bearer token remains server-only, and exact historical point-in-time requests remain separate from this fast path.

The stable rolling-window acceptance returned the three 250-row chart bases from R2 in 255 ms, 122 ms, and 180 ms at a cold edge location, then in 39 ms, 36 ms, and 28 ms from that location's edge cache. The page exposes the current cache state so a slow origin fill, R2 snapshot read, stale response, and edge hit are distinguishable during diagnosis.

Configure the server-only values shown in `.env.example`. Never rename the bearer token with a `NEXT_PUBLIC_` prefix.

When the local machine cannot resolve or reach workers.dev reliably, start the loopback-only SOCKS5 bridge in one terminal:

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
- `npm run dev:market-proxy`: start the loopback-only preview bridge for unreliable local workers.dev routing
- `npm run build`: verify the vinext build output
- `npm test`: build the starter and verify its rendered loading skeleton
- `npm run db:generate`: generate Drizzle migrations after schema changes

## Learn More

- [vinext Documentation](https://github.com/cloudflare/vinext)
- [Drizzle D1 Guide](https://orm.drizzle.team/docs/get-started/d1-new)
