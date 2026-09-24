# Hong Kong, US, and Custom-Symbol Development Roadmap

## Document status

- Created: 2026-08-31
- Scope: stock-indicator-dashboard and mootdx-cf
- Delivery strategy: curated historical coverage first, user-added symbols second, controlled realtime third, public service only after rights approval
- Relationship to the A-share roadmap: additive. The existing A-share ingestion, history, and recovery work remains independent and must not be destabilized by this program.

## Executive decision

The current platform should evolve into a curated multi-market service rather than ingesting every Hong Kong and US listing.

The recommended product sequence is:

1. Publish trustworthy end-of-day charts for a small, administrator-managed set of popular Hong Kong and US stocks.
2. Allow a user to request an additional symbol through a validated admission workflow.
3. Add controlled realtime for the curated universe through shared upstream subscriptions.
4. Consider public realtime only after written market-data rights, entitlement enforcement, load tests, and a cost cap are approved.

This sequence reuses the current D1, R2, Iceberg, Worker, Queue, and chart-snapshot architecture while isolating the parts that are still A-share-specific.

## Product boundaries

### Initial supported instruments

- Common equities and ETFs only.
- Hong Kong listings on XHKG.
- US listings on XNAS, XNYS, and other explicitly approved US venues when required.
- Administrator-curated popular symbols.
- User-requested symbols that pass provider validation and data-readiness checks.

### Explicit non-goals for the first release

- Full Hong Kong or US market ingestion.
- Options, warrants, CBBCs, futures, cryptocurrencies, or OTC instruments.
- Order books, trades, broker queues, or Level 2 data.
- Public market-data API access.
- Trading or brokerage functions.
- Automatic activation of arbitrary user strings.
- Silent substitution between providers or adjustment methods.

## Target architecture

Historical path:

Provider adapters
  → immutable raw responses in R2
  → normalized multi-currency daily bars
  → validation and source comparison
  → Iceberg history and lineage
  → immutable compact chart snapshots in R2
  → Worker authorization and edge cache
  → dashboard

Realtime path:

Provider subscription connection
  → Rust feed shard
  → normalized complete quote state
  → authenticated Worker ingress
  → Durable Object fanout room
  → short-lived browser ticket
  → dashboard

Control path:

Curated catalog and user request
  → D1 symbol admission state
  → provider identity validation
  → historical bootstrap
  → readiness verification
  → activation in the historical and optional realtime universe

## Architectural rules

1. Instrument identity is provider-neutral. Provider symbols are mappings, not primary keys.
2. A listing symbol may change without losing the instrument history.
3. Exchange, currency, timezone, trading calendar, and session rules are explicit data.
4. Every published bar and quote preserves source, observation time, retrieval time, revision, and raw lineage.
5. Adjustment support is declared per provider and instrument. Unsupported bases are unavailable, never simulated.
6. User input cannot directly control provider URLs, R2 keys, Queue names, or realtime room identities.
7. Historical publication and realtime eligibility are separate states.
8. A provider quota is not redistribution permission.
9. Existing A-share identifiers and serving paths remain compatible during the migration.
10. Unknown market rights, expired entitlements, and incomplete symbol admission fail closed.

## Roadmap summary

The effort ranges below assume one experienced engineer working primarily on this program. They exclude vendor contracting, legal review, and waiting for external approvals.

| Milestone | Outcome | Estimated effort | Release boundary |
|---|---|---:|---|
| R0 | Contracts and identity decisions | 1 week | Local only |
| R1 | Curated HK and US historical charts | 2 to 3 weeks | Isolated preview |
| R2 | User-added symbol workflow | 1 to 2 weeks | Authenticated beta |
| R3 | Controlled multi-market realtime | 3 to 5 weeks | Named preview users |
| R4 | Production hardening and public decision | 2 to 4 weeks plus external approvals | Conditional |

The shortest credible route to an authenticated historical beta is approximately four to six engineering weeks. Public realtime has no reliable calendar date until provider rights are documented.

## R0: Contracts, identity, and source decision

### Goal

Remove cross-market ambiguity before introducing new data.

### Work items

- Write an architecture decision record for canonical instrument identity.
- Decide how stable instrument identity survives ticker changes, share-class punctuation, and exchange transfers.
- Add exchange definitions for XHKG, XNAS, XNYS, and any additional approved venue.
- Define market metadata: country, currency, timezone, calendar source, session model, and default delay mode.
- Version the daily-bar contract so currency is explicit for both price and turnover.
- Replace the assumption that every instrument supports none, qfq, and hfq with per-instrument available adjustment bases.
- Define normalized adjustment names and the exact mapping from each provider.
- Define how US premarket, regular, after-hours, and overnight observations map to trading date and session.
- Define how Hong Kong lunch recess, auctions, half days, typhoon sessions, and suspensions are represented.
- Create a provider capability matrix for historical bars, realtime quotes, calendars, corporate actions, adjustments, currencies, corrections, quotas, and rights.
- Select one primary provider for the controlled HK and US pilot and one independent comparison source for acceptance.
- Record provider terms and approval evidence outside the repository; keep only decisions and enforcement state in source control.

### Required fixtures

- Hong Kong ordinary equity with a five-digit display code and leading zero preservation.
- Hong Kong suspended or special-session day.
- US Nasdaq equity.
- US NYSE equity.
- US share class or ticker containing punctuation.
- US split event.
- US ticker or exchange migration.
- ETF in each target market if ETFs remain in scope.

### Exit conditions

- The same listing resolves to one canonical instrument across search, ingestion, storage, API, and UI.
- A ticker or venue change does not break historical continuity.
- Currency and session semantics are unambiguous in the contracts.
- Unsupported adjustment bases produce an explicit unavailable result.
- The source decision distinguishes technical coverage from redistribution rights.

## R1: Curated Hong Kong and US historical charts

### Goal

Publish trustworthy end-of-day charts for a small administrator-managed universe without ingesting either full market.

### Initial acceptance universe

- Five to ten Hong Kong equities.
- Five to ten US equities.
- At least one ETF per market if ETFs are in scope.
- At least one corporate-action case, one suspension or halt case, one high-volume case, and one thin or irregular case.

### Backend work

- Add provider symbol codecs for Hong Kong and US listings.
- Add a historical daily-bar adapter for the selected provider.
- Capture the original provider response before parsing or validation.
- Normalize prices, volume, turnover, currency, trading date, source identity, and adjustment basis.
- Add exchange-calendar adapters and point-in-time calendar revisions.
- Build a targeted multi-market collection command separate from the full A-share campaign machinery.
- Add per-market post-close schedules rather than reusing China close times.
- Publish raw and adjusted series only when the provider explicitly supports the requested basis.
- Extend Iceberg schemas and queries for explicit currency where required.
- Reuse immutable chart snapshots, data-version hashes, R2 object versioning, and edge caching.
- Keep market generation pointers separate by market or use an explicit market dimension in generation identity.
- Preserve source comparison status and do not call a single-source row matched.

### Dashboard work

- Replace A-share-only symbol types with a market-neutral instrument view model.
- Make market tabs render actual supported and favorite counts.
- Add market-aware search and symbol formatting.
- Preserve Hong Kong leading zeros and display US tickers in their conventional case.
- Display currency beside price and turnover where ambiguity is possible.
- Render session, delay, source, as-of time, adjustment basis, and freshness honestly.
- Migrate browser favorites without deleting existing A-share selections.
- Disable unavailable adjustment controls at the instrument level.
- Keep CSV import separate from live provider data and label its source state.

### Verification

- Unit tests for symbol codecs, calendars, session mapping, currency, adjustment mapping, and corporate actions.
- Golden fixtures for raw capture and deterministic replay.
- A live shadow comparison for every acceptance symbol.
- Per-day OHLCV reconciliation for at least 250 completed sessions when history is available.
- API checks for source, currency, revision, adjustment basis, freshness, and lineage.
- Browser verification on desktop and mobile for market switching, favorite switching, adjustment switching, loading, empty, stale, and error states.
- Cache verification for edge hit, R2 snapshot hit, stale refresh, and cold rebuild.

### Exit conditions

- Every acceptance symbol has an immutable, reproducible historical chart snapshot.
- No row has an incorrect currency, exchange date, duplicated date, or silent adjustment substitution.
- Browser candles match the API series exactly.
- A provider outage returns stale or unavailable state without inventing data.
- Existing A-share chart behavior passes regression tests.

## R2: User-added symbol admission

### Goal

Allow future custom symbols without maintaining a complete exchange universe and without letting arbitrary input bypass validation.

### D1 control model

Add control-plane records equivalent to:

- market_catalog: the activated provider-neutral instrument view used by the product.
- provider_symbol_mappings: provider, provider symbol, instrument, effective dates, and verification state.
- universe_memberships: curated or user-requested membership, owner, priority, and lifecycle state.
- symbol_admission_requests: requester, raw query, resolved candidate, status, reason, and timestamps.
- symbol_bootstrap_state: history coverage, snapshot version, retry state, and readiness.
- market_rights_state: market, field family, surface, delay class, effective period, and approval state.

Recommended admission states:

requested → resolving → verified → bootstrapping → ready → active

Terminal or exceptional states:

rejected, unsupported, quota-blocked, rights-blocked, failed, inactive

### Admission workflow

1. Authenticate and rate-limit the requester.
2. Normalize the search query without assuming its exchange.
3. Query the provider security directory or static-information endpoint.
4. Return disambiguation candidates when more than one listing matches.
5. Persist the canonical instrument and provider mapping only after validation.
6. Check market and field rights before any publication.
7. Enqueue a bounded historical bootstrap.
8. Validate coverage, currency, calendar alignment, adjustment semantics, and raw lineage.
9. Publish the immutable chart snapshot.
10. Mark historical readiness active.
11. Evaluate realtime eligibility separately.

### Product behavior

- The user sees progress rather than an empty chart during bootstrap.
- Duplicate requests converge on one admission and one bootstrap.
- Rejected requests expose a safe reason such as unsupported market, ambiguous symbol, provider unavailable, quota reached, or rights unavailable.
- Removing a favorite does not delete shared market history.
- Inactive long-tail symbols can stop realtime subscriptions while retained historical snapshots remain addressable under the approved retention policy.

### Security and abuse controls

- Strict query length and character validation.
- Provider requests built only from resolved catalog records.
- Per-subject request quotas and global admission concurrency limits.
- Idempotent Queue commands and deterministic bootstrap identity.
- No provider credentials, OAuth tokens, or internal object keys in browser responses.
- Administrative controls for blocking, retrying, deactivating, and auditing a symbol.

### Exit conditions

- A new supported HK or US symbol can move from request to a verified chart without a deployment or environment-variable change.
- Duplicate requests do not duplicate upstream history queries or chart objects.
- Unsupported symbols fail safely and leave an auditable decision.
- User favorites persist server-side for authenticated users and remain tenant-isolated.
- Admission cannot broaden market-data rights.

## R3: Controlled multi-market realtime

### Goal

Deliver normalized HK and US quote states to named preview users while sharing upstream subscriptions across all viewers.

### Mandatory architecture change

Do not extend the current per-instrument Durable Object polling loop as the production provider-ingestion model.

Implement the proposed separation:

- Rust feed shards own provider connections, subscriptions, provider credentials, sequence handling, mapping, and normalized state.
- A symbol has one formal upstream publisher in a shard generation.
- The Worker authenticates internal feed ingress and routes each normalized state to the correct fanout shard.
- Durable Objects own browser WebSockets, the latest complete state, resume behavior, slow-consumer handling, and broadcast.
- The gateway authorizes subject, device, market, surface, delay class, and symbol before issuing a short-lived ticket.
- Closed minute windows, if added, flow through an idempotent Queue and persistent validation path; browser delivery does not wait for persistence.

### Provider integration

- Extend the selected provider adapter for HK and US symbols.
- Use provider push subscriptions or an explicitly rate-limited shared snapshot loop.
- Implement a subscription registry with pinned curated symbols and bounded demand-driven custom symbols.
- Keep realtime subscription state separate from historical catalog state.
- Support full snapshot recovery after reconnect or sequence gaps.
- Normalize HKD and USD, sessions, timestamps, halts, and stale states.
- Treat US extended-hours states separately from the regular completed daily candle.
- Never merge provider deltas in the browser.

### Credential handling

- Replace local interactive OAuth dependence with an approved non-interactive service credential flow if the provider supports it.
- If OAuth refresh tokens are required, store them only in an approved encrypted secret or token service with rotation and audit.
- Do not deploy the existing local-only Longbridge OAuth pilot to cloud production unchanged.

### Capacity and recovery tests

- Ordinary distribution across the curated universe.
- At least 50 percent of clients concentrated on one popular symbol.
- Market-open burst and reconnect storm.
- Slow browser consumer.
- Provider disconnect and full snapshot recovery.
- Feed-shard and fanout generation change.
- Duplicate, late, corrected, malformed, and out-of-order events.
- Separate HK and US session transitions.
- Quota exhaustion and rights expiration.

### Exit conditions

- Multiple viewers of one symbol share one upstream subscription.
- No browser connection receives a symbol outside its ticket and entitlement.
- Reconnect restores a complete state without double-counting provider deltas.
- Freshness, browser latency, disconnect recovery, fanout capacity, and provider call rates meet measured preview targets.
- Realtime failures do not corrupt formal daily history.
- Provider credentials and refresh tokens never reach D1 plaintext, logs, browser payloads, or repository files.

## R4: Production hardening and public-service decision

### Goal

Decide whether the authenticated preview can become a public or paid product.

### External approval gates

- Written provider and exchange authorization for each market, field, delay class, audience, device model, cache behavior, derived bar, and retention period.
- Approved professional and non-professional user classification behavior where required.
- Approved attribution, reporting, audit, deletion, and geography controls.
- Approved disaster-feed and correction behavior.
- Approved monthly provider and Cloudflare cost cap with a named owner.

### Engineering gates

- Authorization occurs before cache lookup and WebSocket upgrade.
- Cache identity includes the required public, entitlement-class, or subject-specific scope.
- Unknown or expired rights fail closed.
- D1 control state can be rebuilt or restored without deleting R2 raw or historical evidence.
- Feed, fanout, Queue, snapshot, and dashboard recovery drills pass.
- Load tests establish safe operating limits with headroom rather than relying on platform maximums.
- Cost is measured per active symbol, connection hour, million updates, historical bootstrap, and chart request.
- Monitoring covers provider freshness, missing symbols, corrections, subscription count, Queue age, fanout load, snapshot age, error rate, and browser-displayed data age.
- Runbooks cover provider outage, credential rotation, rights suspension, symbol migration, bad corporate action, and partial-market closure.

### Exit conditions

- Every exposed field has an approved and enforced rights decision.
- Production capacity and unit cost are supported by target-environment measurements.
- Public launch, paid launch, or continued private use is an explicit product decision.
- The rollback plan is rehearsed and preserves raw lineage and historical snapshots.

## Recommended provider strategy

### Controlled private or authenticated pilot

- Keep Tencent, TDX, Tushare, and Eastmoney in their existing A-share roles.
- Evaluate Longbridge as the primary HK and US pilot provider because its official API exposes cross-market security identity, quotes, historical candlesticks, and subscriptions.
- Confirm the exact account tier, market permissions, historical-symbol quota, realtime subscription quota, sessions, and token model before implementation.
- Use an independent source to measure historical disagreements and provider omissions.

### Public product

- Do not assume broker API access includes public redistribution.
- Select a provider or exchange-authorized vendor only after the intended web, mobile, paid, free, cache, derived-data, and retention uses are covered in writing.
- Keep the adapter boundary provider-neutral so the controlled-pilot source can be replaced without changing browser contracts or historical identity.

## Testing matrix

| Layer | Required evidence |
|---|---|
| Identity | Ticker change, exchange move, share class, leading zero, duplicate name |
| Calendar | Holiday, half day, lunch recess, suspension, extended hours, overnight boundary |
| Historical source | Raw replay, missing range, correction, split, dividend, timeout, malformed payload |
| Normalization | Currency, OHLC, volume, turnover, timestamp, adjustment basis |
| Publication | Idempotency, lineage, immutable snapshot, cache invalidation, cold rebuild |
| Admission | Duplicate request, ambiguous result, unsupported symbol, quota block, rights block |
| Realtime | Snapshot, update, reconnect, sequence gap, stale state, halt, slow consumer |
| Authorization | Wrong subject, wrong market, expired right, cache bypass, ticket replay |
| Dashboard | Market tabs, search, favorites, currency, sessions, mobile, empty and error states |
| Operations | Provider outage, Queue backlog, D1 recovery, R2 replay, token rotation, rollback |

## Delivery order inside the repositories

### mootdx-cf

1. Add ADRs and contract versions.
2. Add exchange and market metadata.
3. Add provider-neutral symbol mapping and fixtures.
4. Add historical provider adapter and raw capture.
5. Extend normalized daily bars and Iceberg publication.
6. Add targeted multi-market collection commands and schedules.
7. Add D1 catalog and admission migrations.
8. Add bootstrap and readiness APIs.
9. Extend chart snapshots and market-aware cache identity.
10. Add shared realtime feed shards and authenticated fanout ingress.
11. Add entitlement enforcement, load tests, recovery tests, and runbooks.

### stock-indicator-dashboard

1. Replace A-share-only symbol parsing and types.
2. Add market-aware search and canonical result handling.
3. Migrate favorites to server-backed authenticated storage while preserving local A-share favorites.
4. Add HK and US historical charts and per-instrument adjustment availability.
5. Add custom-symbol request and bootstrap-progress UI.
6. Add currency, session, delay, source, and freshness presentation.
7. Add realtime only after the shared feed path and entitlements pass preview acceptance.

## First implementation slice

The first slice should prove the smallest complete historical path:

- One XHKG equity.
- One XNAS equity.
- Unadjusted daily bars only.
- Explicit HKD and USD.
- Provider symbol resolution.
- Immutable raw capture.
- Normalized publication with source lineage.
- One compact chart snapshot per instrument.
- Authenticated dashboard rendering.
- No realtime and no public access.

This slice passes only when the browser series matches the provider-derived normalized series day by day and both instruments can be rebuilt from retained raw responses.

## Immediate next actions

1. Keep the current A-share market-date changes isolated and complete or checkpoint them before cross-market implementation begins.
2. Write the identity, multi-currency daily-bar, adjustment-basis, and market-generation ADRs.
3. Choose the initial HK and US acceptance symbols and edge-case fixtures.
4. Run a live provider spike for identity lookup, 250 daily bars, adjustment behavior, timestamps, sessions, quotas, and credential renewal.
5. Record the provider comparison and select the controlled-pilot source.
6. Implement R0 before changing the dashboard allowlists.

## Definition of roadmap success

The roadmap succeeds when a verified HK or US symbol can be added without a deployment, produce a reproducible historical chart with correct currency and calendar semantics, optionally join a shared realtime subscription after entitlement checks, and remain fully traceable to its provider evidence without weakening the existing A-share service.
