# US earnings strategy research

A reproducible Python pilot using original SEC disclosures, consolidated US daily prices, a guidance-language baseline and a local text classifier. This project has no order-routing code. Capital allocation remains undecided.

## Current result

The latest run is pilot-v2. The 2024–2025 historical evaluation does not establish a reliable trading edge. Results after assumed costs:

| Strategy | Annualized return | Maximum drawdown | Average invested exposure | Closed trades |
|---|---:|---:|---:|---:|
| SPY buy and hold | 21.28% | -18.75% | 100.0% | — |
| Price only | 7.67% | -17.77% | 87.6% | 342 |
| Earnings + price | 4.16% | -5.71% | 19.5% | 63 |
| Raised guidance + price | 0.88% | -0.91% | 1.8% | 6 |
| ML guidance + price | 1.55% | -1.26% | 4.3% | 14 |

These portfolios have different market exposure. The low drawdowns of the guidance strategies mostly reflect holding cash. Cash earns zero in this test. Closed-trade counts include study-end liquidations: one guidance trade and one ML trade end before their normal 20-session horizon. Event statistics exclude these immature observations.

The ML filter's mean 20-session excess return over same-date SPY is +0.61 percentage points across 13 mature events. A month-cluster bootstrap interval runs from -2.71 to +3.91 percentage points. The guidance filter has only five mature later-period events in four signal months; no confidence interval is reported for that sample. None of these results supports a capital decision by itself.

## Deliverables

- [Executed notebook](notebooks/us-earnings-pilot-v2.ipynb)
- [Local HTML preview](artifacts/pilot-v2/us-earnings-pilot.html)
- [Results and ledgers](artifacts/pilot-v2/)
- [Registered strategy configuration](configs/pilot-v2.json)
- [Registered text-model configuration](configs/text-model-v2.json)
- [Independent source-price reconciliation](artifacts/pilot-v2/independent-audit.json)

The initial v1 notebook and outputs remain available for comparison. No strategy parameter was selected by searching for the best return.

## Reproduce

Use Python 3.12 or 3.13. The environment is pinned in uv.lock. From this directory:

    uv sync --default-index https://pypi.org/simple
    uv run python -m us_earnings backtest --config configs/pilot-v2.json
    uv run python -m us_earnings audit --config configs/pilot-v2.json
    uv run python -m us_earnings notebook --config configs/pilot-v2.json
    uv run pytest -q
    uv run ruff check src tests

Backtesting, auditing and notebook execution use cached data without network access. The executed notebook retains tables and three figures. Its HTML preview omits code inputs; the notebook keeps editable Python code.

For a fresh collection, provide ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY through the environment, or use a local credentials file:

    uv run python -m us_earnings collect --config configs/pilot-v2.json --credentials-file /path/to/local/.env

Only those two market-data keys are loaded from that file. They are used only with Alpaca's data endpoint and are never saved in source metadata, notebooks or logs. Set SEC_USER_AGENT to a descriptive research user agent when collecting SEC documents. No additional subscription was purchased for this run.

Raw data, prices, source texts, fitted models and generated artifacts are excluded from Git. Keep the local data directory to reproduce this exact snapshot; a future download can contain vendor revisions. Protocol hashes prevent silent changes to existing configurations. Use a new protocol/output subdirectory for a new experiment.

## Data and information timing

The sample contains 24 surviving companies selected for an implementation pilot, not a historical index membership sample. It therefore has selection and survivorship bias. There are 46,625 price rows covering 25 symbols including SPY, from August 2018 through December 2025. The strategy period is January 2019 through December 2025.

SEC submissions identify 683 Item 2.02 filings. The collector validates 681 financial exhibits; two other exhibits remain unclassified and excluded. Original exhibit bytes, normalized text, request URLs, fetch timestamps and SHA-256 hashes are retained locally. The events file includes every catalog filing and its status. Zero detected raises means no matching explicit language, not proof that management did not raise guidance.

The signal is evaluated at the close of the first SPY session strictly after the SEC filing date. Entry occurs at the next session open. This avoids using ambiguously represented API acceptance timestamps and can be later than the original earnings announcement. The study does not claim to trade the first public release instant.

[SEC API documentation](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) describes submissions and financial data. [Alpaca historical-bar documentation](https://docs.alpaca.markets/us/reference/stockbars) describes pagination, consolidated SIP coverage, symbol mapping and adjustments. All pages are paginated to completion.

## Strategy

The fixed confirmation requires the stock's close above its trailing 50-session moving average and its daily return above SPY's daily return. Compare:

1. SPY buy and hold.
2. Equal initial weights in the fixed company sample, bought in 2019 and held with evolving weights.
3. Daily price confirmation without an earnings filter.
4. An earnings event plus price confirmation.
5. Explicit raised annual guidance language plus price confirmation.
6. ML-classified raised guidance plus price confirmation, evaluated only from 2023 onward.

Each active strategy targets 10% of opening portfolio NAV per new position, holds at most ten positions, and cannot borrow. Entries rank by the previous session's relative return, with ticker as a tie-breaker. A held ticker cannot be bought again. Unused capacity stays in cash.

The primary holding horizon is 20 trading sessions including the entry day. A fixed stop is two trailing 20-session ATRs below the entry open. Opening gaps through stops fill at the worse open; entry-day stops are allowed. Costs reduce both entry and exit proceeds by 10 basis points per side. The study liquidates remaining positions at its final close and identifies them separately.

Alpaca prices use adjustment=all, incorporating split, dividend and spin-off adjustments. Positions are fractional adjusted total-return units. Stops approximate a risk rule on those units; they are not exact reconstructed historical broker fills. Raw-share corporate-action and dividend cash-flow replay is still required before execution claims.

Diagnostics use 5, 20 and 50 basis points per side, and 5/20/40-session horizons. These are reported without choosing a winner. Taxes, cash interest, financing, variable liquidity, spread dynamics and market impact are not modeled.

## AI model and label limitations

The local model is TF-IDF word unigrams/bigrams followed by balanced logistic regression. It fits only 2019–2022 disclosure text, using lexical weak labels for explicit raised annual guidance. It never sees returns as training targets. Predictions are saved only for filings from 2023 onward. The fixed threshold is 0.5 and scores are not calibrated probabilities.

This is a small machine-learning baseline, not a generative LLM or a numeric financial-statement reasoning system. External pretrained model downloads were unavailable in this environment. A modern LLM analyzing historical events can remember later outcomes; the local historical training boundary avoids that particular test design problem. It does not eliminate weak-label errors or issuer-specific language effects.

The parser intentionally abstains on numeric-only revisions and many implicit comparisons. It does not fully reconcile forecast periods, GAAP/non-GAAP bases, acquisitions, currency effects or operating assumptions. The broad annual-outlook label can include cash-flow outlook changes. A blinded human-labeled validation set and a normalized numeric guidance extractor remain necessary.

## Version history and evaluation status

Before any return inspection, pilot-v1 fixed entry, exit, cost, capacity and model parameters. A source audit corrected false positives such as historical EPS growth appearing near guidance tables.

After initial v1 results revealed unusually sparse guidance coverage, an inspection found omissions for FY25-style periods, low-end/midpoint wording and footnote digits. Pilot-v2 repairs 14 disclosure labels, from 79 to 93 explicit raises. Its classifier is refitted on repaired pre-2023 labels with identical settings. Trading rules and thresholds are unchanged. Original v1 outputs remain preserved; parser differences are listed in parser-repair-audit.csv.

The 2024–2025 period is a later historical evaluation, not a fully blind or prospective holdout. The universe was chosen today and parsing was inspected across years. Further performance-driven changes require a new protocol and a new evaluation sample.

## Validation and next decision

Behavioral tests cover event/entry timing, future-data invariance, missing exhibits, gap and entry-day stops, holding horizons, fees, cash/position limits, initial-capital drawdown, immature events, parser formats, low-sample intervals, and the text model's temporal training boundary.

The independent audit recomputes every saved trade return from cached OHLC prices, independently rebuilds the stop ATR, reconciles final NAV to realized P&L, and verifies all validated exhibit byte hashes. Both notebooks execute top-to-bottom with saved outputs. Rendered figures and the HTML preview are inspected locally.

Before choosing capital, broaden to a historical universe including delisted firms, validate numeric guidance with blinded labels, replay raw corporate actions and realistic fills, and conduct a prospective paper evaluation under frozen rules. No live orders or capital allocation were made.
