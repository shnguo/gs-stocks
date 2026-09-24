# Return and stock-ranking objective

## Scope and status

Implements the agreed bounded comparison: baseline, BOLL/MACD/KDJ inputs, and the same inputs with a return/ranking auxiliary loss. Three seeds (17, 29, 43), four fixed continuation epochs and the same training rows for every arm. All nine fits and 12,828,672 historical five-day paths completed. Independent verification passed. The return/ranking arm is nominated for manual comparison; the main daily model remains unchanged.

## Results

The primary evaluation freezes forecast-only Top20 membership before attaching outcomes. All figures below are equal-model ensemble averages over 71 evaluation dates, from a pool of 192 sampled stocks per date before forecast-availability pairing.

| Arm | Return forecast MAE, percentage points | Observed Top20 extrema scenario | Top20 outcome 10th percentile |
| --- | ---: | ---: | ---: |
| Matched baseline | 4.2798 | 9.56% | 1.61% |
| Indicators | 4.3223 | 9.51% | 1.56% |
| Indicators + return/ranking loss | 4.0905 | 9.63% | 2.06% |

Return forecast MAE improves 4.42% versus baseline, with improvement on 51/71 dates (71.8%). The observed Top20 scenario improves 0.0736 percentage points on average and on 39/71 dates (54.9%); its date-block interval spans zero. The Top20 10th-percentile outcome improves 0.4458 percentage points. These are price-extrema scenarios, not executed profits.

On the 49 dates with fully observed ensemble Top20 outcomes in every arm, mean Top20 scenarios are essentially tied: 9.7864% baseline versus 9.7805% candidate. Return MAE still improves, from 4.5678 to 4.3674 percentage points. This supports retaining the return/ranking objective for improved return estimates and further comparison; a higher average stock-selection return is not established. Missing Top20 outcomes remain visible (about 0.35–0.37 per date) and are never replaced by lower-ranked stocks. High-volatility stocks still account for about 76% of the selected group.

The candidate was nominated using selection-period Top20 lift, without evaluation-based tuning. The historical input-indicator-only gain from the preceding small comparison did not reproduce as an ensemble gain here; all candidates and per-seed tradeoffs are retained.

[Primary report](../artifacts/token-ranking-20260916-v2-frozen-top20/report.md) · [Frozen-Top20 verification](../artifacts/token-ranking-20260916-v2-frozen-top20/verification.json) · [Path verification](../artifacts/token-ranking-20260916-v2-verification.json)

The original five-day OHLCVA token output and published score remain: sell reference price / buy reference price − 1 − 0.25%. The buy reference is the median predicted T low. The exit date is the modal predicted peak day among T+1 through T+4, with earliest-date ties. The sell reference is the median predicted high on that selected day. Ensemble reporting averages each model's medians and normalized date frequencies equally.

## Learned objective

Baseline and indicator control: token CE + 0.05 sampled midpoint loss.

New arm: the same loss + 0.01 reference-return Huber cost + 0.01 within-date pairwise ordering cost. Return differences are normalized by five percentage points; Huber transitions at one normalized unit. Pair ordering uses a stable logistic penalty, weighted by the realized return gap capped at one normalized unit. Equal outcomes carry no arbitrary ordering label.

Training draws two independent bags of eight actual token paths per chosen stock. Each bag uses its own median prices and modal exit date. Observed return uses the actual high on that already-selected date divided by actual T low, less the same cost. Only complete mature five-session labels participate; stock pairs must share a signal date. Four rows per date receive this auxiliary loss, rotating with the fixed epoch shuffles.

Discrete tokens, medians and date selection stay on the actual inference path. Their costs are detached, and gradients flow through the sampled bags' sequence log probabilities. A leave-one-bag-out baseline reduces variance without using its own action. Pair costs contribute to both participating stocks' score gradients but count once in the reported objective. Invalid bags are penalized rather than silently dropped. No separate ranking head or indicator-threshold trading rule is introduced.

This finite-bag training surrogate differs from final 64-path inference, and each model trains individually before equal-model ensemble reporting. Those approximations are explicit parts of the protocol.

## Data and comparisons

| Partition | Rows | Signal dates | Stocks per date |
| --- | ---: | ---: | ---: |
| Training | 7,969 | 251 | 29–32 |
| Selection | 8,640 | 45 | 192 |
| Evaluation | 13,632 | 71 | 192 |

Training IDs are identical to the preceding indicator experiment. Selection/evaluation retain the same dates and expand to the archived 192-stock exchange-stratified pool, selected without looking at future prices. Feature normalization uses training histories only. Five-day labels are embargoed at split boundaries. These are reused development dates with historical-source and tokenizer-pretraining limitations, not an untouched test.

The primary comparison freezes eligibility, ranking and Top20 membership using forecasts alone. Future-label availability never replaces a Top20 stock; reports show known/missing outcome counts and a complete-Top20-date sensitivity view. It uses a shared forecast-eligible stock-date cohort across all three arms and all three seeds; native per-model universes are also retained. The earlier known-label-cohort tables remain a separate diagnostic. Reports show actual Top20, not top 20% of a small sample. Metrics include return forecast MAE, Top20 extrema scenario, lift over the paired universe, rank correlation, lower-tail Top20 outcomes, adverse price movement, volatility concentration, exchange concentration and forecast availability. Results are also broken down by seed and period. Date-block intervals convey uncertainty without an automatic significance gate.

The shadow candidate is nominated using selection-period ensemble Top20 lift only. The evaluation period does not tune loss weights, choose epochs or select the candidate. Nomination is for comparison, not automatic replacement of the current model.

## Inference runtime

The initial v1 execution was stopped during baseline forecasting because it repeatedly recomputed identical causal prefixes. Its partial artifacts remain intact; no candidate outcomes had been inspected. The v2 comparison restarts all arms under the same cached inference backend and unchanged cohorts, losses and training budgets.

The cache reuses historical attention keys/values and frozen decoder prefixes. Sampling retains the original sorted multinomial method, local seed, temperature 1, top-p 1 and 64 paths. Cache tests compare each causal step's logits with full-context inference, check nonzero auxiliary inputs and missing future inputs, and reject context overflow. Actual MPS probes matched sampled tokens and legal-path masks; decoded values agreed within roughly two millionths in normalized units. Different floating-point kernel shapes need not remain bit-identical for every possible draw; independent verification records replay and original-backend comparisons.

## Manual commands

Run or verify/reuse the historical comparison, independently audit its paths and publish the corrected frozen-Top20 report:

    /Users/guo/Documents/stocks/quant-research/scripts/run_token_ranking.sh

Run the usual close validation, main model incremental training and report, then the nominated candidate comparison:

    /Users/guo/Documents/stocks/quant-research/scripts/run_daily_token_comparison.sh

The ordinary manual script also accepts an explicit candidate configuration through --shadow-config. No scheduler is created.

To compare against an existing completed run without refreshing data or repeating main-model training:

    /Users/guo/Documents/stocks/quant-research/scripts/run_token_ranking_shadow.sh --config /Users/guo/Documents/stocks/quant-research/artifacts/token-ranking-20260916-v2-shadow.json --run /Users/guo/Documents/stocks/quant-research/artifacts/daily-token-live-v1/runs/2026-09-15

The candidate configuration is generated only after the experiment finishes. Candidate weights remain frozen research checkpoints, while the main model continues its existing incremental updates. This operational comparison therefore also reflects different training recency. Frozen candidate forecasts, checkpoint/source hashes, current and candidate ranks, reference prices and selected dates are saved separately. Nothing changes the main report pointer or main checkpoint.

Reports completed at or after T's 09:15 Shanghai opening auction are marked late and excluded from prospective scoring. Subsequent manual runs review older timely predictions only after all five sessions mature, preserve their original Top20 and exit dates, and report missing outcomes without substituting lower-ranked stocks. Day-bar extrema remain theoretical price scenarios, not guaranteed fills or executed profits.

## Artifacts

- Protocol: ../configs/token-ranking-v2.json
- Experiment: ../artifacts/token-ranking-20260916-v2/
- Primary report: ../artifacts/token-ranking-20260916-v2-frozen-top20/report.md
- Known-label diagnostic: ../artifacts/token-ranking-20260916-v2/report.md
- Independent receipt: ../artifacts/token-ranking-20260916-v2-verification.json
- Candidate configuration: ../artifacts/token-ranking-20260916-v2-shadow.json
- Forward comparison store: ../artifacts/token-ranking-shadow-v1/

## Verification results

- 61 focused tests passed; one optional tokenizer-environment test was skipped. Real training, decoding and replay used the pinned environment with the actual tokenizer dependencies.
- An exact enumerable gradient test checks the score-function estimator, including pair contributions and action-dependent exit outcomes. A real-data/decoder probe produced finite, nonzero gradients.
- Independent path verification recalculated 256,828 reference-price records, checked 384 feature windows and replayed 576 inputs exactly under the cached backend.
- Original versus cached sampling matched tokens in 17/18 probe batches; the decoder maximum discrepancy was 0.0000073 in normalized units, with identical legality masks on those probes. Cached inference is numerically consistent, not universally bit-identical to the original backend. Every experiment arm uses the same cache backend, and its own saved forecasts replay exactly.
- The frozen-Top20 audit checked 261,300 paired rows, including forecast-only eligibility, fixed ranks, missing outcomes and selection-only nomination.
- Main model configuration, checkpoints/report pointers covered by the live-state snapshot remain unchanged. No scheduler was added.

## Manual comparison smoke run

The existing September 15 daily run was replayed without repeating its training or refreshing market data. The candidate generated 1,033,920 paths for 5,385 inputs and ranked 5,331 stocks; 5,322 stocks are shared with the current model's 5,328-stock ranking. Its publication time is September 17 UTC, after the first forecast session began, so it is explicitly late and permanently excluded from prospective scoring.

Independent verification recalculated all 5,331 candidate ranks/reference prices and exactly replayed 96 inputs. Main state hashes remained unchanged. Repeating both the historical comparison command and the manual candidate command reused completed artifacts without training or forecasting again. All task processes have finished.

A float32 normalization-layout mismatch found during the first manual attempt was fixed by copying each historical array in C order, matching the existing daily input builder. The exact mean/scale checks remain in place, and a dedicated regression test covers this case. The failed attempt and its error evidence are retained separately.

[Candidate comparison report](../artifacts/token-ranking-shadow-v1/runs/2026-09-15/report.md) · [Manual comparison verification](../artifacts/token-ranking-shadow-v1/verification/2026-09-15.json)

## Follow-up: matched incremental bridge

The isolated [matched incremental bridge](token-ranking-incremental-20260917.md) completed on September 17. It updated both historical baseline and candidate on the same September 15 bootstrap rows and verified a fixed 192-stock forecast sample. The frozen comparison described above remains unchanged; the bridge does not automatically promote or schedule the updated candidate.
