# Midpoint loss: matched continuation protocol

The predictor still generates discrete coarse and fine market tokens. The existing frozen decoder turns those tokens into five daily OHLCVA bars. No price rule, auxiliary prediction head or extra input feature is added.

## Training objective

Keep the existing equally weighted five-day coarse/fine cross-entropy loss. Add a sampled-distribution loss for the midpoint of future highest high and lowest low, expressed as percentage-point returns from the signal-date close. Two-day and five-day horizons receive equal weight when both labels are known.

The auxiliary objective is population CRPS: expected absolute error minus half the expected distance between two independent forecast draws. Lower values reward both accurate placement and useful dispersion. For each of eight rows from a shuffled training batch, generate four actual token paths. The decoder is frozen and receives discrete codes, exactly as in inference.

For draw i, let A_i be its absolute midpoint error and D_ij its absolute midpoint distance from draw j. The detached score-function reward is:

R_i = A_i − mean over j ≠ i of D_ij − mean over j ≠ i of A_j.

Multiply R_i by the log probability of its sampled token prefix, then average over draws, known horizons and date-balanced rows. Differentiate only the token log probabilities. The leave-one-out absolute-error term is a baseline independent of draw i and reduces variance without changing the expected gradient. The two-day prefix includes two coarse/fine pairs; the five-day prefix includes five. Fine-token probabilities are conditioned on the actual sampled coarse tokens.

The auxiliary risk estimate uses the off-diagonal pairwise distance average. It estimates population CRPS without the finite-sample bias of the empirical-distribution score and may be negative for an individual set of four draws. The numerical surrogate multiplied by log probabilities is not itself a forecast-error score. Neither quantity should be compared directly with cross-entropy loss magnitudes.

All finite sampled paths enter the training risk, including paths with invalid OHLC ordering. There is no rejection or repair inside this loss. Evaluation separately measures legality and uses the existing requirement of at least 16 legal draws from 64. The objective and evaluation therefore differ in their handling of invalid paths; report this limitation and coverage explicitly.

## Matched experiment

- Three training seeds: 17, 29 and 43.
- Warm starts: the existing refreshed 2024-vintage checkpoints, trained through 2023 and selected using 2024 H1.
- Two arms: unchanged token loss, and token loss plus midpoint loss with coefficient 0.05.
- Two continuation epochs per arm, fixed final checkpoint, reset AdamW, learning rate 0.00003, batch size 256 and gradient clipping at 1.
- Each epoch draws 32 stocks per signal date from 1,902 training dates. Each fit sees 121,728 distinct stock-date rows across two epochs from a 354,007-row eligible pool.
- Seed-matched arms use exactly the same row order, exposure and initial predictor weights. Loading the auxiliary decoder preserves predictor RNG state. Auxiliary sampling uses its own seeded generator; likelihood evaluation has dropout disabled to match inference.
- Architecture stays at 3,720,448 parameters. History length, decoder, sampling temperature and token vocabularies remain fixed.

## Selection and evaluation

Validation uses 896 inputs across 112 dates in 2024 H1, eight stocks per date. Select coefficient 0.05 only when its date-equal midpoint CRPS, averaged equally over both horizons and all three seeds, is lower than the matched cross-entropy continuation. Otherwise select zero. This is reused development validation because the warm starts were already selected on this period.

Record and hash that choice before generating 2024 H2 forecasts. Evaluation covers 3,840 inputs across 120 dates, 32 stocks per date. Both predeclared arms generate 64 paths per input. Compare midpoint MAE/CRPS, average high/low MAE, range MAE/CRPS, raw endpoint interval coverage and interval score. Each seed/horizon uses the intersection of both arms' eligible inputs. Stocks are weighted equally within dates; dates and seeds receive equal weight. Cohorts can differ between seeds.

Also compare both continuations with already saved forecasts from their original checkpoints on a common three-arm cohort. This reference distinguishes the auxiliary loss effect from the effect of additional training. It does not participate in coefficient selection.

Uncertainty uses paired ten-date circular block bootstrap intervals with 2,000 replicates. Intervals crossing zero indicate uncertain magnitude; they are not an automatic veto against a consistent improvement. Report individual seeds, local regressions and changes in usable coverage. These intervals condition on the fitted models and sampled paths; they do not include model-selection uncertainty. Intervals are pointwise and are not adjusted for multiple comparisons.

No new 2025 outcomes enter this experiment. The sealed period beginning August 7, 2025 remains unused. Prior calibration factors are not transferred to new checkpoints. This is a bounded continuation experiment, not evidence about training from scratch, convergence, profitability or an untouched final test.

The fixed tokenizer's original pretraining date coverage remains unresolved. Both arms share it, so this comparison isolates the incremental objective change under that common dependency; it does not establish a fully point-in-time pretrained system. The 2024 H2 window also appeared in earlier development work, although it does not choose this experiment's coefficient.

## Verification

The mathematical test enumerates a two-outcome distribution and confirms that the expected sampled score-function gradient equals the exact population-CRPS gradient. The actual-model pilot checks finite, nonzero token-head gradients, frozen decoder weights, unchanged predictor weights before optimization and independence from future token labels during generation.

Independent verification checks sampled-token likelihoods against stepwise inference, causal decoder prefixes, identical training exposure, checkpoint reloads, every saved forecast chunk, exact token/price replays, explicit pairwise-distance CRPS, coverage masks, date weighting, paired intervals and the chronology of coefficient selection. Source hashes bind the data, initial weights, decoder and existing reference forecasts.
