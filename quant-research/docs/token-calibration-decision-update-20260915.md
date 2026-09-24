# Current research decision: retain interval calibration

Date: 2026-09-15. This addendum records the user-directed interpretation after the completed interval-calibration experiment. Original measurements and its preregistered gate result remain historical evidence.

## Decision

Retain date-weighted interval calibration as the preferred research method for interval reporting. A change that helps across most tested scenarios should be retained with its measured limitations; a confidence interval crossing zero in one window does not by itself negate the observed improvement.

Across 2023 H2 and 2024 H2, mean coverage rises from roughly 67–70% to 77–78%, nearer the nominal 80%. Mean interval score improves in all four window/horizon combinations (1.17–1.89%), and all three predictor seeds improve on the mean score. Median forecasts and point MAE are exactly unchanged.

The tradeoff is wider intervals: roughly 27% at two days and 19% at five days. The 2024 H2 minimum-price interval score worsens 0.65% at two days and 0.35% at five days. The 2024 mean-score confidence intervals cross zero. These qualify the scope and certainty of the improvement.

The original cross-window strict gate was not met. That remains true as a record of the original rule; it is not the current judgment of research usefulness. Current status: useful within the tested scope, preferred research calibration candidate.

## Application

Calibration factors belong to their predictor checkpoint. New predictor weights need calibration fitted on their own earlier validation forecasts. The existing factors must not be copied to a refreshed predictor. This decision does not substitute for prospective testing or change the daily strategy model.

## Next bounded comparison

Compare old predictors trained through 2021 with predictors refreshed through 2022 for 2023 H2 and through 2023 for 2024 H2. Keep the 3.72 million parameter model, adapted decoder, equal output loss and 64-path full-distribution sampling fixed. Evaluate maximum-high and minimum-low point errors separately, with range and distribution scores as supporting evidence.

Evidence: [completed calibration report](token-interval-calibration-20260915.md). Historical decision: artifacts/token-interval-calibration-20260915-v1/research-decision.json. Updated experiment protocol: configs/token-training-refresh-v1.json.
