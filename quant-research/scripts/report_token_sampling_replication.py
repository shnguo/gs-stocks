"""Render verified sampling-replication evidence and bind the finished artifacts."""
import shutil

import numpy as np
import pandas as pd
from token_sampling_replication import BASE, ROOT, checked, read

from quant_research.storage import file_hash, utc_now, write_json


def table(frame):
    def cell(x):
        if isinstance(x, (float, np.floating)):
            return '—' if np.isnan(x) else f'{x:.4f}'
        return str(x)
    return '\n'.join(['| '+' | '.join(frame.columns)+' |', '| '+' | '.join(['---']*len(frame.columns))+' |'] +
        ['| '+' | '.join(cell(x) for x in row)+' |' for row in frame.itertuples(index=False, name=None)])


def main():
    checked()
    replay = read(ROOT / 'replay-verification.json')
    arithmetic = read(ROOT / 'arithmetic-verification.json')
    assert replay['passed'] and arithmetic['passed']
    assert '9 passed' in (ROOT / 'tests.log').read_text()
    write_json(ROOT / 'tests.json', dict(passed=True, count=9, log_sha256=file_hash(ROOT / 'tests.log'),
        command='PYTHONPATH=src:scripts .venv/bin/pytest -q tests/test_forecast_audit.py tests/test_range_decomposition.py tests/test_token_transformer.py::test_sampling_retains_paths_uses_local_seed_and_rolls_context'))
    paired = pd.read_csv(ROOT / 'paired.csv')
    selected = paired.query('cohort == "all27"')
    average = selected.query('draw == 0 and metric == "endpoint_mae"')
    per_seed = average.query('seed != 0 and comparison == "midpoint_vs_current"')[[
        'seed', 'horizon', 'baseline', 'candidate', 'relative_change_pct', 'delta', 'ci_low', 'ci_high']]
    seed43 = selected.query('seed == 43 and metric == "endpoint_mae"')[[
        'draw', 'horizon', 'comparison', 'relative_change_pct', 'delta', 'ci_low', 'ci_high']]
    drift = selected.query('seed == 43 and comparison == "midpoint_vs_ce" and metric == "center_bias"')[[
        'draw', 'horizon', 'delta', 'ci_low', 'ci_high']]
    pooled = average.query('seed == 0')[[
        'horizon', 'comparison', 'relative_change_pct', 'delta', 'ci_low', 'ci_high']]
    sensitivity = paired.query('cohort == "within_draw_seed" and draw == 0 and metric == "endpoint_mae"')[[
        'seed', 'horizon', 'comparison', 'relative_change_pct']]
    # Validate the prose's sign claims before publishing them.
    main43 = selected.query('seed == 43 and draw != 0 and comparison == "midpoint_vs_current" and metric == "endpoint_mae"')
    loss43 = selected.query('seed == 43 and draw != 0 and comparison == "midpoint_vs_ce" and metric == "endpoint_mae"')
    assert len(main43) == len(loss43) == 6 and (main43.delta > 0).all() and (loss43.delta > 0).all()
    assert (drift.delta < 0).all() and (per_seed[per_seed.seed.isin([17, 29])].delta < 0).all()
    ranges = main43.groupby('horizon').relative_change_pct.agg(['min', 'max']).reset_index()
    ranges.to_csv(ROOT / 'seed43-draw-range.csv', index=False)
    report = '''# Frozen-checkpoint sampling replication

## Conclusion

Seed 43's deterioration persists across all three forecast sampling runs. Its downward range-center movement and extra low-price error are not explained by the original single draw set. This supports a stability problem in the learned continuation checkpoints on this development period; it does not identify the causal optimizer mechanism or establish that this training seed is intrinsically bad.

Retain the midpoint-loss direction and the gains in seeds 17 and 29. Keep the current calibrated reference preferred. No training seed or draw replicate was discarded, and no production/default model changed.

## Experiment

Reused the nine original forecast sets and completed eighteen new sets: original, token-only continuation and midpoint-loss continuation checkpoints, each with training seeds 17, 29 and 43, under two additional forecast RNG seeds. Decoder, predictor weights, input IDs, sampling settings, four-input batch shape and 64 paths per input were unchanged. The additional RNG seed ranges are disjoint from each other and from the original.

All 2,801,664 new five-day OHLCVA paths are saved. Together with the original forecasts, the analysis covers 4,202,496 paths across 27 sets. There were no training runs or calibration fits.

The same 2,432 inputs cover 76 signal dates from April 1 through July 22, 2025, with labels ending July 29. The primary cohort requires known labels and at least 16 legal paths in every set. It contains 2,376 two-day inputs and 2,287 five-day inputs on all 76 dates. This excludes only three/seven inputs from the previous all-nine-model shared cohort. A separate within-draw/seed comparison checks sensitivity to that restriction.

Stocks have equal weight within date, dates have equal weight, and sampling replicates have equal weight. We average the errors from each 64-path forecast; this is neither an ensemble of forecast prices nor a pooled 192-path distribution. Negative error changes are better. Absolute differences are percentage points of the signal close. In the tables, draw 0 denotes the three-draw score average, and seed 0 denotes the three-training-seed score average.

## Average result by training seed

Midpoint-loss continuation versus the original/current checkpoint:

'''+table(per_seed)+'''

The gains in seeds 17 and 29 remain. Seed 43's larger downside makes the equal-seed mean worse by 0.67% / 1.29% at two/five days. The two-day date-bootstrap interval crosses zero; the five-day interval is slightly above zero. These intervals are conditional on this fixed collection of models and draws and are not adjusted for multiple comparisons.

## Seed 43: both continuation effects reproduce

'''+table(seed43)+'''

Across the three draws, total high/low MAE deterioration ranges from 3.99% to 4.48% at two days and 7.16% to 8.15% at five days. Relative to matched CE continuation, the midpoint objective worsens it in all six draw/horizon cases: 1.80–2.03% at two days and 2.85–3.78% at five days.

On the three-draw average, token-only continuation adds 2.35% / 4.43% error relative to current; midpoint loss adds 1.92% / 3.23% relative to CE. Their combined result is +4.31% / +7.81% versus current. The percentage changes have different denominators and must not be added.

Original-draw numbers differ slightly from the preceding report because every row here uses the stricter common 27-set cohort. The original paths were reused unchanged and their row-level scores reconcile exactly with the preceding diagnostic.

## The downward shift persists

Change in implied range-center bias for seed 43, midpoint loss versus CE:

'''+table(drift)+'''

The mean center moves downward by 0.129 / 0.325 pp at two/five days relative to CE, with the same sign in every draw. Relative to current, the total downward movement is 0.194 / 0.483 pp. On average, seed 43's predicted-low MAE increases 8.46% / 15.91% versus current, and 3.51% / 6.49% versus CE. Direct sampled-midpoint CRPS also worsens 1.87% / 3.27% versus CE. The finding therefore appears in the sampled midpoint distribution as well as the endpoint medians.

The other seeds do not show the same average loss-specific center movement: seed 29 moves upward at both horizons, while seed 17's movement is small. Seed 17's two-day loss-specific MAE improvement is weak and flips slightly positive in one draw (+0.075%); its three-draw mean still improves. Seed 29 improves against CE at both horizons in every draw. This preserves the distinction between a useful direction and its sensitivity to training initialization/checkpoint state.

## Equal-seed average and cohort sensitivity

'''+table(pooled)+'''

The isolated loss-specific average worsens 0.30% / 0.34% versus CE, with date intervals crossing zero. Both the individual-seed gains and the pooled downside remain relevant. Three draw replicates on the same dates are not three independent market tests and must not be counted as new independent scenarios supporting or rejecting the method.

Within-draw/seed sensitivity, averaged across draws:

'''+table(sensitivity)+'''

## Next optimization

Further resampling is no longer the first priority: the current bounded replication reproduces the main finding. The next controlled training experiment should focus on continuation stability and checkpoint selection, while keeping the model architecture and useful midpoint objective.

Specifically, save every continuation epoch and evaluate price-space metrics, signed center bias and low-price error on predeclared development subperiods. Compare selecting a checkpoint by midpoint-distribution validation with the existing fixed-final continuation, including the original checkpoint as an explicit baseline. Keep all training seeds and fix the selection rule before evaluating later dates. The earlier artifacts saved no per-epoch price-validation trajectory, so this experiment is needed to determine whether earlier stopping can preserve the gains without the seed-43 drift. No such training was started here, and this diagnosis is not evidence that earlier stopping will necessarily solve it.

## Verification and limits

Independent reconstruction passed for all 126,550 usable score rows and all 1,728 paired estimates and date-bootstrap intervals. Every one of 16,416 saved chunks was checked against its row IDs, path array, tokens, legality mask and source hash. Forty-five exact checkpoint-reload replays reproduced 11,520 token/price paths, including the first and last batch of every new set. Nine focused tests passed. Inputs, source forecasts, checkpoints and decoder hashes remained unchanged.

Ready within this scope. This is previously examined development data, not a fresh market holdout. Three RNG replicates test sampling sensitivity; they do not quantify uncertainty from retraining or establish performance in another market period. The date intervals are pointwise and conditional. Existing tokenizer pretraining-provenance limitations remain. The sealed period beginning August 7, 2025 was not accessed.

- [Method](token-sampling-replication-method.md)
- [Detailed evidence](token-sampling-replication-evidence-20260915.md)
- [Frozen protocol](../artifacts/token-sampling-replication-20260915-v1/protocol.json)
- [Independent arithmetic](../artifacts/token-sampling-replication-20260915-v1/arithmetic-verification.json)
- [Path and replay verification](../artifacts/token-sampling-replication-20260915-v1/replay-verification.json)
- [Final receipt](../artifacts/token-sampling-replication-20260915-v1/final-verification.json)
'''
    assert chr(96) not in report
    doc = BASE / 'docs/token-sampling-replication-20260915.md'
    doc.write_text(report)
    evidence = '# Sampling replication: detailed evidence\n\nDraw 0 is the equal-draw score average; seed 0 is the equal-training-seed score average. No prices or paths are pooled. Negative error changes are better.\n\n'
    for name, frame in [('Input support', pd.read_csv(ROOT / 'support.csv')),
                        ('All primary comparisons', selected),
                        ('Score means', pd.read_csv(ROOT / 'summary.csv')),
                        ('Legal-path coverage', pd.read_csv(ROOT / 'coverage.csv'))]:
        evidence += '## '+name+'\n\n'+table(frame)+'\n\n'
    (BASE / 'docs/token-sampling-replication-evidence-20260915.md').write_text(evidence)
    decision = dict(at=utc_now(), status='sampling_replication_confirms_seed43_drift',
        retained='Midpoint objective direction and other-seed improvements; current calibrated reference remains preferred.',
        next_step='Controlled continuation with saved epoch checkpoints and predeclared price-space validation selection.',
        training_runs=0, calibration_fits=0, forecasts_completed=18, reused_forecasts=9,
        all_jobs_finished=True, no_seed_dropped=True, defaults_changed=False, sealed_holdout_used=False)
    write_json(ROOT / 'decision.json', decision)
    entry = ('2026-09-15: [Frozen-checkpoint sampling replication](docs/token-sampling-replication-20260915.md) completed. '
        'Eighteen new forecast sets (2,801,664 paths) and nine reused sets reproduce seed 43\'s regression across all three draws. '
        'Its mean high/low MAE worsens 4.31% / 7.81% versus current at two/five days; seeds 17 and 29 improve 0.48% / 1.06% and 1.89% / 3.16%. '
        'Loss-specific downward center movement persists; a single unlucky draw set does not explain it. '
        'Retain the objective direction and other-seed gains, keep the current calibrated reference preferred, and prioritize continuation/checkpoint stability. '
        'Independent verification covered 126,550 scores, 1,728 paired estimates, 16,416 chunks and 45 exact replays; nine tests passed. '
        'All jobs finished; no training, calibration fits, seed removal, default changes or sealed-holdout use.\n\n')
    for name in ['README.md', 'STATUS.md']:
        path = BASE / name
        original = path.read_text()
        if entry not in original:
            title, rest = original.split('\n', 1)
            path.write_text(title+'\n\n'+entry+rest.lstrip('\n'))
    for name in ['verify_token_sampling_replication.py', 'report_token_sampling_replication.py']:
        shutil.copy2(BASE / 'scripts' / name, ROOT / 'code/scripts' / name)
    checked()
    files = {str(p): file_hash(p) for p in ROOT.rglob('*') if p.is_file() and p.name != 'final-verification.json' and '__pycache__' not in p.parts}
    for name in ['docs/token-sampling-replication-20260915.md', 'docs/token-sampling-replication-method.md',
                 'docs/token-sampling-replication-evidence-20260915.md', 'scripts/token_sampling_replication.py',
                 'scripts/verify_token_sampling_replication.py', 'scripts/report_token_sampling_replication.py']:
        files[str(BASE / name)] = file_hash(BASE / name)
    files.update(read(ROOT / 'sources.json'))
    write_json(ROOT / 'final-verification.json', dict(passed=True, at=utc_now(), files=files,
        consumed_sources_unchanged=len(read(ROOT / 'sources.json')), arithmetic=arithmetic, replay=replay,
        tests_passed=9, all_jobs_finished=True, new_forecast_sets=18, training_runs=0,
        calibration_fits=0, defaults_changed=False, sealed_holdout_used=False))
    print('Report and final receipt completed', len(files), 'bound files', flush=True)


if __name__ == '__main__':
    main()
