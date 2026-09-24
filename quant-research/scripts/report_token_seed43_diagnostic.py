"""Publish bounded diagnostic evidence and a source-bound completion receipt."""
import shutil

import numpy as np
import pandas as pd
from token_seed43_diagnostic import BASE, ROOT, checked, read

from quant_research.storage import file_hash, utc_now, write_json


def table(frame, digits=4):
    def cell(value):
        return f'{value:.{digits}f}' if isinstance(value, (float, np.floating)) else str(value)
    return '\n'.join(['| '+' | '.join(frame.columns)+' |', '| '+' | '.join(['---']*len(frame.columns))+' |']+
        ['| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False, name=None)])


def main():
    checked()
    receipt = read(ROOT / 'independent-verification.json')
    assert receipt['passed']
    assert '8 passed' in (ROOT / 'tests.log').read_text()
    write_json(ROOT / 'tests.json', dict(passed=True, count=8, log_sha256=file_hash(ROOT / 'tests.log'),
        command='PYTHONPATH=src:scripts .venv/bin/pytest -q tests/test_range_decomposition.py tests/test_forecast_audit.py'))
    paired = pd.read_csv(ROOT / 'paired.csv')
    summary = pd.read_csv(ROOT / 'summary.csv')
    cuts = pd.read_csv(ROOT / 'cuts.csv')
    concentration = pd.read_csv(ROOT / 'concentration.csv')
    selection = pd.read_csv(ROOT / 'original-selection-summary.csv')
    support = pd.read_csv(ROOT / 'support.csv')
    p43 = paired.query('seed == 43 and window == "2025transfer"')
    effects = p43.query('metric == "endpoint_mae"')[['comparison', 'horizon', 'base_mae', 'new_mae', 'delta', 'mae_change_pct', 'ci_low', 'ci_high']]
    shares = []
    for h in [2, 5]:
        g = p43.query('horizon == @h and comparison == "midpoint_vs_current"').set_index('metric')
        loss = p43.query('horizon == @h and comparison == "midpoint_vs_ce"').set_index('metric')
        shares.append(dict(horizon=h, center_share_pct=100*g.loc['center_contribution', 'delta']/g.loc['endpoint_mae', 'delta'],
            low_error_share_pct=100*.5*g.loc['low_mae', 'delta']/g.loc['endpoint_mae', 'delta'],
            loss_center_share_pct=100*loss.loc['center_contribution', 'delta']/loss.loc['endpoint_mae', 'delta']))
    shares = pd.DataFrame(shares)
    shares.to_csv(ROOT / 'contribution-shares.csv', index=False)
    drift = paired.query('seed == 43 and comparison == "midpoint_vs_ce" and metric == "center_error"')[[
        'window', 'horizon', 'delta', 'ci_low', 'ci_high']]
    warning = paired.query('seed == 43 and metric == "endpoint_mae"')[['window', 'comparison', 'horizon', 'mae_change_pct']]
    endpoint = paired.query('window == "2025transfer" and metric == "endpoint_mae"')[[
        'seed', 'comparison', 'horizon', 'mae_change_pct']]
    monthly = cuts.query('seed == 43 and window == "2025transfer" and comparison == "midpoint_vs_current" and cut == "month"')[[
        'horizon', 'group', 'dates', 'rows', 'endpoint_mae_delta', 'global_delta_contribution']]
    regimes = cuts.query('seed == 43 and window == "2025transfer" and comparison == "midpoint_vs_current" and cut in ["volatility_group", "trend_group"]')[[
        'horizon', 'cut', 'group', 'dates', 'rows', 'endpoint_mae_delta', 'global_delta_contribution']]
    concentrate = concentration.query('seed == 43 and window == "2025transfer"')[[
        'comparison', 'horizon', 'positive_dates', 'dates', 'total_delta', 'after_top5_removed', 'after_worst10_removed']]
    selection_change = []
    for (seed, h), g in selection.groupby(['seed', 'horizon']):
        g = g.set_index('arm')
        selection_change.append(dict(seed=seed, horizon=h,
            endpoint_mae_change_pct=100*(g.loc['midpoint', 'endpoint_mae']/g.loc['ce', 'endpoint_mae']-1),
            midpoint_crps_change_pct=100*(g.loc['midpoint', 'center_crps']/g.loc['ce', 'center_crps']-1)))
    original_changes = pd.DataFrame(selection_change)
    original_changes.to_csv(ROOT / 'original-selection-changes.csv', index=False)
    text = '''# Seed 43: continuation and midpoint-loss diagnostic

## Conclusion

Retain the midpoint-loss direction and the gains in seeds 17 and 29. Seed 43 has a consistent direction of forecast movement across the saved periods: the midpoint-loss continuation moves the predicted price-range center downward relative to matched token-only continuation in all three periods. In the 2025 transfer window this increases error, especially for the predicted lows.

The deterioration has two parts. Extra token-only training already worsens seed 43; the midpoint objective adds further deterioration. This is evidence about these fixed checkpoints and sampled paths. It does not identify a defective random seed, prove the objective always fails, or isolate the underlying optimization mechanism.

Keep the current calibrated reference preferred while retaining the new candidate and all three seeds. No checkpoint was reselected, prediction corrected or seed dropped.

## Scope and comparability

Recomputed 27 saved forecast sets: three predictor seeds × three checkpoint owners × three periods. Current means the original 2024-vintage checkpoint. CE means two additional epochs of token-only training. Midpoint means the matched two-epoch continuation with the 0.05 midpoint-distribution objective. Each set contains 64 sampled five-day OHLCVA paths per input.

Periods: 112 signal dates in 2024 H1, 120 in 2024 H2, and 76 in April–July 2025. H1 is reused development/selection data. The 2025 window was already inspected; this is retrospective diagnosis, not a fresh confirmation set. The sealed period beginning 2025-08-07 was not used.

For each seed and horizon, all three owners use the same known-label inputs with at least 16 legal paths per owner. Error averages weight stocks equally within a date and dates equally. Units are percentage points of signal close. Negative error changes are better. Calibrated intervals preserve the endpoint medians, so this raw-path point-error diagnosis applies to the calibrated profiles too; it does not re-estimate interval calibration.

## 1. Extra training and the new loss both contribute

Seed 43 high/low mean absolute error (MAE):

'''+table(effects)+'''

The absolute changes add: current → CE plus CE → midpoint equals current → midpoint. The percentage changes have different denominators and must not be added. At two/five days, token-only continuation worsens MAE by 1.90% / 3.89%; the midpoint continuation adds 2.02% / 3.04% versus CE. The resulting total is 3.96% / 7.04% versus current.

The 95% intervals above use 2,000 circular resamples of ten consecutive dates, conditional on these checkpoints and draws. They are pointwise, without a multiple-comparison correction. They do not measure variability across new training runs or new forecast samples.

## 2. Most extra endpoint error comes from range placement and predicted lows

We decompose the two endpoint medians into center = (high + low)/2 and half-width = (high − low)/2. Mean high/low absolute error equals the larger of the absolute center error and absolute half-width error. Averaging the two orders of replacing these components gives contributions that add exactly to the MAE change. This is arithmetic attribution, not causal identification or a deployable mixed model.

'''+table(shares, 2)+'''

Thus center placement accounts for 83.94% / 81.56% of the total deterioration and 92.05% / 94.51% of the loss-specific deterioration. Predicted-low errors account for 83.43% / 90.52% of the total increase in high/low mean MAE.

Seed 43 mean signed center error changes from −0.804 to −0.990 pp at two days and from −2.082 to −2.556 pp at five days. Relative to current, the mean high prediction changes only −0.028 / −0.043 pp, while the mean low prediction moves −0.345 / −0.905 pp. The implied width increases mainly toward the low side. Smaller average width underprediction does not imply better individual width estimates: its arithmetic contribution to endpoint error is also positive.

Actual sampled-midpoint medians and CRPS are scored separately from centers implied by endpoint medians. Those medians need not be equal. Both diagnostics agree on deterioration for seed 43 in 2025; direct midpoint CRPS is in the evidence tables.

## 3. Concentrated in April, but not confined to a few dates or stocks

'''+table(concentrate)+'''

The full midpoint continuation worsens seed 43 on 56/76 dates at both horizons. The isolated midpoint objective worsens it on 52/76 dates. Removing either the five largest positive daily differences or the worst consecutive ten-date block leaves all six seed-43 continuation comparisons worse. These removals are sensitivity checks only; no observations were removed from the reported primary results.

'''+table(monthly)+'''

All four months worsen versus current. April contributes about 71% / 68% of the net two/five-day error increase. Relative to CE, the five-day midpoint objective improves in May, but worsens in April, June and July.

The two/five-day cohorts contain 1,640 / 1,590 distinct instruments. Of these, 955 / 999 have a positive average difference versus current, but the median instrument appears only once. Individual-stock results are too sparse to support stock-specific rules. The five largest positive instrument contributions explain only about 8.1% / 6.6% of the total deterioration.

## 4. Input characteristics

Past volatility uses twenty adjusted-close log returns; trend uses the corresponding twenty-day adjusted-close return. All inputs end on the signal date. Volatility tercile thresholds are calculated solely from the 1,792 H1 inputs. This rule was fixed before computing the diagnostic cuts; the transfer outcomes were already known and this is not a prospective test.

'''+table(regimes)+'''

Each listed group worsens versus current at both horizons. High-volatility observations contribute about 59% / 62% of the total deterioration. Negative past-trend observations contribute about 78% / 73%. These are overlapping cuts, not additive independent causes or a validated routing policy.

The group error column averages within group and date, then over dates where the group exists. The global contribution column preserves the full-cohort date weights and sums to the overall difference within each cut. Group error multiplied by a simple overall row share need not equal its global contribution when group composition changes over dates.

## 5. What could earlier validation tell us?

The original loss selection used 896 inputs, eight per date, and selected midpoint loss on the mean midpoint CRPS across three seeds and two horizons. Its average advantage was only 0.1083%. Seed 43 improved on that criterion at both horizons; it was not the failing seed on the selection metric.

Original selection changes, midpoint versus CE:

'''+table(original_changes)+'''

On the larger, matched 1,792-input H1 calibration forecast set, seed 43 also improves endpoint MAE versus CE by 0.22% / 0.37%, while remaining worse than its original checkpoint by 1.17% / 0.32%. Token-only continuation was already worse than the original by 1.39% / 0.70%. This was a warning about continuation stability, not evidence that the midpoint objective should have been rejected.

A more specific warning is the consistent signed center movement relative to CE:

'''+table(drift)+'''

The downward movement is present in all three periods. Its sign alone is not a failure criterion: it can improve forecasts in a period where downward movement is helpful. In 2024 H2, midpoint continuation still improves seed 43 versus the original at both horizons, although its five-day loss-specific MAE worsens by 0.38%. The 2025 result therefore cannot justify retroactively selecting or rejecting a seed based on its identity.

Saved continuation histories contain training CE, sampled auxiliary loss and legal-path fractions, but no per-epoch out-of-sample price forecast trajectory. Both continuation arms used a fixed final two-epoch checkpoint. The original warm-start selection CE and continuation training CE are different quantities; their numerical levels do not establish improved validation. All three midpoint runs have a larger second-epoch sampled training midpoint CRPS on different sampled rows, so that trace is not a unique warning for seed 43. The saved artifacts cannot show whether an earlier continuation checkpoint would avoid the regression.

## 6. Controls, uncertainty and next experiment

Seeds 17 and 29 retain their 2025 gains:

'''+table(endpoint)+'''

The all-nine-model shared-input sensitivity preserves the finding: seed 43 worsens 3.97% / 7.14%, while seeds 17 and 29 improve at both horizons. Unequal legal-input cohorts do not explain the sign pattern. Legal-path fractions remain around 97% at two days and 93% at five days for seed 43; the error increase is not accompanied by a collapse in usable paths.

Retain the previously observed gains: full continuation improved high/low MAE in 10/12 seed/horizon cases across the two evaluation periods; the isolated midpoint loss improved 7/12. Those counts do not erase seed 43's larger downside, and the downside does not erase the other improvements.

The next bounded experiment should repeat forecast sampling from the same original, CE and midpoint checkpoints with additional fixed draw seeds, retaining all three training seeds and the existing date/stock cohort. Compare signed center movement and low-price MAE as well as overall error. This will test whether the observed drift is stable across sampled paths before changing the training objective again. If it persists, the next training comparison should save per-epoch price validation and test a constrained continuation, with its rule fixed on development data. Neither follow-up was launched here.

## Verification and artifacts

Ready within the reviewed scope, with the development-data and fixed-draw limitations above. Independent code reconstructed every raw-path endpoint, center/width decomposition and midpoint CRPS using explicit pairwise distances; reconciled prior 2024 H2 and 2025 scores; checked input-only features, all attribution rows, date/group weights, concentration removals, shared-seed cohorts, original selection and every paired estimate/interval. Eight focused tests passed. No neural training, forecasting or calibration fitting ran.

- [Detailed evidence](token-seed43-diagnostic-evidence-20260915.md)
- [Protocol](../artifacts/token-seed43-diagnostic-20260915-v1/protocol.json)
- [Independent verification](../artifacts/token-seed43-diagnostic-20260915-v1/independent-verification.json)
- [Final source-bound receipt](../artifacts/token-seed43-diagnostic-20260915-v1/final-verification.json)
- [Analysis script](../scripts/token_seed43_diagnostic.py)
- [Independent verifier](../scripts/verify_token_seed43_diagnostic.py)
'''
    assert chr(96) not in text
    doc = BASE / 'docs/token-seed43-diagnostic-20260915.md'
    doc.write_text(text)
    evidence = '# Seed 43 diagnostic: numerical evidence\n\nAll price-error units are percentage points of signal close. Negative changes improve error. H1 is reused development data.\n\n'
    for title, frame in [('Common input support', support), ('All-nine-model support', pd.read_csv(ROOT / 'all-seed-support.csv')),
            ('All seed and period component scores', summary), ('Seed 43 historical endpoint comparisons', warning),
            ('Seed 43 transfer paired estimates', p43), ('Original selection scores', selection),
            ('Saved training traces', pd.read_csv(ROOT / 'training-history.csv'))]:
        evidence += '## '+title+'\n\n'+table(frame)+'\n\n'
    evidence += 'Per-instrument and input-group contributions are retained in [cuts.csv](../artifacts/token-seed43-diagnostic-20260915-v1/cuts.csv). Individual instrument support is usually sparse.\n'
    (BASE / 'docs/token-seed43-diagnostic-evidence-20260915.md').write_text(evidence)
    entry = ('2026-09-15: [Seed 43 continuation diagnostic](docs/token-seed43-diagnostic-20260915.md) completed from 27 saved forecast sets. '
        'Its 2025 high/low MAE deterioration is mainly center placement (84% / 82% of the two/five-day increase), concentrated in predicted lows. '
        'Token-only continuation already worsens MAE 1.90% / 3.89%; midpoint loss adds 2.02% / 3.04%. '
        'The full regression spans 56/76 dates and survives removal of the worst five dates or ten-date block. '
        'Signed downward movement versus CE was present in both earlier periods, although H1 midpoint scores improved. '
        'Retain the loss direction and other-seed gains; keep the current calibrated reference preferred. '
        'Independent verification covered 140,433 raw-path score rows, 140,433 attributions and 756 paired estimates; eight tests passed. '
        'No training, inference, calibration, seed selection or sealed-holdout use. Next: fixed-checkpoint sampling replication.\n\n')
    for name in ['README.md', 'STATUS.md']:
        path = BASE / name
        original = path.read_text()
        if entry not in original:
            title, rest = original.split('\n', 1)
            path.write_text(title+'\n\n'+entry+rest.lstrip('\n'))
    for name in ['verify_token_seed43_diagnostic.py', 'report_token_seed43_diagnostic.py']:
        shutil.copy2(BASE / 'scripts' / name, ROOT / 'code' / name)
    write_json(ROOT / 'decision.json', dict(at=utc_now(), status='retain_midpoint_direction_current_reference_preferred',
        next_step='Replicate saved-checkpoint forecasts across additional sampling seeds before any new training objective comparison.',
        no_seed_dropped=True, current_defaults_unchanged=True, training_started=False, inference_started=False,
        sealed_holdout_used=False, all_diagnostic_jobs_finished=True))
    checked()
    files = {str(p): file_hash(p) for p in ROOT.rglob('*') if p.is_file() and p.name != 'final-verification.json'}
    for p in [doc, BASE / 'docs/token-seed43-diagnostic-evidence-20260915.md', *[BASE / 'scripts' / n for n in
              ['token_seed43_diagnostic.py', 'verify_token_seed43_diagnostic.py', 'report_token_seed43_diagnostic.py']]]:
        files[str(p)] = file_hash(p)
    files.update(read(ROOT / 'sources.json'))
    write_json(ROOT / 'final-verification.json', dict(passed=True, at=utc_now(), files=files,
        consumed_source_files_unchanged=len(read(ROOT / 'sources.json')), independent=receipt,
        focused_tests=8, neural_jobs_started=0, all_diagnostic_jobs_finished=True,
        defaults_changed=False, sealed_holdout_used=False))
    print('Report and receipt completed:', len(files), 'bound files', flush=True)


if __name__ == '__main__':
    main()
