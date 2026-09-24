"""Render verified refresh evidence without replacing historical decisions."""
import pandas as pd
from token_training_refresh_run import BASE, ROOT, SEEDS, read


def table(frame, columns):
    output = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    output.extend('| ' + ' | '.join(str(x) for x in row) + ' |' for row in frame)
    return '\n'.join(output)


def main():
    verification = read(ROOT / 'independent-verification.json')
    assert verification['passed']
    decision = read(ROOT / 'research-decision.json')
    paired = pd.read_csv(ROOT / 'results/paired.csv', dtype={'seed': str})
    mean = paired[paired.seed == 'mean']
    point = mean[(mean.metric == 'mae') & mean.target.isin(['maximum', 'minimum', 'range'])]
    extrema = mean[(mean.metric == 'mae') & mean.target.isin(['maximum', 'minimum'])]
    per_seed = paired[(paired.metric == 'mae') & paired.target.isin(['maximum', 'minimum']) & (paired.seed != 'mean')]
    metrics = []
    for row in point.itertuples():
        metrics.append([row.window, row.horizon, row.target, f'{row.old:.4f}', f'{row.refreshed:.4f}',
            f'{100 * row.relative_change:+.2f}%', f'[{row.ci_low:+.4f}, {row.ci_high:+.4f}]'])
    combined = []
    for row in mean[(mean.metric.isin(['mae', 'crps'])) & (mean.target == 'high_low')].itertuples():
        combined.append([row.window, row.horizon, row.metric, f'{row.old:.4f}', f'{row.refreshed:.4f}',
            f'{100 * row.relative_change:+.2f}%', f'{100 * row.date_win_fraction:.1f}%'])
    training, coverage = [], []
    for window in ['2023h2', '2024h2']:
        for seed in SEEDS:
            root = ROOT / window / f'predictors/seed{seed}/dense_3720k_equal'
            summary = read(root / 'training/summary.json')
            training.append([window, seed, summary['training_dates'], summary['known_training_pool'],
                summary['unique_examples_visited'], summary['total_examples'], summary['selected_epoch'],
                summary['epochs'], 'reused' if (root / 'training/reuse.json').exists() else 'new fit'])
            cov = pd.read_csv(root / 'comparison/coverage.csv')
            coverage.append(cov.assign(window=window, seed=seed))
    cov = pd.concat(coverage).groupby(['window', 'horizon', 'variant']).agg(
        inputs=('inputs', 'sum'), known=('known', 'sum'), usable=('usable', 'sum'),
        valid_paths=('valid_paths', 'sum'), total_paths=('total_paths', 'sum')).reset_index()
    cov_rows = [[r.window, r.horizon, r.variant, r.usable, r.known, f'{100 * r.valid_paths / r.total_paths:.2f}%']
                for r in cov.itertuples()]
    text = '\n\n'.join([
        '# Training refresh: high- and low-price forecast comparison',
        'Date: 2026-09-15. Completed retrospective research on two historical windows and three training seeds.',
        '## Current conclusion',
        decision['conclusion'],
        f'Maximum-high and minimum-low MAE improve in {(extrema.delta < 0).sum()} of {len(extrema)} window/horizon/target combinations after averaging the three seeds, and in {(per_seed.delta < 0).sum()} of {len(per_seed)} individual seed combinations. These counts describe the tested cases, not the probability of future success.',
        decision['limitations'],
        '## What was compared',
        'Control: the exact three predictors trained on 2016–2021 and selected on 2022 H1, using archived forecasts. Refreshed: train on 2016–2022 and select on 2023 H1 for 2023 H2; train on 2016–2023 and select on 2024 H1 for 2024 H2. Every future label stays inside its partition. Each seed uses the same preselected 32 stocks per signal date, 119 dates in 2023 H2 and 120 dates in 2024 H2.',
        'Both sides use the same 3,720,448 parameter autoregressive token Transformer, 60 history bars, equal five-day token loss, fixed adapted decoder, and 64 sampled paths with temperature 1 and full token distribution. Generation batches contain four inputs and use seed 17 plus batch offset. The decoder was trained through 2021 and selected on 2022 H1. Its weights are identical on both sides.',
        'Four matching completed predictor fits were reused. Two new fits were completed for 2024 H2, seeds 29 and 43. For 2023 H2, saved generated tokens were re-decoded with the fixed adapted decoder after validating source hashes; first-batch generation and decoded prices replay exactly. The 2024 H2 forecasts were newly generated.',
        'This tests the practical refresh package. It jointly changes recent-history coverage, dataset size, optimization exposure, and the validation period used for checkpoint selection. It does not identify the isolated causal effect of newer observations. Both evaluation windows were examined in earlier research; they are not untouched tests. Official tokenizer pretraining coverage remains unresolved.',
        '## Point forecast errors',
        'Point forecasts are medians across legal sampled paths of the future maximum high, minimum low, or high-minus-low range. Errors and confidence-interval endpoints are percentage points of the signal-date close. For example, 1.0 pp is a price error of 1 on a signal close of 100. Relative changes below zero mean lower error. No buy/sell rules enter these targets.',
        table(metrics, ['Window', 'Days', 'Target', 'Old MAE, pp', 'Refreshed MAE, pp', 'Change', '95% CI of difference, pp']),
        '## Combined high/low evidence',
        'MAE measures the median point forecast. CRPS scores the full sampled distribution; smaller is better. Each date has equal weight, then seeds and the high/low targets have equal weight. Date win rate is the fraction of dates with a lower averaged error, not a trading win rate.',
        table(combined, ['Window', 'Days', 'Metric', 'Old', 'Refreshed', 'Change', 'Dates improved']),
        'The 95% intervals use 2,000 circular block bootstrap samples with blocks of 10 dates. They preserve local time dependence but are conditional on these three trained seeds and historical windows; they do not capture all model-selection or market-regime uncertainty. Multiple target comparisons are descriptive.',
        '## Data coverage and training completion',
        table(training, ['Window', 'Seed', 'Train dates', 'Known pool', 'Unique visited', 'Total presentations', 'Best epoch', 'Epochs', 'Fit']),
        'A training epoch visits 32 stocks per date with rotating coverage; selection uses up to 64 per date. At least 16 of the 64 paths must be legal for a row/horizon to be scored. Comparisons use the intersection of usable inputs between the two predictors within each seed. The following coverage counts are totals across the three seeds, so the same stock/date can appear three times.',
        table(cov_rows, ['Window', 'Days', 'Predictor', 'Usable inputs', 'Known-label inputs', 'Legal paths']),
        '## Calibration decision carried forward',
        'Date-weighted interval calibration remains the preferred research interval method because it improves average coverage and interval score across most tested scenarios. The earlier strict-gate result remains preserved as historical evidence. This comparison uses raw forecast distributions to measure the predictor refresh. The 18 old calibration coefficients are tied to the old predictors and were not transferred to new weights. Refreshed predictors require calibration fitted on their own earlier validation forecasts.',
        '## Verification',
        f'All six checkpoint CPU logit reloads match exactly. Split boundaries, checkpoint selection, training-row membership, source and chunk hashes, forecast assembly, legal-path counts, and common cohorts were verified. Independent formulas reproduced {sum(x["score_rows"] for x in verification["runs"]):,} row/target score records, all date-equal aggregates, and {verification["paired_contrasts"]} paired estimates and block intervals. Prior calibration artifacts are unchanged.',
        'The audit caught a rounding issue in the descriptive date-frequency counter: a mathematically tied coverage value could be counted differently under a different averaging order. Differences within 1e-12 are now treated as ties. The original summary is retained, and the correction changes no forecast, error estimate, confidence interval, or MAE date-improvement count. The correction script and receipt are included with the frozen run.',
        'The daily strategy model was not replaced. The sealed holdout beginning 2025-08-07 remains unopened. All jobs for this bounded comparison completed.',
        '## Next action',
        decision['next_action'],
        'Artifacts: artifacts/token-training-refresh-20260915-v1 contains the frozen protocol and source code, source manifest, training checkpoints, generated tokens and paths, common-cohort scores, paired estimates, independent verification and final manifest. The complete per-seed breakdown is results/paired.csv.',
        'Related: [calibration decision update](token-calibration-decision-update-20260915.md), [calibration measurements](token-interval-calibration-20260915.md), [fixed-weight temporal validation](tokenizer-temporal-20260915.md).',
    ]) + '\n'
    assert chr(96) not in text
    (ROOT / 'report.md').write_text(text)
    (BASE / 'docs/token-training-refresh-20260915.md').write_text(text)
    print('Report saved', ROOT / 'report.md')


if __name__ == '__main__':
    main()
