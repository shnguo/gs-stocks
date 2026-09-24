"""Report later-window transfer without revising the frozen profile or prior evidence."""
import pandas as pd
from token_profile_transfer_run import BASE, ROOT, SEEDS, read


def table(rows, columns):
    return '\n'.join(['| ' + ' | '.join(columns) + ' |',
        '| ' + ' | '.join(['---'] * len(columns)) + ' |'] +
        ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def main():
    verification = read(ROOT / 'independent-verification.json')
    assert verification['passed']
    decision = read(ROOT / 'research-decision.json')
    paired = pd.read_csv(ROOT / 'results/paired.csv', dtype={'seed': str})
    mean = paired[paired.seed == 'mean']
    transfer, calibration, targets, seeds = [], [], [], []
    for h in [2, 5]:
        p = mean[(mean.comparison == 'profile_transfer') & (mean.horizon == h) & (mean.target == 'high_low')].set_index('metric')
        distribution = mean[(mean.comparison == 'raw_distribution') & (mean.horizon == h) & (mean.target == 'high_low')].iloc[0]
        transfer.append([h, f'{p.loc["mae", "baseline"]:.4f} → {p.loc["mae", "candidate"]:.4f}',
            f'{100 * p.loc["mae", "relative_change"]:+.2f}%',
            f'{100 * p.loc["interval_score80", "relative_change"]:+.2f}%',
            f'[{p.loc["interval_score80", "ci_low"]:+.4f}, {p.loc["interval_score80", "ci_high"]:+.4f}]',
            f'{100 * distribution.relative_change:+.2f}%'])
        c = mean[(mean.comparison == 'refreshed_calibration') & (mean.horizon == h) & (mean.target == 'high_low')].set_index('metric')
        calibration.append([h, f'{100 * c.loc["coverage80", "baseline"]:.2f}%',
            f'{100 * c.loc["coverage80", "candidate"]:.2f}%', f'{100 * c.loc["width80", "relative_change"]:+.2f}%',
            f'{100 * c.loc["interval_score80", "relative_change"]:+.2f}%',
            f'[{c.loc["interval_score80", "ci_low"]:+.4f}, {c.loc["interval_score80", "ci_high"]:+.4f}]'])
        for target in ['maximum', 'minimum', 'range']:
            t = mean[(mean.comparison == 'profile_transfer') & (mean.horizon == h) & (mean.target == target)].set_index('metric')
            c = mean[(mean.comparison == 'refreshed_calibration') & (mean.horizon == h) & (mean.target == target)].set_index('metric')
            targets.append([h, target, f'{100 * t.loc["mae", "relative_change"]:+.2f}%',
                f'{100 * t.loc["interval_score80", "relative_change"]:+.2f}%',
                f'{100 * t.loc["coverage80", "baseline"]:.2f}% → {100 * t.loc["coverage80", "candidate"]:.2f}%',
                f'{100 * c.loc["interval_score80", "relative_change"]:+.2f}%'])
        for seed in SEEDS:
            s = paired[(paired.seed == str(seed)) & (paired.comparison == 'profile_transfer')
                & (paired.horizon == h) & (paired.target == 'high_low')].set_index('metric')
            seeds.append([seed, h, f'{100 * s.loc["mae", "relative_change"]:+.2f}%',
                f'{100 * s.loc["interval_score80", "relative_change"]:+.2f}%',
                f'{100 * s.loc["coverage80", "candidate"]:.2f}%'])
    coverage = []
    for owner in ['old', 'refreshed']:
        rows = pd.concat([pd.read_csv(ROOT / 'scores' / owner / f'seed{seed}/coverage.csv') for seed in SEEDS])
        summary = rows.groupby('horizon').sum()
        for h, row in summary.iterrows():
            coverage.append([owner, h, int(row.inputs), int(row.known), int(row.usable),
                f'{100 * row.valid_paths / row.total_paths:.2f}%'])
    body = '\n\n'.join([
        '# Frozen-profile transfer to April–July 2025',
        'Date: 2026-09-15 (America/Los_Angeles). Completed transfer research; no predictor retraining, calibration fitting or evaluation-based retuning.',
        '## Conclusion',
        decision['conclusion'],
        decision['limitations'],
        '## Comparison and frozen sources',
        'Evaluation uses 76 signal dates from 2025-04-01 through 2025-07-22, with future labels ending by 2025-07-29. The same 2,432 distinct stock-date inputs are shared by all six runs, selected as the first 32 members of the outcome-independent daily pool. The full five-day label window stays inside the April–July development block.',
        'The refreshed profile uses the three 3,720,448 parameter predictors trained through 2023, selected on 2024 H1, and their checkpoint-specific interval factors fitted on 2024 H1. The control uses three predictors trained through 2021, selected on 2022 H1, with factors fitted on 2023 H1. All weights and factors are the exact previously verified files.',
        'Both vintages use the same adapted decoder, 60 history bars, equal output-loss training recipe, 64 generated paths, temperature 1, full token distribution, batch size four and generation seed 17 plus batch offset. Each comparison uses common usable stock-date/horizon inputs between vintages within each seed. Calibration leaves eligibility and the point median unchanged.',
        '## Does the combined profile transfer?',
        'Maximum-high and minimum-low errors are equally averaged across targets, dates and the three seeds. MAE, interval scores and confidence-interval endpoints are in percentage points of the signal-date close. Negative relative changes mean lower error. CRPS scores only the raw sampled distributions; it is not a claim that interval calibration improves the full distribution.',
        table(transfer, ['Days', 'High/low MAE, old → refreshed', 'MAE change', 'Calibrated interval score change', '95% CI of score difference', 'Raw CRPS change']),
        '## Does the old calibration still help the refreshed predictors?',
        'This comparison holds predictor weights, generated paths, point medians and the common cohort fixed. It applies the frozen 2024 H1 interval factors without updating them on 2025 outcomes. The nominal coverage target is 80%. Lower interval score is better because it penalizes both width and misses.',
        table(calibration, ['Days', 'Raw coverage', 'Calibrated coverage', 'Width change', 'Interval score change', '95% CI of score difference']),
        '## Individual targets',
        table(targets, ['Days', 'Target', 'MAE change versus old', 'Interval score change versus old calibrated', 'Old → refreshed calibrated coverage', 'Calibration-only score change']),
        decision['local_regressions'],
        '## Seed consistency',
        table(seeds, ['Seed', 'Days', 'High/low MAE change', 'Calibrated interval score change', 'Refreshed calibrated coverage']),
        '## Usable inputs and valid paths',
        'Counts below sum over three seeds, so repeated stock-date inputs are counted once per seed. A target horizon requires complete known labels and at least 16 legal draws out of 64. Scores use the common cohort even if the individual-vintage usable counts differ.',
        table(coverage, ['Vintage', 'Days', 'Inputs across seeds', 'Known-label inputs', 'Usable inputs', 'Legal paths']),
        '## Interpretation',
        'This tests transfer of the entire frozen research profile. It does not estimate the isolated causal effect of data recency: the vintages differ in training history, training exposure, checkpoint-selection period and calibration period. The 2025 development block was used by earlier model experiments and is not an untouched final test. Official tokenizer pretraining coverage remains unresolved.',
        'The 95% confidence intervals use 2,000 circular block resamples of 10 dates. They describe uncertainty in this 76-date block conditional on the trained seeds, not all market regimes or all model-selection uncertainty. Subgroup comparisons are descriptive. Local regressions and uncertain estimates qualify earlier gains rather than automatically negating them.',
        '## Verification',
        f'Independent formulas reproduced {sum(verification["run_score_records"].values()):,} raw/calibrated row-target score records, {verification["common_rows"]:,} common-cohort records, all date-equal aggregates, and {verification["paired_estimates_and_intervals"]} paired estimates and confidence intervals. Raw quantiles and pairwise CRPS were recalculated from the saved paths. All six token-and-price first-batch replays matched exactly. Factor/checkpoint hashes, path assembly, legal-path masks, chronology and exact median preservation passed.',
        'Nine focused regression tests passed. The run retained 933,888 five-day sampled paths. Both historical calibration fits and all model weights stayed unchanged. The daily strategy model was not replaced and the sealed holdout beginning 2025-08-07 was not opened.',
        '## Research decision and next step',
        decision['next_action'],
        'Artifacts: artifacts/token-profile-transfer-20260915-v1 contains frozen source, protocol, profiles, source hashes, preflight bounds, token/path chunks, raw quantiles, calibrated scores, matched cohorts, results/paired.csv, independent verification and final manifest. Prior research artifacts remain intact.',
        'Related: [checkpoint-specific calibration](token-refreshed-calibration-20260915.md), [training refresh](token-training-refresh-20260915.md), [calibration decision](token-calibration-decision-update-20260915.md).',
    ]) + '\n'
    assert chr(96) not in body
    (ROOT / 'report.md').write_text(body)
    (BASE / 'docs/token-profile-transfer-20260915.md').write_text(body)
    print('Report saved:', ROOT / 'report.md')


if __name__ == '__main__':
    main()
