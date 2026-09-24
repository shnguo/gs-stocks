"""Report calibrated refreshed forecasts and a matched end-to-end comparison."""
import pandas as pd
from token_refreshed_calibration_run import BASE, ROOT, SEEDS, read


def table(rows, columns):
    return '\n'.join(['| ' + ' | '.join(columns) + ' |',
        '| ' + ' | '.join(['---'] * len(columns)) + ' |'] +
        ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def main():
    verification = read(ROOT / 'independent-verification.json')
    assert verification['passed']
    decision = read(ROOT / 'research-decision.json')
    paired = pd.read_csv(ROOT / 'results/paired.csv', dtype={'seed': str})
    average = paired[paired.seed == 'mean']
    rows, pipeline, targets = [], [], []
    for window in ['2023h2', '2024h2']:
        for h in [2, 5]:
            a = average[(average.comparison == 'calibration') & (average.window == window)
                & (average.horizon == h) & (average.target == 'high_low')].set_index('metric')
            rows.append([window, h, f'{100 * a.loc["coverage80", "baseline"]:.2f}%',
                f'{100 * a.loc["coverage80", "candidate"]:.2f}%',
                f'{100 * a.loc["width80", "relative_change"]:+.2f}%',
                f'{100 * a.loc["interval_score80", "relative_change"]:+.2f}%',
                f'[{a.loc["interval_score80", "ci_low"]:+.4f}, {a.loc["interval_score80", "ci_high"]:+.4f}]'])
            e = average[(average.comparison == 'end_to_end') & (average.window == window)
                & (average.horizon == h) & (average.target == 'high_low')].set_index('metric')
            pipeline.append([window, h, f'{100 * e.loc["coverage80", "baseline"]:.2f}%',
                f'{100 * e.loc["coverage80", "candidate"]:.2f}%',
                f'{100 * e.loc["interval_score80", "relative_change"]:+.2f}%',
                f'{100 * e.loc["mae", "relative_change"]:+.2f}%',
                f'{100 * e.loc["width80", "relative_change"]:+.2f}%'])
            for target in ['maximum', 'minimum', 'range']:
                t = average[(average.comparison == 'calibration') & (average.window == window)
                    & (average.horizon == h) & (average.target == target)].set_index('metric')
                targets.append([window, h, target, f'{100 * t.loc["coverage80", "baseline"]:.2f}%',
                    f'{100 * t.loc["coverage80", "candidate"]:.2f}%',
                    f'{100 * t.loc["interval_score80", "relative_change"]:+.2f}%'])
    factors = []
    fit_data = []
    for window in ['2023h2', '2024h2']:
        fitted = read(ROOT / window / 'fit.json')
        for p in fitted['parameters']:
            factors.append([window, p['seed'], p['horizon'], p['target'], f'{p["scale"]:.4f}', p['dates'], p['rows']])
        for seed in SEEDS:
            quantiles = pd.read_parquet(ROOT / window / 'intervals' / f'calibration{window[:4]}h1' / 'quantiles.parquet')
            sub = quantiles[quantiles.seed == seed]
            fit_data.append([window, seed, sub.date.nunique(), sub.row_id.nunique(),
                're-decoded cached tokens' if seed == 17 else 'new generated tokens'])
    n_quantiles = sum(v['quantiles'] for v in verification['checks'].values() if isinstance(v, dict) and 'quantiles' in v)
    n_scores = sum(v['scores'] for v in verification['checks'].values() if isinstance(v, dict) and 'scores' in v)
    text = '\n\n'.join([
        '# Checkpoint-specific interval calibration after training refresh',
        'Date: 2026-09-15 (America/Los_Angeles). Completed historical research on 2023 H2 and 2024 H2, three predictor seeds per window.',
        '## Conclusion',
        decision['conclusion'],
        decision['limitations'],
        '## Calibration effect on the refreshed predictors',
        'This table equally averages maximum-high and minimum-low metrics across three seeds. Coverage is the fraction of realized targets inside the nominal 80% interval. Lower interval score is better: it accounts for both interval width and misses. Negative score changes indicate improvement. The confidence interval is for the score difference in percentage points of signal-date close.',
        table(rows, ['Window', 'Days', 'Raw coverage', 'Calibrated coverage', 'Width change', 'Interval score change', '95% CI of score difference']),
        'Calibration changes only the lower and upper bounds around the existing median. Every median and every row-level point MAE is exactly unchanged. It does not calibrate the full forecast distribution, so this report makes no claim of improved CRPS from calibration.',
        '## Complete refreshed pipeline versus the previous calibrated models',
        'The comparison below uses exactly matched stock-date inputs. The old models were trained through 2021, and their calibration factors were fitted on 2023 H1 and held fixed for both evaluation windows. Refreshed models were trained through 2022 or 2023 and calibrated on the corresponding 2023 H1 or 2024 H1 forecasts. The adapted decoder and sampler are identical. Thus this comparison includes both predictor refresh and calibration refresh.',
        table(pipeline, ['Window', 'Days', 'Old calibrated coverage', 'Refreshed calibrated coverage', 'Interval score change', 'Point MAE change', 'Width change']),
        'The point-MAE changes reproduce the previous training-refresh comparison on its original common cohorts. The first table uses every usable refreshed-model input; the second uses the old/new intersection. Coverage figures from these tables need not be identical because their cohorts differ.',
        '## Individual targets',
        table(targets, ['Window', 'Days', 'Target', 'Raw coverage', 'Calibrated coverage', 'Interval score change']),
        decision['local_regressions'],
        '## Frozen method and chronology',
        'There are 36 independently fitted factors: two checkpoint vintages × three training seeds × two forecast horizons × three targets. Each factor uses only its own checkpoint’s earlier H1 validation forecasts. The exact predictor hash, decoder hash and sampling settings are attached to each fit. The old 18 factors are not reused.',
        'Within each fit, every signal date has equal total weight. The scale is the weighted empirical 80th percentile of the residual normalized by the left or right raw half-width, with a minimum of one. It widens the interval around the unchanged median. Lower bounds respect the target support: −100% for high/low returns, and zero for range. The numerical half-width floor is 0.000001 percentage points.',
        'All 36 factors were frozen before this run scored either H2 evaluation. Signals and their entire five-day label windows stay inside the corresponding partition. Calibration uses 16 stocks per date, evaluation 32, selected before checking labels. A horizon requires complete known labels and at least 16 legal draws among 64 paths.',
        table(fit_data, ['Checkpoint vintage', 'Seed', 'Calibration dates', 'Usable calibration inputs', 'Forecast source']),
        'The neural networks were not retrained. Two archived validation token sets were verified and decoded with the fixed adapted decoder; four new validation token sets were generated. All six evaluation forecast sets came from the completed training-refresh experiment. Token generation uses 64 paths, temperature 1, full token distribution, batch size four and generation seed 17 plus batch offset.',
        '## Interpretation limits',
        'These are previously examined historical windows. H1 also served for checkpoint selection, so the calibration data are not independent of model selection. Official tokenizer pretraining coverage remains unresolved. The empirical method has no formal future coverage guarantee.',
        'Confidence intervals use 2,000 circular block resamples of 10 dates and describe uncertainty within these historical windows and three trained seeds. They are not an all-subgroup acceptance veto. Width, misses, effect size, consistency and local regressions are considered together.',
        '## Verification and artifacts',
        f'Independent formulas reproduced {n_quantiles:,} raw quantile records and {n_scores:,} raw/calibrated score records, all 36 date-weighted factors, date-equal aggregates and 640 paired estimates and confidence intervals. All six calibration token/price first-batch replays matched exactly. Source hashes, chunk assembly, chronological boundaries, factor-to-checkpoint binding and exact median preservation passed. The matched end-to-end point errors reproduce the prior verified research results.',
        'Nine focused regression tests passed. No daily strategy model was replaced, and the sealed holdout beginning 2025-08-07 was not opened.',
        'Artifacts are in artifacts/token-refreshed-calibration-20260915-v1. The protocol, frozen source, source manifest, two checkpoint-specific fit files, calibration paths, evaluation scores, common-cohort comparisons, verification receipt and final manifest are retained. research-profile.json binds the resulting checkpoint, decoder, sampler and interval-factor files for research reuse.',
        '## Next bounded step',
        decision['next_action'],
        '## Fitted factors',
        table(factors, ['Checkpoint vintage', 'Seed', 'Days', 'Target', 'Scale', 'Dates', 'Rows']),
        'Related: [training refresh comparison](token-training-refresh-20260915.md), [calibration decision](token-calibration-decision-update-20260915.md), [previous calibration measurements](token-interval-calibration-20260915.md).',
    ]) + '\n'
    assert chr(96) not in text
    (ROOT / 'report.md').write_text(text)
    (BASE / 'docs/token-refreshed-calibration-20260915.md').write_text(text)
    print('Report saved:', ROOT / 'report.md')


if __name__ == '__main__':
    main()
