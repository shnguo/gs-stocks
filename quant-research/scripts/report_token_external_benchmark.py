"""Create the repository research note from verified benchmark tables."""
import pandas as pd
from token_external_benchmark import BASE, ROOT, read


def table(rows, columns):
    return '\n'.join(['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---']*len(columns)) + ' |'] +
        ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def main():
    proof = read(ROOT / 'independent-verification.json')
    assert proof['passed']
    decision = read(ROOT / 'research-decision.json')
    daily = pd.read_csv(ROOT / 'results/daily.csv')
    pairs = pd.read_csv(ROOT / 'results/paired.csv', dtype={'seed': str})
    summary = pd.read_csv(ROOT / 'results/summary.csv')
    selected = ['ours_mean_raw', 'ours_mean_calibrated', 'kronos', 'historical']
    names = dict(ours_mean_raw='Self-built, raw', ours_mean_calibrated='Self-built, calibrated',
        kronos='Kronos Base, raw', historical='Historical volatility')
    main_rows, differences, target_rows, seeds = [], [], [], []
    for h in [2, 5]:
        agg = daily[(daily.horizon == h) & daily.target.isin(['maximum', 'minimum'])].groupby('variant')[['mae', 'crps', 'coverage80', 'width80', 'interval_score80']].mean()
        for name in selected:
            row = agg.loc[name]
            main_rows.append([h, names[name], f'{row.mae:.4f}', '—' if pd.isna(row.crps) else f'{row.crps:.4f}',
                f'{100*row.coverage80:.2f}%', f'{row.width80:.4f}', f'{row.interval_score80:.4f}'])
        for against in ['kronos', 'historical']:
            sub = pairs[(pairs.seed == 'mean') & (pairs.kind == 'raw') & (pairs.against == against)
                & (pairs.horizon == h) & (pairs.target == 'high_low')].set_index('metric')
            differences.append([h, names[against], f'{100*sub.loc["mae", "relative_change"]:+.2f}%',
                f'[{sub.loc["mae", "ci_low"]:+.4f}, {sub.loc["mae", "ci_high"]:+.4f}]',
                f'{100*sub.loc["crps", "relative_change"]:+.2f}%',
                f'[{sub.loc["crps", "ci_low"]:+.4f}, {sub.loc["crps", "ci_high"]:+.4f}]',
                f'{100*sub.loc["interval_score80", "relative_change"]:+.2f}%'])
        for target in ['maximum', 'minimum', 'range']:
            for name in selected:
                row = summary[(summary.variant == name) & (summary.horizon == h) & (summary.target == target)].iloc[0]
                target_rows.append([h, target, names[name], f'{row.mae:.4f}', '—' if pd.isna(row.crps) else f'{row.crps:.4f}',
                    f'{100*row.coverage80:.2f}%', f'{row.interval_score80:.4f}'])
        for against in ['kronos', 'historical']:
            for seed in ['17', '29', '43']:
                sub = pairs[(pairs.seed == seed) & (pairs.kind == 'raw') & (pairs.against == against)
                    & (pairs.horizon == h) & (pairs.target == 'high_low')].set_index('metric')
                seeds.append([seed, h, names[against], f'{100*sub.loc["mae", "relative_change"]:+.2f}%',
                    f'{100*sub.loc["crps", "relative_change"]:+.2f}%'])
    cov = pd.read_csv(ROOT / 'results/coverage.csv')
    cov_rows = [[r.variant, r.horizon, int(r.inputs), int(r.known), int(r.usable), f'{100*r.valid_paths/r.total_paths:.2f}%'] for r in cov.itertuples()]
    common = pd.read_parquet(ROOT / 'results/common.parquet')
    cohort = common[['row_id', 'date', 'horizon']].drop_duplicates().groupby('horizon').agg(inputs=('row_id', 'size'), dates=('date', 'nunique'))
    body = '\n\n'.join([
        '# Frozen self-built model versus Kronos Base and historical volatility',
        'Date: 2026-09-15 (America/Los_Angeles). Completed retrospective external benchmark. No training, calibration fitting or evaluation-based retuning.',
        '## Conclusion', decision['conclusion'], decision['limitations'],
        '## Matched scope and comparison',
        'The source cohort contains 2,432 distinct stock-date inputs on 76 signal dates, 2025-04-01 through 2025-07-22, with labels ending no later than 2025-07-29. Both neural pipelines use the same 60 historical OHLCVA bars, signal-anchored price adjustment, normalization statistics and calendar; the historical baseline uses those same source prices. The first 32 members of the outcome-independent daily pool are retained before inspecting outcomes or forecast quality.',
        'The self-built profile contains three frozen 3,720,448 parameter predictors trained through 2023 and selected on 2024 H1, the adapted decoder, and checkpoint-specific 2024 H1 interval factors. Existing 64-path forecasts are reused exactly. Self-built means below average the three models’ scores; they are not forecasts from a 192-path ensemble, and the three seeds are not treated as independent market-date observations.',
        'Kronos Base uses its official tokenizer/decoder and frozen 102,310,592 parameter predictor, revision 2b554741eca47781b64468546e77fef3e85130e6. Its official native inference generates 64 independent five-day paths per input, temperature 1, top-p 1, top-k 0, clip 5 and batches of four. The averaging axis has size one; no paths are averaged away. The native decoder differs from the adapted self-built decoder: this is a full-pipeline comparison, not a capacity or architecture ablation. All 291,840 coarse/fine historical token values match the frozen self-built input prefixes.',
        'The historical-volatility baseline samples 64 contiguous five-day blocks from 59 previous-close OHLC log-return vectors in each 60-day history. The historical mean close drift is subtracted from all four returns, and sampled returns are compounded from the signal close. A hash of seed 17 and stock/date fixes the block starts. Volume and amount are unused zero placeholders; this baseline evaluates prices only. Each external baseline has one frozen Monte Carlo realization, not three independently generated replicas.',
        '## High/low accuracy and interval quality',
        'Maximum future high and minimum future low are scored separately, then equally averaged. Rows are equally weighted within each date, followed by equal date weights. MAE is the absolute error of the sample median. CRPS measures the full sampled distribution; both are lower-is-better. Prices, errors, widths and interval scores are in percentage points of the signal-day close. Coverage is for separate nominal 80% intervals for each target, not a joint executable trading band.',
        table(main_rows, ['Days', 'Model', 'High/low MAE', 'Raw CRPS', '80% coverage', 'Interval width', 'Interval score']),
        'Calibration preserves the self-built medians exactly. Its secondary interval comparison has unequal calibration treatment: neither external baseline is fitted to the self-built calibration period. A better calibrated interval score alone would not demonstrate a better predictor. Calibrated CRPS is omitted because only interval boundaries, not the entire predictive distribution, were calibrated.',
        '## Raw self-built changes relative to each external baseline',
        'Negative relative changes favor the self-built model. Confidence intervals contain absolute differences, in percentage points. Paired 95% intervals use 2,000 circular resamples of 10 signal dates; they are conditional on this development block, these trained seeds and fixed sampled paths. Target/seed comparisons are descriptive and overlap.',
        table(differences, ['Days', 'Baseline', 'MAE change', '95% CI, MAE difference', 'CRPS change', '95% CI, CRPS difference', 'Raw interval score change']),
        '## Individual targets',
        table(target_rows, ['Days', 'Target', 'Model', 'MAE', 'Raw CRPS', '80% coverage', 'Interval score']),
        decision['local_regressions'],
        '## Training-seed consistency',
        table(seeds, ['Self-built seed', 'Days', 'Baseline', 'High/low MAE change', 'High/low CRPS change']),
        '## Availability and common cohort',
        'Each horizon requires all future labels to be known and at least 16 legal paths out of 64. Illegal generated prices are excluded, not repaired. Every reported model comparison uses the intersection of all three self-built predictors and both external baselines; raw/calibrated variants share that intersection. The tables below disclose the excluded coverage, so conditional forecast scores are not presented as whole-population results. This common cohort is smaller than the previous old-versus-refreshed comparison, so absolute self-built scores differ even though its paths and weights are unchanged.',
        table(cov_rows, ['Variant', 'Days', 'Input rows', 'Known-label rows', 'Usable rows', 'Legal paths']),
        table([[h, int(r.inputs), int(r.dates)] for h, r in cohort.iterrows()], ['Days', 'Common stock-date rows', 'Common signal dates']),
        '## Evidence and interpretation limits',
        'This already-examined development period is not an untouched final test. The exact pretraining dates of the official tokenizer and Kronos predictor are not independently established, so temporal leakage cannot be excluded for their pretraining. Differing training histories, decoder weights and calibration prevent isolating an architecture effect. This study evaluates forecasting errors; it does not test fills, costs, attainable extrema or trading profitability.',
        'The previous improvements against the older self-built profile remain valid evidence within their scope. External-baseline strengths or local regressions qualify their practical value; they do not erase those earlier measurements.',
        '## Verification',
        f'Independent calculations reproduced {proof["raw_and_calibrated_score_records"]:,} row-target scores, {proof["common_score_records"]:,} common-cohort records, all date aggregates and {proof["paired_estimates_and_intervals"]} paired estimates and intervals. CRPS was independently recalculated using pairwise path distances. All {proof["historical_paths_reconstructed"]:,} historical paths were reconstructed by sequential compounding. Source histories/statistics, calendar alignment, cohort membership and path validity were verified.',
        f'Kronos generation, path assembly and reload took {read(ROOT / "kronos/lineage.json")["elapsed_seconds"] / 60:.1f} minutes on the local MPS device. Kronos weights were strictly loaded and SHA256-checked, and its first 256 native paths matched exactly after model reload. All three reused self-built runs and their calibration factors were independently rechecked. Fourteen focused regression tests passed. Prior weights, calibration factors and artifacts remain unchanged. The daily strategy model was not replaced, and the sealed holdout beginning 2025-08-07 was not opened.',
        '## Research decision and next step', decision['next_action'],
        'Artifacts: artifacts/token-external-benchmark-20260915-v1 retains the protocol frozen before generation, source hashes, complete matched histories, raw Kronos and historical paths, historical block starts, scores, cohort records, comparison tables, independent verification, runtime and final manifest. The metric source is results/paired.csv and results/summary.csv; the source selection and transformations are reproducible with the frozen scripts in code/scripts.',
        'Related: [later-period transfer](token-profile-transfer-20260915.md), [checkpoint calibration](token-refreshed-calibration-20260915.md), [earlier Kronos Base comparison](kronos-base-range-20260914.md).',
    ]) + '\n'
    assert chr(96) not in body
    (ROOT / 'report.md').write_text(body)
    (BASE / 'docs/token-external-benchmark-20260915.md').write_text(body)
    print('Benchmark research note saved', flush=True)


if __name__ == '__main__':
    main()
