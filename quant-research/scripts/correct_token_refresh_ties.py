"""Treat round-off-sized date differences as ties; preserve all original estimates."""
import shutil

import numpy as np
import pandas as pd
from token_training_refresh_run import ROOT, core, read

from quant_research.storage import file_hash, utc_now, write_json


def main():
    original = ROOT / 'results-before-tie-correction'
    if not original.exists():
        shutil.copytree(ROOT / 'results', original)
    core.verify(original)
    daily = pd.read_csv(original / 'daily.csv')
    paired = pd.read_csv(original / 'paired.csv', dtype={'seed': str})
    result = paired.copy()
    for i, row in paired.iterrows():
        subset = daily[(daily.window == row.window) & (daily.horizon == row.horizon)]
        if row.target == 'high_low':
            subset = subset[subset.target.isin(['maximum', 'minimum'])]
        elif row.target != 'mean':
            subset = subset[subset.target == row.target]
        if row.seed != 'mean':
            subset = subset[subset.seed == int(row.seed)]
        values = subset.groupby(['seed', 'date', 'variant'])[row.metric].mean().unstack().groupby('date').mean()
        result.loc[i, 'date_win_fraction'] = np.mean((values.refreshed - values.old).to_numpy() < -1e-12)
    unaffected = [c for c in paired if c != 'date_win_fraction']
    pd.testing.assert_frame_equal(paired[unaffected], result[unaffected])
    result.to_csv(ROOT / 'results/paired.csv', index=False)
    core.finish(ROOT / 'results')
    definitions = read(ROOT / 'metric-definitions.json')
    note = ' Absolute differences up to 1e-12 are treated as ties to avoid aggregation round-off.'
    if note not in definitions['date_win_fraction']:
        definitions['date_win_fraction'] += note
    write_json(ROOT / 'metric-definitions.json', definitions)
    write_json(ROOT / 'tie-correction.json', dict(at=utc_now(),
        reason='Strict sign comparison counted round-off-sized coverage ties inconsistently under independent averaging order.',
        tolerance=1e-12, changed_rows=int(((paired.date_win_fraction - result.date_win_fraction).abs() > 1e-12).sum()),
        changed_mae_rows=int((((paired.date_win_fraction - result.date_win_fraction).abs() > 1e-12) & (paired.metric == 'mae')).sum()),
        all_error_estimates_and_confidence_intervals_unchanged=True,
        original_sha256=file_hash(original / 'paired.csv'), corrected_sha256=file_hash(ROOT / 'results/paired.csv')))


if __name__ == '__main__':
    main()
