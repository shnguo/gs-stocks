"""Independently reconstruct ideal-timing rankings from frozen sampled paths."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.daily_loop import read, verify
from quant_research.storage import file_hash, utc_now, write_json


def main(run, report):
    verify(run)
    verify(report)
    info, delivery = read(run / 'run.json'), read(report / 'delivery.json')
    assert delivery['ranking_method'] == 't_low_to_future_high'
    assert delivery['run_manifest_sha256'] == file_hash(run / 'manifest.json')
    rows = pd.read_parquet(run / 'inputs/forecast-rows.parquet')
    records = []
    for seed, checkpoint in info['checkpoints'].items():
        assert file_hash(Path(checkpoint['path'])) == checkpoint['sha256']
        paths = np.load(run / f'forecasts/seed{seed}/paths.npy', mmap_mode='r')
        for i, row in enumerate(rows.itertuples()):
            values = []
            for horizon in [2, 5]:
                p = paths[i, :, :horizon].astype(float)
                legal = (np.isfinite(p).all(-1) & (p[..., :4] > 0).all(-1) & (p[..., 4:] >= 0).all(-1)
                    & (p[..., 1] >= p[..., :4].max(-1)) & (p[..., 2] <= p[..., :4].min(-1))).all(-1)
                p = p[legal]
                if len(p) < 16:
                    values = []
                    break
                buy = p[:, 0, 2]
                sell = np.maximum.reduce(p[:, 1:, 1], axis=1)
                returns = (sell-buy)/buy-info['cost_scenario']
                values.append(dict(instrument_id=row.instrument_id, seed=int(seed), horizon=horizon,
                    expected_net_return=returns.mean(), positive_fraction=(returns > 0).mean(),
                    entry_median=np.median(buy), exit_median=np.median(sell), return_q10=np.quantile(returns, .1)))
            records.extend(values)
    expected = pd.DataFrame(records)
    scores = report if (report / 'model-scores.parquet').exists() else run
    saved = pd.read_parquet(scores / 'model-scores.parquet')
    keys = ['instrument_id', 'seed', 'horizon']
    pd.testing.assert_frame_equal(expected.sort_values(keys).reset_index(drop=True),
        saved[expected.columns].sort_values(keys).reset_index(drop=True), check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-10)
    eligible = expected[expected.horizon.eq(5)].groupby('instrument_id').filter(lambda g: g.seed.nunique() == len(info['checkpoints']))
    ranking = eligible.groupby('instrument_id').expected_net_return.mean().reset_index().sort_values(['expected_net_return', 'instrument_id'], ascending=[False, True])
    actual = pd.read_csv(report / 'ranking.csv')
    assert actual.instrument_id.tolist() == ranking.instrument_id.tolist()
    np.testing.assert_allclose(actual.expected_net_return, ranking.expected_net_return, rtol=1e-10, atol=1e-10)
    result = dict(passed=True, at=utc_now(), ranking_method=delivery['ranking_method'], ranked=len(actual),
        checked_model_scores=len(expected), unchanged_run_manifest_sha256=file_hash(run / 'manifest.json'),
        report_manifest_sha256=file_hash(report / 'manifest.json'), checkpoints_verified=len(info['checkpoints']),
        no_training_or_inference=True, same_day_high_excluded=True, paired_within_each_path=True)
    write_json(run.parent.parent / 'verification' / (run.name+'-ideal-timing.json'), result)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    main(args.run, args.report)
