"""Recompute head fits via weighted least squares and audit saved rankings."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.daily_loop import read, verify
from quant_research.storage import file_hash, utc_now, write_json


def independent_fit(history, as_of, ridge):
    t = history[history.label_known & history.date.lt(as_of) & history.label_end.lt(as_of)]
    p, v = t.predicted.to_numpy() * 100, t.volatility_pp.to_numpy()
    x = np.array([p, v, p*v]).T
    w = 1 / t.groupby('date').date.transform('count').to_numpy() / t.date.nunique()
    center = np.average(x, axis=0, weights=w)
    scale = np.sqrt(np.average((x-center)**2, axis=0, weights=w))
    scale[scale <= 1e-12] = 1
    z = np.column_stack([np.ones(len(t)), (x-center)/scale])
    y = (t.actual_extrema_scenario.to_numpy()-t.predicted.to_numpy())*100
    design = np.vstack([z*np.sqrt(w[:, None]), np.diag([0., *([np.sqrt(ridge)]*3)])])
    target = np.r_[y*np.sqrt(w), np.zeros(4)]
    coef = np.linalg.lstsq(design, target, rcond=None)[0]
    return center, scale, coef, t


def audit(root):
    from token_return_calibration import source_frames
    cfg = read(root/'binding.json')['config']
    base = Path(__file__).resolve().parents[1]
    source = base/cfg['source']
    binding = read(root/'binding.json')
    for path, expected in {**binding['sources'], **binding['code']}.items():
        if file_hash(Path(path)) != expected:
            raise ValueError('Changed calibration source or code: '+path)
    native = source_frames(source, 'native', cfg['arm'])
    paired = source_frames(source, 'paired', cfg['arm'])
    training = {s: native[native.seed.eq(s)] for s in cfg['seeds']}
    rows_checked, fits_checked, metric_checks = 0, 0, 0
    expected_dates = [day for day in sorted(native.date.unique())
                      if all(h.loc[h.label_known & h.date.lt(day) & h.label_end.lt(day), 'date'].nunique()
                             >= cfg['minimum_training_dates'] for h in training.values())]
    actual_dates = sorted(p.name for p in (root/'walk-forward').iterdir() if p.is_dir())
    if actual_dates != expected_dates:
        raise ValueError('Incomplete walk-forward date coverage')
    all_daily = []
    for day in expected_dates:
        dest = root/'walk-forward'/day
        verify(dest)
        heads = read(dest/'heads.json')
        daily = pd.read_csv(dest/'daily.csv', dtype={'seed': str})
        all_daily.append(daily)
        for seed in cfg['seeds']:
            center, scale, coef, train = independent_fit(training[seed], day, cfg['ridge'])
            head = heads[seed]
            if (head['as_of'] != day or head['trained_labels_through'] != train.label_end.max()
                    or head['training_rows'] != len(train) or head['training_dates'] != train.date.nunique()):
                raise ValueError('Training lineage differs')
            for got, expected in [(head['center'], center), (head['scale'], scale), (head['coefficients'], coef)]:
                np.testing.assert_allclose(got, expected, atol=1e-9, rtol=1e-9)
            fits_checked += 1
            for mode, source_frame in [('native', native), ('paired', paired)]:
                original = source_frame[source_frame.seed.eq(seed) & source_frame.date.eq(day)].set_index('local_row').sort_index()
                saved = pd.read_parquet(dest/f'{mode}-{seed}.parquet')
                if len(saved) != len(original)*2:
                    raise ValueError('Incomplete ranking coverage')
                p, v = original.predicted.to_numpy()*100, original.volatility_pp.to_numpy()
                expected = original.predicted.to_numpy() + (coef[0] + ((np.array([p, v, p*v]).T-center)/scale) @ coef[1:])/100
                for arm in ['raw', 'calibrated']:
                    g = saved[saved.arm.eq(arm)].set_index('local_row').sort_index()
                    pd.testing.assert_index_equal(g.index, original.index)
                    keep = ['instrument_id', 'date', 'sell_offset', 'buy_reference', 'sell_reference',
                            'label_known', 'actual_extrema_scenario', 'volatility_pp', 'adverse_excursion_pct']
                    pd.testing.assert_frame_equal(g[keep], original[keep])
                    np.testing.assert_allclose(g.predicted, expected if arm == 'calibrated' else original.predicted,
                                               atol=1e-11, rtol=1e-10)
                    ordered = g.sort_values(['predicted', 'instrument_id'], ascending=[False, True])
                    np.testing.assert_array_equal(ordered['rank'], np.arange(1, len(g)+1))
                    top = ordered.iloc[:cfg['top_n']]
                    observed = g[g.label_known]
                    known_top = top[top.label_known]
                    raw_top_ids = original.sort_values(['predicted', 'instrument_id'], ascending=[False, True]).head(cfg['top_n']).index
                    fixed = g[g.index.isin(raw_top_ids) & g.label_known]
                    independently = dict(rows=len(g), known=len(observed), top_known=len(known_top),
                        top_unknown=len(top)-len(known_top),
                        return_mae_pp=np.mean(abs(observed.predicted-observed.actual_extrema_scenario))*100,
                        top_extrema_scenario_pct=known_top.actual_extrema_scenario.mean()*100,
                        top_prediction_bias_pp=(known_top.predicted-known_top.actual_extrema_scenario).mean()*100,
                        top_high_volatility_fraction=(top.volatility_pp >= g.volatility_pp.quantile(.8)).mean(),
                        raw_top20_bias_pp=(fixed.predicted-fixed.actual_extrema_scenario).mean()*100,
                        raw_top20_mae_pp=abs(fixed.predicted-fixed.actual_extrema_scenario).mean()*100)
                    metric = daily[daily.seed.eq(seed) & daily.universe.eq(mode) & daily.arm.eq(arm)].iloc[0]
                    for name, value in independently.items():
                        np.testing.assert_allclose(metric[name], value, atol=1e-10, rtol=1e-10, equal_nan=True)
                        metric_checks += 1
                    rows_checked += len(g)
    verify(root/'validation')
    all_daily = pd.concat(all_daily, ignore_index=True)
    for mode in ['native', 'paired']:
        actual = pd.read_csv(root/'validation'/f'{mode}-daily.csv', dtype={'seed': str})
        expected = all_daily[all_daily.universe.eq(mode)].drop(columns='universe').reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
        for comparison in read(root/'validation'/f'{mode}-comparisons.json'):
            subset = actual[actual.seed.eq(comparison['seed'])]
            a = subset[subset.arm.eq('calibrated')].set_index('date')[comparison['metric']]
            b = subset[subset.arm.eq('raw')].set_index('date')[comparison['metric']]
            delta = (b-a if comparison['metric'] == 'return_mae_pp' else
                     b.abs()-a.abs() if comparison['metric'] == 'top_prediction_bias_pp' else a-b).dropna()
            np.testing.assert_allclose(comparison['improvement'], delta.mean(), atol=1e-10)
            assert comparison['dates_improved'] == int((delta > 1e-10).sum())
    verify(root/'models')
    for seed, history in training.items():
        head = read(root/'models'/f'{seed}.json')
        center, scale, coef, train = independent_fit(history, head['as_of'], cfg['ridge'])
        assert len(train) == int(history.label_known.sum())
        for name, values in [('center', center), ('scale', scale), ('coefficients', coef)]:
            np.testing.assert_allclose(head[name], values, atol=1e-9, rtol=1e-9)
        fits_checked += 1
    for path, expected in read(root/'live-state.json').items():
        if file_hash(Path(path)) != expected:
            raise ValueError('Protected live state changed: '+path)
    write_json(root/'verification.json', dict(passed=True, at=utc_now(),
        independent_method='weighted augmented least squares, separate ranking and metric recomputation',
        evaluation_dates=len(expected_dates), fits_checked=fits_checked, ranking_rows_checked=rows_checked,
        metric_checks=metric_checks, unknown_top20_not_replaced=True,
        native_and_paired=True, seeds=cfg['seeds'], live_state_unchanged=True))
    print('Independent calibration audit passed', rows_checked, 'rows', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    audit(parser.parse_args().root)
