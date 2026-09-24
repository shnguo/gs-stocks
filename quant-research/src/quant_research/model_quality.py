"""Explicit date-level metrics and conservative development advancement gates."""
import numpy as np
import pandas as pd


def label_overlaps(rows):
    ends = rows.groupby('date').label_end.max().sort_index()
    # Future labels start strictly after the signal, so equal boundaries are disjoint.
    return [{'signal':date,'label_end':end,'next_signal':nxt}
            for date,end,nxt in zip(ends.index[:-1],ends.iloc[:-1],ends.index[1:]) if end>nxt]


def weighted_quantile(values, weights, levels=(.1, .5, .9)):
    values, weights = np.asarray(values), np.asarray(weights)
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not good.any():
        raise ValueError('No weighted observations')
    y, w = values[good], weights[good]
    order = np.argsort(y, kind='stable')
    cdf = np.cumsum(w[order]) / w.sum()
    return y[order[np.minimum(np.searchsorted(cdf, levels), len(order)-1)]]


def volatility_scale(x, index, floor):
    value = np.asarray(x[:, index], float)
    if not np.isfinite(value).all() or (value < 0).any():
        raise ValueError('Invalid past volatility')
    return np.maximum(value, floor)


def fit_volatility_baseline(targets, scales, dates):
    standardized = targets / scales[:, None, None]
    result = np.empty((5, 4, 3))
    dates = np.asarray(dates)
    for day in range(5):
        for field in range(4):
            good = np.isfinite(standardized[:, day, field])
            ds = pd.Series(dates[good])
            weights = 1 / ds.map(ds.value_counts()).to_numpy()
            result[day, field] = weighted_quantile(standardized[good, day, field], weights)
    return result


def daily_metrics(rows, targets, forecasts, window):
    records = []
    for model, pred in forecasts.items():
        if pred.shape != (*targets.shape, 3):
            raise ValueError('Forecast axis mismatch')
        for date in sorted(rows.date.unique()):
            ids = rows.date.eq(date).to_numpy()
            y, q = targets[ids], pred[ids]
            valid = np.isfinite(y[:, 4, 3]) & np.isfinite(q[:, 4, 3]).all(1)
            actual, price = y[valid, 4, 3], q[valid, 4, 3]
            error = actual[:, None] - price
            levels = np.array([.1, .5, .9])
            total_error = y[..., None] - q
            finite = np.isfinite(total_error)
            total_loss = np.maximum(levels * total_error, (levels-1) * total_error)
            records.append(dict(window=window, date=date, model=model, input_rows=int(ids.sum()),
                known_rows=int(np.isfinite(y[:, 4, 3]).sum()), scored_rows=int(valid.sum()),
                prediction_fraction=float(np.isfinite(q).all((1, 2, 3)).mean()),
                pinball=float(np.maximum(levels*error,(levels-1)*error).mean()) if valid.any() else np.nan,
                all_ohlc_pinball=float(total_loss[finite].mean()) if finite.any() else np.nan,
                median_mae=float(abs(error[:, 1]).mean()) if valid.any() else np.nan,
                direction_accuracy=float((np.sign(actual)==np.sign(price[:, 1])).mean()) if valid.any() else np.nan,
                coverage80=float(((actual>=price[:, 0]) & (actual<=price[:, 2])).mean()) if valid.any() else np.nan,
                width80=float((price[:, 2]-price[:, 0]).mean()) if valid.any() else np.nan))
    return pd.DataFrame(records)


def assess(daily, protocol, data_audits):
    t = protocol['thresholds']
    windows = daily.groupby(['window','model'], sort=True).agg(
        dates=('pinball','count'), pinball=('pinball','mean'),
        all_ohlc_pinball=('all_ohlc_pinball','mean'), median_mae=('median_mae','mean'),
        coverage80=('coverage80','mean'), width80=('width80','mean'),
        direction_accuracy=('direction_accuracy','mean'), prediction_fraction=('prediction_fraction','min'),
        input_rows=('input_rows','sum'), known_rows=('known_rows','sum'), scored_rows=('scored_rows','sum')).reset_index()
    outcomes = {}
    for model in protocol['learned_candidates']:
        m = windows.loc[windows.model.eq(model)].set_index('window')
        checks = {}
        def check(name, value, observed):
            checks[name] = {'passed': bool(value), 'observed': observed}
        expected = {f'fold-{i:02d}' for i in protocol['fold_indices']}
        check('window_and_date_count', set(m.index)==expected and len(m)>=t['min_windows']
              and (m.dates>=t['min_dates_per_window']).all(), {'windows':len(m),'dates':m.dates.tolist()})
        data_ok = set(data_audits)==expected and all(a['passed'] for a in data_audits.values())
        check('data_audit', data_ok, {w:a['passed'] for w,a in data_audits.items()})
        check('complete_predictions', len(m)>0 and (m.prediction_fraction>=t['min_prediction_fraction']).all()
              and (m.scored_rows==m.known_rows).all(), m.prediction_fraction.tolist())
        for baseline in protocol['baselines']:
            b = windows.loc[windows.model.eq(baseline)].set_index('window')
            comparable = set(b.index)==set(m.index) and len(m)>0
            if not comparable:
                check(baseline+'_comparison',False,None)
                continue
            b = b.reindex(m.index)
            comparable &= bool((m.scored_rows==b.scored_rows).all() and (b.prediction_fraction==1).all())
            ratio = m.pinball / b.pinball
            gain = float(1-m.pinball.mean()/b.pinball.mean())
            wins = int((ratio<1).sum())
            check(baseline+'_pinball_gain', comparable and gain>=t['min_relative_pinball_improvement'], gain)
            check(baseline+'_window_stability', comparable and wins>=t['min_winning_windows']
                  and ratio.max()<=t['max_worst_window_loss_ratio'], {'winning_windows':wins,'worst_loss_ratio':float(ratio.max())})
            overall_ratio = float(m.all_ohlc_pinball.mean()/b.all_ohlc_pinball.mean())
            check(baseline+'_all_ohlc', comparable and overall_ratio<=t['max_all_ohlc_loss_ratio'],overall_ratio)
            mae_ratio = float(m.median_mae.mean()/b.median_mae.mean())
            check(baseline+'_median_mae', comparable and mae_ratio<=t['max_median_mae_ratio'],mae_ratio)
        coverage = float(m.coverage80.mean())
        check('coverage80', t['coverage80_min']<=coverage<=t['coverage80_max'] and
              m.coverage80.between(t['window_coverage80_min'],t['window_coverage80_max']).all(),
              {'mean':coverage,'by_window':m.coverage80.tolist()})
        b = windows.loc[windows.model.eq('volatility')].set_index('window')
        width = float(m.width80.mean()/b.width80.mean()) if len(b) else float('nan')
        check('interval_width', np.isfinite(width) and width<=t['max_width_ratio_to_volatility'],width)
        passed = all(c['passed'] for c in checks.values())
        outcomes[model] = {'passed':passed,'checks':checks,
            'next_step':'decision_value_experiment' if passed else 'revise_price_hypothesis',
            'daily_strategy_ready':False,'statistical_significance_established':False}
    def safe(value):
        if isinstance(value, dict):
            return {k:safe(v) for k,v in value.items()}
        if isinstance(value, list):
            return [safe(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    return windows, safe(outcomes)
