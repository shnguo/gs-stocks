"""Calendar-only rolling partitions and paired date-level development diagnostics."""
import numpy as np
import pandas as pd

PARTITIONS = ("train", "selection", "calibration", "evaluation")


def sample_dates(dates, count):
    if len(dates) < count or count < 1:
        raise ValueError("Insufficient calendar dates for frozen design")
    return [dates[i] for i in np.linspace(0, len(dates)-1, count, dtype=int)]


def validate_date_plan(plan, dates, fold, sealed):
    if set(plan) != set(PARTITIONS) or dates != sorted(set(dates)):
        raise ValueError("Invalid date plan/calendar")
    lookup = {d: i for i, d in enumerate(dates)}
    boundaries = {}
    for j, part in enumerate(PARTITIONS):
        ds = plan[part]
        if not ds or ds != sorted(set(ds)) or any(d not in lookup for d in ds):
            raise ValueError("Invalid partition dates")
        boundary = plan[PARTITIONS[j+1]][0] if j < 3 else min(fold['test_end'], sealed)
        if any(lookup[d]+5 >= len(dates) or dates[lookup[d]+5] >= boundary for d in ds):
            raise ValueError("Five-session label crosses partition or sealed boundary")
        boundaries[part] = boundary
    if plan['train'][0] < fold['train_start'] or any(
            not fold['test_start'] <= d <= fold['test_end'] for d in plan['evaluation']):
        raise ValueError("Dates outside registered development window")
    if fold['purpose'] != 'development' or fold['test_end'] >= sealed:
        raise ValueError("Sealed evaluation forbidden")
    coords = [lookup[d] for d in plan['evaluation']]
    if any(b-a < 5 for a, b in zip(coords, coords[1:])):
        raise ValueError("Evaluation future labels overlap")
    return boundaries


def build_date_plans(dates, fold, sealed, evaluation_count=10):
    """Three ablations: old split, recent split, denser recent training.

    Evaluation signals are five sessions apart; training overlap is allowed.
    Recent selection and calibration each cover 20 calendar sessions, four signals.
    All bounds derive from the known exchange calendar, never price outcomes.
    """
    eligible = [d for i, d in enumerate(dates) if fold['test_start'] <= d <= fold['test_end']
                and i+5 < len(dates) and dates[i+5] < min(fold['test_end'], sealed)]
    evaluation = eligible[::5][:evaluation_count]
    if len(evaluation) != evaluation_count:
        raise ValueError("Insufficient disjoint evaluation dates")
    def before(start, end, count):
        eligible = [d for i, d in enumerate(dates) if start <= d < end
                    and i+5 < len(dates) and dates[i+5] < end]
        return sample_dates(eligible, count)
    validation = [d for d in dates if fold['validation_start'] <= d < fold['test_start']]
    cal_start = validation[len(validation)//2]
    old_start = max(fold['train_start'], str((pd.Timestamp(fold['validation_start'])
                                            - pd.DateOffset(years=1)).date()))
    old = dict(train=before(old_start, fold['validation_start'], 24),
               selection=before(fold['validation_start'], cal_start, 4),
               calibration=before(cal_start, evaluation[0], 12), evaluation=evaluation)
    cal_index = dates.index(evaluation[0])-25
    sel_index = cal_index-25
    start = str((pd.Timestamp(dates[sel_index])-pd.DateOffset(years=1)).date())
    recent = dict(train=before(start, dates[sel_index], 24),
                  selection=before(dates[sel_index], dates[cal_index], 4),
                  calibration=before(dates[cal_index], evaluation[0], 4), evaluation=evaluation)
    dense = {**recent, 'train': before(start, dates[sel_index], 96)}
    plans = dict(legacy24=old, recent24=recent, recent96=dense)
    for plan in plans.values():
        validate_date_plan(plan, dates, fold, sealed)
    return plans


def prediction_diagnostics(rows, targets, pred, train_x, eval_x, names):
    lo, hi = np.quantile(train_x, [.01, .99], axis=0)
    drift = pd.DataFrame({'feature': names, 'train_p01': lo, 'train_p99': hi,
                          'evaluation_outside_train_p01_p99':
                          ((eval_x < lo) | (eval_x > hi)).mean(axis=0)})
    records = []
    for date in sorted(rows.date.unique()):
        ids = rows.date.eq(date).to_numpy()
        y, p = targets[ids, 4, 3], pred[ids, 4, 3]
        valid = np.isfinite(y)
        records.append(dict(date=date, rows=int(ids.sum()),
            predicted_median_mean=float(p[:, 1].mean()),
            predicted_median_std=float(p[:, 1].std()),
            predicted_positive_fraction=float((p[:, 1] > 0).mean()),
            median_signed_error=float((p[valid, 1]-y[valid]).mean()),
            **{f'actual_below_q{int(q*100)}': float((y[valid] <= p[valid, j]).mean())
               for j, q in enumerate([.1, .5, .9])}))
    return pd.DataFrame(records), drift


def paired_comparison(daily, candidate, baseline, seed=17, repetitions=2000):
    cols = ['window', 'date']
    a = daily.loc[daily.model.eq(candidate)].set_index(cols)
    b = daily.loc[daily.model.eq(baseline)].set_index(cols)
    if a.index.has_duplicates or b.index.has_duplicates or set(a.index) != set(b.index):
        raise ValueError('Paired date cohorts differ')
    b = b.reindex(a.index)
    if not np.array_equal(a.scored_rows, b.scored_rows):
        raise ValueError('Scored cohorts differ')
    result = {}
    rng = np.random.default_rng(seed)
    for metric in ['pinball', 'median_mae', 'all_ohlc_pinball']:
        pairs = [np.column_stack([a.loc[w, metric], b.loc[w, metric]])
                 for w in sorted(a.index.get_level_values('window').unique())]
        if any(not np.isfinite(x).all() for x in pairs):
            raise ValueError('Missing paired metrics')
        gains = [1-x[:, 0].mean()/x[:, 1].mean() for x in pairs]
        # Circular two-date blocks retain nearby-date dependence within each window.
        draws = []
        for _ in range(repetitions):
            means = []
            for x in pairs:
                starts = rng.integers(len(x), size=(len(x)+1)//2)
                ids = np.column_stack([starts, (starts+1) % len(x)]).ravel()[:len(x)]
                means.append(x[ids].mean(axis=0))
            ma, mb = np.mean(means, axis=0)
            draws.append(1-ma/mb)
        means = np.mean([x.mean(axis=0) for x in pairs], axis=0)
        result[metric] = {'relative_improvement': float(1-means[0]/means[1]),
                          'winning_windows': int(sum(g > 0 for g in gains)),
                          'by_window': gains,
                          'conditional_block_bootstrap_95': np.quantile(draws, [.025, .975]).tolist()}
    return result
