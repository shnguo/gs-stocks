"""Project individual sampled OHLC paths onto fixed plans, retaining abstentions."""
import numpy as np

from quant_research.plan_targets import plan_targets
from quant_research.plan_value import HEADS, PROBABILITIES, date_weights, head_targets


def path_heads(paths, reference, min_valid=8, min_filled=4):
    paths, reference = np.asarray(paths), np.asarray(reference)
    if paths.ndim != 4 or paths.shape[2:] != (5, 6) or reference.shape != (len(paths),):
        raise ValueError('Expected signal/sample/five-day/OHLCVA paths and signal references')
    if min_valid < 1 or min_filled < 1 or min_valid > paths.shape[1]:
        raise ValueError('Invalid sample thresholds')
    n, samples = paths.shape[:2]
    valid = np.isfinite(paths).all((2, 3))
    valid &= (paths[..., :4] > 0).all((2, 3)) & (paths[..., 4:] >= 0).all((2, 3))
    valid &= (paths[..., 1] >= paths[..., :4].max(-1)).all(2)
    valid &= (paths[..., 2] <= paths[..., :4].min(-1)).all(2)
    labels = dict(reference=np.repeat(reference, samples),
        future=paths[..., :4].reshape(-1, 5, 4), valid=np.repeat(valid.reshape(-1, 1), 5, 1))
    outcomes = plan_targets(labels)
    targets = head_targets(outcomes)
    result, counts = {}, {}
    for h in HEADS:
        y = targets[h].reshape(n, samples, 12)
        known = np.isfinite(y)
        count = known.sum(1)
        mean = np.divide(np.where(known, y, 0).sum(1), count,
            out=np.full((n, 12), np.nan), where=count > 0)
        minimum = min_valid if h == HEADS[0] else min_filled
        mean[(count < minimum) | (valid.sum(1)[:, None] < min_valid)] = np.nan
        result[h], counts[h] = mean, count
    return result, {'valid_paths': valid, 'known_path_counts': counts,
        'ambiguous_paths': outcomes['ambiguous'].reshape(n, samples, 12).sum(1)}


def calibrate_sparse(predictions, outcomes, dates, min_dates=4):
    """Same calibration family as baseline; insufficient support keeps identity."""
    from sklearn.linear_model import LogisticRegression
    targets, result = head_targets(outcomes), {}
    dates = np.asarray(dates)
    for h in HEADS:
        result[h] = []
        for p in range(predictions[h].shape[1]):
            y, pred = targets[h][:, p], predictions[h][:, p]
            known = np.isfinite(y) & np.isfinite(pred)
            state = {'known_rows': int(known.sum()), 'known_dates': len(set(dates[known]))}
            if state['known_dates'] < min_dates:
                state.update(kind='identity', reason='insufficient calibration dates')
            else:
                w = date_weights(dates, known)
                if h in PROBABILITIES:
                    if np.unique(y[known]).size < 2 or np.std(pred[known]) < 1e-12:
                        state.update(kind='constant', value=float(np.average(y[known], weights=w)))
                    else:
                        prob = np.clip(pred[known], 1e-6, 1-1e-6)
                        model = LogisticRegression(C=1., solver='lbfgs', random_state=17)
                        model.fit(np.log(prob/(1-prob))[:, None], y[known], sample_weight=w*len(w)/w.sum())
                        state.update(kind='sigmoid', slope=float(model.coef_[0, 0]),
                            intercept=float(model.intercept_[0]))
                else:
                    state.update(kind='offset', offset=float(np.average(y[known]-pred[known], weights=w)))
            result[h].append(state)
    return result


def apply_sparse_calibration(predictions, calibration):
    result = {}
    for h, values in predictions.items():
        out = values.copy()
        for p, state in enumerate(calibration[h]):
            known = np.isfinite(values[:, p])
            pred = values[known, p]
            if state['kind'] == 'constant':
                out[known, p] = state['value']
            elif state['kind'] == 'sigmoid':
                prob = np.clip(pred, 1e-6, 1-1e-6)
                logit = np.clip(state['slope']*np.log(prob/(1-prob))+state['intercept'], -40, 40)
                out[known, p] = 1/(1+np.exp(-logit))
            elif state['kind'] == 'offset':
                out[known, p] = pred+state['offset']
            elif state['kind'] != 'identity':
                raise ValueError('Unknown sparse calibration kind')
        result[h] = np.maximum(out, 0) if h == HEADS[4] else out
    return result
