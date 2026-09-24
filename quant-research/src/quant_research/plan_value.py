"""Per-plan conditional mean/risk heads with separately fitted calibration.

Means describe observed, resolved non-ambiguous daily-bar scenarios. They do not
identify returns on censored fills or prove execution. Probability of resolution
is a separate target so coverage cannot disappear into the mean estimate.
"""
import numpy as np
import pandas as pd

HEADS = ('fill_probability', 'resolution_probability_given_fill',
         'net_return_given_resolved_fill', 'loss_probability_given_resolved_fill',
         'downside_given_resolved_fill')
PROBABILITIES = {HEADS[0], HEADS[1], HEADS[3]}


def make_date_plan(dates, fold, protocol):
    from quant_research.rolling_price import sample_dates, validate_date_plan
    lookup = {d: i for i, d in enumerate(dates)}
    def eligible(start, end):
        return [d for d in dates if start <= d < end and lookup[d]+5 < len(dates)
                and dates[lookup[d]+5] < end]
    evaluation = eligible(fold['test_start'], fold['test_end'])[::5][:protocol['evaluation_dates']]
    if len(evaluation) != protocol['evaluation_dates']:
        raise ValueError('Insufficient independent evaluation dates')
    cal_start = dates[lookup[evaluation[0]]-(protocol['calibration_dates']*5+5)]
    sel_start = dates[lookup[cal_start]-(protocol['selection_dates']*5+5)]
    train_start = max(fold['train_start'], str((pd.Timestamp(sel_start)-pd.DateOffset(years=1)).date()))
    plan = dict(train=sample_dates(eligible(train_start, sel_start), protocol['train_dates']),
        selection=eligible(sel_start, cal_start)[::5][:protocol['selection_dates']],
        calibration=eligible(cal_start, evaluation[0])[::5][:protocol['calibration_dates']],
        evaluation=evaluation)
    if any(len(plan[k]) != protocol[f'{k}_dates'] for k in ['train', 'selection', 'calibration', 'evaluation']):
        raise ValueError('Partition dates differ from registered counts')
    validate_date_plan(plan, dates, fold, protocol['sealed_holdout_start'])
    return plan


def head_targets(outcomes):
    filled, conditional = outcomes['filled'], outcomes['conditional_net_return']
    return dict(zip(HEADS, [filled,
        np.where(filled == 1, np.isfinite(conditional).astype(float), np.nan),
        conditional, outcomes['conditional_loss'], outcomes['conditional_downside']]))


def date_weights(dates, valid):
    dates = pd.Series(np.asarray(dates)[valid])
    if dates.empty:
        raise ValueError('No observed labels')
    return 1/dates.map(dates.value_counts()).to_numpy()


def fit_heads(train_x, train_dates, train_outcomes, select_x, select_dates,
              select_outcomes, directory, protocol, categorical_features=None):
    import lightgbm as lgb
    train_y, select_y = head_targets(train_outcomes), head_targets(select_outcomes)
    directory.mkdir()
    metadata = {'heads': {}, 'baseline': {}, 'scope': 'resolved non-ambiguous scenarios only'}
    parameters = protocol['tree_parameters']
    for head in HEADS:
        metadata['heads'][head], metadata['baseline'][head] = [], []
        for plan in range(train_y[head].shape[1]):
            y, v = train_y[head][:, plan], select_y[head][:, plan]
            good, vg = np.isfinite(y), np.isfinite(v)
            if good.sum() < 100 or vg.sum() < 20:
                raise ValueError(f'Insufficient labels for {head} plan {plan}')
            weights, vw = date_weights(train_dates, good), date_weights(select_dates, vg)
            mean = float(np.average(y[good], weights=weights))
            metadata['baseline'][head].append(mean)
            state = {'plan': plan, 'training_known': int(good.sum()),
                     'selection_known': int(vg.sum()), 'baseline': mean}
            if np.unique(y[good]).size == 1:
                state.update(kind='constant', value=mean)
            else:
                options = dict(n_estimators=protocol['iterations'],
                    learning_rate=parameters['learning_rate'], num_leaves=parameters['num_leaves'],
                    min_child_samples=parameters['min_child_samples'], reg_lambda=parameters['reg_lambda'],
                    n_jobs=parameters['n_jobs'], random_state=protocol['seed'], verbosity=-1,
                    deterministic=True, force_col_wise=True)
                fit_options = {}
                if categorical_features is not None:
                    columns = list(categorical_features)
                    if (not columns or len(set(columns)) != len(columns)
                            or any(not isinstance(i, int) or not 0 <= i < train_x.shape[1]
                                   for i in columns)):
                        raise ValueError('Invalid categorical feature indices')
                    options.update(protocol['categorical_parameters'])
                    fit_options['categorical_feature'] = columns
                if head in PROBABILITIES:
                    model = lgb.LGBMClassifier(objective='binary', **options)
                else:
                    # Squared error estimates a conditional mean, unlike median/L1 regression.
                    model = lgb.LGBMRegressor(objective='regression', **options)
                model.fit(train_x[good], y[good], sample_weight=weights,
                    eval_set=[(select_x[vg], v[vg])], eval_sample_weight=[vw],
                    callbacks=[lgb.early_stopping(parameters['early_stopping_rounds'], verbose=False)],
                    **fit_options)
                filename = f'{head}-plan{plan:02d}.txt'
                model.booster_.save_model(str(directory/filename))
                state.update(kind='booster', file=filename, best_iteration=model.best_iteration_)
            metadata['heads'][head].append(state)
        print('Completed plan head', head, flush=True)
    return metadata


def predict_heads(x, directory, metadata):
    import lightgbm as lgb
    result = {}
    for head in HEADS:
        columns = []
        for state in metadata['heads'][head]:
            if state['kind'] == 'constant':
                columns.append(np.full(len(x), state['value']))
            else:
                columns.append(lgb.Booster(model_file=str(directory/state['file'])).predict(x, num_threads=4))
        value = np.column_stack(columns)
        result[head] = np.clip(value, 0, 1) if head in PROBABILITIES else value
        if head == HEADS[4]:
            result[head] = np.maximum(value, 0)
    return result


def baseline_heads(size, metadata):
    return {head: np.broadcast_to(metadata['baseline'][head], (size, len(metadata['baseline'][head]))).copy()
            for head in HEADS}


def fit_calibration(predictions, outcomes, dates):
    from sklearn.linear_model import LogisticRegression
    ys, result = head_targets(outcomes), {}
    for head in HEADS:
        result[head] = []
        for plan in range(ys[head].shape[1]):
            y, pred = ys[head][:, plan], predictions[head][:, plan]
            valid = np.isfinite(y) & np.isfinite(pred)
            if not valid.any():
                raise ValueError('No independent calibration labels')
            weights = date_weights(dates, valid)
            state = {'known_rows': int(valid.sum()),
                     'known_dates': len(set(np.asarray(dates)[valid]))}
            if head in PROBABILITIES:
                if np.unique(y[valid]).size < 2 or np.std(pred[valid]) < 1e-12:
                    state.update(kind='constant', value=float(np.average(y[valid], weights=weights)))
                else:
                    prob = np.clip(pred[valid], 1e-6, 1-1e-6)
                    logits = np.log(prob/(1-prob))[:, None]
                    model = LogisticRegression(C=1., solver='lbfgs', random_state=17)
                    model.fit(logits, y[valid], sample_weight=weights*len(weights)/weights.sum())
                    state.update(kind='sigmoid', slope=float(model.coef_[0, 0]),
                                 intercept=float(model.intercept_[0]))
            else:
                state.update(kind='offset', offset=float(np.average(y[valid]-pred[valid], weights=weights)))
            result[head].append(state)
    return result


def apply_calibration(predictions, calibration):
    result = {}
    for head, values in predictions.items():
        out = np.empty_like(values)
        for plan, state in enumerate(calibration[head]):
            pred = values[:, plan]
            if state['kind'] == 'constant':
                out[:, plan] = state['value']
            elif state['kind'] == 'sigmoid':
                p = np.clip(pred, 1e-6, 1-1e-6)
                logit = np.clip(state['slope']*np.log(p/(1-p))+state['intercept'], -40, 40)
                out[:, plan] = 1/(1+np.exp(-logit))
            else:
                out[:, plan] = pred+state['offset']
        result[head] = np.maximum(out, 0) if head == HEADS[4] else out
    return result


def choose_research_plan(predictions):
    fill, resolution, mean = [predictions[h] for h in HEADS[:3]]
    eligible = (fill >= .3) & (resolution >= .9) & (mean > 0)
    scores = np.where(eligible, fill*mean, -np.inf)
    index = scores.argmax(axis=1)
    index[~eligible.any(axis=1)] = -1
    return index


def head_metrics(rows, outcomes, forecasts, window):
    targets, records = head_targets(outcomes), []
    for model, prediction in forecasts.items():
        for date in sorted(rows.date.unique()):
            ids = rows.date.eq(date).to_numpy()
            for head in HEADS:
                actual, predicted = targets[head][ids].ravel(), prediction[head][ids].ravel()
                valid = np.isfinite(actual)
                if not np.isfinite(predicted).all():
                    raise ValueError('Incomplete predictions')
                y, p = actual[valid], predicted[valid]
                record = dict(window=window, model=model, date=date, head=head,
                    total_plan_rows=len(actual), known_rows=int(valid.sum()),
                    mae=float(np.abs(p-y).mean()) if len(y) else np.nan,
                    mse=float(np.mean((p-y)**2)) if len(y) else np.nan,
                    mean_prediction=float(p.mean()) if len(y) else np.nan,
                    mean_actual=float(y.mean()) if len(y) else np.nan)
                if head in PROBABILITIES and len(y):
                    clipped = np.clip(p, 1e-6, 1-1e-6)
                    record.update(brier=float(np.mean((p-y)**2)),
                        log_loss=float(np.mean(-y*np.log(clipped)-(1-y)*np.log1p(-clipped))))
                records.append(record)
    return pd.DataFrame(records)


def calibration_bins(rows, outcomes, forecasts, window):
    targets, records = head_targets(outcomes), []
    for model, prediction in forecasts.items():
        for head in HEADS:
            y, p = targets[head].ravel(), prediction[head].ravel()
            ds = np.repeat(rows.date.to_numpy(), targets[head].shape[1])
            edges = np.unique(np.quantile(p, [.2, .4, .6, .8]))
            bins = np.searchsorted(edges, p, side='left')
            for group in np.unique(bins):
                selected, known = bins == group, (bins == group) & np.isfinite(y)
                weights = date_weights(ds, known) if known.any() else None
                records.append(dict(window=window, model=model, head=head, bin=int(group),
                    plan_rows=int(selected.sum()), known_rows=int(known.sum()),
                    known_dates=len(set(ds[known])),
                    predicted_mean=float(np.average(p[known], weights=weights)) if known.any() else np.nan,
                    actual_mean=float(np.average(y[known], weights=weights)) if known.any() else np.nan,
                    prediction_min=float(p[selected].min()), prediction_max=float(p[selected].max())))
    return pd.DataFrame(records)


def compare_heads(metrics, repetitions=2000):
    rng, result = np.random.default_rng(17), {}
    for head in HEADS:
        metric = 'brier' if head in PROBABILITIES else 'mse'
        selected = metrics.loc[metrics['head'].eq(head)]
        a = selected.loc[selected.model.eq('learned')].set_index(['window', 'date'])
        b = selected.loc[selected.model.eq('empirical')].set_index(['window', 'date'])
        if a.index.has_duplicates or b.index.has_duplicates or set(a.index) != set(b.index):
            raise ValueError('Unpaired head evaluation')
        b = b.reindex(a.index)
        if not np.array_equal(a.known_rows, b.known_rows):
            raise ValueError('Head observation masks differ')
        pairs = [np.column_stack([a.loc[w, metric], b.loc[w, metric]])
                 for w in sorted(a.index.get_level_values('window').unique())]
        if any(not np.isfinite(x).all() for x in pairs):
            result[head] = {'error': 'insufficient observed dates'}
            continue
        differences = [float(np.mean(x[:, 0]-x[:, 1])) for x in pairs]
        draws = []
        for _ in range(repetitions):
            values = []
            for x in pairs:
                starts = rng.integers(len(x), size=(len(x)+1)//2)
                indices = np.column_stack([starts, (starts+1) % len(x)]).ravel()[:len(x)]
                values.append(np.mean(x[indices, 0]-x[indices, 1]))
            draws.append(np.mean(values))
        result[head] = {'metric': metric, 'learned_minus_empirical': float(np.mean(differences)),
            'by_window': differences, 'winning_windows': sum(v < 0 for v in differences),
            'conditional_two_date_block_95': np.quantile(draws, [.025, .975]).tolist()}
    return result
