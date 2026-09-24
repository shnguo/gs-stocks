"""Frozen development diagnostics for predictive ordering and opportunity coverage.

All cuts depend only on predictions and same-date identities. Every registered
cut is reported; none is selected as a deployable strategy from these results.
"""
import argparse
import hashlib
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.plan_targets import plan_targets
from quant_research.plan_value import HEADS, choose_research_plan
from quant_research.price_strategy import TradeAssumptions
from quant_research.storage import file_hash, utc_now, write_json


def read(p):
    return json.loads(p.read_text())


def load(p):
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def verify(root, files):
    for name, expected in files.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or file_hash(path) != expected:
            raise ValueError(f'Changed artifact: {path}')


def prediction_groups(rows, pred, fractions):
    """Return one plan per stock and group membership without reading outcomes."""
    chosen = choose_research_plan(pred)
    records = []
    tie = np.array([hashlib.sha256(s.encode()).hexdigest() for s in rows.instrument_id])
    for date in sorted(rows.date.unique()):
        candidates = np.flatnonzero(rows.date.eq(date).to_numpy() & (chosen >= 0))
        mean = pred[HEADS[2]][candidates, chosen[candidates]]
        fill = pred[HEADS[0]][candidates, chosen[candidates]]
        for score_name, score in [('conditional_mean', mean), ('fill_times_conditional_mean', fill*mean)]:
            order = candidates[np.lexsort((tie[candidates], -score))]
            for fraction in fractions:
                count = int(np.ceil(len(order)*fraction))
                records.append((date, score_name, 'top_fraction', fraction, order[:count]))
            for group in range(5):
                # Equal-count bins with deterministic handling of tied forecasts.
                member = order[np.arange(len(order))*5//max(len(order), 1) == group]
                records.append((date, score_name, 'quintile_best_first', group+1, member))
    return chosen, records


def mean_or_nan(x):
    x = np.asarray(x, float)
    return float(x[np.isfinite(x)].mean()) if np.isfinite(x).any() else np.nan


def evaluate_groups(rows, pred, base, stress, chosen, groups, fold, model):
    records = []
    for date, score_name, kind, value, ids in groups:
        ix = (ids, chosen[ids])
        conditional = base['conditional_net_return'][ix]
        net = base['scenario_order_net_return'][ix]
        stress_net = stress['scenario_order_net_return'][ix]
        fill = base['filled'][ix]
        known_conditional = np.isfinite(conditional)
        pr_mean = pred[HEADS[2]][ix]
        pr_loss = pred[HEADS[3]][ix]
        records.append(dict(window=f'fold-{fold:02d}', model=model, date=date, score=score_name,
            group_kind=kind, group_value=value, cohort_stocks=int(rows.date.eq(date).sum()),
            eligible_stocks=int(((chosen >= 0) & rows.date.eq(date).to_numpy()).sum()), selected=len(ids),
            fill_known=int(np.isfinite(fill).sum()), filled=int((fill == 1).sum()),
            return_known=int(np.isfinite(net).sum()), return_unknown=int((~np.isfinite(net)).sum()),
            stress_return_known=int(np.isfinite(stress_net).sum()),
            stress_return_unknown=int((~np.isfinite(stress_net)).sum()),
            ambiguous=int(base['ambiguous'][ix].sum()), conditional_known=int(known_conditional.sum()),
            predicted_conditional_mean=mean_or_nan(pr_mean),
            predicted_conditional_mean_on_known=mean_or_nan(pr_mean[known_conditional]),
            observed_conditional_mean=mean_or_nan(conditional),
            known_order_scenario_mean=mean_or_nan(net),
            stress_known_order_scenario_mean=mean_or_nan(stress_net),
            observed_conditional_loss=mean_or_nan(base['conditional_loss'][ix]),
            predicted_loss_on_known=mean_or_nan(pr_loss[known_conditional]),
            observed_conditional_downside=mean_or_nan(base['conditional_downside'][ix]),
            conditional_loss_brier=mean_or_nan((pr_loss-base['conditional_loss'][ix])**2),
            observed_fill_rate=mean_or_nan(fill), predicted_fill_probability=mean_or_nan(pred[HEADS[0]][ix]),
            score_distinct_values=len(np.unique(pr_mean if score_name == 'conditional_mean' else pred[HEADS[0]][ix]*pr_mean))))
    return pd.DataFrame(records)


def freeze(a):
    out, prior = a.output.resolve(), a.prior.resolve()
    rolling = getattr(a,'rolling',None)
    if bool(rolling)==bool(a.transformer):
        raise ValueError('Choose exactly one Transformer or rolling comparison')
    parents = {}
    candidates=[('lightgbm',prior)]
    if not rolling:
        candidates.append(('transformer',a.transformer.resolve()))
    for name, root in candidates:
        state = read(root/'run-status.json')
        if state['status'] != 'completed' or not read(root/'verification.json')['passed']:
            raise ValueError('Parent is not complete and verified')
        verify(root, state['files'])
        parents[name] = dict(root=str(root), status_sha256=file_hash(root/'run-status.json'))
    models=['lightgbm_raw', 'lightgbm_calibrated', 'empirical_raw', 'empirical_calibrated', 'transformer_raw', 'transformer_calibrated']
    if rolling:
        rolling=rolling.resolve()
        verify(rolling,read(rolling/'frozen-manifest.json'))
        if Path(read(rolling/'experiment.json')['prior'])!=prior:
            raise ValueError('Rolling model uses a different evaluation baseline')
        parents['rolling']=dict(root=str(rolling),status_sha256=None,frozen_sha256=file_hash(rolling/'frozen-manifest.json'))
        models=['lightgbm_raw','lightgbm_calibrated','rolling_raw','rolling_calibrated']
    out.mkdir()
    write_json(out/'protocol.json', dict(protocol_id='plan-opportunity-rolling-v1' if rolling else 'plan-opportunity-diagnostic-v1', registered_at=utc_now(),
        fold_indices=read(prior/'protocol.json')['fold_indices'], fractions=[.01, .05, .10, .20, .50, 1.0],
        models=models,
        plan_selection='Unchanged parent eligibility: fill>=0.3, resolution>=0.9, conditional mean>0; maximize fill*conditional mean per stock',
        cuts='Within-date prediction-only ranks; identity SHA256 breaks ties; ceil top counts and five bins; retain empty dates',
        interpretation='All cuts descriptive on already-researched development windows; no threshold selection or quality promotion',
        outcomes='Known conservative daily-bar scenario means, unknowns and ambiguity explicit; not unconditional expected return or portfolio PnL',
        aggregation='First equal weight dates within each window, then equal weight windows; report available dates and windows for every mean',
        sealed_holdout_start=read(prior/'protocol.json')['sealed_holdout_start'], executable=False))
    write_json(out/'parents.json', parents)
    shutil.copy2(Path(__file__), out/'implementation.py')
    module_root = Path(__file__).resolve().parents[1]/'src/quant_research'
    write_json(out/'source-modules.json', {str(f): file_hash(f) for f in module_root.glob('*.py')})
    write_json(out/'frozen-manifest.json', {p.name: file_hash(p) for p in out.iterdir() if p.is_file()})


def run(a):
    out = a.output.resolve()
    verify(out, read(out/'frozen-manifest.json'))
    for path, expected in read(out/'source-modules.json').items():
        if file_hash(Path(path)) != expected:
            raise ValueError('Diagnostic model dependency changed')
    protocol, parents = read(out/'protocol.json'), read(out/'parents.json')
    roots = {k: Path(v['root']) for k, v in parents.items()}
    completions={}
    for name, root in roots.items():
        state=read(root/'run-status.json')
        if state['status']!='completed' or not read(root/'verification.json')['passed']:
            raise ValueError('All forecasting parents must finish and pass independent verification')
        expected=parents[name].get('status_sha256')
        if expected is not None and file_hash(root/'run-status.json') != expected:
            raise ValueError('Parent status changed')
        if parents[name].get('frozen_sha256') and file_hash(root/'frozen-manifest.json')!=parents[name]['frozen_sha256']:
            raise ValueError('Registered rolling protocol changed')
        verify(root,state['files'])
        completions[name]=dict(status_sha256=file_hash(root/'run-status.json'),verification_sha256=file_hash(root/'verification.json'))
    marker = out/'run-status.json'
    if marker.exists():
        raise FileExistsError('Preserve previous attempts')
    write_json(out/'source-completions.json',completions)
    write_json(marker, dict(status='running', pid=os.getpid(), started_at=utc_now()))
    reports = []
    try:
        for fold in protocol['fold_indices']:
            dest = out/f'fold-{fold:02d}'
            dest.mkdir()
            parent = roots['lightgbm']/dest.name
            rows = pd.read_parquet(parent/'evaluation-rows.parquet')
            if rows.label_end.max() >= protocol['sealed_holdout_start']:
                raise ValueError('Sealed labels excluded')
            # Build and save every choice before any outcome arrays are loaded.
            group_sets = {}
            for model in protocol['models']:
                name, version = model.rsplit('_', 1)
                root = roots.get(name,roots['lightgbm'])
                prefix = 'learned' if name == 'lightgbm' else name
                suffix = '_raw' if version == 'raw' else ''
                pred = load(root/dest.name/f'{prefix}{suffix}-evaluation.npz')
                chosen, groups = prediction_groups(rows, pred, protocol['fractions'])
                np.savez_compressed(dest/f'{model}-choices.npz', chosen=chosen,
                    **{f'group_{i}': g[4] for i, g in enumerate(groups)})
                write_json(dest/f'{model}-groups.json', [dict(date=g[0], score=g[1], kind=g[2], value=g[3]) for g in groups])
                group_sets[model] = (pred, chosen, groups)
            labels = load(parent/'evaluation.npz')
            assumptions = TradeAssumptions()
            base = plan_targets(labels)
            stress = plan_targets(labels, replace(assumptions, commission=assumptions.commission*2,
                minimum_fee=assumptions.minimum_fee*2, sell_tax=assumptions.sell_tax*2, slippage_bps=assumptions.slippage_bps*2))
            for model, (pred, chosen, groups) in group_sets.items():
                report = evaluate_groups(rows, pred, base, stress, chosen, groups, fold, model)
                report.to_csv(dest/f'{model}-daily.csv', index=False)
                reports.append(report)
            print('Completed opportunity diagnostic', fold, flush=True)
        daily = pd.concat(reports, ignore_index=True)
        daily.to_csv(out/'daily.csv', index=False)
        keys = ['model', 'score', 'group_kind', 'group_value']
        metrics = [c for c in daily.select_dtypes(include='number').columns if c != 'group_value']
        grouped = daily.groupby(['window', *keys], dropna=False)[metrics]
        window = grouped.mean()
        window.to_csv(out/'window-means.csv')
        grouped.count().to_csv(out/'window-known-date-counts.csv')
        overall = window.groupby(keys)[metrics]
        overall.mean().to_csv(out/'overall-window-equal-means.csv')
        overall.count().to_csv(out/'overall-known-window-counts.csv')
        write_json(marker, dict(status='completed', finished_at=utc_now(), quality_promotion=False, files={
            str(p.relative_to(out)): file_hash(p) for p in out.rglob('*') if p.is_file() and p.name != 'run-status.json'}))
    except BaseException as exc:
        write_json(marker, dict(status='failed', error=repr(exc), updated_at=utc_now()))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['freeze', 'run'])
    for name in ['output', 'prior', 'transformer', 'rolling']:
        p.add_argument('--'+name, type=Path, required=name == 'output')
    a = p.parse_args()
    {'freeze': freeze, 'run': run}[a.stage](a)


if __name__ == '__main__':
    main()
