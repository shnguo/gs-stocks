"""Diagnose partition drift and the exact effect of frozen return calibration."""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.plan_targets import plan_targets
from quant_research.plan_value import HEADS
from quant_research.storage import file_hash, utc_now, write_json


def read(path):
    return json.loads(path.read_text())


def load(path, fields=None):
    with np.load(path) as z:
        return {k: z[k] for k in (fields if fields is not None else z.files)}


def check(root, files):
    for name, expected in files.items():
        p = (root/name).resolve()
        if not p.is_relative_to(root) or file_hash(p) != expected:
            raise ValueError('Parent evidence changed')


def offset_effect(y, raw, corrected, offset):
    known = np.isfinite(y)
    if not known.any():
        return {k: np.nan for k in ['actual_mean', 'raw_mean', 'corrected_mean', 'raw_mse',
            'corrected_mse', 'mse_change', 'offset_cross_term', 'offset_square_term']}
    y, raw, corrected = [a[known].astype(float) for a in [y, raw, corrected]]
    if not np.isfinite(raw).all() or not np.isfinite(corrected).all():
        raise ValueError('Frozen baseline forecast unexpectedly missing')
    np.testing.assert_allclose(raw+offset, corrected, rtol=0, atol=1e-12)
    raw_mse, corrected_mse = np.mean((raw-y)**2), np.mean((corrected-y)**2)
    cross, square = 2*offset*np.mean(raw-y), offset**2
    np.testing.assert_allclose(corrected_mse-raw_mse, cross+square, rtol=1e-8, atol=1e-12)
    return dict(actual_mean=float(y.mean()), raw_mean=float(raw.mean()), corrected_mean=float(corrected.mean()),
        raw_mse=float(raw_mse), corrected_mse=float(corrected_mse), mse_change=float(corrected_mse-raw_mse),
        offset_cross_term=float(cross), offset_square_term=float(square))


def freeze(a):
    prior, out = a.prior.resolve(), a.output.resolve()
    state = read(prior/'run-status.json')
    if state['status'] != 'completed' or not read(prior/'verification.json')['passed']:
        raise ValueError('Parent not verified')
    check(prior, state['files'])
    out.mkdir()
    protocol = read(prior/'protocol.json')
    write_json(out/'protocol.json', dict(registered_at=utc_now(), fold_indices=protocol['fold_indices'],
        partitions=['train', 'selection', 'calibration', 'evaluation'], sealed_holdout_start=protocol['sealed_holdout_start'],
        outcome='Same resolved non-ambiguous filled-order net-return target; every fixed plan, date equally weighted',
        calibration_effect='Exact paired MSE change from frozen scalar offsets; no refitting or deployment',
        scope='Already researched development windows; partition drift is descriptive, not causal attribution',
        executable=False, quality_promotion=False))
    write_json(out/'parent.json', dict(root=str(prior), status_sha256=file_hash(prior/'run-status.json')))
    shutil.copy2(Path(__file__), out/'implementation.py')
    base = Path(__file__).resolve().parents[1]/'src/quant_research'
    write_json(out/'module-hashes.json', {str(base/n):file_hash(base/n) for n in ['plan_targets.py', 'plan_value.py', 'price_strategy.py', 'storage.py']})
    write_json(out/'frozen-manifest.json', {p.name:file_hash(p) for p in out.iterdir() if p.is_file()})


def run(a):
    out = a.output.resolve()
    check(out, read(out/'frozen-manifest.json'))
    for p, h in read(out/'module-hashes.json').items():
        if file_hash(Path(p)) != h:
            raise ValueError('Model dependency changed')
    source, protocol = read(out/'parent.json'), read(out/'protocol.json')
    prior = Path(source['root'])
    if file_hash(prior/'run-status.json') != source['status_sha256']:
        raise ValueError('Parent changed')
    check(prior, read(prior/'run-status.json')['files'])
    marker = out/'run-status.json'
    if marker.exists():
        raise FileExistsError('Preserve previous attempt')
    write_json(marker, dict(status='running', pid=os.getpid(), started_at=utc_now()))
    records, shifts, ages, budgets = [], [], [], []
    try:
        for fold in protocol['fold_indices']:
            parent = prior/f'fold-{fold:02d}'
            cfg, models = read(parent/'config.json'), read(parent/'models.json')
            axis = read(Path(cfg['root'])/'panel-h5/manifest.json')['dates']
            row_parts = {part:pd.read_parquet(parent/f'{part}-rows.parquet') for part in protocol['partitions']}
            first_eval = min(cfg['dates']['evaluation'])
            for part in protocol['partitions']:
                rows = row_parts[part]
                if sorted(rows.date.unique()) != cfg['dates'][part] or rows.label_end.max() >= protocol['sealed_holdout_start']:
                    raise ValueError('Unexpected dates or sealed outcome')
                if part != 'evaluation':
                    next_part = protocol['partitions'][protocol['partitions'].index(part)+1]
                    if rows.label_end.max() >= min(cfg['dates'][next_part]):
                        raise ValueError('Partition outcomes overlap following signal dates')
                    ages.append(dict(window=parent.name, partition=part, last_signal=rows.date.max(),
                        last_label_end=rows.label_end.max(), first_evaluation_signal=first_eval,
                        signal_age_calendar_days=(pd.Timestamp(first_eval)-pd.Timestamp(rows.date.max())).days,
                        signal_age_sessions=axis.index(first_eval)-axis.index(rows.date.max()),
                        label_age_sessions=axis.index(first_eval)-axis.index(rows.label_end.max())))
                labels = load(parent/f'{part}.npz', ['reference', 'future', 'valid', 'upper', 'lower'])
                target = plan_targets(labels)
                for date, ids in rows.groupby('date', sort=True).indices.items():
                    for plan in range(12):
                        y = target['conditional_net_return'][ids, plan]
                        known = np.isfinite(y)
                        values = y[known]
                        filled = target['filled'][ids, plan]
                        records.append(dict(window=parent.name, partition=part, date=date, plan=plan, stock_rows=len(ids),
                            filled=int((filled == 1).sum()), fill_unknown=int(np.isnan(filled).sum()),
                            ambiguous=int(target['ambiguous'][ids, plan].sum()), conditional_known=int(known.sum()),
                            conditional_mean=float(values.mean()) if len(values) else np.nan,
                            conditional_std=float(values.std()) if len(values) else np.nan,
                            conditional_q10=float(np.quantile(values, .1)) if len(values) else np.nan,
                            conditional_q90=float(np.quantile(values, .9)) if len(values) else np.nan,
                            conditional_loss_rate=float((values < 0).mean()) if len(values) else np.nan))
                if part == 'evaluation':
                    raw = load(parent/'learned_raw-evaluation.npz')[HEADS[2]]
                    corrected = load(parent/'learned-evaluation.npz')[HEADS[2]]
                    cal = read(parent/'calibration.json')['learned'][HEADS[2]]
                    for date, ids in rows.groupby('date', sort=True).indices.items():
                        for plan in range(12):
                            if cal[plan]['kind'] != 'offset':
                                raise ValueError('Return calibration is not scalar offset')
                            y = target['conditional_net_return'][ids, plan]
                            effect = offset_effect(y, raw[ids, plan], corrected[ids, plan], cal[plan]['offset'])
                            shifts.append(dict(window=parent.name, date=date, plan=plan, known_rows=int(np.isfinite(y).sum()),
                                offset=cal[plan]['offset'], **effect))
                del labels, target
            for head in HEADS:
                for plan, spec in enumerate(models['heads'][head]):
                    iteration = spec.get('best_iteration')
                    budgets.append(dict(window=parent.name, head=head, plan=plan, kind=spec['kind'],
                        best_iteration=iteration, registered_cap=read(prior/'protocol.json')['iterations'],
                        best_at_cap=iteration == read(prior/'protocol.json')['iterations']))
            print('Completed temporal diagnostic', fold, flush=True)
        daily = pd.DataFrame(records)
        daily.to_csv(out/'partition-daily.csv', index=False)
        summary = daily.groupby(['window', 'partition', 'plan']).conditional_mean.agg(['mean','count'])
        summary.to_csv(out/'partition-plan-means.csv')
        summary['mean'].groupby(['window', 'partition']).mean().unstack('partition').to_csv(out/'partition-means.csv')
        pd.DataFrame(shifts).to_csv(out/'offset-effects.csv', index=False)
        pd.DataFrame(shifts).groupby('window')[['raw_mse','corrected_mse','mse_change','offset_cross_term','offset_square_term']].mean().to_csv(out/'offset-window-means.csv')
        pd.DataFrame(ages).to_csv(out/'partition-ages.csv', index=False)
        pd.DataFrame(budgets).to_csv(out/'tree-budget.csv', index=False)
        write_json(marker, dict(status='completed', finished_at=utc_now(), files={
            p.name:file_hash(p) for p in out.iterdir() if p.is_file() and p.name != 'run-status.json'}))
    except BaseException as exc:
        write_json(marker, dict(status='failed', error=repr(exc), updated_at=utc_now()))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['freeze','run'])
    p.add_argument('--prior', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    {'freeze':freeze, 'run':run}[a.stage](a)


if __name__ == '__main__':
    main()
