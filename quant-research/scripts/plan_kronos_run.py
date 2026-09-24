"""Frozen post-pretraining-cutoff three-model plan comparison from sampled paths."""
import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from price_pilot import load_arrays, verify_files

from quant_research.plan_paths import apply_sparse_calibration, calibrate_sparse, path_heads
from quant_research.plan_targets import plan_targets
from quant_research.plan_value import HEADS, PROBABILITIES, head_targets
from quant_research.storage import file_hash, utc_now, write_json


def read(path):
    return json.loads(path.read_text())


def preflight_rows(rows, manifest, protocol):
    axis = {d: i for i, d in enumerate(manifest['dates'])}
    for row in rows.itertuples():
        t = axis.get(row.date, -1)
        if t < 59 or t+5 >= len(axis):
            raise ValueError(f'Incomplete Kronos input calendar: {row.date}')
        if manifest['dates'][t+5] != row.label_end:
            raise ValueError('Kronos calendar and label horizon differ')
        if row.label_end >= protocol['sealed_holdout_start']:
            raise ValueError('Kronos horizon crosses sealed holdout')


def register_reuse(previous, inputs, protocol, chosen, prior, tf, bundle):
    state = read(previous/'run-status.json')
    if state['status'] not in ['failed', 'completed']:
        raise ValueError('Reuse requires a terminal previous attempt')
    verify_files(previous, read(previous/'frozen-manifest.json'))
    old = read(previous/'experiment.json')
    if read(previous/'protocol.json') != protocol or read(previous/'identities.json') != chosen:
        raise ValueError('Reuse protocol or identity cohort differs')
    for key, path in [('prior', prior), ('transformer', tf), ('bundle', bundle)]:
        if Path(old[key]) != path:
            raise ValueError('Reuse parent or pretrained bundle differs')
    oldinputs = Path(old.get('input_panel', str(bundle/'inputs')))
    om, nm = read(oldinputs/'manifest.json'), read(inputs/'manifest.json')
    if file_hash(oldinputs/'manifest.json') != old['input_manifest_sha256']:
        raise ValueError('Original input manifest changed')
    if (om['instruments'] != nm['instruments'] or om['fields'] != nm['fields']
            or om['dataset_id'] != nm['dataset_id'] or nm['dates'][:len(om['dates'])] != om['dates']):
        raise ValueError('Input axes are not an exact extension')
    if file_hash(oldinputs/'values.npy') != om['values_sha256']:
        raise ValueError('Original input values changed')
    ov, nv = [np.load(root/'values.npy', mmap_mode='r') for root in [oldinputs, inputs]]
    for stock in range(len(ov)):
        np.testing.assert_array_equal(ov[stock], nv[stock, :len(om['dates'])])
    files = {}
    for fold in protocol['fold_indices']:
        dest = previous/f'fold-{fold:02d}'
        if (dest/'completed.json').exists():
            verify_files(dest, read(dest/'completed.json')['files'])
        for part in ['calibration', 'evaluation']:
            rows = pd.read_parquet(previous/f'fold-{fold:02d}-{part}-rows.parquet')
            expected = pd.read_parquet(prior/f'fold-{fold:02d}'/f'{part}-rows.parquet')
            expected = expected.loc[expected.instrument_id.isin(chosen)].copy()
            expected['parent_row_index'] = expected.index
            pd.testing.assert_frame_equal(rows, expected.reset_index(drop=True))
            chunks = sorted(dest.glob(f'{part}-paths-*.npz'))
            if chunks:
                preflight_rows(rows, om, protocol)
                for f in [dest/f'{part}-input-ledger.json', *chunks]:
                    files[str(f.relative_to(previous))] = file_hash(f)
    return dict(previous=str(previous), input_panel=str(oldinputs), previous_status=state,
        input_manifest_sha256=file_hash(oldinputs/'manifest.json'), old_prefix_values_verified=int(ov.size), files=files)


def freeze(args):
    prior, tf, bundle, out = [p.resolve() for p in [args.prior, args.transformer, args.bundle, args.output]]
    protocol = read(args.protocol)
    if read(prior/'run-status.json')['status'] != 'completed' or not read(prior/'verification.json')['passed']:
        raise ValueError('Unverified LightGBM baseline')
    verify_files(prior, read(prior/'run-status.json')['files'])
    inputs = (args.inputs or bundle/'inputs').resolve()
    manifest, provenance = read(inputs/'manifest.json'), read(bundle/'pretrained-provenance.json')
    verify_files(bundle, provenance['files'])
    if file_hash(inputs/'values.npy') != manifest['values_sha256']:
        raise ValueError('Raw panel changed')
    first = pd.read_parquet(prior/'fold-13/calibration-rows.parquet')
    universe = first.loc[first.date.eq(first.date.min()), 'instrument_id'].unique()
    chosen = []
    for exchange in ['xshg', 'xshe', 'xbse']:
        identities = [s for s in universe if s.split('.')[1] == exchange]
        order = sorted(identities, key=lambda s: hashlib.sha256(f"{protocol['seed']}:{s}".encode()).hexdigest())
        chosen.extend(order[:protocol['identities_per_exchange']])
    for fold in protocol['fold_indices']:
        cfg = read(prior/f'fold-{fold:02d}'/'config.json')
        if cfg['dataset_id'] != manifest['dataset_id']:
            raise ValueError('Different underlying dataset')
        if min(cfg['dates']['calibration']) <= protocol['paper_pretraining_cutoff']:
            raise ValueError('Calibration overlaps paper pretraining period')
        if max(cfg['dates']['evaluation']) >= protocol['sealed_holdout_start']:
            raise ValueError('Sealed holdout boundary')
        for part in ['calibration', 'evaluation']:
            rows = pd.read_parquet(prior/f'fold-{fold:02d}'/f'{part}-rows.parquet')
            preflight_rows(rows.loc[rows.instrument_id.isin(chosen)], manifest, protocol)
    reuse = register_reuse(args.reuse.resolve(), inputs, protocol, chosen, prior, tf, bundle) if args.reuse else None
    out.mkdir()
    write_json(out/'protocol.json', protocol)
    write_json(out/'identities.json', chosen)
    write_json(out/'experiment.json', {'created_at': utc_now(), 'prior': str(prior), 'transformer': str(tf),
        'bundle': str(bundle), 'parent_status_sha256': file_hash(prior/'run-status.json'),
        'transformer_frozen_sha256': file_hash(tf/'frozen-manifest.json'),
        'pretrained_provenance_sha256': file_hash(bundle/'pretrained-provenance.json'),
        'input_panel': str(inputs), 'input_manifest_sha256': file_hash(inputs/'manifest.json'),
        'exact_checkpoint_cutoff_independently_verified': False, 'executable': False})
    if reuse:
        write_json(out/'reuse-manifest.json', reuse)
    for fold in protocol['fold_indices']:
        for part in ['calibration', 'evaluation']:
            rows = pd.read_parquet(prior/f'fold-{fold:02d}'/f'{part}-rows.parquet')
            selected = rows.loc[rows.instrument_id.isin(chosen)].copy()
            selected['parent_row_index'] = selected.index
            selected.to_parquet(out/f'fold-{fold:02d}-{part}-rows.parquet', index=False)
    base = Path(__file__).resolve().parents[1]
    shutil.copytree(base/'src/quant_research', out/'code/src/quant_research', ignore=shutil.ignore_patterns('__pycache__'))
    (out/'code/scripts').mkdir()
    for name in ['plan_kronos_run.py', 'price_pilot.py']:
        shutil.copy2(base/'scripts'/name, out/'code/scripts'/name)
    shutil.copy2(base/'uv.lock', out/'code/uv.lock')
    write_json(out/'frozen-manifest.json', {str(f.relative_to(out)): file_hash(f) for f in out.rglob('*') if f.is_file()})
    print('Frozen', out, 'identities', len(chosen), flush=True)


def matched_metrics(rows, outcomes, models, fold, version):
    targets, records = head_targets(outcomes), []
    for date in sorted(rows.date.unique()):
        ids = rows.date.eq(date).to_numpy()
        for h in HEADS:
            y = targets[h][ids].ravel()
            available = np.ones(len(y), bool)
            for pred in models.values():
                available &= np.isfinite(pred[h][ids].ravel())
            known = available & np.isfinite(y)
            for name, prediction in models.items():
                pr = prediction[h][ids].ravel()
                item = dict(window=f'fold-{fold:02d}', version=version, model=name, date=date, head=h,
                    cohort_plan_rows=len(y), label_known=int(np.isfinite(y).sum()),
                    model_available=int(np.isfinite(pr).sum()), common_available=int(available.sum()),
                    common_known=int(known.sum()),
                    mse=float(np.mean((pr[known]-y[known])**2)) if known.any() else None,
                    mean_prediction=float(pr[known].mean()) if known.any() else None,
                    mean_actual=float(y[known].mean()) if known.any() else None)
                if h in PROBABILITIES:
                    item['brier'] = item['mse']
                records.append(item)
    return pd.DataFrame(records)


def run(args):
    from quant_research.price_kronos import native_paths
    out = args.output.resolve()
    verify_files(out, read(out/'frozen-manifest.json'))
    e, protocol = read(out/'experiment.json'), read(out/'protocol.json')
    prior, tf, bundle = [Path(e[k]) for k in ['prior', 'transformer', 'bundle']]
    inputs = Path(e.get('input_panel', str(bundle/'inputs')))
    reuse = read(out/'reuse-manifest.json') if (out/'reuse-manifest.json').exists() else None
    if reuse:
        verify_files(Path(reuse['previous']), reuse['files'])
    if file_hash(prior/'run-status.json') != e['parent_status_sha256']:
        raise ValueError('LightGBM parent changed')
    if file_hash(tf/'frozen-manifest.json') != e['transformer_frozen_sha256']:
        raise ValueError('Transformer protocol changed')
    state = read(tf/'run-status.json')
    if state['status'] != 'completed' or not read(tf/'verification.json')['passed']:
        raise ValueError('Transformer must be finished and verified before GPU inference')
    verify_files(tf, state['files'])
    verify_files(prior, read(prior/'run-status.json')['files'])
    if file_hash(bundle/'pretrained-provenance.json') != e['pretrained_provenance_sha256']:
        raise ValueError('Checkpoint provenance changed')
    if file_hash(inputs/'manifest.json') != e['input_manifest_sha256']:
        raise ValueError('Input manifest changed')
    manifest = read(inputs/'manifest.json')
    if file_hash(inputs/'values.npy') != manifest['values_sha256']:
        raise ValueError('Raw panel changed')
    raw = np.load(inputs/'values.npy', mmap_mode='r')
    stock_axis = {s: i for i, s in enumerate(manifest['instruments'])}
    date_axis = {d: i for i, d in enumerate(manifest['dates'])}
    marker = out/'run-status.json'
    if marker.exists():
        raise FileExistsError('Preserve previous attempts')
    write_json(marker, dict(status='running', pid=os.getpid(), started_at=utc_now(),
        transformer_status_sha256=file_hash(tf/'run-status.json')))
    reports = []
    try:
        for fold in protocol['fold_indices']:
            dest = out/f'fold-{fold:02d}'
            dest.mkdir()
            parts, predictions = {}, {}
            for part_number, part in enumerate(['calibration', 'evaluation']):
                rows = pd.read_parquet(out/f'fold-{fold:02d}-{part}-rows.parquet')
                preflight_rows(rows, manifest, protocol)
                labels = load_arrays(prior/f'fold-{fold:02d}'/f'{part}.npz')
                labels = {k: v[rows.parent_row_index.to_numpy()] for k, v in labels.items()}
                paths = np.full((len(rows), protocol['path_samples'], 5, 6), np.nan, np.float32)
                ledger, windows, past, future, slots = [], [], [], [], []
                for i, row in rows.iterrows():
                    s, t = stock_axis.get(row.instrument_id), date_axis[row.date]
                    if t+5 >= len(manifest['dates']) or manifest['dates'][t+5] >= protocol['sealed_holdout_start']:
                        raise ValueError('Forecast horizon crosses sealed boundary')
                    window = None if s is None else raw[s, t-59:t+1]
                    good = (window is not None and window.shape == (60, 7) and np.isfinite(window).all()
                        and (window[:, :4] > 0).all() and (window[:, 6] > 0).all() and (window[:, 4:6] >= 0).all())
                    ledger.append(dict(row_index=i, instrument_id=row.instrument_id, date=row.date, input_available=bool(good)))
                    if good:
                        # Ensure the historical close used to normalize paths matches the plan reference.
                        np.testing.assert_allclose(window[-1, 3], labels['reference'][i], rtol=1e-6, atol=1e-6)
                        windows.append(window.copy())
                        past.append(manifest['dates'][t-59:t+1])
                        future.append(manifest['dates'][t+1:t+6])
                        slots.append(i)
                write_json(dest/f'{part}-input-ledger.json', ledger)
                ledger_key = f'fold-{fold:02d}/{part}-input-ledger.json'
                if reuse and ledger_key in reuse['files']:
                    if read(Path(reuse['previous'])/ledger_key) != ledger:
                        raise ValueError('Reused input availability changed')
                for start in range(0, len(slots), protocol['chunk_size']):
                    end = min(start+protocol['chunk_size'], len(slots))
                    seed = protocol['seed']+fold*100000+part_number*10000+start
                    chunk = dest/f'{part}-paths-{start:05d}.npz'
                    key = str(chunk.relative_to(out))
                    reused = bool(reuse and key in reuse['files'])
                    if reused:
                        original = Path(reuse['previous'])/key
                        if file_hash(original) != reuse['files'][key]:
                            raise ValueError('Reused chunk changed after registration')
                        with np.load(original) as saved:
                            np.testing.assert_array_equal(saved['row_indices'], slots[start:end])
                            assert saved['seed'].item() == seed
                            generated = saved['paths'].copy()
                        if generated.shape != (end-start, protocol['path_samples'], 5, 6):
                            raise ValueError('Reused path shape differs')
                        shutil.copy2(original, chunk)
                    else:
                        generated = native_paths(bundle, np.stack(windows[start:end]), past[start:end], future[start:end],
                            samples=protocol['path_samples'], seed=seed, device=protocol['device'])
                        np.savez_compressed(chunk, row_indices=slots[start:end], paths=generated, seed=seed)
                    paths[np.array(slots[start:end])] = generated
                    write_json(out/'progress.json', dict(status='running', fold=fold, partition=part,
                        completed_rows=end, total_rows=len(slots), reused_chunk=reused, updated_at=utc_now()))
                pred, coverage = path_heads(paths, labels['reference'], protocol['min_valid_paths'], protocol['min_filled_paths'])
                np.savez_compressed(dest/f'{part}-path-coverage.npz', valid_paths=coverage['valid_paths'],
                    ambiguous_paths=coverage['ambiguous_paths'], **coverage['known_path_counts'])
                np.savez_compressed(dest/f'raw-{part}.npz', **pred)
                parts[part], predictions[part] = (rows, labels), pred
            cr, cl = parts['calibration']
            calibration = calibrate_sparse(predictions['calibration'], plan_targets(cl), cr.date,
                min_dates=protocol['calibration_min_dates'])
            write_json(dest/'calibration.json', calibration)
            calibrated = apply_sparse_calibration(predictions['evaluation'], calibration)
            np.savez_compressed(dest/'calibrated-evaluation.npz', **calibrated)
            rows, labels = parts['evaluation']
            ids = rows.parent_row_index.to_numpy()
            for version, kp in [('raw', predictions['evaluation']), ('calibrated', calibrated)]:
                suffix = '_raw' if version == 'raw' else ''
                models = {'kronos': kp}
                for name, root, prefix in [('lightgbm', prior, 'learned'), ('transformer', tf, 'transformer')]:
                    models[name] = {h: v[ids] for h, v in load_arrays(root/f'fold-{fold:02d}'/f'{prefix}{suffix}-evaluation.npz').items()}
                reports.append(matched_metrics(rows, plan_targets(labels), models, fold, version))
            write_json(dest/'completed.json', dict(status='completed', updated_at=utc_now(), files={
                str(f.relative_to(dest)): file_hash(f) for f in dest.rglob('*') if f.is_file()}))
        pd.concat(reports, ignore_index=True).to_csv(out/'matched-head-metrics.csv', index=False)
        write_json(out/'assessment.json', dict(quality_promotion=False, executable=False,
            inference='native frozen model paths', comparison_scope='two post-paper-cutoff development windows',
            missing_prediction_policy='per-head common finite forecast cohort, availability separately reported',
            checkpoint_cutoff_independently_verified=False))
        write_json(out/'progress.json', dict(status='completed', updated_at=utc_now()))
        write_json(marker, dict(status='completed', finished_at=utc_now(), files={
            str(f.relative_to(out)): file_hash(f) for f in out.rglob('*') if f.is_file()
            and f.name != 'run-status.json' and '__pycache__' not in f.parts}))
    except BaseException as exc:
        write_json(marker, dict(status='failed', error=repr(exc), updated_at=utc_now()))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['freeze', 'run'])
    for name in ['prior', 'transformer', 'bundle', 'protocol', 'inputs', 'reuse']:
        p.add_argument('--'+name, type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    globals()[args.stage](args)


if __name__ == '__main__':
    main()
