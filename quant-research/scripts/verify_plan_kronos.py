"""Audit path retention, scalar plan scenarios, sparse calibration and matched metrics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.plan_paths import path_heads
from quant_research.plan_targets import plan_targets
from quant_research.plan_value import HEADS, head_targets
from quant_research.price_strategy import TradeAssumptions, candidate_plans, trade_diagnostic
from quant_research.storage import file_hash, utc_now, write_json


def read(p):
    return json.loads(p.read_text())


def load(p):
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def hashes(root, files):
    for name, expected in files.items():
        p = (root/name).resolve()
        if not p.is_relative_to(root) or file_hash(p) != expected:
            raise ValueError(f'Changed evidence: {name}')
    return len(files)


def scalar_heads(paths, reference, minimum, min_filled):
    values = np.full((len(paths), 12, 5), np.nan)
    good = np.isfinite(paths).all((1, 2)) & (paths[:, :, :4] > 0).all((1, 2))
    good &= (paths[:, :, 4:] >= 0).all((1, 2))
    good &= (paths[:, :, 1] >= paths[:, :, :4].max(2)).all(1)
    good &= (paths[:, :, 2] <= paths[:, :, :4].min(2)).all(1)
    for i in np.flatnonzero(good):
        for j, plan in enumerate(candidate_plans(reference, TradeAssumptions())):
            r = trade_diagnostic(paths[i, :, :4].astype(float), np.ones(5, bool), plan, TradeAssumptions())
            if r['filled'] is not None:
                values[i, j, 0] = float(r['filled'])
            if r['filled']:
                resolved = r['net_return'] is not None and not r['ambiguous']
                values[i, j, 1] = float(resolved)
                if resolved:
                    net = r['net_return']
                    values[i, j, 2:] = [net, float(net < 0), max(-net, 0)]
    result = np.full((12, 5), np.nan)
    for j in range(12):
        for h in range(5):
            v = values[:, j, h]
            known = np.isfinite(v)
            threshold = minimum if h == 0 else min_filled
            if good.sum() >= minimum and known.sum() >= threshold:
                result[j, h] = v[known].mean()
    return result, int(good.sum())*12


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output.resolve()
    state, e, protocol = [read(out/n) for n in ['run-status.json', 'experiment.json', 'protocol.json']]
    if state['status'] != 'completed':
        raise ValueError('Kronos inference incomplete')
    if (out/'verification.json').exists():
        raise FileExistsError('Preserve prior verification')
    prior, tf, bundle = [Path(e[k]) for k in ['prior', 'transformer', 'bundle']]
    inputs = Path(e.get('input_panel', str(bundle/'inputs')))
    counts = dict(hashes=hashes(out, state['files']), retained_paths=0, scalar_plan_scenarios=0,
        path_head_values=0, matched_head_metrics=0)
    if file_hash(prior/'run-status.json') != e['parent_status_sha256']:
        raise ValueError('Parent changed')
    for root in [prior, tf]:
        counts['hashes'] += hashes(root, read(root/'run-status.json')['files'])
    if not read(tf/'verification.json')['passed']:
        raise ValueError('Transformer not verified')
    if file_hash(bundle/'pretrained-provenance.json') != e['pretrained_provenance_sha256']:
        raise ValueError('Pretrained checkpoint provenance changed')
    if file_hash(inputs/'manifest.json') != e['input_manifest_sha256']:
        raise ValueError('Raw input manifest changed')
    input_manifest = read(inputs/'manifest.json')
    if file_hash(inputs/'values.npy') != input_manifest['values_sha256']:
        raise ValueError('Historical raw inputs changed')
    counts['hashes'] += hashes(bundle, read(bundle/'pretrained-provenance.json')['files'])
    if (out/'reuse-manifest.json').exists():
        reuse = read(out/'reuse-manifest.json')
        original = Path(reuse['previous'])
        counts['hashes'] += hashes(original, reuse['files'])
        counts['hashes'] += hashes(out, reuse['files'])
        oldinputs = Path(reuse['input_panel'])
        old_manifest = read(oldinputs/'manifest.json')
        assert file_hash(oldinputs/'manifest.json') == reuse['input_manifest_sha256']
        assert file_hash(oldinputs/'values.npy') == old_manifest['values_sha256']
        assert old_manifest['instruments'] == input_manifest['instruments']
        assert old_manifest['fields'] == input_manifest['fields']
        assert input_manifest['dates'][:len(old_manifest['dates'])] == old_manifest['dates']
        ov, nv = [np.load(p/'values.npy', mmap_mode='r') for p in [oldinputs, inputs]]
        for stock in range(len(ov)):
            np.testing.assert_array_equal(ov[stock], nv[stock, :len(old_manifest['dates'])])
        counts['reused_input_values'] = int(ov.size)
        counts['reused_chunks'] = sum(n.endswith('.npz') for n in reuse['files'])
    universe = pd.read_parquet(prior/'fold-13/calibration-rows.parquet')
    universe = universe.loc[universe.date.eq(universe.date.min()), 'instrument_id'].unique()
    chosen = []
    for exchange in ['xshg', 'xshe', 'xbse']:
        ids = [s for s in universe if s.split('.')[1] == exchange]
        chosen.extend(sorted(ids, key=lambda s: hashlib.sha256(f"{protocol['seed']}:{s}".encode()).hexdigest())[:protocol['identities_per_exchange']])
    assert chosen == read(out/'identities.json')
    recorded_metrics = pd.read_csv(out/'matched-head-metrics.csv')
    summaries = []
    for fold in protocol['fold_indices']:
        dest = out/f'fold-{fold:02d}'
        for part in ['calibration', 'evaluation']:
            parent_rows = pd.read_parquet(prior/f'fold-{fold:02d}'/f'{part}-rows.parquet')
            rows = pd.read_parquet(out/f'fold-{fold:02d}-{part}-rows.parquet')
            expected_rows = parent_rows.loc[parent_rows.instrument_id.isin(chosen)].copy()
            expected_rows['parent_row_index'] = expected_rows.index
            pd.testing.assert_frame_equal(rows, expected_rows.reset_index(drop=True))
            if rows.date.min() <= protocol['paper_pretraining_cutoff']:
                raise ValueError('Pretraining period overlap')
            if rows.label_end.max() >= protocol['sealed_holdout_start']:
                raise ValueError('Outcome horizon reaches sealed holdout')
            date_axis = {d: i for i, d in enumerate(input_manifest['dates'])}
            for row in rows.itertuples():
                t = date_axis[row.date]
                assert t >= 59 and t+5 < len(date_axis)
                assert input_manifest['dates'][t+5] == row.label_end
            parent_labels = load(prior/f'fold-{fold:02d}'/f'{part}.npz')
            labels = {h: a[rows.parent_row_index.to_numpy()] for h, a in parent_labels.items()}
            paths = np.full((len(rows), protocol['path_samples'], 5, 6), np.nan, np.float32)
            seen = set()
            for chunk in sorted(dest.glob(f'{part}-paths-*.npz')):
                c = load(chunk)
                ids = c['row_indices']
                start = int(chunk.stem.rsplit('-', 1)[1])
                assert c['seed'].item() == protocol['seed']+fold*100000+(['calibration', 'evaluation'].index(part))*10000+start
                assert len(set(ids)) == len(ids) and not (set(ids) & seen)
                assert ((ids >= 0) & (ids < len(rows))).all()
                paths[ids] = c['paths']
                seen.update(ids)
            ledger = read(dest/f'{part}-input-ledger.json')
            assert seen == {r['row_index'] for r in ledger if r['input_available']}
            pred, coverage = path_heads(paths, labels['reference'], protocol['min_valid_paths'], protocol['min_filled_paths'])
            raw = load(dest/f'raw-{part}.npz')
            for h in HEADS:
                np.testing.assert_allclose(pred[h], raw[h], rtol=0, atol=0, equal_nan=True)
                counts['path_head_values'] += pred[h].size
            stored_coverage = load(dest/f'{part}-path-coverage.npz')
            np.testing.assert_array_equal(coverage['valid_paths'], stored_coverage['valid_paths'])
            np.testing.assert_array_equal(coverage['ambiguous_paths'], stored_coverage['ambiguous_paths'])
            for h in HEADS:
                np.testing.assert_array_equal(coverage['known_path_counts'][h], stored_coverage[h])
            # Selection is independent of actual returns; explicitly include valid-path cases.
            sample = set(np.linspace(0, len(rows)-1, 24, dtype=int))
            valid_rows = np.flatnonzero(coverage['valid_paths'].any(1))
            if len(valid_rows):
                sample.update(valid_rows[np.linspace(0, len(valid_rows)-1, min(24, len(valid_rows)), dtype=int)])
            for i in sorted(sample):
                independent, scenarios = scalar_heads(paths[i], labels['reference'][i], protocol['min_valid_paths'], protocol['min_filled_paths'])
                for h, name in enumerate(HEADS):
                    np.testing.assert_allclose(independent[:, h], raw[name][i], rtol=1e-10, atol=1e-12, equal_nan=True)
                counts['scalar_plan_scenarios'] += scenarios
            counts['retained_paths'] += len(seen)*protocol['path_samples']
            summaries.append(dict(fold=fold, partition=part, selected_rows=len(rows),
                input_available=len(seen), valid_paths=int(coverage['valid_paths'].sum()),
                total_drawn_paths=len(seen)*protocol['path_samples'],
                rows_with_minimum_paths=int((coverage['valid_paths'].sum(1) >= protocol['min_valid_paths']).sum()),
                finite_forecasts={h: int(np.isfinite(pred[h]).sum()) for h in HEADS}))
        cal = load(dest/'calibrated-evaluation.npz')
        calibration = read(dest/'calibration.json')
        for h in HEADS:
            np.testing.assert_array_equal(np.isfinite(raw[h]), np.isfinite(cal[h]))
            for j, spec in enumerate(calibration[h]):
                p = raw[h][:, j]
                expected = p.copy()
                good = np.isfinite(p)
                if spec['kind'] == 'constant':
                    expected[good] = spec['value']
                elif spec['kind'] == 'offset':
                    expected[good] += spec['offset']
                elif spec['kind'] == 'sigmoid':
                    z = np.clip(p[good], 1e-6, 1-1e-6)
                    z = np.clip(spec['slope']*np.log(z/(1-z))+spec['intercept'], -40, 40)
                    expected[good] = 1/(1+np.exp(-z))
                else:
                    assert spec['kind'] == 'identity'
                if h == HEADS[4]:
                    expected = np.maximum(expected, 0)
                np.testing.assert_allclose(expected, cal[h][:, j], rtol=0, atol=1e-12, equal_nan=True)
        targets = head_targets(plan_targets(labels))
        for version, kp in [('raw', raw), ('calibrated', cal)]:
            suffix = '_raw' if version == 'raw' else ''
            models = {'kronos': kp}
            for name, root, prefix in [('lightgbm', prior, 'learned'), ('transformer', tf, 'transformer')]:
                models[name] = {h: a[rows.parent_row_index.to_numpy()] for h, a in load(root/f'fold-{fold:02d}'/f'{prefix}{suffix}-evaluation.npz').items()}
            for day in sorted(rows.date.unique()):
                mask = rows.date.eq(day).to_numpy()
                for h in HEADS:
                    y = targets[h][mask].ravel()
                    available = np.logical_and.reduce([np.isfinite(pred[h][mask].ravel()) for pred in models.values()])
                    known = available & np.isfinite(y)
                    for name, pred in models.items():
                        p = pred[h][mask].ravel()
                        r = recorded_metrics.loc[recorded_metrics.window.eq(f'fold-{fold:02d}') & recorded_metrics.version.eq(version)
                            & recorded_metrics.date.eq(day) & recorded_metrics['head'].eq(h) & recorded_metrics.model.eq(name)]
                        assert len(r) == 1
                        r = r.iloc[0]
                        assert (r.cohort_plan_rows, r.label_known, r.model_available, r.common_available, r.common_known) == (
                            len(y), np.isfinite(y).sum(), np.isfinite(p).sum(), available.sum(), known.sum())
                        mse = np.mean((p[known]-y[known])**2) if known.any() else np.nan
                        np.testing.assert_allclose(r.mse, mse, rtol=1e-10, atol=1e-12, equal_nan=True)
                        counts['matched_head_metrics'] += 1
    write_json(out/'verification.json', dict(passed=True, checked_at=utc_now(), counts=counts, coverage=summaries,
        limitations='No independent checkpoint training-date manifest; native paths and daily OHLC scenarios do not prove fills or profitability'))
    print(counts, flush=True)


if __name__ == '__main__':
    main()
