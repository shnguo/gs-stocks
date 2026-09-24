"""Audit frozen inputs, restored predictions, calibration and all daily head errors."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from quant_research.plan_targets import plan_targets
from quant_research.plan_transformer import PlanTransformer, predict
from quant_research.plan_value import HEADS, apply_calibration, choose_research_plan, head_targets
from quant_research.storage import file_hash, utc_now, write_json


def read(path):
    return json.loads(path.read_text())


def load(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def hashes(root, files):
    for name, expected in files.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or file_hash(path) != expected:
            raise ValueError(f'Changed evidence: {name}')
    return len(files)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fold', type=int, help='Audit one completed fold without claiming experiment completion')
    args = p.parse_args()
    out = args.output.resolve()
    experiment, protocol = read(out/'experiment.json'), read(out/'protocol.json')
    prior, panel = Path(experiment['prior']), Path(experiment['panel'])
    if file_hash(prior/'run-status.json') != experiment['parent_status_sha256']:
        raise ValueError('Parent status changed')
    count = dict(hashes=hashes(out, read(out/'frozen-manifest.json')), restored_head_values=0,
        restored_price_values=0, daily_head_errors=0, calibrated_values=0)
    count['hashes'] += hashes(prior, read(prior/'run-status.json')['files'])
    if file_hash(panel/'manifest.json') != experiment['panel_manifest_sha256']:
        raise ValueError('Panel manifest changed')
    count['hashes'] += hashes(panel, read(panel/'manifest.json')['files'])
    folds = [args.fold] if args.fold is not None else experiment['fold_indices']
    if args.fold is None:
        state = read(out/'run-status.json')
        if state['status'] != 'completed':
            raise ValueError('Experiment is incomplete')
        count['hashes'] += hashes(out, state['files'])
    torch.set_num_threads(4)
    values = np.load(panel/'values.npy', mmap_mode='r')
    for fold in folds:
        if fold not in experiment['fold_indices']:
            raise ValueError('Unregistered fold')
        parent, dest = prior/f'fold-{fold:02d}', out/f'fold-{fold:02d}'
        count['hashes'] += hashes(dest, read(dest/'completed.json')['files'])
        cfg = read(parent/'config.json')
        if max(cfg['dates']['evaluation']) >= protocol['sealed_holdout_start']:
            raise ValueError('Holdout exposure')
        saved = torch.load(dest/'transformer.pt', map_location='cpu', weights_only=True)
        scaler = load(dest/'scalers.npz')
        model = PlanTransformer(scaler['target_center'], scaler['target_scale'], scaler['prior'], protocol['width'])
        model.load_state_dict(saved['state_dict'])
        result, log = read(dest/'training-result.json'), read(dest/'training-log.json')
        best, epoch = float('inf'), 0
        for row in log:
            if row['selection_net_mse'] < best-protocol['min_delta']:
                best, epoch = row['selection_net_mse'], row['epoch']
        assert epoch == result['selected_epoch'] == saved['selected_epoch']
        np.testing.assert_allclose(best, saved['selection_net_mse'], rtol=0, atol=1e-15)
        assert len(log) >= protocol['min_epochs']
        assert result['epochs_completed'] == len(log)
        for part in ['calibration', 'evaluation']:
            rows = pd.read_parquet(parent/f'{part}-rows.parquet')
            # Choose samples independently of forecasts or realized outcomes.
            ids = np.unique(np.linspace(0, len(rows)-1, 48, dtype=int))
            raw_file = 'raw-calibration.npz' if part == 'calibration' else 'transformer_raw-evaluation.npz'
            raw = load(dest/raw_file)
            restored, prices = predict(model, values, rows.iloc[ids], scaler['mean'], scaler['scale'])
            for h in HEADS:
                np.testing.assert_allclose(restored[h], raw[h][ids], rtol=5e-4, atol=5e-6)
                count['restored_head_values'] += restored[h].size
            np.testing.assert_allclose(prices, np.load(dest/f'price-{part}.npy')[ids], rtol=5e-4, atol=5e-6)
            count['restored_price_values'] += prices.size
        labels = load(parent/'evaluation.npz')
        target = head_targets(plan_targets(labels))
        cal = load(dest/'transformer-evaluation.npz')
        recomputed = apply_calibration(raw, read(dest/'calibration.json'))
        for h in HEADS:
            np.testing.assert_allclose(recomputed[h], cal[h], rtol=1e-7, atol=1e-8)
            count['calibrated_values'] += cal[h].size
        metrics = pd.read_csv(dest/'head-metrics.csv')
        for name, pred in [('transformer_raw', raw), ('transformer', cal)]:
            recorded = pd.read_parquet(dest/f'{name}-chosen.parquet')
            # The choice file preserves all candidate row indices, including -1 abstention.
            chosen = choose_research_plan(pred)
            np.testing.assert_array_equal(chosen, recorded.plan_index)
            for date in sorted(rows.date.unique()):
                ids = rows.date.eq(date).to_numpy()
                for h in HEADS:
                    y, pr = target[h][ids].ravel(), pred[h][ids].ravel()
                    known = np.isfinite(y)
                    m = metrics.loc[metrics.model.eq(name) & metrics.date.eq(date) & metrics['head'].eq(h)].iloc[0]
                    assert int(m.known_rows) == int(known.sum())
                    np.testing.assert_allclose(m.mse, np.mean((y[known]-pr[known])**2), rtol=1e-10, atol=1e-12)
                    count['daily_head_errors'] += 1
    filename = 'verification.json' if args.fold is None else f'verification-fold-{args.fold:02d}.json'
    if (out/filename).exists():
        raise FileExistsError('Preserve verification evidence')
    write_json(out/filename, dict(passed=True, scope='full experiment' if args.fold is None else 'one completed fold',
        folds=folds, counts=count, checked_at=utc_now(), executable=False))
    print(count)


if __name__ == '__main__':
    main()
