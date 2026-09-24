"""Audit one frozen recent-label refit, including maturity and selected budgets."""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from quant_research.plan_targets import plan_targets
from quant_research.plan_value import HEADS, PROBABILITIES, head_targets
from quant_research.storage import file_hash, utc_now, write_json


def read(path):
    return json.loads(path.read_text())


def load(path):
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def check(root, manifest):
    for name, expected in manifest.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root.resolve()) or file_hash(path) != expected:
            raise ValueError(f'Changed evidence: {path}')
    return len(manifest)


def independent_choices(pred, risk):
    fill, resolution, mean, downside = [pd.DataFrame(pred[h]) for h in [HEADS[0], HEADS[1], HEADS[2], HEADS[4]]]
    utility = mean-downside if risk else mean
    score = (fill*utility).where((fill >= .3) & (resolution >= .9) & (mean > 0) & (utility > 0), -np.inf)
    return score.idxmax(axis=1).where(score.max(axis=1).gt(-np.inf), -1).to_numpy()


def audit(root, window, day, target_count_cache=None):
    counts = dict(hashes=check(root, read(root/'frozen-manifest.json')), restored_values=0,
        training_dates=0, choices=0, heads=0)
    e, protocol = read(root/'experiment.json'), read(root/'protocol.json')
    parent = Path(e['parent'])
    assert file_hash(parent/'run-status.json') == e['parent_status_sha256']
    assert file_hash(parent/'verification.json') == e['parent_verification_sha256']
    parent_manifest = read(parent/'run-status.json')['files']
    source = parent/window/'fits'/day
    old = read(source/'config.json')['schedule']
    spec = read(root/'schedule.json')[window][day]
    assert spec['asof'] == day and spec['train'] == old['train'][4:]+old['selection']
    assert spec['removed'] == old['train'][:4] and spec['added'] == old['selection']
    assert len(spec['train']) == len(set(spec['train'])) == 96
    assert spec['latest_label_end'] == old['selection_label_end'] == day
    assert day < spec['forecast_label_end'] < protocol['sealed_holdout_start']
    assert file_hash(source/'models.json') == spec['source_models_sha256']
    assert file_hash(source/'raw-completed.json') == spec['source_completion_sha256']
    counts['hashes'] += check(source, read(source/'raw-completed.json')['files'])
    dest = root/window/'fits'/day
    completed, config, receipts = [read(dest/n) for n in ['completed.json', 'config.json', 'data-receipts.json']]
    assert completed['status'] == 'completed'
    counts['hashes'] += check(dest, completed['files'])
    assert config['schedule'] == spec and config['asof'] == day and config['window'] == window
    assert config['frozen_experiment_sha256'] == file_hash(root/'frozen-manifest.json')
    expected_receipts = {f'day-cache/{d}/{kind}-manifest.json' for d in spec['train'] for kind in ['input', 'label']}
    expected_receipts.add(f'day-cache/{day}/input-manifest.json')
    assert set(receipts['cache_files']) == expected_receipts
    assert receipts['asof'] == day and receipts['latest_training_label'] == day
    assert receipts['training_date_count'] == 96 and receipts['prediction_input_only'] is True
    assert receipts['parent_models_sha256'] == spec['source_models_sha256']
    for name, value in receipts['cache_files'].items():
        assert parent_manifest[name] == value
    counts['hashes'] += check(parent, receipts['cache_files'])
    target_count_cache = {} if target_count_cache is None else target_count_cache
    training_known = {h: np.zeros(12, dtype=int) for h in HEADS}
    for signal in spec['train']:
        directory = parent/'day-cache'/signal
        inp, label = read(directory/'input-manifest.json'), read(directory/'label-manifest.json')
        assert inp['date'] == label['date'] == signal and inp['source_read_dates'] == [signal]
        assert inp['first_requested_asof'] <= day and label['first_requested_asof'] <= day
        assert signal < label['label_end'] <= day
        assert max(label['source_read_dates']) == label['label_end']
        assert label['input_manifest_sha256'] == file_hash(directory/'input-manifest.json')
        counts['hashes'] += check(directory, inp['files'])+check(directory, label['files'])
        if signal not in target_count_cache:
            target = head_targets(plan_targets(load(directory/'labels.npz')))
            target_count_cache[signal] = {h: np.isfinite(values).sum(axis=0) for h, values in target.items()}
        for h in HEADS:
            training_known[h] += target_count_cache[signal][h]
        counts['training_dates'] += 1
    directory = parent/'day-cache'/day
    inp = read(directory/'input-manifest.json')
    assert inp['date'] == day and inp['source_read_dates'] == [day]
    counts['hashes'] += check(directory, inp['files'])
    rows = pd.read_parquet(dest/'prediction-rows.parquet')
    pd.testing.assert_frame_equal(rows, pd.read_parquet(directory/'rows.parquet'))
    pd.testing.assert_frame_equal(rows, pd.read_parquet(source/'prediction-rows.parquet'))
    x = load(directory/'inputs.npz')['x']
    sample = np.unique(np.linspace(0, len(rows)-1, 24, dtype=int))
    pred = load(dest/'raw.npz')
    meta, old_meta = read(dest/'models.json'), read(source/'models.json')
    params = protocol['training']['tree_parameters']
    for head in HEADS:
        assert pred[head].shape == (len(rows), 12) and np.isfinite(pred[head]).all()
        for j, state in enumerate(meta['heads'][head]):
            old_state = old_meta['heads'][head][j]
            iterations = old_state.get('best_iteration', 1)
            assert state['plan'] == j and state['selected_iterations'] == iterations
            assert state['parent_kind'] == old_state['kind']
            assert state['parent_best_iteration'] == old_state.get('best_iteration')
            assert state['training_known'] == training_known[head][j]
            if state['kind'] == 'constant':
                restored = np.full(len(sample), state['value'])
            else:
                booster = lgb.Booster(model_file=str(dest/'models'/state['file']))
                assert booster.current_iteration() == state['actual_iterations'] <= iterations <= 60
                for key, value in {'learning_rate': params['learning_rate'], 'num_leaves': params['num_leaves'],
                    'min_data_in_leaf': params['min_child_samples'], 'lambda_l2': params['reg_lambda'],
                    'num_threads': params['n_jobs'], 'seed': protocol['training']['seed']}.items():
                    assert booster.params[key] == value, (key, booster.params[key], value)
                assert booster.params['device_type'] == 'cpu'
                restored = booster.predict(x[sample], num_threads=1)
            if head in PROBABILITIES:
                restored = np.clip(restored, 0, 1)
            elif head == HEADS[4]:
                restored = np.maximum(restored, 0)
            np.testing.assert_allclose(restored, pred[head][sample, j], atol=1e-12, rtol=1e-12)
            counts['restored_values'] += len(sample)
            counts['heads'] += 1
    choices = load(dest/'choices.npz')
    for rule in ['original', 'risk']:
        np.testing.assert_array_equal(choices[rule], independent_choices(pred, rule == 'risk'))
        counts['choices'] += len(rows)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--window', required=True)
    parser.add_argument('--asof', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    counts = audit(args.experiment.resolve(), args.window, args.asof)
    write_json(args.output, dict(passed=True, checked_at=utc_now(), counts=counts,
        window=args.window, asof=args.asof, verifier_sha256=file_hash(Path(__file__)),
        scope='One refit: all 96 training dates mature by asof, unchanged selected budgets and tree options, all saved heads on24 samples and both plan selectors; not final six-window quality proof'))
    print(counts, flush=True)


if __name__ == '__main__':
    main()
