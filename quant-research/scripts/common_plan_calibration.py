"""Compare model calibration on exactly shared stock/date/plan/known-label support."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from plan_kronos_run import matched_metrics
from plan_value_run import decision_metrics
from price_pilot import load_arrays, verify_files

from quant_research.plan_paths import apply_sparse_calibration, calibrate_sparse
from quant_research.plan_targets import plan_targets
from quant_research.plan_value import HEADS, compare_heads, predict_heads
from quant_research.storage import file_hash, utc_now, write_json


def read(p):
    return json.loads(p.read_text())


def freeze(args):
    root, out = args.kronos.resolve(), args.output.resolve()
    e = read(root/'experiment.json')
    tf, prior = Path(e['transformer']), Path(e['prior'])
    if not read(tf/'verification.json')['passed'] or not read(prior/'verification.json')['passed']:
        raise ValueError('Unverified baseline models')
    out.mkdir()
    write_json(out/'protocol.json', dict(protocol_id='common-plan-calibration-v1', registered_at=utc_now(),
        rationale='Calibration support asymmetry identified before full native comparison results',
        fold_indices=[13, 15], heads=list(HEADS), min_calibration_dates=4,
        calibration='Identical finite raw forecast and known-label support per stock/date/plan/head across all three models; same date-weighted sigmoid/offset family',
        evaluation='Fixed original 192 identities; per-head common forecast and observed-label mask; keep zero-coverage dates',
        decision='Parent fixed-plan selection and base/double costs; retain missing outcomes and abstentions',
        inference='Restore frozen LightGBM boosters; reuse stored raw Transformer and Kronos forecasts. No retraining or extra native sampling',
        promotion=False))
    write_json(out/'experiment.json', dict(kronos=str(root), prior=str(prior), transformer=str(tf),
        kronos_frozen_sha256=file_hash(root/'frozen-manifest.json'),
        prior_status_sha256=file_hash(prior/'run-status.json'), transformer_status_sha256=file_hash(tf/'run-status.json')))
    base = Path(__file__).resolve().parents[1]
    shutil.copytree(base/'src/quant_research', out/'code/src/quant_research', ignore=shutil.ignore_patterns('__pycache__'))
    (out/'code/scripts').mkdir()
    for name in ['common_plan_calibration.py', 'plan_kronos_run.py', 'plan_value_run.py', 'price_pilot.py', 'model_quality_run.py']:
        shutil.copy2(base/'scripts'/name, out/'code/scripts'/name)
    write_json(out/'frozen-manifest.json', {str(p.relative_to(out)): file_hash(p) for p in out.rglob('*') if p.is_file()})
    print('Frozen', out, flush=True)


def run(args):
    out = args.output.resolve()
    verify_files(out, read(out/'frozen-manifest.json'))
    e, protocol = read(out/'experiment.json'), read(out/'protocol.json')
    root, prior, tf = [Path(e[k]) for k in ['kronos', 'prior', 'transformer']]
    if not read(root/'verification.json')['passed']:
        raise ValueError('Native comparison must be verified')
    for base, expected in [(prior, e['prior_status_sha256']), (tf, e['transformer_status_sha256'])]:
        if file_hash(base/'run-status.json') != expected:
            raise ValueError('Baseline status changed')
        verify_files(base, read(base/'run-status.json')['files'])
    if file_hash(root/'frozen-manifest.json') != e['kronos_frozen_sha256']:
        raise ValueError('Native protocol changed')
    verify_files(root, read(root/'run-status.json')['files'])
    marker = out/'run-status.json'
    if marker.exists():
        raise FileExistsError('Preserve previous attempt')
    write_json(marker, dict(status='running', pid=os.getpid(), started_at=utc_now(),
        kronos_verification_sha256=file_hash(root/'verification.json')))
    metrics, decisions = [], []
    try:
        for fold in protocol['fold_indices']:
            dest = out/f'fold-{fold:02d}'
            dest.mkdir()
            parts, labels, raw = {}, {}, {}
            for part in ['calibration', 'evaluation']:
                rows = pd.read_parquet(root/f'fold-{fold:02d}-{part}-rows.parquet')
                ids = rows.parent_row_index.to_numpy()
                parts[part] = rows
                data = load_arrays(prior/f'fold-{fold:02d}'/f'{part}.npz')
                labels[part] = {h: a[ids] for h, a in data.items()}
                native = load_arrays(root/f'fold-{fold:02d}'/f'raw-{part}.npz')
                tr_name = 'raw-calibration.npz' if part == 'calibration' else 'transformer_raw-evaluation.npz'
                temporal = {h: a[ids] for h, a in load_arrays(tf/f'fold-{fold:02d}'/tr_name).items()}
                if part == 'calibration':
                    tree = predict_heads(data['x'][ids], prior/f'fold-{fold:02d}'/'plan-models',
                        read(prior/f'fold-{fold:02d}'/'models.json'))
                else:
                    tree = {h: a[ids] for h, a in load_arrays(prior/f'fold-{fold:02d}'/'learned_raw-evaluation.npz').items()}
                raw[part] = dict(lightgbm=tree, transformer=temporal, kronos=native)
                for name, pred in raw[part].items():
                    np.savez_compressed(dest/f'{name}-raw-{part}.npz', **pred)
            common = {h: np.logical_and.reduce([np.isfinite(pred[h]) for pred in raw['calibration'].values()]) for h in HEADS}
            np.savez_compressed(dest/'common-calibration-forecast-mask.npz', **common)
            calibrations, forecasts = {}, {}
            targets = plan_targets(labels['calibration'])
            for name, pred in raw['calibration'].items():
                masked = {h: np.where(common[h], pred[h], np.nan) for h in HEADS}
                state = calibrate_sparse(masked, targets, parts['calibration'].date, protocol['min_calibration_dates'])
                calibrations[name] = state
                forecasts[name] = apply_sparse_calibration(raw['evaluation'][name], state)
                write_json(dest/f'{name}-calibration.json', state)
                np.savez_compressed(dest/f'{name}-evaluation.npz', **forecasts[name])
                daily, chosen = decision_metrics(parts['evaluation'], forecasts[name], labels['evaluation'], f'fold-{fold:02d}', name)
                decisions.append(daily)
                chosen.to_parquet(dest/f'{name}-chosen.parquet', index=False)
            for h in HEADS:
                for j in range(12):
                    counts = {(s[h][j]['known_rows'], s[h][j]['known_dates']) for s in calibrations.values()}
                    if len(counts) != 1:
                        raise ValueError('Calibration support differs across models')
            report = matched_metrics(parts['evaluation'], plan_targets(labels['evaluation']), forecasts, fold, 'common_calibration')
            report.to_csv(dest/'head-metrics.csv', index=False)
            metrics.append(report)
            write_json(dest/'completed.json', dict(status='completed', files={str(p.relative_to(dest)):file_hash(p) for p in dest.rglob('*') if p.is_file()}))
            print('Completed common calibration', fold, flush=True)
        combined = pd.concat(metrics, ignore_index=True)
        combined.to_csv(out/'head-metrics.csv', index=False)
        pd.concat(decisions, ignore_index=True).to_csv(out/'decision-metrics.csv', index=False)
        comparisons = {}
        for candidate, baseline in [('kronos', 'lightgbm'), ('transformer', 'lightgbm'), ('kronos', 'transformer')]:
            pair = combined.loc[combined.model.isin([candidate, baseline])].rename(columns={'common_known': 'known_rows'}).replace(
                {'model': {candidate: 'learned', baseline: 'empirical'}})
            comparisons[candidate+'_vs_'+baseline] = compare_heads(pair)
        write_json(out/'assessment.json', dict(comparisons=comparisons, executable=False, quality_promotion=False,
            scope='Two post-paper-cutoff development windows and fixed identities; same calibration support; checkpoint cutoff still not independently certified'))
        write_json(marker, dict(status='completed', finished_at=utc_now(), files={str(p.relative_to(out)):file_hash(p)
            for p in out.rglob('*') if p.is_file() and p.name != 'run-status.json' and '__pycache__' not in p.parts}))
    except BaseException as exc:
        write_json(marker, dict(status='failed', error=repr(exc), updated_at=utc_now()))
        raise


def wait_run(args):
    out = args.output.resolve()
    root = Path(read(out/'experiment.json')['kronos'])
    finalizer = args.finalizer.resolve()
    status = read(finalizer/'status.json')
    pid = read(finalizer/'launch.json')['pid']
    expected = None
    if status['status'] != 'completed':
        expected = subprocess.check_output(['ps', '-p', str(pid), '-o', 'command='], text=True).strip()
        if 'finalize_plan_kronos.py' not in expected or root.name not in expected:
            raise ValueError('Finalizer PID mismatch')
    marker = out.parent/(out.name+'-queue.json')
    if marker.exists():
        raise FileExistsError('Preserve queue attempt')
    write_json(marker, dict(status='waiting_for_live_finalizer', pid=os.getpid(), observed_pid=pid, updated_at=utc_now()))
    try:
        while read(finalizer/'status.json')['status'] != 'completed':
            state = read(finalizer/'status.json')
            if state['status'] == 'failed':
                raise RuntimeError('Native finalization failed')
            result = subprocess.run(['ps', '-p', str(pid), '-o', 'command='], text=True, capture_output=True)
            if result.returncode or result.stdout.strip() != expected:
                if read(finalizer/'status.json')['status'] == 'completed':
                    break
                raise RuntimeError('Finalizer disappeared')
            write_json(marker, dict(status='waiting_for_live_finalizer', pid=os.getpid(), observed_pid=pid, updated_at=utc_now()))
            time.sleep(10)
        write_json(marker, dict(status='running_common_calibration', pid=os.getpid(), updated_at=utc_now()))
        subprocess.run([sys.executable, str(out/'code/scripts/common_plan_calibration.py'), 'run', '--output', str(out)], check=True)
        write_json(marker, dict(status='completed_pending_independent_audit', updated_at=utc_now()))
    except BaseException as exc:
        write_json(marker, dict(status='failed', error=repr(exc), updated_at=utc_now()))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['freeze', 'run', 'wait-run'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--kronos', type=Path)
    p.add_argument('--finalizer', type=Path)
    args = p.parse_args()
    {'freeze': freeze, 'run': run, 'wait-run': wait_run}[args.stage](args)


if __name__ == '__main__':
    main()
