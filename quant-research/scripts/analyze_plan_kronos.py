"""Summarize paired path-model errors and same-rule net-cost plan diagnostics."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from plan_value_run import decision_metrics

from quant_research.plan_value import compare_heads
from quant_research.storage import file_hash, utc_now, write_json


def read(p):
    return json.loads(p.read_text())


def load(p):
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.experiment.resolve(), args.output.resolve()
    verified = read(root/'verification.json')
    if not verified['passed'] or read(root/'run-status.json')['status'] != 'completed':
        raise ValueError('Analysis requires completed independent verification')
    out.mkdir()
    e, protocol = read(root/'experiment.json'), read(root/'protocol.json')
    prior, tf = Path(e['prior']), Path(e['transformer'])
    write_json(out/'sources.json', dict(experiment=str(root),
        run_status_sha256=file_hash(root/'run-status.json'), verification_sha256=file_hash(root/'verification.json'),
        source_sha256=file_hash(Path(__file__)), created_at=utc_now()))
    metrics = pd.read_csv(root/'matched-head-metrics.csv')
    comparisons = {}
    for version in ['raw', 'calibrated']:
        for baseline in ['lightgbm', 'transformer']:
            pair = metrics.loc[metrics.version.eq(version) & metrics.model.isin(['kronos', baseline])].copy()
            pair = pair.rename(columns={'common_known': 'known_rows'}).replace({'model': {'kronos': 'learned', baseline: 'empirical'}})
            comparisons[f'{version}_versus_{baseline}'] = compare_heads(pair)
    decisions = []
    for fold in protocol['fold_indices']:
        rows = pd.read_parquet(root/f'fold-{fold:02d}-evaluation-rows.parquet')
        ids = rows.parent_row_index.to_numpy()
        labels = {h: a[ids] for h, a in load(prior/f'fold-{fold:02d}'/'evaluation.npz').items()}
        for version in ['raw', 'calibrated']:
            suffix = '_raw' if version == 'raw' else ''
            models = {'kronos': load(root/f'fold-{fold:02d}'/f'{version}-evaluation.npz')}
            for name, base, prefix in [('lightgbm', prior, 'learned'), ('transformer', tf, 'transformer')]:
                models[name] = {h: a[ids] for h, a in load(base/f'fold-{fold:02d}'/f'{prefix}{suffix}-evaluation.npz').items()}
            for name, prediction in models.items():
                daily, chosen = decision_metrics(rows.reset_index(drop=True), prediction, labels, f'fold-{fold:02d}', name)
                daily['version'] = version
                decisions.append(daily)
                chosen.to_parquet(out/f'fold-{fold:02d}-{name}-{version}-chosen.parquet', index=False)
    daily = pd.concat(decisions, ignore_index=True)
    daily.to_csv(out/'decision-metrics.csv', index=False)
    write_json(out/'summary.json', dict(comparisons=comparisons, coverage=verified['coverage'],
        interpretation='Date-equal paired errors on identical available predictions; missing forecast/label coverage is explicit. Decision selections differ; individual known scenario means are not portfolio or unconditional expected returns.',
        diagnostics='Parent frozen plan selection and costs including double-cost stress, all chosen identities retained, unknown outcomes not imputed',
        executable=False, quality_promotion=False))
    write_json(out/'completed.json', dict(status='completed', files={str(p.relative_to(out)): file_hash(p)
        for p in out.rglob('*') if p.is_file()}, updated_at=utc_now()))
    print('Analysis completed', out, flush=True)


if __name__ == '__main__':
    main()
