"""Post-hoc common-row diagnostics of previously frozen five-day paths."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
ART = BASE / 'artifacts'
OUT = ART / 'token-kronos-path-comparison-20260914-v1'
ROOTS = {
    'decoder_3720k': ART / 'token-capacity-3720k-20260914-v1',
    'decoder_282k': ART / 'token-capacity-282k-20260914-v1',
}
KR = ART / 'plan-kronos-20260914-v2'
SOURCES = {}


def read(p):
    return json.loads(p.read_text())


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def checked(p, expected=None):
    actual = sha(p)
    if expected is not None:
        assert actual == expected, str(p)
    SOURCES[str(p)] = actual


def legal(x):
    return (np.isfinite(x).all((2, 3)) & (x[..., :4] > 0).all((2, 3))
            & (x[..., 4:] >= 0).all((2, 3))
            & (x[..., 1] >= x[..., :4].max(-1)).all(2)
            & (x[..., 2] <= x[..., :4].min(-1)).all(2))


def main():
    OUT.mkdir()
    rows, paths = {}, {}
    for name, root in ROOTS.items():
        assert read(root / 'verification.json')['passed']
        checked(root / 'verification.json')
        manifest = read(root / 'forecast/completed.json')['files']
        for file in ['rows.parquet', 'paths.npz']:
            checked(root / 'forecast' / file, manifest[file])
        rows[name] = pd.read_parquet(root / 'forecast/rows.parquet')
        with np.load(root / 'forecast/paths.npz') as z:
            paths[name] = z['paths']
            np.testing.assert_array_equal(legal(paths[name]), z['valid_paths'])
    assert read(KR / 'verification.json')['passed']
    checked(KR / 'verification.json')
    checked(KR / 'fold-15-evaluation-rows.parquet',
            read(KR / 'frozen-manifest.json')['fold-15-evaluation-rows.parquet'])
    rows['kronos_small'] = pd.read_parquet(KR / 'fold-15-evaluation-rows.parquet')
    paths['kronos_small'] = np.full((len(rows['kronos_small']), 32, 5, 6), np.nan, np.float32)
    occupied = set()
    km = read(KR / 'fold-15/completed.json')['files']
    for p in sorted((KR / 'fold-15').glob('evaluation-paths-*.npz')):
        checked(p, km[p.name])
        with np.load(p) as z:
            ids = z['row_indices']
            assert not occupied.intersection(ids.tolist())
            occupied.update(ids.tolist())
            paths['kronos_small'][ids] = z['paths']
    keys = ['date', 'instrument_id']
    joint = None
    for name, r in rows.items():
        assert not r.duplicated(keys).any()
        f = r[keys].assign(**{name: np.arange(len(r))})
        joint = f if joint is None else joint.merge(f, on=keys, validate='one_to_one')
    joint = joint.sort_values(keys).reset_index(drop=True)
    data_root = ROOTS['decoder_3720k']
    checked(data_root / 'evaluation.npz', read(data_root / 'prepared.json')['files']['evaluation.npz'])
    full_rows = pd.read_parquet(data_root / 'evaluation-rows.parquet')
    assert not full_rows.duplicated(keys).any()
    joint = joint.merge(full_rows[keys].assign(data_index=np.arange(len(full_rows))), on=keys, validate='one_to_one')
    with np.load(data_root / 'evaluation.npz') as z:
        data = {k: z[k][joint.data_index.to_numpy()] for k in ['future', 'valid', 'last', 'scale', 'mean']}
    # Verify both historical panels and normalization for every matched input.
    te, ke = read(data_root / 'experiment.json'), read(KR / 'experiment.json')
    histories = []
    for folder, expected in [(Path(te['inputs']), te['input_manifest_sha256']),
                             (Path(ke['input_panel']), ke['input_manifest_sha256'])]:
        checked(folder / 'manifest.json', expected)
        m = read(folder / 'manifest.json')
        checked(folder / 'values.npy', m['values_sha256'])
        sm, dm = {s: i for i, s in enumerate(m['instruments'])}, {d: i for i, d in enumerate(m['dates'])}
        raw = np.load(folder / 'values.npy', mmap_mode='r')
        h = np.stack([raw[sm[r.instrument_id], dm[r.date]-59:dm[r.date]+1] for r in joint.itertuples()])
        histories.append(h)
    np.testing.assert_array_equal(*histories)
    x = histories[0][..., :6].astype(np.float32, copy=True)
    x[..., :4] *= histories[0][..., 6:7] / histories[0][:, -1:, 6:7]
    np.testing.assert_array_equal(x.mean(1, keepdims=True), data['mean'])
    np.testing.assert_array_equal(x.std(1, keepdims=True)+1e-5, data['scale'])
    # Ensure all models refer to exactly the same future label dates and source labels.
    for name, r in rows.items():
        aligned = r.iloc[joint[name].to_numpy()]
        assert aligned.label_end.tolist() == full_rows.iloc[joint.data_index.to_numpy()].label_end.tolist()
    parent = Path(te['prior']) / 'fold-15/evaluation.npz'
    checked(parent, te['source_files']['fold-15/evaluation.npz'])
    with np.load(parent) as z:
        ix = rows['kronos_small'].iloc[joint.kronos_small.to_numpy()].parent_row_index.to_numpy()
        actual = np.concatenate([z['future'][ix], z['turnover'][ix]], -1)
        np.testing.assert_array_equal(actual, data['future'])
    paths = {name: p[joint[name].to_numpy()] for name, p in paths.items()}
    masks = {name: legal(p) for name, p in paths.items()}
    common = np.logical_and.reduce([v.sum(1) >= 8 for v in masks.values()])
    joint['all_models_eight_legal_paths'] = common
    joint.to_parquet(OUT / 'matched-rows.parquet', index=False)
    records = []
    for name, p in paths.items():
        for i in np.flatnonzero(common):
            sample = p[i, masks[name][i]].astype(np.float64)
            for day in np.flatnonzero(data['valid'][i]):
                for field, fname in enumerate(['open', 'high', 'low', 'close', 'volume', 'amount']):
                    v = sample[:, day, field]
                    y, scale = float(data['future'][i, day, field]), float(data['scale'][i, 0, field])
                    q = np.quantile(v, [.1, .5, .9])
                    # Pairwise CRPS verified against independent sorted formula.
                    crps = np.abs(v-y).mean() - np.abs(v[:, None]-v).mean()/2
                    n = len(v)
                    alternative = np.abs(v-y).mean() - np.sum((2*np.arange(1,n+1)-n-1)*np.sort(v))/n**2
                    np.testing.assert_allclose(crps, alternative, rtol=1e-10, atol=1e-6)
                    records.append(dict(model=name, date=joint.iloc[i].date, day=int(day+1), field=fname,
                        mae=abs(q[1]-y)/scale, crps=crps/scale, coverage80=float(q[0] <= y <= q[2]),
                        persistence_mae=abs(float(data['last'][i, field])-y)/scale))
    frame = pd.DataFrame(records)
    daily = frame.groupby(['model', 'date', 'day', 'field']).agg(
        mae=('mae', 'mean'), crps=('crps', 'mean'), coverage80=('coverage80', 'mean'),
        persistence_mae=('persistence_mae', 'mean'), count=('mae', 'size')).reset_index()
    daily.to_csv(OUT / 'daily-metrics.csv', index=False)
    summary = daily.groupby(['model', 'day', 'field']).agg(
        mae=('mae', 'mean'), crps=('crps', 'mean'), coverage80=('coverage80', 'mean'),
        persistence_mae=('persistence_mae', 'mean'), rows=('count', 'sum')).reset_index()
    summary.to_csv(OUT / 'metrics.csv', index=False)
    result = dict(scope='Post-hoc frozen-path common-row comparison; no refit or resampling',
        matched_rows=len(joint), dates=sorted(joint.date.unique()), common_available_rows=int(common.sum()),
        coverage={name: dict(legal_paths=int(v.sum()), total_paths=int(v.size),
                            rows_with_eight_legal_paths=int((v.sum(1)>=8).sum())) for name,v in masks.items()},
        day5_close=summary.loc[(summary.day==5)&(summary.field=='close')].to_dict('records'),
        verification='Source hashes, identical raw histories/normalization/labels, unique row mapping, two CRPS formulas passed',
        limitations=['Single previously researched development window, post-hoc comparison',
                     'Frozen pretrained Kronos-small vs locally trained scratch predictors; not isolated architecture comparison',
                     'Same 60-day input, 32 paths, T=1, top_p=.9, top_k=0; different frozen random draws and implementations',
                     'Metrics conditional on all models having eight legal full paths',
                     'Kronos checkpoint pretraining date manifest unverified',
                     'No full-cohort Kronos token CE or final holdout evaluation performed'])
    (OUT / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    checked(Path(__file__))
    (OUT / 'sources.json').write_text(json.dumps(SOURCES, indent=2))
    (OUT / 'completed.json').write_text(json.dumps({'passed': True, 'files':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()}}, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
