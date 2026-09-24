"""Audit one completed historical-as-of fit and its immutable data receipts."""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from quant_research.plan_value import HEADS, PROBABILITIES
from quant_research.storage import file_hash, utc_now, write_json


def read(p):
    return json.loads(p.read_text())


def check(root,manifest):
    for name,expected in manifest.items():
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()) or file_hash(p)!=expected:
            raise ValueError(f'Changed evidence: {p}')
    return len(manifest)


def audit(out,window,asof):
    protocol,schedule=[read(out/n) for n in ['protocol.json','schedule.json']]
    dest=out/window/'fits'/asof
    complete=read(dest/'raw-completed.json')
    assert complete['status']=='completed'
    counts=dict(hashes=check(out,read(out/'frozen-manifest.json'))+check(dest,complete['files']),
        cache_inputs=0,cache_labels=0,restored_values=0,best_at_cap=0)
    cfg,receipts=[read(dest/n) for n in ['config.json','data-receipts.json']]
    spec=schedule[window]['fits'][asof]
    assert cfg['schedule']==spec and cfg['asof']==asof
    assert cfg['frozen_experiment_sha256']==file_hash(out/'frozen-manifest.json')
    days=spec['train']+spec['selection']
    expected={f'{day}/{kind}-manifest.json' for day in days for kind in ['input','label']}
    expected.add(f'{asof}/input-manifest.json')
    assert set(receipts['cache_files'])==expected
    assert receipts['asof']==asof and receipts['input_only_prediction_date']==asof
    cache=out/'day-cache'
    counts['hashes']+=check(cache,receipts['cache_files'])
    for day in days+[asof]:
        folder=cache/day
        meta=read(folder/'input-manifest.json')
        assert meta['date']==day and meta['source_read_dates']==[day] and day<=asof
        counts['hashes']+=check(folder,meta['files'])
        counts['cache_inputs']+=1
        if day in days:
            label=read(folder/'label-manifest.json')
            assert label['date']==day and label['label_end']<=asof
            assert label['label_end']<=label['first_requested_asof']
            assert min(label['source_read_dates'])==day and max(label['source_read_dates'])==label['label_end']
            assert label['input_manifest_sha256']==file_hash(folder/'input-manifest.json')
            if day in spec['train']:
                assert label['label_end']<spec['selection'][0]
            counts['hashes']+=check(folder,label['files'])
            counts['cache_labels']+=1
    rows=pd.read_parquet(dest/'prediction-rows.parquet')
    pd.testing.assert_frame_equal(rows,pd.read_parquet(cache/asof/'rows.parquet'))
    with np.load(cache/asof/'inputs.npz') as z:
        x=z['x']
    with np.load(dest/'raw.npz') as z:
        prediction={h:z[h] for h in HEADS}
    ids=np.unique(np.linspace(0,len(rows)-1,24,dtype=int))
    models=read(dest/'models.json')
    for h in HEADS:
        assert prediction[h].shape==(len(rows),12) and np.isfinite(prediction[h]).all()
        for j,m in enumerate(models['heads'][h]):
            if m['kind']=='constant':
                restored=np.full(len(ids),m['value'])
            else:
                booster=lgb.Booster(model_file=str(dest/'models'/m['file']))
                restored=booster.predict(x[ids],num_threads=1)
                assert 1<=m['best_iteration']<=protocol['training']['iterations']
                counts['best_at_cap']+=m['best_iteration']==protocol['training']['iterations']
            if h in PROBABILITIES:
                restored=np.clip(restored,0,1)
            elif h==HEADS[4]:
                restored=np.maximum(restored,0)
            np.testing.assert_allclose(restored,prediction[h][ids,j],rtol=1e-12,atol=1e-12)
            counts['restored_values']+=len(ids)
    return counts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--window',required=True)
    p.add_argument('--asof',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError('Preserve earlier audit')
    count=audit(a.experiment.resolve(),a.window,a.asof)
    write_json(a.output,dict(passed=True,checked_at=utc_now(),window=a.window,asof=a.asof,counts=count,
        scope='One completed as-of fit: frozen protocol, all input/label receipt files and dates, all 60 saved heads on 24 source rows; not whole-experiment or profitability proof'))
    print(count,flush=True)


if __name__=='__main__':
    main()
