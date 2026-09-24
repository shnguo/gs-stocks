"""Independent raw-path extrema and empirical-score recomputation."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE=Path('/Users/guo/Documents/stocks/quant-research')
DATA=BASE/'artifacts/token-history-data-20260915-v1'
ROOT=BASE/'artifacts/token-history-experiment-20260915-v1'
NAMES=['dense_282k_equal','dense_3720k_equal','dense_3720k_weighted']


def legal_prefix(p,h):
    q=p[:,:,:h]
    bars=(np.isfinite(q).all(-1)&(q[...,:4]>0).all(-1)&(q[...,4:]>=0).all(-1)
          &(q[...,1]>=np.maximum(q[...,0],q[...,3]))
          &(q[...,2]<=np.minimum(q[...,0],q[...,3])))
    return bars.all(-1)


def targets(p,ref):
    high=p[...,1].max(-1).astype(np.float64)
    low=p[...,2].min(-1).astype(np.float64)
    if p.ndim==4:
        ref=ref[:,None]
    return np.stack([100*(high/ref-1),100*(low/ref-1),100*(high-low)/ref],-1)


def recompute(fold):
    rows=pd.read_parquet(DATA/'rows.parquet')
    reference=np.load(DATA/'last.npy',mmap_mode='r')
    truth=np.load(DATA/'future.npy',mmap_mode='r')
    known=np.load(DATA/'valid.npy',mmap_mode='r')
    ids=None
    paths={}
    for name in NAMES:
        root=ROOT/fold/name/'forecast'
        assert json.loads((root/'completed.json').read_text())['passed']
        ii=np.load(root/'row-ids.npy')
        if ids is None:
            ids=ii
        np.testing.assert_array_equal(ii,ids)
        paths[name]=np.load(root/'paths.npy',mmap_mode='r')
    output=[]
    for h in [2,5]:
        masks={k:legal_prefix(p,h) for k,p in paths.items()}
        good=known[ids,:h].all(1)&np.logical_and.reduce([m.sum(1)>=8 for m in masks.values()])
        indices=np.flatnonzero(good)
        _,date_index=np.unique(rows.iloc[ids[good]].date.to_numpy(),return_inverse=True)
        day_counts=np.bincount(date_index)
        actual=targets(truth[ids[good],:h],reference[ids[good],3].astype(float))
        for name,p in paths.items():
            scores,errors=[],[]
            for start in range(0,len(indices),128):
                idx=indices[start:start+128]
                mask=masks[name][idx]
                count=mask.sum(1)
                values=targets(p[idx,:,:h],reference[ids[idx],3].astype(float))
                values=np.where(mask[...,None],values,0.)
                y=actual[start:start+len(idx)]
                first=np.where(mask[...,None],np.abs(values-y[:,None]),0).sum(1)/count[:,None]
                distance=np.abs(values[:,:,None]-values[:,None,:])
                paired=mask[:,:,None]&mask[:,None,:]
                second=np.where(paired[...,None],distance,0).sum((1,2))/(2*count[:,None]**2)
                scores.append(first-second)
                median=np.nanmedian(np.where(mask[...,None],values,np.nan),axis=1)
                errors.append(np.abs(median-y))
            crps,mae=np.concatenate(scores),np.concatenate(errors)
            for j,target in enumerate(['maximum','minimum','range']):
                c=np.bincount(date_index,weights=crps[:,j])/day_counts
                e=np.bincount(date_index,weights=mae[:,j])/day_counts
                output.append(dict(fold=fold,horizon=h,model=name,target=target,
                    common_rows=len(indices),dates=len(day_counts),crps=float(c.mean()),mae=float(e.mean())))
    result=pd.DataFrame(output)
    result.to_csv(ROOT/f'{fold}-independent-path-metrics.csv',index=False)
    print(result.groupby(['horizon','model'])[['crps','mae']].mean().to_string(),flush=True)


def compare():
    other=pd.concat([pd.read_csv(ROOT/f'{f}-independent-path-metrics.csv') for f in ['2024h2','2025q2']])
    expected=pd.read_csv(ROOT/'weighted-evaluation/common-metrics.csv')
    keys=['fold','horizon','model','target']
    expected=expected.loc[expected.model.isin(NAMES)].sort_values(keys).reset_index(drop=True)
    other=other.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(other[keys],expected[keys])
    for field in ['crps','mae']:
        np.testing.assert_allclose(other[field],expected[field],rtol=1e-10,atol=1e-10)
    (ROOT/'independent-path-verification.json').write_text(json.dumps(dict(passed=True,
        comparisons=len(other),independent_masks=True,independent_extrema=True,
        independent_pairwise_crps=True,independent_date_aggregation=True),indent=2))
    print('All independent path metrics agree with the frozen evaluator',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['2024h2','2025q2','compare'])
    args=parser.parse_args()
    if args.action=='compare':
        compare()
    else:
        recompute(args.action)
