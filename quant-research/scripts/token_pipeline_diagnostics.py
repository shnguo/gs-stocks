"""Describe representation distortion, including invalid reconstructions, on fixed cohorts."""
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.forecast_audit import TARGETS, extrema
from quant_research.storage import file_hash, write_json
from quant_research.token_transformer import valid_bars

BASE=Path('/Users/guo/Documents/stocks/quant-research')
AUDIT=BASE/'artifacts/token-pipeline-audit-20260915-v1'
DATA=BASE/'artifacts/token-history-data-20260915-v1'
PRIOR=BASE/'artifacts/token-history-experiment-20260915-v1'


def main():
    dest=AUDIT/'distortion-analysis'
    dest.mkdir(exist_ok=True)
    rows=pd.read_parquet(DATA/'rows.parquet')
    a={k:np.load(DATA/f'{k}.npy',mmap_mode='r') for k in ['valid','future','last']}
    records,validity,matched=[],[],[]
    for path in sorted((AUDIT/'reconstruction').glob('*.npz')):
        fold,part=path.stem.split('-')
        with np.load(path) as z:
            ids=z['row_ids']
            if part=='evaluation':
                prior=PRIOR/fold/'dense_3720k_equal/forecast'
                np.testing.assert_array_equal(ids,np.load(prior/'row-ids.npy'))
                predicted=np.load(prior/'paths.npy',mmap_mode='r')
            for h in [2,5]:
                known=a['valid'][ids,:h].all(1)
                reference=a['last'][ids,3].astype(float)
                truth=extrema(a['future'][ids,:h],reference)
                clip=extrema(z['clip_only'][:,:h],reference)
                decoded=extrema(z['tokenizer_clipped'][:,:h],reference)
                clipped=z['price_clipped'][:,:h].any(1)
                for name,mask in [('all',known),('price_clipped',known&clipped),('price_unclipped',known&~clipped)]:
                    for stage,error in [('clipping',clip-truth),('tokenizer_vs_clipped',decoded-clip),('total_reconstruction',decoded-truth)]:
                        for j,target in enumerate(TARGETS):
                            f=pd.DataFrame(dict(date=rows.iloc[ids[mask]].date.to_numpy(),mae=np.abs(error[mask,j]),bias=error[mask,j]))
                            m=f.groupby('date')[['mae','bias']].mean().mean()
                            records.append(dict(fold=fold,partition=part,horizon=h,group=name,stage=stage,target=target,
                                rows=int(mask.sum()),dates=int(f.date.nunique()),mae=float(m.mae),bias=float(m.bias)))
                q=z['tokenizer_clipped'][known,:h]
                order=((q[...,1]<np.maximum(q[...,0],q[...,3]))|(q[...,2]>np.minimum(q[...,0],q[...,3]))).any(1)
                validity.append(dict(fold=fold,partition=part,horizon=h,known_rows=len(q),
                    ohlc_order_invalid=int(order.sum()),negative_turnover=int((q[...,4:]<0).any((1,2)).sum()),
                    nonpositive_price=int((q[...,:4]<=0).any((1,2)).sum()),any_invalid=int((~valid_bars(q).all(1)).sum())))
                if part=='evaluation':
                    legal=valid_bars(predicted[:,:,:h]).all(-1)
                    common=known&(legal.sum(1)>=8)
                    for i in np.flatnonzero(common):
                        med=np.median(extrema(predicted[i,legal[i],:h].astype(float),reference[i]),axis=0)
                        for j,target in enumerate(TARGETS):
                            matched.append(dict(fold=fold,date=rows.iloc[ids[i]].date,horizon=h,target=target,
                                reconstruction_mae=abs(decoded[i,j]-truth[i,j]),forecast_mae=abs(med[j]-truth[i,j])))
    pd.DataFrame(records).to_csv(dest/'component-errors.csv',index=False)
    pd.DataFrame(validity).to_csv(dest/'invalid-reconstruction.csv',index=False)
    matched=pd.DataFrame(matched)
    daily=matched.groupby(['fold','date','horizon','target'])[['reconstruction_mae','forecast_mae']].mean().reset_index()
    daily.groupby(['fold','horizon','target'])[['reconstruction_mae','forecast_mae']].mean().reset_index().to_csv(dest/'matched-forecast-comparison.csv',index=False)
    write_json(dest/'interpretation.json',dict(
        errors='Percentage points of signal-close price; stock means within dates, then date means',
        components='Absolute component errors do not add up; tokenizer_vs_clipped isolates extra round-trip distortion around clipped values',
        invalid_paths='Invalid finite reconstructions remain included in distortion errors; no repair is applied',
        matched_comparison='Reconstruction receives actual future tokens; forecast receives history only. This comparison does not establish an attainable forecast-error floor.',
        executable=False))
    write_json(dest/'completed.json',dict(passed=True,files={p.name:file_hash(p) for p in dest.iterdir() if p.is_file() and p.name!='completed.json'}))


if __name__=='__main__':
    main()
