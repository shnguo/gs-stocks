"""Compare both decoders using exactly the same valid generated token paths."""
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.forecast_audit import SCORES, TARGETS, empirical_score, extrema
from quant_research.storage import file_hash, utc_now, write_json
from quant_research.token_transformer import valid_bars

BASE=Path('/Users/guo/Documents/stocks/quant-research')
ROOT=BASE/'artifacts/tokenizer-reconstruction-20260915-v1'
DATA=BASE/'artifacts/token-history-data-20260915-v1'


def main():
    dest=ROOT/'common-draw-sensitivity'
    dest.mkdir(exist_ok=True)
    rows=pd.read_parquet(DATA/'rows.parquet')
    a={k:np.load(DATA/f'{k}.npy',mmap_mode='r') for k in ['future','valid','last']}
    records=[]
    coverage=[]
    for seed in [17,29,43]:
        folder=ROOT/f'predictors/seed{seed}/dense_3720k_equal/decoder-forecast'
        ids=np.load(folder/'row-ids.npy')
        p={name:np.load(folder/f'{name}-paths.npy',mmap_mode='r') for name in ['frozen','adapted']}
        ref=a['last'][ids,3].astype(float)
        for h in [2,5]:
            legal=valid_bars(p['frozen'][:,:,:h]).all(-1)&valid_bars(p['adapted'][:,:,:h]).all(-1)
            known=a['valid'][ids,:h].all(1)
            good=known&(legal.sum(1)>=16)
            actual=extrema(a['future'][ids,:h].astype(float),ref)
            coverage.append(dict(seed=seed,horizon=h,known=int(known.sum()),usable=int(good.sum()),
                common_legal_paths=int(legal.sum()),total_paths=int(legal.size)))
            for i in np.flatnonzero(good):
                for name in p:
                    values=extrema(p[name][i,legal[i],:h].astype(float),ref[i])
                    scores=empirical_score(values,actual[i])
                    for j,target in enumerate(TARGETS):
                        records.append(dict(seed=seed,variant=name,horizon=h,target=target,date=rows.iloc[ids[i]].date,
                            **{k:float(v[j]) for k,v in scores.items()}))
    frame=pd.DataFrame(records)
    daily=frame.groupby(['seed','variant','date','horizon','target'])[list(SCORES)].mean().reset_index()
    daily.to_csv(dest/'daily.csv',index=False)
    daily.groupby(['seed','variant','horizon','target'])[list(SCORES)].mean().reset_index().to_csv(dest/'metrics.csv',index=False)
    pd.DataFrame(coverage).to_csv(dest/'coverage.csv',index=False)
    write_json(dest/'completed.json',dict(passed=True,at=utc_now(),files={p.name:file_hash(p) for p in dest.iterdir() if p.is_file() and p.name!='completed.json'}))
    print('Common generated-path sensitivity complete',flush=True)


if __name__=='__main__':
    main()
