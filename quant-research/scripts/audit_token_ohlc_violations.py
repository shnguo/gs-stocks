"""Measure the size of OHLC ordering violations without changing score masks."""
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.storage import file_hash, write_json

BASE=Path('/Users/guo/Documents/stocks/quant-research')
ROOT=BASE/'artifacts/token-pipeline-audit-20260915-v1'
DATA=BASE/'artifacts/token-history-data-20260915-v1'


def main():
    out=ROOT/'ohlc-violations'
    out.mkdir(exist_ok=True)
    known=np.load(DATA/'valid.npy',mmap_mode='r')
    last=np.load(DATA/'last.npy',mmap_mode='r')
    rows=[]
    for f in sorted((ROOT/'reconstruction').glob('*.npz')):
        fold,part=f.stem.split('-')
        with np.load(f) as z:
            ids=z['row_ids']
            ok=known[ids].all(1)
            ref=last[ids[ok],3].astype(float)
            for stage in ['clip_only','tokenizer_clipped']:
                p=z[stage][ok].astype(float)
                violation=np.maximum(np.maximum(p[...,0],p[...,3])-p[...,1],p[...,2]-np.minimum(p[...,0],p[...,3]))
                maximum=np.maximum(violation,0).max(1)
                pct=maximum/ref*100
                bad=maximum>0
                rows.append(dict(fold=fold,partition=part,stage=stage,known_rows=len(p),invalid_rows=int(bad.sum()),
                    above_001pct_signal_price=int((pct>.01).sum()),above_010pct_signal_price=int((pct>.1).sum()),
                    maximum_violation_pp=float(pct.max()),median_violation_pp_on_invalid=float(np.median(pct[bad])) if bad.any() else 0))
    pd.DataFrame(rows).to_csv(out/'violations.csv',index=False)
    write_json(out/'interpretation.json',dict(measure='Largest high-below-open/close or low-above-open/close gap across the five future bars, expressed as percentage points of signal close',
        rounding='No tick rounding or repair. Size thresholds diagnose materiality; they do not change any evaluation mask.',horizon=5))
    write_json(out/'completed.json',dict(passed=True,files={p.name:file_hash(p) for p in out.iterdir() if p.is_file() and p.name!='completed.json'}))


if __name__=='__main__':
    main()
