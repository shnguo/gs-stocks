"""Independent reconciliation of Kronos-base extrema and paired summaries."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ART=Path(__file__).resolve().parents[1]/'artifacts'
BASE=ART/'kronos-base-paths-20260914-v2'
OUT=ART/'kronos-base-range-comparison-20260914-v1'


def main():
    complete=json.loads((OUT/'completed.json').read_text())
    for name,expected in complete['files'].items():
        with (OUT/name).open('rb') as f:
            assert hashlib.file_digest(f,'sha256').hexdigest()==expected,name
    rows=pd.read_parquet(OUT/'rows.parquet')
    preds=pd.read_parquet(OUT/'row-predictions.parquet')
    metrics=pd.read_parquet(OUT/'row-metrics.parquet')
    joined=preds.merge(metrics,on=['date','instrument_id','model','target'],validate='one_to_one')
    np.testing.assert_allclose(abs(joined.q50-joined.actual),joined.mae)
    np.testing.assert_allclose(joined.q90-joined.q10,joined.width80)
    np.testing.assert_array_equal(((joined.q10<=joined.actual)&(joined.actual<=joined.q90)).astype(float),joined.coverage80)
    np.testing.assert_allclose(joined.width80+10*np.maximum(joined.q10-joined.actual,0)+10*np.maximum(joined.actual-joined.q90,0),joined.interval_score80)
    with np.load(BASE/'forecast/paths.npz') as z:
        paths=z['paths'].astype(np.float64)
        mask=z['valid_paths']
    independently_valid=np.ones(paths.shape[:2],bool)
    for day in range(5):
        bars=paths[:,:,day]
        independently_valid &= np.isfinite(bars).all(-1)
        independently_valid &= (bars[...,:4]>0).all(-1)&(bars[...,4:]>=0).all(-1)
        independently_valid &= (bars[...,1]>=bars[...,0])&(bars[...,1]>=bars[...,3])
        independently_valid &= (bars[...,2]<=bars[...,0])&(bars[...,2]<=bars[...,3])
        independently_valid &= bars[...,1]>=bars[...,2]
    np.testing.assert_array_equal(independently_valid,mask)
    with np.load(ART/'token-capacity-3720k-20260914-v1/evaluation.npz') as z:
        actual=z['future'][rows.data_index.to_numpy()]
        ref=z['last'][rows.data_index.to_numpy(),3].astype(np.float64)
        assert z['valid'][rows.data_index.to_numpy()].all()
    bpred=preds[preds.model.eq('kronos_base')].set_index(['date','instrument_id','target'])
    bmetric=metrics[metrics.model.eq('kronos_base')].set_index(['date','instrument_id','target'])
    for i,row in enumerate(rows.itertuples()):
        b=paths[row.base_index,mask[row.base_index]]
        high=np.array([max(path[:,1]) for path in b])
        low=np.array([min(path[:,2]) for path in b])
        ah=max(float(day[1]) for day in actual[i])
        al=min(float(day[2]) for day in actual[i])
        target_vectors={'maximum':((high-ref[i])/ref[i]*100,(ah-ref[i])/ref[i]*100),
                        'minimum':((low-ref[i])/ref[i]*100,(al-ref[i])/ref[i]*100),
                        'range':((high-low)/ref[i]*100,(ah-al)/ref[i]*100)}
        for target,(values,y) in target_vectors.items():
            recorded=bpred.loc[(row.date,row.instrument_id,target)]
            np.testing.assert_allclose(recorded.actual,y,atol=1e-12)
            np.testing.assert_allclose(recorded[['q10','q50','q90']].to_numpy(float),np.quantile(values,[.1,.5,.9]),atol=1e-12)
            # Absolute distance minus sum over unique unordered sample pairs.
            spread=sum(abs(values[j]-values[k]) for j in range(len(values)) for k in range(j))/len(values)**2
            crps=sum(abs(v-y) for v in values)/len(values)-spread
            np.testing.assert_allclose(bmetric.loc[(row.date,row.instrument_id,target)].crps,crps,atol=1e-10)
    summary=pd.read_csv(OUT/'metrics.csv')
    for row in summary.itertuples():
        f=joined[joined.model.eq(row.model)&joined.target.eq(row.target)]
        date_errors=[np.mean(abs(g.q50-g.actual)) for _,g in f.groupby('date')]
        np.testing.assert_allclose(row.mae,np.mean(date_errors),atol=1e-12)
    # Each comparison must subtract values on the same eight dates.
    daily=pd.read_csv(OUT/'daily-metrics.csv')
    for r in json.loads((OUT/'summary.json').read_text())['comparisons']:
        p=daily[daily.target.eq(r['target'])].pivot(index='date',columns='model',values=r['metric'])
        np.testing.assert_allclose(r['mean'],(p.kronos_base-p[r['against']]).mean(),atol=1e-12)
    result=dict(passed=True,scored_rows=len(rows),row_target_predictions_checked=len(preds),
                native_base_path_values=int(paths.size),base_crps_rows_independently_checked=len(rows)*3,
                source='Completed artifact hashes and independently calculated extrema, masks, quantiles, CRPS, interval scores and date-equal paired differences',
                comparison_completed_sha256=hashlib.sha256((OUT/'completed.json').read_bytes()).hexdigest())
    (ART/'kronos-base-range-independent-check-20260914.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
