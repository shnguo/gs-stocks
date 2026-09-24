"""Add verified Kronos-base to frozen five-day range diagnostics on identical rows."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from compare_token_kronos_paths import ART, KR, ROOTS, legal, read, sha
from compare_token_range import TARGETS, block_interval, extrema, order_class, score

BASE = ART / 'kronos-base-paths-20260914-v2'
PRIOR = ART / 'token-range-comparison-20260914-v1'
JOINT = ART / 'token-kronos-path-comparison-20260914-v1'
OUT = ART / 'kronos-base-range-comparison-20260914-v1'


def verify_manifest(root, manifest):
    for name, expected in manifest.items():
        assert sha(root/name)==expected, name


def main():
    OUT.mkdir()
    protocol=dict(targets=TARGETS,thresholds=[.02,.05],units='Signal-close percentage points',
        weighting='Date-equal means',min_legal_paths=8,complete_future_required=True,
        cohort='Previous 1349 fully-known common rows intersect base availability; preserve exclusions',
        baselines='Identical frozen historical-volatility and persistence paths',
        comparisons='Base minus small, self-built 3.72M and historical baseline; MAE and CRPS on same rows',
        sampling='Same 60-day inputs, 32 paths, T1 top_p0.9 top_k0; base and small use frozen native upstream',
        sealed_holdout_accessed=False,scope='Post-hoc single development window; no tuning')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    assert read(BASE/'verification.json')['passed']
    verify_manifest(BASE/'forecast',read(BASE/'forecast/completed.json')['files'])
    verify_manifest(PRIOR,read(PRIOR/'completed.json')['files'])
    for path, expected in read(PRIOR/'sources.json').items():
        assert sha(Path(path))==expected,path
    rows=pd.read_parquet(PRIOR/'rows.parquet')
    b_rows=pd.read_parquet(BASE/'rows.parquet').assign(base_index=lambda x:np.arange(len(x)))
    rows=rows.merge(b_rows,on=['date','instrument_id'],validate='one_to_one',how='left')
    assert rows.base_index.notna().all()
    with np.load(BASE/'forecast/paths.npz') as z:
        base_all=z['paths']
        mask_all=legal(base_all)
        np.testing.assert_array_equal(mask_all,z['valid_paths'])
    available=mask_all[rows.base_index.to_numpy()].sum(1)>=8
    rows.assign(base_available=available).to_parquet(OUT/'availability.parquet',index=False)
    rows=rows.loc[available].reset_index(drop=True)
    assert len(rows) and rows.date.nunique()==8
    rows.to_parquet(OUT/'rows.parquet',index=False)
    paths={'kronos_base':base_all[rows.base_index.to_numpy()].astype(np.float64)}
    for name,root in ROOTS.items():
        with np.load(root/'forecast/paths.npz') as z:
            paths[name]=z['paths'][rows[name].to_numpy()].astype(np.float64)
    kp=np.full((1534,32,5,6),np.nan)
    for file in sorted((KR/'fold-15').glob('evaluation-paths-*.npz')):
        with np.load(file) as z:
            kp[z['row_indices']]=z['paths']
    paths['kronos_small']=kp[rows.kronos_small.to_numpy()]
    original=pd.read_parquet(PRIOR/'rows.parquet')[['date','instrument_id']]
    original['prior_index']=np.arange(len(original))
    ix=rows.merge(original,on=['date','instrument_id'],validate='one_to_one').prior_index.to_numpy()
    with np.load(PRIOR/'baseline-paths.npz') as z:
        paths['historical_volatility']=z['paths'][ix]
    with np.load(ROOTS['decoder_3720k']/'evaluation.npz') as z:
        indices=rows.data_index.to_numpy()
        actual,ref=z['future'][indices],z['last'][indices,3].astype(np.float64)
        assert z['valid'][indices].all()
        mean,scale,normalized=z['mean'][indices],z['scale'][indices],z['normalized'][indices,:60]
    with np.load(BASE/'inputs.npz') as z:
        np.testing.assert_array_equal(z['mean'][rows.base_index.to_numpy()],mean)
        np.testing.assert_array_equal(z['scale'][rows.base_index.to_numpy()],scale)
        np.testing.assert_array_equal(z['x'][rows.base_index.to_numpy()],normalized)
    paths['persistence']=np.broadcast_to(ref[:,None,None,None],(len(ref),32,5,6)).copy()
    masks={name:legal(p) for name,p in paths.items()}
    assert all((v.sum(1)>=8).all() for v in masks.values())
    expected,actual_order=extrema(actual,ref),order_class(actual)
    metrics,touch,order,quantiles=[],[],[],[]
    for name,p in paths.items():
        for i in range(len(rows)):
            ps=p[i,masks[name][i]]
            v=extrema(ps,ref[i])
            keys=dict(model=name,date=rows.iloc[i].date,instrument_id=rows.iloc[i].instrument_id)
            for j,target in enumerate(TARGETS):
                metrics.append(dict(**keys,target=target,**score(v[:,j],expected[i,j])))
                q=np.quantile(v[:,j],[.1,.5,.9])
                quantiles.append(dict(**keys,target=target,actual=expected[i,j],q10=q[0],q50=q[1],q90=q[2]))
            for threshold in [.02,.05]:
                for direction,j in [('up',0),('down',1)]:
                    y=expected[i,j]>=threshold*100 if j==0 else expected[i,j]<=-threshold*100
                    hit=v[:,j]>=threshold*100 if j==0 else v[:,j]<=-threshold*100
                    pr=float(hit.mean())
                    touch.append(dict(**keys,direction=direction,threshold=threshold,
                                      brier=(pr-float(y))**2,probability=pr,actual=float(y)))
            probs=np.bincount(order_class(ps),minlength=3)/len(ps)
            truth=np.eye(3)[actual_order[i]]
            order.append(dict(**keys,brier=float(np.sum((probs-truth)**2)),
                accuracy=float(np.argmax(probs)==actual_order[i]),predicted_low_first=probs[0],
                predicted_high_first=probs[1],predicted_ambiguous=probs[2],
                actual_ambiguous=float(actual_order[i]==2)))
    frame=pd.DataFrame(metrics)
    frame.to_parquet(OUT/'row-metrics.parquet',index=False)
    pd.DataFrame(quantiles).to_parquet(OUT/'row-predictions.parquet',index=False)
    # Non-base results must exactly reproduce previously saved row-level scores.
    old=pd.read_parquet(PRIOR/'row-metrics.parquet')
    matched=frame[frame.model.ne('kronos_base')].merge(old,on=['model','date','instrument_id','target'],
                                                      suffixes=('_new','_old'),validate='one_to_one')
    assert len(matched)==len(rows)*5*3
    fields=['mae','crps','coverage80','width80','interval_score80','bias']
    for field in fields:
        np.testing.assert_allclose(matched[field+'_new'],matched[field+'_old'],atol=1e-10,rtol=1e-10)
    daily=frame.groupby(['model','date','target'])[fields].mean().reset_index()
    daily.to_csv(OUT/'daily-metrics.csv',index=False)
    summary=daily.groupby(['model','target'])[fields].mean().reset_index()
    summary.to_csv(OUT/'metrics.csv',index=False)
    td=pd.DataFrame(touch).groupby(['model','date','direction','threshold'])[['brier','probability','actual']].mean().reset_index()
    td.to_csv(OUT/'daily-touch.csv',index=False)
    td.groupby(['model','direction','threshold'])[['brier','probability','actual']].mean().to_csv(OUT/'touch.csv')
    od=pd.DataFrame(order).groupby(['model','date']).mean(numeric_only=True).reset_index()
    od.to_csv(OUT/'daily-order.csv',index=False)
    od.groupby('model').mean(numeric_only=True).to_csv(OUT/'order.csv')
    comparisons=[]
    for target in TARGETS:
        for metric in ['mae','crps']:
            p=daily[daily.target==target].pivot(index='date',columns='model',values=metric)
            for against in ['kronos_small','decoder_3720k','historical_volatility']:
                comparisons.append(dict(target=target,metric=metric,model='kronos_base',against=against,
                    relative_difference=float(p.kronos_base.mean()/p[against].mean()-1),
                    **block_interval((p.kronos_base-p[against]).to_numpy())))
    result=dict(model='NeoQuasar/Kronos-base',parameter_count=102310592,
        original_rows=1510,original_complete_common_rows=len(available),common_scored_rows=len(rows),
        lost_to_base_unavailability=int((~available).sum()),dates=sorted(rows.date.unique()),
        base_coverage=dict(legal_paths=int(mask_all.sum()),total_paths=int(mask_all.size),
                           rows_with_eight_legal_paths=int((mask_all.sum(1)>=8).sum())),
        base_on_original_complete_rows=dict(available_rows=int(available.sum()),total_rows=len(available)),
        metrics=summary.to_dict('records'),comparisons=comparisons,
        verification=dict(passed=True,nonbase_reproduced_row_targets=len(matched),
                          identical_normalized_histories=True,dual_crps_formulas=True),
        limitations=['Single previously researched development window, eight dates and post-hoc comparison',
                     'No final holdout; no trading profit validation',
                     'Base/small pretrained weights versus locally trained scratch predictors',
                     'Pretraining time manifest not independently verified',
                     'Common-valid paths and complete-label selection remain conditional',
                     'Different frozen random draws; 32 paths each; no seed sweep'])
    (OUT/'summary.json').write_text(json.dumps(result,indent=2))
    sources={str(Path(__file__)):sha(Path(__file__)),str(PRIOR/'completed.json'):sha(PRIOR/'completed.json'),
             str(BASE/'verification.json'):sha(BASE/'verification.json'),
             str(BASE/'forecast/completed.json'):sha(BASE/'forecast/completed.json')}
    (OUT/'sources.json').write_text(json.dumps(sources,indent=2))
    (OUT/'completed.json').write_text(json.dumps(dict(passed=True,
        files={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}),indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
