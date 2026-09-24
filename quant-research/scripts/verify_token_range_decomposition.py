"""Independent path formulas, endpoint reconciliation and arithmetic attribution."""
import numpy as np
import pandas as pd
from token_range_decomposition import DATA, FIELDS, PARENT, ROOT, SEEDS, checked, paths_map, read

from quant_research.storage import utc_now, write_json


def legal(p):
    return (np.isfinite(p).all(-1)&(p[...,:4]>0).all(-1)&(p[...,4:]>=0).all(-1)
        &(p[...,1]>=p[...,0])&(p[...,1]>=p[...,3])&(p[...,1]>=p[...,2])
        &(p[...,2]<=p[...,0])&(p[...,2]<=p[...,3])).all(-1)


def equal(a,b,keys):
    pd.testing.assert_frame_equal(a.sort_values(keys).reset_index(drop=True)[b.columns],
        b.sort_values(keys).reset_index(drop=True),check_dtype=False,check_exact=False,rtol=1e-10,atol=1e-10)


def interval(values):
    x=np.asarray(values,float)
    rng=np.random.default_rng(314159)
    starts=rng.integers(len(x),size=(2000,(len(x)+9)//10))
    index=np.stack([(starts+j)%len(x) for j in range(10)],axis=-1).reshape(2000,-1)[:,:len(x)]
    return np.quantile(x[index].mean(1),[.025,.975])


def main():
    checked()
    assert read(ROOT/'run-completed.json')['passed']
    ids=np.load(PARENT/'row-ids.npy')
    original_rows=pd.read_parquet(PARENT/'rows.parquet')
    assert original_rows.date.nunique()==76 and original_rows.label_end.max()<'2025-07-30'<'2025-08-07'
    last=np.load(DATA/'last.npy',mmap_mode='r')
    future=np.load(DATA/'future.npy',mmap_mode='r')
    known=np.load(DATA/'valid.npy',mmap_mode='r')
    paths={k:np.load(v,mmap_mode='r') for k,v in paths_map().items()}
    rows=[]
    counts={}
    for h in [2,5]:
        masks={k:legal(p[:,:,:h]) for k,p in paths.items()}
        common=known[ids,:h].all(1)
        for mask in masks.values():
            common &= mask.sum(1)>=16
        counts[h]=int(common.sum())
        assert counts[h]=={2:2372,5:2253}[h]
        for name,p in paths.items():
            for start in range(0,len(ids),64):
                ix=np.arange(start,min(start+64,len(ids)))
                ix=ix[common[ix]]
                if not len(ix):
                    continue
                ref=last[ids[ix],3].astype(float)
                high=100*(p[ix,:,:h,1].astype(float).max(2)/ref[:,None]-1)
                low=100*(p[ix,:,:h,2].astype(float).min(2)/ref[:,None]-1)
                mh,ml=[np.nanquantile(np.where(masks[name][ix],v,np.nan),.5,axis=1) for v in [high,low]]
                y=future[ids[ix],:h]
                yh=100*(y[:,:,1].max(1)/ref-1)
                yl=100*(y[:,:,2].min(1)/ref-1)
                ce=(mh+ml-yh-yl)/2
                we=(mh-ml-yh+yl)/2
                center=(high+low)/2
                width=high-low
                v=np.stack([center,width],-1)
                truth=np.stack([(yh+yl)/2,yh-yl],-1)
                mask=masks[name][ix]
                v=np.where(mask[...,None],v,0.)
                n=mask.sum(1)[:,None]
                first=np.where(mask[...,None],np.abs(v-truth[:,None]),0.).sum(1)/n
                pairs=mask[:,:,None]&mask[:,None,:]
                distance=np.abs(v[:,:,None]-v[:,None,:])
                crps=first-np.where(pairs[...,None],distance,0.).sum((1,2))/(2*n**2)
                med=np.nanquantile(np.where(mask[...,None],v,np.nan),.5,axis=1)
                for j,i in enumerate(ix):
                    rows.append(dict(variant=name,local_row=int(i),row_id=int(ids[i]),date=original_rows.iloc[i].date,horizon=h,draws=int(mask[j].sum()),
                        high_median=mh[j],low_median=ml[j],actual_high=yh[j],actual_low=yl[j],
                        center=(mh[j]+ml[j])/2,half_width=(mh[j]-ml[j])/2,
                        center_error=ce[j],half_width_error=we[j],center_mae=abs(ce[j]),half_width_mae=abs(we[j]),
                        center_mse=ce[j]**2,half_width_mse=we[j]**2,
                        endpoint_mae=(abs(mh[j]-yh[j])+abs(ml[j]-yl[j]))/2,
                        endpoint_mse=((mh[j]-yh[j])**2+(ml[j]-yl[j])**2)/2,
                        path_center_median=med[j,0],path_range_median=med[j,1],
                        path_center_mae=abs(med[j,0]-truth[j,0]),path_range_mae=abs(med[j,1]-truth[j,1]),
                        path_center_crps=crps[j,0],path_range_crps=crps[j,1],
                        path_center_bias=med[j,0]-truth[j,0],path_range_bias=med[j,1]-truth[j,1]))
    f=pd.DataFrame(rows)
    equal(f,pd.read_parquet(ROOT/'rows.parquet'),['variant','row_id','horizon'])
    np.testing.assert_allclose(f.endpoint_mae,np.maximum(abs(f.center_error),abs(f.half_width_error)),atol=1e-12)
    np.testing.assert_allclose(f.endpoint_mse,f.center_mse+f.half_width_mse,atol=1e-10)
    previous=pd.read_parquet(PARENT/'results/common.parquet')
    for name,g in f.groupby('variant'):
        before=previous[previous.variant==(name+'_raw' if name.startswith('ours_') else name)]
        endpoints=before[before.target.isin(['maximum','minimum'])].groupby(['row_id','horizon']).mae.mean().sort_index()
        g=g.set_index(['row_id','horizon']).sort_index()
        np.testing.assert_allclose(g.endpoint_mae,endpoints,atol=1e-10,rtol=1e-10)
        span=before[before.target=='range'].set_index(['row_id','horizon']).sort_index()
        np.testing.assert_allclose(g.path_range_mae,span.mae,atol=1e-10,rtol=1e-10)
        np.testing.assert_allclose(g.path_range_crps,span.crps,atol=1e-10,rtol=1e-10)
    daily=[]
    for (name,h),g in f.groupby(['variant','horizon']):
        dates,codes=np.unique(g.date,return_inverse=True)
        n=np.bincount(codes)
        values={k:np.bincount(codes,weights=g[k])/n for k in FIELDS}
        for i,date in enumerate(dates):
            daily.append(dict(variant=name,horizon=h,date=date,**{k:v[i] for k,v in values.items()}))
    d=pd.DataFrame(daily)
    mean=d[d.variant.str.startswith('ours_')].groupby(['date','horizon'])[FIELDS].mean().reset_index().assign(variant='ours_mean')
    d=pd.concat([d,mean],ignore_index=True)
    equal(d,pd.read_csv(ROOT/'daily.csv'),['variant','date','horizon'])
    equal(d.groupby(['variant','horizon'])[FIELDS].mean().reset_index(),pd.read_csv(ROOT/'summary.csv'),['variant','horizon'])
    attrs=[]
    for against in ['historical','kronos']:
        b=f[f.variant==against].set_index(['row_id','horizon']).sort_index()
        for seed in SEEDS:
            x=f[f.variant==f'ours_{seed}'].set_index(['row_id','horizon']).sort_index()
            def loss(c,w):
                return (abs(c+w-x.actual_high)+abs(c-w-x.actual_low))/2
            before=loss(b.center,b.half_width)
            after=loss(x.center,x.half_width)
            center=0.5*((loss(x.center,b.half_width)-before)+(after-loss(b.center,x.half_width)))
            width=0.5*((loss(b.center,x.half_width)-before)+(after-loss(x.center,b.half_width)))
            np.testing.assert_allclose(center+width,x.endpoint_mae-b.endpoint_mae,atol=1e-12)
            attrs.append(x[['date']].reset_index().assign(against=against,seed=str(seed),center=center.to_numpy(),half_width=width.to_numpy(),total=(after-before).to_numpy()))
    a=pd.concat(attrs,ignore_index=True)
    equal(a,pd.read_parquet(ROOT/'attribution-rows.parquet'),['against','seed','row_id','horizon'])
    ad=a.groupby(['against','seed','date','horizon'])[['center','half_width','total']].mean().reset_index()
    mean=ad.groupby(['against','date','horizon'])[['center','half_width','total']].mean().reset_index().assign(seed='mean')
    ad=pd.concat([ad,mean],ignore_index=True)
    equal(ad,pd.read_csv(ROOT/'attribution-daily.csv',dtype={'seed':str}),['against','seed','date','horizon'])
    p=pd.read_csv(ROOT/'paired.csv',dtype={'seed':str})
    for r in p.itertuples():
        sub=d[d.horizon==r.horizon].pivot(index='date',columns='variant',values=r.metric).sort_index()
        x,b=sub[f'ours_{r.seed}'],sub[r.against]
        low,high=interval(x-b)
        np.testing.assert_allclose([r.baseline,r.candidate,r.delta,r.ci_low,r.ci_high],[b.mean(),x.mean(),(x-b).mean(),low,high],atol=1e-10,rtol=1e-10)
        if np.isfinite(r.relative_change):
            np.testing.assert_allclose(r.relative_change,x.mean()/b.mean()-1,atol=1e-10)
        assert r.dates==76
    ap=pd.read_csv(ROOT/'attribution-summary.csv',dtype={'seed':str})
    for r in ap.itertuples():
        v=ad[(ad.against==r.against)&(ad.seed==r.seed)&(ad.horizon==r.horizon)].sort_values('date')[r.component]
        low,high=interval(v)
        np.testing.assert_allclose([r.delta,r.ci_low,r.ci_high],[v.mean(),low,high],atol=1e-10,rtol=1e-10)
    assert len(p)==224 and len(ap)==48
    write_json(ROOT/'independent-verification.json',dict(passed=True,at=utc_now(),row_components=len(f),
        paired_estimates=len(p),attribution_estimates=len(ap),attribution_rows=len(a),cohort_counts=counts,
        endpoint_mae_matches_prior=True,direct_range_scores_match_prior=True,all_identities_verified=True,
        pairwise_crps_verified=True,source_hashes_match=True,no_new_inference=True,sealed_holdout_opened=False))
    print('Independent midpoint/width diagnostic verification passed',flush=True)


if __name__=='__main__':
    main()
