"""Independent arithmetic, calibration, lineage and replay audit."""
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import verify_token_profile_transfer as old_audit
from token_history_run import Dataset
from token_midpoint_transfer_run import CURRENT, MID, OWNERS, ROOT, SEEDS, checked, read
from tokenizer_reconstruction_run import load_decoder, verify
from verify_token_pipeline_audit import legal

from quant_research.storage import file_hash, utc_now, write_json
from quant_research.token_transformer import decode_paths, generate_tokens, restore_model


def calibration_audit(data,profiles):
    ids=np.load(ROOT/'calibration-ids.npy')
    count=0
    for owner in ['ce','midpoint']:
        output=[]
        for seed in SEEDS:
            directory=ROOT/'calibration/forecasts'/owner/f'seed{seed}'
            verify(directory)
            p=np.load(directory/'paths.npy',mmap_mode='r')
            for h in [2,5]:
                mask=legal(p[:,:,:h])
                use=data.a['valid'][ids,:h].all(1)&(mask.sum(1)>=16)
                for i in np.flatnonzero(use):
                    ref=float(data.a['last'][ids[i],3])
                    draw=p[i,mask[i],:h].astype(float)
                    hi=draw[:,:,1].max(1)
                    lo=draw[:,:,2].min(1)
                    x=np.column_stack([(hi/ref-1)*100,(lo/ref-1)*100,(hi-lo)/ref*100])
                    y=data.a['future'][ids[i],:h]
                    high,low=y[:,1].max(),y[:,2].min()
                    truth=[(high/ref-1)*100,(low/ref-1)*100,(high-low)/ref*100]
                    q=np.quantile(x,[.1,.5,.9],axis=0,method='linear')
                    for j,target in enumerate(['maximum','minimum','range']):
                        output.append(dict(seed=seed,row_id=int(ids[i]),date=data.rows.iloc[ids[i]].date,horizon=h,target=target,
                            lower=q[0,j],median=q[1,j],upper=q[2,j],actual=truth[j],draws=int(mask[i].sum())))
        found=pd.DataFrame(output)
        saved=pd.read_parquet(ROOT/'fits'/owner/'quantiles.parquet')
        order=['seed','horizon','row_id','target']
        pd.testing.assert_frame_equal(found.sort_values(order).reset_index(drop=True)[saved.columns],
            saved.sort_values(order).reset_index(drop=True),check_dtype=False,atol=1e-10,rtol=1e-10)
        fit=read(Path(profiles[owner]['fit_path']))
        assert fit['quantiles_sha256']==file_hash(ROOT/'fits'/owner/'quantiles.parquet')
        assert fit['decoder_sha256']==profiles['decoder']['sha256']
        assert fit['sampling']==profiles['sampling'] and fit['calibration_only'] and not fit['evaluation_used']
        assert fit['row_ids_sha256']==file_hash(ROOT/'calibration-ids.npy')
        assert len(fit['parameters'])==18
        for param in fit['parameters']:
            seed,h,target=param['seed'],param['horizon'],param['target']
            assert fit['checkpoint_sha256'][str(seed)]==profiles[owner]['checkpoints'][str(seed)]['sha256']
            g=found[(found.seed==seed)&(found.horizon==h)&(found.target==target)]
            low,mid,high,y=[g[k].to_numpy() for k in ['lower','median','upper','actual']]
            residual=np.maximum((mid-y)/np.maximum(mid-low,1e-6),(y-mid)/np.maximum(high-mid,1e-6))
            counts=g.groupby('date').size().to_dict()
            weights=np.array([1/counts[d] for d in g.date])
            ix=np.argsort(residual,kind='stable')
            crossing=np.flatnonzero(np.cumsum(weights[ix])/weights.sum()>=.8)[0]
            scale=max(1.,float(residual[ix[crossing]]))
            np.testing.assert_allclose(param['scale'],scale,atol=1e-12,rtol=1e-12)
            assert len(g)==param['rows']>=1000 and g.date.nunique()==param['dates']>=80
            count+=1
    return count


def replay_audit(data,profiles):
    total=0
    chunks=0
    for stage,owners in [('calibration',['ce','midpoint']),('evaluation',OWNERS)]:
        ids=np.load(ROOT/f'{stage}-ids.npy')
        parent=ROOT/'calibration' if stage=='calibration' else ROOT
        for owner in owners:
            for seed in SEEDS:
                dest=parent/'forecasts'/owner/f'seed{seed}'
                verify(dest)
                meta=read(dest/'lineage.json')
                assert meta['checkpoint_sha256']==profiles[owner]['checkpoints'][str(seed)]['sha256']
                assert meta['decoder_sha256']==profiles['decoder']['sha256']
                assert set(meta['chunks'])=={f'{n:06d}.npz' for n in range(0,len(ids),4)}
                paths=np.load(dest/'paths.npy',mmap_mode='r')
                assert paths.shape==(len(ids),64,5,6)
                for name,sha in meta['chunks'].items():
                    chunk=dest/'chunks'/name
                    assert file_hash(chunk)==sha
                    offset=int(chunk.stem)
                    with np.load(chunk) as z:
                        np.testing.assert_array_equal(z['row_ids'],ids[offset:offset+4])
                        np.testing.assert_array_equal(z['paths'],paths[offset:offset+4])
                        np.testing.assert_array_equal(z['valid'],legal(z['paths']))
                        assert z['s1'].shape==z['s2'].shape==(len(z['row_ids']),64,5)
                        assert ((z['s1']>=0)&(z['s1']<1024)&(z['s2']>=0)&(z['s2']<1024)).all()
                    chunks+=1
                checkpoint=Path(profiles[owner]['checkpoints'][str(seed)]['path'])
                model,_=restore_model(checkpoint,'mps')
                decoder=load_decoder(Path(profiles['decoder']['path']))
                a,b,s,_=data.tensors(ids[:4],'mps')
                pairs=generate_tokens(model,a[:,:60],b[:,:60],s[:,:60],s[:,60:],samples=64,seed=17,temperature=1.,top_p=1.,top_k=0)
                p,v=decode_paths(decoder,pairs,data.a['mean'][ids[:4]],data.a['scale'][ids[:4]],5)
                with np.load(dest/'chunks/000000.npz') as z:
                    np.testing.assert_array_equal(z['paths'],p)
                    np.testing.assert_array_equal(z['valid'],v)
                    for key,value in zip(['s1','s2'],pairs):
                        np.testing.assert_array_equal(z[key],value[:,:,-5:].cpu().numpy())
                total+=1
                del model,decoder
                gc.collect()
                torch.mps.empty_cache()
    return total,chunks


def compare_pairs(daily,pairs,comparisons,owner_field='variant'):
    for row in pairs.itertuples():
        baseline,candidate=comparisons[row.comparison]
        sub=daily[(daily.horizon==row.horizon)&daily[owner_field].isin([baseline,candidate])]
        if hasattr(row,'target'):
            sub=sub[sub.target.isin(['maximum','minimum'])] if row.target=='high_low' else sub[sub.target==row.target]
        if row.seed!='mean':
            sub=sub[sub.seed==int(row.seed)]
        table=sub.pivot_table(index='date',columns=owner_field,values=row.metric,aggfunc='mean').sort_index()
        a,b=table[baseline].to_numpy(),table[candidate].to_numpy()
        difference=b-a
        n=len(difference)
        starts=np.random.default_rng(314159).integers(n,size=(2000,int(np.ceil(n/10))))
        indices=np.concatenate([(starts[:,i,None]+np.arange(10))%n for i in range(starts.shape[1])],axis=1)[:,:n]
        low,high=np.quantile(difference[indices].mean(1),[.025,.975])
        np.testing.assert_allclose([row.baseline,row.candidate,row.delta,row.relative_change,row.ci_low,row.ci_high],
            [a.mean(),b.mean(),difference.mean(),b.mean()/a.mean()-1,low,high],rtol=1e-10,atol=1e-10)
        assert row.dates==len(table)==76


def aggregation_audit(frames,cfg):
    matched=[]
    keys=['row_id','horizon','target']
    for seed in SEEDS:
        shared=None
        for owner in OWNERS:
            f=frames[(owner,seed)]
            index=pd.MultiIndex.from_frame(f[f.variant==owner+'_raw'][keys])
            assert index.is_unique
            shared=index if shared is None else shared.intersection(index)
        for owner in OWNERS:
            f=frames[(owner,seed)]
            matched.append(f[pd.MultiIndex.from_frame(f[keys]).isin(shared)])
    common=pd.concat(matched,ignore_index=True)
    expected=pd.read_parquet(ROOT/'results/common.parquet')
    keys=['seed','row_id','horizon','target','variant']
    pd.testing.assert_frame_equal(common.sort_values(keys).reset_index(drop=True)[expected.columns],
        expected.sort_values(keys).reset_index(drop=True),check_dtype=False,atol=1e-10,rtol=1e-10)
    rows=[]
    for (seed,variant,h,target),g in common.groupby(['seed','variant','horizon','target']):
        dates,codes=np.unique(g.date,return_inverse=True)
        count=np.bincount(codes)
        means={field:np.bincount(codes,weights=g[field])/count for field in old_audit.FIELDS}
        for i,date in enumerate(dates):
            rows.append(dict(seed=seed,variant=variant,horizon=h,target=target,date=date,**{k:v[i] for k,v in means.items()}))
    daily=pd.DataFrame(rows)
    expected=pd.read_csv(ROOT/'results/daily.csv')
    keys=['seed','variant','horizon','target','date']
    pd.testing.assert_frame_equal(daily.sort_values(keys).reset_index(drop=True)[expected.columns],
        expected.sort_values(keys).reset_index(drop=True),check_dtype=False,atol=1e-10,rtol=1e-10)
    pairs=pd.read_csv(ROOT/'results/paired.csv',dtype={'seed':str})
    assert len(pairs)==576
    compare_pairs(daily,pairs,cfg['comparisons'])
    return common,len(pairs)


def midpoint_audit(data,ids,common):
    records=[]
    for seed in SEEDS:
        for h in [2,5]:
            cohort=set(common[(common.seed==seed)&(common.horizon==h)].row_id)
            for owner in OWNERS:
                p=np.load(ROOT/'forecasts'/owner/f'seed{seed}/paths.npy',mmap_mode='r')
                mask=legal(p[:,:,:h])
                for i,row in enumerate(ids):
                    if row not in cohort:
                        continue
                    ref=float(data.a['last'][row,3])
                    draw=p[i,mask[i],:h].astype(float)
                    x=((draw[:,:,1].max(1)+draw[:,:,2].min(1))/(2*ref)-1)*100
                    y=data.a['future'][row,:h]
                    y=((y[:,1].max()+y[:,2].min())/(2*ref)-1)*100
                    records.append(dict(seed=seed,owner=owner,row_id=int(row),date=data.rows.iloc[row].date,horizon=h,
                        mae=abs(np.median(x)-y),crps=abs(x-y).mean()-abs(x[:,None]-x[None,:]).mean()/2))
    frame=pd.DataFrame(records)
    expected=pd.read_parquet(ROOT/'results/midpoint-rows.parquet')
    keys=['seed','owner','row_id','horizon']
    pd.testing.assert_frame_equal(frame.sort_values(keys).reset_index(drop=True)[expected.columns],
        expected.sort_values(keys).reset_index(drop=True),check_dtype=False,atol=1e-10,rtol=1e-10)
    daily=frame.groupby(['seed','owner','date','horizon'])[['mae','crps']].mean().reset_index()
    pd.testing.assert_frame_equal(daily,pd.read_csv(ROOT/'results/midpoint-daily.csv'),check_dtype=False,atol=1e-10,rtol=1e-10)
    pairs=pd.read_csv(ROOT/'results/midpoint-paired.csv',dtype={'seed':str})
    assert len(pairs)==32
    compare_pairs(daily,pairs,dict(current=['current','midpoint'],ce=['ce','midpoint']),'owner')
    return len(frame),len(pairs)


def main():
    checked()
    assert read(ROOT/'run-completed.json')['passed']
    torch.set_num_threads(4)
    data=Dataset()
    cfg,profiles=read(ROOT/'protocol.json'),read(ROOT/'profiles.json')
    frozen,start=read(ROOT/'fits-frozen.json'),read(ROOT/'evaluation-started.json')
    assert frozen['at']<start['at'] and start['fits_frozen_sha256']==file_hash(ROOT/'fits-frozen.json')
    assert frozen['profiles_sha256']==file_hash(ROOT/'profiles.json')
    initial=read(CURRENT/'profiles.json')
    assert profiles['current']==initial['refreshed']
    assert profiles['midpoint']['checkpoints']==read(MID/'candidate-profile.json')['checkpoints']
    for owner in OWNERS:
        fitted=read(Path(profiles[owner]['fit_path']))
        assert file_hash(Path(profiles[owner]['fit_path']))==profiles[owner]['fit_sha256']
        assert fitted['checkpoint_sha256']=={seed:p['sha256'] for seed,p in profiles[owner]['checkpoints'].items()}
        assert fitted['decoder_sha256']==profiles['decoder']['sha256']
    for part in ['calibration','evaluation']:
        ids=np.load(ROOT/f'{part}-ids.npy')
        lo,hi=cfg[part]['dates']
        expected=data.rows[data.rows.date.ge(lo)&data.rows.date.lt(hi)&data.rows.label_end.lt(hi)].groupby('date').head(cfg[part]['rows_per_date']).row_id.to_numpy()
        np.testing.assert_array_equal(ids,expected)
        assert data.rows.iloc[ids].label_end.max()<hi<'2025-08-07'
    factors=calibration_audit(data,profiles)
    print('Verified',factors,'checkpoint-bound calibration factors',flush=True)
    replays,chunks=replay_audit(data,profiles)
    print('Verified',replays,'exact token/price replays and',chunks,'chunks',flush=True)
    old_audit.ROOT=ROOT
    ids=np.load(ROOT/'evaluation-ids.npy')
    frames={}
    for seed in SEEDS:
        for owner in OWNERS:
            frames[(owner,seed)]=old_audit.verify_run(data,ids,owner,seed,profiles)
            print('Verified raw and calibrated scores',seed,owner,flush=True)
    common,pairs=aggregation_audit(frames,cfg)
    midpoint_rows,midpoint_pairs=midpoint_audit(data,ids,common)
    for done in ROOT.rglob('completed.json'):
        verify(done.parent)
    checked()
    write_json(ROOT/'independent-verification.json',dict(passed=True,at=utc_now(),factors=factors,replays=replays,
        replayed_paths=replays*256,forecast_chunks=chunks,raw_and_calibrated_rows=sum(map(len,frames.values())),
        common_rows=len(common),midpoint_rows=midpoint_rows,paired_estimates=pairs+midpoint_pairs,
        medians_and_mae_exactly_preserved=True,calibrated_crps_not_claimed=True,weights_and_sources_unchanged=True,
        calibration_frozen_before_transfer=True,sealed_holdout_opened=False))
    print(read(ROOT/'independent-verification.json'),flush=True)


if __name__=='__main__':
    main()
