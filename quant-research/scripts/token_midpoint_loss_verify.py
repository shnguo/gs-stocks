"""Independent raw-path arithmetic and execution audit for midpoint-loss v1."""
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from token_history_run import Dataset
from token_midpoint_loss_run import DECODER, FIELDS, ROOT, SEEDS, checked, initial
from tokenizer_reconstruction_run import load_decoder, verify

from quant_research.storage import file_hash, utc_now, write_json
from quant_research.token_transformer import decode_paths, generate_tokens, restore_model


def read(path):
    return json.loads(path.read_text())


def legal(x):
    o,h,low,c,v,a=np.moveaxis(x,-1,0)
    return (np.isfinite(x).all(-1)&(x[...,:4]>0).all(-1)&(v>=0)&(a>=0)
        &(h>=o)&(h>=low)&(h>=c)&(low<=o)&(low<=c)).all(-1)


def metrics(draw,y,ref):
    hi=(draw[:,:,1].astype(float).max(1)/ref-1)*100
    lo=(draw[:,:,2].astype(float).min(1)/ref-1)*100
    x=np.array([(hi+lo)/2,hi-lo,hi,lo]).T
    yh=(y[:,1].max()/ref-1)*100
    yl=(y[:,2].min()/ref-1)*100
    target=np.array([(yh+yl)/2,yh-yl,yh,yl])
    low,median,high=np.quantile(x,[.1,.5,.9],axis=0,method='linear')
    mae=abs(median-target)
    # Direct pairwise distances, independent of the production sorted formula.
    crps=abs(x-target).mean(0)-abs(x[:,None,:]-x[None,:,:]).mean((0,1))/2
    coverage=((low<=target)&(target<=high)).astype(float)
    interval=high-low+10*np.maximum(low-target,0)+10*np.maximum(target-high,0)
    return [mae[0],crps[0],mae[2:].mean(),mae[1],crps[1],coverage[2:].mean(),interval[2:].mean()]


def paired(daily,baseline,candidate):
    records=[]
    for seed in [*map(str,SEEDS),'mean']:
        sub=daily if seed=='mean' else daily[daily.seed==int(seed)]
        for h in [2,5]:
            for metric in FIELDS:
                table=sub[sub.horizon==h].groupby(['date','arm'])[metric].mean().unstack().sort_index()
                a,b=table[baseline].to_numpy(),table[candidate].to_numpy()
                delta=b-a
                n=len(a)
                starts=np.random.default_rng(314159).integers(n,size=(2000,int(np.ceil(n/10))))
                indices=np.concatenate([(starts[:,j,None]+np.arange(10))%n for j in range(starts.shape[1])],axis=1)[:,:n]
                lo,hi=np.quantile(delta[indices].mean(1),[.025,.975])
                records.append(dict(seed=seed,horizon=h,metric=metric,baseline=a.mean(),candidate=b.mean(),
                    delta=delta.mean(),relative_change=b.mean()/a.mean()-1,ci_low=lo,ci_high=hi,dates=n))
    return pd.DataFrame(records)


def numerical_audit(data,part):
    ids=np.load(ROOT/f'{part}-ids.npy')
    expected=pd.read_parquet(ROOT/f'{part}-scores/rows.parquet')
    output=[]
    for seed in SEEDS:
        paths={arm:np.load(ROOT/arm/f'seed{seed}'/part/'paths.npy',mmap_mode='r') for arm in ['ce','midpoint']}
        for h in [2,5]:
            masks={arm:legal(p[:,:,:h]) for arm,p in paths.items()}
            common=data.a['valid'][ids,:h].all(1)
            for arm,m in masks.items():
                known=data.a['valid'][ids,:h].all(1)
                common &= m.sum(1)>=16
                coverage=pd.read_csv(ROOT/f'{part}-scores/coverage.csv')
                r=coverage[(coverage.seed==seed)&(coverage.arm==arm)&(coverage.horizon==h)].iloc[0]
                assert r.inputs==len(ids) and r.known==known.sum()
                assert r.usable==(known&(m.sum(1)>=16)).sum()
                assert r.valid_paths==m.sum() and r.total_paths==m.size
            for i in np.flatnonzero(common):
                for arm,p in paths.items():
                    values=metrics(p[i,masks[arm][i],:h],data.a['future'][ids[i],:h],float(data.a['last'][ids[i],3]))
                    output.append(dict(seed=seed,arm=arm,row_id=ids[i],local_row=i,date=data.rows.iloc[ids[i]].date,horizon=h,**dict(zip(FIELDS,values))))
    found=pd.DataFrame(output)
    keys=['seed','arm','row_id','horizon']
    x,y=[f.sort_values(keys).reset_index(drop=True) for f in [found,expected]]
    pd.testing.assert_frame_equal(x[keys+['local_row','date']],y[keys+['local_row','date']],check_dtype=False)
    np.testing.assert_allclose(x[FIELDS],y[FIELDS],atol=1e-10,rtol=1e-10)
    daily=found.groupby(['seed','arm','date','horizon'])[FIELDS].mean().reset_index()
    recorded=pd.read_csv(ROOT/f'{part}-scores/daily.csv')
    pd.testing.assert_frame_equal(daily,recorded,check_dtype=False,atol=1e-10,rtol=1e-10)
    dates=read(ROOT/'splits.json')[part]['dates']
    assert (daily.groupby(['seed','arm','horizon']).size()==dates).all()
    computed=paired(daily,'ce','midpoint')
    recorded=pd.read_csv(ROOT/f'{part}-scores/paired.csv',dtype={'seed':str})
    pd.testing.assert_frame_equal(computed,recorded,check_dtype=False,atol=1e-10,rtol=1e-10)
    return found,len(computed)


def initial_reference(data,frame):
    ids=np.load(ROOT/'evaluation-ids.npy')
    rows=[]
    for seed in SEEDS:
        directory=initial(seed).parent.parent/'forecast'
        np.testing.assert_array_equal(ids,np.load(directory/'row-ids.npy'))
        lineage=read(directory/'lineage.json')
        assert lineage['checkpoint_sha256']==file_hash(initial(seed))
        assert lineage['decoder_sha256']==file_hash(DECODER)
        paths=np.load(directory/'paths.npy',mmap_mode='r')
        for h in [2,5]:
            mask=legal(paths[:,:,:h])
            common=set(frame[(frame.seed==seed)&(frame.horizon==h)].row_id)
            for i in np.flatnonzero(data.a['valid'][ids,:h].all(1)&(mask.sum(1)>=16)):
                if ids[i] not in common:
                    continue
                values=metrics(paths[i,mask[i],:h],data.a['future'][ids[i],:h],float(data.a['last'][ids[i],3]))
                rows.append(dict(seed=seed,arm='initial',row_id=ids[i],local_row=i,date=data.rows.iloc[ids[i]].date,horizon=h,**dict(zip(FIELDS,values))))
    original=pd.DataFrame(rows)
    keys=original[['seed','row_id','horizon']]
    matched=frame.merge(keys,on=['seed','row_id','horizon'],validate='many_to_one')
    all_rows=pd.concat([original,matched],ignore_index=True)
    assert (all_rows.groupby(['seed','row_id','horizon']).arm.nunique()==3).all()
    dest=ROOT/'initial-reference'
    dest.mkdir(exist_ok=True)
    all_rows.to_parquet(dest/'rows.parquet',index=False)
    daily=all_rows.groupby(['seed','arm','date','horizon'])[FIELDS].mean().reset_index()
    daily.to_csv(dest/'daily.csv',index=False)
    daily.groupby(['seed','arm','horizon'])[FIELDS].mean().reset_index().to_csv(dest/'summary.csv',index=False)
    comparisons=pd.concat([paired(daily,'initial',arm).assign(comparison=arm) for arm in ['ce','midpoint']],ignore_index=True)
    comparisons.to_csv(dest/'paired.csv',index=False)
    return len(original),len(comparisons)


def execution_audit(data):
    chunks_count=0
    exposure=[]
    for seed in SEEDS:
        aux_ids=[]
        original=torch.load(initial(seed),map_location='cpu',weights_only=True)
        for epoch in [1,2]:
            orders=[np.load(ROOT/arm/f'seed{seed}/training/epoch{epoch}-ids.npy') for arm in ['ce','midpoint']]
            np.testing.assert_array_equal(*orders)
            assert len(orders[0])==60864 and len(np.unique(orders[0]))==60864
            assert (data.rows.iloc[orders[0]].groupby('date').size()==32).all()
            for start in range(0,len(orders[0]),256):
                batch=orders[0][start:start+256]
                eligible=np.flatnonzero(data.a['valid'][batch,:2].all(1))[:8]
                aux_ids.extend(batch[eligible].tolist())
        exposure.append(dict(seed=seed,ce_examples=121728,auxiliary_examples=len(aux_ids),
            auxiliary_unique_examples=len(set(aux_ids)),auxiliary_dates=int(data.rows.iloc[aux_ids].date.nunique())))
        for arm in ['ce','midpoint']:
            train=ROOT/arm/f'seed{seed}/training'
            verify(train)
            model,saved=restore_model(train/'final.pt','cpu')
            assert saved['epoch']==2 and saved['initial_checkpoint_sha256']==file_hash(initial(seed))
            assert saved['config']==original['config']
            assert saved['seed']==seed and saved['arm']==arm
            assert saved['auxiliary_weight']==(0.05 if arm=='midpoint' else 0.0)
            assert any(not torch.equal(value,original['state_dict'][key]) for key,value in saved['state_dict'].items())
            assert sum(p.numel() for p in model.parameters())==3720448
            model.eval()
            with np.load(train/'reload-reference.npz') as z,torch.inference_mode():
                a,b,s,_=data.tensors(z['ids'],'cpu')
                coarse,fine=model.forecast_logits(a[:,:-1],b[:,:-1],s[:,:-1],a[:,1:],59)
                np.testing.assert_array_equal(coarse.numpy(),z['coarse'])
                np.testing.assert_array_equal(fine.numpy(),z['fine'])
            model=model.to('mps')
            decoder=load_decoder(DECODER)
            for part in ['selection','evaluation']:
                directory=ROOT/arm/f'seed{seed}'/part
                verify(directory)
                ids=np.load(ROOT/f'{part}-ids.npy')
                p=np.load(directory/'paths.npy',mmap_mode='r')
                assert p.shape==(len(ids),64,5,6)
                lineage=read(directory/'lineage.json')
                assert lineage['checkpoint_sha256']==file_hash(train/'final.pt')
                assert lineage['decoder_sha256']==file_hash(DECODER)
                assert set(lineage['chunks'])=={f'{i:06d}.npz' for i in range(0,len(ids),4)}
                for name,sha in lineage['chunks'].items():
                    path=directory/'chunks'/name
                    assert file_hash(path)==sha
                    start=int(path.stem)
                    with np.load(path) as z:
                        np.testing.assert_array_equal(z['row_ids'],ids[start:start+4])
                        np.testing.assert_array_equal(z['paths'],p[start:start+4])
                        np.testing.assert_array_equal(z['valid'],legal(z['paths']))
                        for key in ['s1','s2']:
                            assert z[key].shape==(len(z['row_ids']),64,5)
                            assert (z[key]>=0).all() and (z[key]<1024).all()
                    chunks_count+=1
                a,b,s,_=data.tensors(ids[:4],'mps')
                tokens=generate_tokens(model,a[:,:60],b[:,:60],s[:,:60],s[:,60:],samples=64,seed=17,temperature=1.,top_p=1.,top_k=0)
                replay,valid=decode_paths(decoder,tokens,data.a['mean'][ids[:4]],data.a['scale'][ids[:4]],5)
                np.testing.assert_array_equal(replay,p[:4])
                with np.load(directory/'chunks/000000.npz') as z:
                    np.testing.assert_array_equal(valid,z['valid'])
                    for key,t in zip(['s1','s2'],tokens):
                        np.testing.assert_array_equal(z[key],t[:,:,-5:].cpu().numpy())
            del model,decoder
            gc.collect()
            torch.mps.empty_cache()
    pd.DataFrame(exposure).to_csv(ROOT/'training-exposure.csv',index=False)
    return chunks_count


def policy_audit(data):
    model,_=restore_model(initial(17),'mps')
    model.eval()
    decoder=load_decoder(DECODER)
    ids=np.array(read(ROOT/'pilot.json')['row_ids'][:2])
    a,b,s,_=data.tensors(ids,'mps')
    pairs=generate_tokens(model,a[:,:60],b[:,:60],s[:,:60],s[:,60:],samples=4,seed=2718,temperature=1.,top_p=1.,top_k=0)
    x,y=[p.reshape(-1,65) for p in pairs]
    stamps=s.repeat_interleave(4,0)
    with torch.inference_mode():
        batch_c,batch_f=model.forecast_logits(x[:,:-1],y[:,:-1],stamps[:,:-1],x[:,1:],59)
        for t in range(5):
            n=60+t
            c,hidden=model.decode_s1(x[:,:n],y[:,:n],stamps[:,:n])
            f=model.decode_last_s2(hidden,x[:,n])
            torch.testing.assert_close(c[:,-1],batch_c[:,t],atol=5e-5,rtol=5e-5)
            torch.testing.assert_close(f,batch_f[:,t],atol=5e-5,rtol=5e-5)
    full,_=decode_paths(decoder,pairs,data.a['mean'][ids],data.a['scale'][ids],5)
    prefix,_=decode_paths(decoder,[p[:,:,:62] for p in pairs],data.a['mean'][ids],data.a['scale'][ids],2)
    np.testing.assert_allclose(full[:,:,:2],prefix,atol=1e-4,rtol=1e-5)
    return dict(actual_sampler_logits_match=True,decoder_prefix_causal=True)


def main():
    checked()
    previous=read(ROOT.parent/'token-range-decomposition-20260915-v1/final-verification.json')
    prior_count=0
    for name,sha in previous['files'].items():
        if str(ROOT.parent)+'/' in name:
            assert file_hash(Path(name))==sha,name
            prior_count+=1
    assert read(ROOT/'run-completed.json')['passed']
    torch.set_num_threads(4)
    data=Dataset()
    choice=read(ROOT/'selection.json')
    start=read(ROOT/'evaluation-started.json')
    assert choice['at']<start['at'] and start['selection_sha256']==file_hash(ROOT/'selection.json')
    for part,lo,hi in [('training','2016-01-01','2024-01-01'),('selection','2024-01-01','2024-07-01'),('evaluation','2024-07-01','2025-01-01')]:
        rows=data.rows.iloc[np.load(ROOT/f'{part}-ids.npy')]
        assert rows.date.min()>=lo and rows.label_end.max()<hi<'2025-08-07'
    policy=policy_audit(data)
    print('Policy likelihood and causal decoder verified',flush=True)
    chunks=execution_audit(data)
    print('All checkpoints, chunks and exact token/price replays verified',flush=True)
    selection,n1=numerical_audit(data,'selection')
    daily=selection.groupby(['seed','arm','date','horizon']).center_crps.mean()
    means=daily.groupby('arm').mean()
    assert choice['selected']==('midpoint' if means.midpoint<means.ce else 'ce')
    np.testing.assert_allclose([choice['ce_center_crps'],choice['midpoint_center_crps']],[means.ce,means.midpoint],atol=1e-12)
    evaluation,n2=numerical_audit(data,'evaluation')
    original,n3=initial_reference(data,evaluation)
    checked()
    write_json(ROOT/'independent-verification.json',dict(passed=True,at=utc_now(),raw_score_rows=len(selection)+len(evaluation),
        original_reference_rows=original,paired_estimates_verified=n1+n2,reference_paired_estimates=n3,forecast_chunks=chunks,
        exact_replays=12,replayed_paths=12*4*64,cpu_reload_checks=6,matched_training_exposure=True,
        selection_recorded_before_evaluation=True,source_files_unchanged=len(read(ROOT/'sources.json')),**policy))
    receipt=read(ROOT/'independent-verification.json')
    receipt['prior_artifact_files_unchanged']=prior_count
    write_json(ROOT/'independent-verification.json',receipt)
    print(read(ROOT/'independent-verification.json'),flush=True)


if __name__=='__main__':
    main()
