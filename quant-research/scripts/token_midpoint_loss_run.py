"""Matched CE versus sampled-midpoint-risk continuation; frozen chronology."""
import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from token_history_run import Dataset
from tokenizer_reconstruction_run import finish, load_decoder, verify

from quant_research.midpoint_loss import sampled_midpoint_loss
from quant_research.storage import file_hash, utc_now, write_json
from quant_research.token_history import date_weights, epoch_sample, loss_parts, partition_indices
from quant_research.token_transformer import (
    checkpoint_payload,
    decode_paths,
    generate_tokens,
    restore_model,
    valid_bars,
)

BASE=Path('/Users/guo/Documents/stocks/quant-research')
ART=BASE/'artifacts'
ROOT=ART/'token-midpoint-loss-20260915-v1'
DATA=ART/'token-history-data-20260915-v1'
PROFILE=ART/'token-profile-transfer-20260915-v1/profiles.json'
DECODER=ART/'tokenizer-reconstruction-20260915-v1/decoder-training/price_consistency/best-qualified.pt'
SEEDS=[17,29,43]
FIELDS=['center_mae','center_crps','endpoint_mae','range_mae','range_crps','coverage80','interval_score80']


def read(p):
    return json.loads(p.read_text())


def prepare():
    assert not ROOT.exists()
    cfg=read(BASE/'configs/token-midpoint-loss-v1.json')
    sources={str(PROFILE):file_hash(PROFILE),str(DATA/'completed.json'):file_hash(DATA/'completed.json')}
    profiles=read(PROFILE)
    assert file_hash(DECODER)==profiles['decoder']['sha256']
    sources[str(DECODER)]=file_hash(DECODER)
    for p in profiles['refreshed']['checkpoints'].values():
        assert file_hash(Path(p['path']))==p['sha256']
        sources[p['path']]=p['sha256']
        forecast_dir=Path(p['path']).parent.parent/'forecast'
        verify(forecast_dir)
        for name in ['paths.npy','row-ids.npy','lineage.json','completed.json']:
            sources[str(forecast_dir/name)]=file_hash(forecast_dir/name)
    for name in ['rows.parquet','s1.npy','s2.npy','valid.npy','future.npy','last.npy','mean.npy','scale.npy','calendar-stamps.npy']:
        p=DATA/name
        assert file_hash(p)==read(DATA/'completed.json')['files'][name]
        sources[str(p)]=file_hash(p)
    calendar=DATA/'calendar.json'
    assert file_hash(calendar)==read(DATA/'completed.json')['files']['calendar.json']
    sources[str(calendar)]=file_hash(calendar)
    ROOT.mkdir()
    write_json(ROOT/'protocol.json',cfg)
    data=Dataset()
    dates=read(calendar)
    counts={}
    for part in ['training','selection','evaluation']:
        ids=partition_indices(data.rows,dates,*cfg[part]['dates'])
        if part=='training':
            ids=ids[data.a['valid'][ids].any(1)]
        else:
            ids=data.rows.iloc[ids].groupby('date').head(cfg[part]['rows_per_date']).row_id.to_numpy(int)
        r=data.rows.iloc[ids]
        assert r.date.min()>=cfg[part]['dates'][0] and r.label_end.max()<cfg[part]['dates'][1]<cfg['sealed_holdout_start']
        np.save(ROOT/f'{part}-ids.npy',ids)
        counts[part]=dict(inputs=len(ids),dates=int(r.date.nunique()),first=r.date.min(),last=r.date.max(),label_end=r.label_end.max())
    assert counts['training']['inputs']==354007 and counts['training']['dates']==1902
    write_json(ROOT/'splits.json',counts)
    shutil.copytree(BASE/'src',ROOT/'code/src',ignore=shutil.ignore_patterns('__pycache__'))
    (ROOT/'code/scripts').mkdir()
    for name in ['token_midpoint_loss_run.py','token_history_run.py','tokenizer_reconstruction_run.py']:
        shutil.copy2(BASE/'scripts'/name,ROOT/'code/scripts'/name)
    write_json(ROOT/'sources.json',sources)
    write_json(ROOT/'prepared.json',dict(passed=True,files={str(p.relative_to(ROOT)):file_hash(p) for p in ROOT.rglob('*') if p.is_file()}))
    print('Prepared',counts,flush=True)


def checked():
    for name,h in read(ROOT/'prepared.json')['files'].items():
        assert file_hash(ROOT/name)==h,name
    for name,h in read(ROOT/'sources.json').items():
        assert file_hash(Path(name))==h,name


def initial(seed):
    return Path(read(PROFILE)['refreshed']['checkpoints'][str(seed)]['path'])


def aux(model,decoder,data,ids,seed,weights=None):
    a,b,stamps,_=data.tensors(ids,'mps')
    return sampled_midpoint_loss(model,decoder,a,b,stamps,data.a['mean'][ids],data.a['scale'][ids],
        data.a['future'][ids],data.a['valid'][ids],data.a['last'][ids,3],seed=seed,
        samples=read(ROOT/'protocol.json')['auxiliary']['draws_per_row'],weights=weights)


def pilot():
    checked()
    torch.set_num_threads(4)
    data=Dataset()
    ids=np.load(ROOT/'training-ids.npy')
    ids=ids[data.a['valid'][ids].all(1)]
    ids=ids[np.linspace(0,len(ids)-1,8,dtype=int)]
    model,_=restore_model(initial(17),'mps')
    decoder=load_decoder(DECODER)
    assert all(not p.requires_grad for p in decoder.parameters())
    before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    loss,stats=aux(model,decoder,data,ids,seed=2718)
    loss.backward()
    norms={name:float(p.grad.norm()) for name,p in model.named_parameters() if p.grad is not None}
    assert norms and all(np.isfinite(v) for v in norms.values())
    for name in ['coarse_head.weight','fine_head.weight','fusion.weight']:
        assert norms[name]>0
    assert all(p.grad is None for p in decoder.parameters())
    a,b,stamps,_=data.tensors(ids,'mps')
    pairs=generate_tokens(model,a[:,:60],b[:,:60],stamps[:,:60],stamps[:,60:],samples=4,seed=2718,temperature=1.,top_p=1.,top_k=0)
    paths,_=decode_paths(decoder,pairs,data.a['mean'][ids],data.a['scale'][ids],5)
    crps=[]
    ref=data.a['last'][ids,3].astype(float)
    for h in [2,5]:
        high=paths[:,:,:h,1].astype(float).max(2)
        low=paths[:,:,:h,2].astype(float).min(2)
        v=((high+low)/(2*ref[:,None])-1)*100
        y=data.a['future'][ids,:h]
        target=((y[:,:,1].max(1)+y[:,:,2].min(1))/(2*ref)-1)*100
        crps.append((abs(v-target[:,None]).mean(1)-abs(v[:,:,None]-v[:,None,:]).sum((1,2))/(2*4*3)).mean())
    np.testing.assert_allclose(stats['crps'],np.mean(crps),atol=2e-5,rtol=2e-5)
    # Future token labels are unused during generation and the sampled-risk pass.
    from quant_research.midpoint_loss import sampled_midpoint_loss
    changed_a,changed_b=a.clone(),b.clone()
    changed_a[:,60:]=(changed_a[:,60:]+99)%1024
    changed_b[:,60:]=(changed_b[:,60:]+77)%1024
    other,_=sampled_midpoint_loss(model,decoder,changed_a,changed_b,stamps,data.a['mean'][ids],data.a['scale'][ids],
        data.a['future'][ids],data.a['valid'][ids],data.a['last'][ids,3],seed=2718,samples=4)
    torch.testing.assert_close(loss.detach(),other.detach(),atol=0,rtol=0)
    for k,v in model.state_dict().items():
        torch.testing.assert_close(v.cpu(),before[k],atol=0,rtol=0)
    write_json(ROOT/'pilot.json',dict(passed=True,at=utc_now(),row_ids=ids.tolist(),statistics=stats,
        gradient_norms=norms,decoder_frozen=True,actual_sampled_crps_verified=True,future_token_targets_unused=True,
        model_weights_unchanged=True))
    print('Actual-path gradient pilot passed',stats,flush=True)


def save_model(model,path,**metadata):
    payload=checkpoint_payload(model,**metadata)
    payload['state_dict']={k:v.detach().cpu() for k,v in model.state_dict().items()}
    torch.save(payload,path)


def train(data,seed,arm):
    cfg=read(ROOT/'protocol.json')
    dest=ROOT/arm/f'seed{seed}/training'
    if (dest/'completed.json').exists():
        verify(dest)
        return
    dest.mkdir(parents=True)
    torch.manual_seed(seed)
    model,_=restore_model(initial(seed),'mps')
    cpu_rng=torch.get_rng_state()
    mps_rng=torch.mps.get_rng_state()
    decoder=load_decoder(DECODER) if arm=='midpoint' else None
    torch.set_rng_state(cpu_rng)
    torch.mps.set_rng_state(mps_rng)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['training']['learning_rate'],weight_decay=cfg['training']['weight_decay'])
    ids=np.load(ROOT/'training-ids.npy')
    groups={day:g.row_id.to_numpy(int) for day,g in data.rows.iloc[ids].groupby('date')}
    entries=[]
    visited=set()
    started=time.monotonic()
    coefficient=cfg['arms'][arm]
    for epoch in range(1,3):
        order=epoch_sample(groups,epoch,32,seed)
        visited.update(order.tolist())
        np.save(dest/f'epoch{epoch}-ids.npy',order)
        dw=date_weights(data.rows,order)
        model.train()
        totals=dict(ce=0.,policy=0.,crps=0.,legal_fraction=0.,aux_batches=0)
        for start in range(0,len(order),256):
            batch=order[start:start+256]
            optimizer.zero_grad(set_to_none=True)
            parts=loss_parts(model,*data.tensors(batch,'mps'),60).mean(1)
            weights=torch.as_tensor(dw[start:start+len(batch)],device='mps')
            ce=(parts*weights).mean()
            ce.backward()
            policy=0.
            if coefficient:
                eligible=np.flatnonzero(data.a['valid'][batch,:2].all(1))[:8]
                if len(eligible):
                    weights=weights[torch.as_tensor(eligible,device='mps')]
                    risk,stats=aux(model,decoder,data,batch[eligible],seed=seed*1000000+epoch*10000+start,weights=weights)
                    (coefficient*risk).backward()
                    policy=float(risk.detach())
                    totals['crps']+=stats['crps']
                    totals['legal_fraction']+=stats['legal_fraction']
                    totals['aux_batches']+=1
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            assert torch.isfinite(norm) and torch.isfinite(ce)
            optimizer.step()
            totals['ce']+=float(ce.detach())*len(batch)
            totals['policy']+=policy*len(batch)
            if start%(256*80)==0:
                p=dict(stage='training',seed=seed,arm=arm,epoch=epoch,rows=start+len(batch),total=len(order),at=utc_now())
                write_json(dest/'progress.json',p)
                print(p,flush=True)
        entries.append(dict(epoch=epoch,examples=len(order),ce=totals['ce']/len(order),policy_surrogate=totals['policy']/len(order),
            sampled_midpoint_crps=totals['crps']/max(1,totals['aux_batches']),legal_fraction=totals['legal_fraction']/max(1,totals['aux_batches']),
            elapsed_seconds=time.monotonic()-started))
        write_json(dest/'history.json',entries)
    save_model(model,dest/'final.pt',epoch=2,arm=arm,seed=seed,auxiliary_weight=coefficient,
        dataset_manifest_sha256=file_hash(DATA/'completed.json'),initial_checkpoint_sha256=file_hash(initial(seed)))
    np.save(dest/'visited-row-ids.npy',np.asarray(sorted(visited),dtype=np.int64))
    probe=np.load(ROOT/'selection-ids.npy')[:2]
    model=model.cpu().eval()
    with torch.inference_mode():
        a,b,s,_=data.tensors(probe,'cpu')
        expected=model.forecast_logits(a[:,:-1],b[:,:-1],s[:,:-1],a[:,1:],59)
    loaded,_=restore_model(dest/'final.pt','cpu')
    loaded.eval()
    with torch.inference_mode():
        actual=loaded.forecast_logits(a[:,:-1],b[:,:-1],s[:,:-1],a[:,1:],59)
    for x,y in zip(expected,actual):
        torch.testing.assert_close(x,y,atol=0,rtol=0)
    np.savez_compressed(dest/'reload-reference.npz',ids=probe,coarse=expected[0].numpy(),fine=expected[1].numpy())
    write_json(dest/'summary.json',dict(passed=True,epochs=2,seed=seed,arm=arm,coefficient=coefficient,
        parameters=sum(p.numel() for p in model.parameters()),training_dates=len(groups),known_training_pool=len(ids),
        total_examples=sum(e['examples'] for e in entries),unique_examples=len(visited),cpu_reload_exact=True,
        elapsed_seconds=time.monotonic()-started,initial_checkpoint_sha256=file_hash(initial(seed))))
    finish(dest)
    del model,loaded,optimizer,decoder
    gc.collect()
    torch.mps.empty_cache()


def forecast(data,seed,arm,part):
    dest=ROOT/arm/f'seed{seed}'/part
    if (dest/'completed.json').exists():
        verify(dest)
        return
    dest.mkdir(parents=True)
    chunks=dest/'chunks'
    chunks.mkdir()
    ids=np.load(ROOT/f'{part}-ids.npy')
    model,_=restore_model(ROOT/arm/f'seed{seed}/training/final.pt','mps')
    decoder=load_decoder(DECODER)
    model.eval()
    result=np.lib.format.open_memmap(dest/'paths.npy',mode='w+',dtype=np.float32,shape=(len(ids),64,5,6))
    for start in range(0,len(ids),4):
        batch=ids[start:start+4]
        a,b,s,_=data.tensors(batch,'mps')
        pairs=generate_tokens(model,a[:,:60],b[:,:60],s[:,:60],s[:,60:],samples=64,seed=17+start,temperature=1.,top_p=1.,top_k=0)
        p,v=decode_paths(decoder,pairs,data.a['mean'][batch],data.a['scale'][batch],5)
        np.savez_compressed(chunks/f'{start:06d}.npz',paths=p,valid=v,row_ids=batch,s1=pairs[0][:,:,-5:].cpu().numpy(),s2=pairs[1][:,:,-5:].cpu().numpy())
        result[start:start+len(batch)]=p
        if start%512==0:
            progress=dict(stage=part,seed=seed,arm=arm,rows=start+len(batch),total=len(ids),at=utc_now())
            write_json(dest/'progress.json',progress)
            print(progress,flush=True)
    result.flush()
    a,b,s,_=data.tensors(ids[:4],'mps')
    pairs=generate_tokens(model,a[:,:60],b[:,:60],s[:,:60],s[:,60:],samples=64,seed=17,temperature=1.,top_p=1.,top_k=0)
    replay,valid=decode_paths(decoder,pairs,data.a['mean'][ids[:4]],data.a['scale'][ids[:4]],5)
    np.testing.assert_array_equal(result[:4],replay)
    with np.load(chunks/'000000.npz') as z:
        np.testing.assert_array_equal(z['valid'],valid)
        for k,v in zip(['s1','s2'],pairs):
            np.testing.assert_array_equal(z[k],v[:,:,-5:].cpu().numpy())
    write_json(dest/'lineage.json',dict(passed=True,checkpoint_sha256=file_hash(ROOT/arm/f'seed{seed}/training/final.pt'),
        decoder_sha256=file_hash(DECODER),first_batch_exact=True,chunks={p.name:file_hash(p) for p in chunks.iterdir()}))
    finish(dest)
    del model,decoder,result
    gc.collect()
    torch.mps.empty_cache()


def score(data,part):
    from quant_research.forecast_audit import empirical_score
    ids=np.load(ROOT/f'{part}-ids.npy')
    records=[]
    coverage=[]
    for seed in SEEDS:
        paths={arm:np.load(ROOT/arm/f'seed{seed}'/part/'paths.npy',mmap_mode='r') for arm in ['ce','midpoint']}
        for h in [2,5]:
            masks={k:valid_bars(v[:,:,:h]).all(-1) for k,v in paths.items()}
            known=data.a['valid'][ids,:h].all(1)
            common=known.copy()
            for name,mask in masks.items():
                usable=known&(mask.sum(1)>=16)
                common &= usable
                coverage.append(dict(seed=seed,arm=name,horizon=h,inputs=len(ids),known=int(known.sum()),usable=int(usable.sum()),valid_paths=int(mask.sum()),total_paths=int(mask.size)))
            for i in np.flatnonzero(common):
                ref=float(data.a['last'][ids[i],3])
                y=data.a['future'][ids[i],:h]
                yh=(y[:,1].max()/ref-1)*100
                yl=(y[:,2].min()/ref-1)*100
                for arm,p in paths.items():
                    draw=p[i,masks[arm][i],:h].astype(float)
                    high=(draw[:,:,1].max(-1)/ref-1)*100
                    low=(draw[:,:,2].min(-1)/ref-1)*100
                    values=np.column_stack([(high+low)/2,high-low,high,low])
                    scores=empirical_score(values,np.array([(yh+yl)/2,yh-yl,yh,yl]))
                    records.append(dict(seed=seed,arm=arm,row_id=int(ids[i]),local_row=int(i),date=data.rows.iloc[ids[i]].date,horizon=h,
                        center_mae=scores['mae'][0],center_crps=scores['crps'][0],range_mae=scores['mae'][1],range_crps=scores['crps'][1],
                        endpoint_mae=scores['mae'][2:].mean(),coverage80=scores['coverage80'][2:].mean(),interval_score80=scores['interval_score80'][2:].mean()))
    dest=ROOT/f'{part}-scores'
    dest.mkdir(exist_ok=True)
    frame=pd.DataFrame(records)
    frame.to_parquet(dest/'rows.parquet',index=False)
    pd.DataFrame(coverage).to_csv(dest/'coverage.csv',index=False)
    daily=frame.groupby(['seed','arm','date','horizon'])[FIELDS].mean().reset_index()
    daily.to_csv(dest/'daily.csv',index=False)
    daily.groupby(['seed','arm','horizon'])[FIELDS].mean().reset_index().to_csv(dest/'summary.csv',index=False)
    pairs=[]
    for seed in [*map(str,SEEDS),'mean']:
        sub=daily if seed=='mean' else daily[daily.seed==int(seed)]
        for h in [2,5]:
            for metric in FIELDS:
                p=sub[sub.horizon==h].groupby(['date','arm'])[metric].mean().unstack().sort_index()
                b,x=p.ce.to_numpy(),p.midpoint.to_numpy()
                rng=np.random.default_rng(314159)
                ix=(rng.integers(len(p),size=(2000,int(np.ceil(len(p)/10)),1))+np.arange(10))%len(p)
                lo,hi=np.quantile((x-b)[ix.reshape(2000,-1)[:,:len(p)]].mean(1),[.025,.975])
                pairs.append(dict(seed=seed,horizon=h,metric=metric,baseline=b.mean(),candidate=x.mean(),delta=(x-b).mean(),relative_change=x.mean()/b.mean()-1,ci_low=lo,ci_high=hi,dates=len(p)))
    pd.DataFrame(pairs).to_csv(dest/'paired.csv',index=False)
    finish(dest)
    return daily


def run():
    checked()
    assert read(ROOT/'pilot.json')['passed']
    torch.set_num_threads(4)
    data=Dataset()
    write_json(ROOT/'runtime.json',dict(at=utc_now(),pid=os.getpid(),python=sys.version,torch=str(torch.__version__),numpy=np.__version__,device='mps'))
    for seed in SEEDS:
        for arm in ['ce','midpoint']:
            train(data,seed,arm)
    for seed in SEEDS:
        for arm in ['ce','midpoint']:
            forecast(data,seed,arm,'selection')
    daily=score(data,'selection')
    means=daily.groupby('arm').center_crps.mean()
    selected='midpoint' if means.midpoint<means.ce else 'ce'
    decision=dict(at=utc_now(),selected=selected,weight=read(ROOT/'protocol.json')['arms'][selected],
        ce_center_crps=float(means.ce),midpoint_center_crps=float(means.midpoint),evaluation_used=False,
        rule='Lower mean validation midpoint CRPS, two horizons and three seeds equally weighted')
    if (ROOT/'selection.json').exists():
        old=read(ROOT/'selection.json')
        assert old['selected']==selected
    else:
        write_json(ROOT/'selection.json',decision)
    print('Frozen validation selection',decision,flush=True)
    if not (ROOT/'evaluation-started.json').exists():
        write_json(ROOT/'evaluation-started.json',dict(at=utc_now(),selection_sha256=file_hash(ROOT/'selection.json')))
    for seed in SEEDS:
        for arm in ['ce','midpoint']:
            forecast(data,seed,arm,'evaluation')
    score(data,'evaluation')
    write_json(ROOT/'run-completed.json',dict(passed=True,at=utc_now(),fits=6,forecast_runs=12,
        selection_frozen_before_evaluation=True,sealed_holdout_opened=False,calibration_refitted=False,model_defaults_changed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['prepare','pilot','run'])
    globals()[p.parse_args().stage]()
