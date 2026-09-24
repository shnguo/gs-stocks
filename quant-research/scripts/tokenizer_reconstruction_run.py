"""Frozen-codec decoder adaptation and matched-token forecast validation."""
import argparse
import gc
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import token_history_run as history
import torch
from token_history_run import Dataset, atomic_save

from quant_research.forecast_audit import TARGETS, common_scores, extrema, score_paths
from quant_research.storage import file_hash, utc_now, write_json
from quant_research.token_history import date_weights, epoch_sample, partition_indices
from quant_research.token_transformer import (
    decode_paths,
    generate_tokens,
    load_tokenizer,
    restore_model,
    valid_bars,
)
from quant_research.tokenizer_reconstruction import enable_decoder_training, reconstruction_loss

BASE=Path('/Users/guo/Documents/stocks/quant-research')
OUT=BASE/'artifacts/tokenizer-reconstruction-20260915-v1'
DATA=BASE/'artifacts/token-history-data-20260915-v1'
BUNDLE=BASE/'artifacts/kronos-comparison-20260910-v1'
history.OUT=OUT/'predictors'


def read(p):
    return json.loads(p.read_text())


def finish(folder):
    write_json(folder/'completed.json',dict(passed=True,at=utc_now(),files={p.name:file_hash(p)
        for p in folder.iterdir() if p.is_file() and p.name not in ['completed.json','progress.json']}))


def verify(folder):
    for name,h in read(folder/'completed.json')['files'].items():
        assert file_hash(folder/name)==h,(folder,name)


def initialize(config_path):
    cfg=read(config_path)
    if (OUT/'protocol.json').exists():
        assert read(OUT/'protocol.json')==cfg
        return
    verify(DATA)
    OUT.mkdir()
    write_json(OUT/'protocol.json',cfg)
    shutil.copytree(BASE/'src',OUT/'code/src',ignore=shutil.ignore_patterns('__pycache__'))
    (OUT/'code/scripts').mkdir()
    for name in ['tokenizer_reconstruction_run.py','token_history_run.py']:
        shutil.copy2(BASE/'scripts'/name,OUT/'code/scripts'/name)
    write_json(OUT/'source-manifest.json',dict(dataset=file_hash(DATA/'completed.json'),
        protocol=file_hash(OUT/'protocol.json'),
        files={str(p.relative_to(OUT)):file_hash(p) for p in (OUT/'code').rglob('*.py')}))


class ReconstructionData(Dataset):
    def __init__(self,cfg,seed=17):
        super().__init__()
        self.protocol=cfg
        tr=cfg['forecast_training']
        self.cfg.update(seed=seed,selection_per_date=cfg['forecast_selection_rows_per_date'],
            min_epochs=tr['minimum_epochs'],max_epochs=tr['maximum_epochs'],patience=tr['patience'],
            batch_size=tr['batch_size'],train_per_date_per_epoch=tr['stocks_per_date_per_epoch'],
            learning_rate=tr['learning_rate'],weight_decay=tr['weight_decay'])
        dates=np.asarray(read(BASE/'artifacts/kronos-inputs-20260914-v3/manifest.json')['dates'])
        self.splits={part:partition_indices(self.rows,dates,*cfg['fold'][part],5)
                     for part in ['train','selection','evaluation']}
        split=[]
        for part,ids in self.splits.items():
            r=self.rows.iloc[ids]
            assert r.date.min()>=cfg['fold'][part][0]
            assert r.label_end.max()<cfg['fold'][part][1]<cfg['sealed_holdout_start']
            p=OUT/f'{part}-ids.npy'
            if p.exists():
                np.testing.assert_array_equal(np.load(p),ids)
            else:
                np.save(p,ids)
            split.append(dict(partition=part,rows=len(ids),known_rows=int(self.a['valid'][ids].any(1).sum()),
                dates=int(r.date.nunique()),first=str(r.date.min()),last=str(r.date.max()),last_label=str(r.label_end.max())))
        if not (OUT/'splits.json').exists():
            write_json(OUT/'splits.json',split)

    def selected(self,fold,part,count=None,known=False):
        ids=self.splits[part]
        if count is not None:
            ids=self.rows.iloc[ids].groupby('date',sort=True).head(count).row_id.to_numpy(int)
        if known:
            ids=ids[self.a['valid'][ids].any(1)]
        return ids

    def decoder_batch(self,ids,device):
        pairs=[torch.as_tensor(self.a[k][ids].astype(np.int64),device=device) for k in ['s1','s2']]
        values=[torch.as_tensor(np.array(self.a[k][ids]),device=device,dtype=torch.float32) for k in ['future','mean','scale']]
        known=torch.as_tensor(self.a['valid'][ids],device=device)
        ref=torch.as_tensor(self.a['last'][ids,3],device=device,dtype=torch.float32)
        return pairs,values,known,ref


def reconstruction_scores(data,ids,values):
    rows=[]
    ref=data.a['last'][ids,3].astype(float)
    future=data.a['future'][ids]
    for h in [2,5]:
        ok=data.a['valid'][ids,:h].all(1)
        error=extrema(values[:,:h].astype(float),ref)-extrema(future[:,:h],ref)
        norm=np.abs((values[:,:h]-future[:,:h])/data.a['scale'][ids])
        legal=valid_bars(values[:,:h]).all(1)
        for i in np.flatnonzero(ok):
            for j,target in enumerate(TARGETS):
                rows.append(dict(row_id=int(ids[i]),date=data.rows.iloc[ids[i]].date,horizon=h,target=target,
                    mae=abs(error[i,j]),bias=error[i,j],legal=float(legal[i]),
                    normalized_mae=float(norm[i].mean()),turnover_mae=float(norm[i,:,4:].mean())))
    frame=pd.DataFrame(rows)
    fields=['mae','bias','legal','normalized_mae','turnover_mae']
    daily=frame.groupby(['date','horizon','target'])[fields].mean().reset_index()
    summary=daily.groupby(['horizon','target'])[fields].mean().reset_index()
    aggregate={k:float(v) for k,v in summary[fields].mean().items()}
    return frame,daily,summary,aggregate


@torch.inference_mode()
def reconstruct(tokenizer,data,ids,dest=None):
    tokenizer.eval()
    values=[]
    for start in range(0,len(ids),256):
        batch=ids[start:start+256]
        pairs,(_,mean,scale),_,_=data.decoder_batch(batch,'mps')
        decoded=tokenizer.decode(pairs,half=True)[:,-5:]*scale+mean
        values.append(decoded.cpu().numpy())
    values=np.concatenate(values)
    assert np.isfinite(values).all()
    frame,daily,summary,aggregate=reconstruction_scores(data,ids,values)
    if dest is not None:
        dest.mkdir(parents=True,exist_ok=True)
        np.save(dest/'row-ids.npy',ids)
        np.save(dest/'reconstruction.npy',values)
        frame.to_parquet(dest/'rows.parquet',index=False)
        daily.to_csv(dest/'daily.csv',index=False)
        summary.to_csv(dest/'metrics.csv',index=False)
        write_json(dest/'summary.json',aggregate)
        finish(dest)
    return aggregate


def qualifies(score,baseline,cfg):
    g=cfg['selection']
    return bool(score['mae']<=baseline['mae']*(1-g['minimum_extrema_mae_improvement'])
        and score['legal']>=baseline['legal']+g['minimum_legal_fraction_improvement']
        and score['normalized_mae']<=baseline['normalized_mae']*(1+g['maximum_normalized_mae_increase'])
        and score['turnover_mae']<=baseline['turnover_mae']*(1+g['maximum_turnover_mae_increase']))


def load_decoder(path=None,device='mps'):
    model=load_tokenizer(BUNDLE,device)
    if path is not None:
        saved=torch.load(path,map_location='cpu',weights_only=True)
        assert saved['kind']=='fixed_token_decoder_adaptation_v1'
        assert saved['dataset_sha256']==file_hash(DATA/'completed.json')
        model.load_state_dict(saved['state_dict'])
    return model.eval()


def train_decoder(data,objective,baseline,cfg):
    dest=OUT/'decoder-training'/objective
    if (dest/'completed.json').exists():
        verify(dest)
        return read(dest/'summary.json')
    dest.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(cfg['seed'])
    model=load_decoder()
    params=enable_decoder_training(model)
    frozen={name:p.detach().cpu().clone() for name,p in model.named_parameters() if not p.requires_grad}
    tr=cfg['decoder_training']
    optimizer=torch.optim.AdamW(params,lr=tr['learning_rate'],weight_decay=tr['weight_decay'])
    ids=data.selected(None,'train',known=True)
    groups={day:g.row_id.to_numpy(int) for day,g in data.rows.iloc[ids].groupby('date')}
    selection=data.selected(None,'selection',cfg['decoder_selection_rows_per_date'])
    entries=[]
    visited=set()
    best=float('inf')
    best_qualified=float('inf')
    bad=0
    elapsed=0.
    if (dest/'resume.pt').exists():
        saved=torch.load(dest/'resume.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(saved['state_dict'])
        optimizer.load_state_dict(saved['optimizer'])
        entries,visited=saved['history'],set(saved['visited'])
        best,best_qualified,bad=saved['best'],saved['best_qualified'],saved['bad']
        elapsed=entries[-1]['elapsed_seconds']
        torch.set_rng_state(saved['cpu_rng'])
        torch.mps.set_rng_state(saved['mps_rng'])
    started=time.monotonic()-elapsed
    for epoch in range(len(entries)+1,tr['maximum_epochs']+1):
        if epoch>tr['minimum_epochs'] and bad>=tr['patience']:
            break
        order=epoch_sample(groups,epoch,tr['stocks_per_date_per_epoch'],cfg['seed'])
        visited.update(order.tolist())
        weights=date_weights(data.rows,order)
        model.train()
        total=0.
        count=0
        for start in range(0,len(order),tr['batch_size']):
            batch=order[start:start+tr['batch_size']]
            pairs,(future,mean,scale),known,ref=data.decoder_batch(batch,'mps')
            optimizer.zero_grad(set_to_none=True)
            decoded=model.decode(pairs,half=True)[:,-5:]
            losses=reconstruction_loss(decoded,future,known,mean,scale,ref,objective,cfg['loss'])
            loss=(losses*torch.as_tensor(weights[start:start+len(batch)],device='mps')).mean()
            assert torch.isfinite(loss)
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(params,1.)
            assert torch.isfinite(norm)
            optimizer.step()
            total+=float(loss.detach())*len(batch)
            count+=len(batch)
            if start%(tr['batch_size']*80)==0:
                progress=dict(stage='decoder_training',objective=objective,epoch=epoch,rows=count,total=len(order),loss=total/count,at=utc_now())
                write_json(dest/'progress.json',progress)
                print(progress,flush=True)
        score=reconstruct(model,data,selection)
        entry=dict(epoch=epoch,training_loss=total/count,selection=score,qualifies=qualifies(score,baseline,cfg),
            unique_rows=len(visited),examples=count,elapsed_seconds=time.monotonic()-started)
        entries.append(entry)
        payload=dict(kind='fixed_token_decoder_adaptation_v1',objective=objective,epoch=epoch,selection=score,
            dataset_sha256=file_hash(DATA/'completed.json'),state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()})
        atomic_save(payload,dest/f'epoch-{epoch:02d}.pt')
        if score['mae']<best-1e-6:
            best,bad=score['mae'],0
            atomic_save(payload,dest/'best-unguarded.pt')
        else:
            bad+=1
        if entry['qualifies'] and score['mae']<best_qualified:
            best_qualified=score['mae']
            atomic_save(payload,dest/'best-qualified.pt')
        atomic_save(dict(**payload,optimizer=optimizer.state_dict(),history=entries,visited=sorted(visited),
            best=best,best_qualified=best_qualified,bad=bad,cpu_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state()),dest/'resume.pt')
        write_json(dest/'history.json',entries)
        print('Decoder epoch',objective,entry,flush=True)
    for name,p in model.named_parameters():
        if name in frozen:
            torch.testing.assert_close(p.detach().cpu(),frozen[name],rtol=0,atol=0)
    np.save(dest/'visited-ids.npy',np.asarray(sorted(visited),dtype=np.int64))
    summary=dict(objective=objective,epochs=len(entries),unique_rows=len(visited),known_pool=len(ids),
        trainable_parameters=sum(p.numel() for p in params),frozen_parameters=sum(v.numel() for v in frozen.values()),
        frozen_parameters_unchanged=True,has_qualified_checkpoint=(dest/'best-qualified.pt').exists(),
        stopping='early_stopping' if bad>=tr['patience'] else 'budget_limited')
    write_json(dest/'summary.json',summary)
    finish(dest)
    del model,optimizer,params
    gc.collect()
    torch.mps.empty_cache()
    return summary


def decoder_stage(data,cfg):
    dest=OUT/'decoder-selection'
    if (dest/'completed.json').exists():
        verify(dest)
        return read(dest/'selection.json')
    dest.mkdir(exist_ok=True)
    ids=data.selected(None,'selection',cfg['decoder_selection_rows_per_date'])
    model=load_decoder()
    baseline=reconstruct(model,data,ids,dest/'frozen')
    del model
    choices=[]
    for objective in cfg['decoder_objectives']:
        info=train_decoder(data,objective,baseline,cfg)
        folder=OUT/'decoder-training'/objective
        name='best-qualified.pt' if info['has_qualified_checkpoint'] else 'best-unguarded.pt'
        model=load_decoder(folder/name)
        score=reconstruct(model,data,ids,dest/objective)
        choices.append(dict(objective=objective,checkpoint=str(folder/name),qualified=qualifies(score,baseline,cfg),**score))
        del model
        torch.mps.empty_cache()
    candidates=[c for c in choices if c['qualified']]
    winner=min(candidates,key=lambda x:x['mae']) if candidates else None
    choice=dict(selected=winner['objective'] if winner else 'frozen',checkpoint=winner['checkpoint'] if winner else None,
        candidates=choices,baseline=baseline,selected_at=utc_now(),selection_only=True,evaluation_used=False)
    write_json(dest/'selection.json',choice)
    finish(dest)
    print('Decoder choice',choice,flush=True)
    return choice


def reconstruction_evaluation(data,choice,cfg):
    dest=OUT/'reconstruction-evaluation'
    if (dest/'completed.json').exists():
        verify(dest)
        return
    dest.mkdir(exist_ok=True)
    ids=data.selected(None,'evaluation',cfg['reconstruction_evaluation_rows_per_date'])
    variants={'frozen':None,**{c['objective']:Path(c['checkpoint']) for c in choice['candidates']}}
    for name,path in variants.items():
        model=load_decoder(path)
        reconstruct(model,data,ids,dest/name)
        del model
        torch.mps.empty_cache()
    write_json(dest/'lineage.json',dict(selection_sha256=file_hash(OUT/'decoder-selection/selection.json'),
        selection_frozen_before_evaluation=True,variants={k:str(v) if v else None for k,v in variants.items()}))
    finish(dest)


def forecasts(data,root,choice,cfg):
    dest=root/'decoder-forecast'
    if (dest/'completed.json').exists():
        verify(dest)
        return
    dest.mkdir(exist_ok=True)
    ids=data.selected(None,'evaluation',cfg['forecast_evaluation_rows_per_date'])
    np.save(dest/'row-ids.npy',ids)
    chunks=dest/'chunks'
    chunks.mkdir(exist_ok=True)
    model,_=restore_model(root/'training/best.pt','mps')
    tokenizers={'frozen':load_decoder(),'adapted':load_decoder(Path(choice['checkpoint']))}
    size=cfg['generation_batch_rows']
    for start in range(0,len(ids),size):
        batch=ids[start:start+size]
        path=chunks/f'{start:06d}.npz'
        if path.exists():
            continue
        a,b,stamps,_=data.tensors(batch,'mps')
        pairs=generate_tokens(model,a[:,:60],b[:,:60],stamps[:,:60],stamps[:,60:],seed=cfg['seed']+start,**cfg['sampling'])
        content=dict(row_ids=batch,s1=pairs[0][:,:,-5:].cpu().numpy(),s2=pairs[1][:,:,-5:].cpu().numpy())
        for name,tokenizer in tokenizers.items():
            paths,valid=decode_paths(tokenizer,pairs,data.a['mean'][batch],data.a['scale'][batch],5)
            content[name]=paths
            content[name+'_valid']=valid
        with path.with_suffix('.tmp').open('wb') as f:
            np.savez_compressed(f,**content)
        path.with_suffix('.tmp').replace(path)
        if start%512==0:
            progress=dict(stage='forecast',run=str(root),rows=start+len(batch),total=len(ids),at=utc_now())
            write_json(dest/'progress.json',progress)
            print(progress,flush=True)
    records={}
    coverage=[]
    for name in tokenizers:
        paths=np.lib.format.open_memmap(dest/f'{name}-paths.npy',mode='w+',dtype=np.float32,
            shape=(len(ids),cfg['sampling']['samples'],5,6))
        for start in range(0,len(ids),size):
            with np.load(chunks/f'{start:06d}.npz') as z:
                np.testing.assert_array_equal(z['row_ids'],ids[start:start+size])
                paths[start:start+len(z['row_ids'])]=z[name]
        paths.flush()
        frame,cov=score_paths(paths,data.a['future'][ids],data.a['valid'][ids],data.a['last'][ids,3],
            data.rows.iloc[ids].date.to_numpy(),minimum_fraction=cfg['minimum_valid_fraction'])
        records[name]=frame
        coverage.append(cov.assign(variant=name))
    common,daily,metrics=common_scores(records)
    common.to_parquet(dest/'common.parquet',index=False)
    daily.to_csv(dest/'daily.csv',index=False)
    metrics.to_csv(dest/'metrics.csv',index=False)
    pd.concat(coverage).to_csv(dest/'coverage.csv',index=False)
    a,b,stamps,_=data.tensors(ids[:size],'mps')
    pairs=generate_tokens(model,a[:,:60],b[:,:60],stamps[:,:60],stamps[:,60:],seed=cfg['seed'],**cfg['sampling'])
    with np.load(chunks/'000000.npz') as z:
        for key,v in zip(['s1','s2'],pairs):
            np.testing.assert_array_equal(z[key],v[:,:,-5:].cpu().numpy())
        for name,tokenizer in tokenizers.items():
            p,v=decode_paths(tokenizer,pairs,data.a['mean'][ids[:size]],data.a['scale'][ids[:size]],5)
            np.testing.assert_array_equal(z[name],p)
            np.testing.assert_array_equal(z[name+'_valid'],v)
    write_json(dest/'verification.json',dict(passed=True,shared_generated_tokens=True,first_batch_replay_exact=True,
        predictor_sha256=file_hash(root/'training/best.pt'),decoder_sha256=file_hash(Path(choice['checkpoint'])),
        selection_sha256=file_hash(OUT/'decoder-selection/selection.json'),chunks={p.name:file_hash(p) for p in chunks.iterdir()}))
    finish(dest)
    del model,tokenizers
    gc.collect()
    torch.mps.empty_cache()


def aggregate(cfg):
    dest=OUT/'forecast-results'
    dest.mkdir(exist_ok=True)
    frames=[pd.read_csv(OUT/f'predictors/seed{s}/dense_3720k_equal/decoder-forecast/daily.csv').assign(seed=s) for s in cfg['forecast_seeds']]
    daily=pd.concat(frames,ignore_index=True)
    daily.to_csv(dest/'daily.csv',index=False)
    daily.groupby(['seed','variant','horizon','target']).mean(numeric_only=True).reset_index().to_csv(dest/'metrics.csv',index=False)
    rows=[]
    for h in [2,5]:
        for metric in ['crps','mae','coverage80','interval_score80']:
            paired=daily[daily.horizon==h].groupby(['seed','date','variant'])[metric].mean().unstack('variant')
            for seed in cfg['forecast_seeds']+['mean']:
                p=paired if seed=='mean' else paired.loc[[seed]]
                p=p.groupby('date')[['frozen','adapted']].mean()
                x=p.adapted.to_numpy()
                b=p.frozen.to_numpy()
                rng=np.random.default_rng(314159)
                indices=(rng.integers(0,len(x),size=(cfg['bootstrap_replicates'],int(np.ceil(len(x)/10)),1))+np.arange(10))%len(x)
                indices=indices.reshape(len(indices),-1)[:,:len(x)]
                lo,hi=np.quantile((x-b)[indices].mean(1),[.025,.975])
                rows.append(dict(seed=seed,horizon=h,metric=metric,baseline=float(b.mean()),adapted=float(x.mean()),
                    delta=float((x-b).mean()),relative_change=float(x.mean()/b.mean()-1),ci_low=float(lo),ci_high=float(hi),dates=len(x)))
    pd.DataFrame(rows).to_csv(dest/'paired.csv',index=False)
    finish(dest)


def main():
    global OUT, DATA, BUNDLE
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['init','all'])
    parser.add_argument('--config',type=Path,default=BASE/'configs/tokenizer-reconstruction-v1.json')
    parser.add_argument('--data',type=Path,default=DATA)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--bundle',type=Path,default=BUNDLE)
    args=parser.parse_args()
    OUT=args.output.resolve()
    DATA=args.data.resolve()
    BUNDLE=args.bundle.resolve()
    history.OUT=OUT/'predictors'
    history.DATA=DATA
    history.BUNDLE=BUNDLE
    initialize(args.config.resolve())
    if args.stage=='init':
        return
    cfg=read(OUT/'protocol.json')
    for p,h in read(OUT/'source-manifest.json')['files'].items():
        assert file_hash(OUT/p)==h,p
    torch.set_num_threads(4)
    data=ReconstructionData(cfg)
    choice=decoder_stage(data,cfg)
    reconstruction_evaluation(data,choice,cfg)
    if choice['selected']!='frozen':
        for seed in cfg['forecast_seeds']:
            data.cfg['seed']=seed
            root=history.train(data,f'seed{seed}',cfg['forecast_model'],False)
            forecasts(data,root,choice,cfg)
        aggregate(cfg)
    write_json(OUT/'run-completed.json',dict(passed=True,at=utc_now(),selected=choice['selected'],
        forecast_stage_run=choice['selected']!='frozen',executable=False,sealed_holdout_opened=False))
    write_json(OUT/'completed.json',dict(passed=True,files={
        str(path.relative_to(OUT)):file_hash(path)
        for path in OUT.rglob('*')
        if path.is_file() and path.name not in ['completed.json','progress.json']
    }))


if __name__=='__main__':
    main()
