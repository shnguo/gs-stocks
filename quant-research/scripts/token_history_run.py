"""Run frozen chronological dense baselines, then the authorized horizon-loss test."""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from quant_research.storage import file_hash, utc_now, write_json
from quant_research.token_history import date_weights, epoch_sample, loss_parts
from quant_research.token_transformer import (
    TokenConfig,
    TokenTransformer,
    checkpoint_payload,
    decode_paths,
    generate_tokens,
    load_tokenizer,
    restore_model,
)

BASE=Path('/Users/guo/Documents/stocks/quant-research')
ART=BASE/'artifacts'
DATA=ART/'token-history-data-20260915-v1'
OUT=ART/'token-history-experiment-20260915-v1'
BUNDLE=ART/'kronos-comparison-20260910-v1'


def read(path):
    return json.loads(path.read_text())


def atomic_save(payload,path):
    temp=path.with_suffix('.tmp')
    torch.save(payload,temp)
    temp.replace(path)


def complete(folder):
    write_json(folder/'completed.json',dict(passed=True,files={
        p.name:file_hash(p) for p in folder.iterdir()
        if p.is_file() and p.name not in ['completed.json','progress.json']}))


def verify(folder):
    meta=read(folder/'completed.json')
    assert meta['passed']
    for name,expected in meta['files'].items():
        assert file_hash(folder/name)==expected,(folder,name)


class Dataset:
    def __init__(self):
        self.cfg=read(DATA/'protocol.json')
        self.rows=pd.read_parquet(DATA/'rows.parquet')
        self.stamps=np.load(DATA/'calendar-stamps.npy')
        self.a={k:np.load(DATA/f'{k}.npy',mmap_mode='r') for k in
                ['s1','s2','valid','future','mean','scale','last']}

    def tensors(self,ids,device):
        t=self.rows.iloc[ids].date_index.to_numpy(int)
        stamps=self.stamps[t[:,None]+np.arange(-59,6)]
        return [torch.as_tensor(self.a[k][ids].astype(np.int64),device=device) for k in ['s1','s2']]+[
            torch.as_tensor(stamps,device=device),torch.as_tensor(self.a['valid'][ids],device=device)]

    def selected(self,fold,part,count=None,known=False):
        ids=np.load(DATA/f'{fold}-{part}.npy')
        if count is not None:
            # Selection is made before examining label availability.
            ids=self.rows.iloc[ids].groupby('date',sort=True).head(count).row_id.to_numpy(int)
        if known:
            ids=ids[self.a['valid'][ids].any(1)]
        return ids


@torch.inference_mode()
def selection_score(model,data,ids):
    model.eval()
    values=[]
    for start in range(0,len(ids),data.cfg['batch_size']):
        batch=ids[start:start+data.cfg['batch_size']]
        parts=loss_parts(model,*data.tensors(batch,data.cfg['device']),60)
        values.extend(parts.mean(1).cpu().numpy())
    frame=pd.DataFrame(dict(date=data.rows.iloc[ids].date.to_numpy(),ce=values))
    return float(frame.groupby('date').ce.mean().mean())


def train(data,fold,name,weighted):
    cfg=data.cfg
    root=OUT/fold/(name+('_weighted' if weighted else '_equal'))
    dest=root/'training'
    if (dest/'completed.json').exists():
        verify(dest)
        return root
    dest.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(cfg['seed'])
    model=TokenTransformer(TokenConfig(**cfg['models'][name])).to(cfg['device'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    ids=data.selected(fold,'train',known=True)
    groups={day:g.row_id.to_numpy(int) for day,g in data.rows.iloc[ids].groupby('date')}
    selected=data.selected(fold,'selection',cfg['selection_per_date'],True)
    weights=cfg['loss_experiment_after_dense']['horizon_weights'] if weighted else None
    best,bad,history,visited=float('inf'),0,[],set()
    elapsed=0.
    initial=selection_score(model,data,selected)
    if (dest/'resume.pt').exists():
        saved=torch.load(dest/'resume.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(saved['state_dict'])
        optimizer.load_state_dict(saved['optimizer'])
        history,best,bad,initial=saved['history'],saved['best'],saved['bad'],saved['initial']
        visited=set(saved['visited'])
        elapsed=history[-1]['elapsed_seconds']
        torch.set_rng_state(saved['cpu_rng'])
        torch.mps.set_rng_state(saved['mps_rng'])
    started=time.monotonic()-elapsed
    print('Training',fold,name,'weighted' if weighted else 'equal','dates',len(groups),'known pool',len(ids),flush=True)
    for epoch in range(len(history)+1,cfg['max_epochs']+1):
        if epoch>cfg['min_epochs'] and bad>=cfg['patience']:
            break
        order=epoch_sample(groups,epoch,cfg['train_per_date_per_epoch'],cfg['seed'])
        visited.update(order.tolist())
        dw=date_weights(data.rows,order)
        model.train()
        total,count=0.,0
        for start in range(0,len(order),cfg['batch_size']):
            batch=order[start:start+cfg['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            parts=loss_parts(model,*data.tensors(batch,cfg['device']),60,weights)
            loss=(parts.mean(1)*torch.as_tensor(dw[start:start+len(batch)],device=cfg['device'])).mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):
                raise ValueError('Nonfinite gradients')
            optimizer.step()
            total+=float(loss.detach())*len(batch)
            count+=len(batch)
            if start%(cfg['batch_size']*80)==0:
                progress=dict(stage='training',fold=fold,model=name,weighted=weighted,epoch=epoch,
                    processed=count,total=len(order),loss=total/count,elapsed_seconds=time.monotonic()-started,at=utc_now())
                write_json(dest/'progress.json',progress)
                print(progress,flush=True)
        ce=selection_score(model,data,selected)
        entry=dict(epoch=epoch,training_objective=total/count,selection_ce=ce,
            examples=count,unique_examples_visited=len(visited),elapsed_seconds=time.monotonic()-started)
        history.append(entry)
        payload=checkpoint_payload(model,epoch=epoch,selection_ce=ce,horizon_weights=weights,
            dataset_manifest_sha256=file_hash(DATA/'completed.json'),fold=fold)
        # Save CPU tensors for portable and independently checkable checkpoints.
        payload['state_dict']={k:v.detach().cpu() for k,v in model.state_dict().items()}
        atomic_save(payload,dest/'last.pt')
        if ce<best-cfg['min_delta']:
            best,bad=ce,0
            atomic_save(payload,dest/'best.pt')
        else:
            bad+=1
        atomic_save(dict(**payload,optimizer=optimizer.state_dict(),history=history,best=best,bad=bad,
            initial=initial,visited=sorted(visited),cpu_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state()),dest/'resume.pt')
        write_json(dest/'history.json',history)
        print(fold,name,'weighted' if weighted else 'equal',entry,flush=True)
    model=model.cpu()
    del model,optimizer
    torch.mps.empty_cache()
    model,saved=restore_model(dest/'best.pt','cpu')
    model.eval()
    probe=selected[:2]
    with torch.inference_mode():
        a,b,stamps,_=data.tensors(probe,'cpu')
        logits=model.forecast_logits(a[:,:-1],b[:,:-1],stamps[:,:-1],a[:,1:],59)
    np.savez_compressed(dest/'reload-reference.npz',ids=probe,coarse=logits[0].numpy(),fine=logits[1].numpy())
    np.save(dest/'visited-row-ids.npy',np.asarray(sorted(visited),np.int64))
    write_json(dest/'summary.json',dict(fold=fold,model=name,horizon_weights=weights,
        parameters=sum(p.numel() for p in model.parameters()),known_training_pool=len(ids),training_dates=len(groups),
        unique_examples_visited=len(visited),total_examples=sum(x['examples'] for x in history),
        selected_epoch=saved['epoch'],epochs=len(history),initial_selection_ce=initial,best_selection_ce=best,
        selection_rows=len(selected),selection_dates=int(data.rows.iloc[selected].date.nunique()),
        stopping='early_stopping' if bad>=cfg['patience'] else 'budget_limited',
        elapsed_seconds=history[-1]['elapsed_seconds'],executable=False))
    complete(dest)
    return root


def forecast(data,fold,root):
    cfg=data.cfg
    dest=root/'forecast'
    if (dest/'completed.json').exists():
        verify(dest)
        return
    dest.mkdir(exist_ok=True)
    ids=data.selected(fold,'evaluation',cfg['path_per_date'])
    np.save(dest/'row-ids.npy',ids)
    shape=(len(ids),cfg['path_samples'],5,6)
    model,_=restore_model(root/'training/best.pt',cfg['device'])
    model.eval()
    tokenizer=load_tokenizer(BUNDLE,cfg['device'])
    chunks=dest/'chunks'
    chunks.mkdir(exist_ok=True)
    started=time.monotonic()
    # Batch boundaries and RNG seeds are identical for every objective/model.
    for start in range(0,len(ids),8):
        end=min(start+8,len(ids))
        path=chunks/f'{start:06d}.npz'
        if path.exists():
            with np.load(path) as z:
                np.testing.assert_array_equal(z['row_ids'],ids[start:end])
            continue
        a,b,stamps,_=data.tensors(ids[start:end],cfg['device'])
        pairs=generate_tokens(model,a[:,:60],b[:,:60],stamps[:,:60],stamps[:,60:],
            samples=cfg['path_samples'],seed=cfg['seed']+start,temperature=cfg['temperature'],top_p=cfg['top_p'],top_k=cfg['top_k'])
        paths,valid=decode_paths(tokenizer,pairs,data.a['mean'][ids[start:end]],data.a['scale'][ids[start:end]],5)
        temp=path.with_suffix('.tmp')
        with temp.open('wb') as f:
            np.savez_compressed(f,row_ids=ids[start:end],paths=paths,valid_paths=valid,
                s1=pairs[0][:,:,-5:].cpu().numpy(),s2=pairs[1][:,:,-5:].cpu().numpy())
        temp.replace(path)
        if start%256==0:
            progress=dict(stage='forecast',run=str(root),rows=end,total=len(ids),elapsed_seconds=time.monotonic()-started,at=utc_now())
            write_json(dest/'progress.json',progress)
            print(progress,flush=True)
    output=np.lib.format.open_memmap(dest/'paths.npy',mode='w+',dtype=np.float32,shape=shape)
    for start in range(0,len(ids),8):
        with np.load(chunks/f'{start:06d}.npz') as z:
            output[start:start+len(z['row_ids'])]=z['paths']
    output.flush()
    # Replay first complete batch with a fresh checkpoint and the same local seed.
    del model
    torch.mps.empty_cache()
    model,_=restore_model(root/'training/best.pt',cfg['device'])
    a,b,stamps,_=data.tensors(ids[:8],cfg['device'])
    pairs=generate_tokens(model,a[:,:60],b[:,:60],stamps[:,:60],stamps[:,60:],samples=cfg['path_samples'],
        seed=cfg['seed'],temperature=cfg['temperature'],top_p=cfg['top_p'],top_k=cfg['top_k'])
    paths,valid=decode_paths(tokenizer,pairs,data.a['mean'][ids[:8]],data.a['scale'][ids[:8]],5)
    with np.load(chunks/'000000.npz') as z:
        for k,x in [('paths',paths),('valid_paths',valid),('s1',pairs[0][:,:,-5:].cpu().numpy()),('s2',pairs[1][:,:,-5:].cpu().numpy())]:
            np.testing.assert_array_equal(z[k],x)
    write_json(dest/'verification.json',dict(passed=True,fixed_seed_replay_rows=8,
        checkpoint_sha256=file_hash(root/'training/best.pt'),chunks={p.name:file_hash(p) for p in chunks.iterdir()}))
    write_json(dest/'summary.json',dict(rows=len(ids),dates=int(data.rows.iloc[ids].date.nunique()),
        shape=list(shape),fully_known_labels=int(data.a['valid'][ids].all(1).sum()),elapsed_seconds=time.monotonic()-started))
    complete(dest)
    del tokenizer,model
    gc.collect()
    torch.mps.empty_cache()


def main():
    global DATA, OUT, BUNDLE
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['all','dense','weighted'],default='all')
    parser.add_argument('--data',type=Path,default=DATA)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--bundle',type=Path,default=BUNDLE)
    args=parser.parse_args()
    DATA=args.data.resolve()
    OUT=args.output.resolve()
    BUNDLE=args.bundle.resolve()
    verify(DATA)
    OUT.mkdir(exist_ok=True)
    data=Dataset()
    torch.set_num_threads(4)
    import evaluate_token_history as evaluation
    evaluation.DATA=DATA
    evaluation.OUT=OUT
    if args.stage in ['all','dense']:
        for fold in data.cfg['folds']:
            for name in data.cfg['models']:
                root=train(data,fold['name'],name,False)
                forecast(data,fold['name'],root)
        evaluation.evaluate('dense')
    if args.stage in ['all','weighted']:
        # This is deliberately a hard gate: finish baseline validation first.
        verify(OUT/'dense-evaluation')
        for fold in data.cfg['folds']:
            root=train(data,fold['name'],data.cfg['loss_experiment_after_dense']['model'],True)
            forecast(data,fold['name'],root)
        evaluation.evaluate('weighted')
    write_json(OUT/f'{args.stage}-completed.json',dict(passed=True,at=utc_now()))
    write_json(OUT/'completed.json',dict(passed=True,files={
        str(path.relative_to(OUT)):file_hash(path)
        for path in OUT.rglob('*')
        if path.is_file() and path.name not in ['completed.json','progress.json']
    }))


if __name__=='__main__':
    main()
