"""Three-seed chronological confirmation with independently selected price checkpoints."""
import argparse
import gc
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from token_history_run import DATA, Dataset, atomic_save, complete, selection_score
from token_pipeline_audit import OUT as AUDIT
from token_pipeline_audit import finish, generate, read, verify

from quant_research.forecast_audit import choose_sampler, common_scores, score_paths
from quant_research.storage import file_hash, utc_now, write_json
from quant_research.token_history import date_weights, epoch_sample, loss_parts, partition_indices
from quant_research.token_transformer import (
    TokenConfig,
    TokenTransformer,
    checkpoint_payload,
    restore_model,
)

BASE=Path('/Users/guo/Documents/stocks/quant-research')
OUT=BASE/'artifacts/token-pipeline-confirmation-20260915-v1'


def initialize():
    audit=read(AUDIT/'protocol.json')
    cfg=dict(audit, protocol_id='token-pipeline-confirmation-v1',
        checkpoint_selection_rule=audit['selection'], checkpoint_selection_samples=32,
        ce_selection_rows_per_date=64, seed_interpretation='Training seed varies; evaluation stocks and generation RNG are fixed across seeds',
        sampler_selection_rule='Choose among the four prespecified samplers using seed 17 and 2023 H1 validation only; freeze before 2023 H2 evaluation; later-window audit does not select this setting',
        evaluation_rule='Compare CE checkpoint/baseline sampler with one change at a time; report all seeds; no evaluation-based choice')
    if (OUT/'protocol.json').exists():
        assert cfg==read(OUT/'protocol.json')
        return
    OUT.mkdir()
    write_json(OUT/'protocol.json',cfg)
    code=OUT/'code'
    shutil.copytree(AUDIT/'code/src',code/'src')
    (code/'scripts').mkdir()
    for name in ['token_history_run.py','token_pipeline_audit.py','compare_token_range.py','compare_token_kronos_paths.py']:
        shutil.copy2(AUDIT/'code/scripts'/name,code/'scripts'/name)
    shutil.copy2(BASE/'scripts/token_pipeline_confirmation.py',code/'scripts/token_pipeline_confirmation.py')
    write_json(OUT/'source-manifest.json',dict(audit_protocol=file_hash(AUDIT/'protocol.json'),
        dataset=file_hash(DATA/'completed.json'),files={str(p.relative_to(OUT)):file_hash(p) for p in code.rglob('*.py')}))


class ConfirmationData(Dataset):
    def __init__(self, seed, cfg):
        super().__init__()
        self.cfg.update(seed=seed, selection_per_date=cfg['ce_selection_rows_per_date'],
            min_epochs=cfg['training']['minimum_epochs'], max_epochs=cfg['training']['maximum_epochs'],
            patience=cfg['training']['patience'], train_per_date_per_epoch=cfg['training']['stocks_per_date_per_epoch'],
            batch_size=cfg['training']['batch_size'], learning_rate=cfg['training']['learning_rate'],
            weight_decay=cfg['training']['weight_decay'])
        dates=np.asarray(read(BASE/'artifacts/kronos-inputs-20260914-v3/manifest.json')['dates'])
        self.splits={part:partition_indices(self.rows,dates,*cfg['new_fold'][part],5)
                     for part in ['train','selection','evaluation']}
        for part,ids in self.splits.items():
            r=self.rows.iloc[ids]
            assert r.date.min()>=cfg['new_fold'][part][0]
            assert r.label_end.max()<cfg['new_fold'][part][1]
            assert r.label_end.max()<cfg['sealed_holdout_start']
            path=OUT/f'{part}-ids.npy'
            if path.exists():
                np.testing.assert_array_equal(np.load(path),ids)
            else:
                np.save(path,ids)

    def selected(self,fold,part,count=None,known=False):
        ids=self.splits[part]
        if count is not None:
            ids=self.rows.iloc[ids].groupby('date',sort=True).head(count).row_id.to_numpy(int)
        if known:
            ids=ids[self.a['valid'][ids].any(1)]
        return ids


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
        atomic_save(payload,dest/f'epoch-{epoch:02d}.pt')
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


def score(data,ids,paths,cfg):
    return score_paths(paths,data.a['future'][ids],data.a['valid'][ids],data.a['last'][ids,3],
        data.rows.iloc[ids].date.to_numpy(),cfg['horizons'],cfg['minimum_valid_fraction'])


def choose_checkpoint(data,root,cfg):
    dest=root/'checkpoint-selection'
    if (dest/'completed.json').exists():
        verify(dest)
        return read(dest/'selection.json')
    dest.mkdir(exist_ok=True)
    summary=read(root/'training/summary.json')
    epochs=sorted(set(range(2,summary['epochs']+1,2))|{summary['selected_epoch']})
    ids=data.selected(None,'selection',cfg['checkpoint_selection_rows_per_date'])
    records,coverage={},[]
    for epoch in epochs:
        name='baseline' if epoch==summary['selected_epoch'] else f'epoch_{epoch:02d}'
        paths=generate(data,root/f'training/epoch-{epoch:02d}.pt',ids,dest/name,
            cfg['samplers']['baseline'],cfg,samples=cfg['checkpoint_selection_samples'])
        frame,cov=score(data,ids,paths,cfg)
        records[name]=frame
        coverage.append(cov.assign(variant=name))
    common,daily,metrics=common_scores(records)
    common.to_parquet(dest/'common.parquet',index=False)
    daily.to_csv(dest/'daily.csv',index=False)
    metrics=metrics.assign(fold=cfg['new_fold']['name'],partition='selection')
    cov=pd.concat(coverage).assign(fold=cfg['new_fold']['name'],partition='selection')
    metrics.to_csv(dest/'metrics.csv',index=False)
    cov.to_csv(dest/'coverage.csv',index=False)
    choice=choose_sampler(metrics,cov,cfg['checkpoint_selection_rule'])
    epoch=summary['selected_epoch'] if choice['selected']=='baseline' else int(choice['selected'].split('_')[1])
    choice.update(selected_epoch=epoch,ce_epoch=summary['selected_epoch'],selected_at=utc_now(),
        partition='selection',evaluation_used=False,checkpoint_sha256=file_hash(root/f'training/epoch-{epoch:02d}.pt'))
    write_json(dest/'selection.json',choice)
    finish(dest)
    print('Checkpoint selection',root,choice,flush=True)
    return choice


def choose_confirmation_sampler(data,root,cfg):
    dest=OUT/'sampler-selection'
    if (dest/'completed.json').exists():
        verify(dest)
        return read(dest/'selection.json')
    assert data.cfg['seed']==cfg['confirmation_seeds'][0]
    dest.mkdir(exist_ok=True)
    ids=data.selected(None,'selection',cfg['sampling_selection_rows_per_date'])
    records,coverage={},[]
    for name,settings in cfg['samplers'].items():
        paths=generate(data,root/'training/best.pt',ids,dest/name,settings,cfg)
        frame,cov=score(data,ids,paths,cfg)
        records[name]=frame
        coverage.append(cov.assign(variant=name))
    common,daily,metrics=common_scores(records)
    common.to_parquet(dest/'common.parquet',index=False)
    daily.to_csv(dest/'daily.csv',index=False)
    metrics=metrics.assign(fold=cfg['new_fold']['name'],partition='selection')
    cov=pd.concat(coverage).assign(fold=cfg['new_fold']['name'],partition='selection')
    metrics.to_csv(dest/'metrics.csv',index=False)
    cov.to_csv(dest/'coverage.csv',index=False)
    choice=choose_sampler(metrics,cov,cfg['selection'])
    choice.update(partition='selection',selected_at=utc_now(),evaluation_used=False,
        settings=cfg['samplers'][choice['selected']],dates=[str(data.rows.iloc[ids].date.min()),str(data.rows.iloc[ids].date.max())],
        checkpoint_sha256=file_hash(root/'training/best.pt'),training_seed=int(data.cfg['seed']))
    assert choice['dates'][1]<cfg['new_fold']['evaluation'][0]
    write_json(dest/'selection.json',choice)
    finish(dest)
    print('Chronological sampler selection',choice,flush=True)
    return choice


def evaluate(data,root,choice,sampler,cfg):
    dest=root/'evaluation'
    if (dest/'completed.json').exists():
        verify(dest)
        return
    dest.mkdir(exist_ok=True)
    ids=data.selected(None,'evaluation',cfg['confirmation_evaluation_rows_per_date'])
    np.save(dest/'row-ids.npy',ids)
    ce=root/'training/best.pt'
    cp=root/f"training/epoch-{choice['selected_epoch']:02d}.pt"
    variants={'baseline':(ce,cfg['samplers']['baseline']),
              'sampling':(ce,sampler['settings']), 'price_checkpoint':(cp,cfg['samplers']['baseline'])}
    records,coverage,cache={},[],{}
    lineage={}
    for name,(checkpoint,settings) in variants.items():
        # Same payload may have different serialization hashes; compare selected epochs for reuse.
        epoch=choice['selected_epoch'] if name=='price_checkpoint' else choice['ce_epoch']
        key=(epoch,json.dumps(settings,sort_keys=True))
        if key not in cache:
            cache[key]=(name,generate(data,checkpoint,ids,dest/name,settings,cfg))
        source,paths=cache[key]
        frame,cov=score(data,ids,paths,cfg)
        records[name]=frame
        coverage.append(cov.assign(variant=name))
        lineage[name]=dict(path_variant=source,checkpoint=str(checkpoint),epoch=epoch,settings=settings)
    common,daily,metrics=common_scores(records)
    common.to_parquet(dest/'common.parquet',index=False)
    daily.to_csv(dest/'daily.csv',index=False)
    metrics.to_csv(dest/'metrics.csv',index=False)
    pd.concat(coverage).to_csv(dest/'coverage.csv',index=False)
    write_json(dest/'lineage.json',lineage)
    write_json(dest/'selection-proof.json',dict(checkpoint_selection=file_hash(root/'checkpoint-selection/selection.json'),
        sampler_selection=file_hash(OUT/'sampler-selection/selection.json'),choices_frozen_before_evaluation=True))
    finish(dest)
    print('Evaluation complete',root,flush=True)


def aggregate(cfg):
    dest=OUT/'results'
    dest.mkdir(exist_ok=True)
    frames=[]
    for seed in cfg['confirmation_seeds']:
        root=OUT/f'seed{seed}'/(cfg['confirmation_model']+'_equal')
        verify(root/'evaluation')
        frames.append(pd.read_csv(root/'evaluation/daily.csv').assign(seed=seed))
    daily=pd.concat(frames,ignore_index=True)
    daily.to_csv(dest/'daily.csv',index=False)
    metrics=daily.groupby(['seed','variant','horizon','target']).mean(numeric_only=True).reset_index()
    metrics.to_csv(dest/'metrics.csv',index=False)
    rows=[]
    # Resample shared calendar blocks jointly across seeds; seeds are reported separately.
    for horizon in cfg['horizons']:
        for variant in ['sampling','price_checkpoint']:
            for metric in ['crps','mae','coverage80','interval_score80']:
                m=daily[daily.horizon==horizon].groupby(['seed','date','variant'])[metric].mean().unstack('variant')
                for seed in cfg['confirmation_seeds']+['mean']:
                    paired=m if seed=='mean' else m.loc[[seed]]
                    bydate=paired.groupby('date')[[variant,'baseline']].mean()
                    x=bydate[variant].to_numpy()
                    b=bydate.baseline.to_numpy()
                    rng=np.random.default_rng(314159)
                    indices=(rng.integers(0,len(x),size=(cfg['bootstrap_replicates'],int(np.ceil(len(x)/cfg['bootstrap_block_dates'])),1))
                        +np.arange(cfg['bootstrap_block_dates']))%len(x)
                    indices=indices.reshape(len(indices),-1)[:,:len(x)]
                    delta=(x-b)[indices].mean(1)
                    lo,hi=np.quantile(delta,[.025,.975])
                    rows.append(dict(seed=seed,horizon=horizon,variant=variant,metric=metric,
                        baseline=float(b.mean()),candidate=float(x.mean()),delta=float((x-b).mean()),
                        relative_change=float(x.mean()/b.mean()-1) if b.mean() else None,
                        ci_low=float(lo),ci_high=float(hi),dates=len(x)))
    pd.DataFrame(rows).to_csv(dest/'paired-comparisons.csv',index=False)
    write_json(dest/'interpretation.json',dict(evaluation_used_for_selection=False,
        conditional_inference='Calendar-block bootstrap conditional on these three fitted models and selected stocks; not population uncertainty over all training seeds',
        executable=False,sealed_holdout_opened=False))
    finish(dest)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['init','all'])
    args=p.parse_args()
    initialize()
    if args.stage=='init':
        return
    cfg=read(OUT/'protocol.json')
    for path,h in read(OUT/'source-manifest.json')['files'].items():
        assert file_hash(OUT/path)==h,path
    verify(AUDIT/'sampling')
    sampler=None
    torch.set_num_threads(4)
    for seed in cfg['confirmation_seeds']:
        data=ConfirmationData(seed,cfg)
        root=train(data,f'seed{seed}',cfg['confirmation_model'],False)
        choice=choose_checkpoint(data,root,cfg)
        if sampler is None:
            sampler=choose_confirmation_sampler(data,root,cfg)
        evaluate(data,root,choice,sampler,cfg)
        del data
        gc.collect()
    aggregate(cfg)
    write_json(OUT/'run-completed.json',dict(passed=True,at=utc_now()))


if __name__=='__main__':
    main()
