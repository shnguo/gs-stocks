"""Frozen tokenizer diagnostics and validation-only sampling experiments."""
import argparse
import gc
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from token_history_run import BUNDLE, DATA, Dataset

from quant_research.forecast_audit import (
    TARGETS,
    choose_sampler,
    common_scores,
    extrema,
    score_paths,
)
from quant_research.storage import file_hash, utc_now, write_json
from quant_research.token_transformer import (
    decode_paths,
    generate_tokens,
    load_tokenizer,
    normalized_history,
    restore_model,
    valid_bars,
)

BASE=Path('/Users/guo/Documents/stocks/quant-research')
PRIOR=BASE/'artifacts/token-history-experiment-20260915-v1'
OUT=BASE/'artifacts/token-pipeline-audit-20260915-v1'


def read(p):
    return json.loads(p.read_text())


def finish(folder):
    write_json(folder/'completed.json',dict(passed=True,at=utc_now(),files={p.name:file_hash(p)
        for p in folder.iterdir() if p.is_file() and p.name not in ['completed.json','progress.json']}))


def verify(folder):
    meta=read(folder/'completed.json')
    assert meta['passed']
    for name,h in meta['files'].items():
        assert file_hash(folder/name)==h,(folder,name)


def initialize():
    if (OUT/'protocol.json').exists():
        assert read(OUT/'protocol.json')==read(BASE/'configs/token-pipeline-audit-v1.json')
        return
    for p,h in read(PRIOR/'final-verification.json')['files'].items():
        assert file_hash(Path(p))==h,p
    OUT.mkdir()
    write_json(OUT/'protocol.json',read(BASE/'configs/token-pipeline-audit-v1.json'))
    code=OUT/'code'
    shutil.copytree(BASE/'src',code/'src',ignore=shutil.ignore_patterns('__pycache__'))
    (code/'scripts').mkdir()
    for name in ['token_pipeline_audit.py','token_history_run.py','compare_token_range.py','compare_token_kronos_paths.py']:
        shutil.copy2(BASE/'scripts'/name,code/'scripts'/name)
    write_json(OUT/'source-manifest.json',dict(prior=file_hash(PRIOR/'final-verification.json'),
        dataset=file_hash(DATA/'completed.json'),files={str(p.relative_to(OUT)):file_hash(p) for p in code.rglob('*.py')}))


def reconstruction(data,cfg):
    dest=OUT/'reconstruction'
    if (dest/'completed.json').exists():
        verify(dest)
        return
    dest.mkdir(exist_ok=True)
    tokenizer=load_tokenizer(BUNDLE,cfg['device'])
    raw=np.load(BASE/'artifacts/kronos-inputs-20260914-v3/values.npy',mmap_mode='r')
    records,clipping,fields=[],[],[]
    for fold in cfg['audit_folds']:
        for part in ['selection','evaluation']:
            ids=data.selected(fold,part,cfg['audit_rows_per_date'])
            reconstructed={k:[] for k in ['clip_only','tokenizer_clipped','tokenizer_unclipped_future']}
            price_clipped=[]
            turnover_clipped=[]
            for start in range(0,len(ids),512):
                batch=ids[start:start+512]
                rr=data.rows.iloc[batch]
                stock,t=rr[['stock_index','date_index']].to_numpy(int).T
                history,mean,scale=normalized_history(raw[stock[:,None],t[:,None]+np.arange(-59,1)].copy())
                np.testing.assert_array_equal(mean,data.a['mean'][batch])
                np.testing.assert_array_equal(scale,data.a['scale'][batch])
                valid=data.a['valid'][batch]
                future=data.a['future'][batch]
                values=np.where(valid[...,None],future,mean).astype(np.float32)
                z=(values-mean)/scale
                clipped=np.clip(z,-cfg['clip'],cfg['clip'])
                reconstructed['clip_only'].append(clipped*scale+mean)
                price_clipped.append((np.abs(z[...,:4])>cfg['clip']).any(-1)&valid)
                turnover_clipped.append((np.abs(z[...,4:])>cfg['clip']).any(-1)&valid)
                a,b=[torch.as_tensor(data.a[k][batch].astype(np.int64),device=cfg['device']) for k in ['s1','s2']]
                with torch.inference_mode():
                    decoded=tokenizer.decode([a,b],half=True).cpu().numpy()[:,-5:]*scale+mean
                    unclipped=tokenizer.encode(torch.from_numpy(np.concatenate([history,z],1)).to(cfg['device']),half=True)
                    decoded_unclipped=tokenizer.decode(unclipped,half=True).cpu().numpy()[:,-5:]*scale+mean
                    if start==0:
                        check=tokenizer.encode(torch.from_numpy(np.concatenate([history,clipped],1)[:8]).to(cfg['device']),half=True)
                        for expected,got in zip([a,b],check):
                            np.testing.assert_array_equal(expected[:8].cpu().numpy(),got.cpu().numpy())
                assert np.isfinite(decoded).all() and np.isfinite(decoded_unclipped).all()
                reconstructed['tokenizer_clipped'].append(decoded)
                reconstructed['tokenizer_unclipped_future'].append(decoded_unclipped)
            reconstructed={k:np.concatenate(v) for k,v in reconstructed.items()}
            price_clipped=np.concatenate(price_clipped)
            turnover_clipped=np.concatenate(turnover_clipped)
            np.savez_compressed(dest/f'{fold}-{part}.npz',row_ids=ids,price_clipped=price_clipped,
                turnover_clipped=turnover_clipped,**reconstructed)
            rows=data.rows.iloc[ids]
            reference=data.a['last'][ids,3].astype(float)
            for h in cfg['horizons']:
                known=data.a['valid'][ids,:h].all(1)
                truth=extrema(data.a['future'][ids,:h].astype(float),reference)
                for stage,values in reconstructed.items():
                    predicted=extrema(values[:,:h].astype(float),reference)
                    legal=valid_bars(values[:,:h]).all(1)
                    for i in np.flatnonzero(known):
                        for j,target in enumerate(TARGETS):
                            records.append(dict(fold=fold,partition=part,date=rows.iloc[i].date,row_id=int(ids[i]),
                                horizon=h,stage=stage,target=target,mae=abs(predicted[i,j]-truth[i,j]),
                                bias=predicted[i,j]-truth[i,j],valid_ohlc=float(legal[i])))
                    clipping.append(dict(fold=fold,partition=part,horizon=h,stage=stage,known_rows=int(known.sum()),
                        price_clipped_rows=int((price_clipped[:,:h].any(1)&known).sum()),
                        turnover_clipped_rows=int((turnover_clipped[:,:h].any(1)&known).sum()),valid_ohlc_rows=int((known&legal).sum())))
                for day in range(h):
                    good=data.a['valid'][ids,day]
                    for j,field in enumerate(['open','high','low','close','volume','amount']):
                        error=np.abs(reconstructed['tokenizer_clipped'][:,day,j]-data.a['future'][ids,day,j])/data.a['scale'][ids,0,j]
                        tmp=pd.DataFrame(dict(date=rows.date.to_numpy()[good],error=error[good]))
                        fields.append(dict(fold=fold,partition=part,horizon=h,day=day+1,field=field,
                            historical_scale_mae=float(tmp.groupby('date').error.mean().mean())))
            print('Reconstruction',fold,part,len(ids),'inputs',flush=True)
    frame=pd.DataFrame(records)
    frame.to_parquet(dest/'row-errors.parquet',index=False)
    daily=frame.groupby(['fold','partition','stage','horizon','date','target'])[['mae','bias','valid_ohlc']].mean().reset_index()
    daily.to_csv(dest/'daily-errors.csv',index=False)
    summary=daily.groupby(['fold','partition','stage','horizon','target'])[['mae','bias','valid_ohlc']].mean().reset_index()
    summary.to_csv(dest/'metrics.csv',index=False)
    pd.DataFrame(clipping).to_csv(dest/'clipping-coverage.csv',index=False)
    pd.DataFrame(fields).to_csv(dest/'field-errors.csv',index=False)
    write_json(dest/'interpretation.json',dict(forecasting=False,actual_future_tokens_used=True,
        invalid_decoded_bars_included_in_error=True,unclipped_future='Offline out-of-distribution diagnostic, not an approved preprocessing change',
        sealed_holdout_accessed=False))
    finish(dest)
    del tokenizer
    gc.collect()
    torch.mps.empty_cache()


def generate(data,checkpoint,ids,dest,settings,cfg,samples=None):
    samples=samples or cfg['samples']
    if (dest/'completed.json').exists():
        verify(dest)
        return np.load(dest/'paths.npy',mmap_mode='r')
    dest.mkdir(parents=True,exist_ok=True)
    chunks=dest/'chunks'
    chunks.mkdir(exist_ok=True)
    np.save(dest/'row-ids.npy',ids)
    model,_=restore_model(checkpoint,cfg['device'])
    tokenizer=load_tokenizer(BUNDLE,cfg['device'])
    size=cfg['generation_batch_rows']
    for start in range(0,len(ids),size):
        batch=ids[start:start+size]
        path=chunks/f'{start:06d}.npz'
        if path.exists():
            continue
        a,b,stamps,_=data.tensors(batch,cfg['device'])
        pairs=generate_tokens(model,a[:,:60],b[:,:60],stamps[:,:60],stamps[:,60:],samples=samples,
            seed=cfg['seed']+start,**settings)
        paths,valid=decode_paths(tokenizer,pairs,data.a['mean'][batch],data.a['scale'][batch],5)
        with path.with_suffix('.tmp').open('wb') as f:
            np.savez_compressed(f,row_ids=batch,paths=paths,valid=valid,
                s1=pairs[0][:,:,-5:].cpu().numpy(),s2=pairs[1][:,:,-5:].cpu().numpy())
        path.with_suffix('.tmp').replace(path)
        if start%512==0:
            p=dict(stage='generation',run=str(dest),inputs=min(start+size,len(ids)),total=len(ids),at=utc_now())
            write_json(dest/'progress.json',p)
            print(p,flush=True)
    output=np.lib.format.open_memmap(dest/'paths.npy',mode='w+',dtype=np.float32,shape=(len(ids),samples,5,6))
    for start in range(0,len(ids),size):
        with np.load(chunks/f'{start:06d}.npz') as z:
            np.testing.assert_array_equal(z['row_ids'],ids[start:start+size])
            output[start:start+len(z['row_ids'])]=z['paths']
    output.flush()
    a,b,stamps,_=data.tensors(ids[:size],cfg['device'])
    pairs=generate_tokens(model,a[:,:60],b[:,:60],stamps[:,:60],stamps[:,60:],samples=samples,seed=cfg['seed'],**settings)
    paths,valid=decode_paths(tokenizer,pairs,data.a['mean'][ids[:size]],data.a['scale'][ids[:size]],5)
    with np.load(chunks/'000000.npz') as z:
        np.testing.assert_array_equal(paths,z['paths'])
        np.testing.assert_array_equal(valid,z['valid'])
        for key,v in zip(['s1','s2'],pairs):
            np.testing.assert_array_equal(z[key],v[:,:,-5:].cpu().numpy())
    write_json(dest/'settings.json',dict(checkpoint=str(checkpoint),checkpoint_sha256=file_hash(checkpoint),
        sampling=settings,samples=samples,seed=cfg['seed'],fixed_seed_replay=True,
        chunks={p.name:file_hash(p) for p in chunks.iterdir()}))
    finish(dest)
    del model,tokenizer
    gc.collect()
    torch.mps.empty_cache()
    return output


def sampling(data,cfg):
    dest=OUT/'sampling'
    if (dest/'completed.json').exists():
        verify(dest)
        return
    dest.mkdir(exist_ok=True)
    summaries,coverages=[],[]
    for fold in cfg['audit_folds']:
        ids=data.selected(fold,'selection',cfg['sampling_selection_rows_per_date'])
        records={}
        for name,settings in cfg['samplers'].items():
            paths=generate(data,PRIOR/fold/'dense_3720k_equal/training/best.pt',ids,dest/fold/name,settings,cfg)
            for count in cfg['nested_sample_counts']:
                frame,cov=score_paths(paths[:,:count],data.a['future'][ids],data.a['valid'][ids],data.a['last'][ids,3],
                    data.rows.iloc[ids].date.to_numpy(),minimum_fraction=cfg['minimum_valid_fraction'])
                variant=name if count==cfg['samples'] else name+f'_{count}'
                frame.to_parquet(dest/f'{fold}-{variant}-rows.parquet',index=False)
                records[variant]=frame
                coverages.append(cov.assign(fold=fold,partition='selection',variant=variant))
        common,daily,summary=common_scores(records)
        common.to_parquet(dest/f'{fold}-common.parquet',index=False)
        daily.to_csv(dest/f'{fold}-daily.csv',index=False)
        summaries.append(summary.assign(fold=fold,partition='selection'))
    summary=pd.concat(summaries,ignore_index=True)
    coverage=pd.concat(coverages,ignore_index=True)
    summary.to_csv(dest/'metrics.csv',index=False)
    coverage.to_csv(dest/'coverage.csv',index=False)
    # Sampler selection holds draw count at 64. Nested 32-draw results diagnose Monte Carlo error only.
    selected=choose_sampler(summary.loc[summary.variant.isin(cfg['samplers'])],
        coverage.loc[coverage.variant.isin(cfg['samplers'])],cfg['selection'])
    write_json(dest/'selection.json',dict(**selected,partition='selection',selected_at=utc_now(),
        evaluation_used=False,settings=cfg['samplers'][selected['selected']]))
    print('Sampler choice',selected,flush=True)
    finish(dest)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['init','reconstruction','sampling','all'])
    args=parser.parse_args()
    initialize()
    if args.stage=='init':
        return
    for p,h in read(OUT/'source-manifest.json')['files'].items():
        assert file_hash(OUT/p)==h,p
    torch.set_num_threads(4)
    data=Dataset()
    cfg=read(OUT/'protocol.json')
    if args.stage in ['all','reconstruction']:
        reconstruction(data,cfg)
    if args.stage in ['all','sampling']:
        sampling(data,cfg)


if __name__=='__main__':
    main()
