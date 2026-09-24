"""Independent decoder distortion, frozen-code semantics and matched forecast verification."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from token_history_run import BUNDLE, Dataset
from verify_token_pipeline_audit import compare_group, verify_aggregation

from quant_research.storage import file_hash, utc_now, write_json
from quant_research.token_transformer import load_tokenizer, restore_model
from quant_research.tokenizer_reconstruction import TRAINABLE

BASE=Path('/Users/guo/Documents/stocks/quant-research')
ROOT=BASE/'artifacts/tokenizer-reconstruction-20260915-v1'


def read(path):
    return json.loads(path.read_text())


def verify_reconstruction(folder,data):
    ids=np.load(folder/'row-ids.npy')
    p=np.load(folder/'reconstruction.npy').astype(float)
    expected=pd.read_parquet(folder/'rows.parquet')
    fields=['mae','bias','legal','normalized_mae','turnover_mae']
    records=[]
    for h in [2,5]:
        known=data.a['valid'][ids,:h].all(1)
        q=p[:,:h]
        truth=data.a['future'][ids,:h].astype(float)
        ref=data.a['last'][ids,3].astype(float)
        hi=q[:,:,1].max(1)
        lo=q[:,:,2].min(1)
        yh=truth[:,:,1].max(1)
        yl=truth[:,:,2].min(1)
        error=np.stack([(hi-yh)/ref*100,(lo-yl)/ref*100,((hi-lo)-(yh-yl))/ref*100],1)
        legal=(np.isfinite(q).all(-1)&(q[...,:4]>0).all(-1)&(q[...,4:]>=0).all(-1)
            &(q[...,1]>=q[...,0])&(q[...,1]>=q[...,3])&(q[...,1]>=q[...,2])
            &(q[...,2]<=q[...,0])&(q[...,2]<=q[...,3])).all(1)
        norm=np.abs(q-truth)/data.a['scale'][ids]
        for i in np.flatnonzero(known):
            for j,target in enumerate(['maximum','minimum','range']):
                records.append(dict(row_id=int(ids[i]),date=data.rows.iloc[ids[i]].date,horizon=h,target=target,
                    mae=abs(error[i,j]),bias=error[i,j],legal=float(legal[i]),normalized_mae=norm[i].mean(),turnover_mae=norm[i,:,4:].mean()))
    found=pd.DataFrame(records)
    keys=['row_id','horizon','target']
    x=found.sort_values(keys).reset_index(drop=True)
    y=expected.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(x[keys+['date']],y[keys+['date']],check_dtype=False)
    np.testing.assert_allclose(x[fields],y[fields],rtol=1e-10,atol=1e-10)
    summary=[]
    for (h,target),g in found.groupby(['horizon','target']):
        _,codes=np.unique(g.date,return_inverse=True)
        counts=np.bincount(codes)
        summary.append(dict(horizon=h,target=target,**{k:float((np.bincount(codes,weights=g[k])/counts).mean()) for k in fields}))
    x=pd.DataFrame(summary).sort_values(['horizon','target']).reset_index(drop=True)
    y=pd.read_csv(folder/'metrics.csv').sort_values(['horizon','target']).reset_index(drop=True)
    np.testing.assert_allclose(x[fields],y[fields],rtol=1e-10,atol=1e-10)
    aggregate=read(folder/'summary.json')
    for k in fields:
        np.testing.assert_allclose(aggregate[k],x[k].mean(),rtol=1e-10,atol=1e-10)
    return len(found)


def frozen_and_reload(data,choice):
    control=load_tokenizer(BUNDLE,'cpu')
    control_state=control.state_dict()
    ids=np.load(ROOT/'selection-ids.npy')[:4]
    pair=[torch.as_tensor(data.a[k][ids].astype(np.int64)) for k in ['s1','s2']]
    torch.manual_seed(9)
    probe=torch.randn(4,65,6)
    with torch.no_grad():
        encoded=control.encode(probe,half=True)
    result=[]
    for candidate in choice['candidates']:
        model=load_tokenizer(BUNDLE,'cpu')
        saved=torch.load(candidate['checkpoint'],map_location='cpu',weights_only=True)
        model.load_state_dict(saved['state_dict'])
        frozen=0
        for k,v in model.state_dict().items():
            if not k.startswith(TRAINABLE):
                torch.testing.assert_close(v,control_state[k],atol=0,rtol=0)
                frozen+=v.numel()
        with torch.no_grad():
            codes=model.encode(probe,half=True)
            prefix=model.encode(probe[:,:60],half=True)
            for a,b,c in zip(codes,encoded,prefix):
                torch.testing.assert_close(a,b,atol=0,rtol=0)
                torch.testing.assert_close(a[:,:60],c,atol=0,rtol=0)
            full=model.decode(pair,half=True)
            earlier=model.decode([p[:,:60] for p in pair],half=True)
            torch.testing.assert_close(full[:,:60],earlier,rtol=1e-5,atol=1e-5)
            raw=full[:,-5:].numpy()*data.a['scale'][ids]+data.a['mean'][ids]
        folder=ROOT/'decoder-selection'/candidate['objective']
        stored_ids=np.load(folder/'row-ids.npy')
        positions=pd.Index(stored_ids).get_indexer(ids)
        assert (positions>=0).all()
        stored=np.load(folder/'reconstruction.npy')[positions]
        np.testing.assert_allclose((raw-data.a['mean'][ids])/data.a['scale'][ids],
            (stored-data.a['mean'][ids])/data.a['scale'][ids],rtol=2e-4,atol=2e-5)
        result.append(dict(objective=candidate['objective'],epoch=saved['epoch'],frozen_state_elements=frozen,
            encoded_tokens_identical=True,encoder_and_decoder_causal=True,cpu_mps_reconstruction_close=True,
            max_cpu_mps_raw_difference_by_field={name:float(np.abs(raw-stored)[...,j].max()) for j,name in enumerate(['open','high','low','close','volume','amount'])},
            max_cpu_mps_difference_in_historical_scale=float((np.abs(raw-stored)/data.a['scale'][ids]).max())))
    return result


def contrasts(cfg):
    frames=[pd.read_csv(ROOT/f'predictors/seed{s}/dense_3720k_equal/decoder-forecast/daily.csv').assign(seed=s) for s in cfg['forecast_seeds']]
    source=pd.concat(frames,ignore_index=True)
    rows=pd.read_csv(ROOT/'forecast-results/paired.csv',dtype={'seed':str})
    for row in rows.itertuples(index=False):
        frame=source[source.horizon==row.horizon]
        if row.seed!='mean':
            frame=frame[frame.seed==int(row.seed)]
        b=[]
        x=[]
        dates=sorted(set(frame.date))
        for date in dates:
            p=frame[frame.date==date]
            b.append(p.loc[p.variant=='frozen',row.metric].mean())
            x.append(p.loc[p.variant=='adapted',row.metric].mean())
        b,x=np.asarray(b),np.asarray(x)
        np.testing.assert_allclose([row.baseline,row.adapted,row.delta,row.relative_change],
            [b.mean(),x.mean(),(x-b).mean(),x.mean()/b.mean()-1],rtol=1e-10,atol=1e-10)
        rng=np.random.default_rng(314159)
        starts=rng.integers(len(dates),size=(2000,int(np.ceil(len(dates)/10))))
        samples=[]
        for blocks in starts:
            indices=[(int(start)+i)%len(dates) for start in blocks for i in range(10)][:len(dates)]
            samples.append(float(sum((x-b)[indices])/len(dates)))
        np.testing.assert_allclose([row.ci_low,row.ci_high],np.quantile(samples,[.025,.975]),rtol=1e-10,atol=1e-10)
    return len(rows)


def main():
    torch.set_num_threads(4)
    data=Dataset()
    cfg=read(ROOT/'protocol.json')
    completed=read(ROOT/'run-completed.json')
    assert completed['passed']
    choice=read(ROOT/'decoder-selection/selection.json')
    assert choice['selection_only'] and not choice['evaluation_used']
    counts={}
    for stage in ['decoder-selection','reconstruction-evaluation']:
        for folder in (ROOT/stage).iterdir():
            if folder.is_dir() and (folder/'reconstruction.npy').exists():
                counts[f'{stage}/{folder.name}']=verify_reconstruction(folder,data)
    baseline=read(ROOT/'decoder-selection/frozen/summary.json')
    candidates=[]
    g=cfg['selection']
    for c in choice['candidates']:
        x=read(ROOT/'decoder-selection'/c['objective']/'summary.json')
        passed=(x['mae']<=baseline['mae']*(1-g['minimum_extrema_mae_improvement'])
            and x['legal']>=baseline['legal']+g['minimum_legal_fraction_improvement']
            and x['normalized_mae']<=baseline['normalized_mae']*(1+g['maximum_normalized_mae_increase'])
            and x['turnover_mae']<=baseline['turnover_mae']*(1+g['maximum_turnover_mae_increase']))
        assert bool(passed)==c['qualified']
        if passed:
            candidates.append((x['mae'],c['objective']))
        seen=np.load(ROOT/'decoder-training'/c['objective']/'visited-ids.npy')
        assert set(seen).issubset(set(np.load(ROOT/'train-ids.npy')))
        assert data.a['valid'][seen].any(1).all()
    assert choice['selected']==(min(candidates)[1] if candidates else 'frozen')
    model_checks=frozen_and_reload(data,choice)
    for part in ['train','selection','evaluation']:
        r=data.rows.iloc[np.load(ROOT/f'{part}-ids.npy')]
        assert r.date.min()>=cfg['fold'][part][0]
        assert r.label_end.max()<cfg['fold'][part][1]<cfg['sealed_holdout_start']
    if completed['forecast_stage_run']:
        for seed in cfg['forecast_seeds']:
            root=ROOT/f'predictors/seed{seed}/dense_3720k_equal'
            folder=root/'decoder-forecast'
            visited=np.load(root/'training/visited-row-ids.npy')
            assert set(visited).issubset(set(np.load(ROOT/'train-ids.npy')))
            assert data.a['valid'][visited].any(1).all()
            assert len(visited)==read(root/'training/summary.json')['unique_examples_visited']
            ids=np.load(folder/'row-ids.npy')
            mapping={name:np.load(folder/f'{name}-paths.npy',mmap_mode='r') for name in ['frozen','adapted']}
            expected=pd.read_parquet(folder/'common.parquet')
            counts[f'forecast-{seed}']=compare_group(mapping,ids,expected,data.rows,data.a['future'],data.a['valid'],data.a['last'])
            verify_aggregation(expected,pd.read_csv(folder/'daily.csv'),pd.read_csv(folder/'metrics.csv'))
            model,_=restore_model(root/'training/best.pt','cpu')
            model.eval()
            with np.load(root/'training/reload-reference.npz') as z,torch.no_grad():
                a,b,stamps,_=data.tensors(z['ids'],'cpu')
                logits=model.forecast_logits(a[:,:-1],b[:,:-1],stamps[:,:-1],a[:,1:],59)
                np.testing.assert_array_equal(logits[0].numpy(),z['coarse'])
                np.testing.assert_array_equal(logits[1].numpy(),z['fine'])
        counts['paired_contrasts_and_intervals']=contrasts(cfg)
    files={}
    for p in ROOT.rglob('completed.json'):
        x=read(p)
        assert x['passed']
        for name,h in x['files'].items():
            assert file_hash(p.parent/name)==h,(p,name)
        files[str(p)]=file_hash(p)
    for p in (ROOT/'predictors').rglob('verification.json') if (ROOT/'predictors').exists() else []:
        for name,h in read(p)['chunks'].items():
            assert file_hash(p.parent/'chunks'/name)==h
    for p,h in read(ROOT/'source-manifest.json')['files'].items():
        assert file_hash(ROOT/p)==h,p
    write_json(ROOT/'independent-verification.json',dict(passed=True,at=utc_now(),counts=counts,decoder_checks=model_checks,files=files,
        masks_and_price_metrics_independently_recomputed=True,selection_rule_independently_recomputed=True,
        token_semantics_preserved=True,causality_verified=True,sealed_holdout_opened=False))
    print('Independent verification passed',counts,flush=True)


if __name__=='__main__':
    main()
