"""Independent raw-quantile, fitting-boundary and calibrated-score checks."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from token_history_run import DATA, Dataset
from token_interval_calibration_run import FIELDS, PRIOR, ROOT, read
from verify_token_pipeline_audit import legal

from quant_research.storage import file_hash, utc_now, write_json
from quant_research.token_transformer import restore_model


def check_quantiles(stage,data):
    dest=ROOT/'intervals'/stage['name']
    meta=read(dest/'source.json')
    expected=pd.read_parquet(dest/'quantiles.parquet')
    assert file_hash(dest/'quantiles.parquet')==meta['quantiles_sha256']
    frames=[]
    for seed in [17,29,43]:
        folder=Path(meta['source'])/f'predictors/seed{seed}/dense_3720k_equal/decoder-forecast'
        ids=np.load(folder/'row-ids.npy')
        p=np.load(folder/'adapted-paths.npy',mmap_mode='r')
        ref=data.a['last'][ids,3].astype(float)
        for h in [2,5]:
            mask=legal(p[:,:,:h])
            known=data.a['valid'][ids,:h].all(1)
            valid=known&(mask.sum(1)>=16)
            for start in range(0,len(ids),64):
                pos=np.arange(start,min(start+64,len(ids)))
                pos=pos[valid[pos]]
                if not len(pos):
                    continue
                v=p[pos,:,:h].astype(float)
                high,low=v[...,1].max(-1),v[...,2].min(-1)
                x=np.stack([(high/ref[pos,None]-1)*100,(low/ref[pos,None]-1)*100,
                    (high-low)/ref[pos,None]*100],-1)
                x=np.where(mask[pos,:,None],x,np.nan)
                q=np.nanquantile(x,[.1,.5,.9],axis=1)
                truth=data.a['future'][ids[pos],:h].astype(float)
                yh,yl=truth[:,:,1].max(-1),truth[:,:,2].min(-1)
                y=np.stack([(yh/ref[pos]-1)*100,(yl/ref[pos]-1)*100,(yh-yl)/ref[pos]*100],-1)
                for k,i in enumerate(pos):
                    for j,target in enumerate(['maximum','minimum','range']):
                        frames.append(dict(stage=stage['name'],seed=seed,row_id=int(ids[i]),date=data.rows.iloc[ids[i]].date,
                            horizon=h,target=target,lower=q[0,k,j],median=q[1,k,j],upper=q[2,k,j],actual=y[k,j],draws=int(mask[i].sum())))
    found=pd.DataFrame(frames)
    keys=['seed','row_id','horizon','target']
    x=found.sort_values(keys).reset_index(drop=True)
    y=expected.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(x[keys+['stage','date','draws']],y[keys+['stage','date','draws']],check_dtype=False)
    np.testing.assert_allclose(x[['lower','median','upper','actual']],y[['lower','median','upper','actual']],rtol=1e-10,atol=1e-10)
    assert x.date.min()>=stage['dates'][0]
    assert data.rows.iloc[x.row_id].label_end.max()<stage['dates'][1]
    for name,h in meta['files'].items():
        assert file_hash(Path(name))==h
    return found


def check_fit(frame,cfg,fitted):
    assert fitted['calibration_only'] and not fitted['evaluation_used']
    assert frame.stage.eq(cfg['calibration']['name']).all()
    assert fitted['calibration_quantiles_sha256']==file_hash(ROOT/'intervals'/cfg['calibration']['name']/'quantiles.parquet')
    assert fitted['protocol_sha256']==file_hash(ROOT/'protocol.json')
    for p in fitted['parameters']:
        g=frame[(frame.seed==p['seed'])&(frame.horizon==p['horizon'])&(frame.target==p['target'])]
        assert len(g)==p['rows'] and g.date.nunique()==p['dates']
        left=np.maximum(g['median']-g.lower,cfg['fit']['epsilon_pp'])
        right=np.maximum(g.upper-g['median'],cfg['fit']['epsilon_pp'])
        residual=np.maximum((g['median']-g.actual)/left,(g.actual-g['median'])/right).to_numpy()
        counts=g.groupby('date').size()
        weights=np.asarray([1/counts[day] for day in g.date])
        scale=p['scale']
        assert scale>=1 and np.isfinite(scale)
        assert p['epsilon']==cfg['fit']['epsilon_pp']
        assert np.isclose(np.maximum(1,residual),scale,rtol=1e-12,atol=1e-12).any()
        covered=weights[residual<=scale+1e-10].sum()/weights.sum()
        assert covered>=.8-1e-10
        if scale>1:
            below=weights[residual<scale-1e-10].sum()/weights.sum()
            assert below<=.8+1e-10
    assert len(fitted['parameters'])==18


def check_scores(frame,cfg,fitted):
    stage=frame.stage.iloc[0]
    dest=ROOT/'intervals'/stage
    expected=pd.read_parquet(dest/'scores.parquet')
    found=[]
    params={(p['seed'],p['horizon'],p['target']):p for p in fitted['parameters']}
    for row in frame.itertuples(index=False):
        p=params[(row.seed,row.horizon,row.target)]
        for variant in ['raw','calibrated']:
            lo,mid,hi=row.lower,row.median,row.upper
            if variant=='calibrated':
                lo=max(cfg['fit']['lower_support'][row.target],mid-p['scale']*max(mid-lo,p['epsilon']))
                hi=mid+p['scale']*max(hi-mid,p['epsilon'])
            y=row.actual
            found.append(dict(seed=row.seed,row_id=row.row_id,date=row.date,horizon=row.horizon,target=row.target,
                variant=variant,lower=lo,median=mid,upper=hi,mae=abs(mid-y),coverage80=float(lo<=y<=hi),width80=hi-lo,
                interval_score80=hi-lo+10*(lo-y if y<lo else 0)+10*(y-hi if y>hi else 0),
                lower_miss=float(y<lo),upper_miss=float(y>hi)))
    found=pd.DataFrame(found)
    keys=['seed','row_id','horizon','target','variant']
    x=found.sort_values(keys).reset_index(drop=True)
    y=expected.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(x[keys+['date']],y[keys+['date']],check_dtype=False)
    np.testing.assert_allclose(x[['lower','median','upper']+FIELDS],y[['lower','median','upper']+FIELDS],rtol=1e-10,atol=1e-10)
    matched=found.pivot(index=['seed','row_id','horizon','target'],columns='variant',values=['median','mae'])
    for field in ['median','mae']:
        np.testing.assert_array_equal(matched[field]['raw'],matched[field]['calibrated'])
    daily=[]
    for (seed,variant,h,target),g in found.groupby(['seed','variant','horizon','target']):
        dates,codes=np.unique(g.date,return_inverse=True)
        count=np.bincount(codes)
        values={name:np.bincount(codes,weights=g[name])/count for name in FIELDS}
        for i,day in enumerate(dates):
            daily.append(dict(stage=stage,seed=seed,variant=variant,date=day,horizon=h,target=target,
                **{name:float(v[i]) for name,v in values.items()}))
    daily=pd.DataFrame(daily)
    keys=['stage','seed','variant','date','horizon','target']
    x=daily.sort_values(keys).reset_index(drop=True)
    y=pd.read_csv(dest/'daily.csv').sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(x[keys],y[keys],check_dtype=False)
    np.testing.assert_allclose(x[FIELDS],y[FIELDS],atol=1e-10,rtol=1e-10)
    summary=daily.groupby(['stage','seed','variant','horizon','target'])[FIELDS].mean().reset_index()
    pd.testing.assert_frame_equal(summary,pd.read_csv(dest/'metrics.csv'),check_exact=False,rtol=1e-10,atol=1e-10)
    assert read(dest/'frozen-fit.json')['fit_sha256']==file_hash(ROOT/'fit.json')
    return daily,len(found)


def contrasts(source,cfg):
    expected=pd.read_csv(ROOT/'results/paired.csv',dtype={'seed':str})
    for row in expected.itertuples(index=False):
        g=source[(source.stage==row.stage)&(source.horizon==row.horizon)]
        if row.target!='mean':
            g=g[g.target==row.target]
        if row.seed!='mean':
            g=g[g.seed==int(row.seed)]
        dates=sorted(set(g.date))
        b,x=[],[]
        for day in dates:
            values=g[g.date==day]
            b.append(values.loc[values.variant=='raw',row.metric].mean())
            x.append(values.loc[values.variant=='calibrated',row.metric].mean())
        b,x=np.asarray(b),np.asarray(x)
        np.testing.assert_allclose([row.baseline,row.calibrated,row.delta,row.relative_change],
            [b.mean(),x.mean(),(x-b).mean(),x.mean()/b.mean()-1],atol=1e-10,rtol=1e-10)
        rng=np.random.default_rng(314159)
        starts=rng.integers(len(dates),size=(2000,int(np.ceil(len(dates)/10))))
        sample=[]
        for blocks in starts:
            ids=[(int(start)+i)%len(dates) for start in blocks for i in range(10)][:len(dates)]
            sample.append(float(sum((x-b)[ids])/len(dates)))
        np.testing.assert_allclose([row.ci_low,row.ci_high],np.quantile(sample,[.025,.975]),atol=1e-10,rtol=1e-10)
    assert len(expected)==256
    return len(expected)


def main():
    assert read(ROOT/'run-completed.json')['passed']
    torch.set_num_threads(4)
    cfg=read(ROOT/'protocol.json')
    fitted=read(ROOT/'fit.json')
    data=Dataset()
    for name in ['rows.parquet','s1.npy','s2.npy','future.npy','valid.npy','mean.npy','scale.npy','last.npy','calendar-stamps.npy']:
        assert file_hash(DATA/name)==read(DATA/'completed.json')['files'][name]
    counts={}
    source=[]
    for stage in [cfg['calibration']]+cfg['evaluations']:
        frame=check_quantiles(stage,data)
        if stage['name']==cfg['calibration']['name']:
            check_fit(frame,cfg,fitted)
        else:
            assert cfg['calibration']['dates'][1]<=stage['dates'][0]<stage['dates'][1]<cfg['sealed_holdout_start']
        daily,count=check_scores(frame,cfg,fitted)
        counts[stage['name']]=dict(quantiles=len(frame),scores=count)
        if stage['name']!=cfg['calibration']['name']:
            source.append(daily)
    source=pd.concat(source,ignore_index=True)
    counts['contrasts_and_intervals']=contrasts(source,cfg)
    for seed in cfg['forecast_seeds']:
        old=PRIOR/f'predictors/seed{seed}/dense_3720k_equal/training'
        model,_=restore_model(old/'best.pt','cpu')
        model.eval()
        with np.load(old/'reload-reference.npz') as z,torch.no_grad():
            a,b,stamps,_=data.tensors(z['ids'],'cpu')
            logits=model.forecast_logits(a[:,:-1],b[:,:-1],stamps[:,:-1],a[:,1:],59)
            np.testing.assert_array_equal(logits[0].numpy(),z['coarse'])
            np.testing.assert_array_equal(logits[1].numpy(),z['fine'])
    for folder in ROOT.glob('*/predictors/*/dense_3720k_equal/decoder-forecast'):
        proof=read(folder/'verification.json')
        assert proof['first_batch_replay_exact'] and proof['shared_generated_tokens']
        seed_name=folder.parent.parent.name
        original=PRIOR/f'predictors/{seed_name}/dense_3720k_equal/training/best.pt'
        copied=folder.parent/'training/best.pt'
        assert file_hash(copied)==file_hash(original)==proof['predictor_sha256']
        stage_root=folder.parents[3]
        selected=stage_root/'decoder-selection/selection.json'
        assert file_hash(selected)==file_hash(PRIOR/'decoder-selection/selection.json')==proof['selection_sha256']
        assert file_hash(Path(read(selected)['checkpoint']))==proof['decoder_sha256']
        for name,h in proof['chunks'].items():
            assert file_hash(folder/'chunks'/name)==h
    files={}
    for done in ROOT.rglob('completed.json'):
        obj=read(done)
        assert obj['passed']
        for name,h in obj['files'].items():
            assert file_hash(done.parent/name)==h
        files[str(done)]=file_hash(done)
    for name,h in read(ROOT/'source-manifest.json')['files'].items():
        assert file_hash(ROOT/name)==h
    for name,h in read(ROOT/'source-manifest.json')['prior_files'].items():
        assert file_hash(Path(name))==h
    write_json(ROOT/'independent-verification.json',dict(passed=True,at=utc_now(),counts=counts,files=files,
        calibration_only_fit=True,median_and_mae_exactly_unchanged=True,raw_quantiles_independently_recomputed=True,
        fitted_scales_independently_checked=True,sealed_holdout_opened=False))
    print('Interval calibration independently verified',counts,flush=True)


if __name__=='__main__':
    main()
