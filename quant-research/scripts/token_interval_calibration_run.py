"""Fit empirical intervals on validation forecasts, then evaluate frozen scales."""
import argparse
import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tokenizer_reconstruction_run as core
import torch

from quant_research.forecast_audit import TARGETS, extrema
from quant_research.storage import file_hash, utc_now, write_json
from quant_research.token_intervals import apply_scale, fit_scale, interval_metrics
from quant_research.token_transformer import valid_bars

BASE=Path('/Users/guo/Documents/stocks/quant-research')
ROOT=BASE/'artifacts/token-interval-calibration-20260915-v1'
PRIOR=BASE/'artifacts/tokenizer-reconstruction-20260915-v1'
TRANSFER=BASE/'artifacts/tokenizer-temporal-20260915-v1'
FIELDS=['mae','coverage80','width80','interval_score80','lower_miss','upper_miss']


def read(p):
    return json.loads(p.read_text())


def initialize():
    cfg=read(BASE/'configs/token-interval-calibration-v1.json')
    if (ROOT/'protocol.json').exists():
        assert read(ROOT/'protocol.json')==cfg
        return
    for name,h in read(TRANSFER/'final-verification.json')['files'].items():
        if Path(name).is_relative_to(BASE/'artifacts'):
            assert file_hash(Path(name))==h,name
    ROOT.mkdir()
    write_json(ROOT/'protocol.json',cfg)
    shutil.copytree(PRIOR/'code',ROOT/'code')
    shutil.copy2(BASE/'src/quant_research/token_intervals.py',ROOT/'code/src/quant_research/token_intervals.py')
    shutil.copy2(BASE/'scripts/token_interval_calibration_run.py',ROOT/'code/scripts/token_interval_calibration_run.py')
    shutil.copy2(BASE/'tests/test_token_intervals.py',ROOT/'test_token_intervals.py')
    sources={str(TRANSFER/'final-verification.json'):file_hash(TRANSFER/'final-verification.json'),
        str(core.DATA/'completed.json'):file_hash(core.DATA/'completed.json')}
    choice=read(PRIOR/'decoder-selection/selection.json')
    sources[choice['checkpoint']]=file_hash(Path(choice['checkpoint']))
    for stage in [cfg['calibration']]+[x for x in cfg['evaluations'] if not x.get('reuse')]:
        dest=ROOT/stage['name']
        (dest/'decoder-selection').mkdir(parents=True)
        shutil.copy2(PRIOR/'decoder-selection/selection.json',dest/'decoder-selection/selection.json')
        stage_cfg=copy.deepcopy(read(PRIOR/'protocol.json'))
        stage_cfg['fold'].update(name=stage['name'],evaluation=stage['dates'])
        stage_cfg['forecast_evaluation_rows_per_date']=stage['rows_per_date']
        write_json(dest/'protocol.json',stage_cfg)
        for seed in cfg['forecast_seeds']:
            old=PRIOR/f'predictors/seed{seed}/dense_3720k_equal/training'
            training=dest/f'predictors/seed{seed}/dense_3720k_equal/training'
            training.mkdir(parents=True)
            for name in ['best.pt','summary.json','reload-reference.npz']:
                shutil.copy2(old/name,training/name)
                sources[str(old/name)]=file_hash(old/name)
            core.finish(training)
    write_json(ROOT/'source-manifest.json',dict(at=utc_now(),prior_files=sources,
        protocol_sha256=file_hash(ROOT/'protocol.json'),
        files={str(p.relative_to(ROOT)):file_hash(p) for p in (ROOT/'code').rglob('*.py')}))


def forecast_stage(stage,cfg):
    if stage.get('reuse'):
        return BASE/stage['reuse']
    dest=ROOT/stage['name']
    core.OUT=dest
    stage_cfg=read(dest/'protocol.json')
    data=core.ReconstructionData(stage_cfg)
    ids=data.selected(None,'evaluation',stage['rows_per_date'])
    r=data.rows.iloc[ids]
    assert r.date.min()>=stage['dates'][0] and r.label_end.max()<stage['dates'][1]<cfg['sealed_holdout_start']
    assert r.groupby('date').size().eq(stage['rows_per_date']).all()
    write_json(dest/'preflight.json',dict(passed=True,at=utc_now(),inputs=len(ids),dates=int(r.date.nunique()),
        first_signal=str(r.date.min()),last_signal=str(r.date.max()),last_target=str(r.label_end.max()),refitted=False))
    choice=read(dest/'decoder-selection/selection.json')
    for seed in cfg['forecast_seeds']:
        root=dest/f'predictors/seed{seed}/dense_3720k_equal'
        core.verify(root/'training')
        core.forecasts(data,root,choice,stage_cfg)
    return dest


def extract_quantiles(folder,data,stage,seed):
    ids=np.load(folder/'row-ids.npy')
    paths=np.load(folder/'adapted-paths.npy',mmap_mode='r')
    ref=data.a['last'][ids,3].astype(float)
    records=[]
    for h in [2,5]:
        legal=valid_bars(paths[:,:,:h]).all(-1)
        known=data.a['valid'][ids,:h].all(1)
        truth=extrema(data.a['future'][ids,:h].astype(float),ref)
        for i in np.flatnonzero(known&(legal.sum(1)>=16)):
            values=extrema(paths[i,legal[i],:h].astype(float),ref[i])
            q=np.quantile(values,[.1,.5,.9],axis=0)
            for j,target in enumerate(TARGETS):
                records.append(dict(stage=stage,seed=seed,row_id=int(ids[i]),date=data.rows.iloc[ids[i]].date,
                    horizon=h,target=target,lower=q[0,j],median=q[1,j],upper=q[2,j],actual=truth[i,j],draws=int(legal[i].sum())))
    return pd.DataFrame(records)


def stage_quantiles(stage,source,data,cfg):
    dest=ROOT/'intervals'/stage['name']
    if (dest/'quantiles.parquet').exists():
        meta=read(dest/'source.json')
        assert file_hash(dest/'quantiles.parquet')==meta['quantiles_sha256']
        return pd.read_parquet(dest/'quantiles.parquet')
    dest.mkdir(parents=True,exist_ok=True)
    frames=[]
    sources={}
    for seed in cfg['forecast_seeds']:
        folder=source/f'predictors/seed{seed}/dense_3720k_equal/decoder-forecast'
        core.verify(folder)
        frame=extract_quantiles(folder,data,stage['name'],seed)
        assert frame.date.min()>=stage['dates'][0]
        assert data.rows.iloc[frame.row_id].label_end.max()<stage['dates'][1]
        frames.append(frame)
        for name in ['adapted-paths.npy','row-ids.npy','completed.json']:
            sources[str(folder/name)]=file_hash(folder/name)
    frame=pd.concat(frames,ignore_index=True)
    frame.to_parquet(dest/'quantiles.parquet',index=False)
    write_json(dest/'source.json',dict(source=str(source),files=sources,quantiles_sha256=file_hash(dest/'quantiles.parquet')))
    return frame


def fit(frame,cfg):
    if set(frame.stage)!={cfg['calibration']['name']}:
        raise ValueError('Only calibration-period labels can fit interval scales')
    fitted=[]
    for (seed,h,target),g in frame.groupby(['seed','horizon','target']):
        info=fit_scale(g[['lower','median','upper']].to_numpy(),g.actual.to_numpy(),g.date.to_numpy(),
            cfg['fit']['coverage'],cfg['fit']['epsilon_pp'])
        assert info['dates']>=cfg['fit']['minimum_dates'] and info['rows']>=cfg['fit']['minimum_rows']
        fitted.append(dict(seed=int(seed),horizon=int(h),target=target,**info))
    assert len(fitted)==18
    return dict(at=utc_now(),calibration_only=True,evaluation_used=False,parameters=fitted,
        protocol_sha256=file_hash(ROOT/'protocol.json'),
        calibration_quantiles_sha256=file_hash(ROOT/'intervals'/cfg['calibration']['name']/'quantiles.parquet'))


def evaluate(frame,fitted,cfg,dest):
    frames=[]
    params={(x['seed'],x['horizon'],x['target']):x for x in fitted['parameters']}
    for (seed,h,target),g in frame.groupby(['seed','horizon','target']):
        q=g[['lower','median','upper']].to_numpy()
        p=params[(seed,h,target)]
        calibrated=apply_scale(q,p['scale'],cfg['fit']['lower_support'][target],p['epsilon'])
        np.testing.assert_array_equal(q[:,1],calibrated[:,1])
        for variant,values in [('raw',q),('calibrated',calibrated)]:
            out=g[['stage','seed','row_id','date','horizon','target','actual','draws']].copy()
            out['variant']=variant
            out[['lower','median','upper']]=values
            for name,score in interval_metrics(values,g.actual.to_numpy()).items():
                out[name]=score
            frames.append(out)
    scored=pd.concat(frames,ignore_index=True)
    scored.to_parquet(dest/'scores.parquet',index=False)
    daily=scored.groupby(['stage','seed','variant','date','horizon','target'])[FIELDS].mean().reset_index()
    daily.to_csv(dest/'daily.csv',index=False)
    daily.groupby(['stage','seed','variant','horizon','target'])[FIELDS].mean().reset_index().to_csv(dest/'metrics.csv',index=False)
    write_json(dest/'frozen-fit.json',dict(fit_sha256=file_hash(ROOT/'fit.json'),median_exactly_unchanged=True))
    core.finish(dest)
    return daily


def aggregate(frames,cfg):
    dest=ROOT/'results'
    dest.mkdir(exist_ok=True)
    daily=pd.concat(frames,ignore_index=True)
    daily.to_csv(dest/'daily.csv',index=False)
    records=[]
    for stage in [x['name'] for x in cfg['evaluations']]:
        for h in [2,5]:
            for target in list(TARGETS)+['mean']:
                subset=daily[(daily.stage==stage)&(daily.horizon==h)]
                if target!='mean':
                    subset=subset[subset.target==target]
                for metric in ['mae','coverage80','width80','interval_score80']:
                    paired=subset.groupby(['seed','date','variant'])[metric].mean().unstack('variant')
                    for seed in cfg['forecast_seeds']+['mean']:
                        p=paired if seed=='mean' else paired.loc[[seed]]
                        p=p.groupby('date')[['raw','calibrated']].mean()
                        b,x=p.raw.to_numpy(),p.calibrated.to_numpy()
                        rng=np.random.default_rng(314159)
                        block=cfg['bootstrap_block_dates']
                        ids=(rng.integers(len(x),size=(cfg['bootstrap_replicates'],int(np.ceil(len(x)/block)),1))+np.arange(block))%len(x)
                        ids=ids.reshape(len(ids),-1)[:,:len(x)]
                        lo,hi=np.quantile((x-b)[ids].mean(1),[.025,.975])
                        records.append(dict(stage=stage,seed=seed,horizon=h,target=target,metric=metric,
                            baseline=b.mean(),calibrated=x.mean(),delta=(x-b).mean(),relative_change=x.mean()/b.mean()-1,
                            ci_low=lo,ci_high=hi,dates=len(x)))
    pd.DataFrame(records).to_csv(dest/'paired.csv',index=False)
    core.finish(dest)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['init','all'])
    args=parser.parse_args()
    initialize()
    cfg=read(ROOT/'protocol.json')
    manifest=read(ROOT/'source-manifest.json')
    assert file_hash(ROOT/'protocol.json')==manifest['protocol_sha256']
    for name,h in manifest['files'].items():
        assert file_hash(ROOT/name)==h
    for name,h in manifest['prior_files'].items():
        assert file_hash(Path(name))==h
    if args.stage=='init':
        return
    torch.set_num_threads(4)
    source=forecast_stage(cfg['calibration'],cfg)
    data=core.Dataset()
    frame=stage_quantiles(cfg['calibration'],source,data,cfg)
    if not (ROOT/'fit.json').exists():
        write_json(ROOT/'fit.json',fit(frame,cfg))
    fitted=read(ROOT/'fit.json')
    assert fitted['protocol_sha256']==file_hash(ROOT/'protocol.json')
    assert fitted['calibration_quantiles_sha256']==file_hash(ROOT/'intervals'/cfg['calibration']['name']/'quantiles.parquet')
    evaluate(frame,fitted,cfg,ROOT/'intervals'/cfg['calibration']['name'])
    print('Calibration frozen',fitted,flush=True)
    daily=[]
    for stage in cfg['evaluations']:
        source=forecast_stage(stage,cfg)
        frame=stage_quantiles(stage,source,data,cfg)
        daily.append(evaluate(frame,fitted,cfg,ROOT/'intervals'/stage['name']))
        print('Interval evaluation completed',stage['name'],flush=True)
    aggregate(daily,cfg)
    write_json(ROOT/'run-completed.json',dict(passed=True,at=utc_now(),fit_sha256=file_hash(ROOT/'fit.json'),
        weights_refitted=False,median_unchanged=True,sealed_holdout_opened=False))


if __name__=='__main__':
    main()
