"""Checkpoint-specific calibration and frozen three-profile transfer comparison."""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import token_profile_transfer_run as transfer
import tokenizer_reconstruction_run as core
import torch
from token_history_run import DATA, Dataset

from quant_research.forecast_audit import TARGETS, empirical_score, extrema
from quant_research.storage import file_hash, utc_now, write_json
from quant_research.token_history import partition_indices
from quant_research.token_intervals import fit_scale
from quant_research.token_transformer import valid_bars

BASE=Path('/Users/guo/Documents/stocks/quant-research')
ART=BASE/'artifacts'
ROOT=ART/'token-midpoint-transfer-20260915-v1'
MID=ART/'token-midpoint-loss-20260915-v1'
CURRENT=ART/'token-profile-transfer-20260915-v1'
SEEDS=[17,29,43]
OWNERS=['current','ce','midpoint']
FIELDS=transfer.FIELDS


def read(path):
    return json.loads(path.read_text())


def prepare():
    assert not ROOT.exists()
    cfg=read(BASE/'configs/token-midpoint-transfer-v1.json')
    sources={}
    for prior in [MID,CURRENT]:
        final=read(prior/'final-verification.json')
        assert final['passed']
        for name,h in final['files'].items():
            if Path(name).is_relative_to(ART):
                assert file_hash(Path(name))==h,name
        sources[str(prior/'final-verification.json')]=file_hash(prior/'final-verification.json')
    selected=read(MID/'candidate-profile.json')
    assert selected['objective_weight']==0.05
    profiles=read(CURRENT/'profiles.json')
    out=dict(decoder=profiles['decoder'],sampling=cfg['sampling'],fit_settings=profiles['fit_settings'],current=profiles['refreshed'])
    for owner in ['ce','midpoint']:
        out[owner]=dict(checkpoints={str(seed):dict(path=str(MID/owner/f'seed{seed}/training/final.pt'),
            sha256=file_hash(MID/owner/f'seed{seed}/training/final.pt')) for seed in SEEDS})
    assert out['midpoint']['checkpoints']==selected['checkpoints']
    for owner in OWNERS:
        for item in out[owner]['checkpoints'].values():
            sources[item['path']]=item['sha256']
    for path in [MID/'candidate-profile.json',MID/'selection.json',CURRENT/'profiles.json',CURRENT/'row-ids.npy',
                 Path(out['current']['fit_path']),Path(out['decoder']['path'])]:
        sources[str(path)]=file_hash(path)
    assert sources[out['current']['fit_path']]==out['current']['fit_sha256']
    for seed in SEEDS:
        directory=CURRENT/'forecasts/refreshed'/f'seed{seed}'
        core.verify(directory)
        for p in directory.rglob('*'):
            if p.is_file() and p.name!='progress.json':
                sources[str(p)]=file_hash(p)
    meta=read(DATA/'completed.json')
    for name in ['completed.json','rows.parquet','s1.npy','s2.npy','valid.npy','future.npy','mean.npy','scale.npy','last.npy','calendar-stamps.npy','calendar.json']:
        path=DATA/name
        if name!='completed.json':
            assert file_hash(path)==meta['files'][name]
        sources[str(path)]=file_hash(path)
    ROOT.mkdir()
    data=Dataset()
    dates=read(DATA/'calendar.json')
    splits={}
    for part in ['calibration','evaluation']:
        spec=cfg[part]
        ids=partition_indices(data.rows,dates,*spec['dates'])
        ids=data.rows.iloc[ids].groupby('date').head(spec['rows_per_date']).row_id.to_numpy(int)
        r=data.rows.iloc[ids]
        assert len(ids)==spec['inputs'] and r.date.nunique()==spec['dates_count']
        assert r.date.min()>=spec['dates'][0] and r.label_end.max()<spec['dates'][1]<cfg['sealed_holdout_start']
        np.save(ROOT/f'{part}-ids.npy',ids)
        splits[part]=dict(inputs=len(ids),dates=int(r.date.nunique()),first=r.date.min(),last=r.date.max(),label_end=r.label_end.max())
    np.testing.assert_array_equal(np.load(ROOT/'evaluation-ids.npy'),np.load(CURRENT/'row-ids.npy'))
    for seed in SEEDS:
        np.testing.assert_array_equal(np.load(ROOT/'calibration-ids.npy'),np.load(ART/f'token-refreshed-calibration-20260915-v1/2024h2/calibration-forecasts/seed{seed}/row-ids.npy'))
    write_json(ROOT/'splits.json',splits)
    write_json(ROOT/'protocol.json',cfg)
    write_json(ROOT/'initial-profiles.json',out)
    write_json(ROOT/'sources.json',sources)
    shutil.copytree(BASE/'src',ROOT/'code/src',ignore=shutil.ignore_patterns('__pycache__'))
    (ROOT/'code/scripts').mkdir()
    for name in ['token_midpoint_transfer_run.py','token_profile_transfer_run.py','token_history_run.py','tokenizer_reconstruction_run.py']:
        shutil.copy2(BASE/'scripts'/name,ROOT/'code/scripts'/name)
    write_json(ROOT/'prepared.json',dict(at=utc_now(),files={str(p.relative_to(ROOT)):file_hash(p) for p in ROOT.rglob('*') if p.is_file()}))
    print('Prepared',splits,flush=True)


def checked():
    for name,h in read(ROOT/'prepared.json')['files'].items():
        assert file_hash(ROOT/name)==h,name
    for name,h in read(ROOT/'sources.json').items():
        assert file_hash(Path(name))==h,name


def fit(data,owner,profiles):
    dest=ROOT/'fits'/owner
    if (dest/'completed.json').exists():
        core.verify(dest)
        return dest/'fit.json'
    dest.mkdir(parents=True)
    ids=np.load(ROOT/'calibration-ids.npy')
    records=[]
    for seed in SEEDS:
        directory=ROOT/'calibration/forecasts'/owner/f'seed{seed}'
        core.verify(directory)
        assert read(directory/'lineage.json')['checkpoint_sha256']==profiles[owner]['checkpoints'][str(seed)]['sha256']
        p=np.load(directory/'paths.npy',mmap_mode='r')
        for h in [2,5]:
            mask=valid_bars(p[:,:,:h]).all(-1)
            known=data.a['valid'][ids,:h].all(1)
            ref=data.a['last'][ids,3].astype(float)
            y=extrema(data.a['future'][ids,:h].astype(float),ref)
            for i in np.flatnonzero(known&(mask.sum(1)>=16)):
                draws=extrema(p[i,mask[i],:h].astype(float),ref[i])
                q=np.quantile(draws,[.1,.5,.9],axis=0)
                for j,target in enumerate(TARGETS):
                    records.append(dict(seed=seed,row_id=int(ids[i]),date=data.rows.iloc[ids[i]].date,horizon=h,target=target,
                        lower=q[0,j],median=q[1,j],upper=q[2,j],actual=y[i,j],draws=int(mask[i].sum())))
    frame=pd.DataFrame(records)
    frame.to_parquet(dest/'quantiles.parquet',index=False)
    settings=read(ROOT/'protocol.json')['fit']
    params=[]
    for (seed,h,target),g in frame.groupby(['seed','horizon','target']):
        info=fit_scale(g[['lower','median','upper']].to_numpy(),g.actual.to_numpy(),g.date.to_numpy(),settings['coverage'],settings['epsilon_pp'])
        assert info['rows']>=settings['minimum_rows'] and info['dates']>=settings['minimum_dates']
        params.append(dict(seed=int(seed),horizon=int(h),target=target,**info))
    assert len(params)==18
    write_json(dest/'fit.json',dict(at=utc_now(),parameters=params,calibration_only=True,evaluation_used=False,
        checkpoint_sha256={k:v['sha256'] for k,v in profiles[owner]['checkpoints'].items()},decoder_sha256=profiles['decoder']['sha256'],
        quantiles_sha256=file_hash(dest/'quantiles.parquet'),row_ids_sha256=file_hash(ROOT/'calibration-ids.npy'),
        calibration_dates=read(ROOT/'protocol.json')['calibration']['dates'],sampling=profiles['sampling']))
    core.finish(dest)
    return dest/'fit.json'


def paired(daily,cfg):
    records=[]
    for comparison,(baseline,candidate) in cfg['comparisons'].items():
        metrics=['crps'] if comparison.startswith('raw_') else ['mae','coverage80','width80','interval_score80']
        for h in [2,5]:
            for target in ['maximum','minimum','range','high_low']:
                sub=daily[(daily.horizon==h)&daily.variant.isin([baseline,candidate])]
                sub=sub[sub.target.isin(['maximum','minimum'])] if target=='high_low' else sub[sub.target==target]
                for metric in metrics:
                    by=sub.groupby(['seed','date','variant'])[metric].mean().unstack()
                    for seed in [*map(str,SEEDS),'mean']:
                        x=(by if seed=='mean' else by.loc[[int(seed)]]).groupby('date').mean()
                        a,b=x[baseline].to_numpy(),x[candidate].to_numpy()
                        rng=np.random.default_rng(314159)
                        ix=(rng.integers(len(x),size=(2000,int(np.ceil(len(x)/10)),1))+np.arange(10))%len(x)
                        lo,hi=np.quantile((b-a)[ix.reshape(2000,-1)[:,:len(x)]].mean(1),[.025,.975])
                        records.append(dict(comparison=comparison,seed=seed,horizon=h,target=target,metric=metric,
                            baseline=a.mean(),candidate=b.mean(),delta=(b-a).mean(),relative_change=b.mean()/a.mean()-1,ci_low=lo,ci_high=hi,dates=len(x)))
    return pd.DataFrame(records)


def aggregate(frames,cfg):
    frame=pd.concat(frames,ignore_index=True)
    keys=['seed','row_id','horizon','target']
    count=frame.groupby(keys).variant.transform('nunique')
    common=frame[count==6].copy()
    dest=ROOT/'results'
    dest.mkdir(exist_ok=True)
    common.to_parquet(dest/'common.parquet',index=False)
    daily=common.groupby(['seed','variant','date','horizon','target'])[FIELDS].mean().reset_index()
    assert (daily.groupby(['seed','variant','horizon','target']).size()==76).all()
    daily.to_csv(dest/'daily.csv',index=False)
    paired(daily,cfg).to_csv(dest/'paired.csv',index=False)
    core.finish(dest)
    return common


def midpoint_scores(data,ids,common):
    rows=[]
    for seed in SEEDS:
        for h in [2,5]:
            cohort=set(common[(common.seed==seed)&(common.horizon==h)].row_id)
            for owner in OWNERS:
                p=np.load(ROOT/'forecasts'/owner/f'seed{seed}/paths.npy',mmap_mode='r')
                mask=valid_bars(p[:,:,:h]).all(-1)
                for i,row_id in enumerate(ids):
                    if row_id not in cohort:
                        continue
                    ref=float(data.a['last'][row_id,3])
                    draw=p[i,mask[i],:h].astype(float)
                    values=((draw[:,:,1].max(1)+draw[:,:,2].min(1))/(2*ref)-1)*100
                    y=data.a['future'][row_id,:h]
                    target=((y[:,1].max()+y[:,2].min())/(2*ref)-1)*100
                    score=empirical_score(values[:,None],np.array([target]))
                    rows.append(dict(seed=seed,owner=owner,row_id=int(row_id),date=data.rows.iloc[row_id].date,horizon=h,
                        mae=score['mae'][0],crps=score['crps'][0]))
    frame=pd.DataFrame(rows)
    frame.to_parquet(ROOT/'results/midpoint-rows.parquet',index=False)
    daily=frame.groupby(['seed','owner','date','horizon'])[['mae','crps']].mean().reset_index()
    daily.to_csv(ROOT/'results/midpoint-daily.csv',index=False)
    # Reuse the aggregation routine with a single meaningful midpoint target.
    records=[]
    for baseline in ['current','ce']:
        for h in [2,5]:
            for metric in ['mae','crps']:
                for seed in [*map(str,SEEDS),'mean']:
                    sub=daily[(daily.horizon==h)&daily.owner.isin([baseline,'midpoint'])]
                    if seed!='mean':
                        sub=sub[sub.seed==int(seed)]
                    p=sub.groupby(['date','owner'])[metric].mean().unstack()
                    a,b=p[baseline].to_numpy(),p.midpoint.to_numpy()
                    rng=np.random.default_rng(314159)
                    ix=(rng.integers(len(p),size=(2000,int(np.ceil(len(p)/10)),1))+np.arange(10))%len(p)
                    lo,hi=np.quantile((b-a)[ix.reshape(2000,-1)[:,:len(p)]].mean(1),[.025,.975])
                    records.append(dict(comparison=baseline,seed=seed,horizon=h,metric=metric,baseline=a.mean(),candidate=b.mean(),
                        delta=(b-a).mean(),relative_change=b.mean()/a.mean()-1,ci_low=lo,ci_high=hi,dates=len(p)))
    pd.DataFrame(records).to_csv(ROOT/'results/midpoint-paired.csv',index=False)
    core.finish(ROOT/'results')


def run():
    checked()
    torch.set_num_threads(4)
    write_json(ROOT/'runtime.json',dict(at=utc_now(),pid=os.getpid(),python=sys.version,torch=str(torch.__version__),device='mps'))
    data=Dataset()
    cfg=read(ROOT/'protocol.json')
    profiles=read(ROOT/'initial-profiles.json')
    transfer.ROOT=ROOT/'calibration'
    ids=np.load(ROOT/'calibration-ids.npy')
    for seed in SEEDS:
        for owner in ['ce','midpoint']:
            transfer.forecast(data,ids,owner,seed,profiles)
    for owner in ['ce','midpoint']:
        path=fit(data,owner,profiles)
        profiles[owner].update(fit_path=str(path),fit_sha256=file_hash(path))
    if (ROOT/'profiles.json').exists():
        assert read(ROOT/'profiles.json')==profiles
    else:
        write_json(ROOT/'profiles.json',profiles)
        write_json(ROOT/'fits-frozen.json',dict(at=utc_now(),profiles_sha256=file_hash(ROOT/'profiles.json'),factors=36,
            fits={owner:profiles[owner]['fit_sha256'] for owner in ['ce','midpoint']},calibration_only=True))
    print('36 factors frozen before 2025 generation',flush=True)
    transfer.ROOT=ROOT
    ids=np.load(ROOT/'evaluation-ids.npy')
    if not (ROOT/'evaluation-started.json').exists():
        write_json(ROOT/'evaluation-started.json',dict(at=utc_now(),fits_frozen_sha256=file_hash(ROOT/'fits-frozen.json')))
    frames=[]
    for seed in SEEDS:
        dest=ROOT/'forecasts/current'/f'seed{seed}'
        if not dest.exists():
            shutil.copytree(CURRENT/'forecasts/refreshed'/f'seed{seed}',dest)
        core.verify(dest)
        for owner in OWNERS:
            if owner!='current':
                transfer.forecast(data,ids,owner,seed,profiles)
            frames.append(transfer.score(data,ids,owner,seed,profiles))
    common=aggregate(frames,cfg)
    midpoint_scores(data,ids,common)
    checked()
    write_json(ROOT/'run-completed.json',dict(passed=True,at=utc_now(),training_runs=0,calibration_forecast_runs=6,
        transfer_forecast_runs=6,reused_transfer_runs=3,calibration_factors=36,sampled_paths=(1792+2432)*64*6,
        weights_retuned=False,sealed_holdout_opened=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','run'])
    globals()[parser.parse_args().stage]()
