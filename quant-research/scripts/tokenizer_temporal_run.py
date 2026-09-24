"""Fixed-checkpoint temporal transfer using the verified decoder experiment code."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tokenizer_reconstruction_run as core
import torch

from quant_research.storage import file_hash, utc_now, write_json

BASE=Path('/Users/guo/Documents/stocks/quant-research')
ROOT=BASE/'artifacts/tokenizer-temporal-20260915-v1'
PRIOR=BASE/'artifacts/tokenizer-reconstruction-20260915-v1'


def read(p):
    return json.loads(p.read_text())


def initialize():
    cfg=read(BASE/'configs/tokenizer-temporal-v1.json')
    if (ROOT/'protocol.json').exists():
        assert cfg==read(ROOT/'protocol.json')
        return
    old=read(PRIOR/'final-verification.json')
    assert old['passed']
    for name,h in old['files'].items():
        if Path(name).is_relative_to(BASE/'artifacts'):
            assert file_hash(Path(name))==h,name
    ROOT.mkdir()
    write_json(ROOT/'protocol.json',cfg)
    shutil.copytree(PRIOR/'code',ROOT/'code')
    shutil.copy2(BASE/'scripts/tokenizer_temporal_run.py',ROOT/'code/scripts/tokenizer_temporal_run.py')
    (ROOT/'decoder-selection').mkdir()
    shutil.copy2(PRIOR/'decoder-selection/selection.json',ROOT/'decoder-selection/selection.json')
    choice=read(ROOT/'decoder-selection/selection.json')
    source={str(PRIOR/'final-verification.json'):file_hash(PRIOR/'final-verification.json'),
        choice['checkpoint']:file_hash(Path(choice['checkpoint'])),
        str(core.DATA/'completed.json'):file_hash(core.DATA/'completed.json')}
    for seed in cfg['forecast_seeds']:
        old=PRIOR/f'predictors/seed{seed}/dense_3720k_equal/training'
        dest=ROOT/f'predictors/seed{seed}/dense_3720k_equal/training'
        dest.mkdir(parents=True)
        for name in ['best.pt','summary.json','history.json','visited-row-ids.npy','reload-reference.npz']:
            shutil.copy2(old/name,dest/name)
            source[str(old/name)]=file_hash(old/name)
            assert file_hash(dest/name)==source[str(old/name)]
        write_json(dest/'reused-checkpoint.json',dict(source=str(old),refitted=False,at=utc_now()))
        core.finish(dest)
    write_json(ROOT/'source-manifest.json',dict(at=utc_now(),prior_files=source,
        files={str(p.relative_to(ROOT)):file_hash(p) for p in (ROOT/'code').rglob('*.py')},
        protocol_sha256=file_hash(ROOT/'protocol.json')))


def bind():
    core.OUT=ROOT
    core.history.OUT=ROOT/'predictors'


def preflight(data,cfg):
    old_ids=np.load(PRIOR/'evaluation-ids.npy')
    ids=data.splits['evaluation']
    assert not np.intersect1d(old_ids,ids).size
    for part in ['train','selection']:
        np.testing.assert_array_equal(data.splits[part],np.load(PRIOR/f'{part}-ids.npy'))
    assert data.rows.iloc[ids].date.min()>data.rows.iloc[old_ids].label_end.max()
    selected=data.selected(None,'evaluation',cfg['forecast_evaluation_rows_per_date'])
    assert len(np.unique(selected))==len(selected)
    assert data.rows.iloc[selected].groupby('date').size().eq(32).all()
    assert data.rows.iloc[selected].label_end.max()<cfg['fold']['evaluation'][1]<cfg['sealed_holdout_start']
    write_json(ROOT/'preflight.json',dict(passed=True,at=utc_now(),refit=False,
        previous_evaluation_disjoint=True,training_and_selection_unchanged=True,
        evaluation_inputs=len(selected),evaluation_dates=int(data.rows.iloc[selected].date.nunique()),
        first_signal=str(data.rows.iloc[selected].date.min()),last_signal=str(data.rows.iloc[selected].date.max()),
        last_target=str(data.rows.iloc[selected].label_end.max()),sealed_holdout_opened=False))


def target_contrasts(cfg):
    daily=pd.read_csv(ROOT/'forecast-results/daily.csv')
    records=[]
    for h in cfg['horizons']:
        for target in cfg['comparison_targets']:
            subset=daily[daily.horizon==h]
            if target!='mean':
                subset=subset[subset.target==target]
            for metric in ['crps','mae','coverage80','interval_score80']:
                paired=subset.groupby(['seed','date','variant'])[metric].mean().unstack('variant')
                for seed in cfg['forecast_seeds']+['mean']:
                    p=paired if seed=='mean' else paired.loc[[seed]]
                    p=p.groupby('date')[['frozen','adapted']].mean()
                    b,x=p.frozen.to_numpy(),p.adapted.to_numpy()
                    rng=np.random.default_rng(314159)
                    block=cfg['bootstrap_block_dates']
                    indices=(rng.integers(len(x),size=(cfg['bootstrap_replicates'],int(np.ceil(len(x)/block)),1))+np.arange(block))%len(x)
                    indices=indices.reshape(len(indices),-1)[:,:len(x)]
                    lo,hi=np.quantile((x-b)[indices].mean(1),[.025,.975])
                    records.append(dict(seed=seed,horizon=h,target=target,metric=metric,baseline=b.mean(),adapted=x.mean(),
                        delta=(x-b).mean(),relative_change=x.mean()/b.mean()-1,ci_low=lo,ci_high=hi,dates=len(x)))
    pd.DataFrame(records).to_csv(ROOT/'forecast-results/target-paired.csv',index=False)
    core.finish(ROOT/'forecast-results')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['init','all'])
    args=parser.parse_args()
    initialize()
    bind()
    cfg=read(ROOT/'protocol.json')
    manifest=read(ROOT/'source-manifest.json')
    assert file_hash(ROOT/'protocol.json')==manifest['protocol_sha256']
    for name,h in manifest['files'].items():
        assert file_hash(ROOT/name)==h,name
    for name,h in manifest['prior_files'].items():
        assert file_hash(Path(name))==h,name
    torch.set_num_threads(4)
    data=core.ReconstructionData(cfg)
    preflight(data,cfg)
    if args.stage=='init':
        return
    choice=read(ROOT/'decoder-selection/selection.json')
    reconstruction=ROOT/'reconstruction-evaluation'
    ids=data.selected(None,'evaluation',cfg['reconstruction_evaluation_rows_per_date'])
    for name,checkpoint in [('frozen',None),('adapted',Path(choice['checkpoint']))]:
        dest=reconstruction/name
        if (dest/'completed.json').exists():
            core.verify(dest)
        else:
            model=core.load_decoder(checkpoint)
            core.reconstruct(model,data,ids,dest)
            del model
            torch.mps.empty_cache()
    for seed in cfg['forecast_seeds']:
        root=ROOT/f'predictors/seed{seed}/dense_3720k_equal'
        core.verify(root/'training')
        core.forecasts(data,root,choice,cfg)
    core.aggregate(cfg)
    target_contrasts(cfg)
    write_json(ROOT/'run-completed.json',dict(passed=True,at=utc_now(),refitted=False,
        decoder=choice['selected'],forecast_seeds=cfg['forecast_seeds'],sealed_holdout_opened=False))


if __name__=='__main__':
    main()
