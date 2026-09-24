"""Independent numerical, lineage, and chronological checks of pipeline experiments."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.storage import file_hash, utc_now, write_json

BASE=Path('/Users/guo/Documents/stocks/quant-research')
DATA=BASE/'artifacts/token-history-data-20260915-v1'
AUDIT=BASE/'artifacts/token-pipeline-audit-20260915-v1'
CONF=BASE/'artifacts/token-pipeline-confirmation-20260915-v1'
TARGETS=['maximum','minimum','range']
FIELDS=['crps','mae','coverage80','width80','interval_score80','bias']


def read(path):
    return json.loads(path.read_text())


def hashes(root):
    files={}
    for done in root.rglob('completed.json'):
        meta=read(done)
        assert meta['passed']
        for name,expected in meta['files'].items():
            path=done.parent/name
            assert file_hash(path)==expected,path
        files[str(done)]=file_hash(done)
    for meta in root.rglob('settings.json'):
        x=read(meta)
        assert x['fixed_seed_replay']
        assert file_hash(Path(x['checkpoint']))==x['checkpoint_sha256']
        for name,h in x['chunks'].items():
            assert file_hash(meta.parent/'chunks'/name)==h
    return files


def legal(p):
    return (np.isfinite(p).all(-1)&(p[...,:4]>0).all(-1)&(p[...,4:]>=0).all(-1)
        &(p[...,1]>=p[...,0])&(p[...,1]>=p[...,3])&(p[...,1]>=p[...,2])
        &(p[...,2]<=p[...,0])&(p[...,2]<=p[...,3])).all(-1)


def extrema(p,ref):
    high=p[...,1].astype(float).max(-1)
    low=p[...,2].astype(float).min(-1)
    if p.ndim==4:
        ref=ref[:,None]
    return np.stack([(high/ref-1)*100,(low/ref-1)*100,(high-low)/ref*100],-1)


def independent_rows(paths,ids,rows,future,known,last):
    output=[]
    for horizon in [2,5]:
        masks=legal(paths[:,:,:horizon])
        ok=known[ids,:horizon].all(1)&(masks.sum(1)>=max(2,int(np.ceil(paths.shape[1]/4))))
        for start in range(0,len(ids),64):
            idx=np.arange(start,min(start+64,len(ids)))
            idx=idx[ok[idx]]
            if not len(idx):
                continue
            mask=masks[idx]
            count=mask.sum(1)[:,None]
            p=extrema(paths[idx,:,:horizon],last[ids[idx],3].astype(float))
            y=extrema(future[ids[idx],:horizon],last[ids[idx],3].astype(float))
            p=np.where(mask[...,None],p,0.)
            first=np.where(mask[...,None],np.abs(p-y[:,None]),0).sum(1)/count
            distance=np.abs(p[:,:,None]-p[:,None,:])
            pair=mask[:,:,None]&mask[:,None,:]
            crps=first-np.where(pair[...,None],distance,0).sum((1,2))/(2*count**2)
            q10,median,q90=np.nanquantile(np.where(mask[...,None],p,np.nan),[.1,.5,.9],axis=1)
            score=q90-q10+10*np.maximum(q10-y,0)+10*np.maximum(y-q90,0)
            for k,i in enumerate(idx):
                for j,target in enumerate(TARGETS):
                    output.append(dict(local_row=int(i),date=rows.iloc[ids[i]].date,horizon=horizon,target=target,
                        crps=crps[k,j],mae=abs(median[k,j]-y[k,j]),bias=median[k,j]-y[k,j],
                        coverage80=float(q10[k,j]<=y[k,j]<=q90[k,j]),width80=q90[k,j]-q10[k,j],interval_score80=score[k,j]))
    return pd.DataFrame(output)


def compare_group(mapping,ids,expected,rows,future,known,last):
    frames=[]
    for variant,paths in mapping.items():
        frames.append(independent_rows(paths,ids,rows,future,known,last).assign(variant=variant))
    result=pd.concat(frames,ignore_index=True)
    keys=['local_row','horizon','target']
    n=result.groupby(keys).variant.transform('nunique')
    result=result[n==len(mapping)]
    keys=['variant','local_row','horizon','target']
    x=result.sort_values(keys).reset_index(drop=True)
    y=expected.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(x[keys+['date']],y[keys+['date']],check_dtype=False)
    for field in FIELDS:
        np.testing.assert_allclose(x[field],y[field],atol=1e-10,rtol=1e-10)
    return len(x)


def verify_aggregation(frame,expected_daily,expected_summary):
    daily=[]
    summary=[]
    for (variant,horizon,target),g in frame.groupby(['variant','horizon','target']):
        dates,codes=np.unique(g.date.to_numpy(),return_inverse=True)
        counts=np.bincount(codes)
        values={field:np.bincount(codes,weights=g[field].to_numpy())/counts for field in FIELDS}
        for i,day in enumerate(dates):
            daily.append(dict(variant=variant,horizon=horizon,target=target,date=day,
                **{field:float(v[i]) for field,v in values.items()}))
        summary.append(dict(variant=variant,horizon=horizon,target=target,
            **{field:float(v.mean()) for field,v in values.items()}))
    for found,expected,keys in [(pd.DataFrame(daily),expected_daily,['variant','date','horizon','target']),
                                 (pd.DataFrame(summary),expected_summary,['variant','horizon','target'])]:
        x=found.sort_values(keys).reset_index(drop=True)
        y=expected.sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(x[keys],y[keys],check_dtype=False)
        np.testing.assert_allclose(x[FIELDS],y[FIELDS],atol=1e-10,rtol=1e-10)


def reconstruction(rows,future,known,last):
    stored=pd.read_parquet(AUDIT/'reconstruction/row-errors.parquet')
    checked=0
    for path in (AUDIT/'reconstruction').glob('*.npz'):
        fold,part=path.stem.split('-')
        with np.load(path) as z:
            ids=z['row_ids']
            ref=last[ids,3].astype(float)
            for horizon in [2,5]:
                eligible=known[ids,:horizon].all(1)
                truth=extrema(future[ids,:horizon],ref)
                for stage in ['clip_only','tokenizer_clipped','tokenizer_unclipped_future']:
                    error=extrema(z[stage][:,:horizon],ref)-truth
                    actual=stored[(stored.fold==fold)&(stored.partition==part)&(stored.horizon==horizon)&(stored.stage==stage)]
                    for j,target in enumerate(TARGETS):
                        x=actual[actual.target==target].set_index('row_id').loc[ids[eligible]]
                        np.testing.assert_allclose(x.mae,np.abs(error[eligible,j]),atol=1e-10,rtol=1e-10)
                        np.testing.assert_allclose(x.bias,error[eligible,j],atol=1e-10,rtol=1e-10)
                        checked+=len(x)
    return checked


def verify_contrasts(cfg):
    sources=[]
    for seed in cfg['confirmation_seeds']:
        p=CONF/f'seed{seed}/dense_3720k_equal/evaluation/daily.csv'
        sources.append(pd.read_csv(p).assign(seed=seed))
    source=pd.concat(sources,ignore_index=True)
    stored=pd.read_csv(CONF/'results/daily.csv')
    keys=['seed','variant','date','horizon','target']
    pd.testing.assert_frame_equal(source.sort_values(keys).reset_index(drop=True),
                                  stored.sort_values(keys).reset_index(drop=True))
    pairs=pd.read_csv(CONF/'results/paired-comparisons.csv',dtype={'seed':str})
    for row in pairs.itertuples(index=False):
        frame=source[source.horizon==row.horizon]
        if row.seed!='mean':
            frame=frame[frame.seed==int(row.seed)]
        # One average per shared calendar date, then equal weighting across dates.
        calendar=sorted(set(frame.date))
        baseline=[]
        candidate=[]
        for day in calendar:
            x=frame[frame.date==day]
            baseline.append(float(x.loc[x.variant=='baseline',row.metric].mean()))
            candidate.append(float(x.loc[x.variant==row.variant,row.metric].mean()))
        b=np.asarray(baseline)
        x=np.asarray(candidate)
        delta=x-b
        np.testing.assert_allclose([row.baseline,row.candidate,row.delta],
            [b.mean(),x.mean(),delta.mean()],atol=1e-10,rtol=1e-10)
        np.testing.assert_allclose(row.relative_change,x.mean()/b.mean()-1,atol=1e-10,rtol=1e-10)
        rng=np.random.default_rng(314159)
        starts=rng.integers(len(calendar),size=(cfg['bootstrap_replicates'],int(np.ceil(len(calendar)/10))))
        samples=[]
        for blocks in starts:
            pick=[(int(start)+offset)%len(calendar) for start in blocks for offset in range(10)][:len(calendar)]
            samples.append(float(sum(delta[pick])/len(calendar)))
        np.testing.assert_allclose([row.ci_low,row.ci_high],np.quantile(samples,[.025,.975]),atol=1e-10,rtol=1e-10)
    return len(pairs)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['audit','confirmation'])
    args=parser.parse_args()
    rows=pd.read_parquet(DATA/'rows.parquet')
    future=np.load(DATA/'future.npy',mmap_mode='r')
    known=np.load(DATA/'valid.npy',mmap_mode='r')
    last=np.load(DATA/'last.npy',mmap_mode='r')
    counts={}
    if args.stage=='audit':
        counts['reconstruction_row_scores']=reconstruction(rows,future,known,last)
        for fold in ['2024h2','2025q2']:
            root=AUDIT/'sampling'/fold
            mapping={}
            ids=None
            for name in read(AUDIT/'protocol.json')['samplers']:
                ii=np.load(root/name/'row-ids.npy')
                if ids is None:
                    ids=ii
                np.testing.assert_array_equal(ids,ii)
                p=np.load(root/name/'paths.npy',mmap_mode='r')
                mapping[name]=p
                mapping[name+'_32']=p[:,:32]
            expected=pd.read_parquet(AUDIT/f'sampling/{fold}-common.parquet')
            counts[fold]=compare_group(mapping,ids,expected,rows,future,known,last)
            metrics=pd.read_csv(AUDIT/'sampling/metrics.csv')
            verify_aggregation(expected,pd.read_csv(AUDIT/f'sampling/{fold}-daily.csv'),metrics[metrics.fold==fold])
        root=AUDIT
    else:
        cfg=read(CONF/'protocol.json')
        for part in ['train','selection','evaluation']:
            r=rows.iloc[np.load(CONF/f'{part}-ids.npy')]
            assert r.date.min()>=cfg['new_fold'][part][0]
            assert r.label_end.max()<cfg['new_fold'][part][1]
            counts[part+'_dates']=int(r.date.nunique())
        sampler=read(CONF/'sampler-selection/selection.json')
        assert sampler['dates'][1]<cfg['new_fold']['evaluation'][0]
        assert sampler['partition']=='selection' and not sampler['evaluation_used']
        for seed in cfg['confirmation_seeds']:
            r=CONF/f'seed{seed}/dense_3720k_equal'
            selection=read(r/'checkpoint-selection/selection.json')
            assert selection['partition']=='selection' and not selection['evaluation_used']
            history=read(r/'training/history.json')
            visited=np.load(r/'training/visited-row-ids.npy')
            assert known[visited].any(1).all()
            assert set(visited).issubset(set(np.load(CONF/'train-ids.npy')))
            assert len(visited)==read(r/'training/summary.json')['unique_examples_visited']
            assert len(history)>=cfg['training']['minimum_epochs']
            folders=[(r/'checkpoint-selection','checkpoint'),(r/'evaluation','evaluation')]
            if seed==cfg['confirmation_seeds'][0]:
                folders.append((CONF/'sampler-selection','sampler'))
            for folder,label in folders:
                mapping={}
                ids=None
                if label=='evaluation':
                    names={k:v['path_variant'] for k,v in read(folder/'lineage.json').items()}
                else:
                    names={p.name:p.name for p in folder.iterdir() if p.is_dir() and (p/'paths.npy').exists()}
                for name,source in names.items():
                    ii=np.load(folder/source/'row-ids.npy')
                    if ids is None:
                        ids=ii
                    np.testing.assert_array_equal(ids,ii)
                    mapping[name]=np.load(folder/source/'paths.npy',mmap_mode='r')
                expected=pd.read_parquet(folder/'common.parquet')
                counts[f'{seed}-{label}']=compare_group(mapping,ids,expected,rows,future,known,last)
                verify_aggregation(expected,pd.read_csv(folder/'daily.csv'),pd.read_csv(folder/'metrics.csv'))
        counts['paired_contrasts_and_block_intervals']=verify_contrasts(cfg)
        root=CONF
    files=hashes(root)
    for path,h in read(BASE/'artifacts/token-history-experiment-20260915-v1/final-verification.json')['files'].items():
        assert file_hash(Path(path))==h,path
    write_json(root/'independent-verification.json',dict(passed=True,at=utc_now(),counts=counts,files=files,
        method='Independent legal-path masks, explicit pairwise empirical CRPS, extrema, all six score fields, common row cohorts, independent bincount date means and summary means',
        prior_experiment_unchanged=True,sealed_holdout_opened=False))
    print(counts,flush=True)


if __name__=='__main__':
    main()
