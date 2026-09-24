"""Frozen midpoint/width diagnosis on saved external-benchmark paths."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.forecast_audit import empirical_score
from quant_research.range_decomposition import attribute_mae_change, point_components
from quant_research.storage import file_hash, utc_now, write_json
from quant_research.token_transformer import valid_bars

BASE=Path('/Users/guo/Documents/stocks/quant-research')
ART=BASE/'artifacts'
ROOT=ART/'token-range-decomposition-20260915-v1'
PARENT=ART/'token-external-benchmark-20260915-v1'
TRANSFER=ART/'token-profile-transfer-20260915-v1'
DATA=ART/'token-history-data-20260915-v1'
SEEDS=[17,29,43]
FIELDS=['endpoint_mae','endpoint_mse','center_mae','half_width_mae','center_mse','half_width_mse',
    'center_error','half_width_error','path_center_mae','path_center_crps','path_range_mae',
    'path_range_crps','path_center_bias','path_range_bias']


def read(p):
    return json.loads(p.read_text())


def paths_map():
    return {'historical':PARENT/'historical/paths.npy','kronos':PARENT/'kronos/paths.npy',
        **{f'ours_{s}':TRANSFER/f'forecasts/refreshed/seed{s}/paths.npy' for s in SEEDS}}


def prepare():
    assert not ROOT.exists()
    assert read(PARENT/'final-verification.json')['passed']
    sources={}
    for name,h in read(PARENT/'final-verification.json')['files'].items():
        if Path(name).is_relative_to(ART):
            assert file_hash(Path(name))==h,name
    for p in [PARENT/'final-verification.json',PARENT/'independent-verification.json',PARENT/'row-ids.npy',
              PARENT/'rows.parquet',PARENT/'results/common.parquet',PARENT/'results/paired.csv']:
        sources[str(p)]=file_hash(p)
    for p in paths_map().values():
        assert file_hash(p)==read(p.parent/'completed.json')['files'][p.name]
        sources[str(p)]=file_hash(p)
    for name in ['future.npy','valid.npy','last.npy','rows.parquet']:
        p=DATA/name
        assert file_hash(p)==read(DATA/'completed.json')['files'][name]
        sources[str(p)]=file_hash(p)
    ROOT.mkdir()
    cfg=dict(protocol_id='token-range-decomposition-v1',at=utc_now(),horizons=[2,5],seeds=SEEDS,
        cohort='Exact external-benchmark shared cohort: 2372 two-day and 2253 five-day inputs on 76 dates; five raw models with at least 16 legal paths.',
        primary='Self-built versus historical volatility; Kronos Base retained as context.',
        definitions=dict(center='(median(high) + median(low))/2',half_width='(median(high) - median(low))/2',
            target_center='(actual highest high + actual lowest low)/2',target_half_width='(actual high - actual low)/2',
            unit='All endpoints are returns in percentage points of signal close.',
            endpoint_mae='mean absolute high/low error = max(abs(center error), abs(half-width error))',
            endpoint_mse='mean squared high/low error = center squared error + half-width squared error',
            direct_distribution='Separately score per-path midpoint and full range distributions. Their medians can differ from components implied by endpoint medians.'),
        attribution='For endpoint MAE, average the two orders of replacing historical center/half-width with the self-built components. The contributions add exactly to the error difference. Arithmetic diagnostic only, no causal training attribution or hybrid model selection.',
        hypothesis='The width/range forecasts improve while centre/location forecasts remain weaker than historical volatility; test rather than assume.',
        aggregation='Equal input weights within date, then equal dates; self-built mean averages seed scores, not paths.',
        bootstrap=dict(replicates=2000,block_dates=10,seed=314159),
        frozen='No new inference, training, calibration, prediction rules or evaluation-based fitting.',
        sealed_holdout_start='2025-08-07',limitations='Already-examined development period; conditional uncertainty, unresolved pretrained weights date coverage, no trading-profit claim.')
    write_json(ROOT/'protocol.json',cfg)
    shutil.copytree(PARENT/'code/src',ROOT/'code/src',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(BASE/'src/quant_research/range_decomposition.py',ROOT/'code/src/quant_research/range_decomposition.py')
    (ROOT/'code/scripts').mkdir()
    shutil.copy2(BASE/'scripts/token_range_decomposition.py',ROOT/'code/scripts/token_range_decomposition.py')
    write_json(ROOT/'sources.json',sources)
    write_json(ROOT/'prepared.json',dict(passed=True,files={str(p.relative_to(ROOT)):file_hash(p) for p in ROOT.rglob('*') if p.is_file()}))


def checked():
    for name,h in read(ROOT/'sources.json').items():
        assert file_hash(Path(name))==h,name
    for name,h in read(ROOT/'prepared.json')['files'].items():
        assert file_hash(ROOT/name)==h,name


def ci(delta):
    delta=np.asarray(delta,float)
    rng=np.random.default_rng(314159)
    idx=(rng.integers(len(delta),size=(2000,int(np.ceil(len(delta)/10)),1))+np.arange(10))%len(delta)
    return np.quantile(delta[idx.reshape(2000,-1)[:,:len(delta)]].mean(1),[.025,.975])


def run():
    checked()
    assert not (ROOT/'rows.parquet').exists()
    cohort=pd.read_parquet(PARENT/'results/common.parquet')[['local_row','row_id','date','horizon']].drop_duplicates()
    ids=np.load(PARENT/'row-ids.npy')
    last=np.load(DATA/'last.npy',mmap_mode='r')
    future=np.load(DATA/'future.npy',mmap_mode='r')
    known=np.load(DATA/'valid.npy',mmap_mode='r')
    records=[]
    for name,path in paths_map().items():
        paths=np.load(path,mmap_mode='r')
        for h in [2,5]:
            sub=cohort[cohort.horizon==h].sort_values('local_row')
            assert len(sub)=={2:2372,5:2253}[h] and sub.date.nunique()==76
            legal=valid_bars(paths[:,:,:h]).all(-1)
            for row in sub.itertuples():
                i,rid=row.local_row,row.row_id
                assert ids[i]==rid and known[rid,:h].all() and legal[i].sum()>=16
                ref=float(last[rid,3])
                p=paths[i,legal[i],:h].astype(float)
                high=(p[...,1].max(-1)/ref-1)*100
                low=(p[...,2].min(-1)/ref-1)*100
                y=future[rid,:h].astype(float)
                yh=(y[:,1].max()/ref-1)*100
                yl=(y[:,2].min()/ref-1)*100
                mh,ml=np.median(high),np.median(low)
                components=point_components(mh,ml,yh,yl)
                values=np.column_stack([(high+low)/2,high-low])
                target=np.array([(yh+yl)/2,yh-yl])
                scores=empirical_score(values,target)
                np.testing.assert_allclose(components['endpoint_mae'],max(components['center_mae'],components['half_width_mae']),atol=1e-12)
                np.testing.assert_allclose(components['endpoint_mse'],components['center_mse']+components['half_width_mse'],atol=1e-10)
                records.append(dict(variant=name,local_row=i,row_id=rid,date=row.date,horizon=h,draws=int(legal[i].sum()),
                    high_median=mh,low_median=ml,actual_high=yh,actual_low=yl,**components,
                    path_center_median=np.median(values[:,0]),path_range_median=np.median(values[:,1]),
                    path_center_mae=scores['mae'][0],path_range_mae=scores['mae'][1],
                    path_center_crps=scores['crps'][0],path_range_crps=scores['crps'][1],
                    path_center_bias=scores['bias'][0],path_range_bias=scores['bias'][1]))
        print('Scored components:',name,flush=True)
    frame=pd.DataFrame(records)
    frame.to_parquet(ROOT/'rows.parquet',index=False)
    daily=frame.groupby(['variant','date','horizon'])[FIELDS].mean().reset_index()
    avg=daily[daily.variant.str.startswith('ours_')].groupby(['date','horizon'])[FIELDS].mean().reset_index()
    daily=pd.concat([daily,avg.assign(variant='ours_mean')],ignore_index=True)
    daily.to_csv(ROOT/'daily.csv',index=False)
    daily.groupby(['variant','horizon'])[FIELDS].mean().reset_index().to_csv(ROOT/'summary.csv',index=False)
    attrib=[]
    for against in ['historical','kronos']:
        b=frame[frame.variant==against].set_index(['row_id','horizon']).sort_index()
        for seed in SEEDS:
            x=frame[frame.variant==f'ours_{seed}'].set_index(['row_id','horizon']).sort_index()
            pd.testing.assert_index_equal(b.index,x.index)
            a=attribute_mae_change(b.center_error,b.half_width_error,x.center_error,x.half_width_error)
            np.testing.assert_allclose(a['total'],x.endpoint_mae-b.endpoint_mae,atol=1e-12)
            np.testing.assert_allclose(a['center']+a['half_width'],a['total'],atol=1e-12)
            out=x[['date']].reset_index().assign(against=against,seed=str(seed),**a)
            attrib.append(out)
    attribution=pd.concat(attrib,ignore_index=True)
    attribution.to_parquet(ROOT/'attribution-rows.parquet',index=False)
    ad=attribution.groupby(['against','seed','date','horizon'])[['center','half_width','total']].mean().reset_index()
    avg=ad.groupby(['against','date','horizon'])[['center','half_width','total']].mean().reset_index().assign(seed='mean')
    ad=pd.concat([ad,avg],ignore_index=True)
    ad.to_csv(ROOT/'attribution-daily.csv',index=False)
    pairs=[]
    for against in ['historical','kronos']:
        for seed in [*map(str,SEEDS),'mean']:
            for h in [2,5]:
                d=daily[daily.horizon==h].pivot(index='date',columns='variant',values=FIELDS).sort_index()
                for metric in FIELDS:
                    b,x=d[metric][against],d[metric][f'ours_{seed}']
                    low,high=ci(x-b)
                    pairs.append(dict(against=against,seed=seed,horizon=h,metric=metric,baseline=b.mean(),candidate=x.mean(),
                        delta=(x-b).mean(),relative_change=(x.mean()/b.mean()-1) if metric not in ['center_error','half_width_error','path_center_bias','path_range_bias'] else np.nan,
                        ci_low=low,ci_high=high,dates=len(d)))
    pd.DataFrame(pairs).to_csv(ROOT/'paired.csv',index=False)
    attributed=[]
    for (against,seed,h),g in ad.groupby(['against','seed','horizon']):
        g=g.sort_values('date')
        for field in ['center','half_width','total']:
            low,high=ci(g[field])
            attributed.append(dict(against=against,seed=seed,horizon=h,component=field,delta=g[field].mean(),ci_low=low,ci_high=high,dates=len(g)))
    pd.DataFrame(attributed).to_csv(ROOT/'attribution-summary.csv',index=False)
    write_json(ROOT/'run-completed.json',dict(passed=True,at=utc_now(),rows=len(frame),attribution_rows=len(attribution),
        paired_estimates=len(pairs),attributed_estimates=len(attributed),new_inference=False,trained=False,calibration_refitted=False,sealed_holdout_opened=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','run'])
    globals()[parser.parse_args().stage]()
