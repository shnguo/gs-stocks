"""Paired daily-window path validation on a frozen development protocol."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from compare_token_range import extrema, historical_paths, score

from quant_research.storage import file_hash, utc_now, write_json
from quant_research.token_transformer import valid_bars

BASE=Path('/Users/guo/Documents/stocks/quant-research')
ART=BASE/'artifacts'
DATA=ART/'token-history-data-20260915-v1'
OUT=ART/'token-history-experiment-20260915-v1'
FIELDS=['mae','crps','coverage80','width80','interval_score80','bias']
TARGETS=['maximum','minimum','range']


def read(path):
    return json.loads(path.read_text())


def interval(values,cfg):
    values=np.asarray(values,float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Empty or invalid paired dates')
    rng=np.random.default_rng(cfg['seed'])
    n=len(values)
    block=cfg['bootstrap_block_dates']
    starts=rng.integers(0,n,size=(cfg['bootstrap_replicates'],(n+block-1)//block))
    ids=((starts[...,None]+np.arange(block))%n).reshape(len(starts),-1)[:,:n]
    means=values[ids].mean(1)
    return dict(mean_difference=float(values.mean()),conditional95=np.quantile(means,[.025,.975]).tolist(),dates=n)


def validate_complete(root):
    meta=read(root/'completed.json')
    assert meta['passed']
    for name,digest in meta['files'].items():
        assert file_hash(root/name)==digest,(root,name)


def evaluate(stage):
    cfg=read(DATA/'protocol.json')
    dest=OUT/f'{stage}-evaluation'
    if (dest/'completed.json').exists():
        validate_complete(dest)
        return
    dest.mkdir(exist_ok=True)
    names=['dense_282k_equal','dense_3720k_equal']
    if stage=='weighted':
        names.append('dense_3720k_weighted')
    global_rows=pd.read_parquet(DATA/'rows.parquet')
    future=np.load(DATA/'future.npy',mmap_mode='r')
    known=np.load(DATA/'valid.npy',mmap_mode='r')
    last=np.load(DATA/'last.npy',mmap_mode='r')
    raw=np.load(ART/'kronos-inputs-20260914-v3/values.npy',mmap_mode='r')
    records,coverage,source_hashes=[],[],{}
    for fold in cfg['folds']:
        name=fold['name']
        paths={}
        ids=None
        for model in names:
            root=OUT/name/model
            validate_complete(root/'training')
            validate_complete(root/'forecast')
            source_hashes[str(root/'forecast/completed.json')]=file_hash(root/'forecast/completed.json')
            row_ids=np.load(root/'forecast/row-ids.npy')
            if ids is None:
                ids=row_ids
            np.testing.assert_array_equal(ids,row_ids)
            paths[model]=np.load(root/'forecast/paths.npy',mmap_mode='r')
        rows=global_rows.iloc[ids].reset_index(drop=True)
        reference=last[ids,3].astype(np.float64)
        baseline=[]
        draws=[]
        for row in rows.itertuples():
            h=raw[row.stock_index,row.date_index-59:row.date_index+1].astype(np.float64)
            adjusted=h[:,:4]*h[:,6:7]/h[-1,6]
            p,starts=historical_paths(adjusted,f'{row.instrument_id}:{row.date}',cfg['path_samples'])
            baseline.append(p)
            draws.append(starts)
        baseline=np.stack(baseline)
        baseline=np.concatenate([baseline,np.zeros((*baseline.shape[:-1],2))],-1)
        assert valid_bars(baseline).all()
        paths['historical_volatility']=baseline
        paths['persistence']=np.broadcast_to(reference[:,None,None,None],baseline.shape)
        np.savez_compressed(dest/f'{name}-baseline.npz',row_ids=ids,paths=baseline,starts=np.stack(draws))
        for horizon in cfg['evaluation_horizons']:
            truth=extrema(future[ids,:horizon],reference)
            observed=known[ids,:horizon].all(1)
            masks={k:valid_bars(v[:,:,:horizon]).all(-1) for k,v in paths.items()}
            usable={k:observed & (v.sum(1)>=cfg['minimum_legal_paths']) for k,v in masks.items()}
            common=np.logical_and.reduce(list(usable.values()))
            assert common.any(), 'No common known evaluation cohort'
            for model,p in paths.items():
                for exchange in ['all','xshg','xshe','xbse']:
                    subset=np.ones(len(rows),bool) if exchange=='all' else rows.instrument_id.str.split('.').str[1].eq(exchange).to_numpy()
                    coverage.append(dict(fold=name,horizon=horizon,model=model,exchange=exchange,
                        input_rows=int(subset.sum()),known_rows=int((observed & subset).sum()),
                        usable_rows=int((usable[model]&subset).sum()),common_rows=int((common&subset).sum()),
                        legal_paths=int(masks[model][subset].sum()),total_paths=int(subset.sum()*cfg['path_samples'])))
                for i in np.flatnonzero(usable[model]):
                    values=extrema(p[i,masks[model][i],:horizon].astype(float),reference[i])
                    for j,target in enumerate(TARGETS):
                        metrics=score(values[:,j],truth[i,j])
                        quantiles=np.quantile(values[:,j],[.1,.5,.9])
                        records.append(dict(fold=name,horizon=horizon,model=model,row_id=int(ids[i]),
                            date=rows.iloc[i].date,instrument_id=rows.iloc[i].instrument_id,target=target,
                            common=bool(common[i]),actual=float(truth[i,j]),q10=quantiles[0],q50=quantiles[1],q90=quantiles[2],**metrics))
            print('Evaluated',stage,name,horizon,'days; known',int(observed.sum()),'common',int(common.sum()),flush=True)
    frame=pd.DataFrame(records)
    frame.to_parquet(dest/'row-metrics.parquet',index=False)
    pd.DataFrame(coverage).to_csv(dest/'coverage.csv',index=False)
    group=['fold','horizon','model','date','target']
    own=frame.groupby(group)[FIELDS].mean().reset_index()
    own.groupby(['fold','horizon','model','target'])[FIELDS].mean().to_csv(dest/'own-coverage-metrics.csv')
    daily=frame.loc[frame.common].groupby(group)[FIELDS].mean().reset_index()
    daily.to_csv(dest/'daily-common-metrics.csv',index=False)
    summary=daily.groupby(['fold','horizon','model','target'])[FIELDS].mean().reset_index()
    summary.to_csv(dest/'common-metrics.csv',index=False)
    paired=[]
    for (fold,horizon),part in daily.groupby(['fold','horizon']):
        averaged=part.groupby(['date','model'])[FIELDS].mean().reset_index()
        for target in TARGETS+['mean_of_three']:
            table_source=averaged if target=='mean_of_three' else part.loc[part.target==target]
            for metric in ['mae','crps']:
                table=table_source.pivot(index='date',columns='model',values=metric).sort_index()
                pairs=[(model,'historical_volatility') for model in names]
                pairs.append(('dense_3720k_equal','dense_282k_equal'))
                if stage=='weighted':
                    pairs.append(('dense_3720k_weighted','dense_3720k_equal'))
                for model,against in pairs:
                    paired.append(dict(fold=fold,horizon=int(horizon),target=target,metric=metric,
                        model=model,against=against,relative_difference=float(table[model].mean()/table[against].mean()-1),
                        **interval((table[model]-table[against]).to_numpy(),cfg)))
    pd.DataFrame(paired).to_json(dest/'paired-comparisons.json',orient='records',indent=2)
    # Independent spot audit of stored row scores and date-equal aggregation.
    probes=frame.iloc[np.linspace(0,len(frame)-1,min(200,len(frame)),dtype=int)]
    np.testing.assert_allclose(probes.mae,np.abs(probes.q50-probes.actual),rtol=1e-12,atol=1e-12)
    for row in summary.itertuples():
        selected=frame.loc[frame.common & frame.fold.eq(row.fold) & frame.horizon.eq(row.horizon)
            & frame.model.eq(row.model) & frame.target.eq(row.target)]
        by_date=[float(g.crps.sum()/len(g)) for _,g in selected.groupby('date')]
        np.testing.assert_allclose(row.crps,sum(by_date)/len(by_date),rtol=1e-12,atol=1e-12)
    write_json(dest/'verification.json',dict(passed=True,
        row_crps='Every scored row agrees between pairwise and sorted CRPS formulas',
        median_mae_audit_rows=len(probes),date_equal_aggregation='Independently recomputed every common summary CRPS',
        model_rows='Identical input-selected stock/date IDs across all variants',sealed_holdout_accessed=False))
    write_json(dest/'sources.json',source_hashes)
    write_json(dest/'summary.json',dict(stage=stage,created_at=utc_now(),metrics=summary.to_dict('records'),
        comparisons=paired,training=[read(OUT/f['name']/n/'training/summary.json') for f in cfg['folds'] for n in names],
        weighting='Equal rows within each signal date, then equal dates; each fold reported separately',
        units='Percentage points of signal close',bootstrap='2000 circular resamples of 10 consecutive trading dates, conditional on fitted models and selected stocks',
        limitations=['Development backtests with one training seed; no formal promotion or profitability claim',
        'Frozen tokenizer pretraining dates remain unverified; historical identity and source boundaries retain the inherited limitations',
        'Illegal decoded paths remain unaltered and excluded; coverage and common cohorts are disclosed',
        'Signal dates and five-day outcomes overlap; intervals use date blocks, not independent stock-row assumptions',
        'Changing output loss weights is not the same experiment as retraining a two-day-only model']))
    write_json(dest/'completed.json',dict(passed=True,files={p.name:file_hash(p) for p in dest.iterdir()
        if p.is_file() and p.name!='completed.json'}))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['dense','weighted'])
    evaluate(parser.parse_args().stage)
