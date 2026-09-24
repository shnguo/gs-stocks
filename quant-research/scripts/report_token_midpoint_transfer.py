"""Evidence tables for the independently verified midpoint transfer experiment."""
import pandas as pd
from token_midpoint_transfer_run import BASE, MID, ROOT, read

from quant_research.storage import file_hash, utc_now, write_json


def table(frame):
    def fmt(x):
        if pd.isna(x):
            return '—'
        return f'{x:.6f}' if isinstance(x,float) else str(x)
    return '\n'.join(['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |',
        *['| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)]])


def main():
    check=read(ROOT/'independent-verification.json')
    assert check['passed']
    pairs=pd.read_csv(ROOT/'results/paired.csv',dtype={'seed':str})
    midpoint=pd.read_csv(ROOT/'results/midpoint-paired.csv',dtype={'seed':str})
    daily=pd.read_csv(ROOT/'results/daily.csv')
    means=daily.groupby(['variant','horizon','target'])[['mae','coverage80','width80','interval_score80','crps']].mean().reset_index()
    means.to_csv(ROOT/'results/summary.csv',index=False)
    high_low=daily[daily.target.isin(['maximum','minimum'])].groupby(['variant','horizon'])[['mae','coverage80','width80','interval_score80','crps']].mean().reset_index()
    high_low.to_csv(ROOT/'results/high-low-summary.csv',index=False)
    monthly=[]
    for comparison in ['pipeline_transfer','loss_transfer']:
        baseline,candidate=read(ROOT/'protocol.json')['comparisons'][comparison]
        for h in [2,5]:
            for metric in ['mae','interval_score80']:
                p=daily[(daily.horizon==h)&daily.target.isin(['maximum','minimum'])].groupby(['date','variant'])[metric].mean().unstack()
                for month,g in p.groupby(p.index.str[:7]):
                    monthly.append(dict(comparison=comparison,horizon=h,metric=metric,month=month,dates=len(g),
                        baseline=g[baseline].mean(),candidate=g[candidate].mean(),relative_change=g[candidate].mean()/g[baseline].mean()-1,
                        improved_date_fraction=float((g[candidate]<g[baseline]).mean())))
    monthly=pd.DataFrame(monthly)
    monthly.to_csv(ROOT/'results/monthly.csv',index=False)
    # Descriptive context from the earlier fixed-checkpoint, common-three-arm result.
    previous=MID/'initial-reference/daily.csv'
    assert file_hash(previous)==read(MID/'final-verification.json')['files'][str(previous)]
    old=pd.read_csv(previous).rename(columns={'arm':'owner'})
    old['owner']=old.owner.replace({'initial':'current'})
    center=pd.read_csv(ROOT/'results/midpoint-daily.csv').rename(columns={'mae':'center_mae','crps':'center_crps'})
    raw=daily[daily.variant.str.endswith('_raw')].copy()
    raw['owner']=raw.variant.str.removesuffix('_raw')
    keys=['seed','owner','date','horizon']
    endpoints=raw[raw.target.isin(['maximum','minimum'])].groupby(keys).mae.mean().rename('endpoint_mae').reset_index()
    ranges=raw[raw.target=='range'][keys+['mae','crps']].rename(columns={'mae':'range_mae','crps':'range_crps'})
    new=center.merge(endpoints,on=keys,validate='one_to_one').merge(ranges,on=keys,validate='one_to_one')
    consistency=[]
    for window,frame in [('2024H2',old),('2025AprJul',new)]:
        for baseline in ['current','ce']:
            for h in [2,5]:
                for metric in ['center_mae','center_crps','endpoint_mae','range_mae','range_crps']:
                    for seed in ['17','29','43','mean']:
                        f=frame[frame.horizon==h]
                        if seed!='mean':
                            f=f[f.seed==int(seed)]
                        t=f.groupby(['date','owner'])[metric].mean().unstack()
                        consistency.append(dict(window=window,baseline=baseline,seed=seed,horizon=h,metric=metric,
                            relative_change=t.midpoint.mean()/t[baseline].mean()-1,dates=len(t)))
    consistency=pd.DataFrame(consistency)
    consistency.to_csv(ROOT/'results/cross-window-consistency.csv',index=False)
    common=pd.read_parquet(ROOT/'results/common.parquet')
    cohort=common[(common.variant=='current_raw')&(common.target=='maximum')].groupby(['seed','horizon']).agg(rows=('row_id','count'),dates=('date','nunique')).reset_index()
    parts=['# Midpoint calibration and transfer: evidence','',f'Generated {utc_now()}. Read with the main report and frozen protocol.','',
        'Errors and interval widths use percentage-point returns. Coverage is a proportion. Relative changes are fractions, so −0.01 means a 1% reduction. Confidence bounds describe absolute candidate-minus-baseline differences. Lower errors and interval scores are better; coverage targets 0.8. Calibrated CRPS is intentionally unavailable because interval scaling does not define a full calibrated distribution.','',
        '## Average high/low metrics','',table(high_low),'','## Common cohorts','',table(cohort)]
    for comparison in ['pipeline_transfer','loss_transfer','midpoint_calibration','ce_calibration','raw_pipeline','raw_loss']:
        p=pairs[(pairs.comparison==comparison)&(pairs.seed=='mean')]
        parts+=['',f'## {comparison}','',table(p[['horizon','target','metric','baseline','candidate','relative_change','ci_low','ci_high']])]
    parts+=['','## Raw midpoint distribution','',table(midpoint[midpoint.seed=='mean'][['comparison','horizon','metric','baseline','candidate','relative_change','ci_low','ci_high']]),
        '','## Per-seed high/low changes','',table(pairs[(pairs.seed!='mean')&pairs.target.eq('high_low')&pairs.comparison.isin(['pipeline_transfer','loss_transfer'])&pairs.metric.isin(['mae','interval_score80'])][['comparison','seed','horizon','metric','relative_change','ci_low','ci_high']]),
        '','## Descriptive monthly consistency','', 'These summaries are descriptive and do not select factors, checkpoints or the loss weight. July is a partial month.','',table(monthly)]
    parts+=['','## Descriptive consistency across the two tested periods','',
        'Both periods use the same fixed predictor checkpoints and a common three-profile cohort within each seed/horizon. Values are raw point/distribution comparisons. These already examined development windows are not independent untouched tests. This summary does not change the frozen selection or calibration.','',
        table(consistency[consistency.seed=='mean'])]
    factors=[]
    for owner in ['ce','midpoint']:
        factors += [dict(owner=owner,**p) for p in read(ROOT/'fits'/owner/'fit.json')['parameters']]
    factors=pd.DataFrame(factors)
    factors.to_csv(ROOT/'results/factors.csv',index=False)
    parts+=['','## Calibration factors','',table(factors[['owner','seed','horizon','target','scale','rows','dates']])]
    coverage=[]
    for owner in ['current','ce','midpoint']:
        for seed in [17,29,43]:
            g=pd.read_csv(ROOT/'scores'/owner/f'seed{seed}/coverage.csv')
            g['owner'],g['seed']=owner,seed
            g['legal_fraction']=g.valid_paths/g.total_paths
            coverage.append(g)
    coverage=pd.concat(coverage,ignore_index=True)
    coverage.to_csv(ROOT/'results/coverage.csv',index=False)
    parts+=['','## Forecast availability','',table(coverage[['owner','seed','horizon','inputs','known','usable','legal_fraction']]),
        '','## Independent verification','',table(pd.DataFrame([dict(check=k,value=str(v)) for k,v in check.items()]))]
    text='\n'.join(parts)+'\n'
    assert chr(96) not in text
    path=BASE/'docs/token-midpoint-transfer-evidence-20260915.md'
    path.write_text(text)
    write_json(ROOT/'evidence-report.json',dict(at=utc_now(),path=str(path),sha256=file_hash(path),
        descriptive_prior_source=dict(path=str(previous),sha256=file_hash(previous))))
    print(high_low.to_string(index=False))
    print(midpoint[midpoint.seed=='mean'].to_string(index=False))


if __name__=='__main__':
    main()
