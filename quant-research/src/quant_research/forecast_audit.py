"""Small, explicit numerical helpers for forecast-pipeline diagnostics."""
import numpy as np
import pandas as pd

from .token_transformer import valid_bars

TARGETS=('maximum','minimum','range')
SCORES=('mae','crps','coverage80','width80','interval_score80','bias')


def extrema(paths,reference):
    hi=paths[...,1].max(-1)
    lo=paths[...,2].min(-1)
    return np.stack([(hi/reference-1)*100,(lo/reference-1)*100,(hi-lo)/reference*100],-1)


def empirical_score(values,actual):
    values=np.asarray(values,np.float64)
    actual=np.asarray(actual,np.float64)
    if values.ndim!=2 or actual.shape!=(values.shape[1],) or not np.isfinite(values).all():
        raise ValueError('Invalid distribution')
    n=len(values)
    if n<2:
        raise ValueError('At least two paths required')
    q10,median,q90=np.quantile(values,[.1,.5,.9],axis=0)
    crps=np.abs(values-actual).mean(0)-((2*np.arange(1,n+1)-n-1)[:,None]*np.sort(values,axis=0)).sum(0)/n**2
    score=q90-q10+10*np.maximum(q10-actual,0)+10*np.maximum(actual-q90,0)
    return dict(mae=np.abs(median-actual),crps=crps,coverage80=((q10<=actual)&(actual<=q90)).astype(float),
        width80=q90-q10,interval_score80=score,bias=median-actual)


def score_paths(paths,future,known,reference,dates,horizons=(2,5),minimum_fraction=.25):
    """Scores include coverage masks; callers form common cohorts before comparison."""
    output=[]
    coverage=[]
    for h in horizons:
        legal=valid_bars(paths[:,:,:h]).all(-1)
        observed=known[:,:h].all(1)
        usable=observed & (legal.sum(1)>=max(2,int(np.ceil(paths.shape[1]*minimum_fraction))))
        truth=extrema(future[:,:h].astype(float),reference.astype(float))
        coverage.append(dict(horizon=h,inputs=len(paths),known=int(observed.sum()),usable=int(usable.sum()),
            valid_paths=int(legal.sum()),total_paths=int(legal.size),usable_fraction=float(usable.mean())))
        for i in np.flatnonzero(usable):
            values=extrema(paths[i,legal[i],:h].astype(float),float(reference[i]))
            scores=empirical_score(values,truth[i])
            for j,target in enumerate(TARGETS):
                output.append(dict(local_row=int(i),date=dates[i],horizon=h,target=target,
                    **{k:float(v[j]) for k,v in scores.items()}))
    return pd.DataFrame(output),pd.DataFrame(coverage)


def common_scores(records):
    """Require every variant on a row/horizon; preserve within-date weighting."""
    frame=pd.concat([v.assign(variant=k) for k,v in records.items()],ignore_index=True)
    keys=['local_row','horizon','target']
    counts=frame.groupby(keys).variant.nunique()
    eligible=counts[counts==len(records)].reset_index()[keys]
    common=frame.merge(eligible,on=keys,validate='many_to_one')
    daily=common.groupby(['variant','date','horizon','target'])[list(SCORES)].mean().reset_index()
    return common,daily,daily.groupby(['variant','horizon','target'])[list(SCORES)].mean().reset_index()


def choose_sampler(summary,coverage,config):
    """All inputs must originate in selection partitions, never evaluation results."""
    if set(summary.partition)!={'selection'} or set(coverage.partition)!={'selection'}:
        raise ValueError('Sampler selection may use validation partitions only')
    means=summary.groupby(['fold','variant'])[list(SCORES)].mean()
    cov=coverage.groupby(['fold','variant']).usable_fraction.min()
    rows=[]
    for variant in sorted(set(summary.variant)):
        ratios=[]
        eligible=True
        for fold in sorted(set(summary.fold)):
            x,b=means.loc[(fold,variant)],means.loc[(fold,'baseline')]
            ratios.append(float(x.crps/b.crps))
            eligible &= x.mae<=b.mae*(1+config['maximum_relative_mae_increase'])
            eligible &= cov.loc[(fold,variant)]>=cov.loc[(fold,'baseline')]-config['maximum_usable_coverage_drop']
            if config['require_interval_score_noninferior']:
                eligible &= x.interval_score80<=b.interval_score80
        rows.append(dict(variant=variant,relative_crps=float(np.mean(ratios)-1),eligible=bool(eligible)))
    choices=[x for x in rows if x['eligible'] and x['relative_crps']<=-config['minimum_relative_improvement']]
    winner=min(choices,key=lambda x:x['relative_crps'])['variant'] if choices else config['fallback']
    return dict(selected=winner,candidates=rows)
