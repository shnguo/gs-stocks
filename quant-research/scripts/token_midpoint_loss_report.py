"""Build the evidence tables after independent verification."""
import json

import pandas as pd
from token_midpoint_loss_run import BASE, ROOT

from quant_research.storage import file_hash, utc_now, write_json


def table(frame):
    def value(x):
        return f'{x:.6f}' if isinstance(x,float) else str(x)
    return '\n'.join(['| '+' | '.join(frame.columns)+' |',
        '| '+' | '.join(['---']*len(frame.columns))+' |',
        *['| '+' | '.join(value(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)]])


def main():
    verification=json.loads((ROOT/'independent-verification.json').read_text())
    assert verification['passed']
    selection=json.loads((ROOT/'selection.json').read_text())
    parts=['# Midpoint-loss evidence tables','',f'Generated {utc_now()}. All errors use percentage-point returns. Lower errors and interval scores are better. Coverage targets 80%; its direction alone is not a quality verdict.','',
        '## Frozen validation choice','',f"Selected arm: {selection['selected']}; coefficient {selection['weight']}. Mean midpoint CRPS: CE {selection['ce_center_crps']:.6f}, midpoint {selection['midpoint_center_crps']:.6f}."]
    for part in ['selection','evaluation']:
        paired=pd.read_csv(ROOT/f'{part}-scores/paired.csv',dtype={'seed':str})
        parts+=['',f'## {part.title()}: midpoint versus matched CE continuation','',
            table(paired[paired.seed=='mean'][['horizon','metric','baseline','candidate','relative_change','ci_low','ci_high']]),
            '', 'Relative change is a fraction; 0.01 means a 1% increase. Confidence bounds are absolute candidate-minus-baseline differences.','',
            '### Per-seed error changes','',
            table(paired[(paired.seed!='mean')&paired.metric.isin(['center_mae','center_crps','endpoint_mae','range_mae','range_crps'])][['seed','horizon','metric','relative_change','ci_low','ci_high']])]
        cov=pd.read_csv(ROOT/f'{part}-scores/coverage.csv')
        cov['legal_fraction']=cov.valid_paths/cov.total_paths
        cov['usable_fraction']=cov.usable/cov.known
        parts+=['','### Path and input coverage','',table(cov[['seed','arm','horizon','inputs','known','usable','legal_fraction','usable_fraction']])]
        rows=pd.read_parquet(ROOT/f'{part}-scores/rows.parquet')
        cohort=rows[rows.arm=='ce'].groupby(['seed','horizon']).agg(stock_dates=('row_id','count'),dates=('date','nunique')).reset_index()
        parts+=['','### Common cohorts','',table(cohort)]
    paired=pd.read_csv(ROOT/'initial-reference/paired.csv',dtype={'seed':str})
    parts+=['','## Continuations versus their original checkpoints','',
        'These comparisons use a separate common three-arm cohort. They do not enter coefficient selection. All forecasts are raw; previous calibration factors are not applied to new checkpoints.','',
        table(paired[paired.seed=='mean'][['comparison','horizon','metric','baseline','candidate','relative_change','ci_low','ci_high']])]
    daily=pd.read_csv(ROOT/'evaluation-scores/daily.csv')
    monthly=[]
    for horizon in [2,5]:
        for metric in ['center_crps','endpoint_mae','range_mae']:
            paired_dates=daily[daily.horizon==horizon].groupby(['date','arm'])[metric].mean().unstack()
            paired_dates['month']=paired_dates.index.str[:7]
            for month,group in paired_dates.groupby('month'):
                monthly.append(dict(month=month,horizon=horizon,metric=metric,dates=len(group),
                    baseline=group.ce.mean(),candidate=group.midpoint.mean(),
                    relative_change=group.midpoint.mean()/group.ce.mean()-1,
                    improved_date_fraction=float((group.midpoint<group.ce).mean())))
    monthly=pd.DataFrame(monthly)
    monthly.to_csv(ROOT/'evaluation-monthly.csv',index=False)
    parts+=['','## Descriptive monthly consistency','',
        'These month-level summaries describe variation after the coefficient was frozen. They are not additional selection criteria or independent experiments.','',table(monthly)]
    parts+=['','## Training loss and exposure','',table(pd.read_csv(ROOT/'training-summary.csv')),
        '',table(pd.read_csv(ROOT/'training-exposure.csv')),
        '', 'Only the listed auxiliary examples receive midpoint gradients; every listed CE example receives token-loss gradients. Each auxiliary example uses four draws.','',
        '## Verification','',table(pd.DataFrame([{'check':key,'value':str(value)} for key,value in verification.items()]))]
    text='\n'.join(parts)+'\n'
    assert chr(96) not in text
    output=BASE/'docs/token-midpoint-loss-evidence-20260915.md'
    output.write_text(text)
    write_json(ROOT/'evidence-report.json',dict(at=utc_now(),path=str(output),sha256=file_hash(output)))
    print(output)


if __name__=='__main__':
    main()
