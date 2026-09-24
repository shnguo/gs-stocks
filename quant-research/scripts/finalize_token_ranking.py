"""Freeze forecast-only Top20 before attaching labels; nominate and verify."""
import argparse
import fcntl
from pathlib import Path

import numpy as np
import pandas as pd
from token_features_run import read, verify_manifest
from token_history_run import Dataset
from token_ranking_run import comparisons
from verify_token_ranking import independent_stats, verify

from quant_research.storage import file_hash, utc_now, write_json

BASE=Path(__file__).resolve().parents[1]


def freeze_and_score(frame, top_n=20):
    """Future-label availability cannot change eligibility, ranks or membership."""
    records, ranked=[],[]
    for (arm,seed,day),g in frame.groupby(['arm','seed','date']):
        g=g.sort_values(['predicted','instrument_id'],ascending=[False,True]).copy()
        g['rank']=np.arange(1,len(g)+1)
        ranked.append(g)
        if len(g)<top_n:
            continue
        top=g.head(top_n)
        observed=g[g.label_known]
        top_observed=top[top.label_known]
        records.append(dict(arm=arm,seed=seed,date=day,rows=len(g),known=len(observed),top_rows=top_n,
            top_known=len(top_observed),top_unknown=top_n-len(top_observed),
            return_mae_pp=float(observed.absolute_error.mean()*100),
            top_extrema_scenario_pct=float(top_observed.actual_extrema_scenario.mean()*100),
            top_lift_pp=float((top_observed.actual_extrema_scenario.mean()-observed.actual_extrema_scenario.mean())*100),
            rank_ic=float(observed.predicted.rank().corr(observed.actual_extrema_scenario.rank())) if len(observed)>2 and observed.predicted.nunique()>1 and observed.actual_extrema_scenario.nunique()>1 else np.nan,
            top_q10_pct=float(top_observed.actual_extrema_scenario.quantile(.1)*100),
            top_adverse_excursion_pct=float(top_observed.adverse_excursion_pct.mean()),
            top_high_volatility_fraction=float((top.volatility_pp>=g.volatility_pp.quantile(.8)).mean()),
            largest_exchange_fraction=float(top.instrument_id.str.split('.').str[1].value_counts(normalize=True).max()),
            top_prediction_bias_pp=float((top_observed.predicted-top_observed.actual_extrema_scenario).mean()*100)))
    return pd.concat(ranked,ignore_index=True),pd.DataFrame(records)


def build_part(root,dest,cfg,data,part):
    ids=np.load(root/f'{part}-ids.npy')
    rows=data.rows.iloc[ids].reset_index(drop=True)
    volatility=np.load(root/f'{part}-volatility.npy')
    result=[]
    for arm in cfg['variants']:
        arrays={str(s):np.load(root/f'seed{s}'/arm/part/'paths.npy',mmap_mode='r') for s in cfg['seeds']}
        for start in range(0,len(ids),256):
            end=min(start+256,len(ids))
            local=np.arange(start,end)
            stats={s:independent_stats(x[start:end],cfg['minimum_legal_paths']) for s,x in arrays.items()}
            future=data.a['future'][ids[local]].astype(float)
            known=data.a['valid'][ids[local]].all(1)
            for seed in [str(s) for s in cfg['seeds']]+['ensemble']:
                used=list(stats.values()) if seed=='ensemble' else [stats[seed]]
                okay=np.logical_and.reduce([x[0] for x in used])  # FORECASTS ONLY.
                day=np.mean([x[1] for x in used],axis=0).argmax(1)+1
                buy=np.mean([x[2] for x in used],axis=0)
                sell=np.mean([x[3] for x in used],axis=0)[np.arange(end-start),day-1]
                predicted=sell/buy-1-cfg['cost']
                outcome=np.where(known,future[np.arange(end-start),day,1]/future[:,0,2]-1-cfg['cost'],np.nan)
                adverse=np.array([(future[i,:day[i]+1,2].min()/future[i,0,2]-1)*100 if known[i] else np.nan for i in range(len(local))])
                result.append(pd.DataFrame(dict(local_row=local[okay],instrument_id=rows.iloc[local[okay]].instrument_id.to_numpy(),
                    date=rows.iloc[local[okay]].date.to_numpy(),arm=arm,seed=seed,label_known=known[okay],
                    sell_offset=day[okay],buy_reference=buy[okay],sell_reference=sell[okay],predicted=predicted[okay],
                    actual_extrema_scenario=outcome[okay],absolute_error=abs(predicted-outcome)[okay],
                    volatility_pp=volatility[local][okay],adverse_excursion_pct=adverse[okay])))
    all_rows=pd.concat(result,ignore_index=True)
    native,native_daily=freeze_and_score(all_rows,cfg['top_n'])
    native.to_parquet(dest/f'{part}-native-ranking.parquet',index=False)
    native_daily.to_csv(dest/f'{part}-native-daily.csv',index=False)
    counts=all_rows.groupby('local_row').size()
    shared=counts[counts==len(cfg['variants'])*(len(cfg['seeds'])+1)].index
    paired,daily=freeze_and_score(all_rows[all_rows.local_row.isin(shared)],cfg['top_n'])
    paired.to_parquet(dest/f'{part}-paired-ranking.parquet',index=False)
    daily.to_csv(dest/f'{part}-daily.csv',index=False)
    summary=daily.groupby(['arm','seed']).mean(numeric_only=True).reset_index()
    summary.to_csv(dest/f'{part}-summary.csv',index=False)
    ensemble=daily[daily.seed=='ensemble']
    full_dates=ensemble.groupby('date').top_unknown.sum().eq(0)
    complete=daily[daily.date.isin(full_dates[full_dates].index)]
    complete.groupby(['arm','seed']).mean(numeric_only=True).to_csv(dest/f'{part}-complete-top20-dates.csv')
    daily.assign(period=np.where(daily.date<'2025-01-01','2024','2025')).groupby(['arm','seed','period']).mean(numeric_only=True).to_csv(dest/f'{part}-periods.csv')
    daily.groupby(['arm','seed']).top_extrema_scenario_pct.quantile(.1).to_csv(dest/f'{part}-poor-date-decile.csv')
    # Assert the essential invariant by hiding every outcome and reranking.
    no_labels=paired.copy()
    no_labels['label_known']=False
    no_labels['actual_extrema_scenario']=np.nan
    no_labels['absolute_error']=np.nan
    again,_=freeze_and_score(no_labels,cfg['top_n'])
    keys=['arm','seed','date','instrument_id']
    pd.testing.assert_series_equal(paired.set_index(keys)['rank'].sort_index(),again.set_index(keys)['rank'].sort_index())
    return daily,summary


def verify_frozen_report(root,dest,cfg):
    checked=0
    for part in ['selection','evaluation']:
        native=pd.read_parquet(dest/f'{part}-native-ranking.parquet')
        paired=pd.read_parquet(dest/f'{part}-paired-ranking.parquet')
        coverage=pd.read_csv(root/f'{part}-coverage.csv',dtype={'seed':str})
        keys=['arm','seed','local_row']
        expected=coverage[coverage.forecast_eligible].rename(columns={'variant':'arm'})
        assert set(map(tuple,native[keys].to_numpy()))==set(map(tuple,expected[keys].to_numpy()))
        prior=pd.read_csv(root/f'{part}-all-ranking.csv',dtype={'seed':str}).set_index(keys).sort_index()
        observed=native[native.label_known].set_index(keys).sort_index()
        pd.testing.assert_index_equal(prior.index,observed.index)
        for column in ['predicted','buy_reference','sell_reference','sell_offset']:
            np.testing.assert_allclose(prior[column],observed[column],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(prior.actual_extrema_scenario,observed.actual_extrema_scenario,rtol=1e-5,atol=2e-7)
        assert native.loc[~native.label_known,'actual_extrema_scenario'].isna().all()
        daily=pd.read_csv(dest/f'{part}-daily.csv',dtype={'seed':str}).set_index(['arm','seed','date'])
        for key,g in paired.groupby(['arm','seed','date']):
            ordered=g.sort_values(['predicted','instrument_id'],ascending=[False,True])
            np.testing.assert_array_equal(ordered['rank'],np.arange(1,len(g)+1))
            if len(g)<20:
                continue
            top=ordered.iloc[:20]
            visible=top[top.label_known]
            r=daily.loc[key]
            assert r.top_known==len(visible) and r.top_unknown==20-len(visible)
            np.testing.assert_allclose(r.top_extrema_scenario_pct,visible.actual_extrema_scenario.mean()*100,rtol=1e-12)
            np.testing.assert_allclose(r.return_mae_pp,g[g.label_known].absolute_error.mean()*100,rtol=1e-12)
            checked+=len(g)
    selection=pd.read_csv(dest/'selection-summary.csv',dtype={'seed':str})
    chosen=selection[(selection.seed=='ensemble')&(selection.arm!='baseline')].sort_values(['top_lift_pp','arm'],ascending=[False,True]).iloc[0].arm
    assert read(dest/'nomination.json')['arm']==chosen
    write_json(dest/'verification.json',dict(passed=True,at=utc_now(),checked_paired_rows=checked,
        eligibility_uses_forecasts_only=True,unknown_top20_not_replaced=True,
        known_prices_match_independent_raw_path_audit=True,selection_only_nomination=True))


def finalize(root):
    cfg=read(root/'protocol.json')
    verify_manifest(root,'completed.json')
    dest=root.parent/(root.name+'-frozen-top20')
    if (dest/'completed.json').exists():
        verify_manifest(dest,'completed.json')
        if read(dest/'binding.json')['source_manifest_sha256']!=file_hash(root/'completed.json'):
            raise ValueError('Frozen Top20 audit belongs to another experiment')
        publish_shadow_binding(root,dest,cfg)
        return dest
    dest.mkdir(exist_ok=True)
    write_json(dest/'binding.json',dict(source=str(root),source_manifest_sha256=file_hash(root/'completed.json'),
        code_sha256=file_hash(Path(__file__)),rule='Rank all forecast-eligible stocks before attaching observed labels; retain missing Top20 outcomes without substitution.'))
    data=Dataset()
    sd,sm=build_part(root,dest,cfg,data,'selection')
    chosen=sm[(sm.seed=='ensemble')&(sm.arm!='baseline')].sort_values(['top_lift_pp','arm'],ascending=[False,True]).iloc[0]
    write_json(dest/'nomination.json',dict(arm=chosen.arm,selection_lift_pp=float(chosen.top_lift_pp),
        rule=cfg['nomination_rule'],forecast_only_top20=True,selection_summary_sha256=file_hash(dest/'selection-summary.csv'),
        evaluation_used=False,at=utc_now(),live_promoted=False))
    ed,em=build_part(root,dest,cfg,data,'evaluation')
    comp=comparisons(ed,cfg)
    write_json(dest/'comparison.json',comp)
    lines=['# Return/ranking comparison: frozen Top20', '',
        'Primary evaluation freezes stock eligibility, ranking, Top20 membership and exit dates using forecasts only. Unknown future outcomes remain missing and never cause replacement by a lower-ranked stock.', '',
        'Three seeds, three arms, four fixed epochs; 192 sampled stocks per date before forecast-availability pairing. Five-day OHLCVA token output and sell-reference / buy-reference − 1 − 0.25% score are preserved.', '',
        '## Evaluation: equal-model ensemble', '',
        '| Arm | Return MAE, pp | Top20 observed extrema scenario | Lift over observed paired universe, pp | Top20 q10 | Known Top20 per date | High-volatility share |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in em[em.seed=='ensemble'].itertuples():
        lines.append(f'| {r.arm} | {r.return_mae_pp:.4f} | {r.top_extrema_scenario_pct:.2f}% | {r.top_lift_pp:+.4f} | {r.top_q10_pct:.2f}% | {r.top_known:.2f}/20 | {r.top_high_volatility_fraction:.1%} |')
    lines+=['', 'Unknown Top20 outcomes are excluded from observed means, not filled or counted as zero. This can bias those means if missingness is informative. The complete-Top20-date sensitivity table uses only dates with all 20 outcomes observed for every ensemble arm; native-universe results are also retained.', '',
        '## Paired ensemble improvement', '', '| Comparison | Metric | Mean improvement | Dates improved | Date-block 95% interval |', '| --- | --- | ---: | ---: | --- |']
    for r in comp:
        if r['seed']=='ensemble':
            lo,hi=r['bootstrap_95']
            lines.append(f'| {r["arm"]} vs {r["comparator"]} | {r["metric"]} | {r["improvement"]:+.4f} | {r["date_win_fraction"]:.1%} | [{lo:+.4f}, {hi:+.4f}] |')
    lines+=['', '## Candidate for manual comparison', '',
        f'Selection-only nomination: **{chosen.arm}**, using selection ensemble Top20 lift. No loss-weight tuning, evaluation-based model choice or main-model replacement.', '',
        '## Interpretation and evidence', '',
        'These are actual price-extrema scenarios on forecast-selected dates, not executed profits. The experiment covers a fixed exchange-stratified sample, not whole-market Top20. All per-seed, calendar-period, poor-date, native-universe and complete-Top20 sensitivity results are retained. Date win fractions are sample frequencies, not future success probabilities. Neither one weaker metric nor an interval crossing zero automatically negates another gain.', '',
        'The original experiment report conditions its diagnostic ranking on known labels. This corrected report is the primary stock-selection result; it preserves Top20 membership before label review. Model training, checkpoints and forecast paths are unchanged by this correction.', '',
        cfg['limitations'], '']
    (dest/'report.md').write_text('\n'.join(lines))
    verify_frozen_report(root,dest,cfg)
    files={str(p.relative_to(dest)):file_hash(p) for p in dest.rglob('*') if p.is_file() and p.name!='completed.json'}
    write_json(dest/'completed.json',dict(passed=True,at=utc_now(),files=files))
    publish_shadow_binding(root,dest,cfg)
    return dest


def publish_shadow_binding(root,dest,cfg):
    chosen=read(dest/'nomination.json')['arm']
    shadow_path=root.parent/(root.name+'-shadow.json')
    shadow=read(shadow_path)
    shadow['arm']=chosen
    shadow['checkpoints']={str(seed):dict(path=str(root/f'seed{seed}'/chosen/'model.pt'),sha256=file_hash(root/f'seed{seed}'/chosen/'model.pt')) for seed in cfg['seeds']}
    shadow['nomination_evidence']=dict(path=str(dest/'completed.json'),sha256=file_hash(dest/'completed.json'))
    write_json(shadow_path,shadow)


def run(config,prepare_only=False):
    if prepare_only:
        return
    cfg=read(config)
    root=BASE/cfg['output']
    receipt=root.parent/(root.name+'-verification.json')
    if receipt.exists():
        audit=read(receipt)
        if not audit['passed'] or audit['experiment_sha256']!=file_hash(root/'completed.json'):
            raise ValueError('Verification receipt differs')
    else:
        verify(root)
    dest=finalize(root)
    print('Primary frozen-Top20 report',dest/'report.md',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=BASE/'configs/token-ranking-v2.json')
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args()
    config=read(args.config)
    root=BASE/config['output']
    with root.with_suffix('.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Experiment or finalization is already running') from None
        run(args.config,args.prepare_only)
