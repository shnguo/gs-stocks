"""Audit all rolling fits, historical calibrators, fixed choices and final metrics."""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from verify_plan_asof_fit import audit, check, read

from quant_research.plan_targets import plan_targets
from quant_research.plan_value import HEADS, PROBABILITIES, head_targets
from quant_research.price_strategy import TradeAssumptions
from quant_research.storage import file_hash, utc_now, write_json


def load(path):
    with np.load(path) as z:
        return {k:z[k] for k in z.files}


def calibrator_check(predictions,targets,dates,states,raw,corrected):
    count=0
    for h in HEADS:
        for j,spec in enumerate(states[h]):
            y,p=targets[h][:,j],predictions[h][:,j]
            known=np.isfinite(y)&np.isfinite(p)
            unique,inverse,frequency=np.unique(dates[known],return_inverse=True,return_counts=True)
            w=1/frequency[inverse]
            assert spec['known_rows']==int(known.sum()) and spec['known_dates']==len(unique)
            expected_kind='offset'
            if h in PROBABILITIES:
                expected_kind='constant' if np.unique(y[known]).size<2 or np.std(p[known])<1e-12 else 'sigmoid'
            assert spec['kind']==expected_kind
            if expected_kind=='offset':
                value=np.average(y[known]-p[known],weights=w)
                np.testing.assert_allclose(spec['offset'],value,rtol=1e-10,atol=1e-12)
                result=raw[h][:,j]+value
            elif expected_kind=='constant':
                value=np.average(y[known],weights=w)
                np.testing.assert_allclose(spec['value'],value,rtol=1e-10,atol=1e-12)
                result=np.full(len(raw[h]),value)
            else:
                probability=np.clip(p[known],1e-6,1-1e-6)
                model=LogisticRegression(C=1.,solver='lbfgs',random_state=17)
                model.fit(np.log(probability/(1-probability))[:,None],y[known],sample_weight=w*len(w)/w.sum())
                np.testing.assert_allclose([spec['slope'],spec['intercept']],
                    [model.coef_[0,0],model.intercept_[0]],rtol=1e-6,atol=1e-8)
                value=np.clip(raw[h][:,j],1e-6,1-1e-6)
                logit=np.clip(spec['slope']*np.log(value/(1-value))+spec['intercept'],-40,40)
                result=1/(1+np.exp(-logit))
            if h==HEADS[4]:
                result=np.maximum(result,0)
            # Neural outputs can be float32; retain strict float64 checks while
            # allowing storage/operation rounding at the actual output precision.
            epsilon = np.finfo(corrected[h].dtype).eps
            np.testing.assert_allclose(result, corrected[h][:, j],
                rtol=max(1e-10, 8*epsilon), atol=max(1e-12, epsilon))
            count+=1
    return count


def independent_choice(pred):
    fill,resolution,mean=[pred[h] for h in HEADS[:3]]
    scores=pd.DataFrame(fill*mean).where((fill>=.3)&(resolution>=.9)&(mean>0),-np.inf)
    chosen=scores.idxmax(axis=1).to_numpy()
    chosen[scores.max(axis=1).eq(-np.inf)]=-1
    return chosen


def mean(values):
    return pd.Series(values,dtype=float).mean()


def verify_decision_assembly(parts, combined):
    keys = ['window', 'model', 'date']
    expected = pd.concat(parts, ignore_index=True)
    for frame in [expected, combined]:
        assert not frame[keys].isna().any().any()
        assert not frame.duplicated(keys).any()
    pd.testing.assert_frame_equal(expected.sort_values(keys).reset_index(drop=True),
                                  combined.sort_values(keys).reset_index(drop=True))


def verify_comparisons(metrics,assessment):
    for key,reported in assessment['comparisons'].items():
        candidate,baseline=key.split('_vs_')
        rng=np.random.default_rng(17)
        for h in HEADS:
            selected=metrics.loc[metrics['head'].eq(h)]
            columns=selected.pivot(index=['window','date'],columns='model',values='mse')
            differences=[(part[candidate]-part[baseline]).to_numpy() for _,part in columns.groupby(level='window')]
            by_window=[float(d.mean()) for d in differences]
            item=reported[h]
            np.testing.assert_allclose(item['by_window'],by_window,atol=1e-12)
            np.testing.assert_allclose(item['learned_minus_empirical'],np.mean(by_window),atol=1e-12)
            assert item['winning_windows']==sum(v<0 for v in by_window)
            draws=[]
            for _ in range(2000):
                values=[]
                for d in differences:
                    starts=rng.integers(len(d),size=(len(d)+1)//2)
                    indices=np.column_stack([starts,(starts+1)%len(d)]).ravel()[:len(d)]
                    values.append(d[indices].mean())
                draws.append(np.mean(values))
            np.testing.assert_allclose(item['conditional_two_date_block_95'],np.quantile(draws,[.025,.975]),atol=1e-12)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    out=a.output.resolve()
    state,e,s,protocol=[read(out/n) for n in ['run-status.json','experiment.json','schedule.json','protocol.json']]
    if state['status']!='completed' or (out/'verification.json').exists():
        raise ValueError('Requires complete, not previously verified rolling experiment')
    count=dict(hashes=check(out,state['files']),fits=0,restored_values=0,cache_inputs=0,cache_labels=0,
        best_at_cap=0,calibrators=0,known_source_values=0,metric_rows=0,chosen_rows=0,decision_dates=0)
    prior=Path(e['prior'])
    assert file_hash(prior/'run-status.json')==e['parent_status_sha256']
    count['hashes']+=check(prior,read(prior/'run-status.json')['files'])
    for sub in ['panel-h5','snapshot']:
        count['hashes']+=check(Path(e['source'])/sub,read(Path(e['source'])/sub/'manifest.json')['files'])
    cache=out/'day-cache'
    final=read(out/'forecast-stage-completed.json')
    expected_markers={f'{window}/fits/{day}/raw-completed.json' for window,info in s.items() for day in info['fits']}
    expected_markers.update(f'{window}/fits/{p["date"]}/calibration-completed.json' for window,info in s.items() for p in info['predictions'])
    assert set(final['prediction_markers'])==expected_markers and final['fits']==120
    count['hashes']+=check(out,final['prediction_markers'])
    reports,decision_reports=[],[]
    for window,info in s.items():
        for asof in info['fits']:
            checked=audit(out,window,asof)
            for k,value in checked.items():
                count[k]+=value
            count['fits']+=1
        for item in info['predictions']:
            day=item['date']
            directory=out/window/'fits'/day
            receipt=read(directory/'calibration-receipts.json')
            assert receipt['asof']==day and receipt['maximum_label_end']<=day
            assert set(receipt['earlier_raw_completions'])==set(item['calibration_prediction_dates'])
            assert receipt['raw_completion_sha256']==file_hash(directory/'raw-completed.json')
            completed=read(directory/'calibration-completed.json')
            assert read(directory/'raw-completed.json')['completed_at']<=completed['completed_at']<=final['completed_at']
            count['hashes']+=check(directory,completed['files'])
            required_cache={f'{d}/{kind}-manifest.json' for d in item['calibration_prediction_dates'] for kind in ['input','label']}
            assert set(receipt['cache_files'])==required_cache
            count['hashes']+=check(cache,receipt['cache_files'])
            predictions,targets,dates=[],[],[]
            for past in item['calibration_prediction_dates']:
                parent=out/window/'fits'/past
                assert receipt['earlier_raw_completions'][past]==file_hash(parent/'raw-completed.json')
                earlier=read(parent/'raw-completed.json')
                meta=read(cache/past/'label-manifest.json')
                assert past<day and meta['label_end']<=day
                assert earlier['completed_at']<=meta['created_at']<=completed['completed_at']
                rows=pd.read_parquet(parent/'prediction-rows.parquet')
                pd.testing.assert_frame_equal(rows,pd.read_parquet(cache/past/'rows.parquet'))
                predictions.append(load(parent/'raw.npz'))
                targets.append(head_targets(plan_targets(load(cache/past/'labels.npz'))))
                dates.extend(rows.date.tolist())
            raw,corrected=load(directory/'raw.npz'),load(directory/'calibrated.npz')
            count['calibrators']+=calibrator_check({h:np.concatenate([p[h] for p in predictions]) for h in HEADS},
                {h:np.concatenate([t[h] for t in targets]) for h in HEADS},np.asarray(dates),
                read(directory/'calibration.json'),raw,corrected)
            for name,pred in [('raw',raw),('calibrated',corrected)]:
                chosen=pd.read_parquet(directory/f'{name}-chosen.parquet')
                pd.testing.assert_frame_equal(chosen[['date','instrument_id']],pd.read_parquet(directory/'prediction-rows.parquet')[['date','instrument_id']])
                np.testing.assert_array_equal(chosen.plan_index,independent_choice(pred))
        rows=pd.read_parquet(prior/window/'evaluation-rows.parquet')
        original=load(prior/window/'evaluation.npz')
        pieces=[{**load(cache/item['date']/'inputs.npz'),**load(cache/item['date']/'labels.npz')} for item in info['predictions']]
        for key,value in original.items():
            combined=np.concatenate([part[key] for part in pieces])
            np.testing.assert_array_equal(combined,value)
            count['known_source_values']+=combined.size
        base=plan_targets(original)
        assumptions=TradeAssumptions()
        stress=plan_targets(original,replace(assumptions,commission=assumptions.commission*2,
            minimum_fee=assumptions.minimum_fee*2,sell_tax=assumptions.sell_tax*2,slippage_bps=assumptions.slippage_bps*2))
        targets=head_targets(base)
        models={name:load(out/window/f'{name}-evaluation.npz') for name in ['rolling','rolling_raw']}
        models.update({name:load(prior/window/filename) for name,filename in [('baseline27','learned-evaluation.npz'),('baseline27_raw','learned_raw-evaluation.npz')]})
        for name,filename in [('rolling','calibrated.npz'),('rolling_raw','raw.npz')]:
            for h in HEADS:
                expected=np.concatenate([load(out/window/'fits'/item['date']/filename)[h] for item in info['predictions']])
                np.testing.assert_array_equal(expected,models[name][h])
        metrics=pd.read_csv(out/window/'head-metrics.csv')
        assert len(metrics)==8*len(models)*len(HEADS)
        for date,ids in rows.groupby('date').indices.items():
            for name,pred in models.items():
                for h in HEADS:
                    actual,predicted=targets[h][ids].ravel(),pred[h][ids].ravel()
                    good=np.isfinite(actual)
                    y,pr=actual[good],predicted[good]
                    item=metrics.loc[metrics.model.eq(name)&metrics.date.eq(date)&metrics['head'].eq(h)].iloc[0]
                    assert item.known_rows==good.sum() and item.total_plan_rows==len(actual)
                    for key,value in [('mse',np.mean((y-pr)**2)),('mae',np.mean(np.abs(y-pr))),('mean_actual',y.mean()),('mean_prediction',pr.mean())]:
                        np.testing.assert_allclose(item[key],value,rtol=1e-10,atol=1e-12)
                    if h in PROBABILITIES:
                        prob=np.clip(pr,1e-6,1-1e-6)
                        np.testing.assert_allclose(item.brier,item.mse,atol=1e-12)
                        np.testing.assert_allclose(item.log_loss,np.mean(-y*np.log(prob)-(1-y)*np.log1p(-prob)),atol=1e-12)
                    count['metric_rows']+=1
        reports.append(metrics)
        for name in ['rolling','rolling_raw']:
            chosen=independent_choice(models[name])
            recorded=pd.read_parquet(out/window/f'{name}-chosen.parquet')
            np.testing.assert_array_equal(recorded.plan_index,chosen)
            for h in HEADS:
                value=np.where(chosen>=0,models[name][h][np.arange(len(rows)),np.maximum(chosen,0)],np.nan)
                np.testing.assert_allclose(recorded[h],value,atol=1e-12,equal_nan=True)
            assert not recorded.executable.any()
            daily=pd.read_csv(out/window/f'{name}-decision-metrics.csv')
            for date in rows.date.unique():
                ids=np.flatnonzero(rows.date.eq(date).to_numpy()&(chosen>=0))
                ix=(ids,chosen[ids])
                filled,net,stressed=base['filled'][ix],base['scenario_order_net_return'][ix],stress['scenario_order_net_return'][ix]
                expected=dict(candidate_stock_rows=int(rows.date.eq(date).sum()),selected=len(ids),fill_unknown=np.isnan(filled).sum(),
                    filled=(filled==1).sum(),return_unknown=np.isnan(net).sum(),ambiguous=base['ambiguous'][ix].sum(),
                    known_selected_scenario_mean=mean(net),stress_known_selected_scenario_mean=mean(stressed),stress_return_unknown=np.isnan(stressed).sum())
                item=daily.loc[daily.date.eq(date)].iloc[0]
                for key,value in expected.items():
                    np.testing.assert_allclose(item[key],value,atol=1e-12,equal_nan=True)
                count['decision_dates']+=1
            decision_reports.append(daily)
            count['chosen_rows']+=len(rows)
        print('Verified rolling window',window,flush=True)
    combined=pd.concat(reports,ignore_index=True)
    pd.testing.assert_frame_equal(combined,pd.read_csv(out/'head-metrics.csv'))
    verify_decision_assembly(decision_reports,pd.read_csv(out/'decision-metrics.csv'))
    verify_comparisons(combined,read(out/'assessment.json'))
    assert count['fits']==120 and count['calibrators']==48*60
    write_json(out/'verification.json',dict(passed=True,checked_at=utc_now(),counts=count,
        scope='All 120 fit receipts and saved heads on 24 rows each, all 48 historical calibration supports and fitted states, all evaluation labels and forecast assembly, daily metrics, choices, costs and block contrasts; retrospective daily-close availability assumption, not execution or profitability proof'))
    print(count,flush=True)


if __name__=='__main__':
    main()
