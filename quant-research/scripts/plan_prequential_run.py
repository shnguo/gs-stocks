"""Chronological rolling plan models with calibration from earlier saved forecasts."""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from plan_asof_data import AsOfData, arrays, read, verify
from plan_prequential_schedule import make_schedule
from plan_value_run import decision_metrics

from quant_research.plan_targets import plan_targets
from quant_research.plan_value import (
    HEADS,
    apply_calibration,
    choose_research_plan,
    compare_heads,
    fit_calibration,
    fit_heads,
    head_metrics,
    predict_heads,
)
from quant_research.storage import file_hash, utc_now, write_json


def files(root):
    return {str(p.relative_to(root)):file_hash(p) for p in root.rglob('*') if p.is_file()
        and '__pycache__' not in p.parts}


def freeze(a):
    out,prior,schedule_root = [p.resolve() for p in [a.output,a.prior,a.schedule]]
    state = read(prior/'run-status.json')
    if state['status'] != 'completed' or not read(prior/'verification.json')['passed']:
        raise ValueError('Unverified baseline')
    verify(prior,state['files'])
    verify(schedule_root,read(schedule_root/'frozen-manifest.json'))
    if not read(schedule_root/'verification.json')['passed']:
        raise ValueError('Unverified as-of schedule')
    schedule_protocol = read(schedule_root/'protocol.json')
    if schedule_protocol['parent'] != str(prior) or schedule_protocol['parent_status_sha256'] != file_hash(prior/'run-status.json'):
        raise ValueError('Schedule refers to a different baseline')
    schedule = read(schedule_root/'schedule.json')
    parent_protocol = read(prior/'protocol.json')
    base = Path(__file__).resolve().parents[1]
    for name in ['plan_value.py','plan_targets.py','price_strategy.py']:
        if file_hash(base/'src/quant_research'/name) != file_hash(prior/'code/src/quant_research'/name):
            raise ValueError('Core targets or tree implementation differs from baseline')
    source = Path(read(prior/'fold-01/config.json')['root'])
    accepted = read(source/'development-acceptance.json')
    if not accepted['passed'] or accepted['formal_ready'] is not False:
        raise ValueError('Unexpected source acceptance')
    verify(source,accepted['evidence'])
    panel,snapshot = [read(source/n/'manifest.json') for n in ['panel-h5','snapshot']]
    if file_hash(source/'panel-h5/manifest.json') != accepted['panels']['5']:
        raise ValueError('Panel is not the accepted version')
    verify(source/'panel-h5',panel['files'])
    verify(source/'snapshot',snapshot['files'])
    assert panel['dataset_id'] == snapshot['dataset_id'] == accepted['dataset_id']
    for fold,s in schedule.items():
        cfg = read(prior/fold/'config.json')
        expected = make_schedule(panel['dates'],cfg['dates']['evaluation'],cfg['fold']['train_start'],parent_protocol['sealed_holdout_start'])
        if s != expected:
            raise ValueError('As-of schedule differs from registered rules')
    out.mkdir()
    write_json(out/'protocol.json',dict(protocol_id='plan-prequential-v1',schedule=schedule_protocol,
        baseline_protocol=parent_protocol,
        training={**{k:parent_protocol[k] for k in ['iterations','seed','tree_parameters','sealed_holdout_start']},
            'train_dates':96,'selection_dates':4},
        calibration='Identical baseline sigmoid/offset family; twelve earlier as-of predictions, labels matured by current signal close',
        model_scope='LightGBM 27 features and all five plan heads; no industry changes in this timing contrast',
        comparison='Same baseline evaluation stock/date/plan labels; report raw and calibrated separately',
        causal_scope='Whole update pipeline intervention; shorter selection interval also changes; no isolated recency causal claim',
        quality_promotion=False,executable=False))
    write_json(out/'schedule.json',schedule)
    write_json(out/'experiment.json',dict(created_at=utc_now(),prior=str(prior),source=str(source),
        parent_status_sha256=file_hash(prior/'run-status.json'),dataset_id=panel['dataset_id'],
        source_manifests={n:file_hash(source/n) for n in ['development-acceptance.json','panel-h5/manifest.json','snapshot/manifest.json']},
        schedule_frozen_sha256=file_hash(schedule_root/'frozen-manifest.json')))
    shutil.copytree(base/'src/quant_research',out/'code/src/quant_research',ignore=shutil.ignore_patterns('__pycache__'))
    (out/'code/scripts').mkdir()
    for name in ['plan_prequential_run.py','plan_asof_data.py','plan_prequential_schedule.py','plan_value_run.py','price_pilot.py','model_quality_run.py']:
        shutil.copy2(base/'scripts'/name,out/'code/scripts'/name)
    shutil.copy2(base/'uv.lock',out/'code/uv.lock')
    write_json(out/'frozen-manifest.json',files(out))
    print('Frozen',out,'fits',sum(len(s['fits']) for s in schedule.values()),flush=True)


def context(out):
    verify(out,read(out/'frozen-manifest.json'))
    e,p,s = [read(out/n) for n in ['experiment.json','protocol.json','schedule.json']]
    store = AsOfData(e['source'],out/'day-cache',p['training']['sealed_holdout_start'])
    for name,h in e['source_manifests'].items():
        if file_hash(Path(e['source'])/name) != h:
            raise ValueError('Source manifest changed')
    return e,p,s,store


def fit(a):
    out = a.output.resolve()
    e,p,s,store = context(out)
    spec = s[a.window]['fits'][a.asof]
    dest = out/a.window/'fits'/a.asof
    dest.mkdir(parents=True)
    write_json(dest/'config.json',dict(window=a.window,asof=a.asof,schedule=spec,dataset_id=e['dataset_id'],
        frozen_experiment_sha256=file_hash(out/'frozen-manifest.json'),started_at=utc_now()))
    train_rows,train = store.stack(spec['train'],a.asof)
    select_rows,selection = store.stack(spec['selection'],a.asof)
    if spec['train_label_end'] >= spec['selection'][0] or spec['selection_label_end'] > a.asof:
        raise ValueError('Immature or overlapping training/selection labels')
    # Only targets already complete at this simulated time are supplied to training.
    train_y,select_y = [plan_targets(data) for data in [train,selection]]
    metadata = fit_heads(train['x'],train_rows.date,train_y,selection['x'],select_rows.date,select_y,dest/'models',p['training'])
    write_json(dest/'models.json',metadata)
    receipts = store.label_receipt([*spec['train'],*spec['selection']],a.asof)
    del train,selection,train_y,select_y
    rows,data = store.inputs(a.asof,a.asof)
    pred = predict_heads(data['x'],dest/'models',metadata)
    rows.to_parquet(dest/'prediction-rows.parquet',index=False)
    np.savez_compressed(dest/'raw.npz',**pred)
    receipts.update(store.input_receipt([a.asof]))
    write_json(dest/'data-receipts.json',dict(asof=a.asof,cache_files=receipts,
        input_only_prediction_date=a.asof,latest_training_label=spec['train_label_end'],latest_selection_label=spec['selection_label_end']))
    write_json(dest/'raw-completed.json',dict(status='completed',completed_at=utc_now(),files=files(dest)))
    print('Completed as-of fit',a.window,a.asof,flush=True)


def calibrate_at(out,window,day,spec,store):
    dest = out/window/'fits'/day
    marker = dest/'calibration-completed.json'
    if marker.exists():
        verify(dest,read(marker)['files'])
        return
    dependencies,parts,targets,dates = {},[],[],[]
    for past in spec['calibration_prediction_dates']:
        previous = out/window/'fits'/past
        completed = read(previous/'raw-completed.json')
        verify(previous,completed['files'])
        if read(previous/'config.json')['asof'] != past or past >= day:
            raise ValueError('Calibration forecast was not made at an earlier as-of')
        rows,data = store.labels(past,day)
        pd.testing.assert_frame_equal(rows,pd.read_parquet(previous/'prediction-rows.parquet'))
        parts.append(arrays(previous/'raw.npz'))
        target = plan_targets(data)
        targets.append({k:target[k] for k in ['filled','conditional_net_return','conditional_loss','conditional_downside']})
        dates.extend(rows.date.tolist())
        dependencies[past] = file_hash(previous/'raw-completed.json')
    predictions = {h:np.concatenate([p[h] for p in parts]) for h in HEADS}
    outcomes = {k:np.concatenate([p[k] for p in targets]) for k in targets[0]}
    calibration = fit_calibration(predictions,outcomes,np.array(dates))
    raw = arrays(dest/'raw.npz')
    corrected = apply_calibration(raw,calibration)
    write_json(dest/'calibration.json',calibration)
    np.savez_compressed(dest/'calibrated.npz',**corrected)
    rows = pd.read_parquet(dest/'prediction-rows.parquet')
    for name,pred in [('raw',raw),('calibrated',corrected)]:
        selected = rows[['date','instrument_id']].copy()
        selected['plan_index'] = choose_research_plan(pred)
        selected.to_parquet(dest/f'{name}-chosen.parquet',index=False)
    write_json(dest/'calibration-receipts.json',dict(asof=day,earlier_raw_completions=dependencies,
        cache_files=store.label_receipt(spec['calibration_prediction_dates'],day),
        maximum_label_end=max(spec['calibration_label_ends']),raw_completion_sha256=file_hash(dest/'raw-completed.json')))
    names=['calibration.json','calibrated.npz','raw-chosen.parquet','calibrated-chosen.parquet','calibration-receipts.json']
    write_json(marker,dict(status='completed',completed_at=utc_now(),files={n:file_hash(dest/n) for n in names}))


def evaluate(out,e,p,s,store):
    metrics,decisions = [],[]
    prior = Path(e['prior'])
    for window,info in s.items():
        parent = prior/window
        rows = pd.read_parquet(parent/'evaluation-rows.parquet')
        dest = out/window
        gathered = {name:{h:[] for h in HEADS} for name in ['rolling_raw','rolling']}
        labels_parts=[]
        for item in info['predictions']:
            day=item['date']
            directory=dest/'fits'/day
            verify(directory,read(directory/'raw-completed.json')['files'])
            verify(directory,read(directory/'calibration-completed.json')['files'])
            input_rows,data=store.labels(day,item['evaluation_label_end'])
            expected=rows.loc[rows.date.eq(day)].reset_index(drop=True)
            pd.testing.assert_frame_equal(input_rows,expected[input_rows.columns])
            labels_parts.append(data)
            for name,filename in [('rolling_raw','raw.npz'),('rolling','calibrated.npz')]:
                pred=arrays(directory/filename)
                chosen=pd.read_parquet(directory/('raw-chosen.parquet' if name=='rolling_raw' else 'calibrated-chosen.parquet'))
                np.testing.assert_array_equal(chosen.plan_index,choose_research_plan(pred))
                for h in HEADS:
                    gathered[name][h].append(pred[h])
        labels={k:np.concatenate([d[k] for d in labels_parts]) for k in labels_parts[0]}
        original=arrays(parent/'evaluation.npz')
        for k,v in original.items():
            np.testing.assert_array_equal(labels[k],v)
        forecasts={name:{h:np.concatenate(v) for h,v in values.items()} for name,values in gathered.items()}
        for name,filename in [('baseline27_raw','learned_raw-evaluation.npz'),('baseline27','learned-evaluation.npz')]:
            forecasts[name]=arrays(parent/filename)
        for name,pred in forecasts.items():
            if name.startswith('rolling'):
                np.savez_compressed(dest/f'{name}-evaluation.npz',**pred)
                daily,chosen=decision_metrics(rows,pred,labels,window,name)
                daily.to_csv(dest/f'{name}-decision-metrics.csv',index=False)
                chosen.to_parquet(dest/f'{name}-chosen.parquet',index=False)
                decisions.append(daily)
        report=head_metrics(rows,plan_targets(labels),forecasts,window)
        report.to_csv(dest/'head-metrics.csv',index=False)
        metrics.append(report)
    combined=pd.concat(metrics,ignore_index=True)
    combined.to_csv(out/'head-metrics.csv',index=False)
    pd.concat(decisions,ignore_index=True).to_csv(out/'decision-metrics.csv',index=False)
    contrasts={}
    for candidate,baseline in [('rolling','baseline27'),('rolling_raw','baseline27_raw'),('rolling','rolling_raw')]:
        pair=combined.loc[combined.model.isin([candidate,baseline])].replace({'model':{candidate:'learned',baseline:'empirical'}})
        contrasts[candidate+'_vs_'+baseline]=compare_heads(pair)
    write_json(out/'assessment.json',dict(comparisons=contrasts,quality_promotion=False,executable=False,
        scope='Predeclared simulated daily-close information; retrospective dataset; all six development windows, no final holdout or portfolio proof'))


def run(a):
    out=a.output.resolve()
    e,p,s,store=context(out)
    marker=out/'run-status.json'
    if marker.exists():
        raise FileExistsError('Preserve previous run attempts')
    prior=Path(e['prior'])
    if file_hash(prior/'run-status.json')!=e['parent_status_sha256']:
        raise ValueError('Baseline changed')
    verify(prior,read(prior/'run-status.json')['files'])
    for sub in ['panel-h5','snapshot']:
        verify(Path(e['source'])/sub,read(Path(e['source'])/sub/'manifest.json')['files'])
    write_json(marker,dict(status='running',pid=os.getpid(),started_at=utc_now()))
    completed=0
    total=sum(len(info['fits']) for info in s.values())
    try:
        for window,info in s.items():
            evaluations={x['date']:x for x in info['predictions']}
            for day in info['fits']:
                dest=out/window/'fits'/day
                write_json(out/'progress.json',dict(status='fitting',window=window,asof=day,completed_fits=completed,total_fits=total,updated_at=utc_now()))
                if (dest/'raw-completed.json').exists():
                    verify(dest,read(dest/'raw-completed.json')['files'])
                    config=read(dest/'config.json')
                    if config['schedule']!=info['fits'][day] or config['frozen_experiment_sha256']!=file_hash(out/'frozen-manifest.json'):
                        raise ValueError('Completed fit belongs to another protocol')
                else:
                    log=out/f'{window}-{day}.log'
                    with log.open('x') as stream:
                        subprocess.run([sys.executable,str(out/'code/scripts/plan_prequential_run.py'),'fit','--output',str(out),
                            '--window',window,'--asof',day],env={**os.environ,'PYTHONPATH':str(out/'code/src')},
                            stdout=stream,stderr=subprocess.STDOUT,check=True)
                if day in evaluations:
                    calibrate_at(out,window,day,evaluations[day],store)
                completed+=1
                print('Completed rolling fit',completed,'/',total,window,day,flush=True)
        write_json(out/'forecast-stage-completed.json',dict(completed_at=utc_now(),fits=completed,
            prediction_markers={str(f.relative_to(out)):file_hash(f) for f in out.glob('fold-*/fits/*/*completed.json')}))
        write_json(out/'progress.json',dict(status='evaluating_frozen_forecasts',completed_fits=completed,total_fits=total,updated_at=utc_now()))
        evaluate(out,e,p,s,store)
        for sub in ['panel-h5','snapshot']:
            verify(Path(e['source'])/sub,read(Path(e['source'])/sub/'manifest.json')['files'])
        write_json(out/'progress.json',dict(status='completed',completed_fits=completed,total_fits=total,updated_at=utc_now()))
        evidence=files(out)
        evidence.pop('run-status.json',None)
        write_json(marker,dict(status='completed',finished_at=utc_now(),files=evidence))
    except BaseException as exc:
        write_json(marker,dict(status='failed',error=repr(exc),completed_fits=completed,updated_at=utc_now()))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['freeze','fit','run'])
    for name in ['output','prior','schedule']:
        parser.add_argument('--'+name,type=Path,required=name=='output')
    parser.add_argument('--window')
    parser.add_argument('--asof')
    a=parser.parse_args()
    {'freeze':freeze,'fit':fit,'run':run}[a.stage](a)


if __name__=='__main__':
    main()
