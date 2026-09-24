"""Freeze and run a bounded multi-window model-quality experiment."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from price_pilot import load_arrays, verify_files

from quant_research.model_quality import (
    assess,
    daily_metrics,
    fit_volatility_baseline,
    label_overlaps,
    volatility_scale,
)
from quant_research.price_strategy import FIELDS, ordered_quantiles
from quant_research.storage import file_hash, utc_now, write_json


def read(path):
    return json.loads(path.read_text())


def freeze(args):
    root,out=args.root.resolve(),args.output.resolve()
    protocol=read(args.protocol)
    schedule=read(root/'schedule.json')
    folds=[schedule['development'][i] for i in protocol['fold_indices']]
    if any(f['test_end']>=schedule['sealed_holdout_start'] or f['purpose']!='development' for f in folds):
        raise ValueError('Invalid development windows')
    out.mkdir()
    shutil.copy2(args.protocol,out/'protocol.json')
    write_json(out/'experiment.json',{'created_at':utc_now(),'root':str(root),'folds':folds,
        'schedule_sha256':file_hash(root/'schedule.json'),
        'acceptance_sha256':file_hash(root/'development-acceptance.json'),
        'protocol_sha256':file_hash(out/'protocol.json'),
        'sealed_holdout_start':schedule['sealed_holdout_start'],'daily_strategy_ready':False})
    base=Path(__file__).resolve().parents[1]
    shutil.copytree(base/'src/quant_research',out/'code/src/quant_research',ignore=shutil.ignore_patterns('__pycache__'))
    (out/'code/scripts').mkdir()
    for name in ['price_pilot.py','model_quality_run.py']:
        shutil.copy2(base/'scripts'/name,out/'code/scripts'/name)
    shutil.copy2(base/'uv.lock',out/'code/uv.lock')
    write_json(out/'frozen-manifest.json',{str(f.relative_to(out)):file_hash(f) for f in out.rglob('*') if f.is_file()})
    print('Frozen protocol and source',out,flush=True)


def scaled(window, protocol):
    import lightgbm as lgb
    cfg=read(window/'config.json')
    train,selection=load_arrays(window/'train.npz'),load_arrays(window/'selection.npz')
    evaluation=load_arrays(window/'evaluation.npz')
    rows=pd.read_parquet(window/'train-rows.parquet')
    val_rows=pd.read_parquet(window/'selection-rows.parquet')
    names=read(Path(cfg['root'])/'panel-h5/manifest.json')['feature_names']
    col=names.index('volatility_20')
    scales={k:volatility_scale(d['x'],col,protocol['volatility_floor'])
            for k,d in [('train',train),('selection',selection),('evaluation',evaluation)]}
    b=fit_volatility_baseline(train['targets'],scales['train'],rows.date)
    np.save(window/'volatility-evaluation.npy',b[None]*scales['evaluation'][:,None,None,None])
    np.save(window/'volatility-baseline.npy',b)
    target=train['targets']/scales['train'][:,None,None]
    val=selection['targets']/scales['selection'][:,None,None]
    prediction=np.empty((len(evaluation['x']),5,4,3))
    directory=window/'lightgbm-scaled-models'
    directory.mkdir()
    weights=1/rows.groupby('date').date.transform('size').to_numpy()
    val_weights=1/val_rows.groupby('date').date.transform('size').to_numpy()
    log=[]
    for day in range(5):
        for field in range(4):
            y,v=target[:,day,field],val[:,day,field]
            good,vg=np.isfinite(y),np.isfinite(v)
            if good.sum()<100 or vg.sum()<20:
                raise ValueError('Insufficient observed labels')
            for j,q in enumerate([.1,.5,.9]):
                model=lgb.LGBMRegressor(objective='quantile',alpha=q,n_estimators=protocol['iterations'],
                    learning_rate=.05,num_leaves=15,min_child_samples=100,reg_lambda=1.,n_jobs=4,
                    random_state=protocol['seed'],verbosity=-1,deterministic=True,force_col_wise=True)
                model.fit(train['x'][good],y[good],sample_weight=weights[good],
                    eval_set=[(selection['x'][vg],v[vg])],eval_sample_weight=[val_weights[vg]],
                    callbacks=[lgb.early_stopping(10,verbose=False)])
                model.booster_.save_model(str(directory/f'd{day+1}-{FIELDS[field]}-q{q}.txt'))
                prediction[:,day,field,j]=model.predict(evaluation['x'])*scales['evaluation']
                log.append({'day':day+1,'field':FIELDS[field],'q':q,'best_iteration':model.best_iteration_})
            print('Scaled LightGBM',day+1,FIELDS[field],'complete',flush=True)
    np.save(window/'lightgbm_scaled-evaluation.npy',ordered_quantiles(prediction))
    write_json(window/'scaled-training.json',{'models':log,'volatility_index':col,
        'volatility_floor':protocol['volatility_floor'],'selection':'selection partition only in normalized units',
        'target':'raw log quote change / signal volatility20; forecasts restored to raw log quote units'})


def audit_window(window,protocol):
    cfg=read(window/'config.json')
    boundaries={}
    for part in ['train','selection','calibration','evaluation']:
        r=pd.read_parquet(window/f'{part}-rows.parquet')
        boundaries[part]={'start':r.date.min(),'end':r.date.max(),'label_end':r.label_end.max()}
        if r.duplicated(['date','instrument_id']).any():
            raise ValueError('Duplicate sample')
    separated=all(boundaries[a]['label_end']<boundaries[b]['start']
                  for a,b in [('train','selection'),('selection','calibration'),('calibration','evaluation')])
    sealed=boundaries['evaluation']['label_end']<cfg['sealed_holdout_start']
    r=pd.read_parquet(window/'evaluation-rows.parquet')
    cohort=pd.read_parquet(window/'evaluation-cohort.parquet')
    labels=load_arrays(window/'evaluation.npz')
    known=np.isfinite(labels['targets'][:,4,3])
    frame=r.assign(known=known,exchange=r.instrument_id.str.split('.').str[1])
    coverage=frame.groupby('exchange').known.agg(['count','sum','mean']).to_dict('index')
    complete=set(coverage)=={'xbse','xshe','xshg'}
    t=protocol['thresholds']
    known_fraction=float(known.sum()/len(cohort))
    per_exchange_cohort=cohort.instrument_id.str.split('.').str[1].value_counts()
    for exchange,result in coverage.items():
        result['cohort_rows']=int(per_exchange_cohort[exchange])
        result['known_fraction_of_cohort']=result['sum']/result['cohort_rows']
    labels_ordered=not np.any(labels['valid'][:,1:] & ~labels['valid'][:,:-1])
    finite_x=np.isfinite(labels['x']).all()
    passed=separated and sealed and complete and labels_ordered and finite_x and known_fraction>=t['min_known_day5_fraction'] and all(
        v['known_fraction_of_cohort']>=t['min_exchange_known_fraction'] for v in coverage.values())
    return {'passed':bool(passed),'boundaries':boundaries,'sealed_holdout_untouched':bool(sealed),
        'partitions_separated':bool(separated),'evaluation_cohort':len(cohort),'evaluation_inputs':len(r),
        'known_day5_fraction':known_fraction,'exchange_coverage':coverage,
        'label_barriers_monotonic':bool(labels_ordered),'finite_past_features':bool(finite_x),
        'evaluation_label_overlaps':label_overlaps(r),
        'scope':'structural development audit; not verified-PIT or execution readiness'}


def run(args):
    out=args.output.resolve()
    verify_files(out,read(out/'frozen-manifest.json'))
    experiment,protocol=read(out/'experiment.json'),read(out/'protocol.json')
    root=Path(experiment['root'])
    for filename,key in [('schedule.json','schedule_sha256'),('development-acceptance.json','acceptance_sha256')]:
        if file_hash(root/filename)!=experiment[key]:
            raise ValueError('Changed experiment input')
    status=out/'run-status.json'
    if status.exists():
        raise FileExistsError('Run already attempted; retain artifacts and start a new version')
    write_json(status,{'status':'running','started_at':utc_now(),'pid':os.getpid()})
    env=os.environ.copy()
    env['PYTHONPATH']=str(out/'code/src')
    all_daily=[]
    audits={}
    try:
        for index in protocol['fold_indices']:
            name=f'fold-{index:02d}'
            window=out/name
            print('Preparing',name,flush=True)
            cmd=[sys.executable,str(out/'code/scripts/price_pilot.py'),'prepare','--root',str(root),'--output',str(window),
                '--fold',str(index),'--train-dates',str(protocol['train_dates']),
                '--validation-dates',str(protocol['selection_dates']),'--calibration-dates',str(protocol['calibration_dates']),
                '--eval-dates',str(protocol['evaluation_dates']),'--iterations',str(protocol['iterations'])]
            with (out/f'{name}-prepare.log').open('w') as log:
                subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            audit=audit_window(window,protocol)
            write_json(window/'data-audit.json',audit)
            audits[name]=audit
            print(name,'data audit',audit['passed'],flush=True)
            if not audit['passed']:
                raise ValueError(f'Data gate failed for {name}; no training on this window')
            env_fold=env.copy()
            env_fold['PYTHONPATH']=str(window/'code')
            with (out/f'{name}-tree.log').open('w') as log:
                subprocess.run([sys.executable,str(window/'code/price_pilot.py'),'tree','--output',str(window)],
                    env=env_fold,stdout=log,stderr=subprocess.STDOUT,check=True)
            tree=read(window/'tree-status.json')
            if tree['status']!='completed':
                raise ValueError('Incomplete raw tree stage')
            verify_files(window,tree['files'])
            print(name,'raw model completed',flush=True)
            with (out/f'{name}-scaled.log').open('w') as log:
                subprocess.run([sys.executable,str(out/'code/scripts/model_quality_run.py'),'scaled',
                    '--output',str(out),'--window',name],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            files={str(f.relative_to(window)):file_hash(f) for pattern in
                ['lightgbm-scaled-models/*.txt','volatility-*.npy','lightgbm_scaled-evaluation.npy','scaled-training.json']
                for f in window.glob(pattern)}
            write_json(window/'scaled-status.json',{'status':'completed','files':files,'updated_at':utc_now()})
            r=pd.read_parquet(window/'evaluation-rows.parquet')
            y=load_arrays(window/'evaluation.npz')['targets']
            predictions={m:np.load(window/f'{m}-evaluation.npy') for m in protocol['models']}
            daily=daily_metrics(r,y,predictions,name)
            daily.to_csv(window/'daily-metrics.csv',index=False)
            all_daily.append(daily)
            print(name,'evaluated',len(r),'inputs',flush=True)
            write_json(out/'progress.json',{'status':'running','completed_windows':len(all_daily),
                'total_windows':len(protocol['fold_indices']),'updated_at':utc_now()})
        daily=pd.concat(all_daily,ignore_index=True)
        daily.to_csv(out/'daily-metrics.csv',index=False)
        table,gates=assess(daily,protocol,audits)
        table.to_csv(out/'window-metrics.csv',index=False)
        write_json(out/'assessment.json',{'created_at':utc_now(),'protocol_sha256':file_hash(out/'protocol.json'),
            'models':gates,'data_audits':audits,'daily_strategy_ready':False,
            'uncertainty':'Three historically exposed development quarters; no significance claim or final promotion.'})
        lines=['# 模型质量：多窗口开发验收','',
            '本次评价模型，不生成用户交易建议。门槛在运行前冻结，属于开发筛选标准，不是盈利保证。',
            '日期等权、窗口等权；所有模型在同窗口相同行上比较。方向准确率仅诊断，不作为单独提升依据。','',
            '| 窗口 | 模型 | T+4收盘损失 | 80%覆盖 | 区间宽度 | 中位数MAE |',
            '|---|---|---:|---:|---:|---:|']
        for r in table.itertuples():
            lines.append(f'| {r.window} | {r.model} | {r.pinball:.6f} | {r.coverage80:.1%} | {r.width80:.4f} | {r.median_mae:.5f} |')
        lines+=['','区间宽度和MAE均为对数价格单位。','']
        for model,result in gates.items():
            state='允许进入决策增益实验' if result['passed'] else '未通过价格质量门槛'
            lines += [f'## {model}：{state}','']
            for name,check in result['checks'].items():
                lines.append(f"- {'通过' if check['passed'] else '未通过'}：{name}；观测值 {json.dumps(check['observed'],ensure_ascii=False)}")
            lines.append('')
        lines += ['三个历史开发窗口不足以证明统计显著性；最终留出集封存。',
            '原始训练快照仍缺少完整时点与执行证据。本轮结构审计通过不代表可以发布盘前买卖建议。',
            '未通过的候选保留失败记录，下一版假设及阈值变更必须单独登记，不能回改本次协议。']
        (out/'report.md').write_text('\n'.join(lines)+'\n')
        write_json(status,{'status':'completed','finished_at':utc_now(),'files':{
            name:file_hash(out/name) for name in ['assessment.json','daily-metrics.csv','window-metrics.csv','report.md']}})
        write_json(out/'progress.json',{'status':'completed','completed_windows':len(all_daily),
            'total_windows':len(protocol['fold_indices']),'updated_at':utc_now()})
    except BaseException as exc:
        write_json(out/'data-audits-partial.json',audits)
        write_json(status,{'status':'failed','updated_at':utc_now(),'error':repr(exc)})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['freeze','run','scaled'])
    parser.add_argument('--root',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--protocol',type=Path)
    parser.add_argument('--window')
    args=parser.parse_args()
    if args.stage=='freeze':
        freeze(args)
    elif args.stage=='run':
        run(args)
    else:
        verify_files(args.output,read(args.output/'frozen-manifest.json'))
        window=args.output/args.window
        verify_files(window,read(window/'prepared-manifest.json'))
        scaled(window,read(args.output/'protocol.json'))


if __name__=='__main__':
    main()
