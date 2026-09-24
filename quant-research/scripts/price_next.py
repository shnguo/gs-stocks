"""Frozen follow-up: interval calibration and a small quantile Transformer."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from price_pilot import load_arrays, verify_files

from quant_research.price_calibration import apply_intervals, fit_intervals
from quant_research.price_stability import (
    aggregate_dates,
    date_metrics,
    stress_selected_orders,
    trade_metrics,
    verify_inputs,
)
from quant_research.price_strategy import (
    PlanCalibrator,
    TradeAssumptions,
    candidate_plans,
    choose_plan,
    trade_diagnostic,
)
from quant_research.storage import file_hash, utc_now, write_json


def read(path):
    return json.loads(path.read_text())


def prepare(args):
    p, out = args.prior.resolve(), args.output.resolve()
    verify_inputs(p)
    cfg = read(p / 'config.json')
    panel = Path(cfg['root']) / 'panel-h5'
    manifest = read(panel / 'manifest.json')
    assert file_hash(panel / 'manifest.json') == cfg['source_hashes']['panel_manifest']
    verify_files(panel, manifest['files'])
    assert manifest['dataset_id'] == cfg['dataset_id']
    assert len(cfg['dates']['calibration']) == 12
    split = cfg['dates']['calibration'][6]
    rows = pd.read_parquet(p / 'calibration-rows.parquet')
    assert rows.loc[rows.date < split, 'label_end'].max() < split
    assert pd.read_parquet(p / 'evaluation-rows.parquet').label_end.max() < cfg['sealed_holdout_start']
    out.mkdir()
    config = {'created_at': utc_now(), 'prior': str(p), 'dataset_id': cfg['dataset_id'],
              'parent_config_sha256': file_hash(p/'config.json'),
              'panel_manifest_sha256': file_hash(panel/'manifest.json'),
              'seed': 17, 'device': args.device, 'max_epochs': 4, 'batch_size': 512,
              'lookback': 60, 'features': 27, 'width': 48, 'layers': 2, 'heads': 4,
              'target': '5xOHLCx(q10,q50,q90) log raw quote relative to signal close',
              'selection': 'earliest best date-equal selection pinball; patience2; no evaluation tuning',
              'interval_calibration_dates': cfg['dates']['calibration'][:6],
              'plan_calibration_dates': cfg['dates']['calibration'][6:],
              'evaluation_dates': cfg['dates']['evaluation'],
              'interval_method': 'shared-cohort date-equal empirical score widening; no coverage guarantee',
              'evaluation_status': 'already exposed development dates; sealed holdout unopened',
              'executable': False, 'formal_ready': False}
    write_json(out/'config.json', config)
    code = out/'code'
    shutil.copytree(Path(__file__).resolve().parents[1]/'src/quant_research', code/'quant_research',
                    ignore=shutil.ignore_patterns('__pycache__'))
    for name in ['price_next.py', 'price_pilot.py']:
        shutil.copy2(Path(__file__).with_name(name), code/name)
    shutil.copy2(Path(__file__).resolve().parents[1]/'uv.lock', out/'uv.lock')
    if args.reuse_trained:
        trained = args.reuse_trained.resolve()
        stage = read(trained/'train-status.json')
        assert stage['status'] == 'completed'
        verify_files(trained,read(trained/'prepared-manifest.json'))
        verify_files(trained,stage['files'])
        old_config = read(trained/'config.json')
        assert {k:v for k,v in old_config.items() if k!='created_at'} == {
            k:v for k,v in config.items() if k!='created_at'}
        for name in ['price_transformer.py','price_calibration.py']:
            assert file_hash(trained/'code/quant_research'/name) == file_hash(code/'quant_research'/name)
        for name in stage['files']:
            shutil.copy2(trained/name,out/name)
        shutil.copy2(trained/'train-status.json',out/'train-status.json')
        write_json(out/'training-reuse.json',{'from':str(trained),
            'training_source_manifest_sha256':file_hash(trained/'prepared-manifest.json'),
            'training_status_sha256':file_hash(trained/'train-status.json'),
            'reason':'report call signature repair; same frozen model and predictions'})
    write_json(out/'prepared-manifest.json', {str(f.relative_to(out)): file_hash(f)
        for f in out.rglob('*') if f.is_file()})


def train_stage(out, cfg):
    from quant_research.price_transformer import predict, train
    p = Path(cfg['prior'])
    parent = read(p/'config.json')
    panel = Path(parent['root'])/'panel-h5'
    assert file_hash(panel/'manifest.json') == cfg['panel_manifest_sha256']
    verify_files(panel, read(panel/'manifest.json')['files'])
    values = np.load(panel/'values.npy', mmap_mode='r')
    rows = {s: pd.read_parquet(p/f'{s}-rows.parquet') for s in ['train','selection','calibration','evaluation']}
    model, mean, scale, log = train(values, rows['train'], load_arrays(p/'train.npz'),
        rows['selection'], load_arrays(p/'selection.npz'), out, cfg)
    for part in ['calibration','evaluation']:
        np.save(out/f'transformer-{part}.npy', predict(model, values, rows[part], mean, scale))
    write_json(out/'progress.json', {'status':'completed', 'epochs':len(log), 'updated_at':utc_now()})


def report(out, cfg):
    p = Path(cfg['prior'])
    parent = read(p/'config.json')
    a = TradeAssumptions(**parent['assumptions'])
    rows = {s:pd.read_parquet(p/f'{s}-rows.parquet') for s in ['calibration','evaluation']}
    data = {s:load_arrays(p/f'{s}.npz') for s in rows}
    models = ['naive','lightgbm','kronos','transformer']
    pred = {s:{m:np.load((out if m=='transformer' else p)/f'{m}-{s}.npy') for m in models} for s in rows}
    common = {s:np.logical_and.reduce([np.isfinite(q).all((1,2,3)) for q in pred[s].values()]) for s in rows}
    interval_ids = np.flatnonzero(common['calibration'] & rows['calibration'].date.isin(cfg['interval_calibration_dates']))
    plan_ids = np.flatnonzero(common['calibration'] & rows['calibration'].date.isin(cfg['plan_calibration_dates']))
    eval_ids = np.flatnonzero(common['evaluation'])
    assert not set(interval_ids) & set(plan_ids)
    widening, calibrated = {}, {s:{} for s in rows}
    for m in models:
        fit = fit_intervals(pred['calibration'][m][interval_ids], data['calibration']['targets'][interval_ids],
                            rows['calibration'].date.iloc[interval_ids])
        if not np.isfinite(fit['offsets']).all():
            raise ValueError('Insufficient interval calibration evidence')
        widening[m] = {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in fit.items()}
        for s in rows:
            calibrated[s][m] = apply_intervals(pred[s][m], fit['offsets'])
            np.save(out/f'{m}-{s}-calibrated.npy', calibrated[s][m])
    write_json(out/'interval-calibration.json', widening)
    all_daily, price_summary = [], []
    for variant, predictions in [('raw',pred),('calibrated',calibrated)]:
        daily = date_metrics(rows['evaluation'],data['evaluation']['targets'],predictions['evaluation'])
        daily['variant'] = variant
        all_daily.append(daily)
        price_summary.extend([{**r,'variant':variant} for r in aggregate_dates(daily) if r['subset']=='all_dates'])
    pd.concat(all_daily).to_csv(out/'daily-price-metrics.csv',index=False)

    def outcomes(part, ids):
        d = data[part]
        return [[trade_diagnostic(d['future'][i],d['valid'][i],plan,a,d['upper'][i],d['lower'][i])
                 for plan in candidate_plans(d['reference'][i],a)] for i in ids]
    cal_outcomes, eval_outcomes = outcomes('calibration',plan_ids), outcomes('evaluation',eval_ids)
    axis = read(Path(parent['root'])/'panel-h5/manifest.json')['dates']
    plan_records, forecast_frames = [], []
    from quant_research.price_strategy import FIELDS
    for variant, predictions in [('raw',pred),('calibrated',calibrated)]:
        for m in models:
            estimates = PlanCalibrator().fit(predictions['calibration'][m][plan_ids], cal_outcomes).estimate(
                predictions['evaluation'][m][eval_ids])
            for j,i in enumerate(eval_ids):
                row = rows['evaluation'].iloc[i]
                choice = choose_plan(estimates[j])
                base = dict(model=f'{m}_{variant}',date=row.date,instrument_id=row.instrument_id,
                    common_comparison=True,executable=False,signal='observe',status='no_order',
                    net_return=0.,filled=False,buy=np.nan,take_profit=np.nan,stop=np.nan,
                    buy_valid_until=axis[row.date_index+1],time_exit_date=row.label_end,
                    decision_reason='no_candidate_with_sufficient_calibrated_utility')
                if choice is not None:
                    plan = candidate_plans(data['evaluation']['reference'][i],a)[choice]
                    base.update(signal='research_candidate',buy=plan.buy,take_profit=plan.take_profit,
                        stop=plan.stop,candidate_index=choice,decision_reason='hypothetical_candidate_only')
                    base.update(estimates[j][choice])
                    base.update(eval_outcomes[j][choice])
                plan_records.append(base)
            quote = np.exp(predictions['evaluation'][m][eval_ids]) * data['evaluation']['reference'][eval_ids,None,None,None]
            r = rows['evaluation'].iloc[eval_ids]
            forecast_frames.append(pd.DataFrame(dict(model=f'{m}_{variant}',date=np.repeat(r.date.to_numpy(),20),
                instrument_id=np.repeat(r.instrument_id.to_numpy(),20),day=np.tile(np.repeat(np.arange(1,6),4),len(r)),
                field=np.tile(FIELDS,len(r)*5),q10=quote[...,0].ravel(),q50=quote[...,1].ravel(),q90=quote[...,2].ravel(),executable=False)))
    plans = pd.DataFrame(plan_records)
    plans.to_csv(out/'trade-plans.csv',index=False)
    stressed = stress_selected_orders(plans,rows['evaluation'],data['evaluation'],a)
    stressed.to_parquet(out/'fixed-order-cost-stress.parquet',index=False)
    pd.concat(forecast_frames).to_parquet(out/'common-price-forecasts.parquet',index=False)
    trades = [r for r in trade_metrics(stressed) if r['scope']=='common']
    result = {'created_at':utc_now(),'price':price_summary,'trades':trades,
        'interval_calibration_rows':len(interval_ids),'plan_calibration_rows':len(plan_ids),
        'common_evaluation_rows':len(eval_ids),'executable':False,'formal_ready':False,
        'limitations':['already exposed development dates','empirical calibration without coverage guarantee',
                      'Kronos pretraining overlap unknown','unknown trade outcomes excluded; no portfolio NAV']}
    write_json(out/'summary.json',result)
    lines = ['# 价格区间校准与 Transformer 对照','',
        '固定的 12 个已使用开发日期，按日期等权。最终留出集封存，全部计划不可执行。',
        f'前 6 校准日期共 {len(interval_ids)} 行用于区间修正；后 6 日期共 {len(plan_ids)} 行用于买卖计划。',
        '四模型在相同校准行及共同评估样本上比较。区间修正只扩大上下界，中位数不变。','',
        '| 范围 | 模型 | 版本 | 第5日分位损失 | 80%覆盖 |','|---|---|---|---:|---:|']
    for r in price_summary:
        if r['scope']=='full' and r['model']=='kronos':
            continue
        lines.append(f"| {r['scope']} | {r['model']} | {r['variant']} | {r['mean_daily_pinball']:.6f} | {r['mean_daily_coverage80']:.1%} |")
    lines += ['', '交易表只包含共同预测样本，原始和校准版本分别在后半校准期决定订单；每个版本内部成本压力保持订单不变。','',
              '| 模型版本 | 成本倍数 | 候选 | 未知 | 关闭 | 已知结果每单净收益 |','|---|---:|---:|---:|---:|---:|']
    for r in trades:
        net = '无订单' if r['mean_net_per_resolved_order'] is None else f"{r['mean_net_per_resolved_order']:.3%}"
        lines.append(f"| {r['model']} | {r['cost_multiplier']} | {r['candidate_orders']} | {r['unresolved_orders']} | {r['closed']} | {net} |")
    lines += ['', '训练专用数据缺少真实成交契约，逐单收益存在未知结果删失，不能视为组合收益。',
        '校准日期有限且时间分布会变化，经验扩宽不保证未来 80% 覆盖。Kronos 预训练历史重叠未排除。',
        'Transformer 是单种子、最多四轮的小规模比较；输入为60日×27特征，LightGBM仅当日27项，Kronos为60日OHLCVA。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','train','report'])
    parser.add_argument('--prior',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='mps')
    parser.add_argument('--reuse-trained',type=Path)
    args=parser.parse_args()
    if args.stage=='prepare':
        return prepare(args)
    out=args.output
    verify_files(out,read(out/'prepared-manifest.json'))
    cfg=read(out/'config.json')
    p=Path(cfg['prior'])
    assert file_hash(p/'config.json')==cfg['parent_config_sha256']
    verify_inputs(p)
    status=out/f'{args.stage}-status.json'
    if status.exists():
        raise FileExistsError('Stage already attempted; use a new experiment')
    if args.stage=='report':
        stage=read(out/'train-status.json')
        assert stage['status']=='completed'
        verify_files(out,stage['files'])
    before={f.name for f in out.iterdir()}
    write_json(status,{'status':'running','started_at':utc_now()})
    try:
        (train_stage if args.stage=='train' else report)(out,cfg)
        files={f.name:file_hash(f) for f in out.iterdir() if f.is_file() and f.name not in before
               and f!=status and not f.name.endswith('.log')}
        write_json(status,{'status':'completed','finished_at':utc_now(),'files':files})
    except BaseException as e:
        write_json(status,{'status':'failed','error':str(e),'updated_at':utc_now()})
        raise


if __name__=='__main__':
    main()
