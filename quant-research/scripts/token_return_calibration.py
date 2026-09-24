"""Manual walk-forward training and validation of the learned return head."""
import argparse
import fcntl
import shutil
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from finalize_token_ranking import freeze_and_score
from finalize_token_three_ideas import compare, summarize_daily

from quant_research.daily_loop import read, verify, write_manifest
from quant_research.return_calibration import fit_head, mature_rows, rank_head
from quant_research.storage import file_hash, utc_now, write_json

BASE = Path(__file__).resolve().parents[1]


def source_frames(source, mode, arm):
    labels = pd.read_parquet(source/'evaluation-inputs/rows.parquet')
    parts = []
    manifest = read(source/'validation/completed.json')['files']
    for path in sorted((source/'validation'/mode).glob('*.parquet')):
        if file_hash(path) != manifest[str(path.relative_to(source/'validation'))]:
            raise ValueError('Changed source rankings')
        part = pd.read_parquet(path, filters=[('arm', '=', arm)])
        part = part.merge(labels[['row_id', 'instrument_id', 'date', 'label_end']],
                          left_on=['local_row', 'instrument_id', 'date'],
                          right_on=['row_id', 'instrument_id', 'date'],
                          how='left', validate='many_to_one')
        if part.label_end.isna().any():
            raise ValueError('Missing label maturity metadata')
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def evaluate_cross_section(frame, head, top_n):
    raw = frame.copy()
    raw['arm'] = 'raw'
    # Detach labels before correction and ranking; reattach by row identity.
    forecasts = frame.drop(columns=['label_known', 'actual_extrema_scenario',
                                     'absolute_error', 'adverse_excursion_pct'])
    adjusted = rank_head(forecasts, head)
    adjusted = adjusted.merge(frame[['local_row', 'label_known', 'actual_extrema_scenario',
                                     'adverse_excursion_pct']], on='local_row', validate='one_to_one')
    adjusted['absolute_error'] = abs(adjusted.predicted - adjusted.actual_extrema_scenario)
    adjusted['arm'] = 'calibrated'
    ranks, daily = freeze_and_score(pd.concat([raw, adjusted], ignore_index=True), top_n)
    # Also measure error on the original Top20 to separate calibration from selection.
    raw_top = raw.sort_values(['predicted', 'instrument_id'], ascending=[False, True]).head(top_n)
    for i, row in daily.iterrows():
        fixed = ranks[(ranks.arm == row.arm) & ranks.local_row.isin(raw_top.local_row) & ranks.label_known]
        daily.loc[i, 'raw_top20_bias_pp'] = (fixed.predicted - fixed.actual_extrema_scenario).mean() * 100
        daily.loc[i, 'raw_top20_mae_pp'] = abs(fixed.predicted - fixed.actual_extrema_scenario).mean() * 100
    return ranks, daily


def run(config):
    cfg = read(config)
    source, root = BASE/cfg['source'], BASE/cfg['output']
    root.mkdir(parents=True, exist_ok=True)
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        files = [source/'completed.json', source/'validation/completed.json',
                 source/'evaluation-inputs/rows.parquet', source/'completion-audit/verification.json',
                 source/'prepared.json', source/'protocol.json']
        code = [Path(__file__), BASE/'src/quant_research/return_calibration.py',
                BASE/'scripts/verify_token_return_calibration.py',
                BASE/'scripts/finalize_token_ranking.py', BASE/'scripts/finalize_token_three_ideas.py']
        binding = dict(config=cfg, sources={str(p): file_hash(p) for p in files},
                       code={str(p): file_hash(p) for p in code})
        if (root/'binding.json').exists():
            if read(root/'binding.json') != binding:
                raise ValueError('Changed protocol/source/code; use a new output')
        else:
            write_json(root/'binding.json', binding)
            for path in code:
                snapshot = root/'code'/path.relative_to(BASE)
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, snapshot)
            live = [BASE/'configs/daily-token-v1.json', BASE/'configs/token-ranking-daily-v1.json',
                    BASE/'artifacts/daily-token-live-v1/current.json',
                    BASE/'artifacts/daily-token-live-v1/latest.json',
                    BASE/'artifacts/token-ranking-daily-v1/current.json']
            write_json(root/'live-state.json', {str(p): file_hash(p) for p in live if p.exists()})
        if (root/'completed.json').exists():
            complete = read(root/'completed.json')
            for name, path in [('validation_sha256', root/'validation/manifest.json'),
                               ('model_sha256', root/'models/manifest.json'),
                               ('verification_sha256', root/'verification.json')]:
                if file_hash(path) != complete[name]:
                    raise ValueError('Changed completed calibration artifact')
            verify(root/'validation')
            verify(root/'models')
            return root/'validation/report.md'
        if not read(source/'completion-audit/verification.json')['passed']:
            raise ValueError('Source audit must pass')
        completed = read(source/'completed.json')
        if completed['validation_manifest_sha256'] != file_hash(source/'validation/completed.json'):
            raise ValueError('Changed source validation manifest')
        if completed['prepared_sha256'] != file_hash(source/'prepared.json'):
            raise ValueError('Changed source preparation manifest')
        for name in ['evaluation-inputs/rows.parquet', 'protocol.json']:
            if file_hash(source/name) != read(source/'prepared.json')['files'][name]:
                raise ValueError('Changed source protocol or label dates')
        native = source_frames(source, 'native', cfg['arm'])
        paired = source_frames(source, 'paired', cfg['arm'])
        days = sorted(native.date.unique())
        training = {s: native[native.seed.eq(s)].copy() for s in cfg['seeds']}
        metrics, warmup = [], []
        for day in days:
            if any(mature_rows(f, day).date.nunique() < cfg['minimum_training_dates'] for f in training.values()):
                warmup.append(day)
                continue
            dest = root/'walk-forward'/day
            if (dest/'manifest.json').exists():
                verify(dest)
                metrics.append(pd.read_csv(dest/'daily.csv', dtype={'seed': str}))
                continue
            dest.mkdir(parents=True, exist_ok=True)
            heads, daily_parts = {}, []
            for seed, history in training.items():
                head = fit_head(history, day, ridge=cfg['ridge'], min_dates=cfg['minimum_training_dates'])
                heads[seed] = head
                for mode, frame in [('native', native), ('paired', paired)]:
                    section = frame[frame.seed.eq(seed) & frame.date.eq(day)].copy()
                    ranks, daily = evaluate_cross_section(section, head, cfg['top_n'])
                    ranks.to_parquet(dest/f'{mode}-{seed}.parquet', index=False)
                    daily['universe'] = mode
                    daily_parts.append(daily)
            daily = pd.concat(daily_parts, ignore_index=True)
            daily.to_csv(dest/'daily.csv', index=False)
            write_json(dest/'heads.json', heads)
            write_manifest(dest)
            metrics.append(daily)
            write_json(root/'progress.json', dict(stage='walk_forward', date=day, at=utc_now()))
            print('Validated', day, flush=True)
        if not metrics:
            raise ValueError('No dates after warm-up')
        validation = root/'validation'
        validation.mkdir(exist_ok=True)
        daily = pd.concat(metrics, ignore_index=True)
        comparison_cfg = dict(cfg, comparisons=[['calibrated', 'raw']], seeds=[17, 29, 43])
        for mode in ['native', 'paired']:
            subset = daily[daily.universe.eq(mode)].drop(columns='universe')
            summarize_daily(subset, validation, mode)
            write_json(validation/f'{mode}-comparisons.json', compare(subset, comparison_cfg))
        last_label = native.label_end.max()
        as_of = (date.fromisoformat(last_label) + timedelta(days=1)).isoformat()
        for seed, history in training.items():
            write_json(root/'models'/f'{seed}.json', fit_head(history, as_of,
                       ridge=cfg['ridge'], min_dates=cfg['minimum_training_dates']))
        write_json(root/'models/identity.json', dict(arm=cfg['arm'], cost=cfg['cost'],
            source=str(source), source_manifest_sha256=file_hash(source/'validation/completed.json'),
            protocol_sha256=file_hash(root/'binding.json'), seeds=cfg['seeds'],
            status='research_candidate_not_live_promoted', at=utc_now()))
        write_manifest(root/'models')
        write_json(validation/'coverage.json', dict(source_dates=len(days), warmup_dates=warmup,
            evaluation_dates=sorted(daily.date.unique()), native_rows=len(native), paired_rows=len(paired),
            training_embargo='label_end strictly before signal date',
            untouched_holdout=False, hyperparameter_search=False))
        lines = ['# 收益校准层：逐日向前验证', '',
                 f'原模型保留技术指标及收益/排序损失。{len(warmup)} 个日期用于起步，'
                 f'{daily.date.nunique()} 个日期比较；每个日期仅使用此前已经成熟的五日结果。', '',
                 '校准层学习原始预测收益、历史波动率及二者交互与预测误差的关系。'
                 '参数由数据拟合，买卖参考价格、退出日期和 OHLCVA 路径保持原含义。', '',
                 '## 全部可预测股票，三模型组合', '',
                 '| 版本 | 收益 MAE（百分点） | Top20 情景收益 | Top20 高估（百分点） | 原 Top20 高估（百分点） |',
                 '| --- | ---: | ---: | ---: | ---: |']
        summary = daily[(daily.universe == 'native') & (daily.seed == 'ensemble')].groupby('arm').mean(numeric_only=True)
        for arm in ['raw', 'calibrated']:
            r = summary.loc[arm]
            lines.append(f'| {arm} | {r.return_mae_pp:.4f} | {r.top_extrema_scenario_pct:.2f}% | '
                         f'{r.top_prediction_bias_pp:+.2f} | {r.raw_top20_bias_pp:+.2f} |')
        lines += ['', '## 日期改善比例', '', '| 指标 | 改善日期 | 平均改善 | 日期分块 95% 区间 |',
                  '| --- | ---: | ---: | --- |']
        for r in read(validation/'native-comparisons.json'):
            if r['seed'] == 'ensemble':
                lines.append(f'| {r["metric"]} | {r["dates_improved"]}/{r["dates"]} | '
                             f'{r["improvement"]:+.4f} | {r["bootstrap_95"]} |')
        lines += ['', '各随机种子、共同股票池、分时期、较差日期以及完整 Top20 标签的结果另存表格。', '',
                  '## 使用边界', '',
                  '这些日期此前已经用于研究，本次属于带时间隔离的开发验证，不是全新独立测试。'
                  '情景收益按实际 T 最低价买入、预测选定日期最高价卖出并扣除 0.25% 成本，'
                  '不代表实际成交。未知结果保留排名、不以更低排名补位。', '',
                  '校准预期收益是学习后的结果，不等于原始卖出参考价/买入参考价减成本；'
                  '日报分别展示两者，不反向伪造价格。最终模型以全部成熟历史拟合，'
                  '只供后续日期使用，不计入上述验证结果。', '',
                  '未改变主模型、主报告或定时任务。候选输出需在未来日期继续观察。']
        (validation/'report.md').write_text('\n'.join(lines)+'\n')
        write_manifest(validation)
        from verify_token_return_calibration import audit
        audit(root)
        write_json(root/'completed.json', dict(passed=True, at=utc_now(),
            validation_sha256=file_hash(validation/'manifest.json'),
            model_sha256=file_hash(root/'models/manifest.json'),
            verification_sha256=file_hash(root/'verification.json'), live_promoted=False))
        return validation/'report.md'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=BASE/'configs/token-return-calibration-v1.json')
    print(run(parser.parse_args().config), flush=True)
