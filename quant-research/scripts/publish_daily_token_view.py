"""Human-readable daily prices and watchlist export from an immutable token run."""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.daily_loop import read, verify, write_manifest
from quant_research.daily_token import (
    LOW_HIGH,
    OPEN_CLOSE,
    rank_paths,
    scenario_description,
)
from quant_research.storage import file_hash, utc_now, write_json
from quant_research.token_transformer import valid_bars


def timing_summary(run, ranks, info, method):
    """Equal-model peak-date frequencies and prices for the stated reference date."""
    rows = pd.read_parquet(run / 'inputs/forecast-rows.parquet')
    positions = dict(zip(rows.instrument_id, range(len(rows))))
    indices = np.array([positions[s] for s in ranks.instrument_id])
    frequency, buy_prices, sell_prices, day_returns, day_positive = [], [], [], [], []
    for seed in info['checkpoints']:
        paths = np.load(run / f'forecasts/seed{seed}/paths.npy', mmap_mode='r')
        model_frequency, model_buy, model_sell, model_returns, model_positive = [], [], [], [], []
        for i in indices:
            x = paths[i].astype(float)
            x = x[valid_bars(x).all(-1)]
            if len(x) < 16:
                raise ValueError('Insufficient valid paths for the published ranking')
            buy = x[:, 0, 2 if method == LOW_HIGH else 0]
            exits = x[:, 1:, 1] if method == LOW_HIGH else x[:, 1:, 3]
            peak = exits.argmax(1) if method == LOW_HIGH else np.full(len(x), 3)
            model_frequency.append(np.bincount(peak, minlength=4)/len(x))
            model_buy.append(np.median(buy))
            model_sell.append(np.median(exits, axis=0))
            net = exits/buy[:, None]-1-info['cost_scenario']
            model_returns.append(net.mean(0))
            model_positive.append((net > 0).mean(0))
        frequency.append(model_frequency)
        buy_prices.append(model_buy)
        sell_prices.append(model_sell)
        day_returns.append(model_returns)
        day_positive.append(model_positive)
    frequency = np.mean(frequency, axis=0)
    selected = frequency.argmax(1)  # Earliest date breaks ties deterministically.
    row_ids = np.arange(len(ranks))
    dates = info['horizon_dates']
    result = pd.DataFrame(dict(instrument_id=ranks.instrument_id.to_numpy(), buy_date=dates[0],
        buy_reference_price=np.mean(buy_prices, axis=0),
        sell_reference_date=[dates[d+1] for d in selected],
        sell_reference_price=np.mean(sell_prices, axis=0)[row_ids, selected],
        sell_date_frequency=frequency[row_ids, selected],
        selected_day_expected_net_return=np.mean(day_returns, axis=0)[row_ids, selected],
        selected_day_positive_fraction=np.mean(day_positive, axis=0)[row_ids, selected]))
    result['reference_price_net_return'] = result.sell_reference_price/result.buy_reference_price-1-info['cost_scenario']
    for d in range(4):
        result[f'sell_t{d+1}_frequency'] = frequency[:, d]
    return result


def rank_reference_returns(ranks):
    ranks = ranks.rename(columns={'expected_net_return': 'path_mean_ideal_net_return',
        'positive_fraction': 'path_ideal_positive_fraction', 'rank': 'path_mean_rank'}).copy()
    ranks['expected_net_return'] = ranks.reference_price_net_return
    ranks['positive_fraction'] = ranks.selected_day_positive_fraction
    ranks = ranks.sort_values(['expected_net_return', 'instrument_id'], ascending=[False, True]).reset_index(drop=True)
    ranks['rank'] = np.arange(1, len(ranks)+1)
    return ranks


def report_header(info, ranks, method):
    ideal = method == LOW_HIGH
    text = '# 股票预测与收益排名\n\n'
    text += f'行情截至：{info["signal_date"]} 收盘。预测交易日：'+ '、'.join(info['horizon_dates'])+'。\n\n'
    text += '本报告为盘后补发研究报告，不计入前向效果验证。\n\n' if not info['prospective'] else '本报告在首个预测交易日开盘前发布。\n\n'
    text += '**排名依据：预期净收益率 = 卖出参考价 ÷ 买入参考价 − 1 − 往返成本；按此收益率从高到低排序。**\n\n'
    text += f'收益均扣除 {info["cost_scenario"]:.2%} 的往返成本假设。价格单位为元。\n\n'
    text += '买入日为 T；卖出参考日取各模型等权汇总后，预测最高价出现频率最高的日期，并列时取较早日期。卖出参考价是该日预测最高价的模型中位数均值；买入参考价是 T 日预测最低价的模型中位数均值。\n\n' if ideal else '买入、卖出参考日期固定为 T 与 T+4，参考价格为对应开盘、收盘价格的模型中位数均值。\n\n'
    text += '**日线模型只能提供日期，无法给出当天几点几分的买卖时刻。极值价不保证成交。收益率使用参考价的完整精度计算，表格价格保留四位小数。**\n\n'
    text += '| 排名 | 股票 | 预期净收益率（参考价测算） | 买入日期 | 买入参考价 | 卖出参考日期 | 该日卖出参考价 | 卖出日期路径占比 |\n| --- | --- | --- | --- | --- | --- | --- | --- |\n'
    for r in ranks.head(info['top_n']).itertuples():
        text += f'| {r.rank} | {r.name}（{r.instrument_id}） | {r.expected_net_return:.2%} | {r.buy_date} | {r.buy_reference_price:.4f} | {r.sell_reference_date} | {r.sell_reference_price:.4f} | {r.sell_date_frequency:.1%} |\n'
    return text


def publish(run, ranking_method=None):
    run = Path(run)
    verify(run)
    info = read(run / 'run.json')
    method = ranking_method or info.get('ranking_method', OPEN_CLOSE)
    scenario_description(method)  # Reject unsupported method names before deriving any files.
    revised = method != info.get('ranking_method', OPEN_CLOSE)
    store = run.parent.parent
    previous_view = store / 'reports' / (run.name+'-'+method if revised else run.name)
    dest = previous_view.with_name(previous_view.name+'-reference-return-v1')
    if not (dest / 'manifest.json').exists():
        if revised:
            if (previous_view / 'manifest.json').exists():
                verify(previous_view)
                delivery = read(previous_view / 'delivery.json')
                if delivery['run_manifest_sha256'] != file_hash(run / 'manifest.json') or delivery['ranking_method'] != method:
                    raise ValueError('Cached ranking source or method changed')
                ranks = pd.read_csv(previous_view / 'ranking.csv')
                dest.mkdir(parents=True, exist_ok=True)
                for name in ['model-scores.parquet', 'summary.parquet']:
                    shutil.copyfile(previous_view / name, dest / name)
                published = delivery['publication_at']
            else:
                cfg = dict(read(run / 'binding.json')['config'], ranking_method=method)
                ranks = rank_paths(run, cfg, dest)
                published = utc_now()
            info = dict(info, ranking_method=method, published_at=published)
            info['prospective'] = pd.Timestamp(info['published_at']) < pd.Timestamp(info['horizon_dates'][0]+'T09:15:00', tz='Asia/Shanghai')
            info['mode'] = 'revised_research_watchlist' if info['prospective'] else 'late_research_watchlist'
        else:
            ranks = pd.read_csv(run / 'ranking.csv')
        timing = timing_summary(run, ranks, info, method)
        ranks = rank_reference_returns(ranks.merge(timing, on='instrument_id', validate='one_to_one'))
        # This is a new ranking publication, not a backdated presentation revision.
        info = dict(info, published_at=utc_now())
        info['prospective'] = pd.Timestamp(info['published_at']) < pd.Timestamp(info['horizon_dates'][0]+'T09:15:00', tz='Asia/Shanghai')
        quantiles = pd.read_parquet(run / 'daily-quantiles.parquet')
        counts = quantiles.groupby(['instrument_id', 'day', 'field']).seed.transform('nunique')
        mean = quantiles[counts == len(info['checkpoints'])].groupby(['instrument_id', 'day', 'field'])[['q10', 'q50', 'q90']].mean().reset_index()
        mean['date'] = mean.day.map(dict(enumerate(info['horizon_dates'])))
        dest.mkdir(parents=True, exist_ok=True)
        mean.to_csv(dest / 'daily-ohlcva.csv', index=False)
        ranks.to_csv(dest / 'ranking.csv', index=False)
        text = report_header(info, ranks, method)
        text += '\n## 个股买卖参考与每日预测\n\n以下价格为各模型中位数的等权平均；完整分位数保存在数据文件中。日期路径占比不是实际卖出成功率。\n'
        for row in ranks.head(info['top_n']).itertuples():
            values = mean[mean.instrument_id == row.instrument_id].pivot(index='date', columns='field', values='q50')
            text += f'\n### {row.rank}. {row.name} · {row.instrument_id}\n\n'
            text += f'买入参考：{row.buy_date}，{row.buy_reference_price:.4f} 元。卖出参考：{row.sell_reference_date}，{row.sell_reference_price:.4f} 元。\n\n'
            text += f'预期净收益率（参考价测算）：**{row.expected_net_return:.2%}**。计算方式：卖出参考价 ÷ 买入参考价 − 1 − {info["cost_scenario"]:.2%}，使用未四舍五入的参考价计算；不代表已实现收益。\n\n'
            if method == LOW_HIGH:
                text += '卖出日期路径分布：'+'；'.join(f'{info["horizon_dates"][d]}：{getattr(row, f"sell_t{d}_frequency"):.1%}' for d in range(1,5))+'。\n\n'
            text += '| 日期 | 开盘价 | 最高价 | 最低价 | 收盘价 | 成交量（股） | 成交额（元） |\n| --- | --- | --- | --- | --- | --- | --- |\n'
            for date, p in values.iterrows():
                text += f'| {date} | {p["open"]:.2f} | {p["high"]:.2f} | {p["low"]:.2f} | {p["close"]:.2f} | {p["volume"]:,.0f} | {p["amount"]:,.0f} |\n'
        text += '\n[完整收益排名与买卖参考](ranking.csv) · [每日六项行情分位数](daily-ohlcva.csv)\n'
        (dest / 'report.md').write_text(text)
        write_json(dest / 'delivery.json', dict(at=utc_now(), source_run=str(run.resolve()),
            run_manifest_sha256=file_hash(run / 'manifest.json'), dates=info['horizon_dates'],
            ranking_method=method, revised=revised, publication_at=info.get('published_at'),
            ranking_score_method='reference_price_ratio', layout_version='reference-return-v1', timing_resolution='trading_date_only',
            sell_date_rule='equal_model_peak_date_frequency_earliest_tie' if method == LOW_HIGH else 'fixed_T_plus_4',
            ranked=len(ranks), daily_rows=len(mean), top_n=info['top_n'], prospective=info['prospective']))
        write_manifest(dest)
    verify(dest)
    if read(dest / 'delivery.json')['run_manifest_sha256'] != file_hash(run / 'manifest.json'):
        raise ValueError('Delivery source changed')
    delivery = read(dest / 'delivery.json')
    write_json(store / 'reference-publications' / (run.name+'.json'), dict(report=str(dest.resolve()),
        manifest_sha256=file_hash(dest / 'manifest.json'), publication_at=delivery['publication_at']))
    latest = store / 'latest.json'
    if not latest.exists() or read(latest)['signal_date'] <= info['signal_date']:
        write_json(latest, dict(run=str(run.resolve()), report=str((dest / 'report.md').resolve()),
            ranking_method=method, ranking_score_method='reference_price_ratio', ranking=str((dest / 'ranking.csv').resolve()), daily_forecasts=str((dest / 'daily-ohlcva.csv').resolve()), signal_date=info['signal_date']))
    return dest / 'report.md'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--ranking-method')
    args = parser.parse_args()
    print(publish(args.run, args.ranking_method))
