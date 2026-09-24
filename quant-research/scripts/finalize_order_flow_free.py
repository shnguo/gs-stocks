"""Audit raw flow inputs, matched exposure and all saved forecast rankings."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from finalize_token_ranking import freeze_and_score
from finalize_token_three_ideas import compare, summarize_daily
from verify_token_ranking import independent_stats

from quant_research.daily_loop import read, verify, write_manifest
from quant_research.order_flow_features import FEATURES
from quant_research.return_ranking_loss import reference_prices
from quant_research.storage import file_hash, utc_now, write_json
from quant_research.token_indicators import FEATURES as INDICATORS
from quant_research.token_transformer import restore_model

BASE = Path(__file__).resolve().parents[1]


def audit_features(root, cfg, rows):
    """Independent explicit session slicing of raw response amounts / turnover."""
    source = BASE/cfg['price_source']
    verify(source/'snapshot')
    bars = pd.read_parquet(source/'snapshot/bars.parquet')
    calendar = read(root/'inputs/calendar.json')
    arrays = np.load(root/'inputs/features.npy', mmap_mode='r')
    extra = arrays[:, -1, len(INDICATORS):]
    assert np.isnan(arrays[:, :-1, len(INDICATORS):]).all()
    expected = np.full(extra.shape, np.nan, np.float64)
    summary = read(root/'source-summary.json')
    for symbol, group in rows.groupby('instrument_id'):
        folder = root/'source'/symbol
        verify(folder)
        assert file_hash(folder/'manifest.json') == summary['source_manifests'][symbol]
        if not (folder/'response.json').exists():
            continue
        data = read(folder/'response.json').get('data')
        if not data:
            continue
        assert data['code'] == symbol.split('.')[-1]
        flows = {}
        for line in data['klines']:
            cells = line.split(',')
            assert cells[0] not in flows
            flows[cells[0]] = [float(v) if v not in ('-', '', 'null') else np.nan for v in cells[2:6]]
        amount = bars[bars.instrument_id.eq(symbol)].set_index('date').amount
        for r in group.itertuples():
            end = calendar.index(r.date)-cfg['flow_lag_sessions']
            for j, window in enumerate([1, 5, 20]):
                days = calendar[end-window+1:end+1]
                if len(days) != window:
                    continue
                values = np.array([flows.get(d, [np.nan]*4) for d in days])
                turnover = amount.reindex(days).to_numpy(float)
                if not np.isfinite(turnover).all() or (turnover <= 0).any():
                    continue
                expected[r.row_id, j*4:j*4+4] = values.sum(0)/turnover.sum()
    np.testing.assert_allclose(extra, expected, atol=1e-7, rtol=1e-6, equal_nan=True)
    training = rows.part.eq('training').to_numpy()
    values = expected[training]
    with np.load(root/'inputs/normalizer.npz') as normalizer:
        center = np.nanmean(values, axis=0)
        scale = np.nanstd(values, axis=0)
        scale = np.where(scale > 1e-6, scale, 1.)
        np.testing.assert_allclose(normalizer['center'], center, atol=1e-7, rtol=1e-6)
        np.testing.assert_allclose(normalizer['scale'], scale, atol=1e-7, rtol=1e-6)
    assert rows[rows.part.eq('training')].label_end.max() < rows[rows.part.eq('evaluation')].date.min()
    return int(expected.size)


def all_paths(root, cfg, seed, arm, ids):
    dest = root/f'seed{seed}'/arm/'forecast'
    verify(dest)
    result, identities = [], []
    for chunk in sorted(p for p in dest.iterdir() if p.is_dir()):
        verify(chunk)
        identity = read(chunk/'identity.json')
        assert identity['model_sha256'] == file_hash(dest.parent/'training/model.pt')
        values = np.load(chunk/'paths.npy')
        assert values.shape == (len(identity['ids']), cfg['samples'], 5, 6)
        identities.extend(identity['ids'])
        result.append(values)
    np.testing.assert_array_equal(identities, ids)
    return np.concatenate(result)


def finalize(root):
    cfg = read(root/'protocol.json')
    for path, expected in {**read(root/'binding.json')['sources'], **read(root/'binding.json')['code']}.items():
        if file_hash(Path(path)) != expected:
            raise ValueError('Changed source/code: '+path)
    verify(root/'inputs')
    rows = pd.read_parquet(root/'inputs/rows.parquet')
    feature_checks = audit_features(root, cfg, rows)
    trained = []
    for seed in cfg['seeds']:
        matching = []
        for arm in cfg['variants']:
            dest = root/f'seed{seed}'/arm/'training'
            verify(dest)
            info = read(dest/'training.json')
            matching.append(info)
            model, saved = restore_model(dest/'model.pt')
            assert tuple(model.config.auxiliary_features) == INDICATORS+FEATURES
            assert saved['trained_labels_through'] < cfg['evaluation'][0]
            assert saved['parent_sha256'] == info['parent_sha256']
            extra_weights = model.auxiliary_projection[0].weight[:, len(INDICATORS):len(INDICATORS)+len(FEATURES)]
            if arm == 'control':
                assert torch.count_nonzero(extra_weights) == 0
            else:
                assert torch.count_nonzero(extra_weights) > 0
            trained.append(dict(seed=seed, arm=arm, **info))
        for key in ['rows', 'epochs', 'parameters', 'exposure_sha256', 'parent_sha256']:
            assert matching[0][key] == matching[1][key], key
    evaluation = rows[rows.part.eq('evaluation')].reset_index(drop=True)
    ids = evaluation.row_id.to_numpy()
    future = np.load(root/'inputs/future.npy', mmap_mode='r')[ids]
    known = np.load(root/'inputs/valid.npy', mmap_mode='r')[ids].all(1)
    volatility = np.load(root/'inputs/volatility.npy', mmap_mode='r')[ids]
    frames, probes = [], 0
    for arm in cfg['variants']:
        paths = {str(s): all_paths(root, cfg, s, arm, ids) for s in cfg['seeds']}
        stats = {s: independent_stats(v, cfg['minimum_legal_paths']) for s, v in paths.items()}
        for seed in [str(s) for s in cfg['seeds']]+['ensemble']:
            selected = list(paths) if seed == 'ensemble' else [seed]
            used = [stats[s] for s in selected]
            eligible = np.logical_and.reduce([x[0] for x in used])
            offset = np.mean([x[1] for x in used], axis=0).argmax(1)+1
            buy = np.mean([x[2] for x in used], axis=0)
            sell = np.mean([x[3] for x in used], axis=0)[np.arange(len(ids)), offset-1]
            score = sell/buy-1-cfg['cost']
            f = evaluation[['instrument_id', 'date']].copy()
            f['local_row'] = ids
            f['arm'], f['seed'] = arm, seed
            f['buy_reference'], f['sell_reference'], f['sell_offset'], f['predicted'] = buy, sell, offset, score
            f['volatility_pp'] = volatility
            f = f[eligible].copy()
            # Rank input eligibility and exit choice contain no target information.
            for day, group in evaluation.groupby('date'):
                indices = group.index.to_numpy()
                for i in indices[np.linspace(0, len(indices)-1, min(8, len(indices)), dtype=int)]:
                    ref = reference_prices(np.stack([paths[s][i] for s in selected]), cfg['cost'], cfg['minimum_legal_paths'])
                    if ref is None:
                        assert not eligible[i]
                    else:
                        assert eligible[i] and ref['sell_offset'] == offset[i]
                        np.testing.assert_allclose([ref['predicted'], ref['buy_reference'], ref['sell_reference']],
                                                   [score[i], buy[i], sell[i]], atol=1e-12, rtol=1e-12)
                    probes += 1
            actual = future[np.arange(len(ids)), offset, 1]/future[:, 0, 2]-1-cfg['cost']
            f['label_known'] = known[f.index]
            f['actual_extrema_scenario'] = np.where(known, actual, np.nan)[f.index]
            f['absolute_error'] = abs(f.predicted-f.actual_extrema_scenario)
            adverse = np.array([(future[i, :offset[i]+1, 2].min()/future[i, 0, 2]-1)*100 if known[i] else np.nan for i in range(len(ids))])
            f['adverse_excursion_pct'] = adverse[f.index]
            frames.append(f)
        del paths, stats
    frame = pd.concat(frames, ignore_index=True)
    counts = frame.groupby('local_row').size()
    shared = counts[counts.eq(len(cfg['variants'])*(len(cfg['seeds'])+1))].index
    dest = root/'validation'
    dest.mkdir(exist_ok=True)
    compare_cfg = dict(cfg, comparisons=[['size_flow', 'control']])
    checked = 0
    for mode, subset in [('native', frame), ('paired', frame[frame.local_row.isin(shared)])]:
        rankings, daily = freeze_and_score(subset, cfg['top_n'])
        hidden = subset.copy()
        hidden['label_known'] = False
        hidden[['actual_extrema_scenario', 'absolute_error', 'adverse_excursion_pct']] = np.nan
        blind, _ = freeze_and_score(hidden, cfg['top_n'])
        pd.testing.assert_frame_equal(rankings[['local_row', 'arm', 'seed', 'rank', 'sell_offset']],
                                      blind[['local_row', 'arm', 'seed', 'rank', 'sell_offset']])
        # Separate scalar top selection and errors check the vectorized report.
        for (arm, seed, day), g in rankings.groupby(['arm', 'seed', 'date']):
            order = sorted(g.to_dict('records'), key=lambda x: (-x['predicted'], x['instrument_id']))
            top = [r for r in order[:cfg['top_n']] if r['label_known']]
            metric = daily[daily.arm.eq(arm) & daily.seed.eq(seed) & daily.date.eq(day)].iloc[0]
            assert metric.top_unknown == cfg['top_n']-len(top)
            np.testing.assert_allclose(metric.top_extrema_scenario_pct,
                np.mean([r['actual_extrema_scenario'] for r in top])*100 if top else np.nan, equal_nan=True)
            observed = [r for r in order if r['label_known']]
            np.testing.assert_allclose(metric.return_mae_pp,
                np.mean([abs(r['predicted']-r['actual_extrema_scenario']) for r in observed])*100, equal_nan=True)
        rankings.to_parquet(dest/f'{mode}-rankings.parquet', index=False)
        summarize_daily(daily, dest, mode)
        write_json(dest/f'{mode}-comparisons.json', compare(daily, compare_cfg))
        markets = []
        for exchange in ['xshg', 'xshe', 'xbse']:
            _, per_day = freeze_and_score(subset[subset.instrument_id.str.split('.').str[1].eq(exchange)], cfg['top_n'])
            per_day['exchange'] = exchange
            markets.append(per_day)
        pd.concat(markets, ignore_index=True).to_csv(dest/f'{mode}-exchange-daily.csv', index=False)
        checked += len(rankings)
    for path, expected in read(root/'live-state.json').items():
        assert file_hash(Path(path)) == expected, path
    coverage = read(root/'inputs/coverage.json')
    write_json(dest/'verification.json', dict(passed=True, at=utc_now(), feature_values_independently_checked=feature_checks,
        raw_path_reference_checks=probes, ranking_rows_checked=checked, matched_exposure=True,
        equal_parameter_count=True, no_future_label_ranking=True, live_state_unchanged=True,
        active_buy_sell_tested=False, size_net_flow_tested=True, complete_frozen_pilot=True,
        full_market_validation=False, dates=int(evaluation.date.nunique()), inputs=len(evaluation)))
    summary = pd.read_csv(dest/'paired-summary.csv', dtype={'seed': str})
    lines = ['# 免费大小单净流入：Transformer 配对试验', '',
        '实际测试两组：保留技术指标及收益/排序损失的对照；在相同模型上增加大小单净流入的候选。'
        '主动买卖四组方案因账户权限不足未执行，且不以净额冒充买卖总额。', '',
        f'沪深北各 128 只，共 384 只预先固定股票；训练 {coverage["parts"]["training"]["rows"]} 条、'
        f'{coverage["parts"]["training"]["dates"]} 个日期；验证 {len(evaluation)} 条、{evaluation.date.nunique()} 个日期。', '',
        '资金流统一滞后一个交易日，以成交额归一化，使用小/中/大/超大单的 1、5、20 日净额。'
        '仅在信号日最后一个历史 token 注入。未来资金流保持未知。'
        '两组参数数目、初始化、训练行、批次顺序、四轮训练预算及损失相同。', '',
        '## 共同可预测股票池', '',
        '| 随机种子 | 版本 | 收益 MAE（百分点） | Top20 情景收益 | Top20 高估（百分点） | 高波动组占比 |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for seed in [str(s) for s in cfg['seeds']]+['ensemble']:
        for arm in cfg['variants']:
            r = summary[summary.seed.eq(seed) & summary.arm.eq(arm)].iloc[0]
            lines.append(f'| {seed} | {arm} | {r.return_mae_pp:.4f} | {r.top_extrema_scenario_pct:.2f}% | '
                         f'{r.top_prediction_bias_pp:+.2f} | {r.top_high_volatility_fraction:.1%} |')
    lines += ['', '## 改善日期与幅度', '', '| 模型 | 指标 | 平均改善 | 改善日期 | 日期分块 95% 区间 |',
              '| --- | --- | ---: | ---: | --- |']
    for r in read(dest/'paired-comparisons.json'):
        if r['metric'] not in ['return_mae_pp', 'top_extrema_scenario_pct', 'top_prediction_bias_pp']:
            continue
        lo, hi = r['bootstrap_95']
        lines.append(f'| {r["seed"]} | {r["metric"]} | {r["improvement"]:+.4f} | '
            f'{r["dates_improved"]}/{r["dates"]} | [{lo:+.4f}, {hi:+.4f}] |')
    lines += ['', '## 边界', '', cfg['limitations'], '',
        '本次为 384 只股票的近期开发试验，不能当成全 A 股验证。'
        '日期改善占比是历史描述，不是未来胜率。主模型没有替换，没有新增定时任务。', '',
        '情景收益采用实际 T 日最低价买入、模型预先选定 T+1 至 T+4 某日的实际最高价卖出，'
        '扣除 0.25% 成本。不是事后挑选最高卖出日，也不代表成交利润。'
        '未知结果保留排名、不补位。原始股票池、共同股票池及三个市场的结果另存。']
    (dest/'report.md').write_text('\n'.join(lines)+'\n')
    write_manifest(dest)
    write_json(root/'completed.json', dict(passed=True, at=utc_now(),
        validation_sha256=file_hash(dest/'manifest.json'), report=str(dest/'report.md'),
        scope='size-tier net flow matched sampled Transformer pilot', live_promoted=False))
    print('Completed', dest/'report.md', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    finalize(parser.parse_args().root)
