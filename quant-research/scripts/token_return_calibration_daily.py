"""Publish a learned-return candidate beside the manually updated ranking model."""
import argparse
import fcntl
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.daily_loop import digest, read, verify, write_manifest
from quant_research.daily_token import future_known, panel
from quant_research.return_calibration import fit_head, rank_head, volatility_pp
from quant_research.storage import file_hash, utc_now, write_json

BASE = Path(__file__).resolve().parents[1]


def mature_publications(store, bars, as_of, cost):
    """Only timely saved RAW forecasts can provide new calibration supervision."""
    groups = {s: g.set_index('date') for s, g in bars.groupby('instrument_id')}
    records, lineage = [], {}
    for path in sorted((store/'runs').glob('*/publication.json')):
        verify(path.parent)
        info = read(path)
        if not info['prospective'] or info['horizon_dates'][-1] >= as_of:
            continue
        if info['cost'] != cost:
            raise ValueError('Historical cost differs')
        lineage[str(path.parent/'manifest.json')] = file_hash(path.parent/'manifest.json')
        raw = pd.read_csv(path.parent/'raw-ranking.csv')
        dates = [info['signal_date'], *info['horizon_dates']]
        for row in raw.itertuples():
            if row.instrument_id not in groups:
                continue
            block = groups[row.instrument_id].reindex(dates)
            if not future_known(block, 0, 5):
                continue
            actual = block.loc[row.sell_reference_date, 'high']/block.loc[row.buy_date, 'low']-1-cost
            records.append(dict(instrument_id=row.instrument_id, date=info['signal_date'],
                label_end=info['horizon_dates'][-1], label_known=True,
                predicted=row.expected_net_return, volatility_pp=row.volatility_pp,
                actual_extrema_scenario=float(actual)))
    return pd.DataFrame(records), lineage


def publish(matched, config):
    cfg = read(config)
    fitted, store = BASE/cfg['output'], BASE/cfg['daily_store']
    matched = Path(matched).resolve()
    store.mkdir(parents=True, exist_ok=True)
    with (store/'cycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify(matched)
        receipt = read(matched.with_name(matched.name+'-verification.json'))
        if not receipt['passed'] or receipt['run_manifest_sha256'] != file_hash(matched/'manifest.json'):
            raise ValueError('Matched daily model audit required')
        completed = read(fitted/'completed.json')
        if (not completed['passed'] or completed['verification_sha256'] != file_hash(fitted/'verification.json')
                or completed['model_sha256'] != file_hash(fitted/'models/manifest.json')):
            raise ValueError('Calibration completion hashes differ')
        verify(fitted/'models')
        if not read(fitted/'verification.json')['passed']:
            raise ValueError('Calibration audit required')
        source = BASE/cfg['source']
        original = read(source/'protocol.json')
        protocol = read(matched/'prepared/binding.json')['config']
        if protocol['experiment'] != original['prior']:
            raise ValueError('Calibration backbone lineage differs')
        for key in ['samples', 'seeds', 'minimum_legal_paths', 'cost', 'return_ranking_loss']:
            if protocol[key] != original[key]:
                raise ValueError('Incompatible calibration protocol: '+key)
        if read(fitted/'binding.json')['config'] != cfg:
            raise ValueError('Calibration configuration differs from trained head')
        info = read(matched/'prepared/input.json')
        signal = info['signal_date']
        dest = store/'runs'/signal
        code = [Path(__file__), BASE/'src/quant_research/return_calibration.py']
        binding = dict(config_sha256=digest(cfg), matched=str(matched),
            matched_sha256=file_hash(matched/'manifest.json'),
            fitted_sha256=file_hash(fitted/'completed.json'),
            code={str(p): file_hash(p) for p in code})
        if (dest/'manifest.json').exists():
            verify(dest)
            if read(dest/'binding.json') != binding:
                raise ValueError('Changed candidate publication binding')
            return dest/'report.md'
        if dest.exists():
            raise ValueError('Incomplete publication exists; preserve it and use a new daily store')
        meta, bars, _ = panel(Path(info['source']))
        if (meta['price_data_through'] != signal or bars.date.max() != signal
                or file_hash(Path(info['source'])/'snapshot/manifest.json') != info['snapshot_sha256']):
            raise ValueError('Changed signal-close source snapshot')
        raw = pd.read_csv(matched/'indicators_ranking-ranking.csv')
        history_dates = meta['dates'][meta['dates'].index(signal)-20:meta['dates'].index(signal)+1]
        groups = {s: g.set_index('date') for s, g in bars.groupby('instrument_id')}
        raw['volatility_pp'] = [volatility_pp(groups[s].reindex(history_dates).close,
                                             groups[s].reindex(history_dates).factor)
                                for s in raw.instrument_id]
        # Confirm raw return still exactly means the displayed reference-price ratio.
        np.testing.assert_allclose(raw.expected_net_return,
            raw.sell_reference_price/raw.buy_reference_price-1-cfg['cost'], atol=1e-12, rtol=1e-12)
        head = read(fitted/'models/ensemble.json')
        added, lineage = mature_publications(store, bars, signal, cfg['cost'])
        if len(added):
            from token_return_calibration import source_frames
            historic = source_frames(source, 'native', cfg['arm'])
            historic = historic[historic.seed.eq('ensemble')]
            train = pd.concat([historic, added], ignore_index=True)
            head = fit_head(train, signal, ridge=cfg['ridge'], min_dates=cfg['minimum_training_dates'])
        candidate = rank_head(raw.assign(date=signal, predicted=raw.expected_net_return), head)
        candidate['reference_price_net_return'] = candidate.raw_predicted
        candidate['expected_net_return'] = candidate.predicted
        # Immutable dates/prices are carried through; only learned return and rank differ.
        original_rows = raw.set_index('instrument_id')
        candidate_rows = candidate.set_index('instrument_id').reindex(original_rows.index)
        for field in ['buy_date', 'sell_reference_date', 'buy_reference_price', 'sell_reference_price']:
            np.testing.assert_array_equal(original_rows[field], candidate_rows[field])
        stamp = utc_now()
        prospective = bool(pd.Timestamp(stamp) < pd.Timestamp(info['horizon_dates'][0]+'T09:15:00', tz='Asia/Shanghai'))
        publication = dict(signal_date=signal, horizon_dates=info['horizon_dates'], completed_at=stamp,
            prospective=prospective, cost=cfg['cost'], top_n=cfg['top_n'],
            ranking_files={'raw': 'raw-ranking.csv', 'calibrated': 'calibrated-ranking.csv'},
            model_transfer='Historical calibration applied to incrementally updated ranking backbone; prospective performance pending')
        dest.mkdir(parents=True)
        write_json(dest/'binding.json', binding)
        write_json(dest/'head.json', head)
        write_json(dest/'training-update.json', dict(new_mature_rows=len(added),
            new_mature_dates=int(added.date.nunique()) if len(added) else 0,
            publication_lineage=lineage, snapshot_sha256=info['snapshot_sha256']))
        write_json(dest/'publication.json', publication)
        raw.to_csv(dest/'raw-ranking.csv', index=False)
        candidate.to_csv(dest/'calibrated-ranking.csv', index=False)
        lines = ['# 收益校准模型候选报告', '',
            f'行情截至 {signal}；预测日期：'+ '、'.join(info['horizon_dates'])+'。', '',
            '按校准后预期情景收益排序。校准层从已成熟预测误差学习，'
            '保留收益/排序损失 Transformer 的价格路径与买卖日期。', '',
            '参考价测算收益 = 卖出参考价 ÷ 买入参考价 − 1 − 0.25%。'
            '校准后收益是另一个学习结果，不能由这组原始价格直接相除得到。', '',
            f'本次新增成熟训练样本 {len(added)} 条，校准层训练标签截至 {head["trained_labels_through"]}。', '',
            '及时前向候选，结果待成熟。' if prospective else '历史补发候选报告，不计入前向效果验证。', '',
            '| 排名 | 股票 | 校准后预期收益 | 参考价测算收益 | 买入日 | 买入参考价 | 卖出日 | 卖出参考价 |',
            '| ---: | --- | ---: | ---: | --- | ---: | --- | ---: |']
        for r in candidate.head(cfg['top_n']).itertuples():
            lines.append(f'| {r.rank} | {r.name}（{r.instrument_id}） | {r.expected_net_return:.2%} | '
                f'{r.reference_price_net_return:.2%} | {r.buy_date} | {r.buy_reference_price:.4f} | '
                f'{r.sell_reference_date} | {r.sell_reference_price:.4f} |')
        lines += ['', '历史校准向增量训练模型迁移的效果需由后续及时预测验证。'
                  '日线极值不保证成交，收益不是已实现利润。本文件为候选模型输出。']
        (dest/'report.md').write_text('\n'.join(lines)+'\n')
        write_manifest(dest)
        latest = store/'latest.json'
        if not latest.exists() or read(latest)['signal_date'] <= signal:
            write_json(latest, dict(signal_date=signal, report=str(dest/'report.md'),
                                   manifest_sha256=file_hash(dest/'manifest.json')))
        from token_ranking_shadow import review_shadow
        review_shadow(store, Path(info['source']))
        return dest/'report.md'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=BASE/'configs/token-return-calibration-v1.json')
    parser.add_argument('--run', type=Path, help='Completed matched ranking run; defaults to current matched state')
    args = parser.parse_args()
    matched = args.run or Path(read(BASE/'artifacts/token-ranking-daily-v1/current.json')['root'])
    print(publish(matched, args.config), flush=True)
