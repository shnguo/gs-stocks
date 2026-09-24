"""Bounded research-only capture of rolling Eastmoney size-tier net flows."""
import argparse
import concurrent.futures
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.daily_loop import read, verify, write_manifest
from quant_research.daily_token import FIELDS, panel
from quant_research.order_flow_features import parse_eastmoney
from quant_research.storage import file_hash, utc_now, write_json
from quant_research.token_history import stratified_pool
from quant_research.token_transformer import valid_bars

BASE = Path(__file__).resolve().parents[1]


def listed_on(record, day):
    end = record.delisted_at
    return bool(record.listed_at <= day and (pd.isna(end) or end == '' or end > day))


def collect(config):
    cfg = read(config)
    root, source = BASE/cfg['output'], BASE/cfg['price_source']
    root.mkdir(parents=True, exist_ok=True)
    if (root/'protocol.json').exists() and read(root/'protocol.json') != cfg:
        raise ValueError('Changed flow protocol; use new output')
    write_json(root/'protocol.json', cfg)
    meta, bars, instruments = panel(source)
    day = cfg['universe_as_of']
    t = meta['dates'].index(day)
    dates = meta['dates'][t-59:t+1]
    candidates = []
    for symbol, group in bars[bars.date.le(day)].groupby('instrument_id'):
        h = group.set_index('date').reindex(dates)
        x = h[FIELDS].to_numpy(float)
        if len(x) != 60 or not np.isfinite(x).all() or not valid_bars(x[:, :6]).all() or (x[:, 6] <= 0).any():
            continue
        if symbol not in instruments.index:
            continue
        record = instruments.loc[symbol]
        if not listed_on(record, day):
            continue
        candidates.append(symbol)
    chosen = stratified_pool(np.arange(len(candidates)), candidates, day, cfg['stocks'])
    symbols = [candidates[i] for i in chosen]
    if len(symbols) != cfg['stocks']:
        raise ValueError('Insufficient historical cohort for the frozen stock count')
    universe = dict(symbols=symbols, historical_candidates=len(candidates), as_of=day,
        snapshot_sha256=file_hash(source/'snapshot/manifest.json'),
        exchange_counts=pd.Series(symbols).str.split('.').str[1].value_counts().to_dict())
    if (root/'universe.json').exists() and read(root/'universe.json') != universe:
        raise ValueError('Changed frozen cohort')
    write_json(root/'universe.json', universe)
    def one(symbol):
        dest = root/'source'/symbol
        if (dest/'manifest.json').exists():
            verify(dest)
            return read(dest/'capture.json')
        dest.mkdir(parents=True, exist_ok=True)
        code = symbol.split('.')[-1]
        secid = ('1.' if symbol.split('.')[1] == 'xshg' else '0.') + code
        params = dict(secid=secid, fields1='f1,f2,f3,f7',
            fields2=','.join('f'+str(i) for i in range(51, 66)), lmt='120')
        url = 'https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get?'+urllib.parse.urlencode(params)
        record = dict(instrument_id=symbol, provider='eastmoney', params=params, captured_at=utc_now())
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'mootdx-cf-rs/0.1', 'Referer': 'https://quote.eastmoney.com/'})
            with urllib.request.urlopen(request, timeout=15) as response:
                raw = response.read(2*1024*1024+1)
            if len(raw) > 2*1024*1024:
                raise ValueError('Response too large')
            (dest/'response.json').write_bytes(raw)
            frame = parse_eastmoney(raw, code)
            frame.to_parquet(dest/'flows.parquet', index=False)
            record.update(status='captured' if len(frame) else 'no_data', rows=len(frame),
                first=frame.date.min() if len(frame) else None, last=frame.date.max() if len(frame) else None,
                raw_sha256=file_hash(dest/'response.json'))
        except Exception as error:
            record.update(status='failed', error_type=type(error).__name__, message=str(error)[:200])
        write_json(dest/'capture.json', record)
        write_manifest(dest)
        time.sleep(.5)
        return record
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for i, record in enumerate(pool.map(one, symbols), 1):
            records.append(record)
            if i % 32 == 0:
                print('Captured', i, len(symbols), flush=True)
    pd.DataFrame(records).to_parquet(root/'source-coverage.parquet', index=False)
    write_json(root/'source-summary.json', dict(stocks=len(symbols),
        statuses=pd.Series([r['status'] for r in records]).value_counts().to_dict(), at=utc_now(),
        active_buy_sell_available=False, history='rolling recent window',
        source_manifests={s: file_hash(root/'source'/s/'manifest.json') for s in symbols}))
    print(read(root/'source-summary.json')['statuses'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=BASE/'configs/token-order-flow-free-v1.json')
    collect(parser.parse_args().config)
