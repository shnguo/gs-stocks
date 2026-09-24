"""Past-only daily input/label cache for the registered rolling experiment."""
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.price_strategy import price_labels
from quant_research.storage import file_hash, utc_now, write_json


def read(path):
    return json.loads(path.read_text())


def arrays(path):
    with np.load(path) as z:
        return {k:z[k] for k in z.files}


def verify(directory, manifest):
    for name, expected in manifest.items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory.resolve()) or file_hash(path) != expected:
            raise ValueError('Changed cached evidence')


class AsOfData:
    def __init__(self, source, cache, sealed):
        self.source, self.cache = Path(source).resolve(), Path(cache).resolve()
        self.manifest = read(self.source/'panel-h5/manifest.json')
        self.dates = self.manifest['dates']
        self.axis = {d:i for i,d in enumerate(self.dates)}
        self.sealed = sealed
        self.values = np.load(self.source/'panel-h5/values.npy', mmap_mode='r')
        self.cache.mkdir(parents=True, exist_ok=True)

    def inputs(self, day, asof):
        if day not in self.axis or day > asof or asof >= self.sealed:
            raise ValueError('Input date unavailable at this as-of')
        directory = self.cache/day
        marker = directory/'input-manifest.json'
        if marker.exists():
            meta = read(marker)
            if meta['date'] != day or meta['dataset_id'] != self.manifest['dataset_id']:
                raise ValueError('Cached input belongs to another source')
            verify(directory,meta['files'])
        else:
            if directory.exists():
                raise ValueError('Incomplete input cache; preserve failed attempt')
            # Deliberately exclude all future-return/label columns in the sample table.
            rows = pd.read_parquet(self.source/'panel-h5/samples.parquet', filters=[('date','==',day)],
                columns=['date','instrument_id','stock_index','date_index','risk_status','trading_eligible','source_is_st'])
            rows = rows.sort_values(['date','instrument_id']).reset_index(drop=True)
            if rows.empty or rows.duplicated(['date','instrument_id']).any() or not rows.date.eq(day).all():
                raise ValueError('Invalid or missing signal cohort')
            quote = pd.read_parquet(self.source/'snapshot/bars.parquet',filters=[('date','==',day)],
                columns=['instrument_id','date','close'])
            refs = quote.set_index(['instrument_id','date'],verify_integrity=True).close.reindex(
                pd.MultiIndex.from_frame(rows[['instrument_id','date']])).to_numpy(float)
            good = np.isfinite(refs) & (refs > 0)
            rows['input_status'] = np.where(good,'available','missing_signal_quote')
            directory.mkdir()
            rows.to_parquet(directory/'cohort.parquet',index=False)
            rows = rows.loc[good].reset_index(drop=True)
            if rows.empty:
                raise ValueError('No usable input; preserve cohort evidence')
            if not (rows.date_index.to_numpy() == self.axis[day]).all():
                raise ValueError('Input calendar coordinates differ')
            x = self.values[rows.stock_index.to_numpy(int),rows.date_index.to_numpy(int)].copy()
            if not np.isfinite(x).all():
                raise ValueError('Input features missing')
            rows.to_parquet(directory/'rows.parquet',index=False)
            np.savez_compressed(directory/'inputs.npz',x=x,reference=refs[good])
            write_json(marker,dict(date=day,dataset_id=self.manifest['dataset_id'],created_at=utc_now(),
                first_requested_asof=asof,source_read_dates=[day],source_read_kind='past input columns only',files={
                    name:file_hash(directory/name) for name in ['cohort.parquet','rows.parquet','inputs.npz']}))
        return pd.read_parquet(directory/'rows.parquet'),arrays(directory/'inputs.npz')

    def labels(self, day, asof):
        t = self.axis.get(day,-1)
        if t < 0 or t+5 >= len(self.dates) or self.dates[t+5] > asof or asof >= self.sealed:
            raise ValueError('Five-day labels have not matured by this as-of')
        rows,data = self.inputs(day,asof)
        directory = self.cache/day
        marker = directory/'label-manifest.json'
        if marker.exists():
            meta = read(marker)
            if meta['date'] != day or meta['label_end'] != self.dates[t+5] or meta['label_end'] > asof:
                raise ValueError('Cached label maturity differs')
            if meta['input_manifest_sha256'] != file_hash(directory/'input-manifest.json'):
                raise ValueError('Label cohort changed')
            verify(directory,meta['files'])
        else:
            if (directory/'labels.npz').exists():
                raise ValueError('Incomplete label cache; preserve prior attempt')
            days = self.dates[t:t+6]
            bars = pd.read_parquet(self.source/'snapshot/bars.parquet',filters=[('date','in',days)])
            actions = pd.read_parquet(self.source/'snapshot/actions.parquet',filters=[('ex_date','in',days)])
            if (bars.date > asof).any() or (actions.ex_date > asof).any():
                raise ValueError('Source read returned future evidence')
            boundary = (date.fromisoformat(asof)+timedelta(days=1)).isoformat()
            labels = price_labels(bars,rows,self.dates,actions,boundary)
            np.testing.assert_array_equal(labels['reference'],data['reference'])
            np.savez_compressed(directory/'labels.npz',**labels)
            write_json(marker,dict(date=day,label_end=days[-1],first_requested_asof=asof,created_at=utc_now(),
                source_read_dates=days,input_manifest_sha256=file_hash(directory/'input-manifest.json'),
                files={'labels.npz':file_hash(directory/'labels.npz')}))
        labels = arrays(directory/'labels.npz')
        np.testing.assert_array_equal(labels['reference'],data['reference'])
        return rows,{**data,**labels}

    def stack(self, days, asof):
        parts = [self.labels(d,asof) for d in days]
        return pd.concat([p[0] for p in parts],ignore_index=True),{
            k:np.concatenate([p[1][k] for p in parts]) for k in parts[0][1]}

    def input_receipt(self, days):
        return {f'{d}/input-manifest.json':file_hash(self.cache/d/'input-manifest.json') for d in days}

    def label_receipt(self, days, asof):
        result = self.input_receipt(days)
        for day in days:
            path = self.cache/day/'label-manifest.json'
            if read(path)['label_end'] > asof:
                raise ValueError('Receipt contains unavailable labels')
            result[f'{day}/label-manifest.json'] = file_hash(path)
        return result
