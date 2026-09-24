"""Extend only historical OHLCVA inputs needed by the frozen plan cohorts."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.storage import file_hash, utc_now, write_json


def read(p):
    return json.loads(p.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['training-root', 'parent-experiment', 'old-inputs', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    root, parent, oldroot, out = [x.resolve() for x in [a.training_root, a.parent_experiment, a.old_inputs, a.output]]
    panel = read(root/'panel-h5/manifest.json')
    old = read(oldroot/'manifest.json')
    snapshot = read(root/'snapshot/manifest.json')
    protocol = read(parent/'protocol.json')
    rows = [pd.read_parquet(parent/f'fold-{f:02d}-{s}-rows.parquet') for f in protocol['fold_indices'] for s in ['calibration', 'evaluation']]
    end = max(r.label_end.max() for r in rows)
    if end >= protocol['sealed_holdout_start']:
        raise ValueError('Requested raw horizon crosses sealed boundary')
    if not (panel['dataset_id'] == old['dataset_id'] == snapshot['dataset_id']):
        raise ValueError('Dataset differs')
    if panel['instruments'] != old['instruments'] or file_hash(root/'snapshot/bars.parquet') != old['raw_bars_sha256']:
        raise ValueError('Raw source identity differs')
    if file_hash(oldroot/'values.npy') != old['values_sha256']:
        raise ValueError('Old input cache changed')
    if file_hash(root/'panel-h5/values.npy') != panel['files']['values.npy']:
        raise ValueError('Historical suspension mask changed')
    dates = [d for d in panel['dates'] if d <= end]
    if dates[:len(old['dates'])] != old['dates']:
        raise ValueError('Date axes are not a prefix extension')
    date_axis = {d: i for i, d in enumerate(dates)}
    for r in rows:
        for row in r.itertuples():
            t = date_axis[row.date]
            if t < 59 or t+5 >= len(dates) or dates[t+5] != row.label_end:
                raise ValueError('Required historical window or future calendar is incomplete')
    fields = old['fields']
    frame = pd.read_parquet(root/'snapshot/bars.parquet', columns=['instrument_id', 'date', *fields, 'sequence_id'], filters=[('date', '<=', end)])
    groups = frame.groupby('instrument_id', sort=False).indices
    out.mkdir()
    write_json(out/'source.json', dict(created_at=utc_now(), parent=str(parent), training_root=str(root), old_inputs=str(oldroot),
        end_date=end, sealed_holdout_start=protocol['sealed_holdout_start'], source_sha256=file_hash(Path(__file__)),
        old_manifest_sha256=file_hash(oldroot/'manifest.json'), parent_frozen_sha256=file_hash(parent/'frozen-manifest.json')))
    values = np.lib.format.open_memmap(out/'values.npy', mode='w+', dtype=np.float32, shape=(len(panel['instruments']), len(dates), len(fields)))
    values[:] = np.nan
    mask_values = np.load(root/'panel-h5/values.npy', mmap_mode='r')
    old_values = np.load(oldroot/'values.npy', mmap_mode='r')
    mask_index = panel['feature_names'].index('unpriced_suspension')
    count = 0
    for index, symbol in enumerate(panel['instruments']):
        raw = frame.iloc[groups.get(symbol, [])].set_index('date').reindex(dates)
        carry = (mask_values[index, :len(dates), mask_index] == 1) & raw.close.isna()
        sequence = raw.sequence_id.ffill().fillna('')
        episodes = sequence.ne(sequence.shift()).cumsum()
        close, factor = raw.close.groupby(episodes).ffill(), raw.factor.groupby(episodes).ffill()
        for field in fields[:4]:
            raw.loc[carry, field] = close.loc[carry]
        raw.loc[carry, 'factor'] = factor.loc[carry]
        raw.loc[carry, ['volume', 'amount']] = 0.
        values[index] = raw[fields].to_numpy(np.float32)
        np.testing.assert_array_equal(values[index, :len(old['dates'])], old_values[index])
        count += old_values[index].size
        if index % 500 == 0:
            write_json(out/'progress.json', dict(status='extending_inputs', completed=index, total=len(panel['instruments']), updated_at=utc_now()))
    values.flush()
    manifest = {**old, 'dates': dates, 'values_sha256': file_hash(out/'values.npy'),
        'fold': None, 'fold_index': None, 'scope': 'exact required plan input horizon; no labels loaded from sealed period',
        'end_date': end, 'old_prefix_values_verified': count, 'completed': True}
    write_json(out/'manifest.json', manifest)
    write_json(out/'progress.json', dict(status='completed', end_date=end, old_prefix_values_verified=count, updated_at=utc_now()))
    print('Prepared bounded extension', end, 'unchanged historical values', count, flush=True)


if __name__ == '__main__':
    main()
