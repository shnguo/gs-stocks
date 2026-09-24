"""Dated retrospective industry snapshots, with missing membership retained.

Provider update dates are not certified publication times. These features are
research-only and never establish point-in-time production readiness.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.storage import file_hash

INDUSTRY_FEATURES = ('industry_known', 'industry_peer_count',
    'industry_mean_return_1', 'industry_mean_return_5', 'industry_mean_return_20',
    'industry_mean_volatility_20', 'relative_to_industry_return_1',
    'relative_to_industry_return_5', 'relative_to_industry_return_20',
    'relative_to_industry_volatility_20')


def read_snapshot(directory, requested_date):
    directory = Path(directory).resolve()
    meta = json.loads((directory/'manifest.json').read_text())
    if (meta['status'] != 'completed' or meta['provider'] != 'baostock'
            or meta['query'] != {'api': 'stock_industry', 'code': '', 'date': requested_date}):
        raise ValueError('Incomplete or incorrectly scoped industry snapshot')
    for spec in meta['raw_files']:
        p = (directory/spec['file']).resolve()
        if not p.is_relative_to(directory) or file_hash(p) != spec['sha256']:
            raise ValueError('Industry source evidence changed')
    if file_hash(directory/'records.json') != meta['records_sha256']:
        raise ValueError('Industry records changed')
    records = json.loads((directory/'records.json').read_text())
    if not records['pagination_complete']:
        raise ValueError('Incomplete industry pagination')
    frame = pd.DataFrame(records['rows'], columns=records['fields'])
    if frame.code.duplicated().any() or (frame.updateDate > requested_date).any():
        raise ValueError('Duplicate or future industry assignment')
    # B-shares and index identities are outside this A-share research panel.
    keep = frame.code.str.match(r'^(sh\.6\d{5}|sz\.[03]\d{5})$')
    frame = frame.loc[keep].copy()
    frame['instrument_id'] = frame.code.str[:2].map({'sh': 'cn.xshg.', 'sz': 'cn.xshe.'})+frame.code.str[3:]
    frame['requested_date'] = requested_date
    frame['fetched_at'] = meta['fetched_at']
    frame['published_at'] = None
    frame['information_vintage'] = 'retrospective_reconstructed'
    return frame


def industry_context(x, rows, feature_names, snapshots, min_peers=5, max_snapshot_age_days=0):
    if x.shape != (len(rows), len(feature_names)) or not np.isfinite(x).all():
        raise ValueError('Invalid stock features')
    if rows.duplicated(['date', 'instrument_id']).any() or min_peers < 2:
        raise ValueError('Invalid cohort or peer threshold')
    if not isinstance(max_snapshot_age_days, int) or not 0 <= max_snapshot_age_days <= 6:
        raise ValueError('Invalid explicitly registered source lag')
    source = ['return_1', 'return_5', 'return_20', 'volatility_20']
    positions = [feature_names.index(k) for k in source]
    out = np.full((len(rows), len(INDUSTRY_FEATURES)), np.nan, dtype=np.float32)
    out[:, :2] = 0
    for date in sorted(rows.date.unique()):
        if date not in snapshots:
            raise ValueError(f'Missing requested historical snapshot: {date}')
        snapshot = snapshots[date]
        age = (pd.Timestamp(date)-pd.to_datetime(snapshot.requested_date)).dt.days
        if ((age < 0).any() or (age > max_snapshot_age_days).any()
                or snapshot.requested_date.nunique() > 1
                or (snapshot.updateDate > snapshot.requested_date).any()
                or snapshot.instrument_id.duplicated().any()):
            raise ValueError('Industry source is not the requested as-of snapshot')
        ids = np.flatnonzero(rows.date.to_numpy() == date)
        members = snapshot.set_index('instrument_id').industry.reindex(rows.instrument_id.iloc[ids])
        known = members.notna() & members.ne('')
        groups = members.to_numpy()
        values = x[ids][:, positions].astype(float)
        for industry in sorted(set(members.loc[known])):
            peers = np.flatnonzero(groups == industry)
            target = ids[peers]
            out[target, 1] = len(peers)
            if len(peers) < min_peers:
                continue
            mean = values[peers].mean(axis=0)
            out[target, 0] = 1
            out[target, 2:6] = mean
            out[target, 6:] = values[peers]-mean
    return out
