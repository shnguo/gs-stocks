"""Train-only dictionaries for retrospective industry and identifier categories.

Code families are explicitly not certified historical exchange/board membership.
Unknown categories remain missing (-1); no current security-master backfill.
"""
import numpy as np
import pandas as pd

CATEGORY_FEATURES = ('industry_category', 'identity_exchange', 'identity_code_family',
                     'stock_identity')


def identity_categories(instrument_id):
    parts = instrument_id.split('.')
    if len(parts) != 3 or parts[0] != 'cn' or len(parts[2]) != 6 or not parts[2].isdigit():
        raise ValueError('Invalid canonical instrument identity')
    exchange, code = parts[1:]
    if exchange == 'xshg' and code.startswith('6'):
        family = 'sh_68' if code.startswith(('688', '689')) else 'sh_other_a'
    elif exchange == 'xshe' and code.startswith(('0', '3')):
        family = 'sz_30' if code.startswith(('300', '301')) else 'sz_other_a'
    elif exchange == 'xbse':
        # Preserve the canonical source identity; do not assign a historical
        # BSE listing status to its predecessor-market observations.
        family = 'bj_identity'
    else:
        raise ValueError('Unsupported canonical instrument identity')
    return exchange, family


def raw_categories(rows, snapshots, max_age_days=6):
    if rows.duplicated(['date', 'instrument_id']).any():
        raise ValueError('Duplicate stock-date')
    result = pd.DataFrame(index=range(len(rows)), columns=CATEGORY_FEATURES, dtype=object)
    identities = [identity_categories(v) for v in rows.instrument_id]
    result['identity_exchange'] = [v[0] for v in identities]
    result['identity_code_family'] = [v[1] for v in identities]
    result['stock_identity'] = rows.instrument_id.to_numpy()
    for day, positions in rows.groupby('date', sort=True).indices.items():
        frame = snapshots[day]
        age = (pd.Timestamp(day)-pd.to_datetime(frame.requested_date)).dt.days
        if (frame.instrument_id.duplicated().any() or frame.requested_date.nunique() != 1
                or not age.between(0, max_age_days).all()
                or (frame.updateDate > frame.requested_date).any()):
            raise ValueError('Invalid or future industry source')
        known = frame.industry.notna() & frame.industry.ne('')
        known &= frame.industryClassification.notna() & frame.industryClassification.ne('')
        # Tuple encoding avoids collisions between taxonomy and industry text.
        import json
        mapping = {r.instrument_id: json.dumps([r.industryClassification, r.industry],
                                              ensure_ascii=False)
                   for r in frame.loc[known].itertuples()}
        result.loc[positions, 'industry_category'] = [
            mapping.get(v) for v in rows.instrument_id.iloc[positions]]
    return result


def fit_dictionary(train):
    return {name: sorted(set(train[name].dropna())) for name in train.columns}


def encode_categories(frame, dictionary):
    if list(frame.columns) != list(dictionary):
        raise ValueError('Category schema changed')
    columns = []
    for name, values in dictionary.items():
        if values != sorted(set(values)) or len(values) >= 2**24:
            raise ValueError('Invalid category dictionary')
        mapping = {v: i for i, v in enumerate(values)}
        columns.append(frame[name].map(mapping).fillna(-1).to_numpy(dtype=np.float32))
    return np.column_stack(columns)
