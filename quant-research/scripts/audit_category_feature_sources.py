"""Record current feature availability without acquiring data or reading sealed labels."""
import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from quant_research.storage import file_hash, utc_now, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prior', type=Path, required=True)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    prior, snapshot, out = a.prior.resolve(), a.snapshot.resolve(), a.output.resolve()
    out.mkdir()
    protocol = json.loads((prior/'protocol.json').read_text())
    source_hashes = {str(snapshot/f):file_hash(snapshot/f) for f in ['manifest.json', 'instruments.parquet']}
    schemas = {name:pq.read_schema(snapshot/name).names for name in ['bars.parquet', 'instruments.parquet']}
    instruments = pd.read_parquet(snapshot/'instruments.parquet').set_index('instrument_id')
    assert not instruments.index.duplicated().any()
    records = []
    for fold in protocol['fold_indices']:
        for part in ['train', 'selection', 'calibration', 'evaluation']:
            path = prior/f'fold-{fold:02d}'/f'{part}-rows.parquet'
            source_hashes[str(path)] = file_hash(path)
            rows = pd.read_parquet(path, columns=['date', 'instrument_id'])
            assert rows.date.max() < protocol['sealed_holdout_start']
            listed = pd.to_datetime(rows.instrument_id.map(instruments.listed_at), errors='coerce')
            day = pd.to_datetime(rows.date)
            records.append(dict(fold=fold, partition=part, rows=len(rows),
                missing_listing_date=int(listed.isna().sum()),
                listing_after_observation=int((listed > day).sum()),
                bse_canonical_rows_before_establishment=int((rows.instrument_id.str.startswith('cn.xbse.') & rows.date.lt('2021-11-15')).sum())))
    pd.DataFrame(records).to_csv(out/'listing-consistency.csv', index=False)
    write_json(out/'audit.json', dict(created_at=utc_now(), schemas=schemas, inputs=source_hashes,
        checks=records,
        available_inputs=['canonical stock identity', 'identifier exchange/code group', 'dated reconstructed industry snapshots'],
        absent_bar_features=['turnover_rate', 'float_shares', 'float_market_cap', 'total_market_cap'],
        listing_age_status='Listing dates exist in reconstructed current security master; consistency counts do not certify historical lineage. Do not enable until provider lineage and BSE predecessor handling verified.',
        collection_status='Existing Rust daily request excludes turn and valuation fields; existing raw captures cannot restore fields that were not requested.',
        next_data_candidate='Dedicated new Rust capture schema for historical turnover, preserving original daily schema. Validate SH/SZ samples, suspension blanks, units and source scope before scaling; BSE missingness explicit.',
        float_cap_status='Do not use current share counts historically. Any estimate from volume and rounded turnover must be separately named and carry denominator/rounding uncertainty; it is not directly observed float market cap.',
        executable=False))
    print(pd.DataFrame(records).groupby('partition').sum(numeric_only=True).to_string())


if __name__ == '__main__':
    main()
