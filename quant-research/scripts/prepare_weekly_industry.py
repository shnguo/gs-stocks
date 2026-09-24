"""Register past-only weekly source reuse without changing model dates."""
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from quant_research.industry_context import read_snapshot
from quant_research.storage import file_hash, utc_now, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['old-root', 'output', 'source-doc', 'protocol']:
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    old, out = a.old_root.resolve(), a.output.resolve()
    doc = a.source_doc.read_text()
    if '每周一' not in doc:
        raise ValueError('Official source cadence not confirmed')
    dates = json.loads((old/'training-dates.json').read_text())
    anchors, mapping = {}, {}
    for day in sorted(dates):
        key = pd.Timestamp(day).isocalendar()[:2]
        anchors.setdefault(key, day)
        mapping[day] = anchors[key]
    available = {}
    for batch in sorted(old.glob('batch-v*')):
        if batch.is_dir():
            for path in sorted(batch.glob('*/manifest.json')):
                meta = json.loads(path.read_text())
                day = path.parent.name
                if meta['status'] == 'completed' and day in mapping:
                    read_snapshot(path.parent, day)
                    available[day] = str(path.parent)
    comparisons = []
    for day, source in mapping.items():
        if day != source and day in available and source in available:
            cols = ['instrument_id', 'industry', 'industryClassification', 'updateDate']
            left = read_snapshot(available[day], day)[cols].sort_values('instrument_id').reset_index(drop=True)
            right = read_snapshot(available[source], source)[cols].sort_values('instrument_id').reset_index(drop=True)
            pd.testing.assert_frame_equal(left, right)
            comparisons.append(dict(date=day, source_date=source, equal=True))
    out.mkdir()
    shutil.copy2(a.source_doc, out/'official-source-doc.md')
    write_json(out/'source-doc-receipt.json', dict(url='https://www.baostock.com/helpdocs/api/markdown/stockIndustry.md',
        method='POST', body={}, captured_at=utc_now(), sha256=file_hash(out/'official-source-doc.md')))
    write_json(out/'target-source-map.json', mapping)
    write_json(out/'training-dates.json', sorted(set(mapping.values())))
    write_json(out/'seed-index.json', {d: available[d] for d in sorted(set(mapping.values())) if d in available})
    write_json(out/'reuse-evidence.json', dict(target_dates=len(dates), source_dates=len(set(mapping.values())),
        observed_same_week_comparisons=comparisons, created_at=utc_now(),
        meaning='Observed equality supports source reuse, not certified original publication time'))
    protocol = json.loads(a.protocol.read_text())
    protocol.update(protocol_id='industry-ablation-v2', snapshot_cadence='weekly-earliest-required-day',
        max_snapshot_age_days=6, source_frequency='Official weekly Monday update; retain earliest required date within ISO week',
        target_dates_preserved=len(dates), source_dates=len(set(mapping.values())))
    write_json(out/'protocol.json', protocol)
    print('Registered', len(dates), 'model dates;', len(set(mapping.values())), 'source dates;',
        len(json.loads((out/'seed-index.json').read_text())), 'verified seeds;', len(comparisons), 'equal pairs')


if __name__ == '__main__':
    main()
