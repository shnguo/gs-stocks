import numpy as np
import pandas as pd
import pytest

from quant_research.industry_context import industry_context


def test_industry_uses_requested_snapshot_and_keeps_bse_missing():
    names = ['return_1', 'return_5', 'return_20', 'volatility_20']
    x = np.array([[.01, .02, .03, .04], [.03, .04, .05, .06], [.9, .8, .7, .6]])
    rows = pd.DataFrame(dict(date=['2021-06-30']*3, instrument_id=['a.xshg', 'b.xshe', 'c.xbse']))
    s = pd.DataFrame(dict(instrument_id=['a.xshg', 'b.xshe'], industry=['bank', 'bank'],
                          updateDate=['2021-06-14']*2, requested_date=['2021-06-30']*2))
    result = industry_context(x, rows, names, {'2021-06-30': s}, min_peers=2)
    assert len(result) == len(rows)
    np.testing.assert_equal(result[:, 0], [1, 1, 0])
    assert result[0, 2] == pytest.approx(.02)
    assert np.isnan(result[2, 2:]).all()
    changed = s.copy()
    changed.updateDate = '2024-06-24'
    with pytest.raises(ValueError, match='as-of'):
        industry_context(x, rows, names, {'2021-06-30': changed}, min_peers=2)
    with pytest.raises(ValueError, match='Missing'):
        industry_context(x, rows, names, {}, min_peers=2)


def test_source_codes_join_canonical_panel_identifiers(tmp_path):
    from quant_research.industry_context import read_snapshot
    from quant_research.storage import file_hash, write_json
    records = dict(fields=['updateDate', 'code', 'code_name', 'industry', 'industryClassification'],
        rows=[['2021-06-14', 'sh.600000', 'Pudong', 'bank', 'source'],
              ['2021-06-14', 'sz.000001', 'PingAn', 'bank', 'source'],
              ['2021-06-14', 'sh.900901', 'B share', 'bank', 'source']],
        pagination_complete=True)
    write_json(tmp_path/'records.json', records)
    write_json(tmp_path/'manifest.json', dict(status='completed', provider='baostock',
        query=dict(api='stock_industry', code='', date='2021-06-30'),
        raw_files=[], records_sha256=file_hash(tmp_path/'records.json'), fetched_at='2026-09-14'))
    snapshot = read_snapshot(tmp_path, '2021-06-30')
    assert snapshot.instrument_id.tolist() == ['cn.xshg.600000', 'cn.xshe.000001']
    rows = pd.DataFrame(dict(date=['2021-06-30']*3,
        instrument_id=['cn.xshg.600000', 'cn.xshe.000001', 'cn.xbse.830001']))
    result = industry_context(np.ones((3, 4)), rows,
        ['return_1', 'return_5', 'return_20', 'volatility_20'], {'2021-06-30': snapshot}, min_peers=2)
    np.testing.assert_equal(result[:, 0], [1, 1, 0])


def test_weekly_source_never_uses_future_or_excessively_old_membership():
    names = ['return_1', 'return_5', 'return_20', 'volatility_20']
    rows = pd.DataFrame(dict(date=['2021-06-18']*2, instrument_id=['a', 'b']))
    snapshot = pd.DataFrame(dict(instrument_id=['a', 'b'], industry=['bank']*2,
        updateDate=['2021-06-14']*2, requested_date=['2021-06-14']*2))
    features = np.array([[1., 2., 3., 4.], [3., 4., 5., 6.]])
    with pytest.raises(ValueError, match='as-of'):
        industry_context(features, rows, names, {'2021-06-18': snapshot}, min_peers=2)
    result = industry_context(features, rows, names, {'2021-06-18': snapshot}, min_peers=2, max_snapshot_age_days=6)
    assert result[0, 2] == 2
    assert snapshot.requested_date.eq('2021-06-14').all()
    for source_date in ['2021-06-19', '2021-06-10']:
        changed = snapshot.copy()
        changed.requested_date = source_date
        changed.updateDate = source_date
        with pytest.raises(ValueError, match='as-of'):
            industry_context(features, rows, names, {'2021-06-18': changed}, min_peers=2, max_snapshot_age_days=6)
