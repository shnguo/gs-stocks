from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest
from test_research_review import market, protocol

from quant_research.daily_loop import read, write_manifest
from quant_research.research_analysis import (
    aggregate_reports,
    build_analysis_review,
    data_freshness,
    price_state,
    summarize,
    trend,
)
from quant_research.research_review import freeze_decision, refresh_research
from quant_research.storage import file_hash, write_json


def frozen_market(tmp_path):
    snapshot, store, run = market(tmp_path)
    report = store / 'reports' / run.name
    rows = []
    for symbol in snapshot.tables['instruments'].instrument_id:
        for day, date in enumerate(snapshot.dates[-5:]):
            bar = snapshot.tables['bars'].query('instrument_id == @symbol and date == @date').iloc[0]
            for field in ('open', 'high', 'low', 'close'):
                rows.append(dict(instrument_id=symbol, day=day, date=date, field=field,
                                 q10=bar[field] * .9, q50=bar[field] * 1.05, q90=bar[field] * 1.1))
    pd.DataFrame(rows).to_csv(report / 'daily-ohlcva.csv', index=False)
    write_manifest(report)
    write_json(store / 'reference-publications' / (run.name + '.json'),
               {'report': str(report), 'manifest_sha256': file_hash(report / 'manifest.json')})
    return snapshot, store, freeze_decision(run, protocol())


def test_price_metrics_use_each_forecast_field_and_preserve_frozen_inputs(tmp_path):
    snapshot, _, bundle = frozen_market(tmp_path)
    before = file_hash(bundle / 'manifest.json')
    result = build_analysis_review(bundle, snapshot, now='2024-01-01T00:00:00Z')
    analysis = read(result / 'analysis.json')
    arm = analysis['arms'][0]
    assert len(arm['candidates']) == 6
    assert arm['quality']['known'] == arm['quality']['expected'] == 30
    assert arm['quality']['mean_absolute_percentage_error'] == pytest.approx(.05)
    assert arm['quality']['mean_bias'] == pytest.approx(.05)
    assert all(row['known'] == 30 and row['band_hit_rate'] == 1 for row in arm['fields'])
    assert all(row['known'] == 6 for row in arm['horizons'])
    assert arm['candidates'][0]['trend']['status'] == 'comparable'
    assert file_hash(bundle / 'manifest.json') == before
    assert build_analysis_review(bundle, snapshot, now='2024-01-01T00:00:00Z') == result
    assert pd.read_parquet(result / 'price-comparisons.parquet').shape[0] == 120


def test_current_session_prices_cannot_be_used_before_close(tmp_path):
    snapshot, _, bundle = frozen_market(tmp_path)
    first = snapshot.dates[-5]
    before = read(build_analysis_review(bundle, snapshot, now=first+'T06:59:59Z') / 'analysis.json')['arms'][0]
    after = read(build_analysis_review(bundle, snapshot, now=first+'T07:00:00Z') / 'analysis.json')['arms'][0]
    assert before['quality']['pending'] == 30 and before['quality']['known'] == 0
    assert before['quality']['mean_absolute_percentage_error'] is None
    assert all(p['actual'] is None for c in before['candidates'] for p in c['path'])
    assert after['quality']['known'] == 6 and after['quality']['pending'] == 24


def test_missing_observations_and_missing_predictions_are_distinct(tmp_path):
    snapshot, _, bundle = frozen_market(tmp_path)
    first, last = snapshot.dates[-5], snapshot.dates[-1]
    snapshot.tables['bars'] = snapshot.tables['bars'].loc[lambda frame: ~frame.date.eq(first)]
    forecast = pd.read_csv(bundle / 'daily-predictions.csv')
    forecast = forecast.loc[~(forecast.date.eq(last) & forecast.field.eq('close'))]
    forecast.to_csv(bundle / 'daily-predictions.csv', index=False)
    write_manifest(bundle)
    arm = read(build_analysis_review(bundle, snapshot, now='2024-01-01T00:00:00Z') / 'analysis.json')['arms'][0]
    assert arm['quality']['known'] == 18 and arm['quality']['missing'] == 12
    path = arm['candidates'][0]['path']
    assert path[0]['status'] == 'price_missing' and path[0]['actual'] is None
    assert path[-1]['status'] == 'prediction_missing' and path[-1]['actual'] is not None


@pytest.mark.parametrize('key,value', [('factor', 2), ('sequence_id', 'relisted'), ('label_sequence_id', None)])
def test_price_boundaries_are_excluded(key, value):
    anchor = dict(open=10, high=11, low=9, close=10, factor=1, volume=100,
                  sequence_id='old', label_sequence_id='old')
    assert price_state(anchor, anchor) == 'comparable'
    changed = {**anchor, key: value}
    assert price_state(anchor, changed) == 'boundary_unknown'


def test_unknown_trade_status_is_not_inactive_and_explicit_halt_is_excluded():
    row = dict(open=10, high=11, low=9, close=10, factor=1, volume=100, source_trade_status=float('nan'))
    assert price_state(row, row) == 'comparable'
    assert price_state(row, {**row, 'source_trade_status': 0}) == 'inactive_session'
    assert price_state(row, {**row, 'volume': 0}) == 'inactive_session'
    assert price_state(row, {**row, 'close': None}) == 'price_missing'
    assert price_state(row, {**row, 'volume': None}) == 'price_missing'
    assert price_state(None, row) == 'anchor_missing'


@pytest.mark.parametrize('mutation', ['duplicate', 'wrong_day', 'reversed_quantiles'])
def test_invalid_frozen_predictions_are_rejected(tmp_path, mutation):
    snapshot, _, bundle = frozen_market(tmp_path)
    path = bundle / 'daily-predictions.csv'
    rows = pd.read_csv(path)
    if mutation == 'duplicate':
        rows = pd.concat([rows, rows.iloc[:1]])
    elif mutation == 'wrong_day':
        rows.loc[0, 'day'] = 4
    else:
        rows.loc[0, 'q10'] = rows.loc[0, 'q90'] * 2
    rows.to_csv(path, index=False)
    write_manifest(bundle)
    with pytest.raises(ValueError):
        build_analysis_review(bundle, snapshot, now='2024-01-01T00:00:00Z')


def test_aggregate_is_record_weighted_and_separates_late_and_synthetic_reports():
    rows = [dict(field='close', status='comparable', predicted=12, actual=10, q10=9, q90=13, anchor=10)]
    first = summarize(rows)
    second = summarize([{**rows[0], 'predicted': 10} for _ in range(3)])
    reports = [dict(synthetic=False, arms=[dict(arm='control', publication={'prospective': False}, quality=q)]) for q in (first, second)]
    reports.append({**deepcopy(reports[0]), 'synthetic': True})
    forward = deepcopy(reports[0])
    forward['arms'][0]['publication']['prospective'] = True
    reports.append(forward)
    result = {(r['scope'], r['arm']): r for r in aggregate_reports(reports)}
    late = result['retrospective', 'control']
    assert late['signal_dates'] == 2 and late['known'] == 4
    assert late['mean_absolute_percentage_error'] == pytest.approx(.05)
    assert late['baseline_mean_absolute_percentage_error'] == 0
    assert late['direction_agreement'] == .75
    assert result['prospective', 'control']['known'] == 1
    assert summarize([])['mean_absolute_percentage_error'] is None


def test_history_trend_is_signal_bound_and_requires_consecutive_completed_dates():
    dates = pd.bdate_range('2024-01-01', periods=23).strftime('%Y-%m-%d').tolist()
    rows = pd.DataFrame([dict(date=date, open=i+10, high=i+11, low=i+9, close=i+10,
                             factor=1, volume=100) for i, date in enumerate(dates)])
    known = trend(rows, dates[20], dates[:21])
    assert known['return_20'] == 2
    assert known['return_5'] == pytest.approx(30/25-1)
    assert trend(rows.drop(index=10), dates[20], dates[:21])['return_20'] is None
    rows.loc[1, 'factor'] = 2
    assert trend(rows, dates[20], dates[:21])['return_20'] is None


def test_freshness_uses_last_completed_session_and_exchange_holidays():
    snapshot = SimpleNamespace(tables={
        'calendar': pd.DataFrame({'date': ['2026-09-24', '2026-09-25', '2026-09-28', '2026-09-29'], 'is_open': [True, False, True, True]}),
        'bars': pd.DataFrame({'date': ['2026-09-24', '2026-09-28'], 'close': [10, 11]})}, manifest={'declaration': {}})
    assert data_freshness(snapshot, '2026-09-29T06:00:00Z')['status'] == 'current'
    late = data_freshness(snapshot, '2026-09-29T07:00:00Z')
    assert late['missing_sessions'] == ['2026-09-29']
    assert late['expected_through'] == '2026-09-29'
    with pytest.raises(ValueError, match='timezone'):
        data_freshness(snapshot, '2026-09-29')


def test_analysis_refresh_does_not_construct_execution_or_simulate(tmp_path, monkeypatch):
    import quant_research.research_review as review
    snapshot, store, bundle = frozen_market(tmp_path)
    declaration = read(snapshot.path/'manifest.json')
    declaration.pop('schema_version')
    declaration['declaration']['purpose'] = 'training'
    declaration['declaration']['st_history_verified'] = False
    write_json(snapshot.path/'manifest.json', declaration)
    monkeypatch.setattr(review, 'build_execution_snapshot', lambda *a, **k: pytest.fail('Execution must not run'))
    monkeypatch.setattr(review, 'simulate', lambda *a, **k: pytest.fail('Simulation must not run'))
    first = refresh_research(store, snapshot.path.parent, protocol(), analysis_only=True)
    pointer = read(store/'decision-publications'/ (bundle.parent.name+'.json'))
    second = refresh_research(store, snapshot.path.parent, protocol(), analysis_only=True)
    summary = read(second['summary'])
    assert summary['mode'] == 'analysis' and 'matching' not in summary
    assert summary['analysis_reports'][0]['arms'][0]['quality']['known'] == 30
    assert summary['prospective_mature_dates'] == 0
    assert first['evidence'] == second['evidence']
    assert read(store/'decision-publications'/ (bundle.parent.name+'.json')) == pointer
    assert not (store/'execution-evidence-publication.json').exists()
    assert not (store/'execution-snapshots').exists()
    with pytest.raises(ValueError, match='explicit execution'):
        refresh_research(store, snapshot.path.parent, protocol(), analysis_only=True, execution_evidence='unused')
