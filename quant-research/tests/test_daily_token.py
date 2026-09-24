import numpy as np
import pandas as pd
import pytest
from daily_token_cycle import select_signal
from publish_daily_token_view import publish, rank_reference_returns, timing_summary
from review_daily_token import review

from quant_research.daily_loop import read, write_manifest
from quant_research.daily_token import (
    LOW_HIGH,
    OPEN_CLOSE,
    future_known,
    path_statistics,
    publish_state,
)
from quant_research.storage import write_json


def six_days():
    return pd.DataFrame(dict(open=[10.]*6, high=[11.]*6, low=[9.]*6, close=[10.5]*6,
        volume=[100.]*6, amount=[1000.]*6, factor=[1.]*6,
        sequence_id=['a']*6, label_sequence_id=['b']*6, source_trade_status=[1]*6))


def test_labels_require_maturity_and_unchanged_adjustment_identity():
    x = six_days()
    assert future_known(x, 0, 5)
    assert not future_known(x.iloc[:5], 0, 5)
    for column, value in [('factor', 2.), ('label_sequence_id', 'changed'), ('source_trade_status', 0), ('volume', 0)]:
        other = x.copy()
        other.loc[3, column] = value
        assert not future_known(other, 0, 5)


def test_after_close_calendar_uses_trading_dates_and_shanghai_time():
    cal = pd.DataFrame(dict(date=['2026-09-24', '2026-09-25', '2026-09-28'], is_open=[True, False, True]))
    assert select_signal(cal, '2026-09-28T08:00:00Z', '16:30') == '2026-09-24'
    assert select_signal(cal, '2026-09-28T08:30:00Z', '16:30') == '2026-09-28'
    with pytest.raises(ValueError, match='not completed'):
        select_signal(cal, '2026-09-28T06:59:00Z', '16:30', '2026-09-28')
    with pytest.raises(ValueError, match='Timezone'):
        select_signal(cal, '2026-09-28', '16:30')


def test_path_rank_uses_future_open_and_terminal_close_not_best_possible_exit():
    paths = np.ones((64, 5, 6))*10
    paths[:, :, 1] = 100  # a high excursion is not treated as a realizable exit
    paths[:, :, 2] = 9
    paths[:, :, 3] = 11
    stats = path_statistics(paths, .0025)
    for value in stats:
        assert value['expected_net_return'] == pytest.approx(.0975)
        assert value['positive_fraction'] == 1
    paths[:49, 4, 2] = 12  # illegal OHLC; no repaired prices
    assert path_statistics(paths, .0025) is None


def test_ideal_timing_pairs_paths_excludes_same_day_high_and_keeps_losses():
    paths = np.broadcast_to([10., 12., 1., 10., 100., 1000.], (64, 5, 6)).copy()
    paths[:, 0, :4] = [30., 1000., 10., 30.]
    paths[32:, 0, 2] = 20.
    paths[32:, 1:, 1] = 21.
    result = path_statistics(paths, .0025, LOW_HIGH)
    assert result[1]['expected_net_return'] == pytest.approx((.2+.05)/2-.0025)
    assert result[1]['entry_median'] == 15.
    assert result[1]['exit_median'] == 16.5
    paths[:, 1, 1] = 11.
    assert path_statistics(paths, .0025, LOW_HIGH)[0]['expected_net_return'] == pytest.approx((.1-.45)/2-.0025)
    assert path_statistics(paths, .0025, LOW_HIGH)[1]['expected_net_return'] == pytest.approx((.2+.05)/2-.0025)
    paths[:, 0, :4] = [40., 1000., 40., 40.]
    assert path_statistics(paths, .0025, LOW_HIGH)[1]['expected_net_return'] < 0
    with pytest.raises(ValueError, match='Unknown ranking'):
        path_statistics(paths, .0025, 'unknown')


def test_pointer_recovers_after_completed_run_without_regressing_later_day(tmp_path):
    run = tmp_path / 'runs/2026-09-15'
    write_json(run / 'run.json', dict(signal_date='2026-09-15', trained_signal_through='2026-09-08',
        checkpoints={}, published_at='2026-09-15T09:00:00Z'))
    write_json(run / 'binding.json', dict(source='/observed/source'))
    write_manifest(run)
    publish_state(tmp_path, run)
    assert read(tmp_path / 'current.json')['signal_date'] == '2026-09-15'
    write_json(tmp_path / 'current.json', dict(signal_date='2026-09-16', sentinel=True))
    publish_state(tmp_path, run)
    assert read(tmp_path / 'current.json')['sentinel']


@pytest.mark.parametrize('method', [OPEN_CLOSE, LOW_HIGH])
def test_forward_scorecard_and_daily_export_preserve_run(tmp_path, method, monkeypatch):
    # Publication timing is part of the fixture, not the machine's wall clock.
    monkeypatch.setattr('publish_daily_token_view.utc_now', lambda: '2026-09-16T09:00:00Z')
    source, store = tmp_path / 'source', tmp_path / 'store'
    dates = ['2026-09-16', '2026-09-17', '2026-09-18', '2026-09-21', '2026-09-22', '2026-09-23']
    symbols = ['cn.xshg.600001', 'cn.xshe.000001']
    snapshot = source / 'snapshot'
    snapshot.mkdir(parents=True)
    bars = pd.concat([six_days().assign(instrument_id=s, date=dates) for s in symbols], ignore_index=True)
    bars.to_parquet(snapshot / 'bars.parquet', index=False)
    pd.DataFrame(dict(instrument_id=symbols, name=['A', 'B'], current_member=True)).to_parquet(snapshot / 'instruments.parquet', index=False)
    write_manifest(snapshot)
    write_json(source / 'panel-h5/manifest.json', dict(price_data_through=dates[-1], dates=dates))
    run = store / 'runs' / dates[0]
    write_json(run / 'run.json', dict(signal_date=dates[0], horizon_dates=dates[1:], prospective=True,
        ranking_method=method, published_at='2026-09-16T09:00:00Z', mode='post_close_watchlist', forecast_inputs=2,
        top_n=1, cost_scenario=.0025, checkpoints={str(s): {} for s in [17, 29, 43]}))
    write_json(run / 'binding.json', dict(config={'seeds': [17,29,43], 'round_trip_cost_scenario': .0025}))
    (run / 'inputs').mkdir()
    pd.DataFrame(dict(instrument_id=symbols, name=['A', 'B'])).to_parquet(run / 'inputs/forecast-rows.parquet', index=False)
    forecast = pd.DataFrame([dict(instrument_id=s, name=name, horizon=h, expected_net_return=.03-i*.05,
        positive_fraction=.6, rank=i+1) for i, (s, name) in enumerate(zip(symbols, ['A', 'B'])) for h in [2, 5]])
    forecast.to_parquet(run / 'summary.parquet', index=False)
    forecast[forecast.horizon == 5].to_csv(run / 'ranking.csv', index=False)
    quantiles = []
    for seed in [17, 29, 43]:
        folder = run / f'forecasts/seed{seed}'
        folder.mkdir(parents=True)
        sample = np.broadcast_to(six_days()[['open', 'high', 'low', 'close', 'volume', 'amount']].to_numpy()[1:], (2, 64, 5, 6)).copy()
        np.save(folder / 'paths.npy', sample)
        for s in symbols:
            for day in range(5):
                for field, value in zip(['open', 'high', 'low', 'close', 'volume', 'amount'], [10, 11, 9, 10.5, 100, 1000]):
                    quantiles.append(dict(instrument_id=s, seed=seed, day=day, field=field, q10=value, q50=value, q90=value))
    pd.DataFrame(quantiles).to_parquet(run / 'daily-quantiles.parquet', index=False)
    (run / 'report.md').write_text('# Watchlist\n')
    write_manifest(run)
    original = (run / 'manifest.json').read_bytes()
    scores = review(store, source)
    assert len(scores) == 2 and all(s['known'] == 2 for s in scores)
    assert all(s['endpoint_mae'] == pytest.approx(0) for s in scores)
    assert all(s['brier'] == pytest.approx(.16) for s in scores)
    expected_return = 11/9-1-.0025 if method == LOW_HIGH else .0475
    assert all(s['top_mean_net_scenario'] == pytest.approx(expected_return) for s in scores)
    assert all(s['ranking_method'] == method for s in scores)
    assert len(review(store, source)) == 2  # persisted review is reused
    report = publish(run)
    assert len(pd.read_csv(report.parent / 'daily-ohlcva.csv')) == 60
    assert '2026-09-23' in report.read_text()
    assert (run / 'manifest.json').read_bytes() == original
    assert publish(run) == report
    updated_scores = review(store, source)
    assert all(s['ranking_score_method'] == 'reference_price_ratio' for s in updated_scores)
    if method == OPEN_CLOSE:
        assert updated_scores[0]['return_known'] == 0
        assert updated_scores[0]['top_mean_net_scenario'] is None
        assert updated_scores[1]['top_mean_net_scenario'] == pytest.approx(.0475)
    else:
        assert all(s['top_mean_net_scenario'] == pytest.approx(11/9-1-.0025) for s in updated_scores)
    if method == OPEN_CLOSE:
        revised = publish(run, LOW_HIGH)
        assert revised != report
        assert '卖出参考价 ÷ 买入参考价' in revised.read_text()
        assert pd.read_csv(revised.parent / 'ranking.csv').expected_net_return.iloc[0] == pytest.approx(11/9-1-.0025)
        assert (run / 'manifest.json').read_bytes() == original
        assert publish(run, LOW_HIGH) == revised
    write_json(store / 'latest.json', dict(signal_date='2026-09-24', sentinel=True))
    assert publish(run) == report
    assert read(store / 'latest.json')['sentinel']


def test_sell_date_uses_equal_model_frequency_and_date_specific_price(tmp_path):
    (tmp_path / 'inputs').mkdir()
    pd.DataFrame({'instrument_id': ['A']}).to_parquet(tmp_path / 'inputs/forecast-rows.parquet', index=False)
    dates = ['2026-09-16', '2026-09-17', '2026-09-18', '2026-09-21', '2026-09-22']
    info = dict(checkpoints={str(s): {} for s in [17, 29, 43]}, horizon_dates=dates, cost_scenario=.0025)
    ranks = pd.DataFrame({'instrument_id': ['A']})
    for seed in [17, 29, 43]:
        dest = tmp_path / f'forecasts/seed{seed}'
        dest.mkdir(parents=True)
        x = np.broadcast_to([10., 11., 9., 10., 100., 1000.], (1, 64, 5, 6)).copy()
        x[:, :, 0, 1] = 1000.
        x[:, :, 1 if seed == 17 else 2, 1] = 14.
        if seed != 17:
            x[:, 16:, :, 1] = 0.  # 16 valid paths vs 64 in seed 17: models still have equal weight.
        np.save(dest / 'paths.npy', x)
    result = timing_summary(tmp_path, ranks, info, LOW_HIGH).iloc[0]
    assert result.buy_date == dates[0]
    assert result.sell_reference_date == dates[2]
    assert result.sell_date_frequency == pytest.approx(2/3)
    assert result.sell_reference_price == pytest.approx(13.)
    assert result.selected_day_expected_net_return == pytest.approx(13/9-1-.0025)
    assert sum(result[f'sell_t{d}_frequency'] for d in range(1,5)) == pytest.approx(1.)
    path = tmp_path / 'forecasts/seed43/paths.npy'
    x = np.load(path)
    x[:, :16, 2, 1] = 11.
    x[:, :16, 3, 1] = 14.
    np.save(path, x)
    result = timing_summary(tmp_path, ranks, info, LOW_HIGH).iloc[0]
    assert result.sell_reference_date == dates[1]  # Equal frequencies choose the earliest date.


def test_reference_return_controls_rank_instead_of_extreme_path_mean():
    rows = pd.DataFrame(dict(instrument_id=['A', 'B'], expected_net_return=[.9, .1], positive_fraction=[.9, .7],
        rank=[1,2], reference_price_net_return=[.02,.08], selected_day_positive_fraction=[.6,.8]))
    result = rank_reference_returns(rows)
    assert result.instrument_id.tolist() == ['B', 'A']
    assert result.expected_net_return.tolist() == [.08,.02]
    assert result['rank'].tolist() == [1,2]
    assert result.path_mean_ideal_net_return.tolist() == [.1,.9]
