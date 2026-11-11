from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from us_earnings.backtest import cluster_interval, event_returns, metrics, simulate
from us_earnings.signals import build_signals, guidance_label, price_panel
from us_earnings.text_model import fit_text_model


def fixtures(n=85):
    days = pd.bdate_range("2023-09-01", periods=n)
    rows = []
    for ticker, drift in [("SPY", 0.10), ("AAA", 0.20), ("BBB", 0.30)]:
        for i, day in enumerate(days):
            opening = 100 + i * drift
            rows.append(dict(ticker=ticker, date=day, open=opening, high=opening + 1,
                             low=opening - 1, close=opening + 0.1, volume=1000))
    config = dict(start=str(days[55].date()), end=str(days[-1].date()), benchmark="SPY",
                  ma_sessions=50, atr_sessions=20, holding_sessions=20,
                  stop_atr_multiple=2.0, cost_bps_per_side=10,
                  max_positions=2, position_fraction=0.5)
    panel, calendar = price_panel(pd.DataFrame(rows), config)
    return config, panel, calendar


def one_signal(days, ticker="AAA", index=56):
    return pd.DataFrame([dict(ticker=ticker, signal_date=days[index - 1], entry_date=days[index],
                              atr=2.0, rank=0.01, earnings=True, guidance=True, accession="example")])


@pytest.mark.parametrize("text", [
    "Company raises full-year 2024 earnings guidance.",
    "We are increasing our annual revenue outlook.",
    "Full-year guidance raised following strong results.",
    "Company increases 2024 adjusted EPS guidance.",
])
def test_source_bound_positive_guidance(text):
    result = guidance_label(text)
    assert result["raised_guidance"]
    assert result["evidence"] in text


@pytest.mark.parametrize("text", [
    "We raised the dividend and reaffirmed full-year guidance.",
    "We do not raise our full-year earnings guidance.",
    "We previously raised annual earnings guidance.",
    "Last quarter we raised annual earnings guidance.",
    "Revenue increased in 2024. Full-year guidance is unchanged.",
    "The full-year outlook is $5.00 to $5.50 per share.",
    "Non-GAAP EPS increased 16% year over year. Q3 FY 2024 guidance: revenue growth of 4%.",
    "Due to increasing uncertainty in our business outlook, we are not providing full-year 2020 guidance.",
])
def test_parser_abstains_instead_of_inventing_revisions(text):
    assert not guidance_label(text)["raised_guidance"]


def test_friday_filing_waits_for_monday_close_and_tuesday_open():
    config, panel, calendar = fixtures()
    friday = next(day for day in calendar[55:70] if day.dayofweek == 4)
    events = pd.DataFrame([dict(ticker="AAA", filingDate=str(friday.date()),
                                status="ok", raised_guidance=True, accessionNumber="event")])
    signals, audit = build_signals(panel, calendar, events, config)
    matched = signals.loc[signals.guidance].iloc[0]
    assert audit.signal_date.iloc[0].strftime("%Y-%m-%d") == (friday.to_pydatetime() + timedelta(days=3)).strftime("%Y-%m-%d")
    assert matched.entry_date.strftime("%Y-%m-%d") == (friday.to_pydatetime() + timedelta(days=4)).strftime("%Y-%m-%d")


def test_missing_exhibit_does_not_become_an_earnings_trade():
    config, panel, calendar = fixtures()
    events = pd.DataFrame([dict(ticker="AAA", filingDate=str(calendar[55].date()),
                                status="missing", raised_guidance=False, accessionNumber="event")])
    signals, _ = build_signals(panel, calendar, events, config)
    assert not signals.earnings.any()


def test_future_prices_do_not_change_past_signals():
    config, panel, calendar = fixtures()
    events = pd.DataFrame([dict(ticker="AAA", filingDate=str(calendar[55].date()),
                                status="ok", raised_guidance=True, accessionNumber="event")])
    original, _ = build_signals(panel, calendar, events, config)
    # Change a future close, then recompute all indicators from raw inputs.
    changed = pd.concat([f.assign(ticker=t).reset_index() for t, f in panel.items()])
    mask = (changed.ticker == "AAA") & (changed.date >= calendar[70])
    for col in ["open", "high", "low", "close"]:
        changed.loc[mask, col] *= 10
    new_panel, new_calendar = price_panel(changed, config)
    altered, _ = build_signals(new_panel, new_calendar, events, config)
    pd.testing.assert_frame_equal(original.loc[original.signal_date < calendar[70]].reset_index(drop=True),
                                  altered.loc[altered.signal_date < calendar[70]].reset_index(drop=True))


def test_same_day_information_is_rejected():
    config, panel, days = fixtures()
    signal = one_signal(days)
    signal["signal_date"] = signal.entry_date
    with pytest.raises(ValueError, match="same-day"):
        simulate(panel, days, signal, config)


def test_gap_stop_fills_at_worse_open_and_accounts_for_both_costs():
    config, panel, days = fixtures()
    config["position_fraction"] = 1.0
    signal = one_signal(days)
    entry = panel["AAA"].at[days[56], "open"]
    panel["AAA"].loc[days[57], ["open", "high", "low", "close"]] = [90, 91, 89, 90]
    curve, trades, _ = simulate(panel, days, signal, config)
    trade = trades.iloc[0]
    assert trade.exit_reason == "stop"
    assert trade.exit_price == pytest.approx(90 * 0.999)
    assert trade.net_return == pytest.approx((90 * 0.999) / (entry * 1.001) - 1)
    assert curve.nav.iloc[-1] == pytest.approx(1 + trade.pnl)


def test_entry_day_stop_and_exact_holding_horizon():
    config, panel, days = fixtures()
    _, trades, _ = simulate(panel, days, one_signal(days), config)
    assert trades.iloc[0].holding_sessions == 20
    assert trades.iloc[0].exit_date == days[75]
    panel["AAA"].at[days[56], "low"] = 1
    _, stopped, _ = simulate(panel, days, one_signal(days), config)
    assert stopped.iloc[0].holding_sessions == 1
    assert stopped.iloc[0].exit_reason == "stop"


def test_cash_and_position_limits_apply_to_simultaneous_candidates():
    config, panel, days = fixtures()
    config["max_positions"] = 1
    signals = pd.concat([one_signal(days), one_signal(days, "BBB")], ignore_index=True)
    curve, trades, skipped = simulate(panel, days, signals, config)
    assert curve.positions.max() == 1
    assert curve.cash.min() >= -1e-10
    assert len(trades) == 1
    assert skipped.reason.tolist() == ["position_limit"]


def test_drawdown_includes_initial_capital_and_empty_strategies():
    curve = pd.DataFrame(dict(nav=[0.9, 0.95], exposure=[1.0, 1.0]))
    assert metrics(curve)["max_drawdown"] == pytest.approx(-0.1)
    config, panel, days = fixtures()
    empty = one_signal(days).iloc[:0]
    flat, trades, _ = simulate(panel, days, empty, config)
    assert np.all(flat.nav == 1)
    assert len(trades) == 0


def test_event_comparison_has_identical_dates_and_drops_immature_events():
    config, panel, days = fixtures()
    signals = pd.concat([one_signal(days), one_signal(days, index=80)], ignore_index=True)
    result = event_returns(panel, days, signals, config)
    assert len(result) == 1
    assert result.exit_date.iloc[0] == days[75]
    assert result.excess_return.iloc[0] == pytest.approx(
        result.stock_return.iloc[0] - result.benchmark_return.iloc[0])


def test_price_validation_rejects_missing_sessions_and_duplicate_keys():
    config, panel, _ = fixtures()
    prices = pd.concat([f.assign(ticker=t).reset_index() for t, f in panel.items()])
    with pytest.raises(ValueError, match="Duplicate"):
        price_panel(pd.concat([prices, prices.iloc[:1]]), config)
    with pytest.raises(ValueError, match="missing or extra"):
        price_panel(prices.iloc[1:], config)


def test_text_model_never_fits_later_labels_or_vocabulary(tmp_path):
    rows = []
    for i in range(14):
        path = tmp_path / f"text-{i}.txt"
        raised = i % 2 == 0
        if i < 12:
            text = "We raise full-year earnings guidance." if raised else "We maintain annual guidance."
        else:
            text = "UniqueFutureVocabulary raises annual guidance."
        path.write_text(text)
        rows.append(dict(status="ok", text_file=path.name, filingDate="2022-01-01" if i < 12 else "2024-01-01",
                         raised_guidance=raised, accessionNumber=str(i), source_sha256=str(i)))
    events = pd.DataFrame(rows)
    settings = dict(training_end="2022-12-31", evaluation_start="2023-01-01", min_df=1,
                    max_features=100, C=1.0, seed=42, threshold=0.5)
    predicted, metadata, model = fit_text_model(events, tmp_path, settings)
    assert metadata["training_rows"] == 12
    assert len(predicted) == 2
    assert "uniquefuturevocabulary" not in model.named_steps["tfidf"].vocabulary_
    events.loc[12:, "raised_guidance"] = ~events.loc[12:, "raised_guidance"]
    _, altered_metadata, altered_model = fit_text_model(events, tmp_path, settings)
    np.testing.assert_array_equal(model.named_steps["classifier"].coef_,
                                  altered_model.named_steps["classifier"].coef_)
    assert metadata["training_inputs_sha256"] == altered_metadata["training_inputs_sha256"]


@pytest.mark.parametrize("text", [
    "Company raises outlook for FY25.",
    "Company raises guidance for FY25.",
    "Company raises low end of full year FY25 revenue guidance.",
    "Company increasing the midpoint for Full-Year 2024 operational sales 5 and adjusted operational EPS guidance.",
    "Company increases Full-Year 2024 operational sales 5 guidance.",
])
def test_v2_repairs_explicit_guidance_formats(text):
    assert guidance_label(text, "lexical-v1.2")["raised_guidance"]


def test_one_event_cannot_produce_a_degenerate_confidence_interval():
    events = pd.DataFrame(dict(signal_month=["2024-01"], excess_return=[0.02]))
    interval = cluster_interval(events, seed=42)
    assert interval["mean_excess"] == 0.02
    assert interval["lower_95"] is None
    assert interval["upper_95"] is None

