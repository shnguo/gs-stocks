import numpy as np
import pandas as pd
import pytest

from quant_research.features import build_panel
from quant_research.labels import gross_wealth_return, wealth_state
from quant_research.storage import validate_tables


def test_holder_cash_is_not_the_diluted_exchange_reference_amount():
    dates = pd.Index(["2023-04-12", "2023-04-13", "2023-04-14"])
    actions = pd.DataFrame([dict(ex_date=dates[1], split_ratio=1., cash_per_share=.96)])
    shares, cash = wealth_state(dates, actions)
    expected = (23.16 + .96) / 24.11 - 1
    assert gross_wealth_return(24.11, 23.16, 0, 1, shares, cash) == pytest.approx(expected)
    assert gross_wealth_return(23.16, 23.16, 1, 2, shares, cash) == 0


def test_dividends_are_not_reinvested_and_split_uses_prior_share_entitlement():
    dates = pd.Index(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"])
    actions = pd.DataFrame([
        dict(ex_date=dates[1], split_ratio=2., cash_per_share=1.),
        dict(ex_date=dates[2], split_ratio=1., cash_per_share=.5),
    ])
    shares, cash = wealth_state(dates, actions)
    np.testing.assert_array_equal(shares, [1, 2, 2, 2])
    np.testing.assert_array_equal(cash, [0, 1, 2, 2])
    assert gross_wealth_return(10, 4.5, 0, 3, shares, cash) == pytest.approx(.1)


def test_panel_labels_use_actual_cash_and_keep_unpaid_receivables(snapshot):
    symbol = "synthetic.xshe.000001"
    dates = snapshot.dates
    snapshot.tables["actions"] = pd.concat([snapshot.tables["actions"], pd.DataFrame([
        dict(instrument_id=symbol, ex_date=dates[203], pay_date=dates[210],
             split_ratio=1., cash_per_share=.96)])], ignore_index=True)
    panel = build_panel(snapshot, 5, risk_policy="include")
    row = panel.samples.loc[(panel.samples.instrument_id == symbol) &
                            (panel.samples.date == dates[200])].iloc[0]
    bars = snapshot.tables["bars"].set_index(["instrument_id", "date"])
    expected = (bars.loc[(symbol, dates[206]), "open"] + .96) / bars.loc[
        (symbol, dates[201]), "open"] - 1
    assert row.forward_return == pytest.approx(expected)


def test_price_continuity_does_not_release_unknown_holder_terms(snapshot):
    bars = snapshot.tables["bars"]
    bars["sequence_id"] = bars.instrument_id
    bars["label_sequence_id"] = bars.instrument_id
    symbol = "synthetic.xshe.000001"
    after = (bars.instrument_id == symbol) & (bars.date >= snapshot.dates[203])
    bars.loc[after, "label_sequence_id"] += "/holder-unresolved"
    panel = build_panel(snapshot, 5, risk_policy="include")
    row = panel.samples.loc[(panel.samples.instrument_id == symbol) &
                            (panel.samples.date == snapshot.dates[200])].iloc[0]
    assert row.label_status == "unverified_corporate_action_boundary"
    assert np.isnan(row.forward_return)
    assert np.isfinite(panel.values[int(row.stock_index), int(row.date_index)]).all()


def test_unknown_source_state_and_pay_date_stay_unknown_in_training(snapshot):
    snapshot.tables["bars"]["source_is_st"] = np.nan
    snapshot.tables["actions"]["pay_date"] = ""
    validate_tables(snapshot.tables, require_execution_rules=False)
    with pytest.raises(ValueError, match="entitlement/pay dates"):
        validate_tables(snapshot.tables, require_execution_rules=True)
    snapshot.tables["bars"].loc[0, "source_is_st"] = 2
    with pytest.raises(ValueError, match="explicitly unknown"):
        validate_tables(snapshot.tables, require_execution_rules=False)
