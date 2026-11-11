import pandas as pd

from quant_research.earnings_comparison import paired_decisions, reconcile_paired_accounts
from quant_research.earnings_ledger import simulate_earnings_account
from quant_research.earnings_selection import (
    frozen_selection_features,
    match_frozen_event_selections,
    select_earnings_arms,
)
from quant_research.fixtures import create_fixture


def test_four_arm_selection_frozen_matching_and_independent_accounts_work_together(tmp_path):
    snapshot = create_fixture(tmp_path / "source", tmp_path / "snapshot", stocks=24)
    mapping = {"XSHG": "shanghai_main", "XSHE": "shenzhen_main", "XBSE": "beijing"}
    meta, bars = snapshot.tables["instruments"], snapshot.tables["bars"]
    meta["board"] = meta.board.map(mapping)
    snapshot.tables["rules"]["board"] = snapshot.tables["rules"].board.map(mapping)
    offsets = bars.date.map({day: i for i, day in enumerate(snapshot.dates)})
    bars["open"], bars["close"] = 10. + offsets * .001, 10. + offsets * .001
    bars["high"], bars["low"] = bars.close + .1, bars.close - .1
    bars["upper_limit"], bars["lower_limit"] = bars.close * 1.1, bars.close * .9
    bars["factor"], bars["volume"], bars["amount"] = 1., 4_000_000., 40_000_000.
    snapshot.tables["actions"] = snapshot.tables["actions"].iloc[:0]
    industries = meta[["instrument_id", "industry"]].assign(
        start_date=snapshot.dates[0], end_date="", known_at="2020-01-01T00:00:00Z")
    caps = industries.drop(columns="industry").assign(float_market_cap=1_000_000_000.)
    day, at = snapshot.dates[150], snapshot.dates[150] + "T10:00:00Z"
    features = frozen_selection_features(snapshot, day, at, industries, caps)
    # Deliberately synthetic event and model fixtures; these flags cannot qualify a real experiment.
    events = pd.DataFrame([dict(announcement_id=str(i), instrument_id=meta.instrument_id.iloc[i],
        fiscal_period="2023-06-30", available_at=snapshot.dates[149] + "T10:00:00Z",
        document_kind="earnings_forecast", profit_metric="attributable_to_parent_same_reporting_period",
        net_profit_lower_cny=140., net_profit_upper_cny=160., prior_same_period_profit_cny=100.,
        disclosed_yoy_lower=.40, disclosed_yoy_upper=.60, financial_fields_verified=True,
        first_numeric_version_verified=True, historical_pit_verified=False,
        availability_basis="local_observation", pdf_sha256="a" * 64) for i in [0, 1]])
    model = features[["date", "instrument_id"]].assign(model_rank=range(len(features)),
        available_at=day + "T09:00:00Z", model_manifest_sha256="b" * 64)
    candidates = select_earnings_arms(features, events, snapshot.dates, model_ranks=model)
    fees = pd.DataFrame([dict(board=board, start_date=snapshot.dates[0], end_date="",
        commission_rate=.0005, minimum_commission=5., transfer_rate=.00001,
        stamp_duty=.0005, slippage_bps=10.) for board in mapping.values()])

    def simulate(decisions):
        return simulate_earnings_account(snapshot, decisions, fees, industries, day, snapshot.dates[180])

    accounts = {name: simulate(rows) for name, rows in candidates["arms"].items()}
    assert all(not account["fills"].empty for account in accounts.values())
    coverage = pd.DataFrame([dict(exchange=exchange, start_date=snapshot.dates[100], end_date=day,
        known_at=day + "T09:00:00Z", complete=True, source_sha256="c" * 64) for exchange in ["XSHG", "XSHE"]])
    matches = match_frozen_event_selections(accounts["E1"]["selections"], features, events, coverage, snapshot.dates)
    assert matches["intended_budget_coverage"] == 1. and len(matches["pairs"]) == 2
    decisions = paired_decisions(matches["pairs"], features)
    result = reconcile_paired_accounts(simulate(decisions["E1_matched"]),
                                      simulate(decisions["P_matched_exposure"]), matches["pairs"])
    assert result["internal_arithmetic_reconciled"]
    assert result["daily_accounts"].daily_net_return_difference.abs().max() == 0.
    assert result["pair_diagnostics"].same_planned_budget.all()
    assert all(account["summary"]["formal_profit_qualified"] is False for account in accounts.values())
    assert result["formal_profit_qualified"] is False
