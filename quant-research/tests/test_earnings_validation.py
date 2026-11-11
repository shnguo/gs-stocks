import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_earnings_ledger import account as ledger_fixture
from test_earnings_ledger import decisions, run

from quant_research.earnings_comparison import PAIR_COLUMNS
from quant_research.earnings_validation import (
    account_metrics,
    evaluate_research_diagnostics,
    matched_budget_coverage,
    mature_filled_events,
    normal_holding_capacity,
    paired_moving_block_interval,
    remove_best_five_pairs,
    stability_block_diagnostics,
)
from quant_research.fixtures import create_fixture


@pytest.fixture(name="account")
def earnings_account(snapshot):
    return ledger_fixture.__wrapped__(snapshot)


def protocol():
    return json.loads((Path(__file__).parents[1] / "configs/earnings-drift-validation-20261012-v1.json").read_text())


def frozen_metadata(selected):
    result = selected.copy()
    result["fiscal_period"] = result.date  # Deliberately artificial report IDs, no strategy claims.
    return result


def saturated_account(account, observed=252):
    snapshot = account[0]
    snapshot.tables["risk_events"] = snapshot.tables["risk_events"].iloc[:0].copy()
    selected = decisions(snapshot, indices=range(50, 50 + observed - 1, 20),
                         symbols=snapshot.tables["instruments"].instrument_id.tolist())
    selected["trade_id"] = [f"E1/{r.date}/{r.instrument_id}" for r in selected.itertuples()]
    result = run(account, selected, end=50 + observed - 1)
    return result, frozen_metadata(selected), snapshot.dates[50:50 + observed]


def scenario_accounts(account, observed=120):
    snapshot = account[0]
    left, right = snapshot.tables["instruments"].instrument_id.iloc[:2]
    selected = decisions(snapshot, symbols=[left])
    selected["trade_id"] = "E1/artificial"
    controls = decisions(snapshot, symbols=[right])
    controls["trade_id"] = "P_matched/E1/artificial"
    pairs = pd.DataFrame([dict(date=snapshot.dates[50], decision_at=selected.decision_at.iloc[0],
        treated=left, control=right, treated_trade_id="E1/artificial",
        control_trade_id="P_matched/E1/artificial", selection_order=0, intended_budget=30_000.)])
    scenarios = {}
    for name, multiplier, delay in [("base", 1, 0), ("stress", 2, 0), ("delayed", 1, 1)]:
        options = dict(friction_multiplier=multiplier, delay_sessions=delay, end=50 + observed - 1)
        treated = run(account, selected, **options)
        scenarios[name] = dict(full=copy.deepcopy(treated), treated=treated,
                               control=run(account, controls, **options))
    return scenarios, pairs, frozen_metadata(selected), snapshot.dates[50:50 + observed]


def test_normal_holding_bound_is_reproduced_by_actual_six_slot_ledger(account):
    result, metadata, dates = saturated_account(account)
    counted = mature_filled_events(result, metadata, dates)
    bound = normal_holding_capacity(252)
    assert len(counted) == 78 and counted.mature.sum() == 72
    assert bound["no_early_exit_mature_trade_upper_bound"] == counted.mature.sum()
    assert bound["minimum_sessions_for_event_count_at_full_capacity"] == 342
    assert not bound["event_count_feasible_without_early_exits"]
    assert normal_holding_capacity(504)["no_early_exit_mature_trade_upper_bound"] == 150
    assert normal_holding_capacity(120)["no_early_exit_mature_trade_upper_bound"] == 30
    assert not bound["formal_profit_qualified"]


def test_maturity_counts_actual_unique_fills_not_unfilled_candidates_or_end_holdings(account):
    result, metadata, dates = saturated_account(account)
    # The last six are still marked in NAV but cannot contribute matured event evidence.
    assert result["summary"]["terminal_positions"] == 6
    counted = mature_filled_events(result, metadata, dates)
    assert counted.loc[~counted.mature, "label_end"].isna().all()
    extra = metadata.iloc[[0]].copy()
    extra["trade_id"] = "E1/unfilled"
    extra["fiscal_period"] = "1999-12-31"
    assert len(mature_filled_events(result, pd.concat([metadata, extra]), dates)) == 78
    duplicate = metadata.copy()
    duplicate.loc[6, "fiscal_period"] = duplicate.loc[0, "fiscal_period"]
    with pytest.raises(ValueError, match="duplicate frozen company/report"):
        mature_filled_events(result, duplicate, dates)


def test_maturity_rejects_changed_company_and_late_identity_clock(account):
    scenarios, _, metadata, dates = scenario_accounts(account)
    changed = metadata.copy()
    changed.loc[0, "instrument_id"] = "different_company"
    with pytest.raises(ValueError, match="company identities differ"):
        mature_filled_events(scenarios["base"]["full"], changed, dates)
    changed = metadata.copy()
    changed.loc[0, "decision_at"] = dates[1] + "T02:00:00Z"
    with pytest.raises(ValueError, match="timely pre-frozen"):
        mature_filled_events(scenarios["base"]["full"], changed, dates)


def test_account_metrics_reject_deleted_cash_date_nan_and_forged_return(account):
    result = run(account)
    dates = account[0].dates[50:81]
    changed = copy.deepcopy(result)
    changed["nav"] = changed["nav"].iloc[1:]
    with pytest.raises(ValueError, match="entire expected observed calendar"):
        account_metrics(changed, dates)
    for value in [np.nan, .5]:
        changed = copy.deepcopy(result)
        changed["nav"].loc[0, "daily_return"] = value
        with pytest.raises(ValueError):
            account_metrics(changed, dates)
    metrics = account_metrics(result, dates)
    assert metrics["net_total_return"] == pytest.approx(-45.6 / 200000)
    assert metrics["maximum_drawdown"] == pytest.approx(-45.6 / 200000)


def test_block_interval_matches_independent_loop_and_uses_arithmetic_annualization():
    left = np.linspace(-.001, .003, 63)
    right = left / 2
    result = paired_moving_block_interval(left, right, repetitions=200)
    rng = np.random.default_rng(20261012)
    samples = []
    for _ in range(200):
        indices = []
        for start in rng.integers(0, 63 - 20 + 1, size=math.ceil(63 / 20)):
            indices.extend(range(start, start + 20))
        samples.append(sum(left[i] - right[i] for i in indices[:63]) / 63 * 252)
    assert [result["interval_lower"], result["interval_upper"]] == pytest.approx(np.quantile(samples, [.025, .975]))
    assert result["annualized_mean_daily_difference"] == pytest.approx(.126)
    assert result["annualized_mean_daily_difference"] != pytest.approx(
        np.prod(1 + left) ** 4 - np.prod(1 + right) ** 4)
    # Common indices preserve an identical treatment/control time series exactly.
    same = paired_moving_block_interval(left, left)
    assert same["interval_lower"] == same["interval_upper"] == 0
    assert result == paired_moving_block_interval(left, right, repetitions=200)


@pytest.mark.parametrize("left,right", [(np.zeros(40), np.zeros(39)), (np.zeros(39), np.zeros(39)),
                                      (np.r_[np.zeros(39), np.nan], np.zeros(40))])
def test_bootstrap_never_drops_rows_or_shrinks_block(left, right):
    with pytest.raises(ValueError):
        paired_moving_block_interval(left, right)


@pytest.mark.parametrize("observed", [120, 253])
def test_off_schedule_suppresses_interval_and_all_results_stay_research_only(account, observed):
    scenarios, pairs, metadata, dates = scenario_accounts(account, observed=observed)
    result = evaluate_research_diagnostics(scenarios, pairs, metadata, dates, protocol())
    assert result["primary_interval"] is None
    assert not result["numerical_gate_results"]["exact_single_look_length"]
    assert not result["numerical_gate_results"]["mature_unique_filled_events"]
    assert result["mature_unique_filled_events"] == result["mature_active_entry_dates"] == 1
    assert result["matched_intended_budget_coverage"] == 1
    assert result["scenarios"]["base"]["annualized_mean_daily_difference"] == 0
    assert not result["formal_profit_qualified"]


def test_one_sided_failed_control_entry_remains_in_primary_account_difference(account):
    snapshot = account[0]
    control = snapshot.tables["instruments"].instrument_id.iloc[1]
    mask = snapshot.tables["bars"].instrument_id.eq(control) & snapshot.tables["bars"].date.eq(snapshot.dates[51])
    snapshot.tables["bars"].loc[mask, "upper_limit"] = 10.
    scenarios, pairs, metadata, dates = scenario_accounts(account)
    result = evaluate_research_diagnostics(scenarios, pairs, metadata, dates, protocol())
    base = result["scenarios"]["base"]
    assert base["actual_exposure"]["one_sided_or_different_entry_pairs"] == 1
    assert base["actual_exposure"]["one_sided_holding_pair_dates"] == 20
    assert base["annualized_mean_daily_difference"] < 0  # Trading costs versus retained cash.
    assert result["matched_intended_budget_coverage"] == 1


def test_empty_cash_accounts_have_no_invented_coverage_or_events(account):
    snapshot = account[0]
    dates = snapshot.dates[50:170]
    empty_decisions = decisions(snapshot).iloc[:0]
    empty_decisions["trade_id"] = pd.Series(dtype=str)
    empty_decisions["fiscal_period"] = pd.Series(dtype=str)
    scenarios = {}
    for name, multiplier, delay in [("base", 1, 0), ("stress", 2, 0), ("delayed", 1, 1)]:
        empty = run(account, empty_decisions, end=169, friction_multiplier=multiplier, delay_sessions=delay)
        scenarios[name] = dict(full=empty, treated=empty, control=empty)
    result = evaluate_research_diagnostics(scenarios, pd.DataFrame(columns=sorted(PAIR_COLUMNS)),
                                          empty_decisions, dates, protocol())
    assert result["mature_unique_filled_events"] == 0
    assert result["matched_intended_budget_coverage"] is None
    assert result["scenarios"]["base"]["full"]["net_total_return"] == 0
    assert not result["formal_profit_qualified"]


def test_exact_252_session_report_does_not_promote_an_insufficient_sample(account):
    scenarios, pairs, metadata, dates = scenario_accounts(account, observed=252)
    result = evaluate_research_diagnostics(scenarios, pairs, metadata, dates, protocol())
    assert result["primary_interval"]["block"] == 20
    assert result["primary_interval"]["repetitions"] == 10000
    assert result["primary_interval"]["interval_lower"] == 0
    assert len(result["complete_63_session_block_excess"]) == 4
    assert not result["numerical_gates_all_passed"] and not result["formal_profit_qualified"]


def test_scenario_and_frozen_budget_mismatches_are_rejected(account):
    scenarios, pairs, metadata, dates = scenario_accounts(account)
    changed = copy.deepcopy(scenarios)
    changed["stress"]["full"]["summary"]["friction_multiplier"] = 1
    with pytest.raises(ValueError, match="cost or delay differs"):
        evaluate_research_diagnostics(changed, pairs, metadata, dates, protocol())
    changed_pairs = pairs.copy()
    changed_pairs.loc[0, "intended_budget"] += 1
    with pytest.raises(ValueError, match="intended budget differs"):
        matched_budget_coverage(scenarios["base"]["full"], changed_pairs)
    assert matched_budget_coverage(scenarios["base"]["full"], pd.DataFrame(columns=sorted(PAIR_COLUMNS))) == 0


@pytest.fixture(scope="module")
def extended_account(tmp_path_factory):
    root = tmp_path_factory.mktemp("earnings-504")
    snapshot = create_fixture(root / "source", root / "snapshot", stocks=6, days=580)
    return ledger_fixture.__wrapped__(snapshot)


@pytest.mark.parametrize("observed", [120, 252, 503, 504, 505])
def test_v2_has_only_one_interval_look_and_keeps_annualization_and_sample_gates(extended_account, observed):
    design = json.loads((Path(__file__).parents[1] / "configs/earnings-drift-validation-20261012-v2.json").read_text())
    scenarios, pairs, metadata, dates = scenario_accounts(extended_account, observed=observed)
    result = evaluate_research_diagnostics(scenarios, pairs, metadata, dates, design)
    assert (result["primary_interval"] is not None) == (observed == 504)
    assert result["numerical_gate_results"]["exact_single_look_length"] == (observed == 504)
    assert result["single_formal_look_sessions"] == 504
    assert not result["numerical_gate_results"]["mature_unique_filled_events"]
    assert not result["numerical_gate_results"]["mature_active_entry_dates"]
    assert not result["formal_profit_qualified"]
    assert result["stability"]["required_complete_blocks"] == 8
    assert result["stability"]["minimum_positive_blocks"] == 6
    if observed == 504:
        assert result["primary_interval"]["annualized_mean_daily_difference"] == 0
        assert result["stability"]["complete_blocks"] == 8


def test_v2_actual_saturated_ledger_can_reach_100_events_but_does_not_assume_60_entry_dates(extended_account):
    result, metadata, dates = saturated_account(extended_account, observed=504)
    counted = mature_filled_events(result, metadata, dates)
    assert len(counted) == 156 and counted.mature.sum() == 150
    assert counted.loc[counted.mature, "entry_date"].nunique() == 25
    capacity = normal_holding_capacity()
    assert capacity["observed_sessions"] == 504
    assert capacity["event_count_feasible_without_early_exits"]
    assert capacity["no_early_exit_mature_trade_upper_bound"] == 150


@pytest.mark.parametrize("positive_blocks,passed", [(5, False), (6, True), (8, True)])
def test_v2_stability_retains_three_quarters_positive_fraction(positive_blocks, passed):
    design = json.loads((Path(__file__).parents[1] / "configs/earnings-drift-validation-20261012-v2.json").read_text())
    daily = np.repeat([.001] * positive_blocks + [-.001] * (8 - positive_blocks), 63)
    result = stability_block_diagnostics(daily, design)
    assert result["numerical_gate_passed"] == passed
    assert result["complete_block_excess"][0] == pytest.approx(.252)
    assert result["positive_blocks"] == positive_blocks
    assert not stability_block_diagnostics(daily[:252], design)["numerical_gate_passed"]
    broken = copy.deepcopy(design)
    broken["validation"]["stability"]["required_complete_blocks"] = 4
    with pytest.raises(ValueError, match="exact formal horizon"):
        stability_block_diagnostics(daily, broken)


def test_best_five_removes_their_controls_and_keeps_cash_without_dropping_dates():
    dates = pd.bdate_range("2023-01-02", periods=40).strftime("%Y-%m-%d").tolist()
    treated_profits, control_profits = np.arange(1, 7) * 10., np.array([15., 100., 100., 100., 100., 100.])
    comparison = {"initial_capital": 200000.}
    frames = {}
    for arm, profits in [("treated", treated_profits), ("control", control_profits)]:
        rows = [dict(date=date, trade_id=f"{arm}/{i}", net_pnl_cny=float(profit * j / 39))
                for j, date in enumerate(dates) for i, profit in enumerate(profits)]
        frame = pd.DataFrame(rows)
        comparison[arm + "_attribution"] = dict(daily_trade_attribution=frame,
            terminal_trade_attribution=frame.loc[frame.date == dates[-1]])
        frames["nav_" + arm] = 200000 + np.linspace(0, profits.sum(), 40)
    comparison["daily_accounts"] = pd.DataFrame({"date": dates, **frames})
    pairs = pd.DataFrame([dict(treated_trade_id=f"treated/{i}", control_trade_id=f"control/{i}") for i in range(6)])
    result = remove_best_five_pairs(comparison, pairs, [f"treated/{i}" for i in range(6)])
    assert result["removed_control_trade_ids"] == [f"control/{i}" for i in [5, 4, 3, 2, 1]]
    remaining_left, remaining_right = 200000 + np.linspace(0, 10, 40), 200000 + np.linspace(0, 15, 40)
    expected = sum((remaining_left[i] / (remaining_left[i - 1] if i else 200000) - 1)
                   - (remaining_right[i] / (remaining_right[i - 1] if i else 200000) - 1) for i in range(40)) / 40 * 252
    assert result["annualized_mean_daily_difference"] == pytest.approx(expected)
    assert result["annualized_mean_daily_difference"] < 0
    assert not remove_best_five_pairs(comparison, pairs, ["treated/0"])["available"]
