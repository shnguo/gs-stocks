"""Research diagnostics for the unregistered earnings protocol, never profit acceptance.

Keep every observed cash date and every frozen pair. Input accounts are independently
reconciled before comparison. Numerical gates are diagnostics: registration, original
source acceptance and prospective operation remain separate required work.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .earnings_comparison import attribute_account, reconcile_paired_accounts, validate_pairs


def normal_holding_capacity(observed_sessions=504, slots=6, horizon=20, minimum_events=100):
    """Upper bound with no early exits; day zero is the first frozen decision close."""
    if any(type(v) is not int or v < 1 for v in [observed_sessions, slots, horizon, minimum_events]):
        raise ValueError("Capacity inputs must be positive integers")
    # First entry day 1, first mature exit day 1+horizon. Replacement can enter that day.
    cycles = max(0, (observed_sessions - 2) // horizon)
    bound = slots * cycles
    return {"observed_sessions": observed_sessions, "slots": slots, "horizon": horizon,
            "no_early_exit_mature_trade_upper_bound": bound,
            "minimum_sessions_for_event_count_at_full_capacity":
                math.ceil(minimum_events / slots) * horizon + 2,
            "minimum_events": minimum_events, "event_count_feasible_without_early_exits": bound >= minimum_events,
            "scope": "normal_holding_only_not_a_bound_when_forced_risk_exits_free_slots_early",
            "formal_profit_qualified": False}


def _calendar(dates):
    dates = list(dates)
    if (not dates or any(not isinstance(d, str) or pd.Timestamp(d).strftime("%Y-%m-%d") != d for d in dates)
            or dates != sorted(set(dates))):
        raise ValueError("Expected observed session calendar must be nonempty, unique and sorted")
    return dates


def account_metrics(account, dates):
    dates = _calendar(dates)
    if account["nav"].date.tolist() != dates:
        raise ValueError("Account must retain the entire expected observed calendar, including cash dates")
    attribution = attribute_account(account)
    initial = float(account["summary"]["initial_capital"])
    values = account["nav"].nav.to_numpy(float)
    if np.any(values <= 0):
        raise ValueError("Nonpositive account NAV cannot use ordinary percentage return diagnostics")
    peaks = np.maximum.accumulate(np.r_[initial, values])[1:]
    return {"observed_sessions": len(dates), "initial_capital": initial,
            "net_total_return": float(values[-1] / initial - 1),
            "cagr": float((values[-1] / initial) ** (252 / len(dates)) - 1),
            "maximum_drawdown": float(np.min(values / peaks - 1)),
            "mean_cash_weight": float((account["nav"].cash / account["nav"].nav).mean()),
            "terminal_stale_marks": bool(attribution["terminal_trade_attribution"].stale_mark.any()),
            "formal_profit_qualified": False}


def paired_moving_block_interval(treated, control, *, block=20, repetitions=10000,
                                 seed=20261012, interval=.95):
    """Noncircular moving blocks; common indices, final block truncated to original n.

    Returns percentile limits of 252*mean daily return differences. No NaN dropping,
    automatic block shortening, independent arm resampling, or CAGR substitution.
    """
    left, right = np.asarray(treated, dtype=float), np.asarray(control, dtype=float)
    if (left.ndim != 1 or left.shape != right.shape or not np.isfinite(left).all()
            or not np.isfinite(right).all()):
        raise ValueError("Paired bootstrap requires equally sized finite daily return vectors")
    if (type(block) is not int or block < 1 or len(left) < 2 * block
            or type(repetitions) is not int or repetitions < 1 or not 0 < interval < 1):
        raise ValueError("Invalid fixed block design or insufficient observed sessions")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(left) - block + 1, size=(repetitions, math.ceil(len(left) / block)))
    indices = (starts[:, :, None] + np.arange(block)).reshape(repetitions, -1)[:, :len(left)]
    differences = left - right
    samples = differences[indices].mean(axis=1) * 252
    alpha = (1 - interval) / 2
    low, high = np.quantile(samples, [alpha, 1 - alpha], method="linear")
    return {"annualized_mean_daily_difference": float(differences.mean() * 252),
            "interval_lower": float(low), "interval_upper": float(high),
            "block": block, "repetitions": repetitions, "seed": seed, "interval": interval,
            "method": "noncircular_moving_block_percentile_linear_quantile_truncate_last_block",
            "formal_profit_qualified": False}


def mature_filled_events(account, frozen_events, dates, horizon=20):
    dates = _calendar(dates)
    required = {"date", "decision_at", "trade_id", "instrument_id", "fiscal_period"}
    if required - set(frozen_events):
        raise ValueError("Maturity needs pre-frozen company/report identities")
    metadata = frozen_events.copy()
    if (metadata.trade_id.duplicated().any() or metadata.duplicated(["instrument_id", "fiscal_period"]).any()
            or not metadata[list(required)].map(lambda v: isinstance(v, str) and bool(v.strip())).all().all()):
        raise ValueError("Missing or duplicate frozen company/report identity")
    if not metadata.fiscal_period.map(lambda v: pd.Timestamp(v).strftime("%Y-%m-%d") == v).all():
        raise ValueError("Invalid fiscal period identity")
    if type(horizon) is not int or horizon < 1:
        raise ValueError("Invalid maturity horizon")
    fills = account["fills"]
    if fills.empty:
        return pd.DataFrame(columns=[*metadata.columns, "entry_date", "label_end", "mature"])
    buys = fills.loc[fills.side == "buy", ["trade_id", "instrument_id", "date"]].rename(
        columns={"date": "entry_date", "instrument_id": "filled_instrument"})
    if buys.trade_id.duplicated().any() or not set(buys.trade_id) <= set(metadata.trade_id):
        raise ValueError("Actual entries need one frozen event identity each")
    joined = buys.merge(metadata, on="trade_id", validate="one_to_one")
    if not joined.filled_instrument.eq(joined.instrument_id).all():
        raise ValueError("Frozen event and filled company identities differ")
    indices = {d: i for i, d in enumerate(dates)}
    ends = []
    for row in joined.itertuples():
        clock = pd.Timestamp(row.decision_at)
        if (row.entry_date not in indices or row.date not in indices or pd.isna(clock) or clock.tzinfo is None
                or clock.tz_convert("Asia/Shanghai").strftime("%Y-%m-%d") != row.date
                or not pd.Timestamp(row.date + "T15:00:00+08:00") <= clock
                < pd.Timestamp(row.entry_date + "T09:00:00+08:00")):
            raise ValueError("Actual entry lacks a timely pre-frozen observed decision")
        end_index = indices[row.entry_date] + horizon
        ends.append(dates[end_index] if end_index < len(dates) else None)
    joined["label_end"] = ends
    joined["mature"] = joined.label_end.notna()
    return joined.drop(columns="filled_instrument")


def matched_budget_coverage(full_account, pairs):
    validate_pairs(pairs)
    selections = full_account["selections"]
    selected = selections.loc[selections.status == "selected_before_open"]
    if selected.trade_id.duplicated().any():
        raise ValueError("Duplicate frozen full-account selections")
    if pairs.empty:
        return 0. if not selected.empty else None
    if not set(pairs.treated_trade_id) <= set(selected.trade_id):
        raise ValueError("Match was not a frozen full-account selection")
    joined = pairs.merge(selected, left_on="treated_trade_id", right_on="trade_id",
                         suffixes=("_pair", "_full"), validate="one_to_one")
    if (not joined.date_pair.eq(joined.date_full).all() or not joined.treated.eq(joined.instrument_id).all()
            or not joined.decision_at_pair.map(pd.Timestamp).eq(joined.decision_at_full.map(pd.Timestamp)).all()
            or not np.allclose(joined.intended_budget, joined.budget, atol=1e-8, rtol=0)):
        raise ValueError("Frozen matching identity, clock or intended budget differs from full account")
    if not np.isfinite(selected.budget).all() or selected.budget.le(0).any():
        raise ValueError("Invalid intended budget denominator")
    return float(pairs.intended_budget.sum() / selected.budget.sum())


def remove_best_five_pairs(comparison, pairs, mature_trade_ids):
    """Subtract both sides' cumulative P&L; leave vacated money in cash, never resize.

    Choose the five most profitable *mature matched* E1 events (including losses if
    necessary), ordered by terminal net CNY P&L then trade ID. Immature events remain.
    This is an explicit draft convention to register before any outcome evaluation.
    """
    daily = comparison["daily_accounts"]
    terminal = comparison["treated_attribution"]["terminal_trade_attribution"]
    ranking = terminal.loc[terminal.trade_id.isin(mature_trade_ids)].sort_values(
        ["net_pnl_cny", "trade_id"], ascending=[False, True])
    chosen = ranking.trade_id.head(5).tolist()
    if len(chosen) < 5:
        return {"available": False, "reason": "fewer_than_five_mature_matched_filled_events",
                "removed_treated_trade_ids": chosen, "formal_profit_qualified": False}
    controls = pairs.set_index("treated_trade_id").loc[chosen, "control_trade_id"].tolist()
    # Initial capital is carried explicitly, not inferred from day-one NAV.
    initial = float(comparison["initial_capital"])
    adjusted = {}
    for side, ids in [("treated", chosen), ("control", controls)]:
        attribution = comparison[f"{side}_attribution"]["daily_trade_attribution"]
        removed = attribution.loc[attribution.trade_id.isin(ids)].groupby("date").net_pnl_cny.sum()
        nav = daily[f"nav_{side}"].to_numpy(float) - removed.reindex(daily.date, fill_value=0).to_numpy(float)
        if np.any(nav <= 0):
            raise ValueError("Best-five removal produced nonpositive NAV")
        adjusted[side] = nav / np.r_[initial, nav[:-1]] - 1
    return {"available": True, "removed_treated_trade_ids": chosen, "removed_control_trade_ids": controls,
            "annualized_mean_daily_difference": float((adjusted["treated"] - adjusted["control"]).mean() * 252),
            "policy": "remove_matched_pair_cumulative_pnl_keep_cash_no_reallocation",
            "formal_profit_qualified": False}


def stability_block_diagnostics(daily_difference, protocol):
    """Fixed consecutive session blocks; v1 retains its original four-block rule.

    V2 keeps the original 75% positive-block fraction over eight 63-session blocks.
    Annualization remains 252; extending observation does not change a year's scale.
    """
    values = np.asarray(daily_difference, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Stability needs every finite daily difference")
    design = protocol["validation"]
    setting = design.get("stability", {"block_sessions": 63, "required_complete_blocks": 4,
        "minimum_positive_blocks": protocol["proposed_acceptance_gates"]["positive_excess_quarters_at_least"]})
    block, count, minimum = (setting[key] for key in ["block_sessions", "required_complete_blocks", "minimum_positive_blocks"])
    if (any(type(v) is not int or v < 1 for v in [block, count, minimum]) or minimum > count
            or block * count != design["single_formal_look_sessions"]
            or minimum != protocol["proposed_acceptance_gates"]["positive_excess_quarters_at_least"]):
        raise ValueError("Stability block rule must cover the exact formal horizon and agree with its gate")
    excess = [float(values[i:i + block].mean() * 252) for i in range(0, len(values) - block + 1, block)]
    at_look = len(values) == design["single_formal_look_sessions"]
    positives = sum(v > 0 for v in excess)
    return dict(block_sessions=block, required_complete_blocks=count, minimum_positive_blocks=minimum,
        complete_block_excess=excess, complete_blocks=len(excess), positive_blocks=positives,
        numerical_gate_passed=at_look and len(excess) == count and positives >= minimum,
        formal_profit_qualified=False)


def evaluate_research_diagnostics(scenarios, pairs, frozen_events, dates, protocol):
    """Compare base/stress/delayed account triples without accepting a formal result.

    Each scenario supplies full E1, matched E1 and matched price control accounts.
    The external observed calendar is mandatory; it cannot be inferred after deletions.
    A CI is emitted only at the exact design look, never at the operational review.
    """
    dates = _calendar(dates)
    if set(scenarios) != {"base", "stress", "delayed"}:
        raise ValueError("Need base, doubled-friction and one-session-delay scenarios")
    design = protocol["validation"]
    results, comparisons = {}, {}
    expected = {"base": (1, 0), "stress": (2, 0), "delayed": (1, 1)}
    capital, horizon = protocol["capital_cny"], protocol["execution"]["primary_holding_sessions"]
    for name, accounts in scenarios.items():
        if set(accounts) != {"full", "treated", "control"}:
            raise ValueError("Each scenario needs full E1 and both matched accounts")
        metrics = {}
        for arm, account in accounts.items():
            summary = account["summary"]
            if (summary["initial_capital"] != capital or summary["horizon"] != horizon
                    or (summary["friction_multiplier"], summary["delay_sessions"]) != expected[name]):
                raise ValueError("Scenario account capital, horizon, cost or delay differs from design")
            metrics[arm] = account_metrics(account, dates)
        paired = reconcile_paired_accounts(accounts["treated"], accounts["control"], pairs)
        paired["initial_capital"] = capital
        comparisons[name] = paired
        metrics["annualized_mean_daily_difference"] = float(paired["daily_accounts"].daily_net_return_difference.mean() * 252)
        pair_status = paired["pair_diagnostics"]
        exposure = paired["daily_pair_exposure"]
        metrics["actual_exposure"] = {
            "frozen_pairs": len(pairs),
            "one_sided_or_different_entry_pairs": int(pair_status.entry_status.eq(
                "different_or_one_sided_entry").sum()) if not pair_status.empty else 0,
            "both_unfilled_pairs": int(pair_status.entry_status.eq("both_unfilled").sum()) if not pair_status.empty else 0,
            "same_planned_budget_pairs": int(pair_status.same_planned_budget.sum()) if not pair_status.empty else 0,
            "one_sided_holding_pair_dates": int(exposure.holding_status.isin(
                ["treated_only", "control_only"]).sum()) if not exposure.empty else 0,
            "actual_exposure_equality_accepted": False}
        results[name] = metrics
    base = scenarios["base"]
    mature = mature_filled_events(base["full"], frozen_events, dates, horizon)
    mature_ids = mature.loc[mature.mature, "trade_id"].tolist()
    mature_dates = int(mature.loc[mature.mature, "entry_date"].nunique())
    coverage = matched_budget_coverage(base["full"], pairs)
    at_look = len(dates) == design["single_formal_look_sessions"]
    ci = None
    if at_look:
        b = design["bootstrap"]
        daily = comparisons["base"]["daily_accounts"]
        ci = paired_moving_block_interval(daily.daily_return_treated, daily.daily_return_control,
             block=b["paired_date_blocks"], repetitions=b["repetitions"], seed=b["seed"], interval=b["interval"])
    daily_difference = comparisons["base"]["daily_accounts"].daily_net_return_difference.to_numpy(float)
    stability = stability_block_diagnostics(daily_difference, protocol)
    best_five = remove_best_five_pairs(comparisons["base"], pairs, mature_ids)
    gates = protocol["proposed_acceptance_gates"]
    diagnostics = {
        "exact_single_look_length": at_look,
        "mature_unique_filled_events": len(mature_ids) >= design["minimum_mature_unique_company_report_events"],
        "mature_active_entry_dates": mature_dates >= design["minimum_active_entry_dates"],
        "net_account_return_positive": results["base"]["full"]["net_total_return"] > 0,
        "annualized_matched_excess_hurdle": results["base"]["annualized_mean_daily_difference"] >= gates["annualized_net_excess_over_matched_price_control"],
        "interval_lower_above_zero": ci is not None and ci["interval_lower"] > 0,
        "maximum_account_drawdown": results["base"]["full"]["maximum_drawdown"] >= -gates["maximum_account_drawdown"],
        "stress_account_and_excess_positive": results["stress"]["full"]["net_total_return"] > 0 and results["stress"]["annualized_mean_daily_difference"] > 0,
        "delayed_account_and_excess_positive": results["delayed"]["full"]["net_total_return"] > 0 and results["delayed"]["annualized_mean_daily_difference"] > 0,
        "stability_blocks_positive": stability["numerical_gate_passed"],
        "remove_best_five_excess_positive": best_five["available"] and best_five["annualized_mean_daily_difference"] > 0,
        "matched_intended_budget_coverage": coverage is not None and coverage >= gates["minimum_matched_weight_coverage"],
        "terminal_marks_not_stale": all(not m[a]["terminal_stale_marks"] for m in results.values() for a in ["full", "treated", "control"]),
    }
    return {"status": "research_diagnostics_only_unregistered_not_established",
            "formal_profit_qualified": False, "numerical_gate_results": diagnostics,
            "numerical_gates_all_passed": all(diagnostics.values()),
            "protocol_version": protocol["version"],
            "single_formal_look_sessions": design["single_formal_look_sessions"],
            "observed_sessions": len(dates), "mature_unique_filled_events": len(mature_ids),
            "mature_active_entry_dates": mature_dates, "matched_intended_budget_coverage": coverage,
            "scenarios": results, "primary_interval": ci,
            "complete_63_session_block_excess": stability["complete_block_excess"],
            "stability": stability, "best_five_sensitivity": best_five,
            "normal_holding_capacity": normal_holding_capacity(len(dates),
                protocol["portfolio"]["maximum_positions"], horizon,
                design["minimum_mature_unique_company_report_events"]),
            "registration_clarifications_required": ["noncircular bootstrap and percentile quantile method",
                "matched subset primary statistic and full E1 absolute account gates",
                "mature actual fills as company/report count",
                "fixed consecutive session blocks versus calendar-quarter stability",
                "matched mature top-five paired P&L removal with no reallocation"],
            "external_acceptance_still_required": ["original source and independent numeric acceptance",
                "complete first-version and execution evidence", "actual broker minimum fee",
                "registered code protocol environment and timely prospective operation"]}
