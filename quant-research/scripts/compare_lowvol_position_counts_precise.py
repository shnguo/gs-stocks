"""Fixed low-vol signal with equal target budgets across position counts.

Derived from the frozen simulator. Corporate actions and execution constraints are
unchanged; only retained-position resizing and configurable portfolio breadth differ.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from compare_strategy_directions import (
    Position,
    features,
    fees,
    load_inputs,
    save,
    statistics,
)

from quant_research.storage import file_hash, utc_now


def whole_lots(shares):
    """Ignore sub-micro-share floating noise; cash/fee checks still prevent borrowing."""
    return math.floor((shares + 1e-7) / 100) * 100


def simulate_balanced_target(
    dates, symbols, instruments, arrays, actions, scores, adv20, protocol, scenario, name, output
):
    start = int(np.searchsorted(dates, protocol["evaluation_start"]))
    positions = {}
    cash = float(protocol["initial_cash"])
    receivables = []
    fills, orders, events, nav, holdings = [], [], [], [], []
    history = {}
    uncertain_reserve = 0.0
    first = start + scenario["rebalance_offset"] + scenario["delay_sessions"]
    due = set(range(first, len(dates), protocol["rebalance_sessions"]))
    dividend_delay = scenario.get("dividend_delay_days", 60)
    delisted = instruments.delisted_at.tolist()
    for i in range(start, len(dates)):
        day = dates[i]
        day_actions = {j: (split, dividend) for j, split, dividend in actions.get(i, [])}
        for j, (split, dividend) in day_actions.items():
            if j not in positions:
                continue
            pos = positions[j]
            old_shares = pos.shares
            amount = old_shares * dividend * 0.8
            if amount:
                due_date = str(date.fromisoformat(day) + timedelta(days=dividend_delay))
                receivables.append((due_date, amount, j))
                events.append(
                    dict(
                        date=day,
                        instrument_id=symbols[j],
                        kind="dividend_entitlement",
                        amount=amount,
                        gross=old_shares * dividend,
                        due_date=due_date,
                    )
                )
            pos.shares = float(np.floor(old_shares * split + 1e-8))
            pos.last_price = (pos.last_price - dividend) / split
            if split != 1:
                pos.lock_until = i + 5
                events.append(
                    dict(
                        date=day,
                        instrument_id=symbols[j],
                        kind="split",
                        amount=0.0,
                        old_shares=old_shares,
                        new_shares=pos.shares,
                    )
                )
        # Account for an unresolved change even when the holding is sold today.
        for j, pos in positions.items():
            seq = int(arrays["label_sequence_id"][i, j])
            if (
                seq >= 0
                and pos.source_sequence >= 0
                and seq != pos.source_sequence
                and not pos.uncertain
            ):
                pos.uncertain = True
                uncertain_reserve += pos.shares * pos.last_price
                events.append(
                    dict(
                        date=day,
                        instrument_id=symbols[j],
                        kind="unresolved_label_barrier",
                        amount=pos.shares * pos.last_price,
                    )
                )
            if seq >= 0:
                pos.source_sequence = seq
        outstanding = []
        for payable, amount, j in receivables:
            if payable <= day:
                cash += amount
                events.append(
                    dict(date=day, instrument_id=symbols[j], kind="dividend_paid", amount=amount)
                )
            else:
                outstanding.append((payable, amount, j))
        receivables = outstanding

        def execute(j, side, budget=0.0, signal=None, reason="rebalance", sell_shares=None):
            nonlocal cash
            order = dict(
                date=day,
                instrument_id=symbols[j],
                side=side,
                reason=reason,
                signal_date=dates[signal] if signal is not None else "",
                status="blocked",
            )
            orders.append(order)
            opening = arrays["open"][i, j]
            if (
                not np.isfinite(opening)
                or opening <= 0
                or arrays["source_trade_status"][i, j] != 1
                or not arrays["volume"][i, j] > 0
            ):
                order["block_reason"] = "no_verified_trading_bar"
                return
            if arrays["source_is_st"][i, j] not in (0.0, 1.0):
                order["block_reason"] = "unknown_risk_and_limit_status"
                return
            if side == "buy" and arrays["source_is_st"][i, j] != 0:
                order["block_reason"] = "st_or_unknown"
                return
            if side == "sell" and (i <= positions[j].bought or i < positions[j].lock_until):
                order["block_reason"] = "t_plus_or_new_share_lock"
                return
            previous = arrays["close"][i - 1, j]
            if not np.isfinite(previous):
                order["block_reason"] = "prior_reference_missing"
                return
            split, dividend = day_actions.get(j, (1.0, 0.0))
            reference = (previous - dividend) / split
            pct = 0.05 if arrays["source_is_st"][i, j] == 1 else 0.10
            upper = math.floor(reference * (1 + pct) * 100 + 0.5) / 100
            lower = math.floor(reference * (1 - pct) * 100 + 0.5) / 100
            slip = scenario["slippage_bps"] / 10000
            price = opening * (1 + slip if side == "buy" else 1 - slip)
            if (side == "buy" and (opening >= upper - 1e-8 or price > upper)) or (
                side == "sell" and (opening <= lower + 1e-8 or price < lower)
            ):
                order["block_reason"] = "adverse_opening_limit_assumption"
                return
            capacity = adv20[i - 1, j] * protocol["adv_participation"]
            if not np.isfinite(capacity) or capacity <= 0:
                order["block_reason"] = "prior_liquidity_missing"
                return
            if side == "buy":
                quantity = min(
                    whole_lots(budget / opening),
                    whole_lots(cash / price),
                    whole_lots(capacity / price),
                )
                while (
                    quantity > 0
                    and quantity * price
                    + sum(fees(quantity * price, day, side, scenario["commission_multiplier"]))
                    > cash + 1e-8
                ):
                    quantity -= 100
            else:
                requested = (
                    positions[j].shares
                    if sell_shares is None
                    else min(positions[j].shares, sell_shares)
                )
                quantity = (
                    requested if capacity >= requested * price else whole_lots(capacity / price)
                )
                if quantity < positions[j].shares:
                    quantity = whole_lots(quantity)
            if quantity <= 0:
                order["block_reason"] = "insufficient_cash_lot_or_capacity"
                return
            notional = quantity * price
            commission, transfer, stamp = fees(
                notional, day, side, scenario["commission_multiplier"]
            )
            cost = commission + transfer + stamp
            if side == "buy":
                cash -= notional + cost
                if j in positions:
                    positions[j].shares += quantity
                    positions[j].lock_until = max(positions[j].lock_until, i + 1)
                else:
                    positions[j] = Position(
                        quantity, opening, i, i + 1, int(arrays["label_sequence_id"][i, j])
                    )
            else:
                cash += notional - cost
                positions[j].shares -= quantity
                if positions[j].shares <= 1e-8:
                    del positions[j]
            if cash < -1e-7:
                raise AssertionError("Borrowed cash")
            order.update(status="filled", shares=quantity)
            fills.append(
                dict(
                    date=day,
                    instrument_id=symbols[j],
                    side=side,
                    shares=quantity,
                    price=price,
                    reference_open=opening,
                    notional=notional,
                    commission=commission,
                    transfer=transfer,
                    stamp=stamp,
                    fees=cost,
                    slippage=abs(price - opening) * quantity,
                    reason=reason,
                    signal_date=order["signal_date"],
                )
            )

        for j in list(positions):
            if arrays["source_is_st"][i - 1, j] != 0 or (delisted[j] and day >= delisted[j]):
                execute(j, "sell", reason="dated_risk_exit")
        if i in due:
            signal = i - 1 - scenario["delay_sessions"]
            row = scores[signal]
            ranked = np.argsort(-row, kind="stable")
            ranked = ranked[np.isfinite(row[ranked])]
            prior_positions, prior_equity = history.get(
                signal, (set(), float(protocol["initial_cash"]))
            )
            target = [
                int(j) for j in ranked[: protocol["retention_buffer_rank"]] if j in prior_positions
            ]
            target = target[: protocol["max_positions"]]
            for j in ranked:
                if len(target) >= protocol["max_positions"]:
                    break
                if j not in target:
                    target.append(int(j))
            for j in list(positions):
                if j not in target:
                    execute(j, "sell", signal=signal)
            target_amount = prior_equity * protocol["position_budget_fraction"]
            # A single deterministic rounding plan keeps total invested value
            # close to 90% across N. Combine each name into one net order.
            planned = {}
            for j in target:
                opening = arrays["open"][i, j]
                if np.isfinite(opening) and opening > 0:
                    planned[j] = whole_lots(target_amount / opening)
            if len(planned) == protocol["max_positions"]:
                residual = prior_equity * 0.9 - sum(
                    planned[j] * arrays["open"][i, j] for j in planned
                )
                ordered = sorted(
                    planned,
                    key=lambda j: (
                        -(target_amount - planned[j] * arrays["open"][i, j]),
                        symbols[j],
                    ),
                )
                for j in ordered:
                    lot_value = arrays["open"][i, j] * 100
                    if lot_value <= residual + 1e-8:
                        planned[j] += 100
                        residual -= lot_value
            # Equal target budgets for retained positions as well as new names.
            # Decisions use the completed signal date; opening is the execution price.
            for j in target:
                opening = arrays["open"][i, j]
                if j in positions and np.isfinite(opening) and opening > 0:
                    desired = planned[j]
                    excess = positions[j].shares - desired
                    if excess > 0:
                        execute(
                            j,
                            "sell",
                            signal=signal,
                            reason="resize_to_equal_target",
                            sell_shares=excess,
                        )
            # Blocked entries are not replaced with other candidates.
            for j in target:
                opening = arrays["open"][i, j]
                if j in positions and np.isfinite(opening) and opening > 0:
                    remaining = max(0.0, (planned[j] - positions[j].shares) * opening)
                    if remaining >= opening * 100:
                        execute(j, "buy", remaining, signal, reason="resize_to_equal_target")
                elif j not in positions and len(positions) < protocol["max_positions"]:
                    execute(j, "buy", planned[j] * opening, signal)
        stale = uncertain_count = 0
        uncertain_value = 0.0
        for j, pos in positions.items():
            current = arrays["close"][i, j]
            if np.isfinite(current) and current > 0:
                pos.last_price = current
            else:
                stale += 1
            if pos.uncertain or not np.isfinite(current):
                uncertain_count += 1
                uncertain_value += pos.shares * pos.last_price
            holdings.append(
                dict(
                    date=day,
                    instrument_id=symbols[j],
                    shares=pos.shares,
                    mark=pos.last_price,
                    value=pos.shares * pos.last_price,
                    stale=not np.isfinite(current),
                    uncertain=pos.uncertain,
                )
            )
        market_value = sum(p.shares * p.last_price for p in positions.values())
        receivable = sum(v[1] for v in receivables)
        equity = cash + market_value + receivable
        nav.append(
            dict(
                date=day,
                equity=equity,
                cash=cash,
                market_value=market_value,
                receivable=receivable,
                positions=len(positions),
                stale=stale,
                uncertain_count=uncertain_count,
                uncertain_value=uncertain_value,
                cumulative_uncertainty_reserve=uncertain_reserve,
            )
        )
        history[i] = (set(positions), equity)
    output.mkdir(parents=True)
    frames = {
        "nav": pd.DataFrame(nav),
        "fills": pd.DataFrame(fills),
        "orders": pd.DataFrame(orders),
        "cash_events": pd.DataFrame(events),
        "holdings": pd.DataFrame(holdings),
    }
    for key, frame in frames.items():
        frame.to_parquet(output / (key + ".parquet"), index=False)
    result = statistics(frames["nav"], protocol["initial_cash"])
    result.update(
        strategy=name,
        scenario=scenario["name"],
        trades=len(fills),
        completed_sales=sum(f["side"] == "sell" for f in fills),
        commission=sum(f["commission"] for f in fills),
        stamp=sum(f["stamp"] for f in fills),
        transfer=sum(f["transfer"] for f in fills),
        slippage=sum(f["slippage"] for f in fills),
        minimum_fee_fills=int(sum(abs(f["commission"] - 5.0) < 1e-9 for f in fills)),
        blocked_reasons=dict(
            Counter(v["block_reason"] for v in orders if v["status"] == "blocked")
        ),
        unresolved_barrier_events=sum(v["kind"] == "unresolved_label_barrier" for v in events),
        cumulative_uncertainty_reserve=uncertain_reserve,
        permanent_uncertainty_haircut_return=(nav[-1]["equity"] - uncertain_reserve)
        / protocol["initial_cash"]
        - 1,
        input_assumptions_not_exchange_execution_certification=True,
        formal_profit_qualified=False,
    )
    save(output / "summary.json", result)
    return result, frames["nav"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    cfg = json.loads((root / "protocol.json").read_text())
    prior_root = Path("artifacts/strategy-screen-20261013-v1")
    old = json.loads((prior_root / "protocol.json").read_text())
    source = Path(__file__).with_name("compare_strategy_directions.py")
    assert file_hash(source) == file_hash(prior_root / "code/compare_strategy_directions.py")
    freeze = json.loads((root / "implementation-freeze.json").read_text())
    for path, digest in freeze["files"].items():
        assert file_hash(Path(path)) == digest, path
    output = root / "results"
    output.mkdir()
    save(
        output / "run-start.json",
        dict(
            started_at=utc_now(),
            protocol_sha256=file_hash(root / "protocol.json"),
            no_sealed_returns=True,
        ),
    )
    dates, symbols, instruments, arrays, actions = load_inputs(old, output)
    all_scores, adv = features(arrays, old)
    scores = all_scores["lowvol60"]
    del all_scores
    summaries = []
    for count in cfg["position_counts"]:
        p = dict(
            old,
            max_positions=count,
            position_budget_fraction=0.9 / count,
            retention_buffer_rank=2 * count,
        )
        for scenario in cfg["scenarios"]:
            item, _ = simulate_balanced_target(
                dates,
                symbols,
                instruments,
                arrays,
                actions,
                scores,
                adv,
                p,
                scenario,
                f"lowvol_{count}",
                output / scenario["name"] / f"lowvol_{count}",
            )
            item.update(position_count=count, allocation_method="balanced_lots_90_target")
            summaries.append(item)
            print(
                "equal_target",
                count,
                scenario["name"],
                {
                    k: item[k]
                    for k in [
                        "net_return",
                        "max_drawdown",
                        "mean_exposure",
                        "unresolved_barrier_events",
                    ]
                },
                flush=True,
            )
    save(
        output / "comparison.json",
        dict(
            completed_at=utc_now(),
            summaries=summaries,
            source_signal_unchanged=True,
            formal_profit_qualified=False,
        ),
    )
    pd.DataFrame(
        [{k: v for k, v in row.items() if not isinstance(v, (dict, list))} for row in summaries]
    ).to_csv(output / "comparison.csv", index=False)


if __name__ == "__main__":
    main()
