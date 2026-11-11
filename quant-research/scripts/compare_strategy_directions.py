"""Fixed-rule, retrospective strategy screen. Never promotes production source flags."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from quant_research.storage import file_hash, utc_now


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def fees(notional, day, side, multiplier=1):
    commission = max(5., notional * 0.0005 * multiplier)
    transfer = notional * (0.00002 if day < "2022-04-29" else 0.00001)
    stamp = notional * (0.001 if day < "2023-08-28" else 0.0005) if side == "sell" else 0.
    return commission, transfer, stamp


def load_inputs(protocol, output):
    root = Path(protocol["source"])
    manifest = json.loads((root / "manifest.json").read_text())
    if file_hash(root / "manifest.json") != protocol["source_manifest_sha256"]:
        raise ValueError("Frozen source manifest changed")
    for name, expected in manifest["files"].items():
        if file_hash(root / name) != expected:
            raise ValueError("Input changed: " + name)
    instruments = pd.read_parquet(root / "instruments.parquet")
    prefixes = ("000", "001", "002", "003", "600", "601", "603", "605")
    mask = instruments.instrument_id.str.startswith(("cn.xshe.", "cn.xshg.")) & instruments.instrument_id.str.split(".").str[-1].str.startswith(prefixes)
    instruments = instruments.loc[mask].sort_values("instrument_id").reset_index(drop=True)
    symbols = instruments.instrument_id.tolist()
    calendar = pd.read_parquet(root / "calendar.parquet")
    all_dates = sorted(calendar.loc[calendar.is_open.astype(bool) & calendar.exchange.isin(["XSHE", "XSHG"]), "date"].unique())
    dates = [d for d in all_dates if protocol["load_start"] <= d <= protocol["evaluation_end"]]
    if dates[-1] >= protocol["sealed_start"]:
        raise ValueError("Attempt to read sealed prices")
    fields = ["open", "high", "low", "close", "volume", "amount", "factor", "source_is_st", "source_trade_status"]
    filt = ((ds.field("date") >= dates[0]) & (ds.field("date") <= dates[-1]) & ds.field("instrument_id").isin(symbols))
    frame = ds.dataset(root / "bars.parquet").to_table(columns=["date", "instrument_id", *fields, "sequence_id", "label_sequence_id"], filter=filt).to_pandas()
    if frame.duplicated(["date", "instrument_id"]).any() or sorted(frame.date.unique()) != dates:
        raise ValueError("Duplicate bars or missing market sessions")
    ti = pd.Index(dates).get_indexer(frame.date)
    sj = pd.Index(symbols).get_indexer(frame.instrument_id)
    shape = (len(dates), len(symbols))
    arrays = {}
    for field in fields:
        values = pd.to_numeric(frame[field], errors="coerce").to_numpy(float)
        matrix = np.full(shape, np.nan)
        matrix[ti, sj] = values
        arrays[field] = matrix
    for field in ["sequence_id", "label_sequence_id"]:
        values, _ = pd.factorize(frame[field], sort=False)
        matrix = np.full(shape, -1, dtype=np.int32)
        matrix[ti, sj] = values
        arrays[field] = matrix
    listed = pd.Index(all_dates).searchsorted(instruments.listed_at.to_numpy())
    absolute_days = pd.Index(all_dates).get_indexer(dates)
    arrays["age"] = absolute_days[:, None] - listed[None, :]
    actions = pd.read_parquet(root / "actions.parquet")
    actions = actions.loc[actions.instrument_id.isin(symbols) & actions.ex_date.between(dates[0], dates[-1])]
    if actions.duplicated(["instrument_id", "ex_date"]).any() or (actions.split_ratio <= 0).any() or (actions.cash_per_share < 0).any():
        raise ValueError("Invalid corporate action input")
    action_map = defaultdict(list)
    date_index, symbol_index = {d: i for i, d in enumerate(dates)}, {s: j for j, s in enumerate(symbols)}
    outside = 0
    for action in actions.itertuples():
        if action.ex_date not in date_index:
            outside += 1
            continue
        action_map[date_index[action.ex_date]].append((symbol_index[action.instrument_id], float(action.split_ratio), float(action.cash_per_share)))
    profile = dict(created_at=utc_now(), rows=len(frame), symbols=len(symbols), observed_symbols=int(frame.instrument_id.nunique()),
                   first_price_read=min(frame.date), last_price_read=max(frame.date), sealed_price_rows_read=0,
                   source_flag_counts={str(k): int(v) for k, v in frame.groupby(["source_trade_status", "source_is_st"], dropna=False).size().items()},
                   delisted_metadata_symbols=int((instruments.delisted_at != "").sum()),
                   actions=len(actions), actions_outside_trading_calendar=outside,
                   declaration=manifest["declaration"], source_hashes=manifest["files"])
    if outside:
        raise ValueError("Corporate action date outside market calendar")
    save(output / "input-profile.json", profile)
    del frame
    return dates, symbols, instruments, arrays, action_map


def features(arrays, protocol):
    adjusted = pd.DataFrame(arrays["close"] * arrays["factor"])
    ret = adjusted.pct_change(fill_method=None)
    ma5 = adjusted.rolling(5).mean().to_numpy()
    ma60 = adjusted.rolling(60).mean().to_numpy()
    ma120 = adjusted.rolling(120).mean().to_numpy()
    vol60 = ret.rolling(60).std().to_numpy()
    adv20 = pd.DataFrame(arrays["amount"]).rolling(20).mean().to_numpy()
    mom = {days: adjusted.pct_change(days, fill_method=None).to_numpy() for days in [5, 20, 120]}
    # Require 121 consecutive observations inside the same feature sequence.
    seq = arrays["sequence_id"]
    run = np.zeros_like(seq)
    for i in range(1, len(run)):
        same = (seq[i] >= 0) & (seq[i] == seq[i-1]) & np.isfinite(adjusted.iloc[i].to_numpy())
        run[i] = np.where(same, run[i-1]+1, 0)
    eligible = ((arrays["age"] >= protocol["minimum_history_sessions"]) & (run >= 120)
                & (arrays["source_is_st"] == 0) & (arrays["source_trade_status"] == 1)
                & (adv20 >= protocol["minimum_adv20"]) & (arrays["volume"] > 0)
                & (adjusted.to_numpy() >= ma5) & np.isfinite(vol60) & (vol60 > 0))
    # Ascending column order is stable instrument-id tie breaking.
    scores = {"momentum20": mom[20], "trend120": mom[120] / np.maximum(vol60, 1e-12),
              "lowvol60": -vol60, "reversal5": -mom[5], "liquidity6": adv20}
    trend = (adjusted.to_numpy() > ma60) & (ma60 > ma120)
    rising = ma5 > np.roll(ma5, 1, axis=0)
    for name in scores:
        valid = eligible.copy()
        if name == "trend120":
            valid &= trend & (mom[120] > 0)
        elif name == "reversal5":
            valid &= rising
        scores[name] = np.where(valid & np.isfinite(scores[name]), scores[name], -np.inf)
    return scores, adv20


@dataclass
class Position:
    shares: float
    last_price: float
    bought: int
    lock_until: int
    source_sequence: int
    uncertain: bool = False


def statistics(nav, initial):
    values = np.r_[initial, nav.equity.to_numpy()]
    returns = values[1:] / values[:-1] - 1
    dd = values / np.maximum.accumulate(values) - 1
    total = values[-1] / initial - 1
    cagr = (values[-1] / initial) ** (252 / len(nav)) - 1
    mdd = -dd.min()
    years = {}
    for year, group in nav.assign(year=nav.date.str[:4], daily_return=returns).groupby("year"):
        years[year] = float(np.prod(1 + group.daily_return) - 1)
    return dict(net_return=float(total), profit_cny=float(values[-1] - initial), final_equity=float(values[-1]),
                cagr=float(cagr), max_drawdown=float(mdd), calmar=float(cagr/mdd) if mdd else None,
                annual_volatility=float(np.std(returns, ddof=1)*np.sqrt(252)),
                mean_exposure=float((nav.market_value/nav.equity).mean()), annual_returns=years,
                sessions=len(nav), first_date=nav.date.iloc[0], last_date=nav.date.iloc[-1],
                final_unpaid_dividends=float(nav.receivable.iloc[-1]),
                max_uncertain_value=float(nav.uncertain_value.max()),
                conservative_terminal_haircut_return=float((nav.equity.iloc[-1]-nav.uncertain_value.iloc[-1])/initial-1),
                stale_position_days=int(nav.stale.sum()), uncertainty_position_days=int(nav.uncertain_count.sum()))


def simulate(dates, symbols, instruments, arrays, actions, scores, adv20, protocol, scenario, name, output):
    start = int(np.searchsorted(dates, protocol["evaluation_start"]))
    positions = {}
    cash = float(protocol["initial_cash"])
    receivables = []
    fills, orders, events, nav, holdings = [], [], [], [], []
    history = {}
    uncertain_reserve = 0.
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
            amount = old_shares * dividend * .8
            if amount:
                due_date = str(date.fromisoformat(day) + timedelta(days=dividend_delay))
                receivables.append((due_date, amount, j))
                events.append(dict(date=day, instrument_id=symbols[j], kind="dividend_entitlement", amount=amount, gross=old_shares*dividend, due_date=due_date))
            pos.shares = float(np.floor(old_shares * split + 1e-8))
            pos.last_price = (pos.last_price-dividend)/split
            if split != 1:
                pos.lock_until = i+5
                events.append(dict(date=day, instrument_id=symbols[j], kind="split", amount=0., old_shares=old_shares, new_shares=pos.shares))
        # Account for an unresolved change even when the holding is sold today.
        for j, pos in positions.items():
            seq = int(arrays["label_sequence_id"][i, j])
            if seq >= 0 and pos.source_sequence >= 0 and seq != pos.source_sequence and not pos.uncertain:
                pos.uncertain = True
                uncertain_reserve += pos.shares*pos.last_price
                events.append(dict(date=day, instrument_id=symbols[j], kind="unresolved_label_barrier", amount=pos.shares*pos.last_price))
            if seq >= 0:
                pos.source_sequence = seq
        outstanding = []
        for payable, amount, j in receivables:
            if payable <= day:
                cash += amount
                events.append(dict(date=day, instrument_id=symbols[j], kind="dividend_paid", amount=amount))
            else:
                outstanding.append((payable, amount, j))
        receivables = outstanding

        def execute(j, side, budget=0., signal=None, reason="rebalance"):
            nonlocal cash
            order = dict(date=day, instrument_id=symbols[j], side=side, reason=reason, signal_date=dates[signal] if signal is not None else "", status="blocked")
            orders.append(order)
            opening = arrays["open"][i, j]
            if not np.isfinite(opening) or opening <= 0 or arrays["source_trade_status"][i, j] != 1 or not arrays["volume"][i, j] > 0:
                order["block_reason"] = "no_verified_trading_bar"
                return
            if arrays["source_is_st"][i, j] not in (0., 1.):
                order["block_reason"] = "unknown_risk_and_limit_status"
                return
            if side == "buy" and arrays["source_is_st"][i, j] != 0:
                order["block_reason"] = "st_or_unknown"
                return
            if side == "sell" and (i <= positions[j].bought or i < positions[j].lock_until):
                order["block_reason"] = "t_plus_or_new_share_lock"
                return
            previous = arrays["close"][i-1, j]
            if not np.isfinite(previous):
                order["block_reason"] = "prior_reference_missing"
                return
            split, dividend = day_actions.get(j, (1., 0.))
            reference = (previous-dividend)/split
            pct = .05 if arrays["source_is_st"][i, j] == 1 else .10
            upper = math.floor(reference*(1+pct)*100+.5)/100
            lower = math.floor(reference*(1-pct)*100+.5)/100
            slip = scenario["slippage_bps"]/10000
            price = opening*(1+slip if side == "buy" else 1-slip)
            if ((side == "buy" and (opening >= upper-1e-8 or price > upper))
                    or (side == "sell" and (opening <= lower+1e-8 or price < lower))):
                order["block_reason"] = "adverse_opening_limit_assumption"
                return
            capacity = adv20[i-1, j]*protocol["adv_participation"]
            if not np.isfinite(capacity) or capacity <= 0:
                order["block_reason"] = "prior_liquidity_missing"
                return
            if side == "buy":
                quantity = math.floor(min(budget, cash, capacity)/price/100)*100
                while quantity > 0 and quantity*price+sum(fees(quantity*price, day, side, scenario["commission_multiplier"])) > min(cash, budget)+1e-8:
                    quantity -= 100
            else:
                requested = positions[j].shares
                quantity = requested if capacity >= requested*price else math.floor(capacity/price/100)*100
            if quantity <= 0:
                order["block_reason"] = "insufficient_cash_lot_or_capacity"
                return
            notional = quantity*price
            commission, transfer, stamp = fees(notional, day, side, scenario["commission_multiplier"])
            cost = commission+transfer+stamp
            if side == "buy":
                cash -= notional+cost
                positions[j] = Position(quantity, opening, i, i+1, int(arrays["label_sequence_id"][i, j]))
            else:
                cash += notional-cost
                positions[j].shares -= quantity
                if positions[j].shares <= 1e-8:
                    del positions[j]
            if cash < -1e-7:
                raise AssertionError("Borrowed cash")
            order.update(status="filled", shares=quantity)
            fills.append(dict(date=day, instrument_id=symbols[j], side=side, shares=quantity, price=price, reference_open=opening,
                              notional=notional, commission=commission, transfer=transfer, stamp=stamp, fees=cost,
                              slippage=abs(price-opening)*quantity, reason=reason, signal_date=order["signal_date"]))

        for j in list(positions):
            if arrays["source_is_st"][i-1, j] != 0 or (delisted[j] and day >= delisted[j]):
                execute(j, "sell", reason="dated_risk_exit")
        if i in due:
            signal = i-1-scenario["delay_sessions"]
            row = scores[signal]
            ranked = np.argsort(-row, kind="stable")
            ranked = ranked[np.isfinite(row[ranked])]
            prior_positions, prior_equity = history.get(signal, (set(), float(protocol["initial_cash"])))
            target = [int(j) for j in ranked[:protocol["retention_buffer_rank"]] if j in prior_positions]
            target = target[:protocol["max_positions"]]
            for j in ranked:
                if len(target) >= protocol["max_positions"]:
                    break
                if j not in target:
                    target.append(int(j))
            for j in list(positions):
                if j not in target:
                    execute(j, "sell", signal=signal)
            # Exactly the frozen target list: blocked entries are not backfilled.
            for j in target:
                if j not in positions and len(positions) < protocol["max_positions"]:
                    execute(j, "buy", prior_equity*protocol["position_budget_fraction"], signal)
        stale = uncertain_count = 0
        uncertain_value = 0.
        for j, pos in positions.items():
            current = arrays["close"][i, j]
            if np.isfinite(current) and current > 0:
                pos.last_price = current
            else:
                stale += 1
            if pos.uncertain or not np.isfinite(current):
                uncertain_count += 1
                uncertain_value += pos.shares*pos.last_price
            holdings.append(dict(date=day, instrument_id=symbols[j], shares=pos.shares, mark=pos.last_price,
                                 value=pos.shares*pos.last_price, stale=not np.isfinite(current), uncertain=pos.uncertain))
        market_value = sum(p.shares*p.last_price for p in positions.values())
        receivable = sum(v[1] for v in receivables)
        equity = cash+market_value+receivable
        nav.append(dict(date=day, equity=equity, cash=cash, market_value=market_value, receivable=receivable,
                        positions=len(positions), stale=stale, uncertain_count=uncertain_count,
                        uncertain_value=uncertain_value, cumulative_uncertainty_reserve=uncertain_reserve))
        history[i] = (set(positions), equity)
    output.mkdir(parents=True)
    frames = {"nav": pd.DataFrame(nav), "fills": pd.DataFrame(fills), "orders": pd.DataFrame(orders),
              "cash_events": pd.DataFrame(events), "holdings": pd.DataFrame(holdings)}
    for key, frame in frames.items():
        frame.to_parquet(output/(key+".parquet"), index=False)
    result = statistics(frames["nav"], protocol["initial_cash"])
    result.update(strategy=name, scenario=scenario["name"], trades=len(fills),
                  completed_sales=sum(f["side"] == "sell" for f in fills),
                  commission=sum(f["commission"] for f in fills), stamp=sum(f["stamp"] for f in fills),
                  transfer=sum(f["transfer"] for f in fills), slippage=sum(f["slippage"] for f in fills),
                  minimum_fee_fills=int(sum(abs(f["commission"]-5.) < 1e-9 for f in fills)),
                  blocked_reasons=dict(Counter(v["block_reason"] for v in orders if v["status"] == "blocked")),
                  unresolved_barrier_events=sum(v["kind"] == "unresolved_label_barrier" for v in events),
                  cumulative_uncertainty_reserve=uncertain_reserve,
                  permanent_uncertainty_haircut_return=(nav[-1]["equity"]-uncertain_reserve)/protocol["initial_cash"]-1,
                  input_assumptions_not_exchange_execution_certification=True, formal_profit_qualified=False)
    save(output/"summary.json", result)
    return result, frames["nav"]


def bootstrap_difference(a, b, seed=20261013):
    # Paired calendar blocks; inferential output is exploratory across four candidates.
    ar = np.r_[a[0]/200000-1, a[1:]/a[:-1]-1]
    br = np.r_[b[0]/200000-1, b[1:]/b[:-1]-1]
    difference = ar-br
    rng = np.random.default_rng(seed)
    estimates = []
    n = len(difference)
    for _ in range(40):
        starts = rng.integers(0, n-20+1, size=(250, math.ceil(n/20)))
        indices = (starts[:, :, None]+np.arange(20)).reshape(250, -1)[:, :n]
        estimates.extend(difference[indices].mean(axis=1)*252)
    return dict(annualized_mean_daily_excess=float(difference.mean()*252),
                block20_ci95=[float(v) for v in np.quantile(estimates, [.025, .975])], bootstrap_repeats=10000,
                multiple_candidates=True, confirmatory_significance=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.root/"protocol.json").read_text())
    output = args.root/"results"
    output.mkdir()
    save(output/"run-start.json", dict(started_at=utc_now(), protocol_sha256=file_hash(args.root/"protocol.json"),
                                      implementation_sha256=file_hash(Path(__file__)), outcomes_read_before_rule_declaration=False))
    dates, symbols, instruments, arrays, actions = load_inputs(protocol, output)
    score, adv20 = features(arrays, protocol)
    save(output/"signal-coverage.json", {name: {"min": int(np.isfinite(v).sum(axis=1).min()),
         "evaluation_mean": float(np.isfinite(v[np.searchsorted(dates,protocol['evaluation_start']):]).sum(axis=1).mean())} for name, v in score.items()})
    results, daily = [], {}
    for scenario in protocol["scenarios"]:
        for name in score:
            item, nav = simulate(dates, symbols, instruments, arrays, actions, score[name], adv20, protocol,
                                 scenario, name, output/scenario["name"]/name)
            results.append(item)
            if scenario["name"] == "base":
                daily[name] = nav.equity.to_numpy()
            print(scenario["name"], name, json.dumps({k:item[k] for k in ["net_return","max_drawdown","trades","unresolved_barrier_events"]}), flush=True)
    save(output/"comparison.json", dict(completed_at=utc_now(), summaries=results,
         paired_diagnostics={name: bootstrap_difference(values, daily["liquidity6"]) for name, values in daily.items() if name != "liquidity6"},
         formal_profit_qualified=False, historical_research_returns_computed=True, untouched_holdout=False))
    pd.DataFrame([{k:v for k,v in row.items() if not isinstance(v,(dict,list))} for row in results]).to_csv(output/"comparison.csv", index=False)


if __name__ == "__main__":
    main()
