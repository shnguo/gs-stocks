"""Vectorized five-session plan outcomes for conditional-return research.

Daily bars define conservative scenarios, not verified exchange fills. Missing
outcomes remain NaN; known unfilled orders alone have zero order-level return.
"""
import numpy as np

from quant_research.price_strategy import GRID, TradeAssumptions


def plan_targets(labels, assumptions=TradeAssumptions()):
    n = len(labels['reference'])
    ref = np.asarray(labels['reference'], float)
    future = np.asarray(labels['future'], float)
    valid = np.asarray(labels['valid'], bool)
    if future.shape != (n, 5, 4) or valid.shape != (n, 5):
        raise ValueError('Expected five OHLC sessions')
    if not np.isfinite(ref).all() or (ref <= 0).any():
        raise ValueError('Invalid reference prices')
    grid = np.array(GRID)
    tick = assumptions.tick
    buy_limit = np.floor(ref[:, None]*(1+grid[:, 0])/tick+1e-9)*tick
    target = np.ceil(buy_limit*(1+grid[:, 1])/tick-1e-9)*tick
    stop = np.floor(buy_limit*(1-grid[:, 2])/tick+1e-9)*tick
    buy_limit, target, stop = [np.round(a, 8) for a in (buy_limit, target, stop)]
    if (stop <= 0).any() or (target <= buy_limit).any() or (stop >= buy_limit).any():
        raise ValueError('Invalid tick-rounded plan')
    upper = np.asarray(labels.get('upper', np.full((n, 5), np.nan)), float)
    lower = np.asarray(labels.get('lower', np.full((n, 5), np.nan)), float)
    if upper.shape != (n, 5) or lower.shape != (n, 5):
        raise ValueError('Invalid price-limit axes')
    shape = buy_limit.shape
    filled, net, entry, exit_price = [np.full(shape, np.nan) for _ in range(4)]
    exit_day = np.zeros(shape, int)
    ambiguous = np.zeros(shape, bool)
    slip = assumptions.slippage_bps/10000
    o, h, low, c = [future[:, 0, f, None] for f in range(4)]
    available = np.broadcast_to(valid[:, 0, None], shape)
    blocked = np.isfinite(upper[:, 0, None]) & ((o >= upper[:, 0, None]) | (buy_limit > upper[:, 0, None]))
    at_open = (o <= buy_limit) & (o*(1+slip) <= buy_limit)
    touch = low < buy_limit-tick/2
    enter = available & ~blocked & (at_open | touch)
    filled[available] = 0
    filled[enter] = 1
    net[available & ~enter] = 0
    entry[enter] = np.where(at_open, o*(1+slip), buy_limit)[enter]
    active = enter.copy()
    stopped = np.zeros(shape, bool)
    for d in range(1, 5):
        active &= valid[:, d, None]
        if d < assumptions.t_plus:
            continue
        o, h, low, c = [future[:, d, f, None] for f in range(4)]
        previous = stopped.copy()
        stopped |= low <= stop
        blocked = np.isfinite(lower[:, d, None]) & (o <= lower[:, d, None])
        eligible = active & ~blocked
        sell = np.full(shape, np.nan)
        hit_stop = eligible & stopped
        sell[hit_stop] = (np.minimum(o, stop)*(1-slip))[hit_stop]
        ambiguous |= hit_stop & (h > target) & (o < target)
        opening_target = hit_stop & ~previous & (o*(1-slip) >= target)
        sell[opening_target] = target[opening_target]
        ambiguous[opening_target] = False
        hit_target = eligible & ~stopped & ((o*(1-slip) >= target) | (h > target+tick/2))
        sell[hit_target] = target[hit_target]
        if d == 4:
            time_exit = eligible & ~stopped & ~hit_target
            sell[time_exit] = np.broadcast_to(c*(1-slip), shape)[time_exit]
        close = eligible & np.isfinite(sell) & ~(np.isfinite(lower[:, d, None]) & (sell <= lower[:, d, None]))
        proceeds = assumptions.notional*sell[close]/entry[close]
        entry_fee = max(assumptions.minimum_fee, assumptions.notional*assumptions.commission)
        exit_fee = np.maximum(assumptions.minimum_fee, proceeds*assumptions.commission)
        net[close] = (proceeds-exit_fee-proceeds*assumptions.sell_tax-assumptions.notional-entry_fee)/(assumptions.notional+entry_fee)
        exit_price[close], exit_day[close] = sell[close], d+1
        active[close] = False
    # Main learning target excludes bars with unresolved intraday target/stop order.
    conditional = np.where((filled == 1) & ~ambiguous, net, np.nan)
    probability_loss = np.where(np.isfinite(conditional), (conditional < 0).astype(float), np.nan)
    loss_magnitude = np.where(np.isfinite(conditional), np.maximum(-conditional, 0), np.nan)
    return {'buy_limit': buy_limit, 'take_profit': target, 'stop': stop,
            'filled': filled, 'scenario_order_net_return': net,
            'conditional_net_return': conditional, 'conditional_loss': probability_loss,
            'conditional_downside': loss_magnitude, 'entry': entry, 'exit': exit_price,
            'exit_day': exit_day, 'ambiguous': ambiguous,
            'execution_verified': np.zeros(shape, bool)}
