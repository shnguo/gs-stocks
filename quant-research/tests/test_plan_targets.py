from dataclasses import replace

import numpy as np
import pytest

from quant_research.plan_targets import plan_targets
from quant_research.price_strategy import TradeAssumptions, candidate_plans, trade_diagnostic


@pytest.mark.parametrize('stress', [1, 2])
def test_vectorized_targets_match_scalar_scenarios(stress):
    rng = np.random.default_rng(34)
    n = 300
    close = 10*np.exp(np.cumsum(rng.normal(0, .025, (n, 5)), axis=1))
    opening = close*np.exp(rng.normal(0, .015, (n, 5)))
    high = np.maximum(opening, close)*(1+rng.uniform(0, .045, (n, 5)))
    low = np.minimum(opening, close)*(1-rng.uniform(0, .045, (n, 5)))
    future = np.stack([opening, high, low, close], axis=2)
    valid = np.logical_and.accumulate(rng.random((n, 5)) > .03, axis=1)
    upper, lower = np.full((n, 5), np.nan), np.full((n, 5), np.nan)
    upper[::4], lower[::4] = 11, 9
    labels = dict(reference=np.full(n, 10.), future=future, valid=valid, upper=upper, lower=lower)
    a = TradeAssumptions()
    a = replace(a, commission=a.commission*stress, minimum_fee=a.minimum_fee*stress,
                sell_tax=a.sell_tax*stress, slippage_bps=a.slippage_bps*stress)
    targets = plan_targets(labels, a)
    for i in range(n):
        for j, plan in enumerate(candidate_plans(10., a)):
            result = trade_diagnostic(future[i], valid[i], plan, a, upper[i], lower[i])
            for vector, scalar in [('filled', 'filled'), ('scenario_order_net_return', 'net_return'),
                                   ('entry', 'buy_price'), ('exit', 'sell_price')]:
                if result[scalar] is None:
                    assert np.isnan(targets[vector][i, j])
                else:
                    assert targets[vector][i, j] == pytest.approx(float(result[scalar]))
            assert targets['ambiguous'][i, j] == result['ambiguous']
            assert targets['exit_day'][i, j] == (result['exit_day'] or 0)
    assert not targets['execution_verified'].any()


def test_unknown_and_unfilled_and_ambiguous_remain_distinct():
    future = np.tile([[10., 11., 9., 10.]], (3, 5, 1))
    future[1, 0] = [11., 11., 11., 11.]
    valid = np.ones((3, 5), bool)
    valid[0, 0] = False
    result = plan_targets(dict(reference=np.full(3, 10.), future=future, valid=valid))
    assert np.isnan(result['filled'][0]).all()
    assert np.isnan(result['scenario_order_net_return'][0]).all()
    assert (result['filled'][1] == 0).all()
    assert (result['scenario_order_net_return'][1] == 0).all()
    assert np.isnan(result['conditional_net_return'][1]).all()
    assert result['ambiguous'][2].all()
    assert np.isfinite(result['scenario_order_net_return'][2]).all()
    assert np.isnan(result['conditional_net_return'][2]).all()
