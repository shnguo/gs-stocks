"""Research labels measure gross holder wealth, without dividend reinvestment."""
from __future__ import annotations

import numpy as np
import pandas as pd


def wealth_state(dates: pd.Index, actions: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return shares and cumulative cash entitlements per initial share.

    Entitlements are recognized on the ex date, including unpaid receivables.
    They are not spendable cash, reinvestment or an execution simulation. An
    investor entering at the ex-date open does not earn that day's dividend.
    Only verified cash/split distributions belong in this input; rights and
    restructurings require separate economic terms and remain label barriers.
    """
    splits = np.ones(len(dates), dtype=np.float64)
    cash = np.zeros(len(dates), dtype=np.float64)
    if not actions.empty:
        if actions.ex_date.duplicated().any():
            raise ValueError("Aggregate same-day actions before wealth calculation")
        positions = dates.get_indexer(actions.ex_date)
        if (positions < 0).any():
            raise ValueError("Action ex date is outside the market calendar")
        splits[positions] = actions.split_ratio.to_numpy(dtype=np.float64)
        cash[positions] = actions.cash_per_share.to_numpy(dtype=np.float64)
    if (not np.isfinite(splits).all() or not np.isfinite(cash).all()
            or (splits <= 0).any() or (cash < 0).any()):
        raise ValueError("Invalid holder cash/split entitlement")
    shares = np.cumprod(splits)
    prior_shares = np.r_[1., shares[:-1]]
    entitlements = np.cumsum(cash * prior_shares)
    return shares, entitlements


def gross_wealth_return(entry_open: float, exit_open: float, entry: int, exit: int,
                       shares: np.ndarray, entitlements: np.ndarray) -> float:
    if not 0 <= entry < exit < len(shares):
        raise ValueError("Invalid wealth holding interval")
    if not np.isfinite([entry_open, exit_open]).all() or min(entry_open, exit_open) <= 0:
        raise ValueError("Invalid wealth endpoint")
    terminal_shares = shares[exit] / shares[entry]
    dividend_receivable = (entitlements[exit] - entitlements[entry]) / shares[entry]
    return (exit_open * terminal_shares + dividend_receivable) / entry_open - 1
