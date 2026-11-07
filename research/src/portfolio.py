"""Self-financing fractional-unit execution, common to every reward mode."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def action_to_weights(action: np.ndarray, n_assets: int) -> np.ndarray:
    """Box[-1,1] -> nonnegative scores -> simplex; cash is the last entry.

    Corners allow exact zero/full allocations. All -1 maps to all cash.
    Finite out-of-range actions are clipped; NaN/inf and bad shapes are errors.
    """
    values = np.asarray(action, dtype=np.float64)
    if values.shape != (n_assets + 1,) or not np.isfinite(values).all():
        raise ValueError(f"Action must contain {n_assets + 1} finite values")
    scores = (np.clip(values, -1., 1.) + 1.) / 2.
    if scores.sum() == 0:
        scores[-1] = 1.
    return scores / scores.sum()


@dataclass(frozen=True)
class Execution:
    units: np.ndarray
    cash: float
    nav_before: float
    nav_after: float
    signed_trade_notional: np.ndarray
    traded_notional: float
    commission: float
    turnover: float


def rebalance(
    units: np.ndarray, cash: float, prices: np.ndarray,
    target_weights: np.ndarray, commission_rate: float,
) -> Execution:
    """Target post-fee weights without borrowing cash to pay proportional fees.

    Solve B + c*sum(abs(w_i*B - h_i)) = V_pre by bisection.
    Cash is computed from actual fractional-unit trades, never invented/clamped.
    c is charged on each bought/sold dollar of risky assets, excluding cash.
    """
    units = np.asarray(units, dtype=np.float64).copy()
    prices = np.asarray(prices, dtype=np.float64)
    weights = np.asarray(target_weights, dtype=np.float64)
    if (units.ndim != 1 or prices.shape != units.shape or weights.shape != (len(units) + 1,)
            or not np.isfinite(units).all() or not np.isfinite(prices).all()
            or not np.isfinite(weights).all() or (units < 0).any() or (prices <= 0).any()
            or (weights < 0).any() or not np.isclose(weights.sum(), 1., rtol=0, atol=1e-12)):
        raise ValueError("Invalid positions, prices or simplex weights")
    if not np.isfinite(cash) or cash < 0 or not np.isfinite(commission_rate) or not 0 <= commission_rate < 1:
        raise ValueError("Require nonnegative cash and commission_rate in [0,1)")
    holdings = units * prices
    nav = math.fsum([float(cash), *holdings.tolist()])
    if not np.isfinite(nav) or nav <= 0:
        raise ValueError("Pre-trade NAV must be finite and positive")
    current = np.r_[holdings, cash] / nav
    if np.allclose(current, weights, rtol=0, atol=1e-12):
        return Execution(units, float(cash), nav, nav, np.zeros_like(units), 0., 0., 0.)

    def allocate(budget):
        new_units = weights[:-1] * budget / prices
        trades = (new_units - units) * prices
        notional = math.fsum(np.abs(trades).tolist())
        fee = commission_rate * notional
        remaining = math.fsum([float(cash), *(-trades).tolist(), -fee])
        return new_units, trades, notional, fee, remaining

    low, high = 0., nav
    # Take the feasible lower endpoint to leave nonnegative cash even at 100% risk.
    for _ in range(80):
        middle = (low + high) / 2.
        *_, remaining = allocate(middle)
        if remaining >= weights[-1] * middle:
            low = middle
        else:
            high = middle
    new_units, trades, notional, fee, remaining = allocate(low)
    if remaining < 0 or (new_units < 0).any():
        raise FloatingPointError("Self-financing execution produced an infeasible allocation")
    after = math.fsum([remaining, *(new_units * prices).tolist()])
    if not np.isclose(after, nav - fee, rtol=2e-13, atol=1e-12 * nav):
        raise FloatingPointError("Execution NAV does not reconcile with commission")
    return Execution(new_units, remaining, nav, after, trades, notional, fee, notional / nav)
