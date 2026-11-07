"""Financial performance from net NAV/returns, never from shaped reward."""
import math

import numpy as np
import pandas as pd


def financial_metrics(records: pd.DataFrame, initial_capital: float, *, annualization=252, risk_free_annual=0.) -> dict:
    required = ["net_return", "portfolio_value", "turnover", "commission"]
    if records.empty or not all(c in records for c in required):
        raise ValueError("Complete non-empty financial records required")
    a = records[required].to_numpy(dtype=float)
    if (not np.isfinite(a).all() or not all(math.isfinite(x) for x in (initial_capital, annualization, risk_free_annual))
            or initial_capital <= 0 or annualization <= 0 or risk_free_annual <= -1):
        raise ValueError("Invalid financial inputs")
    returns = records["net_return"].to_numpy(dtype=float)
    nav = records["portfolio_value"].to_numpy(dtype=float)
    if (returns <= -1).any() or (nav <= 0).any() or (records[["turnover", "commission"]] < 0).any().any():
        raise ValueError("Nonpositive NAV or negative cost/turnover")
    np.testing.assert_allclose(initial_capital * np.cumprod(1 + returns), nav, rtol=1e-10, atol=1e-8)
    daily_rf = (1 + risk_free_annual) ** (1 / annualization) - 1
    excess = returns - daily_rf
    deviation = float(np.std(excess, ddof=1)) if len(returns) >= 2 else 0.
    path = np.r_[initial_capital, nav]
    total = float(nav[-1] / initial_capital - 1)
    result = {
        "net_sharpe": float(np.mean(excess) / deviation * math.sqrt(annualization)) if deviation > 1e-15 else None,
        "sharpe_status": "defined" if deviation > 1e-15 else "undefined_zero_variance_or_short_sample",
        "cagr": float((nav[-1] / initial_capital) ** (annualization / len(returns)) - 1),
        "maximum_drawdown": float(np.max(1 - path / np.maximum.accumulate(path))),
        "annualized_volatility": float(np.std(returns, ddof=1) * math.sqrt(annualization)) if len(returns) >= 2 else None,
        "portfolio_turnover_mean_daily": float(records["turnover"].mean()),
        "portfolio_turnover_cumulative": float(records["turnover"].sum()),
        "portfolio_turnover_annualized": float(records["turnover"].mean() * annualization),
        "cumulative_net_return": total, "total_commissions": float(records["commission"].sum()),
        "observations": len(returns), "annualization": annualization, "risk_free_annual": risk_free_annual,
    }
    if any(isinstance(x, float) and not math.isfinite(x) for x in result.values()):
        raise FloatingPointError("Nonfinite financial metric")
    return result
