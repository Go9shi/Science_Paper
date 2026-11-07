"""Causal daily close-to-close risk estimates, indexed by execution session."""
from __future__ import annotations

import numpy as np
import pandas as pd


def historical_volatility(
    log_returns: pd.Series | pd.DataFrame,
    window: int = 20,
    annualization: int = 252,
) -> pd.Series | pd.DataFrame:
    """sqrt(K * mean(r**2)) over [t-window, t-1], available before open t.

    Uncentered daily estimator, not intraday realized volatility or sample std.
    Requires an ordered complete session grid. Missing returns propagate through
    the full window; no fill and no partial-window estimates are performed.
    """
    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise ValueError("window must be an integer >= 2")
    if isinstance(annualization, bool) or not isinstance(annualization, int) or annualization <= 0:
        raise ValueError("annualization must be a positive integer")
    if not isinstance(log_returns.index, pd.DatetimeIndex):
        raise ValueError("Returns must have a DatetimeIndex")
    if log_returns.index.has_duplicates or not log_returns.index.is_monotonic_increasing:
        raise ValueError("Returns must have unique chronological dates")
    values = log_returns.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Infinite returns are not allowed")
    return np.sqrt(
        annualization * log_returns.pow(2).shift(1).rolling(window, min_periods=window).mean()
    )


def fit_sigma_ref(train_sigma: pd.Series, *, train_end_exclusive: str) -> float:
    """Fit a frozen positive median on an explicitly bounded training slice."""
    if train_sigma.empty or not isinstance(train_sigma.index, pd.DatetimeIndex):
        raise ValueError("Provide a non-empty training volatility series")
    if train_sigma.index.has_duplicates or not train_sigma.index.is_monotonic_increasing:
        raise ValueError("Training dates must be unique and chronological")
    if (train_sigma.index >= pd.Timestamp(train_end_exclusive)).any():
        raise ValueError("sigma_ref input contains dates outside training")
    if not np.isfinite(train_sigma.to_numpy(dtype=float)).all() or (train_sigma < 0).any():
        raise ValueError("Training volatility must be finite and non-negative")
    reference = float(train_sigma.median())
    if reference <= 0:
        raise ValueError("sigma_ref is zero; do not invent an epsilon from test data")
    return reference
