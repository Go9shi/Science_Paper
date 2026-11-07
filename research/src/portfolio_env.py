"""Gymnasium portfolio environment: pre-open decision, close-to-close reward."""
from __future__ import annotations

from dataclasses import dataclass
import math

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .environment_data import MarketData, PRICE_MODEL
from .portfolio import action_to_weights, rebalance
from .regularization import penalty_coefficient


@dataclass(frozen=True)
class EnvConfig:
    initial_capital: float = 1_000_000.
    commission_bps: float = 5.
    penalty_mode: str = "none"
    lambda_base: float = .001
    adaptive_ratio_cap: float = 5.
    max_episode_steps: int | None = None
    adaptive_alpha: float = 1.
    adaptive_ratio_floor: float = 1e-6
    lambda_min: float = 0.
    lambda_max: float | None = None

    def validate(self):
        if not np.isfinite(self.initial_capital) or self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if not np.isfinite(self.commission_bps) or not 0 <= self.commission_bps < 10000:
            raise ValueError("commission_bps must be in [0,10000)")
        if self.penalty_mode not in {"none", "fixed", "adaptive"}:
            raise ValueError("Unknown penalty mode")
        if not np.isfinite(self.lambda_base) or self.lambda_base < 0:
            raise ValueError("lambda_base must be nonnegative")
        if not np.isfinite(self.adaptive_ratio_cap) or self.adaptive_ratio_cap <= 0:
            raise ValueError("adaptive_ratio_cap must be positive")
        if self.max_episode_steps is not None and (isinstance(self.max_episode_steps, bool)
                or not isinstance(self.max_episode_steps, int) or self.max_episode_steps < 1):
            raise ValueError("max_episode_steps must be a positive integer or None")
        penalty_coefficient(self.penalty_mode, self.lambda_base, 0., 1., alpha=self.adaptive_alpha,
                            ratio_cap=self.adaptive_ratio_cap, ratio_floor=self.adaptive_ratio_floor,
                            lambda_min=self.lambda_min, lambda_max=self.lambda_max)


class PortfolioEnv(gym.Env):
    """One shared financial mechanism; only reward lambda changes across modes.

    No raw Open[t], post-open weights or forward reward enters observation t.
    Episode ending is a known finite horizon (remaining time is observed).
    """
    metadata = {"render_modes": []}

    def __init__(self, data: MarketData, config: EnvConfig | None = None):
        super().__init__()
        self.config = config or EnvConfig()
        self.config.validate()
        self.assets = data.assets
        self.n_assets = len(self.assets)
        self.dates = data.features.index.copy()
        self._locations = data.adjusted_close.index.get_indexer(self.dates)
        self._price_dates = data.adjusted_close.index.copy()
        self._open = data.adjusted_open.to_numpy(dtype=np.float64, copy=True)
        self._close = data.adjusted_close.to_numpy(dtype=np.float64, copy=True)
        self._features = data.features.to_numpy(dtype=np.float64, copy=True)
        self._sigma = data.features[data.sigma_column].to_numpy(dtype=np.float64, copy=True)
        self.sigma_ref = float(data.sigma_ref)
        self.feature_names = tuple(data.features.columns)
        for array in (self._open, self._close, self._features, self._sigma, self._locations):
            array.flags.writeable = False
        self.action_space = spaces.Box(-1., 1., shape=(self.n_assets + 1,), dtype=np.float32)
        # Market features, prior-close portfolio weights, NAV/initial NAV, time left.
        self.observation_space = spaces.Box(-np.inf, np.inf,
            shape=(len(self.feature_names) + self.n_assets + 3,), dtype=np.float32)
        self._done = True

    def _weights(self, close_prices):
        values = np.r_[self.units * close_prices, self.cash]
        return values / math.fsum(values.tolist())

    def _observation(self):
        if self._cursor < len(self.dates):
            loc = self._locations[self._cursor] - 1
            market = self._features[self._cursor]
            remaining = (len(self.dates) - self._cursor) / len(self.dates)
        else:
            loc = self._locations[-1]
            market = np.zeros(len(self.feature_names))
            remaining = 0.
        obs = np.r_[market, self._weights(self._close[loc]),
                    self.nav / self.config.initial_capital, remaining].astype(np.float32)
        if not np.isfinite(obs).all():
            raise FloatingPointError("Observation exceeds finite float32 range")
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        options = options or {}
        if set(options) - {"start_index"}:
            raise ValueError("Only start_index is a supported reset option")
        start = options.get("start_index", 0)
        if isinstance(start, bool) or not isinstance(start, int) or not 0 <= start < len(self.dates):
            raise ValueError("start_index outside this split")
        self._cursor = start
        self._steps = 0
        self._done = False
        self.units = np.zeros(self.n_assets, dtype=np.float64)
        self.cash = float(self.config.initial_capital)
        self.nav = self.cash
        return self._observation(), {
            "decision_date": str(self.dates[start].date()),
            "data_as_of": str(self._price_dates[self._locations[start] - 1].date()),
            "portfolio_value": self.nav, "weights": np.r_[self.units, 1.],
            "price_model": PRICE_MODEL,
        }

    def step(self, action):
        if self._done:
            raise RuntimeError("Episode is done; call reset before step")
        target = action_to_weights(action, self.n_assets)
        cursor = self._cursor
        loc = self._locations[cursor]
        previous_nav = self.nav  # Cash and old units marked at Close[t-1].
        old_units = self.units.copy()
        old_cash = self.cash
        execution = rebalance(old_units, old_cash, self._open[loc], target,
                              self.config.commission_bps / 10000.)
        close_nav = math.fsum([execution.cash, *(execution.units * self._close[loc]).tolist()])
        if not np.isfinite(close_nav) or close_nav <= 0:
            raise FloatingPointError("Nonpositive/nonfinite close NAV under the price model")
        # Zero-fee allocation to the same requested target, same overnight positions.
        gross_close = execution.nav_before * float(
            target[-1] + np.dot(target[:-1], self._close[loc] / self._open[loc]))
        net_log = math.log(close_nav / previous_nav)
        gross_log = math.log(gross_close / previous_nav)
        sigma = float(self._sigma[cursor])
        coefficient = penalty_coefficient(self.config.penalty_mode, self.config.lambda_base,
            sigma, self.sigma_ref, alpha=self.config.adaptive_alpha,
            ratio_cap=self.config.adaptive_ratio_cap, ratio_floor=self.config.adaptive_ratio_floor,
            lambda_min=self.config.lambda_min, lambda_max=self.config.lambda_max)
        penalty = coefficient * execution.turnover
        reward = net_log - penalty
        self.units, self.cash, self.nav = execution.units, execution.cash, close_nav
        self._cursor += 1
        self._steps += 1
        terminated = self._cursor == len(self.dates)
        truncated = bool(not terminated and self.config.max_episode_steps is not None
                         and self._steps >= self.config.max_episode_steps)
        self._done = terminated or truncated
        info = {
            "date": str(self.dates[cursor].date()),
            "data_as_of": str(self._price_dates[loc - 1].date()),
            "price_model": PRICE_MODEL, "portfolio_value": close_nav,
            "nav_previous_close": previous_nav, "nav_pre_trade_open": execution.nav_before,
            "nav_post_trade_open": execution.nav_after, "nav_close": close_nav,
            "counterfactual_gross_nav_close": gross_close,
            "gross_return": gross_close / previous_nav - 1., "net_return": close_nav / previous_nav - 1.,
            "r_gross_log": gross_log, "r_net_log": net_log,
            "commission": execution.commission, "commission_rate": self.config.commission_bps / 10000.,
            "turnover": execution.turnover, "traded_notional": execution.traded_notional,
            "sigma_t": sigma, "lambda_t": coefficient, "regularization_penalty": penalty,
            "reward": reward, "target_weights": target.copy(),
            "pre_trade_weights": np.r_[old_units * self._open[loc], old_cash] / execution.nav_before,
            "weights": self._weights(self._close[loc]).copy(),
            "positions_synthetic_units": self.units.copy(), "cash": self.cash,
            "signed_trade_notional": execution.signed_trade_notional.copy(),
            "execution_prices_synthetic": self._open[loc].copy(),
            "close_prices_synthetic": self._close[loc].copy(),
            "overnight_pnl": execution.nav_before - previous_nav,
            "intraday_pnl": close_nav - execution.nav_after,
        }
        return self._observation(), float(reward), bool(terminated), truncated, info
