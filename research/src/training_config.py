"""Strict pilot configuration; no test split or unregistered PPO alterations."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

from .portfolio_env import EnvConfig


@dataclass(frozen=True)
class TrainingConfig:
    run_kind: str = "smoke"
    learning_rate: float = 3e-4
    n_steps: int = 256
    batch_size: int = 64
    n_epochs: int = 5
    gamma: float = .99
    gae_lambda: float = .95
    clip_range: float = .2
    ent_coef: float = 0.
    vf_coef: float = .5
    total_timesteps: int = 4096
    random_seed: int = 42
    reward_mode: str = "none"
    lambda_0: float = .001
    alpha: float = .5
    lambda_min: float = 0.
    lambda_max: float = .01
    ratio_cap: float = 5.
    ratio_floor: float = 1e-6
    initial_capital: float = 1000.
    commission_bps: float = 5.
    checkpoint_steps: int = 1024
    policy_layers: tuple[int, ...] = (64, 64)
    normalize_observations: bool = True
    device: str = "cpu"
    evaluation_split: str = "validation"
    annualization: int = 252
    risk_free_annual: float = 0.

    def env_config(self, mode=None):
        return EnvConfig(initial_capital=self.initial_capital, commission_bps=self.commission_bps,
            penalty_mode=mode or self.reward_mode, lambda_base=self.lambda_0,
            adaptive_alpha=self.alpha, adaptive_ratio_cap=self.ratio_cap,
            adaptive_ratio_floor=self.ratio_floor, lambda_min=self.lambda_min, lambda_max=self.lambda_max)

    def validate(self):
        if self.run_kind not in {"smoke", "pilot"} or self.reward_mode not in {"none", "fixed", "adaptive"}:
            raise ValueError("Unknown run kind or reward mode")
        if self.evaluation_split != "validation":
            raise ValueError("Stage 3 evaluation is validation-only; test is forbidden")
        for name in ("n_steps", "batch_size", "n_epochs", "total_timesteps", "checkpoint_steps", "annualization"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (self.n_steps < 2 or self.batch_size < 2 or self.n_steps % self.batch_size
                or self.total_timesteps % self.n_steps or self.checkpoint_steps % self.n_steps):
            raise ValueError("Use complete rollouts, divisible minibatches and rollout-aligned checkpoint frequency")
        if self.run_kind == "smoke" and not self.n_steps <= self.total_timesteps <= 8192:
            raise ValueError("Smoke budget must be between one rollout and 8192 steps")
        if self.run_kind == "pilot" and self.total_timesteps > 262144:
            raise ValueError("Stage 3 pilot budget is capped; large searches are out of scope")
        if not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool) or not 0 <= self.random_seed < 2**32:
            raise ValueError("random_seed outside supported range")
        for name in ("learning_rate", "gamma", "gae_lambda", "clip_range", "ent_coef", "vf_coef", "risk_free_annual"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if (self.learning_rate <= 0 or not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1
                or not 0 < self.clip_range < 1 or self.ent_coef < 0 or self.vf_coef < 0 or self.risk_free_annual <= -1):
            raise ValueError("Invalid PPO/metric hyperparameter")
        if self.device not in {"cpu", "cuda"} or not self.policy_layers or any(type(n) is not int or n < 1 for n in self.policy_layers):
            raise ValueError("Use cpu/cuda and positive policy layer sizes")
        if type(self.normalize_observations) is not bool:
            raise ValueError("normalize_observations must be a JSON boolean")
        self.env_config().validate()

    def to_dict(self):
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_dict(cls, payload):
        payload = dict(payload)
        if "policy_layers" in payload:
            payload["policy_layers"] = tuple(payload["policy_layers"])
        config = cls(**payload)
        config.validate()
        return config


def load_training_config(path: Path) -> TrainingConfig:
    return TrainingConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
