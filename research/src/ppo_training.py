"""Controlled SB3 PPO training and validation, to be executed in Google Colab."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan, VecNormalize

from .data import sha256_file, write_json
from .financial_metrics import financial_metrics
from .portfolio_env import PortfolioEnv
from .training_config import TrainingConfig
from .validation import verify_file_map


FINANCIAL_FIELDS = ("date", "data_as_of", "portfolio_value", "nav_previous_close", "net_return",
    "r_net_log", "gross_return", "r_gross_log", "commission", "traded_notional", "turnover",
    "sigma_t", "lambda_t", "regularization_penalty", "reward", "cash")
CORE_LOGS = {"train/loss", "train/policy_gradient_loss", "train/value_loss", "train/entropy_loss", "train/approx_kl"}


def torch_runtime_info():
    return {"intra_op_threads": torch.get_num_threads(),
            "inter_op_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "mkldnn_enabled": torch.backends.mkldnn.enabled}


def first_step_replay_diagnostic(run_directory: Path, validation_data):
    """Separate policy inference differences from execution of the saved action.

    Diagnostic only: no tolerances, no training, no change to original artifacts.
    A matching first step does not establish full-episode equivalence.
    """
    root = Path(run_directory)
    original = pd.read_csv(root / "validation_financials.csv", float_precision="round_trip",
                           keep_default_na=False).iloc[0]
    model, vector, frozen, _ = load_saved_policy(root, validation_data)
    try:
        observation = vector.reset()
        predicted, _ = model.predict(observation, deterministic=True)
        saved = np.asarray(json.loads(original["action_json"]), dtype=model.action_space.dtype)
        _, _, _, infos = vector.step(saved.reshape(1, -1))
        info = infos[0]
        return {"source_run": root.name, "date": str(original["date"]),
                "torch_runtime": torch_runtime_info(),
                "cpu_capability": getattr(torch.backends.cpu, "get_cpu_capability", lambda: "unknown")(),
                "torch_build_config": torch.__config__.show(),
                "matmul_precision": torch.get_float32_matmul_precision(),
                "policy_dtype": str(next(model.policy.parameters()).dtype),
                "normalized_initial_observation": observation.tolist(),
                "saved_action": saved.tolist(), "predicted_action": predicted[0].tolist(),
                "first_action_exact": bool(np.array_equal(saved, predicted[0])),
                "saved_nav": float(original["portfolio_value"]),
                "nav_using_saved_action": float(info["portfolio_value"]),
                "saved_action_execution_exact": bool(info["date"] == original["date"]
                    and info["portfolio_value"] == float(original["portfolio_value"])),
                "normalization_frozen": not frozen.training and not frozen.norm_reward,
                "test_used": False}
    finally:
        vector.close()


def configure_torch_runtime():
    """Apply the same numerical settings before training AND checkpoint inference.

    In particular, reload_only does not call build_ppo, so it cannot rely on
    process-global settings left over by training or by the synthetic tests.
    """
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return torch_runtime_info()


def _require_split(data, split):
    if data.provenance.get("split") != split:
        raise ValueError(f"This operation requires {split} data; test/unknown data are forbidden")


def _contract(data):
    return {"assets": list(data.assets), "feature_names": list(data.features.columns), "sigma_ref": data.sigma_ref,
            **{key: data.provenance.get(key) for key in (
                "stage1_metadata_sha256", "snapshot_id", "snapshot_manifest_sha256", "price_model")}}


def make_vector_env(data, config, *, training, stats_path=None, mode=None):
    base = DummyVecEnv([lambda: Monitor(PortfolioEnv(data, config.env_config(mode)))])
    normalizer = (VecNormalize.load(str(stats_path), base) if stats_path is not None else
                  VecNormalize(base, norm_obs=config.normalize_observations, norm_reward=False,
                               clip_obs=10., gamma=config.gamma))
    normalizer.training = training
    normalizer.norm_reward = False
    return VecCheckNan(normalizer, raise_exception=True, check_inf=True), normalizer


def build_ppo(data, config: TrainingConfig):
    _require_split(data, "train")
    config.validate()
    configure_torch_runtime()
    random.seed(config.random_seed); np.random.seed(config.random_seed); torch.manual_seed(config.random_seed)
    if config.device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA explicitly requested but unavailable; CPU fallback is not silent")
        torch.cuda.manual_seed_all(config.random_seed)
    vector, normalizer = make_vector_env(data, config, training=True)
    model = PPO("MlpPolicy", vector, learning_rate=config.learning_rate, n_steps=config.n_steps,
        batch_size=config.batch_size, n_epochs=config.n_epochs, gamma=config.gamma, gae_lambda=config.gae_lambda,
        clip_range=config.clip_range, ent_coef=config.ent_coef, vf_coef=config.vf_coef,
        seed=config.random_seed, device=config.device, verbose=0,
        policy_kwargs={"net_arch": {"pi": list(config.policy_layers), "vf": list(config.policy_layers)},
                       "activation_fn": torch.nn.Tanh})
    return model, vector, normalizer


def save_checkpoint(model, normalizer, config, directory: Path, contract, *, phase):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    model.save(directory / "model")
    normalizer.save(str(directory / "vecnormalize.pkl"))
    write_json(directory / "config.json", config.to_dict())
    write_json(directory / "checkpoint.json", {
        "status": "complete", "steps": int(model.num_timesteps), "phase": phase,
        "torch_runtime": torch_runtime_info(),
        "data_contract": contract,
        "sha256": {name: sha256_file(directory / name) for name in ("model.zip", "vecnormalize.pkl", "config.json")},
    })


def _normalization_digest(normalizer):
    parts = []
    for name in ("obs_rms", "ret_rms"):
        stats = getattr(normalizer, name, None)
        if stats is not None:
            for field in ("mean", "var", "count"):
                parts.append(np.asarray(getattr(stats, field)).tobytes())
    return hashlib.sha256(b"".join(parts)).hexdigest()


def load_saved_policy(run_directory: Path, validation_data, *, checkpoint=None, reward_mode=None):
    _require_split(validation_data, "validation")
    directory = Path(checkpoint) if checkpoint is not None else Path(run_directory) / "checkpoints" / "final"
    manifest = json.loads((directory / "checkpoint.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest["data_contract"] != _contract(validation_data):
        raise ValueError("Checkpoint is incomplete or belongs to different data/features/reference")
    verify_file_map(directory, manifest["sha256"])
    config = TrainingConfig.from_dict(json.loads((directory / "config.json").read_text(encoding="utf-8")))
    if reward_mode is not None and reward_mode not in {"none", "fixed", "adaptive"}:
        raise ValueError("Invalid evaluation reward mode")
    configure_torch_runtime()
    vector, normalizer = make_vector_env(validation_data, config, training=False,
                                         stats_path=directory / "vecnormalize.pkl", mode=reward_mode)
    model = PPO.load(directory / "model.zip", env=vector, device=config.device)
    return model, vector, normalizer, config


def evaluate_policy_financial(model, vector, normalizer, data, config):
    _require_split(data, "validation")
    if normalizer.training or normalizer.norm_reward:
        raise ValueError("Evaluation normalization must be frozen; reward normalization is forbidden")
    before = _normalization_digest(normalizer)
    observation = vector.reset()
    rows = []
    for index, date in enumerate(data.features.index):
        if not np.isfinite(observation).all():
            raise FloatingPointError("Nonfinite validation observation")
        action, _ = model.predict(observation, deterministic=True)
        again, _ = model.predict(observation, deterministic=True)
        np.testing.assert_array_equal(action, again)
        observation, reward, dones, infos = vector.step(action)
        info = infos[0]
        if info["date"] != str(date.date()) or bool(dones[0]) != (index == len(data.features) - 1):
            raise AssertionError("Validation coverage/horizon mismatch")
        row = {key: info[key] for key in FINANCIAL_FIELDS}
        row["weights_json"] = json.dumps(info["weights"].tolist())
        row["action_json"] = json.dumps(action[0].tolist())
        rows.append(row)
    if before != _normalization_digest(normalizer):
        raise AssertionError("Validation updated normalization statistics")
    records = pd.DataFrame(rows)
    metrics = financial_metrics(records, config.initial_capital, annualization=config.annualization,
                                risk_free_annual=config.risk_free_annual)
    return records, metrics


class AuditCallback(BaseCallback):
    def __init__(self, directory, config, normalizer, contract):
        super().__init__()
        self.directory, self.config, self.normalizer, self.contract = directory, config, normalizer, contract
        self.metrics = []
        self.file = (directory / "training_financials.csv").open("x", encoding="utf-8", newline="", buffering=1024*1024)
        self.writer = csv.DictWriter(self.file, fieldnames=["training_step", *FINANCIAL_FIELDS, "weights_json", "target_weights_json"])
        self.writer.writeheader()

    def capture_metrics(self):
        record = {"training_step": int(self.num_timesteps)}
        for key, value in self.model.logger.name_to_value.items():
            if isinstance(value, (float, int, np.floating, np.integer)):
                value = float(value)
                if not math.isfinite(value):
                    if key == "train/explained_variance":
                        record[key] = None  # Undefined for a constant target; never used for model selection.
                        continue
                    raise FloatingPointError(f"Nonfinite training metric: {key}")
                record[key] = value
        self.metrics.append(record)

    def _on_step(self):
        for name in ("new_obs", "rewards", "actions"):
            if not np.isfinite(np.asarray(self.locals[name])).all():
                raise FloatingPointError(f"Nonfinite PPO {name}")
        info = self.locals["infos"][0]
        for key in FINANCIAL_FIELDS:
            if isinstance(info[key], (float, int, np.number)) and not np.isfinite(info[key]):
                raise FloatingPointError(f"Nonfinite training financial field: {key}")
        for key in ("weights", "target_weights"):
            if not np.isfinite(info[key]).all() or (info[key] < 0).any() or not np.isclose(info[key].sum(), 1.):
                raise AssertionError("Invalid portfolio weights during training")
        self.writer.writerow({"training_step": self.num_timesteps, **{k: info[k] for k in FINANCIAL_FIELDS},
            "weights_json": json.dumps(info["weights"].tolist()),
            "target_weights_json": json.dumps(info["target_weights"].tolist())})
        if self.num_timesteps % self.config.checkpoint_steps == 0:
            self.file.flush()
            save_checkpoint(self.model, self.normalizer, self.config,
                self.directory / "checkpoints" / f"step-{self.num_timesteps:09d}", self.contract, phase="during_rollout")
        return True

    def _on_rollout_end(self):
        for name in ("observations", "rewards", "returns", "advantages"):
            if not np.isfinite(getattr(self.model.rollout_buffer, name)).all():
                raise FloatingPointError(f"Nonfinite rollout {name}")
        self.capture_metrics()

    def _on_training_end(self):
        self.capture_metrics()
        last = self.metrics[-1]
        if not CORE_LOGS.issubset(last):
            raise AssertionError("PPO core optimization metrics missing")
        self.file.flush()

    def close(self):
        self.file.close()


def train_one_run(train_data, validation_data, config: TrainingConfig, output_root: Path, *, git_sha, provenance=None):
    _require_split(train_data, "train"); _require_split(validation_data, "validation")
    config.validate()
    if _contract(train_data) != _contract(validation_data):
        raise ValueError("Training and validation data contracts differ")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + config.reward_mode + "-" + uuid4().hex[:8]
    root = Path(output_root) / run_id
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "config.json", config.to_dict())
    metadata = {"stage": 3, "status": "started", "run_id": run_id, "git_sha": git_sha,
        "config": config.to_dict(), "data_contract": _contract(train_data), "train_provenance": train_data.provenance,
        "validation_provenance": validation_data.provenance, "session_provenance": provenance or {},
        "seed": config.random_seed, "reward_mode": config.reward_mode, "test_used": False,
        "budget": config.total_timesteps, "training_steps": 0, "python": sys.version,
        "runtime": {"platform": platform.platform(), "machine": platform.machine(),
                    "colab_release_tag": os.environ.get("COLAB_RELEASE_TAG"),
                    "device": config.device, "cuda_build": torch.version.cuda,
                    "gpu_name": torch.cuda.get_device_name(0) if config.device == "cuda" and torch.cuda.is_available() else None},
        "versions": {name: importlib.metadata.version(name) for name in (
            "numpy", "pandas", "torch", "gymnasium", "stable-baselines3", "pyarrow")},
        "algorithm": "SB3 PPO with unchanged optimizer/advantage/clipping implementation",
        "normalization": {"observations": config.normalize_observations, "reward": False, "fit_split": "train"},
        "ppo_defaults": {"n_envs": 1, "max_grad_norm": .5, "normalize_advantage": True, "use_sde": False, "target_kl": None}}
    write_json(root / "metadata.json", metadata)
    print("Training run:", root)
    model = vector = logger = callback = None
    try:
        model, vector, normalizer = build_ppo(train_data, config)
        metadata["runtime"]["torch_settings"] = torch_runtime_info()
        write_json(root / "metadata.json", metadata)
        logger = configure(str(root / "logs"), ["csv"])
        model.set_logger(logger)
        callback = AuditCallback(root, config, normalizer, _contract(train_data))
        model.learn(total_timesteps=config.total_timesteps, callback=callback, reset_num_timesteps=True)
        if model.num_timesteps != config.total_timesteps:
            raise AssertionError("Actual training budget differs from the common configured budget")
        model.logger.dump(step=model.num_timesteps)
        save_checkpoint(model, normalizer, config, root / "checkpoints" / "final", _contract(train_data), phase="after_training")
        write_json(root / "training_metrics.json", {"records": callback.metrics, "core_metrics_finite": True,
            "explained_variance_null_means_undefined": True})
        model.logger.close(); callback.close(); vector.close()
        loaded, evaluation, frozen, saved_config = load_saved_policy(root, validation_data)
        records, metrics = evaluate_policy_financial(loaded, evaluation, frozen, validation_data, saved_config)
        evaluation.close()
        # A second independently loaded model must replay exactly on the same runtime.
        loaded2, evaluation2, frozen2, saved_config2 = load_saved_policy(root, validation_data)
        replay, metrics2 = evaluate_policy_financial(loaded2, evaluation2, frozen2, validation_data, saved_config2)
        evaluation2.close()
        pd.testing.assert_frame_equal(records, replay, check_exact=True)
        if metrics != metrics2:
            raise AssertionError("Reloaded deterministic financial metrics differ")
        records.to_csv(root / "validation_financials.csv", index=False)
        write_json(root / "validation_metrics.json", metrics)
        write_json(root / "reload_check.json", {"status": "passed", "same_runtime_reload": True,
            "fresh_session_reload": "not_yet_verified", "exact_actions_and_finances": True,
            "normalization_frozen": True, "test_used": False})
        metadata.update(status="complete", training_steps=int(model.num_timesteps),
            model_path="checkpoints/final/model.zip", validation_split="validation",
            validation_observations=len(records), finished_at_utc=datetime.now(timezone.utc).isoformat())
        metadata["files_sha256"] = {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(root.rglob("*"))
                                     if p.is_file() and p.name not in {"metadata.json", "metadata.sha256"}}
        write_json(root / "metadata.json", metadata)
        (root / "metadata.sha256").write_text(sha256_file(root / "metadata.json") + "  metadata.json\n")
        return root
    except Exception as error:
        metadata.update(status="failed", error_type=type(error).__name__, error=str(error),
                        training_steps=int(model.num_timesteps) if model is not None else 0)
        write_json(root / "metadata.json", metadata)
        raise
    finally:
        if callback is not None:
            callback.close()
        if vector is not None:
            vector.close()
        if logger is not None:
            logger.close()


def replay_saved_run(run_directory: Path, validation_data, output_directory: Path):
    root = Path(run_directory)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "complete" or metadata.get("test_used") is not False:
        raise ValueError("Only completed development runs may be replayed")
    if (root / "metadata.sha256").read_text().split() != [sha256_file(root / "metadata.json"), "metadata.json"]:
        raise ValueError("Run metadata checksum mismatch")
    verify_file_map(root, metadata["files_sha256"])
    runtime_before = torch_runtime_info()
    model, vector, frozen, config = load_saved_policy(root, validation_data)
    try:
        records, metrics = evaluate_policy_financial(model, vector, frozen, validation_data, config)
    finally:
        vector.close()
    previous = pd.read_csv(root / "validation_financials.csv", float_precision="round_trip", keep_default_na=False)
    output = Path(output_directory)
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("Replay evidence must not modify the original run")
    output.mkdir(parents=True, exist_ok=False)
    records.to_csv(output / "validation_replay.csv", index=False)
    write_json(output / "validation_replay_metrics.json", metrics)
    report = {"status": "failed", "source_run": metadata["run_id"],
        "git_sha": metadata["git_sha"], "observations": len(records), "exact_actions_and_finances": False,
        "torch_runtime_before_load": runtime_before, "torch_runtime": torch_runtime_info(),
        "validation_replay_sha256": sha256_file(output / "validation_replay.csv"),
        "validation_replay_metrics_sha256": sha256_file(output / "validation_replay_metrics.json"),
        "test_used": False, "note": "Fresh-session claim requires the user's separately restarted Colab session"}
    try:
        pd.testing.assert_frame_equal(previous, records, check_exact=True, check_dtype=False)
        if json.loads((root / "validation_metrics.json").read_text()) != metrics:
            raise AssertionError("Replay financial metrics differ")
    except AssertionError as error:
        report["error"] = str(error)
        try:
            report["first_step_diagnostic"] = first_step_replay_diagnostic(root, validation_data)
        except Exception as diagnostic_error:
            report["diagnostic_error"] = str(diagnostic_error)
        write_json(output / "replay_report.json", report)
        raise AssertionError(f"Exact replay failed; diagnostic evidence saved to {output}") from error
    report.update(status="passed", exact_actions_and_finances=True)
    write_json(output / "replay_report.json", report)
    return output
