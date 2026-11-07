"""Short predetermined accounting scenarios and artifact verification, not training."""
from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .data import sha256_file
from .environment_data import load_stage1_market
from .portfolio_env import EnvConfig, PortfolioEnv
from .validation import pytest_summary, verify_file_map


def deterministic_rollout(data, common: EnvConfig, actions: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply identical fixed actions to all modes; assert finances and journal identities."""
    if not actions or len(actions) > len(data.features):
        raise ValueError("Action schedule must fit within this data split")
    records, trades = [], []
    baselines = []
    for mode in ("none", "fixed", "adaptive"):
        env = PortfolioEnv(data, replace(common, penalty_mode=mode, max_episode_steps=len(actions)))
        env.reset(seed=42)
        mode_infos = []
        for number, a in enumerate(actions):
            old_units, old_cash = env.units.copy(), env.cash
            _, reward, terminated, truncated, info = env.step(np.asarray(a, dtype=np.float32))
            np.testing.assert_allclose(
                info["nav_close"], info["nav_previous_close"] + info["overnight_pnl"]
                - info["commission"] + info["intraday_pnl"], rtol=1e-12, atol=1e-8)
            np.testing.assert_allclose(info["commission"], info["commission_rate"] * info["traded_notional"], rtol=1e-12)
            np.testing.assert_allclose(reward, info["r_net_log"] - info["lambda_t"] * info["turnover"], rtol=1e-12)
            assert info["cash"] >= 0 and info["portfolio_value"] > 0
            assert np.maximum(info["signed_trade_notional"], 0).sum() + info["commission"] <= (
                old_cash + np.maximum(-info["signed_trade_notional"], 0).sum() + 1e-8)
            if number < len(actions) - 1 and (terminated or truncated):
                raise AssertionError("Episode ended before the planned scenario")
            scalars = {k: v for k, v in info.items() if not isinstance(v, np.ndarray)}
            records.append({"mode": mode, "step": number, **scalars,
                            "target_cash_weight": float(info["target_weights"][-1]),
                            "terminated": terminated, "truncated": truncated})
            for asset, symbol in enumerate(env.assets):
                trades.append({"mode": mode, "date": info["date"], "asset": symbol,
                    "units_before": float(old_units[asset]),
                    "units_after": float(info["positions_synthetic_units"][asset]),
                    "trade_notional": float(info["signed_trade_notional"][asset]),
                    "execution_price_synthetic": float(info["execution_prices_synthetic"][asset]),
                    "close_price_synthetic": float(info["close_prices_synthetic"][asset]),
                    "target_weight": float(info["target_weights"][asset]),
                    "close_weight": float(info["weights"][asset])})
            mode_infos.append(info)
        np.testing.assert_allclose(sum(x["r_net_log"] for x in mode_infos),
            math.log(mode_infos[-1]["portfolio_value"] / common.initial_capital), rtol=1e-11, atol=1e-12)
        if mode == "none":
            baselines = mode_infos
        else:
            for baseline, current in zip(baselines, mode_infos):
                for key in ("portfolio_value", "cash", "turnover", "commission", "weights",
                            "positions_synthetic_units", "r_net_log", "gross_return"):
                    np.testing.assert_array_equal(baseline[key], current[key])
    return pd.DataFrame(records), pd.DataFrame(trades)


def verify_stage2_run(run_directory: Path) -> dict:
    root = Path(run_directory)
    meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if meta.get("stage") != 2 or meta.get("status") not in {"prepared", "complete"}:
        raise ValueError("Not a prepared Stage 2 run")
    if meta["status"] == "complete":
        if (root / "metadata.sha256").read_text().split() != [sha256_file(root / "metadata.json"), "metadata.json"]:
            raise ValueError("Stage 2 metadata checksum mismatch")
    required = {"pytest.xml", "pytest.log", "environment.json", "pip-freeze.txt", "source.bundle",
                "scenario.csv", "trades.csv", "diagnostics.png", "adjusted_open.parquet", "adjusted_close.parquet"}
    if not required.issubset(meta["files_sha256"]):
        raise ValueError("Stage 2 artifacts missing")
    verify_file_map(root, meta["files_sha256"])
    verify_file_map(root / "provenance", meta["source_sha256"])
    tests = pytest_summary(root / "pytest.xml")
    if meta["tests"].get("exit_code") != 0 or any(tests[k] != meta["tests"].get(k) for k in tests):
        raise ValueError("Stage 2 tests do not match JUnit")
    environment = json.loads((root / "environment.json").read_text())
    if (environment.get("future_stack", {}).get("status") != "passed"
            or environment.get("dependency_closure", {}).get("status") != "passed"
            or environment.get("pip_check_policy", {}).get("status") not in {"passed", "passed_with_known_warnings"}):
        raise ValueError("Stage 2 requires the verified CPU RL dependency profile")
    data = load_stage1_market(Path(meta["stage1_directory"]), "train", acknowledge_synthetic_prices=True)
    if data.provenance["stage1_metadata_sha256"] != meta["stage1_metadata_sha256"]:
        raise ValueError("Stage 1 metadata changed during validation")
    for name in ("adjusted_open", "adjusted_close"):
        pd.testing.assert_frame_equal(pd.read_parquet(root / f"{name}.parquet"), getattr(data, name),
                                      check_freq=False, check_exact=True)
    config = EnvConfig(**meta["common_config"])
    scenario, trades = deterministic_rollout(data, config, meta["action_schedule"])
    for name, expected in (("scenario", scenario), ("trades", trades)):
        pd.testing.assert_frame_equal(pd.read_csv(root / f"{name}.csv", float_precision="round_trip"),
                                      expected, check_dtype=False, check_exact=True)
    return {"status": "verified", "stage": 2, "run_id": meta["run_id"], "git_commit": meta["git_commit"],
            "tests": tests, "price_model": data.provenance["price_model"],
            "scenario_steps_per_mode": len(meta["action_schedule"]), "modes": ["none", "fixed", "adaptive"],
            "identical_actions_identical_finances": True, "journal_reconciled": True,
            "source_stage1_run": data.provenance["stage1_run_id"], "sigma_ref": data.sigma_ref,
            "source_split": "train", "scenario_start": str(data.features.index[0].date()),
            "scenario_end": str(data.features.index[len(meta["action_schedule"]) - 1].date()),
            "run_directory": str(root), "training_performed": False,
            "nominal_share_dividend_ledger_validated": False,
            "limitation": "Only the documented synthetic adjusted-price accounting model is checked"}
