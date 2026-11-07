"""Small synthetic PPO/configuration/metric tests; execute in Google Colab only."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.data import build_features, sha256_file, write_json
from src.environment_data import MarketData
from src.financial_metrics import financial_metrics
from src.portfolio_env import PortfolioEnv
from src.ppo_training import (
    build_ppo, evaluate_policy_financial, load_saved_policy, save_checkpoint, train_one_run,
)
from src.regularization import penalty_coefficient
from src.training_config import TrainingConfig, load_training_config
from src.training_data import load_development_market


def synthetic_market(split="train"):
    dates = pd.bdate_range("2018-01-01", periods=12)
    prices = pd.DataFrame({"A": [100., 101., 99., 102., 100., 104., 103., 106., 105., 108., 107., 110.]}, index=dates)
    features = build_features(prices, window=2).iloc[3:]
    features["sigma__SPY"] = .2
    return MarketData(prices, prices, features, .2, provenance={
        "split": split, "synthetic": True, "snapshot_id": "synthetic-test-only",
        "snapshot_manifest_sha256": "synthetic", "stage1_metadata_sha256": "synthetic", "price_model": "synthetic_adjusted_open_close_v1"})


def tiny_config(**kwargs):
    return replace(TrainingConfig(), n_steps=8, batch_size=4, n_epochs=1,
                   total_timesteps=16, checkpoint_steps=8, policy_layers=(8,), **kwargs)


@pytest.mark.parametrize("sigma", [0., .05, .2, 1.])
def test_alpha_zero_is_exactly_fixed_for_all_sigmas(sigma):
    assert penalty_coefficient("adaptive", .001, sigma, .2, alpha=0, lambda_max=.01) == \
           penalty_coefficient("fixed", .001, sigma, .2, alpha=-.5, lambda_max=.01)


def test_sensitivity_can_have_both_signs_and_handles_zero_sigma():
    positive = penalty_coefficient("adaptive", .001, .1, .2, alpha=.5)
    negative = penalty_coefficient("adaptive", .001, .1, .2, alpha=-.5)
    assert positive == pytest.approx(.001 * np.sqrt(.5))
    assert negative == pytest.approx(.001 / np.sqrt(.5))
    assert positive < .001 < negative
    assert penalty_coefficient("adaptive", .001, 0., .2, alpha=-.5, lambda_max=.003) == .003
    assert penalty_coefficient("none", .001, 1., .2, alpha=-.5) == 0.


def test_config_roundtrip_and_budget_validation(tmp_path):
    config = tiny_config(reward_mode="adaptive", alpha=-.5)
    path = tmp_path / "config.json"
    write_json(path, config.to_dict())
    assert load_training_config(path) == config
    with pytest.raises(ValueError, match="test is forbidden"):
        replace(config, evaluation_split="test").validate()
    with pytest.raises(ValueError, match="complete rollouts"):
        replace(config, total_timesteps=17).validate()
    with pytest.raises(ValueError):
        replace(config, lambda_min=.01, lambda_0=.001).validate()


def test_initial_ppo_weights_and_spaces_match_across_modes():
    states = []
    data = synthetic_market()
    for mode in ("none", "fixed", "adaptive"):
        model, vector, _ = build_ppo(data, tiny_config(reward_mode=mode))
        obs = vector.reset()
        assert obs.shape == (1, model.observation_space.shape[0])
        action, _ = model.predict(obs, deterministic=True)
        assert action.shape == (1, len(data.assets) + 1) and np.isfinite(action).all()
        obs, reward, _, infos = vector.step(action)
        assert np.isfinite(obs).all() and np.isfinite(reward).all()
        assert infos[0]["weights"].sum() == pytest.approx(1.)
        states.append({k: v.detach().clone() for k, v in model.policy.state_dict().items()})
        vector.close()
    for key in states[0]:
        assert torch.equal(states[0][key], states[1][key]) and torch.equal(states[0][key], states[2][key])


def test_reward_equivalence_in_full_environment_at_alpha_zero():
    data = synthetic_market()
    config = tiny_config(alpha=0.)
    fixed = PortfolioEnv(data, config.env_config("fixed"))
    adaptive = PortfolioEnv(data, config.env_config("adaptive"))
    fixed.reset(seed=7); adaptive.reset(seed=7)
    for raw in ([1., -1.], [-1., 1.], [0., 0.]):
        a = fixed.step(np.asarray(raw, np.float32))
        b = adaptive.step(np.asarray(raw, np.float32))
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1] == b[1] and a[-1]["portfolio_value"] == b[-1]["portfolio_value"]


def test_checkpoint_reload_and_financial_metrics_ignore_reward_mode(tmp_path):
    train, val = synthetic_market(), synthetic_market("validation")
    config = tiny_config()
    model, vector, normalizer = build_ppo(train, config)
    # Test-only untrained checkpoint, not an experimental result.
    contract = {"assets": list(train.assets), "feature_names": list(train.features), "sigma_ref": train.sigma_ref,
                **{k: train.provenance.get(k) for k in ("stage1_metadata_sha256", "snapshot_id", "snapshot_manifest_sha256", "price_model")}}
    checkpoint = tmp_path / "unit-checkpoint"
    save_checkpoint(model, normalizer, config, checkpoint, contract, phase="unit_test_untrained")
    vector.close()
    outcomes = []
    rewards = []
    for mode in ("none", "fixed", "adaptive"):
        loaded, evaluation, frozen, cfg = load_saved_policy(tmp_path, val, checkpoint=checkpoint, reward_mode=mode)
        rows, metrics = evaluate_policy_financial(loaded, evaluation, frozen, val, cfg)
        outcomes.append((rows[["net_return", "portfolio_value", "turnover", "commission", "action_json"]], metrics))
        rewards.append(rows["reward"].to_numpy())
        evaluation.close()
    pd.testing.assert_frame_equal(outcomes[0][0], outcomes[1][0], check_exact=True)
    pd.testing.assert_frame_equal(outcomes[0][0], outcomes[2][0], check_exact=True)
    assert outcomes[0][1] == outcomes[1][1] == outcomes[2][1]
    assert not np.array_equal(rewards[0], rewards[1])
    with pytest.raises(FileExistsError):
        save_checkpoint(model, normalizer, config, checkpoint, contract, phase="do_not_overwrite")


def test_tiny_synthetic_training_saves_and_reloads(tmp_path):
    root = train_one_run(synthetic_market(), synthetic_market("validation"), tiny_config(), tmp_path,
                         git_sha="0" * 40, provenance={"synthetic_unit_test": True})
    metadata = json.loads((root / "metadata.json").read_text())
    assert metadata["training_steps"] == 16 and metadata["status"] == "complete"
    assert (root / "checkpoints/final/model.zip").is_file()
    assert (root / "checkpoints/step-000000008/checkpoint.json").is_file()
    assert json.loads((root / "reload_check.json").read_text())["status"] == "passed"


def test_financial_metrics_known_path_and_reward_independence():
    rows = pd.DataFrame({"net_return": [.1, -.1], "portfolio_value": [110., 99.],
                         "turnover": [1., .5], "commission": [1., 2.], "reward": [100., -100.]})
    result = financial_metrics(rows, 100., annualization=2)
    assert result["cumulative_net_return"] == pytest.approx(-.01)
    assert result["cagr"] == pytest.approx(-.01)
    assert result["maximum_drawdown"] == pytest.approx(.1)
    assert result["annualized_volatility"] == pytest.approx(.2)
    assert result["net_sharpe"] == pytest.approx(0.)
    assert result["portfolio_turnover_mean_daily"] == .75
    assert result["portfolio_turnover_cumulative"] == 1.5
    rows["reward"] = [1e9, -1e9]
    assert financial_metrics(rows, 100., annualization=2) == result


def test_flat_nav_has_undefined_sharpe_not_fake_finite_score():
    rows = pd.DataFrame({"net_return": [0., 0.], "portfolio_value": [100., 100.], "turnover": [0., 0.], "commission": [0., 0.]})
    metrics = financial_metrics(rows, 100.)
    assert metrics["net_sharpe"] is None and metrics["maximum_drawdown"] == 0.
    rows.loc[1, "net_return"] = np.nan
    with pytest.raises(ValueError):
        financial_metrics(rows, 100.)


def test_training_and_loading_reject_test_data_before_any_run(tmp_path):
    test = synthetic_market("test")
    with pytest.raises(ValueError, match="forbidden"):
        build_ppo(test, tiny_config())
    with pytest.raises(ValueError, match="forbidden"):
        load_saved_policy(tmp_path, test)
    with pytest.raises(ValueError, match="test access is forbidden"):
        load_development_market(tmp_path, "test")


def test_development_adapter_filters_prices_and_never_opens_test(tmp_path, monkeypatch):
    dates = pd.to_datetime(["2009-12-29", "2009-12-30", "2009-12-31", "2010-01-04", "2010-01-05", "2010-01-06",
                            "2019-01-02", "2019-01-03", "2019-01-04", "2022-01-03"])
    prices = pd.DataFrame({"A": 100. * 1.01 ** np.arange(10), "SPY": 200. * 1.02 ** np.arange(10)}, index=dates.rename("Date"))
    features = build_features(prices, window=2)
    train = features.loc[(dates >= "2010-01-01") & (dates < "2019-01-01")]
    val = features.loc[(dates >= "2019-01-01") & (dates < "2022-01-01")]
    tables = {"vendor_open": prices, "vendor_close": prices, "adjusted_close": prices,
              "adjustment_factor": prices / prices, "features_train": train, "features_validation": val}
    for name, table in tables.items():
        table.to_parquet(tmp_path / f"{name}.parquet")
    # No features_test file is created: accessing it would fail the test.
    meta = {"status": "complete", "run_id": "synthetic-only", "sigma_ref": float(train["sigma__SPY"].median()),
            "snapshot": {"id": "synthetic", "manifest_sha256": "synthetic"},
            "config": {"assets": ["A"], "market": "SPY", "window": 2, "annualization": 252,
                       "study_start": "2010-01-01", "validation_start": "2019-01-01", "test_start": "2022-01-01", "end_exclusive": "2026-01-01"},
            "files_sha256": {p.name: sha256_file(p) for p in tmp_path.glob("*.parquet")}}
    write_json(tmp_path / "metadata.json", meta)
    (tmp_path / "metadata.sha256").write_text(sha256_file(tmp_path / "metadata.json") + "  metadata.json\n")
    original = pd.read_parquet
    calls = []

    def bounded(path, **kwargs):
        assert "features_test" not in str(path) and kwargs.get("filters")
        table = original(path, **kwargs)
        assert (table.index < pd.Timestamp("2022-01-01")).all()
        calls.append(Path(path).name)
        return table

    from pathlib import Path
    monkeypatch.setattr(pd, "read_parquet", bounded)
    for split in ("train", "validation"):
        loaded = load_development_market(tmp_path, split)
        assert loaded.provenance["split"] == split
        assert (loaded.adjusted_open.index < pd.Timestamp("2022-01-01")).all()
    assert "features_test.parquet" not in calls
