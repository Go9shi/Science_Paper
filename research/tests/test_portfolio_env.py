"""Analytical financial and causality tests. Execute exclusively in Colab."""
from dataclasses import replace
import json
import math

import numpy as np
import pandas as pd
import pytest

from src.data import build_features, sha256_file, write_json
from src.environment_data import MarketData, adjusted_execution_prices, load_stage1_market
from src.portfolio import action_to_weights, rebalance
from src.portfolio_env import EnvConfig, PortfolioEnv


def toy_market(opens, closes=None, sigma=.2):
    opens = np.asarray(opens, dtype=float)
    if opens.ndim == 1:
        opens = opens[:, None]
    closes = opens.copy() if closes is None else np.asarray(closes, dtype=float).reshape(opens.shape)
    index = pd.bdate_range("2020-01-01", periods=len(opens))
    columns = [f"A{i}" for i in range(opens.shape[1])]
    o, c = pd.DataFrame(opens, index=index, columns=columns), pd.DataFrame(closes, index=index, columns=columns)
    # Synthetic initial history is known to be flat; this fill is fixture-only.
    f = c.pct_change(fill_method=None).shift(1).fillna(0.).add_prefix("lag_return__")
    f["sigma__SPY"] = sigma
    return MarketData(o, c, f.iloc[1:], .2)


def action(weights):
    return (2. * np.asarray(weights) - 1.).astype(np.float32)


def make_env(data, **kwargs):
    return PortfolioEnv(data, EnvConfig(initial_capital=1000., **kwargs))


def test_constant_prices_no_trades_and_all_cash():
    env = make_env(toy_market([100., 100., 100.]), commission_bps=100.)
    env.reset(seed=3)
    for expected_done in (False, True):
        _, reward, terminated, truncated, info = env.step(action([0., 1.]))
        assert reward == 0. and info["portfolio_value"] == 1000.
        assert info["turnover"] == 0. and info["commission"] == 0.
        assert info["cash"] == 1000.
        assert terminated is expected_done and truncated is False


def test_zero_fee_buy_at_open_then_mark_at_close():
    env = make_env(toy_market([100., 100.], [100., 110.]), commission_bps=0.)
    env.reset()
    _, reward, done, _, info = env.step(action([1., 0.]))
    assert done
    assert info["positions_synthetic_units"][0] == pytest.approx(10.)
    assert info["portfolio_value"] == pytest.approx(1100.)
    assert info["turnover"] == pytest.approx(1.)
    assert reward == pytest.approx(math.log(1.1))
    assert info["r_gross_log"] == pytest.approx(info["r_net_log"])


def test_commission_is_financed_before_purchase():
    env = make_env(toy_market([100., 100.], [100., 110.]), commission_bps=100.)
    env.reset()
    _, reward, _, _, info = env.step(action([1., 0.]))
    assert info["nav_pre_trade_open"] == 1000.
    assert info["nav_post_trade_open"] == pytest.approx(1000 / 1.01)
    assert info["commission"] == pytest.approx(1000 * .01 / 1.01)
    assert info["positions_synthetic_units"][0] == pytest.approx(1000 / 1.01 / 100)
    assert info["portfolio_value"] == pytest.approx(1000 * 1.1 / 1.01)
    assert reward == pytest.approx(math.log(1.1 / 1.01))
    assert info["gross_return"] == pytest.approx(.1)
    assert info["cash"] >= 0


def test_buy_and_hold_does_not_charge_for_passive_price_changes():
    env = make_env(toy_market([100., 100., 110.], [100., 110., 121.]), commission_bps=100.)
    env.reset()
    _, first, _, _, a = env.step(action([1., 0.]))
    _, second, _, _, b = env.step(action([1., 0.]))
    assert b["commission"] == 0. and b["turnover"] == 0.
    assert b["positions_synthetic_units"][0] == a["positions_synthetic_units"][0]
    assert b["portfolio_value"] == pytest.approx(1000 * 1.21 / 1.01)
    assert first + second == pytest.approx(math.log(1.21 / 1.01))


def test_drift_and_rebalancing_two_assets_have_known_turnover():
    data = toy_market([[100., 100.], [100., 100.], [100., 100.], [200., 100.]],
                      [[100., 100.], [100., 100.], [200., 100.], [200., 100.]])
    env = make_env(data, commission_bps=0.)
    env.reset()
    env.step(action([.5, .5, 0.]))
    obs, _, _, _, drift = env.step(action([.5, .5, 0.]))
    np.testing.assert_allclose(drift["weights"], [2/3, 1/3, 0.], atol=1e-13)
    weight_offset = len(env.feature_names)
    np.testing.assert_allclose(obs[weight_offset:weight_offset + 3], [2/3, 1/3, 0.], atol=1e-7)
    _, _, _, _, final = env.step(action([.5, .5, 0.]))
    assert final["nav_pre_trade_open"] == pytest.approx(1500.)
    assert final["traded_notional"] == pytest.approx(500.)
    assert final["turnover"] == pytest.approx(1/3)
    np.testing.assert_allclose(final["signed_trade_notional"], [-250., 250.], atol=1e-10)
    np.testing.assert_allclose(final["weights"], [.5, .5, 0.], atol=1e-13)


def test_full_asset_switch_charges_both_sale_and_purchase():
    env = make_env(toy_market(np.full((3, 2), 100.)), commission_bps=100.)
    env.reset()
    _, _, _, _, first = env.step(action([1., 0., 0.]))
    _, _, _, _, second = env.step(action([0., 1., 0.]))
    pre = first["portfolio_value"]
    expected_post = pre * (1 - .01) / (1 + .01)
    assert second["portfolio_value"] == pytest.approx(expected_post)
    assert second["commission"] == pytest.approx(.01 * (pre + expected_post))
    assert second["turnover"] == pytest.approx(1 + expected_post / pre)


def test_new_position_does_not_earn_gap_before_its_open_execution():
    env = make_env(toy_market([100., 200.], [100., 220.]), commission_bps=0.)
    env.reset()
    _, _, _, _, info = env.step(action([1., 0.]))
    assert info["positions_synthetic_units"][0] == pytest.approx(5.)
    assert info["portfolio_value"] == pytest.approx(1100.)
    assert info["overnight_pnl"] == 0.
    assert info["intraday_pnl"] == pytest.approx(100.)


def test_old_position_earns_overnight_gap_before_sale():
    env = make_env(toy_market([100., 100., 200.], [100., 100., 220.]), commission_bps=0.)
    env.reset()
    env.step(action([1., 0.]))
    _, reward, _, _, info = env.step(action([0., 1.]))
    assert info["portfolio_value"] == pytest.approx(2000.)
    assert info["overnight_pnl"] == pytest.approx(1000.)
    assert info["intraday_pnl"] == pytest.approx(0.)
    assert reward == pytest.approx(math.log(2.))


def test_all_cash_ignores_large_market_moves():
    env = make_env(toy_market([100., 200., 20.], [100., 50., 80.]))
    env.reset()
    for _ in range(2):
        _, reward, _, _, info = env.step(action([0., 1.]))
        assert reward == 0. and info["portfolio_value"] == 1000.


def test_self_financing_nonnegative_capital_for_extreme_valid_actions():
    rng = np.random.default_rng(7)
    opens = 100 * np.exp(rng.normal(0, .1, (51, 12)))
    closes = opens * np.exp(rng.normal(0, .1, (51, 12)))
    env = make_env(toy_market(opens, closes), commission_bps=50.)
    env.reset(seed=7)
    old_cash = 1000.
    for _ in range(50):
        _, _, _, _, info = env.step(rng.uniform(-1, 1, 13))
        trades = info["signed_trade_notional"]
        assert np.maximum(trades, 0).sum() + info["commission"] <= old_cash + np.maximum(-trades, 0).sum() + 1e-8
        assert info["cash"] >= 0 and info["portfolio_value"] > 0
        assert (info["positions_synthetic_units"] >= 0).all()
        assert info["weights"].sum() == pytest.approx(1.)
        assert info["nav_pre_trade_open"] - info["nav_post_trade_open"] == pytest.approx(info["commission"], abs=1e-8)
        old_cash = info["cash"]


def test_current_and_future_open_are_absent_from_observations():
    a = toy_market([100., 100., 110., 120.], [100., 105., 115., 125.])
    b = toy_market([100., 999., 9999., 120.], [100., 105., 115., 125.])
    env_a, env_b = make_env(a), make_env(b)
    obs_a, _ = env_a.reset(seed=3)
    obs_b, _ = env_b.reset(seed=3)
    np.testing.assert_array_equal(obs_a, obs_b)
    # First transition with cash is identical; its next obs must not expose Open[2].
    next_a, _, _, _, _ = env_a.step(action([0., 1.]))
    next_b, _, _, _, _ = env_b.step(action([0., 1.]))
    np.testing.assert_array_equal(next_a, next_b)


def test_future_prices_do_not_change_current_transition_or_next_observation():
    a = toy_market([100., 100., 110., 120.], [100., 105., 115., 125.])
    b = toy_market([100., 100., 9999., 9999.], [100., 105., 9999., 9999.])
    env_a, env_b = make_env(a), make_env(b)
    env_a.reset(); env_b.reset()
    x, rx, _, _, ix = env_a.step(action([1., 0.]))
    y, ry, _, _, iy = env_b.step(action([1., 0.]))
    np.testing.assert_array_equal(x, y)
    assert rx == ry and ix["portfolio_value"] == iy["portfolio_value"]


@pytest.mark.parametrize("slice_", [slice(0, 2), slice(2, 4), slice(4, 6)])
def test_episode_never_executes_outside_its_partition(slice_):
    data = toy_market([100.] * 7)
    data = replace(data, features=data.features.iloc[slice_])
    env = make_env(data)
    env.reset()
    dates = []
    while True:
        obs, _, terminated, truncated, info = env.step(action([0., 1.]))
        dates.append(info["date"])
        assert not truncated
        if terminated:
            assert obs[-1] == 0.  # known finite horizon, terminal market padding
            break
    assert dates == [str(x.date()) for x in data.features.index]
    with pytest.raises(RuntimeError, match="call reset"):
        env.step(action([0., 1.]))


def test_time_limit_truncation_keeps_valid_causal_next_observation():
    env = make_env(toy_market([100.] * 4), max_episode_steps=1)
    env.reset()
    obs, _, terminated, truncated, _ = env.step(action([0., 1.]))
    assert not terminated and truncated and obs[-1] == pytest.approx(2/3)
    assert env.observation_space.contains(obs)


def test_natural_horizon_takes_precedence_over_equal_time_limit():
    env = make_env(toy_market([100., 100.]), max_episode_steps=1)
    env.reset()
    _, _, terminated, truncated, _ = env.step(action([0., 1.]))
    assert terminated and not truncated


def test_modes_have_identical_finances_and_distinct_reward_only():
    data = toy_market([[100., 100.], [100., 100.], [120., 80.], [110., 90.]], sigma=.4)
    actions = [action([1., 0., 0.]), action([0., 1., 0.]), action([.5, .5, 0.])]
    records = {}
    for mode in ("none", "fixed", "adaptive"):
        env = make_env(data, penalty_mode=mode, lambda_base=.01, commission_bps=10.)
        env.reset(seed=8)
        records[mode] = [env.step(a)[-1] for a in actions]
    for step in range(3):
        none, fixed, adaptive = (records[m][step] for m in ("none", "fixed", "adaptive"))
        for key in ("portfolio_value", "cash", "turnover", "commission", "r_net_log", "weights", "positions_synthetic_units"):
            np.testing.assert_array_equal(none[key], fixed[key])
            np.testing.assert_array_equal(none[key], adaptive[key])
        assert fixed["reward"] == pytest.approx(none["r_net_log"] - .01 * none["turnover"])
        assert adaptive["reward"] == pytest.approx(none["r_net_log"] - .02 * none["turnover"])


def test_adaptive_cap_and_zero_volatility():
    for sigma, expected in ((0., 0.), (100., .05)):
        env = make_env(toy_market([100., 100.], sigma=sigma), penalty_mode="adaptive", lambda_base=.01)
        env.reset()
        info = env.step(action([1., 0.]))[-1]
        assert info["lambda_t"] == pytest.approx(expected)


@pytest.mark.parametrize("kind", ["split", "dividend"])
def test_corporate_actions_are_implicit_not_applied_twice(kind):
    index = pd.bdate_range("2020-01-01", periods=4)
    vendor = pd.DataFrame({"A": [100., 100., 50., 50.] if kind == "split" else [100., 100., 99., 99.]}, index=index)
    adjusted = pd.DataFrame({"A": [50.] * 4 if kind == "split" else [99.] * 4}, index=index)
    original = vendor.copy(deep=True)
    synthetic = adjusted_execution_prices(vendor, vendor, adjusted)
    np.testing.assert_allclose(synthetic, adjusted)
    features = pd.DataFrame({"sigma__SPY": .2}, index=index[1:])
    env = make_env(MarketData(synthetic, adjusted, features, .2), commission_bps=0.)
    env.reset()
    first = env.step(action([1., 0.]))[-1]
    second = env.step(action([1., 0.]))[-1]
    assert first["portfolio_value"] == pytest.approx(1000.)
    assert second["portfolio_value"] == pytest.approx(1000.)
    np.testing.assert_array_equal(first["positions_synthetic_units"], second["positions_synthetic_units"])
    assert second["cash"] == pytest.approx(0., abs=1e-10)
    pd.testing.assert_frame_equal(vendor, original)


def test_action_mapping_supports_simplex_edges_and_rejects_nan():
    np.testing.assert_array_equal(action_to_weights(np.full(3, -1.), 2), [0., 0., 1.])
    np.testing.assert_array_equal(action_to_weights([3., -3., -3.], 2), [1., 0., 0.])
    np.testing.assert_allclose(action_to_weights([0., 0., 0.], 2), [1/3] * 3)
    with pytest.raises(ValueError, match="finite"):
        action_to_weights([np.nan, 0., 0.], 2)


def test_reset_seed_and_state_are_reproducible():
    env = make_env(toy_market([100.] * 4))
    first, info = env.reset(seed=42)
    a = env.action_space.sample()
    env.step(a)
    second, again = env.reset(seed=42)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(a, env.action_space.sample())
    assert info["portfolio_value"] == again["portfolio_value"] == 1000.


def test_env_uses_private_price_copies():
    data = toy_market([100., 100., 100.])
    env = make_env(data)
    data.adjusted_open.iloc[1, 0] = 1000.
    env.reset()
    info = env.step(action([1., 0.]))[-1]
    assert info["portfolio_value"] == pytest.approx(1000.)


def test_gymnasium_and_sb3_checkers():
    from gymnasium.utils.env_checker import check_env as gym_check
    from stable_baselines3.common.env_checker import check_env as sb3_check

    data = toy_market(np.full((8, 2), 100.))
    gym_check(make_env(data), skip_render_check=True)
    sb3_check(make_env(data), warn=True)


def test_loader_requires_explicit_price_model_acknowledgement(tmp_path):
    with pytest.raises(ValueError, match="acknowledge"):
        load_stage1_market(tmp_path, "train")


def test_loader_verifies_checksums_reference_and_split_boundaries(tmp_path):
    index = pd.bdate_range("2020-01-01", periods=10)
    prices = pd.DataFrame({"A": 100. * 1.01 ** np.arange(10), "SPY": 200. * 1.02 ** np.arange(10)}, index=index)
    f = build_features(prices, window=2).iloc[3:]
    tables = {"vendor_open": prices, "vendor_close": prices, "adjusted_close": prices,
              "adjustment_factor": prices / prices, "features_train": f.iloc[:2],
              "features_validation": f.iloc[2:4], "features_test": f.iloc[4:]}
    for name, table in tables.items():
        table.to_parquet(tmp_path / f"{name}.parquet")
    reference = float(tables["features_train"]["sigma__SPY"].median())
    meta = {"status": "complete", "run_id": "synthetic-test-only", "sigma_ref": reference,
            "config": {"assets": ["A"], "market": "SPY", "study_start": str(index[3].date()),
                       "validation_start": str(index[5].date()), "test_start": str(index[7].date()),
                       "end_exclusive": "2020-02-01", "window": 2, "annualization": 252},
            "files_sha256": {p.name: sha256_file(p) for p in tmp_path.glob("*.parquet")}}
    write_json(tmp_path / "metadata.json", meta)
    (tmp_path / "metadata.sha256").write_text(sha256_file(tmp_path / "metadata.json") + "  metadata.json\n")
    for split in ("train", "validation", "test"):
        data = load_stage1_market(tmp_path, split, acknowledge_synthetic_prices=True)
        assert data.features.index.equals(tables[f"features_{split}"].index)
        assert data.assets == ("A",) and data.sigma_ref == reference
    with (tmp_path / "vendor_open.parquet").open("ab") as stream:
        stream.write(b"damaged")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_stage1_market(tmp_path, "train", acknowledge_synthetic_prices=True)


def test_invalid_financial_parameters_and_market_grids():
    with pytest.raises(ValueError):
        EnvConfig(commission_bps=10000).validate()
    with pytest.raises(ValueError):
        EnvConfig(lambda_base=-1.).validate()
    data = toy_market([100.] * 4)
    with pytest.raises(ValueError, match="preceding close"):
        replace(data, features=pd.DataFrame({"sigma__SPY": .2}, index=data.adjusted_close.index))
    with pytest.raises(ValueError):
        rebalance(np.zeros(1), -1., np.ones(1), np.array([0., 1.]), 0.)


def test_extreme_but_valid_commission_never_borrows_cash():
    env = make_env(toy_market(np.full((3, 2), 100.)), commission_bps=9999.)
    env.reset()
    for weights in ([1., 0., 0.], [0., 1., 0.]):
        info = env.step(action(weights))[-1]
        assert info["cash"] >= 0. and info["portfolio_value"] > 0.
        assert info["nav_post_trade_open"] == pytest.approx(info["nav_pre_trade_open"] - info["commission"])


def test_short_scenario_reconciles_and_is_identical_across_modes():
    from src.env_validation import deterministic_rollout

    data = toy_market([[100., 100.], [100., 100.], [110., 90.], [105., 95.]])
    scalar, ledger = deterministic_rollout(data, EnvConfig(initial_capital=1000.),
        [action([1., 0., 0.]).tolist(), action([0., 1., 0.]).tolist(), action([0., 0., 1.]).tolist()])
    assert len(scalar) == 9 and len(ledger) == 18
    assert set(scalar["mode"]) == {"none", "fixed", "adaptive"}
    assert scalar.groupby("mode")["nav_close"].last().nunique() == 1
    assert scalar.groupby("mode")["terminated"].last().all()


def test_constant_per_asset_rebasing_does_not_change_finances():
    data = toy_market([[100., 100.], [101., 99.], [110., 90.]],
                      [[100., 100.], [103., 98.], [109., 91.]])
    scale = np.array([.01, 1000.])
    rebased = replace(data, adjusted_open=data.adjusted_open * scale,
                      adjusted_close=data.adjusted_close * scale)
    a, b = make_env(data), make_env(rebased)
    a.reset(seed=42); b.reset(seed=42)
    for weights in ([.5, .5, 0.], [0., 1., 0.]):
        ia = a.step(action(weights))[-1]
        ib = b.step(action(weights))[-1]
        for key in ("portfolio_value", "commission", "turnover", "r_net_log", "weights"):
            np.testing.assert_allclose(ia[key], ib[key], rtol=1e-12, atol=1e-10)
