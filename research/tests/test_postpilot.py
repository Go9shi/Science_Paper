"""Synthetic regression tests for Colab. No research files or real test data."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import sha256_file, write_json
from src.environment_data import MarketData
from src.out_of_sample import _authorized_lock, evaluate_locked_test
from src.paired_analysis import paired_analysis, stationary_indices
from src.postpilot import freeze_decisions, verify_decision_lock, seal_report, verified_report
from src.ppo_training import build_ppo, save_checkpoint, load_saved_policy, evaluate_policy_financial, _contract
from src.replay_audit import _exact_tables
from src.training_config import TrainingConfig


def panel(*, identical=False, constant=False):
    """Explicit synthetic series; not an experimental result or fitted policy."""
    dates = pd.bdate_range('2019-01-01', periods=12)
    returns = np.zeros(12) if constant else np.array([.01, -.02, .03, .005, -.01, .02] * 2)
    result = {}
    for mode in ('none', 'fixed', 'adaptive'):
        for seed in (11, 23, 47):
            high = np.array([False, True] * 6)
            turnover = np.full(12, .2)
            if mode == 'adaptive' and not identical:
                turnover[high] = .1
            result[(mode, seed)] = pd.DataFrame({'date': dates.strftime('%Y-%m-%d'),
                'data_as_of': (dates - pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
                'net_return': returns, 'portfolio_value': 1000 * np.cumprod(1 + returns),
                'turnover': turnover, 'commission': np.full(12, .01),
                'sigma_t': np.where(high, .3, .1), 'lambda_t': .001})
    return result


def bootstrap():
    return {'method': 'paired_stationary_circular', 'resamples': 24, 'seed': 7,
            'expected_block_lengths': [2, 4], 'confidence': .95,
            'seed_resampling': False, 'interval': 'percentile'}


def test_paired_identical_paths_have_exact_zero_interval():
    tables = paired_analysis(panel(identical=True), seeds=[11, 23, 47], sigma_ref=.2, bootstrap=bootstrap())
    intervals = tables['bootstrap_intervals']
    primary = intervals[(intervals.regime == 'overall') & (intervals.metric == 'net_sharpe')]
    assert (primary[['point', 'lower', 'upper']] == 0.).all().all()
    assert (primary.invalid_resamples == 0).all()
    defined = intervals[intervals.lower.notna()]
    assert (defined[['point', 'lower', 'upper']] == 0.).all().all()
    assert (tables['paired_metrics'].difference == 0.).all()


def test_turnover_mechanism_uses_common_train_threshold_and_paired_seeds():
    tables = paired_analysis(panel(), seeds=[11, 23, 47], sigma_ref=.2, bootstrap=bootstrap())
    intervals = tables['bootstrap_intervals']
    interaction = intervals[intervals.metric == 'turnover_interaction']
    assert interaction.point.tolist() == pytest.approx([-.1, -.1])
    assert tables['turnover_mechanism_by_seed'].high_minus_low_interaction.tolist() == pytest.approx([-.1] * 3)
    regimes = tables['regime_metrics']
    assert set(regimes.observations) == {6}
    assert regimes.cagr.isna().all() and regimes.maximum_drawdown.isna().all()
    # Path-dependent metrics must not be computed on spliced regime dates.
    assert set(regimes.path_metrics_status) == {'not_defined_for_discontinuous_regime_subset'}


@pytest.mark.parametrize('change', ['dates', 'sigma', 'future', 'missing_seed'])
def test_pairing_rejects_misaligned_or_future_inputs(change):
    frames = panel()
    key = ('adaptive', 11)
    if change == 'dates':
        frames[key].loc[0, 'date'] = '2018-12-31'
    elif change == 'sigma':
        frames[key].loc[0, 'sigma_t'] = .9
    elif change == 'future':
        frames[key].loc[0, 'data_as_of'] = '2030-01-01'
    else:
        frames.pop(key)
    with pytest.raises(ValueError):
        paired_analysis(frames, seeds=[11, 23, 47], sigma_ref=.2, bootstrap=bootstrap())


def test_undefined_sharpe_does_not_create_a_finite_bootstrap_interval():
    tables = paired_analysis(panel(constant=True), seeds=[11, 23, 47], sigma_ref=.2, bootstrap=bootstrap())
    intervals = tables['bootstrap_intervals']
    sharpe = intervals[intervals.metric == 'net_sharpe']
    assert sharpe[['point', 'lower', 'upper']].isna().all().all()
    assert (sharpe.invalid_resamples == bootstrap()['resamples']).all()


def test_stationary_indices_are_reproducible_and_circular():
    a = stationary_indices(12, 4, np.random.default_rng(17))
    b = stationary_indices(12, 4, np.random.default_rng(17))
    np.testing.assert_array_equal(a, b)
    assert len(a) == 12 and (a >= 0).all() and (a < 12).all()
    with pytest.raises(ValueError):
        stationary_indices(12, 13, np.random.default_rng(17))


def test_test_access_disabled_before_reading_files(tmp_path):
    with pytest.raises(ValueError, match='disabled'):
        evaluate_locked_test(tmp_path / 'absent_batch', tmp_path / 'absent_stage1',
            tmp_path / 'absent_seal', tmp_path / 'outputs', expected_lock_sha256='',
            evaluator_sha='0' * 40, project_root=tmp_path)
    assert not (tmp_path / 'outputs').exists()


def test_nonexact_replay_cannot_seal_decisions(tmp_path, monkeypatch):
    # Mock gate objects are unit-test fixtures, not evidence of actual replay.
    monkeypatch.setattr('src.postpilot.verified_batch', lambda *a: ({}, {}))
    monkeypatch.setattr('src.postpilot.verified_report', lambda p, n:
        {'status': 'diagnostic_complete', 'all_27_exact': False})
    monkeypatch.setattr('src.postpilot._verified_test_evidence', lambda *a: {})
    with pytest.raises(ValueError, match='blocked'):
        freeze_decisions(tmp_path, tmp_path, tmp_path, {}, evaluator_sha='0' * 40, project_root=tmp_path)
    assert not (tmp_path / 'decision_lock.json').exists()


def test_lock_and_report_tampering_is_detected(tmp_path):
    (tmp_path / 'evidence.txt').write_text('synthetic fixture')
    seal_report(tmp_path, 'report.json', {'status': 'synthetic_test_only'})
    assert verified_report(tmp_path, 'report.json')['status'] == 'synthetic_test_only'
    (tmp_path / 'evidence.txt').write_text('modified fixture')
    with pytest.raises(ValueError, match='Checksum'):
        verified_report(tmp_path, 'report.json')
    write_json(tmp_path / 'decision_lock.json', {'synthetic': True})
    (tmp_path / 'decision_lock.sha256').write_text('0' * 64 + '  decision_lock.json\n')
    with pytest.raises(ValueError, match='checksum'):
        verify_decision_lock(tmp_path, evaluator_sha='0' * 40, project_root=tmp_path)
    with pytest.raises(ValueError, match='pre-recorded'):
        _authorized_lock(tmp_path, '1' * 64, '0' * 40, tmp_path, True)


def test_test_evaluation_is_explicit_and_preserves_checkpoint_config(tmp_path):
    dates = pd.bdate_range('2018-01-01', periods=12)
    prices = pd.DataFrame({'A': np.arange(100., 112.)}, index=dates)
    features = pd.DataFrame({'observed_return__A': .01, 'sigma__SPY': .2}, index=dates[1:])
    train = MarketData(prices, prices, features, .2, provenance={'split': 'train', 'synthetic': True})
    test = MarketData(prices, prices, features, .2, provenance={'split': 'test', 'synthetic': True})
    config = TrainingConfig(n_steps=8, batch_size=4, n_epochs=1, total_timesteps=16,
                            checkpoint_steps=8, policy_layers=(8,))
    model, vector, normalizer = build_ppo(train, config)
    final = tmp_path / 'checkpoints/final'
    # Untrained random weights solely to test inference API plumbing, not PPO results.
    save_checkpoint(model, normalizer, config, final, _contract(train), phase='synthetic_no_training_fixture')
    vector.close()
    before = sha256_file(final / 'config.json')
    with pytest.raises(ValueError, match='requires validation'):
        load_saved_policy(tmp_path, test)
    loaded, evaluation, frozen, saved = load_saved_policy(tmp_path, test, evaluation_split='test')
    try:
        with pytest.raises(ValueError, match='requires validation'):
            evaluate_policy_financial(loaded, evaluation, frozen, test, saved)
        records, metrics = evaluate_policy_financial(loaded, evaluation, frozen, test, saved, evaluation_split='test')
        assert len(records) == len(features) and metrics['observations'] == len(features)
        assert saved == config and not frozen.training and not frozen.norm_reward
    finally:
        evaluation.close()
    assert sha256_file(final / 'config.json') == before


def test_tiny_float_difference_is_not_exact_replay_pass():
    left = pd.DataFrame({'portfolio_value': [1000., 1001.]})
    right = left.copy()
    right.iloc[1, 0] = np.nextafter(right.iloc[1, 0], np.inf)
    assert _exact_tables(left, right)['exact'] is False


def test_postpilot_notebook_defaults_do_not_open_test_or_replay():
    root = Path(__file__).resolve().parents[1]
    for name in ('05_validation_and_freeze.ipynb', '06_locked_test_evaluation.ipynb'):
        notebook = json.loads((root / 'notebooks' / name).read_text(encoding='utf-8'))
        code = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
        assert 'ENABLE_TRAINING = True' not in code and '.learn(' not in code
        if name.startswith('05'):
            assert 'RUN_REPLAY = False' in code and 'RUN_ANALYSIS = False' in code and 'SEAL_DECISIONS = False' in code
        else:
            assert 'ENABLE_TEST = False' in code and 'LOCK_SHA256 = ""' in code
