"""Known-value and causality checks; execute exclusively in Google Colab."""
import numpy as np
import pandas as pd
import pytest

from src.volatility import fit_sigma_ref, historical_volatility


def test_known_uncentered_volatility_and_lag():
    # At session 4 the three completed returns are .01, -.02, .02.
    index = pd.bdate_range("2020-01-01", periods=5)
    returns = pd.Series([.01, -.02, .02, .8, -.7], index=index)
    sigma = historical_volatility(returns, window=3, annualization=252)
    assert sigma.iloc[:3].isna().all()
    assert sigma.iloc[3] == pytest.approx(np.sqrt(252 * .0003))


def test_constant_return_is_not_centered_standard_deviation():
    returns = pd.Series([.01] * 5, index=pd.bdate_range("2020-01-01", periods=5))
    result = historical_volatility(returns, window=3)
    assert result.iloc[3] == pytest.approx(.01 * np.sqrt(252))


def test_zero_returns_give_zero_and_reference_fails_explicitly():
    returns = pd.Series([0.] * 6, index=pd.bdate_range("2020-01-01", periods=6))
    sigma = historical_volatility(returns, window=3).dropna()
    assert (sigma == 0.).all()
    with pytest.raises(ValueError, match="sigma_ref is zero"):
        fit_sigma_ref(sigma, train_end_exclusive="2021-01-01")


def test_future_returns_do_not_change_past_or_current_sigma():
    index = pd.bdate_range("2020-01-01", periods=8)
    original = pd.Series([.01, -.02, .03, -.01, .02, .01, -.01, .02], index=index)
    changed = original.copy()
    changed.iloc[4:] = 1.
    pd.testing.assert_series_equal(
        historical_volatility(original, window=3).iloc[:5],
        historical_volatility(changed, window=3).iloc[:5],
    )


def test_missing_return_requires_a_full_new_window():
    returns = pd.Series([.01, .02, np.nan, .01, .02, .03, .04],
                        index=pd.bdate_range("2020-01-01", periods=7))
    result = historical_volatility(returns, window=3)
    assert result.iloc[:6].isna().all()
    assert np.isfinite(result.iloc[6])


def test_reference_uses_training_median_and_rejects_test_dates():
    index = pd.to_datetime(["2018-12-27", "2018-12-28", "2018-12-31", "2019-01-02"])
    sigma = pd.Series([.1, .2, .6, 99.], index=index)
    assert fit_sigma_ref(sigma.iloc[:3], train_end_exclusive="2019-01-01") == pytest.approx(.2)
    changed = sigma.copy()
    changed.iloc[3] = 1000.
    assert fit_sigma_ref(changed.iloc[:3], train_end_exclusive="2019-01-01") == pytest.approx(.2)
    with pytest.raises(ValueError, match="outside training"):
        fit_sigma_ref(sigma, train_end_exclusive="2019-01-01")


@pytest.mark.parametrize("values", [[np.nan, .2], [np.inf, .2], [-.1, .2]])
def test_invalid_reference_rejected(values):
    sigma = pd.Series(values, index=pd.bdate_range("2018-01-01", periods=2))
    with pytest.raises(ValueError, match="finite and non-negative"):
        fit_sigma_ref(sigma, train_end_exclusive="2019-01-01")


@pytest.mark.parametrize("window", [0, 1, -1, True, 2.5])
def test_invalid_window(window):
    returns = pd.Series([.01] * 4, index=pd.bdate_range("2020-01-01", periods=4))
    with pytest.raises(ValueError, match="window"):
        historical_volatility(returns, window=window)
