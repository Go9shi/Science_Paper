"""Synthetic numerical tests: run with python -m pytest in Google Colab only."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.data import (
    DataConfig, build_features, chronological_split, compute_returns, corporate_action_audit,
    download_snapshot, expected_sessions, synchronized_prices, validate_bars,
)


def bars(index):
    return pd.DataFrame({
        "Open": 100., "High": 102., "Low": 98., "Close": 101.,
        "Adj Close": 100., "Volume": 1000., "Dividends": 0., "Stock Splits": 0.,
    }, index=index)


def test_known_simple_and_log_returns():
    prices = pd.DataFrame({"A": [100., 110., 99.]}, index=pd.bdate_range("2020-01-01", periods=3))
    simple = compute_returns(prices)
    assert np.isnan(simple.iloc[0, 0])
    np.testing.assert_allclose(simple.iloc[1:, 0], [.1, -.1], atol=1e-14)
    log = compute_returns(prices, "log")
    np.testing.assert_allclose(log.iloc[1:, 0], np.log([1.1, .9]), atol=1e-14)


@pytest.mark.parametrize("bad", [np.nan, np.inf, 0., -1.])
def test_invalid_price_is_not_filled(bad):
    prices = pd.DataFrame({"A": [100., bad, 101.]}, index=pd.bdate_range("2020-01-01", periods=3))
    with pytest.raises(ValueError, match="finite, complete and positive"):
        compute_returns(prices)


def test_missing_session_fails_instead_of_inner_join():
    index = pd.bdate_range("2020-01-01", periods=4)
    with pytest.raises(ValueError, match="calendar mismatch"):
        synchronized_prices({"A": bars(index), "B": bars(index.delete(1))}, index, ("A", "B"))


def test_missing_value_fails_with_date():
    index = pd.bdate_range("2020-01-01", periods=4)
    frame = bars(index)
    frame.loc[index[1], "Adj Close"] = np.nan
    with pytest.raises(ValueError, match="missing/non-finite"):
        validate_bars(frame, index, "A")


def test_duplicate_sessions_fail():
    index = pd.bdate_range("2020-01-01", periods=4)
    frame = pd.concat([bars(index[:2]), bars(index[1:])])
    with pytest.raises(ValueError, match="duplicate dates"):
        validate_bars(frame, index, "A")


def test_unsorted_sessions_fail():
    index = pd.bdate_range("2020-01-01", periods=4)
    with pytest.raises(ValueError, match="not chronological"):
        validate_bars(bars(index[::-1]), index, "A")


def test_ohlc_bounds_fail():
    index = pd.bdate_range("2020-01-01", periods=4)
    frame = bars(index)
    frame.loc[index[1], "High"] = 99.
    with pytest.raises(ValueError, match="OHLC"):
        validate_bars(frame, index, "A")


def test_synchronized_dates_and_asset_order():
    index = pd.bdate_range("2020-01-01", periods=4)
    adjusted, opens, volumes = synchronized_prices({"A": bars(index), "B": bars(index)}, index, ("B", "A"))
    assert adjusted.index.equals(index)
    assert list(adjusted.columns) == ["B", "A"]
    assert (opens == 100.).all().all()
    assert (volumes == 1000.).all().all()


def test_features_ignore_current_and_future_prices():
    index = pd.bdate_range("2020-01-01", periods=9)
    original = pd.DataFrame({"A": [100., 101., 99., 102., 100., 103., 101., 99., 98.]}, index=index)
    changed = original.copy()
    changed.loc[index[5]:, "A"] *= 2
    a = build_features(original, window=3)
    b = build_features(changed, window=3)
    pd.testing.assert_frame_equal(a.loc[:index[5]], b.loc[:index[5]])
    assert a.loc[index[5], "observed_return__A"] == pytest.approx(100 / 102 - 1)
    assert a.loc[index[6], "observed_return__A"] != b.loc[index[6], "observed_return__A"]


def test_chronological_split_half_open_no_overlap_and_full_coverage():
    index = pd.date_range("2020-01-01", periods=8)
    frame = pd.DataFrame({"x": range(8)}, index=index)
    parts = chronological_split(frame, study_start="2020-01-02", validation_start="2020-01-04",
                                test_start="2020-01-06", end_exclusive="2020-01-08")
    assert parts["train"].index.tolist() == list(index[1:3])
    assert parts["validation"].index.tolist() == list(index[3:5])
    assert parts["test"].index.tolist() == list(index[5:7])
    for a, b in (("train", "validation"), ("validation", "test"), ("train", "test")):
        assert parts[a].index.intersection(parts[b].index).empty
        assert parts[a].index.max() < parts[b].index.min()
    pd.testing.assert_frame_equal(pd.concat(parts.values()), frame.iloc[1:7])


def test_split_rejects_bad_boundaries_empty_split_and_missing_features():
    frame = pd.DataFrame({"x": [1., 2., 3.]}, index=pd.date_range("2020-01-01", periods=3))
    with pytest.raises(ValueError, match="strictly increasing"):
        chronological_split(frame, study_start="2020-01-01", validation_start="2020-01-03",
                            test_start="2020-01-02", end_exclusive="2020-01-04")
    with pytest.raises(ValueError, match="Every split"):
        chronological_split(frame, study_start="2020-01-01", validation_start="2020-01-03",
                            test_start="2020-01-05", end_exclusive="2020-01-06")
    frame.iloc[1, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        chronological_split(frame, study_start="2020-01-01", validation_start="2020-01-02",
                            test_start="2020-01-03", end_exclusive="2020-01-04")


def test_calendar_has_holidays_and_exclusive_end():
    assert expected_sessions("2024-07-03", "2024-07-06").tolist() == [
        pd.Timestamp("2024-07-03"), pd.Timestamp("2024-07-05"),
    ]


def test_config_rejects_market_in_assets():
    with pytest.raises(ValueError, match="exclude the market"):
        replace(DataConfig(), assets=("SPY",)).validate()


def test_snapshot_reuse_and_corruption_detection(tmp_path, monkeypatch):
    import yfinance as yf

    config = replace(DataConfig(), assets=("A",))
    calls = []

    def fake_download(symbol, **kwargs):
        calls.append(symbol)
        assert kwargs["auto_adjust"] is False
        return bars(pd.bdate_range("2020-01-01", periods=3))

    monkeypatch.setattr(yf, "download", fake_download)
    location = tmp_path / "snapshot"
    first, manifest = download_snapshot(config, location)
    second, reused = download_snapshot(config, location)
    assert calls == ["A", "SPY"]
    assert manifest == reused
    pd.testing.assert_frame_equal(first["A"], second["A"], check_freq=False)
    with (location / "A.parquet").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checksum mismatch"):
        download_snapshot(config, location)


def test_incomplete_snapshot_is_not_reused(tmp_path):
    with pytest.raises(RuntimeError, match="Incomplete snapshot"):
        download_snapshot(DataConfig(), tmp_path)


def test_cash_dividend_changes_vendor_return_but_not_adjusted_signal():
    index = pd.bdate_range("2020-01-01", periods=4)
    frame = bars(index)
    frame["Close"] = [100., 100., 99., 100.]
    frame["Open"] = frame["Close"]
    frame["High"] = frame["Close"] + 1
    frame["Low"] = frame["Close"] - 1
    frame["Adj Close"] = [99., 99., 99., 100.]
    frame.loc[index[2], "Dividends"] = 1.
    factors, events, contract = corporate_action_audit({"A": frame})
    adjusted = compute_returns(frame[["Adj Close"]])
    vendor = compute_returns(frame[["Close"]])
    assert adjusted.iloc[2, 0] == pytest.approx(0.)
    assert vendor.iloc[2, 0] == pytest.approx(-.01)
    assert factors.iloc[0, 0] == pytest.approx(.99)
    assert events.loc[0, "Date"] == index[2]
    assert events.loc[0, "dividends_vendor"] == 1.
    assert not contract["execution_accounting_validated"]
    assert not contract["same_day_factors_or_actions_are_features"]


def test_split_is_preserved_without_double_adjusting_vendor_prices():
    index = pd.bdate_range("2020-01-01", periods=4)
    frame = bars(index)
    frame["Adj Close"] = frame["Close"]  # Already on vendor split-adjusted basis.
    frame.loc[index[2], "Stock Splits"] = 2.
    original = frame.copy(deep=True)
    factors, events, _ = corporate_action_audit({"A": frame})
    pd.testing.assert_frame_equal(frame, original)
    assert (factors == 1.).all().all()
    assert events.loc[0, "stock_splits_vendor"] == 2.
    assert compute_returns(frame[["Adj Close"]]).iloc[2, 0] == 0.


def test_unexplained_factor_change_is_flagged_not_repaired():
    index = pd.bdate_range("2020-01-01", periods=4)
    frame = bars(index)
    frame["Adj Close"] = [100., 100., 90., 90.]
    _, events, contract = corporate_action_audit({"A": frame})
    assert contract["unexplained_factor_change_rows"] == 1
    assert events.loc[0, "factor_change_without_reported_event"]


def test_current_and_future_execution_prices_and_actions_never_enter_features():
    index = pd.bdate_range("2020-01-01", periods=8)
    frame = bars(index)
    frame["Adj Close"] = [100., 101., 99., 102., 100., 103., 101., 99.]
    baseline = build_features(frame[["Adj Close"]].rename(columns={"Adj Close": "A"}), window=3)
    frame.loc[index[4]:, "Open"] = 1000.
    frame.loc[index[4]:, "Dividends"] = 10.
    frame.loc[index[4]:, "Stock Splits"] = 2.
    result = build_features(frame[["Adj Close"]].rename(columns={"Adj Close": "A"}), window=3)
    pd.testing.assert_frame_equal(baseline, result)


def test_test_prices_do_not_change_training_reference():
    from src.volatility import fit_sigma_ref

    index = pd.bdate_range("2020-01-01", periods=14)
    prices = pd.DataFrame({"SPY": [100., 101., 99., 102., 103., 100., 102.,
                                    104., 101., 105., 106., 103., 104., 108.]}, index=index)
    options = dict(study_start=str(index[4].date()), validation_start=str(index[7].date()),
                   test_start=str(index[10].date()), end_exclusive="2020-02-01")
    before = chronological_split(build_features(prices, window=3), **options)
    changed = prices.copy()
    changed.loc[index[10]:] *= 10
    after = chronological_split(build_features(changed, window=3), **options)
    pd.testing.assert_frame_equal(before["train"], after["train"])
    assert fit_sigma_ref(before["train"]["sigma__SPY"], train_end_exclusive=options["validation_start"]) == \
        fit_sigma_ref(after["train"]["sigma__SPY"], train_end_exclusive=options["validation_start"])
