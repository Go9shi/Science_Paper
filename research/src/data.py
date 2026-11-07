"""Daily Yahoo snapshot acquisition and strict, causal preparation for Colab.

No credentials, training, execution simulator or financial backtest is included.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .volatility import historical_volatility


ASSET_SECTORS = {
    "AAPL": "Information Technology", "MSFT": "Information Technology",
    "JPM": "Financials", "JNJ": "Health Care", "PG": "Consumer Staples",
    "HD": "Consumer Discretionary", "XOM": "Energy", "CAT": "Industrials",
    "VZ": "Communication Services", "NEE": "Utilities", "AMT": "Real Estate",
    "APD": "Materials",
}
BAR_COLUMNS = ("Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits")


@dataclass(frozen=True)
class DataConfig:
    assets: tuple[str, ...] = tuple(ASSET_SECTORS)
    market: str = "SPY"
    download_start: str = "2009-10-01"
    study_start: str = "2010-01-01"
    validation_start: str = "2019-01-01"
    test_start: str = "2022-01-01"
    end_exclusive: str = "2026-01-01"
    window: int = 20
    annualization: int = 252

    @property
    def symbols(self) -> tuple[str, ...]:
        return (*self.assets, self.market)

    def validate(self) -> None:
        dates = [pd.Timestamp(s) for s in (
            self.download_start, self.study_start, self.validation_start,
            self.test_start, self.end_exclusive,
        )]
        if any(a >= b for a, b in zip(dates, dates[1:])):
            raise ValueError("Require download_start < study_start < validation < test < end")
        if not self.assets or len(set(self.symbols)) != len(self.symbols):
            raise ValueError("Assets must be non-empty, unique and exclude the market indicator")
        if self.window < 2 or self.annualization <= 0:
            raise ValueError("Invalid volatility parameters")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    """Replace one JSON file; final manifest is the completion marker on Drive."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def expected_sessions(start: str, end_exclusive: str) -> pd.DatetimeIndex:
    """NYSE daily session labels; includes exceptional closures and early closes."""
    import pandas_market_calendars as mcal

    schedule = mcal.get_calendar("NYSE").schedule(
        start_date=start, end_date=pd.Timestamp(end_exclusive) - pd.Timedelta(days=1)
    )
    return pd.DatetimeIndex(schedule.index).tz_localize(None).rename("Date")


def _check_index(index: pd.Index, label: str) -> None:
    if not isinstance(index, pd.DatetimeIndex) or index.empty:
        raise ValueError(f"{label}: expected non-empty DatetimeIndex")
    if index.tz is not None or index.hasnans or not index.equals(index.normalize()):
        raise ValueError(f"{label}: require timezone-naive daily session labels")
    if index.has_duplicates:
        raise ValueError(f"{label}: duplicate dates: {index[index.duplicated()].tolist()[:5]}")
    if not index.is_monotonic_increasing:
        raise ValueError(f"{label}: dates are not chronological")


def validate_bars(bars: pd.DataFrame, sessions: pd.DatetimeIndex, symbol: str) -> pd.DataFrame:
    """Fail on missing sessions/values or duplicates; never intersect or impute."""
    _check_index(sessions, "calendar")
    _check_index(bars.index, symbol)
    if bars.columns.has_duplicates:
        raise ValueError(f"{symbol}: duplicate columns")
    absent = set(BAR_COLUMNS) - set(bars.columns)
    if absent:
        raise ValueError(f"{symbol}: missing columns {sorted(absent)}")
    missing = sessions.difference(bars.index)
    extra = bars.index.difference(sessions)
    if len(missing) or len(extra):
        raise ValueError(f"{symbol}: calendar mismatch; missing={missing.tolist()[:5]}, extra={extra.tolist()[:5]}")
    clean = bars.loc[sessions, list(BAR_COLUMNS)].copy()
    try:
        values = clean.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{symbol}: non-numeric bar values") from error
    if not np.isfinite(values).all():
        bad = clean.index[~np.isfinite(values).all(axis=1)]
        raise ValueError(f"{symbol}: missing/non-finite observations on {bad.tolist()[:5]}")
    clean = clean.astype(float)
    prices = clean[["Open", "High", "Low", "Close", "Adj Close"]]
    if (prices <= 0).any().any() or (clean[["Volume", "Dividends", "Stock Splits"]] < 0).any().any():
        raise ValueError(f"{symbol}: invalid price, volume or corporate action")
    if ((clean["Low"] > clean[["Open", "Close"]].min(axis=1)) |
        (clean["High"] < clean[["Open", "Close"]].max(axis=1)) |
        (clean["Low"] > clean["High"])).any():
        raise ValueError(f"{symbol}: inconsistent OHLC bounds")
    return clean


def download_snapshot(config: DataConfig, directory: Path, attempts: int = 3) -> tuple[dict[str, pd.DataFrame], dict]:
    """Reuse a checksummed immutable snapshot or fetch each symbol sequentially.

    An incomplete snapshot is never silently reused/overwritten. Choose a new
    directory after inspecting the failed download. No automatic ticker removal.
    """
    import yfinance as yf

    config.validate()
    if not 1 <= attempts <= 3:
        raise ValueError("attempts must be between 1 and 3")
    directory = Path(directory)
    manifest_path = directory / "snapshot.json"
    serialized_config = json.loads(json.dumps(asdict(config)))
    if directory.exists():
        if not manifest_path.is_file():
            raise RuntimeError(f"Incomplete snapshot at {directory}. Inspect it and use a NEW snapshot ID.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != 1 or manifest.get("config") != serialized_config:
            raise ValueError("Snapshot configuration/schema mismatch; use a new snapshot ID")
        expected_names = {f"{s}.parquet" for s in config.symbols}
        if set(manifest.get("sha256", {})) != expected_names:
            raise ValueError("Snapshot manifest has missing/unexpected symbols")
        for name, digest in manifest["sha256"].items():
            if sha256_file(directory / name) != digest:
                raise ValueError(f"Snapshot checksum mismatch: {name}")
        return {s: pd.read_parquet(directory / f"{s}.parquet") for s in config.symbols}, manifest

    directory.mkdir(parents=True, exist_ok=False)
    frames = {}
    for symbol in config.symbols:
        failure = None
        for attempt in range(attempts):
            try:
                frame = yf.download(
                    symbol, start=config.download_start, end=config.end_exclusive,
                    interval="1d", auto_adjust=False, back_adjust=False, repair=False,
                    actions=True, keepna=True, prepost=False, rounding=False,
                    threads=False, progress=False, ignore_tz=True, timeout=30,
                    multi_level_index=False,
                )
                if frame is None or frame.empty:
                    raise ValueError(f"Yahoo returned no data for {symbol}")
                if isinstance(frame.columns, pd.MultiIndex):
                    raise ValueError("Unexpected Yahoo column schema")
                # Daily labels are dates, not UTC instants. Do not shift them by timezone conversion.
                frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).rename("Date")
                frame.to_parquet(directory / f"{symbol}.parquet", index=True)
                frames[symbol] = frame
                break
            except Exception as error:
                failure = error
                if attempt + 1 < attempts:
                    time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"Download failed for {symbol} after {attempts} attempts; snapshot incomplete") from failure
    manifest = {
        "schema": 1, "config": serialized_config,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "Yahoo Finance via yfinance", "yfinance_version": yf.__version__,
        "description": "Vendor tables, not original HTTP payloads; quality validation runs separately",
        "parameters": {"interval": "1d", "auto_adjust": False, "repair": False, "actions": True},
        "sha256": {f"{s}.parquet": sha256_file(directory / f"{s}.parquet") for s in config.symbols},
    }
    write_json(manifest_path, manifest)
    return frames, manifest


def synchronized_prices(
    frames: dict[str, pd.DataFrame], sessions: pd.DatetimeIndex, symbols: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if set(frames) != set(symbols):
        raise ValueError("Downloaded symbols do not match the frozen universe")
    checked = {s: validate_bars(frames[s], sessions, s) for s in symbols}
    adjusted = pd.DataFrame({s: checked[s]["Adj Close"] for s in symbols})
    opens = pd.DataFrame({s: checked[s]["Open"] for s in symbols})
    volumes = pd.DataFrame({s: checked[s]["Volume"] for s in symbols})
    return adjusted, opens, volumes


def compute_returns(prices: pd.DataFrame, kind: str = "simple") -> pd.DataFrame:
    """Close-to-close research returns; first row remains undefined."""
    _check_index(prices.index, "prices")
    if prices.empty or prices.columns.has_duplicates:
        raise ValueError("Prices require non-empty unique columns")
    if not np.isfinite(prices.to_numpy(dtype=float)).all() or (prices <= 0).any().any():
        raise ValueError("Prices must be finite, complete and positive; no imputation")
    if kind == "simple":
        return prices.pct_change(fill_method=None)
    if kind == "log":
        return np.log(prices).diff()
    raise ValueError("kind must be 'simple' or 'log'")


def build_features(adjusted_close: pd.DataFrame, window: int = 20, annualization: int = 252) -> pd.DataFrame:
    """Only data completed BEFORE the indexed session's open enters features."""
    simple = compute_returns(adjusted_close)
    log = compute_returns(adjusted_close, "log")
    features = simple.shift(1).add_prefix("observed_return__")
    sigma = historical_volatility(log, window, annualization).add_prefix("sigma__")
    return features.join(sigma)


def corporate_action_audit(
    frames: dict[str, pd.DataFrame], *, factor_change_tolerance: float = 1e-5,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Archive price-basis diagnostics, not an execution/total-return simulator.

    AdjClose/Close is a retrospective vendor factor. Same-day factors and actions
    belong to the audit table, never the pre-open observation. A positive factor
    does not establish a complete split/dividend accounting model.
    """
    if not frames or not 0 < factor_change_tolerance < 1:
        raise ValueError("Require non-empty frames and a factor tolerance in (0, 1)")
    sessions = next(iter(frames.values())).index
    factors = {}
    events = []
    for symbol, frame in frames.items():
        clean = validate_bars(frame, sessions, symbol)
        factor = clean["Adj Close"] / clean["Close"]
        if not np.isfinite(factor.to_numpy()).all() or (factor <= 0).any():
            raise ValueError(f"{symbol}: invalid adjustment factor")
        factors[symbol] = factor
        gains = frame["Capital Gains"] if "Capital Gains" in frame else pd.Series(0., index=sessions)
        if not np.isfinite(gains.to_numpy(dtype=float)).all() or (gains < 0).any():
            raise ValueError(f"{symbol}: invalid Capital Gains observations")
        change = factor.pct_change(fill_method=None)
        cash_event = clean["Dividends"].ne(0) | gains.ne(0)
        split_event = clean["Stock Splits"].ne(0)
        material_change = change.abs().gt(factor_change_tolerance)
        table = pd.DataFrame({
            "symbol": symbol, "dividends_vendor": clean["Dividends"],
            "stock_splits_vendor": clean["Stock Splits"], "capital_gains_vendor": gains,
            "adjustment_factor": factor, "factor_relative_change": change,
            "factor_change_without_reported_event": material_change & ~(cash_event | split_event),
        }, index=sessions)
        table.index.name = "Date"
        events.append(table.loc[cash_event | split_event | material_change].reset_index())
    event_table = pd.concat(events, ignore_index=True)
    report = {
        "status": "feature_prices_separated_from_execution_prices",
        "execution_accounting_validated": False,
        "execution_accounting_requirement": "Choose a consistent split/share/dividend convention before implementing NAV",
        "adjusted_close_is_execution_price": False,
        "same_day_factors_or_actions_are_features": False,
        "factor_change_tolerance": factor_change_tolerance,
        "audit_rows": len(event_table),
        "unexplained_factor_change_rows": int(event_table["factor_change_without_reported_event"].sum()),
        "provider_preprocessing": "yfinance tables may already be sorted/deduplicated; not raw HTTP or point-in-time",
    }
    return pd.DataFrame(factors, index=sessions), event_table, report


def chronological_split(
    frame: pd.DataFrame, *, study_start: str, validation_start: str,
    test_start: str, end_exclusive: str,
) -> dict[str, pd.DataFrame]:
    """Half-open chronological partitions; caller supplies already causal features."""
    _check_index(frame.index, "split input")
    bounds = [pd.Timestamp(x) for x in (study_start, validation_start, test_start, end_exclusive)]
    if any(a >= b for a, b in zip(bounds, bounds[1:])):
        raise ValueError("Split boundaries must be strictly increasing")
    selected = frame.loc[(frame.index >= bounds[0]) & (frame.index < bounds[-1])]
    if selected.empty or not np.isfinite(selected.to_numpy(dtype=float)).all():
        raise ValueError("Study features are empty or non-finite; check warm-up/data quality")
    parts = {
        name: selected.loc[(selected.index >= lower) & (selected.index < upper)].copy()
        for name, lower, upper in zip(("train", "validation", "test"), bounds, bounds[1:])
    }
    if any(part.empty for part in parts.values()):
        raise ValueError("Every split must contain observations")
    if not pd.concat(parts.values()).index.equals(selected.index):
        raise AssertionError("Partitions do not cover the study period exactly once")
    return parts
