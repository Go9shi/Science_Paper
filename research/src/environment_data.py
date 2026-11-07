"""Explicit Stage 2 synthetic adjusted-price contract; Stage 1 is never rewritten."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import _check_index, build_features, sha256_file
from .volatility import fit_sigma_ref

PRICE_MODEL = "synthetic_adjusted_open_close_v1"


@dataclass(frozen=True)
class MarketData:
    adjusted_open: pd.DataFrame
    adjusted_close: pd.DataFrame
    features: pd.DataFrame
    sigma_ref: float
    sigma_column: str = "sigma__SPY"
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        for name in ("adjusted_open", "adjusted_close", "features"):
            table = getattr(self, name).copy(deep=True)
            _check_index(table.index, name)
            if table.empty or table.columns.has_duplicates or not np.isfinite(table.to_numpy(dtype=float)).all():
                raise ValueError(f"{name}: require complete finite values and unique columns")
            object.__setattr__(self, name, table)
        if not self.adjusted_open.index.equals(self.adjusted_close.index) or not self.adjusted_open.columns.equals(self.adjusted_close.columns):
            raise ValueError("Open and close grids/assets must match exactly")
        if (self.adjusted_open <= 0).any().any() or (self.adjusted_close <= 0).any().any():
            raise ValueError("Synthetic prices must be positive")
        locs = self.adjusted_close.index.get_indexer(self.features.index)
        if (locs < 1).any() or (len(locs) > 1 and not np.all(np.diff(locs) == 1)):
            raise ValueError("Decision dates must be consecutive price sessions with a preceding close")
        if self.sigma_column not in self.features or (self.features[self.sigma_column] < 0).any():
            raise ValueError("Causal nonnegative market sigma feature is required")
        if not np.isfinite(self.sigma_ref) or self.sigma_ref <= 0:
            raise ValueError("Training-only sigma_ref must be finite and positive")

    @property
    def assets(self) -> tuple[str, ...]:
        return tuple(self.adjusted_close.columns)


def adjusted_execution_prices(
    vendor_open: pd.DataFrame, vendor_close: pd.DataFrame, adjusted_close: pd.DataFrame,
) -> pd.DataFrame:
    """Retrospective synthetic Open, NOT a tradable nominal cash-share quote."""
    for frame in (vendor_close, adjusted_close):
        if not vendor_open.index.equals(frame.index) or not vendor_open.columns.equals(frame.columns):
            raise ValueError("Price bases have different grids/assets")
    for frame in (vendor_open, vendor_close, adjusted_close):
        if not np.isfinite(frame.to_numpy(dtype=float)).all() or (frame <= 0).any().any():
            raise ValueError("Price bases must be complete, finite and positive")
    return vendor_open * (adjusted_close / vendor_close)


def load_stage1_market(run_directory: Path, split: str, *, acknowledge_synthetic_prices: bool = False) -> MarketData:
    if not acknowledge_synthetic_prices:
        raise ValueError("Explicitly acknowledge the synthetic adjusted-price execution assumption")
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation or test")
    root = Path(run_directory)
    meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if meta.get("status") != "complete":
        raise ValueError("Stage 1 run is not complete")
    sidecar = (root / "metadata.sha256").read_text(encoding="utf-8").split()
    if len(sidecar) != 2 or sidecar != [sha256_file(root / "metadata.json"), "metadata.json"]:
        raise ValueError("Stage 1 metadata checksum mismatch")
    config = meta["config"]
    assets = list(config["assets"])
    if not assets or config["market"] in assets or len(set(assets)) != len(assets):
        raise ValueError("Invalid Stage 1 asset universe")

    def read(name):
        path = root / f"{name}.parquet"
        if sha256_file(path) != meta["files_sha256"].get(path.name):
            raise ValueError(f"Stage 1 checksum mismatch: {path.name}")
        return pd.read_parquet(path)

    opens, closes, adjusted = (read(n) for n in ("vendor_open", "vendor_close", "adjusted_close"))
    factors = read("adjustment_factor")
    pd.testing.assert_frame_equal(factors, adjusted / closes, check_freq=False, check_exact=True)
    features = read(f"features_{split}")
    train = read("features_train")
    limits = {
        "train": (config["study_start"], config["validation_start"]),
        "validation": (config["validation_start"], config["test_start"]),
        "test": (config["test_start"], config["end_exclusive"]),
    }
    for name, table in (("train", train), (split, features)):
        start, end = limits[name]
        expected = adjusted.index[(adjusted.index >= start) & (adjusted.index < end)]
        if not table.index.equals(expected):
            raise ValueError(f"Incomplete or crossing {name} decision dates")
    # Check feature causality by reproducing the declared lagged Stage 1 pipeline.
    expected_features = build_features(adjusted, config["window"], config["annualization"])
    for table in (train, features):
        pd.testing.assert_frame_equal(table, expected_features.loc[table.index], check_freq=False, check_exact=True)
    sigma_col = f"sigma__{config['market']}"
    reference = fit_sigma_ref(train[sigma_col], train_end_exclusive=config["validation_start"])
    if reference != meta["sigma_ref"]:
        raise ValueError("Stage 1 sigma_ref does not match the training median")
    synthetic_open = adjusted_execution_prices(opens[assets], closes[assets], adjusted[assets])
    return MarketData(
        synthetic_open, adjusted[assets], features, float(meta["sigma_ref"]), sigma_col,
        {"stage1_run_id": meta["run_id"], "stage1_path": str(root), "split": split,
         "price_model": PRICE_MODEL, "input_sha256": meta["files_sha256"],
         "stage1_metadata_sha256": sha256_file(root / "metadata.json")},
    )
