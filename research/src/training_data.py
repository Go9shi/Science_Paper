"""Development-only adapter: never read features_test or materialize 2022+ prices."""
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .data import build_features, sha256_file
from .environment_data import MarketData, PRICE_MODEL, adjusted_execution_prices
from .volatility import fit_sigma_ref


def load_development_market(run_directory: Path, split: str) -> MarketData:
    if split not in {"train", "validation"}:
        raise ValueError("Stage 3 only allows train/validation; test access is forbidden")
    root = Path(run_directory)
    meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if meta.get("status") != "complete":
        raise ValueError("Stage 1 data must be complete")
    if (root / "metadata.sha256").read_text().split() != [sha256_file(root / "metadata.json"), "metadata.json"]:
        raise ValueError("Stage 1 metadata checksum mismatch")
    cfg = meta["config"]
    if (cfg["study_start"], cfg["validation_start"], cfg["test_start"], cfg["end_exclusive"]) != (
            "2010-01-01", "2019-01-01", "2022-01-01", "2026-01-01"):
        raise ValueError("Stage 3 requires the frozen research time protocol")
    cutoff = pd.Timestamp(cfg["validation_start"] if split == "train" else cfg["test_start"])
    reads = []

    def read(name):
        if name not in {"vendor_open", "vendor_close", "adjusted_close", "adjustment_factor", "features_train", "features_validation"}:
            raise ValueError("Unapproved development input")
        path = root / f"{name}.parquet"
        # Hash compressed bytes for integrity, without materializing test observations.
        if sha256_file(path) != meta["files_sha256"].get(path.name):
            raise ValueError(f"Input checksum mismatch: {name}")
        fields = pq.read_schema(path).names
        date_field = "Date" if "Date" in fields else "__index_level_0__"
        if date_field not in fields:
            raise ValueError("Cannot enforce date predicate on this Parquet schema")
        table = pd.read_parquet(path, filters=[(date_field, "<", cutoff)])
        if table.empty or not (table.index < cutoff).all():
            raise ValueError("Development date filter failed")
        reads.append(path.name)
        return table

    opens, closes, adjusted, factor = (read(n) for n in ("vendor_open", "vendor_close", "adjusted_close", "adjustment_factor"))
    pd.testing.assert_frame_equal(factor, adjusted / closes, check_exact=True, check_freq=False)
    train = read("features_train")
    features = train if split == "train" else read("features_validation")
    expected_features = build_features(adjusted, cfg["window"], cfg["annualization"])
    for name, table in (("train", train), (split, features)):
        start = cfg["study_start"] if name == "train" else cfg["validation_start"]
        end = cfg["validation_start"] if name == "train" else cfg["test_start"]
        expected_dates = adjusted.index[(adjusted.index >= start) & (adjusted.index < end)]
        if not table.index.equals(expected_dates):
            raise ValueError("Development split coverage mismatch")
        pd.testing.assert_frame_equal(table, expected_features.loc[table.index], check_exact=True, check_freq=False)
    sigma_column = f"sigma__{cfg['market']}"
    ref = fit_sigma_ref(train[sigma_column], train_end_exclusive=cfg["validation_start"])
    if ref != meta["sigma_ref"]:
        raise ValueError("Training-only reference mismatch")
    assets = cfg["assets"]
    return MarketData(adjusted_execution_prices(opens[assets], closes[assets], adjusted[assets]),
        adjusted[assets], features, ref, sigma_column,
        {"split": split, "stage1_run_id": meta["run_id"], "stage1_path": str(root),
         "snapshot_id": meta["snapshot"]["id"], "snapshot_manifest_sha256": meta["snapshot"]["manifest_sha256"],
         "stage1_metadata_sha256": sha256_file(root / "metadata.json"), "price_model": PRICE_MODEL,
         "read_files": reads, "price_rows_before": str(cutoff.date()), "test_features_read": False})
