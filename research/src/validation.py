"""Read-back validation of a Stage 1 Colab run. Execute only in Colab."""
from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from .data import (
    DataConfig, build_features, chronological_split, compute_returns,
    corporate_action_audit, expected_sessions, sha256_file, synchronized_prices,
)
from .volatility import fit_sigma_ref


def pytest_summary(xml_path: Path) -> dict:
    """Derive success/counts from actual JUnit cases, including collection errors."""
    root = ET.parse(xml_path).getroot()
    cases = list(root.iter("testcase"))
    failures = sum(1 for _ in root.iter("failure"))
    errors = sum(1 for _ in root.iter("error"))
    skipped = sum(1 for _ in root.iter("skipped"))
    # Test-suite-level counts catch collection errors with no corresponding case.
    suites = list(root.iter("testsuite"))
    suite_errors = sum(int(s.get("errors", "0")) for s in suites)
    suite_failures = sum(int(s.get("failures", "0")) for s in suites)
    if not cases or failures or errors or skipped or suite_errors or suite_failures:
        raise ValueError("pytest report is empty or contains failures/errors/skips")
    return {"status": "passed", "tests": len(cases), "failures": 0, "errors": 0, "skipped": 0}


def verify_file_map(directory: Path, hashes: dict[str, str]) -> None:
    if not hashes:
        raise ValueError("Checksum manifest is empty")
    directory = Path(directory).resolve()
    for relative, digest in hashes.items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError(f"Missing/unsafe artifact path: {relative}")
        if sha256_file(path) != digest:
            raise ValueError(f"Checksum mismatch: {relative}")


def verify_saved_run(run_directory: Path) -> dict:
    """Verify disk artifacts against their snapshot and frozen preparation config.

    All splits are inspected for technical integrity only. Reference fitting
    remains train-only. This does not validate a trading execution model.
    """
    run_directory = Path(run_directory)
    metadata = json.loads((run_directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("status") not in {"prepared", "complete"}:
        raise ValueError("Run is not prepared for final validation")
    if metadata["status"] == "complete":
        sidecar = (run_directory / "metadata.sha256").read_text(encoding="utf-8").split()
        if len(sidecar) != 2 or sidecar[1] != "metadata.json" or sidecar[0] != sha256_file(run_directory / "metadata.json"):
            raise ValueError("Final metadata checksum mismatch")
    if metadata.get("tests", {}).get("status") != "passed" or metadata["tests"].get("exit_code") != 0:
        raise ValueError("pytest has not completed successfully")
    required = {
        "features_train.parquet", "features_validation.parquet", "features_test.parquet",
        "adjusted_close.parquet", "vendor_open.parquet", "vendor_close.parquet", "volume.parquet",
        "dividends_vendor.parquet", "stock_splits_vendor.parquet", "adjustment_factor.parquet",
        "corporate_actions.parquet", "price_contract.json", "simple_returns_close_to_close.parquet",
        "log_returns_close_to_close.parquet", "pytest.log", "pytest.xml", "pip-freeze.txt",
        "environment.json", "train_diagnostics.csv", "train_diagnostics.png", "source.bundle",
    }
    hashes = metadata.get("files_sha256", {})
    if not required.issubset(hashes):
        raise ValueError(f"Missing required artifacts: {sorted(required - set(hashes))}")
    verify_file_map(run_directory, hashes)
    verify_file_map(run_directory / "provenance", metadata["source_sha256"])
    environment = json.loads((run_directory / "environment.json").read_text(encoding="utf-8"))
    if (environment.get("project_imports") != "passed" or environment.get("drive_read_write") != "passed"
            or environment.get("direct_versions") != metadata["direct_dependencies"]):
        raise ValueError("Environment/import/Drive preflight is not confirmed")
    tests = pytest_summary(run_directory / "pytest.xml")
    if tests != {k: metadata["tests"][k] for k in tests}:
        raise ValueError("Persisted pytest status/counts do not match JUnit")

    snapshot_dir = Path(metadata["snapshot"]["path"])
    if sha256_file(snapshot_dir / "snapshot.json") != metadata["snapshot"]["manifest_sha256"]:
        raise ValueError("Snapshot manifest changed")
    snapshot = json.loads((snapshot_dir / "snapshot.json").read_text(encoding="utf-8"))
    if snapshot != metadata["snapshot"]["manifest"] or snapshot["config"] != metadata["config"]:
        raise ValueError("Snapshot/config mismatch")
    cfg_dict = dict(metadata["config"])
    cfg_dict["assets"] = tuple(cfg_dict["assets"])
    config = DataConfig(**cfg_dict)
    config.validate()
    if set(snapshot["sha256"]) != {f"{s}.parquet" for s in config.symbols}:
        raise ValueError("Snapshot symbol manifest mismatch")
    verify_file_map(snapshot_dir, snapshot["sha256"])
    frames = {s: pd.read_parquet(snapshot_dir / f"{s}.parquet") for s in config.symbols}
    sessions = expected_sessions(config.download_start, config.end_exclusive)
    adjusted, opens, volume = synchronized_prices(frames, sessions, config.symbols)
    factors, actions, contract = corporate_action_audit(frames)
    expected = {
        "adjusted_close": adjusted, "vendor_open": opens, "volume": volume,
        "vendor_close": pd.DataFrame({s: frames[s]["Close"] for s in config.symbols}),
        "dividends_vendor": pd.DataFrame({s: frames[s]["Dividends"] for s in config.symbols}),
        "stock_splits_vendor": pd.DataFrame({s: frames[s]["Stock Splits"] for s in config.symbols}),
        "adjustment_factor": factors, "corporate_actions": actions,
        "simple_returns_close_to_close": compute_returns(adjusted),
        "log_returns_close_to_close": compute_returns(adjusted, "log"),
    }
    with (run_directory / "price_contract.json").open(encoding="utf-8") as stream:
        if json.load(stream) != contract:
            raise ValueError("Corporate action contract changed")
    features = build_features(adjusted, config.window, config.annualization)
    partitions = chronological_split(
        features, study_start=config.study_start, validation_start=config.validation_start,
        test_start=config.test_start, end_exclusive=config.end_exclusive,
    )
    expected.update({f"features_{name}": part for name, part in partitions.items()})
    summaries = {}
    for name, table in expected.items():
        saved = pd.read_parquet(run_directory / f"{name}.parquet")
        pd.testing.assert_frame_equal(saved, table, check_freq=False, check_exact=True)
        if name.startswith("features_"):
            summaries[name.removeprefix("features_")] = {
                "rows": len(saved), "first_execution_date": str(saved.index.min().date()),
                "last_execution_date": str(saved.index.max().date()),
                "missing_feature_values": int(saved.isna().sum().sum()),
            }
    if summaries != metadata["splits"]:
        raise ValueError("Read-back split summaries differ from metadata")
    reference = fit_sigma_ref(partitions["train"][f"sigma__{config.market}"],
                              train_end_exclusive=config.validation_start)
    if not np.isclose(reference, metadata["sigma_ref"], rtol=1e-12, atol=1e-12):
        raise ValueError("sigma_ref differs from the training-only median")
    return {
        "status": "verified", "run_id": metadata["run_id"], "tests": tests,
        "splits": summaries, "symbols": len(config.symbols), "assets": len(config.assets),
        "snapshot_sessions": len(sessions), "sigma_ref": reference,
        "verified_run_files": len(hashes), "verified_snapshot_files": len(snapshot["sha256"]),
        "run_directory": str(run_directory), "snapshot_directory": str(snapshot_dir),
        "source_commit": metadata["git_commit"], "snapshot_reused": metadata["snapshot"]["reused"],
        "execution_accounting_validated": False,
    }
