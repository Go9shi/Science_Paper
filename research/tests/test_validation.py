"""Artifact-integrity tests on temporary synthetic fixtures; Colab execution only."""
from dataclasses import asdict
import json

import numpy as np
import pandas as pd
import pytest

from src.data import (
    DataConfig, build_features, chronological_split, compute_returns, corporate_action_audit,
    download_snapshot, expected_sessions, sha256_file, synchronized_prices, write_json,
)
from src.validation import pytest_summary, verify_file_map, verify_saved_run
from src.volatility import fit_sigma_ref


def test_junit_counts_are_derived_from_actual_cases(tmp_path):
    report = tmp_path / "pytest.xml"
    report.write_text('<testsuites><testsuite tests="2" failures="0" errors="0">'
                      '<testcase name="synthetic_a"/><testcase name="synthetic_b"/>'
                      '</testsuite></testsuites>', encoding="utf-8")
    assert pytest_summary(report) == {"status": "passed", "tests": 2, "failures": 0, "errors": 0, "skipped": 0}


@pytest.mark.parametrize("contents", [
    '<testsuites/>',
    '<testsuite><testcase><failure/></testcase></testsuite>',
    '<testsuite><testcase><skipped/></testcase></testsuite>',
    '<testsuite errors="1"><testcase/></testsuite>',
])
def test_empty_failed_or_skipped_junit_cannot_mark_run_complete(tmp_path, contents):
    report = tmp_path / "pytest.xml"
    report.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match="empty or contains"):
        pytest_summary(report)


def test_checksum_verification_detects_tampering(tmp_path):
    path = tmp_path / "synthetic.txt"
    path.write_text("original synthetic fixture", encoding="utf-8")
    checksums = {path.name: sha256_file(path)}
    verify_file_map(tmp_path, checksums)
    path.write_text("modified synthetic fixture", encoding="utf-8")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_file_map(tmp_path, checksums)


def test_checksum_manifest_cannot_escape_run_directory(tmp_path):
    with pytest.raises(ValueError, match="Missing/unsafe"):
        verify_file_map(tmp_path, {"../outside.txt": "unused"})


def test_unfinished_run_is_not_validated(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({"status": "started"}), encoding="utf-8")
    with pytest.raises(ValueError, match="not prepared"):
        verify_saved_run(tmp_path)


def test_unexecuted_tests_cannot_mark_run_complete(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({
        "status": "prepared", "tests": {"status": "not_run"},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="pytest has not"):
        verify_saved_run(tmp_path)


@pytest.fixture
def synthetic_prepared_run(tmp_path, monkeypatch):
    """Test-only artifacts in pytest tmpdir; never outputs of a research run."""
    import yfinance as yf

    config = DataConfig(assets=("SYNTH",), download_start="2020-01-02", study_start="2020-01-08",
                        validation_start="2020-01-13", test_start="2020-01-16",
                        end_exclusive="2020-01-22", window=2)
    sessions = expected_sessions(config.download_start, config.end_exclusive)
    close = pd.Series(100. + np.arange(len(sessions)), index=sessions)
    frame = pd.DataFrame({"Open": close, "High": close + 2, "Low": close - 2, "Close": close,
                          "Adj Close": close, "Volume": 1000., "Dividends": 0., "Stock Splits": 0.})
    monkeypatch.setattr(yf, "download", lambda *args, **kwargs: frame.copy())
    snapshot_dir = tmp_path / "synthetic_snapshot"
    frames, snapshot = download_snapshot(config, snapshot_dir)
    adjusted, opens, volume = synchronized_prices(frames, sessions, config.symbols)
    factors, events, contract = corporate_action_audit(frames)
    parts = chronological_split(build_features(adjusted, window=2), study_start=config.study_start,
                                validation_start=config.validation_start, test_start=config.test_start,
                                end_exclusive=config.end_exclusive)
    run = tmp_path / "synthetic_run"
    run.mkdir()
    tables = {
        "adjusted_close": adjusted, "vendor_open": opens, "volume": volume,
        "vendor_close": pd.DataFrame({s: frames[s]["Close"] for s in config.symbols}),
        "dividends_vendor": pd.DataFrame({s: frames[s]["Dividends"] for s in config.symbols}),
        "stock_splits_vendor": pd.DataFrame({s: frames[s]["Stock Splits"] for s in config.symbols}),
        "adjustment_factor": factors, "corporate_actions": events,
        "simple_returns_close_to_close": compute_returns(adjusted),
        "log_returns_close_to_close": compute_returns(adjusted, "log"),
        **{f"features_{name}": part for name, part in parts.items()},
    }
    for name, table in tables.items():
        table.to_parquet(run / f"{name}.parquet")
    for name in ("pytest.log", "pip-freeze.txt", "train_diagnostics.csv", "train_diagnostics.png", "source.bundle"):
        (run / name).write_text("synthetic unit-test fixture only", encoding="utf-8")
    (run / "pytest.xml").write_text('<testsuite errors="0" failures="0"><testcase name="synthetic"/></testsuite>',
                                     encoding="utf-8")
    write_json(run / "environment.json", {"project_imports": "passed", "drive_read_write": "passed",
                                          "direct_versions": {}})
    write_json(run / "price_contract.json", contract)
    provenance = run / "provenance"
    provenance.mkdir()
    (provenance / "fixture.py").write_text("# synthetic source fixture\n", encoding="utf-8")
    metadata = {
        "run_id": "synthetic-unit-test", "status": "prepared", "git_commit": "synthetic-fixture",
        "config": asdict(config), "direct_dependencies": {},
        "tests": {"status": "passed", "exit_code": 0, "tests": 1, "failures": 0, "errors": 0, "skipped": 0},
        "snapshot": {"path": str(snapshot_dir), "manifest": snapshot, "reused": False,
                     "manifest_sha256": sha256_file(snapshot_dir / "snapshot.json")},
        "source_sha256": {"fixture.py": sha256_file(provenance / "fixture.py")},
        "sigma_ref": fit_sigma_ref(parts["train"]["sigma__SPY"], train_end_exclusive=config.validation_start),
        "splits": {name: {"rows": len(part), "first_execution_date": str(part.index.min().date()),
                          "last_execution_date": str(part.index.max().date()), "missing_feature_values": 0}
                   for name, part in parts.items()},
        "files_sha256": {p.relative_to(run).as_posix(): sha256_file(p) for p in run.rglob("*") if p.is_file()},
    }
    write_json(run / "metadata.json", metadata)
    return run


def test_saved_run_is_reconstructed_from_synthetic_snapshot(synthetic_prepared_run):
    report = verify_saved_run(synthetic_prepared_run)
    assert report["status"] == "verified"
    assert report["assets"] == 1
    assert report["tests"]["tests"] == 1
    assert report["splits"]["train"]["rows"] > 0
    assert not report["execution_accounting_validated"]


def test_rehashed_but_wrong_prepared_features_are_detected(synthetic_prepared_run):
    run = synthetic_prepared_run
    target = run / "features_train.parquet"
    wrong = pd.read_parquet(target)
    wrong.iloc[0, 0] += .1
    wrong.to_parquet(target)
    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    metadata["files_sha256"][target.name] = sha256_file(target)
    write_json(run / "metadata.json", metadata)
    with pytest.raises(AssertionError):
        verify_saved_run(run)
