"""Colab-only plan/ledger tests. Mock run artifacts are fixtures, never research results."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest

from src.data import sha256_file, write_json
from src.pilot import prepare_pilot_batch, read_pilot_batch, run_pilot_batch, summarize_pilot_batch
from src.pilot import _context, _save
from src.pilot_plan import load_pilot_plan, resolve_pilot_plan
from src.training_config import TrainingConfig

ROOT = Path(__file__).resolve().parents[1]
GIT = "0" * 40


def test_pilot_plan_equal_budgets():
    resolved = load_pilot_plan(ROOT / "configs/pilot_plan.json")
    assert resolved["budget"] == {"runs": 27, "training_steps": 442368, "by_mode": {
        "none": {"runs": 3, "training_steps": 49152}, "fixed": {"runs": 12, "training_steps": 196608},
        "adaptive": {"runs": 12, "training_steps": 196608}}}
    assert {j["seed"] for j in resolved["jobs"]} == {11, 23, 47}
    assert all(j["config"]["evaluation_split"] == "validation" for j in resolved["jobs"])
    keys = {k for k in resolved["common_config"] if k not in {"random_seed", "reward_mode", "lambda_0", "alpha"}}
    assert all(all(j["config"][k] == resolved["common_config"][k] for k in keys) for j in resolved["jobs"])


def test_pilot_plan_rejects_test_unbalanced_grid_and_budget():
    original = json.loads((ROOT / "configs/pilot_plan.json").read_text())
    for changes in ({"evaluation_split": "test"}, {"seeds": [11, 11]},
                    {"fixed_candidates": original["fixed_candidates"][:-1]}, {"max_runs": 2},
                    {"adaptive_candidates": [{**c, "alpha": .5} for c in original["adaptive_candidates"]]}):
        with pytest.raises(ValueError):
            resolve_pilot_plan({**original, **changes}, TrainingConfig())
    bad = deepcopy(original)
    bad["fixed_candidates"][0]["lambda_0"] = float("nan")
    with pytest.raises(ValueError):
        resolve_pilot_plan(bad, TrainingConfig())


def small_plan():
    plan = json.loads((ROOT / "configs/pilot_plan.json").read_text())
    plan.update(seeds=[11, 23], total_timesteps=16, checkpoint_steps=8,
                fixed_candidates=plan["fixed_candidates"][:2], adaptive_candidates=plan["adaptive_candidates"][:2])
    base = TrainingConfig(n_steps=8, batch_size=4, n_epochs=1, policy_layers=(8,))
    return resolve_pilot_plan(plan, base)


def evidence(tmp_path, name):
    directory = tmp_path / name
    directory.mkdir()
    cases = ["test_pilot_plan_equal_budgets", "test_pilot_disabled_does_not_train",
             "test_pilot_resume_preserves_attempts_and_skips_completed", "test_gymnasium_and_sb3_checkers",
             "test_tiny_synthetic_training_saves_and_reloads"]
    # Deliberate JUnit fixture for gate/ledger tests, not a claim about actual test execution.
    (directory / "pytest.xml").write_text('<testsuite tests="5" errors="0" failures="0">' +
        ''.join(f'<testcase name="{case}"/>' for case in cases) + '</testsuite>')
    write_json(directory / "test_gate.json", {"synthetic_unit_test_fixture": True, "git_sha": GIT,
        "tests": {"status": "passed", "tests": 5, "errors": 0, "failures": 0, "skipped": 0, "exit_code": 0},
        "pytest_xml_sha256": sha256_file(directory / "pytest.xml")})
    return directory


def stub_runner(monkeypatch, calls, *, fail_first=False, null_sharpe=False):
    context = {"data_contract": {"synthetic_unit_test_fixture": True}, "versions": {}, "python": sys.version}
    monkeypatch.setattr("src.pilot._context", lambda *args: context)

    def run(train, validation, config, output, *, git_sha, provenance, on_run_created, stage):
        root = Path(output) / ("synthetic-fixture-" + uuid4().hex)
        root.mkdir(parents=True)
        calls.append(provenance["job_id"])
        # Real train_one_run writes started metadata before invoking this callback.
        write_json(root / "metadata.json", {"status": "started", "training_steps": 0})
        on_run_created(root)
        if fail_first and len(calls) == 1:
            write_json(root / "metadata.json", {"status": "failed", "training_steps": 7})
            raise RuntimeError("Intentional synthetic ledger failure")
        cfg = config.to_dict()
        write_json(root / "config.json", cfg)
        final = root / "checkpoints/final"
        final.mkdir(parents=True)
        (final / "model.zip").write_bytes(b"synthetic fixture only, not a PPO model")
        (final / "vecnormalize.pkl").write_bytes(b"synthetic fixture only, not a pickle")
        write_json(final / "config.json", cfg)
        write_json(final / "checkpoint.json", {"status": "complete", "phase": "after_training",
            "steps": cfg["total_timesteps"], "data_contract": context["data_contract"],
            "sha256": {n: sha256_file(final / n) for n in ("model.zip", "vecnormalize.pkl", "config.json")}})
        # Known mock scores exercise aggregation only; no market returns or strategy results are fabricated.
        score = None if null_sharpe else config.lambda_0 + config.alpha
        write_json(root / "validation_metrics.json", {"net_sharpe": score})
        (root / "validation_financials.csv").write_text("synthetic_fixture_only\n")
        write_json(root / "training_metrics.json", {"synthetic_unit_test_fixture": True})
        write_json(root / "reload_check.json", {"status": "passed", "exact_actions_and_finances": True})
        meta = {"synthetic_unit_test_fixture": True, "stage": stage, "status": "complete", "test_used": False,
            "config": cfg, "git_sha": git_sha, "training_steps": cfg["total_timesteps"], **context,
            "files_sha256": {p.relative_to(root).as_posix(): sha256_file(p) for p in root.rglob("*")
                             if p.is_file() and p.name != "metadata.json"}}
        write_json(root / "metadata.json", meta)
        (root / "metadata.sha256").write_text(sha256_file(root / "metadata.json") + "  metadata.json\n")
        return root
    monkeypatch.setattr("src.pilot.train_one_run", run)


def test_pilot_disabled_does_not_train(tmp_path, monkeypatch):
    calls = []
    stub_runner(monkeypatch, calls)
    batch = prepare_pilot_batch(tmp_path, small_plan(), GIT)
    with pytest.raises(ValueError, match="disabled"):
        run_pilot_batch(batch, object(), object(), tmp_path)
    assert calls == []
    report = summarize_pilot_batch(batch)
    assert report["completed_jobs"] == 0 and report["selection"]["selected"] == {}


def test_pilot_notebook_preview_does_not_start_jobs(tmp_path):
    notebook = json.loads((ROOT / "notebooks/04_pilot_experiments.ipynb").read_text(encoding="utf-8"))
    bootstrap = "".join(notebook["cells"][2]["source"])
    assert "ENABLE_TRAINING = False" in bootstrap and "RUN_TESTS = False" in bootstrap
    batch = prepare_pilot_batch(tmp_path / "batches", small_plan(), GIT)
    attempt = tmp_path / "preview-attempt"
    attempt.mkdir()

    def forbidden(*args, **kwargs):
        raise AssertionError("Preview must not start any training jobs")

    namespace = {"ENABLE_TRAINING": False, "RUN_TESTS": False, "attempt_meta": {"tests": {"status": "not_run"}},
                 "BATCH_DIR": batch, "ATTEMPT_DIR": attempt, "run_pilot_batch": forbidden,
                 "summarize_pilot_batch": summarize_pilot_batch, "write_json": write_json}
    exec(compile("".join(notebook["cells"][15]["source"]), "pilot-preview-cell", "exec"), namespace)
    assert namespace["attempt_meta"]["status"] == "preview_complete"
    assert namespace["BATCH_REPORT"]["completed_jobs"] == 0
    assert not (batch / "runs").exists()


def test_pilot_context_rejects_test_before_touching_data():
    with pytest.raises(ValueError, match="test is forbidden"):
        _context(SimpleNamespace(provenance={"split": "test"}),
                 SimpleNamespace(provenance={"split": "validation"}), small_plan())


def test_pilot_resume_preserves_attempts_and_skips_completed(tmp_path, monkeypatch):
    calls = []
    stub_runner(monkeypatch, calls, fail_first=True)
    resolved = small_plan()
    batch = prepare_pilot_batch(tmp_path / "batches", resolved, GIT)
    with pytest.raises(RuntimeError, match="Intentional"):
        run_pilot_batch(batch, object(), object(), evidence(tmp_path, "attempt-one"), enable_training=True)
    report = summarize_pilot_batch(batch)
    assert report["failed_attempts"] == 1 and report["selection"]["selected"] == {}
    run_pilot_batch(batch, object(), object(), evidence(tmp_path, "attempt-two"), enable_training=True)
    report = summarize_pilot_batch(batch)
    assert report["completed_jobs"] == 10 and report["failed_attempts"] == 1 and report["attempts"] == 11
    assert report["training_steps_known_from_attempts"] == 167
    assert report["selection"]["selected"] == {"fixed": "f02", "adaptive": "a02"}
    previous_calls = list(calls)
    run_pilot_batch(batch, object(), object(), evidence(tmp_path, "attempt-three"), enable_training=True)
    assert calls == previous_calls
    assert prepare_pilot_batch(tmp_path / "batches", resolved, GIT, batch_id=batch.name) == batch
    with pytest.raises(ValueError, match="original plan"):
        prepare_pilot_batch(tmp_path / "batches", resolved, "1" * 40, batch_id=batch.name)
    state, _ = read_pilot_batch(batch)
    state["jobs"][resolved["jobs"][0]["job_id"]]["status"] = "pending"
    write_json(batch / "batch.json", state)
    with pytest.raises(ValueError, match="checksum"):
        read_pilot_batch(batch)


def test_pilot_undefined_sharpe_is_not_replaced_by_zero(tmp_path, monkeypatch):
    calls = []
    stub_runner(monkeypatch, calls, null_sharpe=True)
    batch = prepare_pilot_batch(tmp_path / "batches", small_plan(), GIT)
    run_pilot_batch(batch, object(), object(), evidence(tmp_path, "attempt"), enable_training=True)
    report = summarize_pilot_batch(batch)
    assert report["selection"]["status"] == "no_defined_candidate" and report["selection"]["selected"] == {}


def test_pilot_recovers_completed_manifest_without_retraining(tmp_path, monkeypatch):
    calls = []
    stub_runner(monkeypatch, calls)
    batch = prepare_pilot_batch(tmp_path / "batches", small_plan(), GIT)
    run_pilot_batch(batch, object(), object(), evidence(tmp_path, "attempt-one"), enable_training=True)
    state, resolved = read_pilot_batch(batch)
    last = state["jobs"][resolved["jobs"][-1]["job_id"]]
    # Simulate interruption after final run artifacts exist but before ledger completion.
    last["status"] = "running"
    last["attempts"][-1]["status"] = "started"
    state["status"] = "interrupted"
    _save(batch, state)
    write_json(batch / ".run-lock.json", {"owner": "ended-runtime"})
    previous_calls = list(calls)
    attempt = evidence(tmp_path, "attempt-two")
    run_pilot_batch(batch, object(), object(), attempt, enable_training=True, recover_interrupted=True)
    assert calls == previous_calls and (attempt / "recovered-lock.json").is_file()
    state, _ = read_pilot_batch(batch)
    assert state["status"] == "complete"
    assert state["jobs"][resolved["jobs"][-1]["job_id"]]["attempts"][-1]["status"] == "complete_recovered"


def test_pilot_rejects_missing_tests_and_foreign_lock(tmp_path, monkeypatch):
    calls = []
    stub_runner(monkeypatch, calls)
    batch = prepare_pilot_batch(tmp_path / "batches", small_plan(), GIT)
    attempt = evidence(tmp_path, "attempt")
    write_json(batch / ".run-lock.json", {"owner": "other-session"})
    with pytest.raises(ValueError, match="writer lock"):
        run_pilot_batch(batch, object(), object(), attempt, enable_training=True)
    assert calls == []
    (batch / ".run-lock.json").unlink()
    (attempt / "pytest.xml").write_text('<testsuite tests="1"><testcase name="not_full_suite"/></testsuite>')
    with pytest.raises(ValueError, match="checksum"):
        run_pilot_batch(batch, object(), object(), attempt, enable_training=True)
    assert calls == []
