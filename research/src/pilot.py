"""Durable small-pilot ledger; all numerical execution belongs in Google Colab."""
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
from uuid import uuid4

import pandas as pd
import torch

from .data import sha256_file, write_json
from .pilot_plan import canonical_digest
from .ppo_training import _contract, configure_torch_runtime, torch_runtime_info, train_one_run
from .training_config import TrainingConfig
from .validation import pytest_summary, verify_file_map


def _stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]


def _save(root, state):
    # The checksum is inside the same atomically replaced JSON, avoiding a stale
    # external sidecar if Colab dies between two separate file replacements.
    state["state_sha256"] = canonical_digest({k: v for k, v in state.items() if k != "state_sha256"})
    write_json(Path(root) / "batch.json", state)


def read_pilot_batch(root):
    root = Path(root)
    state = json.loads((root / "batch.json").read_text(encoding="utf-8"))
    if state["state_sha256"] != canonical_digest({k: v for k, v in state.items() if k != "state_sha256"}):
        raise ValueError("Batch checksum mismatch")
    resolved = json.loads((root / "resolved_plan.json").read_text(encoding="utf-8"))
    if resolved["sha256"] != canonical_digest({k: v for k, v in resolved.items() if k != "sha256"}):
        raise ValueError("Resolved-plan checksum mismatch")
    if state["plan_sha256"] != resolved["sha256"] or list(state["jobs"]) != [j["job_id"] for j in resolved["jobs"]]:
        raise ValueError("Batch/plan job set or order mismatch")
    return state, resolved


def prepare_pilot_batch(output_root, resolved, git_sha, *, batch_id=""):
    if not re.fullmatch(r"[0-9a-f]{40}", git_sha):
        raise ValueError("Record the full resolved Git SHA")
    batch_id = batch_id or _stamp()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", batch_id):
        raise ValueError("Unsafe batch ID")
    root = Path(output_root) / batch_id
    if root.exists():
        state, previous = read_pilot_batch(root)
        if previous != resolved or state["git_sha"] != git_sha:
            raise ValueError("Resume requires the original plan, device and Git SHA; otherwise use a new batch")
        return root
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "resolved_plan.json", resolved)
    state = {"schema": 1, "stage": 4, "batch_id": batch_id, "git_sha": git_sha,
        "plan_sha256": resolved["sha256"], "status": "planned", "context": None, "attempts": [],
        "test_used": False, "jobs": {j["job_id"]: {"status": "pending", "attempts": []} for j in resolved["jobs"]}}
    _save(root, state)
    return root


def _context(train, validation, resolved):
    if train.provenance.get("split") != "train" or validation.provenance.get("split") != "validation":
        raise ValueError("Pilot requires train/validation; test is forbidden")
    if _contract(train) != _contract(validation):
        raise ValueError("Train/validation contracts differ")
    for data in (train, validation):
        if (data.provenance.get("stage1_run_id") != resolved["plan"]["stage1_run_id"]
                or data.provenance.get("snapshot_id") != resolved["plan"]["snapshot_id"]):
            raise ValueError("Use the protocol's original Stage 1 run/snapshot")
    configure_torch_runtime()
    cuda_requested = resolved["common_config"]["device"] == "cuda"
    if cuda_requested and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; CPU fallback is forbidden")
    return {"data_contract": _contract(train), "python": sys.version,
        "versions": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "torch", "gymnasium", "stable-baselines3", "pyarrow")},
        "device": resolved["common_config"]["device"],
        "torch_settings": torch_runtime_info(),
        "gpu_name": torch.cuda.get_device_name(0) if cuda_requested else None,
        "cuda_build": torch.version.cuda,
        "cpu_capability": getattr(torch.backends.cpu, "get_cpu_capability", lambda: "unknown")(),
        "torch_build": torch.__config__.show(), "matmul_precision": torch.get_float32_matmul_precision()}


def _checked_run(root, expected_config, state):
    meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if (root / "metadata.sha256").read_text().split() != [sha256_file(root / "metadata.json"), "metadata.json"]:
        raise ValueError("Run metadata checksum mismatch")
    verify_file_map(root, meta["files_sha256"])
    required = {"config.json", "validation_metrics.json", "validation_financials.csv", "training_metrics.json", "reload_check.json",
                "checkpoints/final/checkpoint.json", "checkpoints/final/model.zip", "checkpoints/final/vecnormalize.pkl",
                "checkpoints/final/config.json"}
    if not required.issubset(meta["files_sha256"]):
        raise ValueError("Completed run is missing essential checkpoint/evaluation artifacts")
    if (meta.get("stage") != 4 or meta.get("status") != "complete" or meta.get("test_used") is not False or meta["config"] != expected_config
            or meta["git_sha"] != state["git_sha"] or meta["training_steps"] != expected_config["total_timesteps"]
            or meta["data_contract"] != state["context"]["data_contract"]
            or meta["versions"] != state["context"]["versions"] or meta["python"] != state["context"]["python"]):
        raise ValueError("Run does not match its frozen pilot job/context")
    checkpoint = json.loads((root / "checkpoints/final/checkpoint.json").read_text())
    verify_file_map(root / "checkpoints/final", checkpoint["sha256"])
    if (checkpoint["status"] != "complete" or checkpoint["phase"] != "after_training"
            or checkpoint["steps"] != expected_config["total_timesteps"]
            or checkpoint["data_contract"] != state["context"]["data_contract"]):
        raise ValueError("Final checkpoint does not match the completed job")
    check = json.loads((root / "reload_check.json").read_text())
    if check.get("status") != "passed" or check.get("exact_actions_and_finances") is not True:
        raise ValueError("Same-runtime save/load check did not pass")
    return json.loads((root / "validation_metrics.json").read_text())


def _run_path(batch, relative):
    path = (Path(batch) / relative).resolve()
    if not path.is_relative_to((Path(batch) / "runs").resolve()):
        raise ValueError("Run path must remain inside this batch's runs directory")
    return path


def _verified_test_evidence(directory, expected_sha):
    directory = Path(directory)
    meta = json.loads((directory / "test_gate.json").read_text(encoding="utf-8"))
    if meta["git_sha"] != expected_sha or meta.get("tests", {}).get("exit_code") != 0:
        raise ValueError("Current attempt needs successful tests for this exact code SHA")
    if meta.get("pytest_xml_sha256") != sha256_file(directory / "pytest.xml"):
        raise ValueError("Test evidence checksum mismatch")
    summary = pytest_summary(directory / "pytest.xml")
    cases = {case.get("name") for case in ET.parse(directory / "pytest.xml").getroot().iter("testcase")}
    required = {"test_pilot_plan_equal_budgets", "test_pilot_disabled_does_not_train",
                "test_pilot_resume_preserves_attempts_and_skips_completed",
                "test_gymnasium_and_sb3_checkers", "test_tiny_synthetic_training_saves_and_reloads"}
    if not required.issubset(cases):
        raise ValueError("Required pilot/PPO/checker tests were not collected")
    if summary != {k: meta["tests"][k] for k in summary}:
        raise ValueError("JUnit and attempt test summary differ")
    return {"directory": str(directory), "test_gate_sha256": sha256_file(directory / "test_gate.json"),
            "pytest_xml_sha256": sha256_file(directory / "pytest.xml"), "tests": summary}


def run_pilot_batch(batch, train, validation, attempt_directory, *, enable_training=False, recover_interrupted=False):
    if type(enable_training) is not bool or type(recover_interrupted) is not bool:
        raise ValueError("Execution switches must be booleans")
    if not enable_training:
        raise ValueError("Training is disabled; explicitly enable it in Colab")
    root = Path(batch)
    state, resolved = read_pilot_batch(root)
    evidence = _verified_test_evidence(attempt_directory, state["git_sha"])
    context = _context(train, validation, resolved)
    if state["context"] is not None and state["context"] != context:
        raise ValueError("Resume context changed; keep the original runtime/device/data or create a new batch")
    lock = root / ".run-lock.json"
    owner = Path(attempt_directory).name
    if lock.exists():
        if not recover_interrupted:
            raise ValueError("Batch has a writer lock. End the old runtime, then explicitly enable recovery")
        # Preserve stale-lock evidence; recovery is an explicit single-writer action.
        (Path(attempt_directory) / "recovered-lock.json").write_bytes(lock.read_bytes())
        lock.unlink()
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"owner": owner, "created_at_utc": datetime.now(timezone.utc).isoformat()}, stream)

    def persist():
        if json.loads(lock.read_text())["owner"] != owner:
            raise RuntimeError("Batch writer ownership changed; stop this notebook")
        _save(root, state)

    active_attempt = None
    try:
        # Another writer may have completed work between our first read and lock acquisition.
        state, resolved = read_pilot_batch(root)
        if state["context"] is not None and state["context"] != context:
            raise ValueError("Resume context changed during lock acquisition")
        state.update(context=context, status="running")
        state.pop("error", None)
        state.pop("error_type", None)
        state["attempts"].append({**evidence, "status": "started"})
        persist()
        for job in resolved["jobs"]:
            entry = state["jobs"][job["job_id"]]
            if entry["status"] == "complete":
                _checked_run(_run_path(root, entry["run_directory"]), job["config"], state)
                continue
            if entry["status"] == "running" and entry["attempts"]:
                last = entry["attempts"][-1]
                previous = _run_path(root, last["run_directory"]) if last["run_directory"] else None
                # Recover a completed run if Colab died before updating the batch ledger.
                if previous is not None and (previous / "metadata.sha256").is_file():
                    _checked_run(previous, job["config"], state)
                    last.update(status="complete_recovered", training_steps=job["config"]["total_timesteps"])
                    entry.update(status="complete", run_directory=last["run_directory"])
                    persist()
                    continue
                last["status"] = "interrupted"
                last["training_steps"] = None  # Unknown, never pretend that interrupted work cost zero.
                last["checkpoint_steps_lower_bound"] = 0
                last["checkpoint_errors"] = []
                for checkpoint in previous.glob("checkpoints/*/checkpoint.json") if previous is not None else []:
                    try:
                        manifest = json.loads(checkpoint.read_text())
                        verify_file_map(checkpoint.parent, manifest["sha256"])
                        if manifest["status"] == "complete":
                            last["checkpoint_steps_lower_bound"] = max(last["checkpoint_steps_lower_bound"], manifest["steps"])
                    except (ValueError, OSError, KeyError) as error:
                        last["checkpoint_errors"].append(str(error))
            attempt = {"attempt_id": owner, "status": "started", "run_directory": None}
            active_attempt = attempt
            entry["attempts"].append(attempt)
            entry["status"] = "running"
            persist()

            def created(run):
                attempt["run_directory"] = run.relative_to(root).as_posix()
                persist()

            try:
                run = train_one_run(train, validation, TrainingConfig.from_dict(job["config"]), root / "runs",
                    git_sha=state["git_sha"], provenance={"pilot_batch": str(root), "job_id": job["job_id"],
                        "plan_sha256": resolved["sha256"], "test_evidence": evidence}, on_run_created=created, stage=4)
                _checked_run(run, job["config"], state)
                attempt.update(status="complete", training_steps=job["config"]["total_timesteps"])
                entry.update(status="complete", run_directory=run.relative_to(root).as_posix())
                persist()
            except Exception as error:
                attempt.update(status="failed", error_type=type(error).__name__, error=str(error))
                if attempt["run_directory"]:
                    meta = json.loads((_run_path(root, attempt["run_directory"]) / "metadata.json").read_text())
                    attempt["training_steps"] = meta.get("training_steps")
                entry["status"] = "failed"
                raise
        state["status"] = "complete"
        state["attempts"][-1]["status"] = "complete"
        persist()
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)) and active_attempt is not None and active_attempt["status"] == "started":
            active_attempt.update(status="interrupted", training_steps=None)
        state.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                     error_type=type(error).__name__, error=str(error))
        if state["attempts"] and state["attempts"][-1].get("directory") == str(attempt_directory):
            state["attempts"][-1].update(status=state["status"], error=str(error))
        persist()
        raise
    finally:
        if lock.exists() and json.loads(lock.read_text()).get("owner") == owner:
            lock.unlink()
    return root


def summarize_pilot_batch(batch):
    """Include every scheduled job; select only after the whole matched grid completes."""
    root = Path(batch)
    state, resolved = read_pilot_batch(root)
    rows, histories = [], []
    for job in resolved["jobs"]:
        entry = state["jobs"][job["job_id"]]
        row = {k: job[k] for k in ("job_id", "mode", "candidate_id", "seed")}
        row.update(status=entry["status"], lambda_0=job["config"]["lambda_0"], alpha=job["config"]["alpha"],
                   configured_steps=job["config"]["total_timesteps"], run_directory=entry.get("run_directory"),
                   attempt_count=len(entry["attempts"]))
        if entry["status"] == "complete":
            row.update(_checked_run(_run_path(root, entry["run_directory"]), job["config"], state))
        rows.append(row)
        for index, attempt in enumerate(entry["attempts"], 1):
            histories.append({"job_id": job["job_id"], "mode": job["mode"], "attempt_number": index, **attempt})
    runs = pd.DataFrame(rows)
    runs.to_csv(root / "run_summary.csv", index=False)
    pd.DataFrame(histories, columns=["job_id", "mode", "attempt_number", "attempt_id", "status", "run_directory",
        "training_steps", "checkpoint_steps_lower_bound", "checkpoint_errors", "error_type", "error"]).to_csv(root / "attempt_history.csv", index=False)
    candidates = []
    for (mode, candidate), part in runs.groupby(["mode", "candidate_id"], sort=True):
        values = part["net_sharpe"].tolist() if "net_sharpe" in part else []
        eligible = (len(part) == len(resolved["plan"]["seeds"]) and (part.status == "complete").all()
                    and len(values) == len(part) and all(v is not None and math.isfinite(v) for v in values))
        candidates.append({"mode": mode, "candidate_id": candidate, "scheduled_seeds": len(part),
            "completed_seeds": int((part.status == "complete").sum()), "eligible": bool(eligible),
            "mean_validation_net_sharpe": float(part.net_sharpe.mean()) if eligible else None,
            "sample_std_across_seeds": float(part.net_sharpe.std(ddof=1)) if eligible else None})
    table = pd.DataFrame(candidates)
    table.to_csv(root / "candidate_summary.csv", index=False)
    selection = {"status": "not_selected_incomplete_batch", "validation_only": True,
                 "scope": "best within this finite grid on the same validation, not an independent result", "selected": {}}
    if state["status"] == "complete" and (runs.status == "complete").all():
        selection["status"] = "selected_within_grid"
        for mode in ("fixed", "adaptive"):
            eligible = table[(table["mode"] == mode) & table.eligible]
            if eligible.empty:
                selection["status"] = "no_defined_candidate"
                selection["selected"] = {}
                break
            winner = eligible.sort_values(["mean_validation_net_sharpe", "candidate_id"], ascending=[False, True]).iloc[0]
            selection["selected"][mode] = winner["candidate_id"]
    write_json(root / "selection.json", selection)
    report = {"status": state["status"], "plan_sha256": resolved["sha256"], "git_sha": state["git_sha"],
        "scheduled_budget": resolved["budget"], "completed_jobs": int((runs.status == "complete").sum()),
        "attempts": len(histories), "failed_attempts": sum(a["status"] == "failed" for a in histories),
        "interrupted_attempts": sum(a["status"] == "interrupted" for a in histories),
        "training_steps_known_from_attempts": sum(a.get("training_steps") or 0 for a in histories),
        "unknown_training_cost_attempts": sum(a.get("training_steps") is None for a in histories),
        "test_used": False, "selection": selection,
        "files_sha256": {p.name: sha256_file(p) for p in (root / n for n in (
            "batch.json", "resolved_plan.json", "run_summary.csv", "attempt_history.csv", "candidate_summary.csv", "selection.json"))}}
    write_json(root / "batch_report.json", report)
    (root / "batch_report.sha256").write_text(sha256_file(root / "batch_report.json") + "  batch_report.json\n")
    verify_file_map(root, report["files_sha256"])
    return report
