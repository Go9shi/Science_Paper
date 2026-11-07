"""Validation-only replay diagnostics; also uploadable as a standalone Colab script.

Uses the existing checkout's src modules. Never trains, changes source runs, or
turns approximate agreement into an exact-replay pass. No tolerance is selected.
"""
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys
from uuid import uuid4

import numpy as np
import pandas as pd
import torch

from src.data import sha256_file, write_json
from src.ppo_training import (
    FINANCIAL_FIELDS, _normalization_digest, evaluate_policy_financial, load_saved_policy,
)
from src.validation import verify_file_map

AUDIT_SOURCE = Path(__file__).resolve()


def _exact_tables(left, right):
    try:
        pd.testing.assert_frame_equal(left, right, check_exact=True, check_dtype=False)
        return {"exact": True}
    except AssertionError as error:
        return {"exact": False, "detail": str(error)}


def _teacher_replay(root, data):
    """Predict on saved-action states, but execute only the original actions."""
    original = pd.read_csv(root / "validation_financials.csv", float_precision="round_trip", keep_default_na=False)
    if original["date"].tolist() != [str(d.date()) for d in data.features.index]:
        raise ValueError("Saved and current validation decision dates differ")
    model, vector, frozen, _ = load_saved_policy(root, data)
    before = _normalization_digest(frozen)
    rows, predictions = [], []
    try:
        obs = vector.reset()
        for i, old in original.iterrows():
            saved = np.asarray(json.loads(old["action_json"]), dtype=model.action_space.dtype)
            predicted, _ = model.predict(obs, deterministic=True)
            again, _ = model.predict(obs, deterministic=True)
            if not np.isfinite(predicted).all():
                raise FloatingPointError("Nonfinite teacher-state prediction")
            np.testing.assert_array_equal(predicted, again)
            delta = np.abs(predicted[0].astype(np.float64) - saved.astype(np.float64))
            predictions.append({"date": old["date"], "action_exact": bool(np.array_equal(saved, predicted[0])),
                "max_absolute_action_difference": float(delta.max()),
                "normalized_observation_json": json.dumps(obs[0].tolist()),
                "saved_action_json": json.dumps(saved.tolist()), "predicted_action_json": json.dumps(predicted[0].tolist())})
            obs, _, done, infos = vector.step(saved.reshape(1, -1))
            info = infos[0]
            if info["date"] != old["date"] or bool(done[0]) != (i == len(original) - 1):
                raise AssertionError("Teacher replay date/horizon mismatch")
            rows.append({**{k: info[k] for k in FINANCIAL_FIELDS},
                         "weights_json": json.dumps(info["weights"].tolist()), "action_json": json.dumps(saved.tolist())})
        if before != _normalization_digest(frozen):
            raise AssertionError("Teacher replay updated normalization statistics")
    finally:
        vector.close()
    return original, pd.DataFrame(rows), pd.DataFrame(predictions)


def _closed_loop(root, data):
    model, vector, frozen, config = load_saved_policy(root, data)
    try:
        return evaluate_policy_financial(model, vector, frozen, data, config)
    finally:
        vector.close()


def audit_saved_policies(validation_data, run_directories, output_root):
    """Write three-mode diagnostic evidence; caller runs this only in Google Colab.

    exact_original_replay remains false on any mismatch. Action differences are
    measured on the ORIGINAL trajectory to avoid confusing them with feedback
    from changed portfolio states. Closed-loop differences are retained separately.
    """
    if validation_data.provenance.get("split") != "validation":
        raise ValueError("Replay audit requires validation; test is forbidden")
    if set(run_directories) != {"none", "fixed", "adaptive"}:
        raise ValueError("Provide all three matched reward modes")
    roots = {mode: Path(p) for mode, p in run_directories.items()}
    output_root = Path(output_root)
    if any(output_root.resolve().is_relative_to(p.resolve()) for p in roots.values()):
        raise ValueError("Audit evidence must be outside the original runs")
    group = output_root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    group.mkdir(parents=True, exist_ok=False)
    runtime_before = {"threads": torch.get_num_threads(), "deterministic": torch.are_deterministic_algorithms_enabled()}
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    report = {"schema": 1, "status": "started", "training_performed": False, "test_used": False,
        "thresholds": "none; differences reported without approximate-pass classification",
        "fresh_session_claim": "requires_user_confirmation", "runtime_before": runtime_before,
        "torch_runtime": {"threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "mkldnn": torch.backends.mkldnn.enabled, "matmul_precision": torch.get_float32_matmul_precision(),
            "cpu_capability": getattr(torch.backends.cpu, "get_cpu_capability", lambda: "unknown")(),
            "build_config": torch.__config__.show()},
        "audit_script_sha256": sha256_file(AUDIT_SOURCE), "runs": {}}
    write_json(group / "audit_report.json", report)
    common = None
    try:
        for mode in ("none", "fixed", "adaptive"):
            root = roots[mode]
            metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
            if metadata.get("status") != "complete" or metadata.get("test_used") is not False:
                raise ValueError("Only complete development runs may be audited")
            if (root / "metadata.sha256").read_text().split() != [sha256_file(root / "metadata.json"), "metadata.json"]:
                raise ValueError("Source metadata checksum mismatch")
            verify_file_map(root, metadata["files_sha256"])
            versions = {name: importlib.metadata.version(name) for name in metadata["versions"]}
            if versions != metadata["versions"] or sys.version != metadata["python"]:
                raise ValueError("Use the original recorded Python/library versions")
            if metadata["reward_mode"] != mode:
                raise ValueError("Run reward mode mismatch")
            contract = {"config": {k: v for k, v in metadata["config"].items() if k != "reward_mode"},
                        "data": metadata["data_contract"], "git_sha": metadata["git_sha"]}
            if common is not None and common != contract:
                raise ValueError("The three source runs are not a matched group")
            common = contract
            destination = group / mode
            destination.mkdir(exist_ok=False)
            original, teacher, predictions = _teacher_replay(root, validation_data)
            teacher.to_csv(destination / "saved_action_finances.csv", index=False)
            predictions.to_csv(destination / "policy_on_saved_states.csv", index=False)
            current, metrics = _closed_loop(root, validation_data)
            repeated, metrics2 = _closed_loop(root, validation_data)
            current.to_csv(destination / "new_policy_finances.csv", index=False)
            write_json(destination / "new_policy_metrics.json", metrics)
            comparison = {"source_run": root.name, "source_git_sha": metadata["git_sha"],
                "source_metadata_sha256": sha256_file(root / "metadata.json"),
                "observations": len(original), "saved_action_finances": _exact_tables(original, teacher),
                "policy_actions_on_saved_states_exact": bool(predictions["action_exact"].all()),
                "max_absolute_action_difference_on_saved_states": float(predictions["max_absolute_action_difference"].max()),
                "float32_epsilon_at_one": float(np.finfo(np.float32).eps),
                "closed_loop_original": _exact_tables(original, current),
                "closed_loop_metrics_original_exact": json.loads((root / "validation_metrics.json").read_text()) == metrics,
                "same_runtime_independent_reload": _exact_tables(current, repeated),
                "same_runtime_metrics_exact": metrics == metrics2,
                "max_absolute_nav_difference_closed_loop": float(np.max(np.abs(
                    original["portfolio_value"].to_numpy() - current["portfolio_value"].to_numpy())))}
            write_json(destination / "comparison.json", comparison)
            report["runs"][mode] = comparison
            write_json(group / "audit_report.json", report)
        report.update(status="diagnostic_complete", matched_source_contract=common)
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        write_json(group / "audit_report.json", report)
        raise
    report["files_sha256"] = {p.relative_to(group).as_posix(): sha256_file(p) for p in sorted(group.rglob("*"))
                               if p.is_file() and p.name != "audit_report.json"}
    write_json(group / "audit_report.json", report)
    (group / "audit_report.sha256").write_text(sha256_file(group / "audit_report.json") + "  audit_report.json\n")
    verify_file_map(group, report["files_sha256"])
    print("Replay diagnostics:", group)
    for mode, comparison in report["runs"].items():
        print(mode, "saved-action finances exact:", comparison["saved_action_finances"]["exact"],
              "policy actions exact:", comparison["policy_actions_on_saved_states_exact"],
              "same-runtime reload exact:", comparison["same_runtime_independent_reload"]["exact"],
              "max action difference:", comparison["max_absolute_action_difference_on_saved_states"])
    return group
