"""Small, frozen development-only search plan. No data access or training here."""
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import re

from .training_config import TrainingConfig, load_training_config


def canonical_digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def resolve_pilot_plan(plan, base: TrainingConfig, *, device="cpu"):
    expected = {"schema", "protocol_id", "base_config", "stage1_run_id", "snapshot_id", "evaluation_split", "seeds",
                "total_timesteps", "checkpoint_steps", "max_runs", "max_total_timesteps", "primary_metric",
                "selection_rule", "fixed_candidates", "adaptive_candidates"}
    if set(plan) != expected or plan["schema"] != 1:
        raise ValueError("Unknown pilot-plan schema/fields")
    for field in ("protocol_id", "stage1_run_id", "snapshot_id"):
        if not isinstance(plan[field], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", plan[field]):
            raise ValueError(f"Unsafe {field}")
    if plan["evaluation_split"] != "validation":
        raise ValueError("Pilot is validation-only; test is forbidden")
    if plan["primary_metric"] != "net_sharpe" or plan["selection_rule"] != "mean_across_all_seeds_then_candidate_id":
        raise ValueError("Selection protocol must be fixed before new runs")
    seeds = plan["seeds"]
    if (not isinstance(seeds, list) or len(seeds) < 2 or len(set(seeds)) != len(seeds)
            or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)):
        raise ValueError("Use distinct paired integer seeds")
    fixed, adaptive = plan["fixed_candidates"], plan["adaptive_candidates"]
    if not fixed or len(fixed) != len(adaptive):
        raise ValueError("Fixed/adaptive need equal nonempty candidate budgets")
    ids, values = set(), {"fixed": set(), "adaptive": set()}
    for mode, candidates in (("fixed", fixed), ("adaptive", adaptive)):
        for c in candidates:
            fields = {"id", "lambda_0"} | ({"alpha"} if mode == "adaptive" else set())
            if set(c) != fields or not isinstance(c["id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", c["id"]):
                raise ValueError("Invalid candidate fields/id")
            if c["id"] in ids or c["id"] == "baseline":
                raise ValueError("Candidate IDs must be globally unique")
            ids.add(c["id"])
            params = (c["lambda_0"], c.get("alpha", 0.))
            if (any(type(v) not in (float, int) or not math.isfinite(v) for v in params)
                    or c["lambda_0"] <= 0 or params in values[mode]):
                raise ValueError("Require distinct finite nonzero penalty candidates")
            values[mode].add(params)
    if not any(c["alpha"] < 0 for c in adaptive) or not any(c["alpha"] > 0 for c in adaptive):
        raise ValueError("Include both signs of adaptive sensitivity")
    for field in ("total_timesteps", "checkpoint_steps", "max_runs", "max_total_timesteps"):
        if type(plan[field]) is not int or plan[field] < 1:
            raise ValueError(f"{field} must be a positive integer")
    common = replace(base, run_kind="pilot", device=device, evaluation_split="validation",
                     total_timesteps=plan["total_timesteps"], checkpoint_steps=plan["checkpoint_steps"])
    common.validate()
    jobs = []
    # Seed blocks, interleaved fixed/adaptive candidates; control first in every block.
    candidates = [("none", {"id": "baseline"})]
    for f, a in zip(fixed, adaptive):
        candidates.extend([("fixed", f), ("adaptive", a)])
    for seed in seeds:
        for mode, candidate in candidates:
            changes = {k: v for k, v in candidate.items() if k != "id"}
            cfg = replace(common, random_seed=seed, reward_mode=mode, **changes)
            cfg.validate()
            jobs.append({"job_id": f"s{seed}-{mode}-{candidate['id']}", "mode": mode,
                         "candidate_id": candidate["id"], "seed": seed, "config": cfg.to_dict()})
    total = len(jobs) * common.total_timesteps
    if len(jobs) > min(plan["max_runs"], 30) or total > min(plan["max_total_timesteps"], 524288):
        raise ValueError("Small-pilot budget exceeded; revise the protocol explicitly")
    payload = {"plan": plan, "common_config": common.to_dict(), "jobs": jobs,
               "budget": {"runs": len(jobs), "training_steps": total,
                          "by_mode": {m: {"runs": sum(j["mode"] == m for j in jobs),
                              "training_steps": sum(j["config"]["total_timesteps"] for j in jobs if j["mode"] == m)}
                                      for m in ("none", "fixed", "adaptive")}}}
    payload["sha256"] = canonical_digest(payload)
    return payload


def load_pilot_plan(path: Path, *, device="cpu"):
    path = Path(path)
    plan = json.loads(path.read_text(encoding="utf-8"))
    name = plan.get("base_config")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+\.json", name):
        raise ValueError("base_config must be a JSON file in the same configs directory")
    return resolve_pilot_plan(plan, load_training_config(path.parent / name), device=device)
