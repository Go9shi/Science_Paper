"""Dependency metadata diagnostics; no training or market-data calculations."""
from __future__ import annotations

from importlib import metadata
import re
import sys

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version
from packaging.specifiers import SpecifierSet


def dependency_closure(roots: list[str]) -> dict:
    """Check active base requirements transitively; optional extras are not enabled."""
    pending = list(roots)
    versions = {}
    problems = []
    seen = set()
    while pending:
        name = canonicalize_name(pending.pop())
        if name in seen:
            continue
        seen.add(name)
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            problems.append(f"Missing distribution: {name}")
            continue
        versions[name] = dist.version
        python_requires = dist.metadata.get("Requires-Python")
        if python_requires and Version(".".join(map(str, sys.version_info[:3]))) not in SpecifierSet(python_requires):
            problems.append(f"{name}: Python does not satisfy {python_requires}")
        for raw in dist.requires or []:
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            dependency = canonicalize_name(requirement.name)
            try:
                installed = metadata.version(dependency)
            except metadata.PackageNotFoundError:
                problems.append(f"{name}: missing {requirement}")
                continue
            if requirement.specifier and Version(installed) not in requirement.specifier:
                problems.append(f"{name}: requires {requirement}, installed {installed}")
            pending.append(dependency)
    return {"status": "passed" if not problems else "failed", "versions": versions, "problems": problems}


def classify_pip_check(output: str, returncode: int) -> dict:
    """Allow only two documented, unused Colab-image issues; never hide new ones."""
    warnings = []
    blocking = []
    for line in output.splitlines():
        line = line.strip()
        if not line or line == "No broken requirements found.":
            continue
        known_ipython = re.fullmatch(r"ipython \S+ requires jedi, which is not installed\.", line)
        known_ibis = re.fullmatch(r"ibis-framework \S+ has requirement toolz.+, but you have toolz \S+\.", line)
        if known_ipython or known_ibis:
            warnings.append(line)
        else:
            blocking.append(line)
    if returncode not in (0, 1) or (returncode != 0 and not warnings and not blocking):
        blocking.append(f"pip check did not produce a usable diagnostic (exit {returncode})")
    return {
        "status": "blocked" if blocking else ("passed_with_known_warnings" if warnings else "passed"),
        "global_pip_check_returncode": returncode,
        "known_unused_image_warnings": warnings,
        "blocking_or_unreviewed": blocking,
    }
