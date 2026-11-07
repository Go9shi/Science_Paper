"""Dependency-policy checks, run with the full suite in Colab."""
from src.environment import classify_pip_check, dependency_closure


def test_only_documented_unused_image_conflicts_are_warnings():
    result = classify_pip_check(
        "ipython 7.34.0 requires jedi, which is not installed.\n"
        "ibis-framework 9.5.0 has requirement toolz<1,>=0.11, but you have toolz 1.2.0.\n", 1,
    )
    assert result["status"] == "passed_with_known_warnings"
    assert len(result["known_unused_image_warnings"]) == 2
    assert not result["blocking_or_unreviewed"]


def test_numpy_pandas_or_unknown_conflicts_block():
    for conflict in (
        "numba 0.60.0 has requirement numpy<2.1,>=1.22, but you have numpy 2.2.6.",
        "google-colab 1.0.0 has requirement pandas==2.2.2, but you have pandas 2.2.3.",
        "some-package 1.0 requires something, which is not installed.",
    ):
        result = classify_pip_check(conflict, 1)
        assert result["status"] == "blocked"
        assert result["blocking_or_unreviewed"] == [conflict]


def test_empty_failed_pip_check_cannot_pass():
    assert classify_pip_check("", 1)["status"] == "blocked"
    assert classify_pip_check("No broken requirements found.\n", 0)["status"] == "passed"


def test_missing_root_is_reported(monkeypatch):
    from importlib import metadata

    def missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "distribution", missing)
    report = dependency_closure(["missing-synthetic-root"])
    assert report["status"] == "failed"
    assert report["problems"] == ["Missing distribution: missing-synthetic-root"]
