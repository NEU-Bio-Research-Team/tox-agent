"""A release names its inputs, or it names nothing useful (K12).

A tag names a commit. It does not name the `torch` and `rdkit` versions that
produced the numbers, the base image the services were built on, or the model
registry they serve — and this project reports scientific measurements, so
"which build produced this probability" has to be answerable afterwards rather
than reconstructed from a date.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "devops" / "scripts"))

import release_manifest as rm  # noqa: E402


@pytest.fixture(scope="module")
def manifest() -> dict:
    return rm.build()


def test_every_service_image_is_pinned_to_a_digest(manifest):
    """A tag is not an artifact. `python:3.10-slim` is a different image next
    month, so a rebuild of one commit would be a different release — and the
    predictor, whose output is the measurement, was the one built on a tag
    while control and OCR were pinned.
    """
    assert manifest["unpinned_base_images"] == []
    for service, image in manifest["base_images"].items():
        assert image and "@sha256:" in image, service


def test_the_manifest_records_what_resolved_not_what_was_allowed(manifest):
    """pyproject's ranges say what may be installed. A release has to say what
    was, and `torch`, `rdkit`, `transformers` and `numpy` have no upper bound
    at all while the golden values depend on them."""
    versions = manifest["dependencies"]
    for name in ("torch", "rdkit", "transformers", "numpy"):
        assert name in versions
    resolved = [v for v in versions.values() if v is not None]
    assert resolved, "no tracked dependency resolved; the manifest would be empty"


def test_an_absent_package_is_null_and_never_a_guess(manifest):
    """Null means "not in this environment", which is a fact. Substituting a
    plausible version would make the manifest worse than not having one."""
    assert set(manifest["dependencies"].values()) - {None} == {
        v for v in manifest["dependencies"].values() if isinstance(v, str)
    }


def test_the_served_weights_are_named(manifest):
    registry = manifest["predictor_registry"]
    assert registry["sha256"].startswith("sha256:")
    assert "herg-tox21-chemberta-v1" in registry["models"]


def test_a_dirty_tree_is_recorded_as_dirty(manifest):
    """A release built from uncommitted changes is not reproducible from the
    commit it names, and the manifest says so rather than implying it is."""
    assert manifest["git"]["clean"] in (True, False)
    assert manifest["git"]["commit"]


def test_check_accepts_the_tree_it_was_generated_from(tmp_path, manifest):
    path = tmp_path / "release.json"
    path.write_text(json.dumps(manifest))
    assert rm.main(["--check", str(path)]) == 0


@pytest.mark.parametrize(
    "mutate,expected",
    [
        (lambda m: m["dependencies"].__setitem__("torch", "0.0.1+fake"), "dependencies.torch"),
        (lambda m: m["predictor_registry"].__setitem__("sha256", "sha256:" + "0" * 64),
         "predictor_registry.sha256"),
        (lambda m: m["base_images"].__setitem__("predictor", "python:3.10-slim@sha256:" + "0" * 64),
         "base_images.predictor"),
        (lambda m: m["git"].__setitem__("commit", "0" * 40), "git.commit"),
    ],
)
def test_check_reports_what_moved(manifest, mutate, expected):
    saved = json.loads(json.dumps(manifest))
    mutate(saved)
    problems = rm.differences(saved, manifest)
    assert any(expected in problem for problem in problems), problems


def test_a_package_missing_from_one_environment_is_not_a_difference(manifest):
    """The control image has no torch and the predictor image has no alembic.
    Comparing a manifest across those would otherwise be all noise."""
    saved = json.loads(json.dumps(manifest))
    saved["dependencies"]["torch"] = None
    assert not [p for p in rm.differences(saved, manifest) if "torch" in p]
