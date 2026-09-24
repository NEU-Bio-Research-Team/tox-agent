"""Which manifest this install reads, and what it says when there is none.

I23: Settings named ``src/artifacts/predictor-manifest.yaml`` as its default —
a path no layout has ever had — and because that value was never ``None`` it
shadowed bootstrap.py's correct derivation. Docker overrode TOXPRED_MANIFEST,
so only developers running the app with defaults hit it, and what they saw was
a bare FileNotFoundError for a path they had never configured.
"""
from __future__ import annotations

import pytest

from toxpred.scientific.bootstrap import DEFAULT_MANIFEST, resolve_manifest
from toxpred.settings import Settings


def test_unset_env_leaves_the_manifest_for_the_install_to_resolve(monkeypatch):
    monkeypatch.delenv("TOXPRED_MANIFEST", raising=False)
    assert Settings.from_env().manifest_path is None


def test_blank_env_is_treated_as_unset_not_as_the_current_directory(monkeypatch):
    monkeypatch.setenv("TOXPRED_MANIFEST", "   ")
    assert Settings.from_env().manifest_path is None


def test_explicit_env_wins(monkeypatch, tmp_path):
    manifest = tmp_path / "custom.yaml"
    manifest.write_text("schema_version: 1\n")
    monkeypatch.setenv("TOXPRED_MANIFEST", str(manifest))
    assert Settings.from_env().manifest_path == manifest
    assert resolve_manifest(manifest) == manifest


def test_the_default_manifest_is_the_registry_that_ships_with_this_checkout():
    assert DEFAULT_MANIFEST.name == "predictor-manifest.yaml"
    assert DEFAULT_MANIFEST.parent.name == "registry"
    assert DEFAULT_MANIFEST.is_file(), DEFAULT_MANIFEST
    # The path the bug produced. Asserting its absence keeps the regression
    # legible if someone reintroduces a src-relative default.
    assert "src" not in DEFAULT_MANIFEST.parts[-3:]


def test_default_resolution_needs_no_environment(monkeypatch):
    monkeypatch.delenv("TOXPRED_MANIFEST", raising=False)
    assert resolve_manifest(Settings.from_env().manifest_path) == DEFAULT_MANIFEST


def test_a_missing_manifest_names_the_resolved_path_and_how_it_was_chosen(tmp_path):
    absent = tmp_path / "nowhere" / "predictor-manifest.yaml"
    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_manifest(absent)
    message = str(excinfo.value)
    assert str(absent) in message
    assert "TOXPRED_MANIFEST" in message
