"""Tool permissions come from a validated manifest (RETHINK §4.10, W9-09)."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from toxagent.flags import FLAGS
from toxagent.tools import profile_manifest as pm
from toxagent.tools.registry import FLAG_GATED_TOOLS, PROFILE_MANIFEST, PROFILES

KNOWN_FLAGS = frozenset(flag.name for flag in FLAGS)
SHIPPED = json.loads(pm.DEFAULT_PATH.read_text())


def _parse(document):
    return pm.parse(document, known_flags=KNOWN_FLAGS)


def test_the_registry_reads_the_shipped_manifest():
    manifest = pm.load()
    assert PROFILES == dict(manifest.profiles)
    assert FLAG_GATED_TOOLS == dict(manifest.flag_gated_tools)
    assert PROFILE_MANIFEST.content_sha256 == manifest.content_sha256
    assert manifest.read_only == frozenset({"audit_readonly"})
    assert all(manifest.descriptions[name] for name in manifest.profiles)


def _broken(mutate):
    document = copy.deepcopy(SHIPPED)
    mutate(document)
    return document


@pytest.mark.parametrize(("mutate", "message"), [
    (lambda d: d.update(schema_version="tool-profiles-v0"), "schema_version"),
    (lambda d: d["profiles"].update({"Bad Name": d["profiles"]["analysis"]}), "not an identifier"),
    (lambda d: d["profiles"]["analysis"].update(description=""), "needs a description"),
    (lambda d: d["profiles"]["analysis"].update(tools=[]), "at least one tool"),
    (lambda d: d["profiles"]["analysis"]["tools"].append("get_analysis_slice"), "more than once"),
    (lambda d: d["profiles"]["analysis"]["tools"].append("rm -rf"), "not a tool identifier"),
    (lambda d: d["profiles"]["report_build"]["tools"].append("submit_grounded_answer"),
     "one way to finish"),
    (lambda d: d["profiles"]["audit_readonly"]["tools"].append("update_scientific_case"),
     "write or finish"),
    (lambda d: d["profiles"]["audit_readonly"]["tools"].append("submit_grounded_answer"),
     "write or finish"),
    (lambda d: d["flag_gated_tools"].update(get_scientific_case="no_such_flag"),
     "not a declared rollout flag"),
    (lambda d: d["flag_gated_tools"].update(unknown_tool="scientific_case_v1"), "in no profile"),
])
def test_a_manifest_breaking_a_rule_is_refused(mutate, message):
    with pytest.raises(pm.ProfileManifestError, match=message):
        _parse(_broken(mutate))


def test_an_unreadable_manifest_fails_the_load(tmp_path):
    path = tmp_path / "tool_profiles.json"
    path.write_text("{not json")
    with pytest.raises(pm.ProfileManifestError, match="cannot read"):
        pm.load(path)


def test_the_hash_follows_the_content():
    changed = _broken(lambda d: d["profiles"]["analysis"].update(description="Another purpose."))
    assert _parse(changed).content_sha256 != _parse(SHIPPED).content_sha256


def test_every_manifest_tool_is_one_the_code_can_register(monkeypatch):
    """Drift, both ways: a manifest cannot name a tool no code provides, and no
    registered tool can be missing from every profile."""
    from toxagent.tools.bootstrap import build_registry

    for flag_name in set(FLAG_GATED_TOOLS.values()):
        monkeypatch.setenv(f"TOXAGENT_FLAG_{flag_name.upper()}", "1")
    registry = build_registry(
        database=None, predictor=None, create_analysis=None,
        research_provider=SimpleNamespace(name="stub"),
        compound_provider=SimpleNamespace(name="stub"),
    )
    registered = set(registry.names())
    listed = set().union(*PROFILES.values())
    assert listed - registered == set(), "the manifest names tools no code registers"
    assert registered - listed == set(), "a registered tool appears in no profile"
