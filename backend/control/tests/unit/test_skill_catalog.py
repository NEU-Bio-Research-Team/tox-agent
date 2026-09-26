"""The scientific skill catalog (ADR 0012, RETHINK §4.8–§4.9)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from toxagent.config import PACKAGE_ROOT
from toxagent.domain import decision_state as ds
from toxagent.application import skill_catalog as catalog_module
from toxagent.application.skill_catalog import SkillCatalogError, load_catalog, render_index, render_static

SHIPPED = PACKAGE_ROOT / "agent_profiles"
CASE_TOOLS = ["get_scientific_case", "update_scientific_case"]


def test_the_shipped_catalog_loads_and_every_skill_is_active():
    catalog = load_catalog(SHIPPED)
    assert [s.skill_id for s in catalog.skills] == [
        "assess-conflicting-evidence", "critique-case", "interpret-model-attribution",
    ]
    assert {s.status for s in catalog.skills} == {"active"}
    for skill in catalog.skills:
        assert len(skill.content_sha256) == 64
        assert "SKILL.md" in skill.file_hashes


def test_critique_case_triggers_on_recommendations_not_on_model_questions():
    """W9-03: the pilot read critique-case on 9 of 10 cases (precision 0.44);
    the extra reads were attribution, reliability and scope questions."""
    skill = next(s for s in load_catalog(SHIPPED).skills if s.skill_id == "critique-case")
    assert "recommend a course of action" in skill.description
    assert "Not for questions that only ask what the model predicts" in skill.description
    assert skill.version == "1.1.0"


def test_loading_twice_gives_the_same_hashes():
    assert load_catalog(SHIPPED).catalog_sha256 == load_catalog(SHIPPED).catalog_sha256


def test_a_skill_is_offered_only_with_every_tool_it_requires():
    catalog = load_catalog(SHIPPED)
    everything = CASE_TOOLS + ["get_attribution", "get_evidence_record"]
    assert len(catalog.available("decision_support", everything)) == 3
    # Without the case tools nothing is offered: each skill writes to the case.
    assert catalog.available("decision_support", ["get_attribution", "get_evidence_record"]) == ()
    without_attribution = catalog.available("decision_support", CASE_TOOLS + ["get_evidence_record"])
    assert "interpret-model-attribution" not in {s.skill_id for s in without_attribution}
    # And only to the profiles it names.
    assert catalog.available("report_build", everything) == ()


def test_offering_never_adds_a_tool():
    catalog = load_catalog(SHIPPED)
    tools = CASE_TOOLS + ["get_attribution", "get_evidence_record"]
    before = list(tools)
    catalog.available("decision_support", tools)
    assert tools == before
    for skill in catalog.skills:
        assert "tools" not in skill.manifest and "permissions" not in skill.manifest


def test_the_dynamic_index_carries_descriptions_not_instructions():
    skills = load_catalog(SHIPPED).skills
    index = render_index(skills)
    for skill in skills:
        assert skill.description in index
        assert skill.body not in index
    assert render_index([]) == ""


def test_the_static_arm_composes_bodies_and_references():
    skills = load_catalog(SHIPPED).skills
    static = render_static(skills)
    for skill in skills:
        assert skill.body in static
        for text in skill.references.values():
            assert text in static


def test_a_missing_catalog_directory_is_an_empty_catalog(tmp_path):
    assert load_catalog(tmp_path).skills == ()


@pytest.fixture
def copy(tmp_path) -> Path:
    shutil.copytree(SHIPPED / "scientific_skills", tmp_path / "scientific_skills")
    return tmp_path


def _edit_manifest(root: Path, skill: str, **changes) -> None:
    path = root / "scientific_skills" / skill / "skill.manifest.json"
    data = json.loads(path.read_text())
    data.update(changes)
    path.write_text(json.dumps(data))


@pytest.mark.parametrize("changes, message", [
    ({"skill_id": "other-name"}, "skill_id must equal"),
    ({"version": "1.0"}, "MAJOR.MINOR.PATCH"),
    ({"status": "live"}, "status must be"),
    ({"risk_tier": "none"}, "risk_tier"),
    ({"allowed_profiles": ["shell"]}, "allowed_profiles"),
    ({"required_capabilities": ["bash"]}, "unknown tools"),
    ({"references": []}, "references declared"),
])
def test_a_malformed_manifest_fails_the_load(copy, changes, message):
    _edit_manifest(copy, "assess-conflicting-evidence", **changes)
    with pytest.raises(SkillCatalogError, match=message):
        load_catalog(copy)


def test_a_name_that_does_not_match_its_directory_fails(copy):
    path = copy / "scientific_skills" / "critique-case" / "SKILL.md"
    path.write_text(path.read_text().replace("name: critique-case", "name: critique_case"))
    with pytest.raises(SkillCatalogError, match="not a valid Agent Skills name"):
        load_catalog(copy)


def test_an_undeclared_reference_file_fails(copy):
    (copy / "scientific_skills" / "critique-case" / "references").mkdir()
    (copy / "scientific_skills" / "critique-case" / "references" / "extra.md").write_text("x")
    with pytest.raises(SkillCatalogError, match="references declared"):
        load_catalog(copy)


def test_a_draft_is_loaded_but_never_offered(copy):
    _edit_manifest(copy, "critique-case", status="draft")
    catalog = load_catalog(copy)
    assert catalog.get("critique-case").status == "draft"
    offered = catalog.available("decision_support", CASE_TOOLS + ["get_attribution",
                                                                  "get_evidence_record"])
    assert "critique-case" not in {s.skill_id for s in offered}


def test_editing_a_reference_changes_the_skill_hash(copy):
    before = load_catalog(copy).get("assess-conflicting-evidence").content_sha256
    path = copy / "scientific_skills" / "assess-conflicting-evidence" / "references" / \
        "scope-comparison-checklist.md"
    path.write_text(path.read_text() + "\nOne more row.\n")
    after = load_catalog(copy).get("assess-conflicting-evidence").content_sha256
    assert before != after


def test_references_stay_under_the_size_cap():
    for skill in load_catalog(SHIPPED).skills:
        for text in skill.references.values():
            assert len(text) <= catalog_module.MAX_REFERENCE_CHARS


def _state():
    return ds.initial(session_id="ses_x", run_id="run_x", goal="g", subject_refs=[],
                      budget_snapshot={})


def test_the_static_arm_counts_every_offered_skill_as_read():
    pins = [{"skill_id": "a", "version": "1.0.0", "content_sha256": "sha256:1"}]
    state = ds.record_skills_offered(_state(), mode="static", offered=pins)
    assert state.skills["loaded"] == pins


def test_a_dynamic_read_is_recorded_once_with_its_references():
    pin = {"skill_id": "a", "version": "1.0.0", "content_sha256": "sha256:1"}
    state = ds.record_skills_offered(_state(), mode="dynamic", offered=[pin])
    assert state.skills["loaded"] == []
    state = ds.record_skill_loaded(state, pin=pin)
    again = ds.record_skill_loaded(state, pin=pin)
    assert again is state
    state = ds.record_skill_loaded(state, pin=pin, reference="r.md")
    assert state.skills["loaded"] == [pin]
    assert state.skills["references_loaded"] == [{**pin, "reference": "r.md"}]
    assert ds.DecisionSupportStateV1.from_dict(state.to_dict()) == state


def test_a_final_state_records_no_more_skills():
    state = ds.finalize(_state(), run_status="cancelled")
    assert ds.record_skill_loaded(state, pin={"skill_id": "a"}) is state
