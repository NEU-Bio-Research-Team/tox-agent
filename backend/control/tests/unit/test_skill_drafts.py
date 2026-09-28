"""Skill drafts: proposed, reviewed by an expert, never offered (RETHINK §4.8, W9-11)."""
from __future__ import annotations

import pytest

from toxagent.application.investigation import skill_drafts
from toxagent.application.investigation.skill_catalog import load_catalog
from toxagent.config import PACKAGE_ROOT
from toxagent.domain import skill_draft as sd

CATALOG = load_catalog(PACKAGE_ROOT / "agent_profiles")
BODY = "# Weigh species differences\n\n" + "Check whether the species of each source matches the question. " * 4


def _package(**overrides):
    fields = dict(
        skill_id="weigh-species-differences",
        description="Use when evidence comes from a species other than the one the decision is about.",
        body=BODY, required_capabilities=["get_evidence_record"],
        output_contract="uncertainties of kind assay_mismatch naming the species gap",
    )
    fields.update(overrides)
    return skill_drafts.compose_package(**fields)


def _draft(**overrides) -> sd.SkillDraft:
    skill_md, manifest = _package()
    fields = dict(
        id="skd_" + "1" * 32, skill_id="weigh-species-differences", version="0.1.0",
        status="proposed", author=sd.DraftAuthor(actor="user", subject_id="user-1"),
        rationale="rat data kept being used for a human question", skill_md=skill_md,
        manifest=manifest, references={}, content_sha256="x" * 64, created_at="t", updated_at="t",
    )
    fields.update(overrides)
    return sd.SkillDraft(**fields)


def test_a_composed_package_passes_the_catalogs_own_validator():
    skill_md, manifest = _package()
    digest = skill_drafts.validate_package(skill_md, manifest, {}, catalog=CATALOG)
    assert len(digest) == 64
    assert manifest["status"] == "draft"


@pytest.mark.parametrize(("overrides", "message"), [
    ({"required_capabilities": ["bash"]}, "unknown tools"),
    ({"skill_id": "Not_A_Name"}, "not a valid Agent Skills name"),
    ({"skill_id": "critique-case", "version": "1.0.0"}, "needs a higher version"),
])
def test_a_package_the_catalog_would_refuse_is_refused(overrides, message):
    skill_md, manifest = _package(**overrides)
    with pytest.raises(skill_drafts.DraftRefused, match=message):
        skill_drafts.validate_package(skill_md, manifest, {}, catalog=CATALOG)


def test_a_draft_cannot_arrive_already_active():
    skill_md, manifest = _package()
    with pytest.raises(skill_drafts.DraftRefused, match="status is 'draft'"):
        skill_drafts.validate_package(skill_md, {**manifest, "status": "active"}, {}, catalog=CATALOG)


def test_a_revision_of_a_shipped_skill_needs_a_higher_version():
    skill_md, manifest = _package(skill_id="critique-case", version="1.2.0")
    skill_drafts.validate_package(skill_md, manifest, {}, catalog=CATALOG)


def test_review_needs_an_expert_who_is_not_the_author():
    draft = _draft()
    with pytest.raises(sd.InvalidDraftTransition, match="'expert' role"):
        sd.review(draft, reviewer="user-2", reviewer_roles=frozenset(), decision="approve",
                  note="fine", at="t")
    with pytest.raises(sd.InvalidDraftTransition, match="other than its author"):
        sd.review(draft, reviewer="user-1", reviewer_roles=frozenset({"expert"}),
                  decision="approve", note="fine", at="t")
    with pytest.raises(sd.InvalidDraftTransition, match="states its reason"):
        sd.review(draft, reviewer="user-3", reviewer_roles=frozenset({"expert"}),
                  decision="approve", note=" ", at="t")
    approved = sd.review(draft, reviewer="user-3", reviewer_roles=frozenset({"expert"}),
                         decision="approve", note="useful on the rat/human cases", at="t2")
    assert approved.status == "approved" and approved.review.reviewer == "user-3"
    with pytest.raises(sd.InvalidDraftTransition, match="only a proposed draft"):
        sd.review(approved, reviewer="user-4", reviewer_roles=frozenset({"expert"}),
                  decision="reject", note="changed my mind", at="t3")


def test_only_an_approved_draft_has_a_package_and_it_is_active():
    draft = _draft()
    with pytest.raises(sd.InvalidDraftTransition, match="only an approved draft"):
        draft.package()
    approved = sd.review(draft, reviewer="user-3", reviewer_roles=frozenset({"expert"}),
                         decision="approve", note="ok", at="t")
    files = approved.package()
    assert '"status": "active"' in files["skill.manifest.json"]
    assert set(files) == {"SKILL.md", "skill.manifest.json"}


def test_only_the_author_withdraws_a_proposed_draft():
    draft = _draft()
    with pytest.raises(sd.InvalidDraftTransition, match="only its author"):
        sd.withdraw(draft, by="user-2", at="t")
    assert sd.withdraw(draft, by="user-1", at="t").status == "withdrawn"


def test_a_draft_round_trips():
    draft = _draft()
    assert sd.SkillDraft.from_dict(draft.to_dict()) == draft
