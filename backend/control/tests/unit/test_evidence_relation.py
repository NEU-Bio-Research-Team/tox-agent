"""domain/evidence_relation.py — the decision_support relation type (ADS plan
section 9.3, ADR 0010)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from toxagent.domain.evidence_relation import (
    Applicability,
    Directness,
    EvidenceRelationAssessment,
    EvidenceScope,
    RelationLabel,
    SourceClass,
    SourceRef,
    Strength,
    new_proposition_id,
)

NOW = datetime(2026, 9, 15, tzinfo=timezone.utc)
SESSION_ID = "ses_" + "1" * 32
RUN_ID = "run_" + "2" * 32


def _assessment(**overrides):
    defaults = dict(
        session_id=SESSION_ID,
        run_id=RUN_ID,
        proposition_id=new_proposition_id(),
        source_ref=SourceRef(SourceClass.PREDICTOR_FACT, "obs_" + "3" * 32),
        relation=RelationLabel.SUPPORTS,
        directness=Directness.DIRECT,
        applicability=Applicability.OK,
        strength=Strength.MODERATE,
        reason_codes=("endpoint_match",),
        scope=EvidenceScope(endpoint="herg"),
        now=NOW,
    )
    defaults.update(overrides)
    return EvidenceRelationAssessment.create(**defaults)


def test_a_supporting_relation_needs_a_reason_code():
    with pytest.raises(ValueError, match="reason code"):
        _assessment(relation=RelationLabel.CONTRADICTS, reason_codes=())


def test_insufficient_and_not_applicable_do_not_need_a_reason_code():
    # "we looked and found nothing that bears on this" and "this source does
    # not apply" are complete answers on their own.
    _assessment(relation=RelationLabel.INSUFFICIENT, reason_codes=())
    _assessment(relation=RelationLabel.NOT_APPLICABLE, reason_codes=())


def test_ids_are_prefixed_and_scoped_to_the_run():
    assessment = _assessment()
    assert assessment.id.startswith("evr_")
    assert assessment.proposition_id.startswith("prop_")
    assert assessment.session_id == SESSION_ID
    assert assessment.run_id == RUN_ID


def test_to_dict_matches_the_plan_shape():
    assessment = _assessment()
    payload = assessment.to_dict()
    assert set(payload) == {
        "id", "proposition_id", "source_ref", "relation", "directness",
        "applicability", "strength", "reason_codes", "scope", "created_at",
        # TAB-Suite Wave 2: synthesis lineage and the ontology's provenance.
        "input_refs", "assessor", "method_version",
    }
    assert payload["source_ref"] == {
        "source_class": "predictor_fact", "source_id": "obs_" + "3" * 32,
    }
    assert payload["relation"] == "supports"
    assert payload["scope"]["endpoint"] == "herg"
    assert payload["scope"]["species"] is None


def test_relation_label_matches_the_plans_five_values():
    assert {member.value for member in RelationLabel} == {
        "supports", "contradicts", "contextual", "insufficient", "not_applicable",
    }


def test_a_malformed_session_id_is_rejected():
    with pytest.raises(ValueError):
        _assessment(session_id="not-a-session-id")
