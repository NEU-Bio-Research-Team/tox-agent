"""Classification and citation/basis validation (plan sections 9.2, 9.3)."""
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from toxagent.domain.evidence import EvidenceRecord, EvidenceStatus, SourceType
from toxagent.domain.ids import new_id
from toxagent.domain.observation import Observation, ObservationKind, Producer
from toxagent.validation.citations import (
    validate_basis,
    validate_citations,
    validate_recommendation_basis,
)
from toxagent.validation.classification import validate_classification
from toxagent.validation.wire import ClaimCandidate

NOW = datetime(2026, 9, 4, tzinfo=timezone.utc)


def an_observation() -> Observation:
    return Observation.create(
        session_id=new_id("ses"), run_id=new_id("run"), producer=Producer.PREDICTOR,
        kind=ObservationKind.PREDICTION, schema_version="v1",
        canonical_payload={
            "predictions": {"herg": {"label": "non_blocker", "probability_blocker": 0.2}},
            "applicability": {"status": "ok", "method": "element_rules_v1", "reasons": []},
        },
        model_projection={}, provenance={}, now=NOW,
    )


def claim(**overrides) -> ClaimCandidate:
    defaults = dict(
        claim_id=new_id("clm"), kind="classification", text="x",
        field_path="predictions.herg.label", source_value="non_blocker", rendered_value="non_blocker",
    )
    defaults.update(overrides)
    return ClaimCandidate(**defaults)


# --- classification --------------------------------------------------------

def test_the_exact_canonical_label_passes():
    observation = an_observation()
    result = validate_classification(claim(observation_id=observation.id), observation)
    assert result == []


def test_non_blocker_cannot_be_rendered_as_safe():
    """Plan section 9.2: an alias belongs to the renderer, never the claim."""
    observation = an_observation()
    result = validate_classification(
        claim(observation_id=observation.id, rendered_value="safe"), observation
    )
    assert [v.code for v in result] == ["claim_rendered_value_is_an_alias"]


def test_applicability_ok_cannot_be_stored_as_a_different_value():
    observation = an_observation()
    result = validate_classification(
        claim(
            observation_id=observation.id, field_path="applicability.status",
            source_value="in_distribution", rendered_value="in_distribution",
        ),
        observation,
    )
    assert "claim_source_value_mismatch" in [v.code for v in result]


def test_a_numeric_field_cannot_be_claimed_as_a_classification():
    observation = an_observation()
    result = validate_classification(
        claim(
            observation_id=observation.id, field_path="predictions.herg.probability_blocker",
            source_value=0.2,
        ),
        observation,
    )
    assert [v.code for v in result] == ["claim_field_not_classification"]


def test_a_missing_observation_is_reported():
    result = validate_classification(claim(observation_id=new_id("obs")), None)
    assert [v.code for v in result] == ["claim_observation_not_found"]


# --- basis -------------------------------------------------------------

def test_a_scientific_claim_with_neither_observation_nor_citation_has_no_basis():
    result = validate_basis(
        claim(kind="scientific", field_path=None, source_value=None, rendered_value=None),
        has_observation_basis=False,
    )
    assert [v.code for v in result] == ["claim_has_no_basis"]


def test_a_scientific_claim_with_a_citation_needs_no_observation():
    result = validate_basis(
        claim(
            kind="scientific", field_path=None, source_value=None, rendered_value=None,
            citation_ids=[new_id("evd")],
        ),
        has_observation_basis=False,
    )
    assert result == []


def test_a_limitation_claim_needs_no_basis_at_all():
    result = validate_basis(
        claim(kind="limitation", field_path=None, source_value=None, rendered_value=None),
        has_observation_basis=False,
    )
    assert result == []


# --- citations -----------------------------------------------------------

def accepted_evidence() -> EvidenceRecord:
    record = EvidenceRecord.create(
        session_id=new_id("ses"), provider="europepmc", provider_record_id="PMC1",
        source_type=SourceType.ARTICLE, title="t", retrieved_at=NOW,
    )
    return record.to_status(EvidenceStatus.NORMALIZED).to_status(EvidenceStatus.ACCEPTED)


def test_an_accepted_citation_passes():
    evidence = accepted_evidence()
    result = validate_citations(
        claim(kind="scientific", citation_ids=[evidence.id]), {evidence.id: evidence},
        read_evidence_ids=frozenset({evidence.id}),
    )
    assert result == []


def test_an_accepted_but_unread_citation_is_refused():
    """W3-07 (remaining-plan): a search result already carries enough to
    construct a citation (title, identifier — tools/definitions/evidence.py's
    _SEARCH_RESULT_FIELDS) without the model ever calling get_evidence_record
    to actually read it. read_evidence_ids is the read-tracking submit_answer.py
    builds from this run's own completed tool_calls; the default (no read
    tracked) must refuse, not silently pass."""
    evidence = accepted_evidence()
    result = validate_citations(
        claim(kind="scientific", citation_ids=[evidence.id]), {evidence.id: evidence}
    )
    assert [v.code for v in result] == ["citation_not_read"]


def test_an_unknown_citation_is_refused():
    result = validate_citations(claim(kind="scientific", citation_ids=[new_id("evd")]), {})
    assert [v.code for v in result] == ["citation_not_found"]


def test_a_not_yet_accepted_citation_is_refused():
    record = EvidenceRecord.create(
        session_id=new_id("ses"), provider="europepmc", provider_record_id="PMC2",
        source_type=SourceType.ARTICLE, title="t", retrieved_at=NOW,
    )  # still "retrieved", never promoted
    result = validate_citations(
        claim(kind="scientific", citation_ids=[record.id]), {record.id: record}
    )
    assert [v.code for v in result] == ["citation_not_accepted"]


def test_a_recommendations_basis_must_name_real_claims():
    result = validate_recommendation_basis(0, ["clm_" + "z" * 32], frozenset({"clm_" + "a" * 32}))
    assert [v.code for v in result] == ["recommendation_basis_unknown"]


def test_a_recommendations_basis_that_matches_is_fine():
    known = "clm_" + "a" * 32
    assert validate_recommendation_basis(0, [known], frozenset({known})) == []
