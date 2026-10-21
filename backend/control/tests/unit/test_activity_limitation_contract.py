"""The activity bridge's limitations, through the boundary an answer crosses.

A projection that requires a limitation the answer vocabulary does not know
fails both ways at once: a claim that leaves the caveat out is accepted,
because nothing triggers it, and a claim that states it is refused as
``unknown_limitation_code``. Both were true before this file existed. So these
tests go projection -> resolver -> validator rather than calling the trigger
table on its own.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import get_args

import pytest

from tests.unit.test_activity_projections import NOW, _run
from toxagent.api import responses
from toxagent.application.activity.projections import (
    ACTIVITY_LIMITATIONS,
    NOT_ADMITTED_LIMITATION,
    POLICY_LIMITATION,
    TOXICITY_NOT_RUN_LIMITATION,
    activity_observation,
    prioritisation_observation,
    screening_state_observation,
)
from toxagent.domain.answer import LimitationCode
from toxagent.validation.answer.claim_resolver import resolve_draft
from toxagent.validation.answer.draft_wire import GroundedAnswerDraftV2
from toxagent.validation.answer.validator import validate_candidate
from toxagent.validation.limitations import TEXT

LATER = datetime(2026, 10, 21, tzinfo=timezone.utc)
INTERVAL = "activity_interval_is_marginal_coverage"
PIC50 = "pic50_not_therapeutic_efficacy"


def _validate(observation, claims: list[dict], limitations: list[str], markdown: str):
    draft = GroundedAnswerDraftV2(
        answer_markdown=markdown,
        claims=claims,
        limitations=[{"code": code, "text": ""} for code in limitations],
    )
    observations = {observation.id: observation}
    resolved = resolve_draft(draft, observations_by_id=observations)
    assert resolved.candidate is not None, resolved.violations
    return validate_candidate(
        resolved.candidate,
        session_id=observation.session_id, run_id=observation.run_id,
        candidate_generation=1, observations_by_id=observations, evidence_by_id={},
        language="en", now=LATER, claimed_values=resolved.claimed_values,
    )


def _codes(result) -> list[str]:
    return [v.code for v in result.violations]


def _estimate(*, admitted: bool):
    screening, snapshot, _ = _run()
    return activity_observation(
        snapshot, screening_id=screening.id, model_admitted=admitted, now=NOW
    )


def _pic50_claim(field_path: str = "estimate.predicted_pic50") -> list[dict]:
    return [{
        "local_ref": "pic50", "kind": "numeric", "text": "Estimated pIC50.",
        "observation_id": None, "field_path": field_path, "transform": "round:1",
    }]


def _cite(observation, claims: list[dict]) -> list[dict]:
    return [{**claim, "observation_id": observation.id} for claim in claims]


# ── the vocabulary ────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "code",
    sorted({*ACTIVITY_LIMITATIONS, NOT_ADMITTED_LIMITATION, POLICY_LIMITATION,
            TOXICITY_NOT_RUN_LIMITATION}),
)
def test_every_limitation_a_projection_requires_is_one_an_answer_can_declare(code):
    member = LimitationCode(code)
    assert set(TEXT[member]) >= {"en", "vi"}
    assert code in get_args(responses.LimitationCode)


def test_the_api_and_the_domain_name_the_same_limitations():
    assert set(get_args(responses.LimitationCode)) == {c.value for c in LimitationCode}


# ── the estimate ──────────────────────────────────────────────────────────

def test_a_pic50_claim_without_its_limitations_is_refused():
    """The first half of the defect: this used to be accepted."""
    observation = _estimate(admitted=False)

    result = _validate(observation, _cite(observation, _pic50_claim()), [],
                       "The estimated pIC50 is {{pic50}}.")

    assert _codes(result) == ["missing_required_limitation"]
    assert set(result.violations[0].expected) == {PIC50, NOT_ADMITTED_LIMITATION}


def test_a_pic50_claim_with_every_limitation_its_observation_lists_is_accepted():
    """The second half: stating all three used to be refused as unknown."""
    observation = _estimate(admitted=False)
    assert set(observation.required_limitations) == {
        INTERVAL, PIC50, NOT_ADMITTED_LIMITATION,
    }

    result = _validate(observation, _cite(observation, _pic50_claim()),
                       list(observation.required_limitations),
                       "The estimated pIC50 is {{pic50}}.")

    assert result.ok, result.violations
    assert {l.code.value for l in result.answer.limitations} == set(
        observation.required_limitations
    )


def test_an_admitted_model_does_not_require_the_not_admitted_caveat():
    observation = _estimate(admitted=True)
    result = _validate(observation, _cite(observation, _pic50_claim()), [PIC50],
                       "The estimated pIC50 is {{pic50}}.")
    assert result.ok, result.violations


def test_the_interval_caveat_is_required_of_a_claim_that_reads_the_interval():
    observation = _estimate(admitted=True)
    claims = _cite(observation, _pic50_claim("estimate.prediction_interval.0"))

    refused = _validate(observation, claims, [PIC50], "The interval starts at {{pic50}}.")
    accepted = _validate(observation, claims, [PIC50, INTERVAL],
                         "The interval starts at {{pic50}}.")

    assert _codes(refused) == ["missing_required_limitation"]
    assert refused.violations[0].expected == [INTERVAL]
    assert accepted.ok, accepted.violations


def test_a_claim_naming_no_field_is_held_to_every_caveat_the_estimate_carries():
    """It cannot be shown not to have used the interval. (Such a claim is
    also refused for having no basis; the caveat is owed regardless.)"""
    observation = _estimate(admitted=True)
    claims = [{
        "local_ref": "reading", "kind": "scientific",
        "text": "The estimate places this ligand among the more potent candidates.",
        "observation_id": observation.id,
    }]
    result = _validate(observation, claims, [PIC50], "A reading of the estimate.")
    missing = [v for v in result.violations if v.code == "missing_required_limitation"]
    assert [v.expected for v in missing] == [[INTERVAL]]


def test_the_activity_applicability_field_does_not_pull_in_the_toxicity_caveat():
    """``applicability_is_rule_based`` describes the toxicity predictor's
    element whitelist. An activity estimate's ligand/target domain is not
    that, so a claim reading it must not be told to say it is."""
    observation = _estimate(admitted=True)
    claims = [{
        "local_ref": "domain", "kind": "classification",
        "text": "The ligand is in the model's domain.",
        "observation_id": observation.id, "field_path": "applicability.ligand",
    }]
    result = _validate(observation, claims, [PIC50], "The ligand is {{domain}}.")
    assert result.ok, result.violations


# ── the decision and the stage ────────────────────────────────────────────

def test_a_claim_on_the_policy_decision_must_say_it_is_a_rule():
    screening, _, decision = _run()
    observation = prioritisation_observation(
        decision, session_id=screening.session_id, run_id=screening.run_id,
        screening_id=screening.id, now=NOW,
    )
    claims = [{
        "local_ref": "status", "kind": "classification", "text": "Prioritised.",
        "observation_id": observation.id, "field_path": "status",
    }]

    refused = _validate(observation, claims, [], "Status: {{status}}.")
    accepted = _validate(observation, claims, [POLICY_LIMITATION], "Status: {{status}}.")

    assert refused.violations[0].expected == [POLICY_LIMITATION]
    assert accepted.ok, accepted.violations


def test_a_claim_on_a_screening_whose_toxicity_did_not_run_must_say_so():
    """The critical-error case: not run is not negative."""
    screening, _, _ = _run()
    observation = screening_state_observation(screening, run_id=screening.run_id, now=NOW)
    assert observation.required_limitations == (TOXICITY_NOT_RUN_LIMITATION,)
    claims = [{
        "local_ref": "stage", "kind": "classification", "text": "Screening status.",
        "observation_id": observation.id, "field_path": "status",
    }]

    refused = _validate(observation, claims, [], "Status: {{stage}}.")
    accepted = _validate(observation, claims, [TOXICITY_NOT_RUN_LIMITATION],
                         "Status: {{stage}}.")

    assert refused.violations[0].expected == [TOXICITY_NOT_RUN_LIMITATION]
    assert accepted.ok, accepted.violations


def test_citing_a_whole_interval_names_the_element_paths():
    """Live run 2026-10-21: the first draft cited estimate.prediction_interval
    as a number and spent its one correction finding .0 and .1."""
    observation = _estimate(admitted=True)
    draft = GroundedAnswerDraftV2(
        answer_markdown="The interval is {{pic50}}.",
        claims=_cite(observation, _pic50_claim("estimate.prediction_interval")),
        limitations=[],
    )
    resolved = resolve_draft(draft, observations_by_id={observation.id: observation})
    (violation,) = resolved.violations
    assert violation.code == "claim_field_not_numeric"
    assert "estimate.prediction_interval.0" in violation.message
