"""The candidate wire shape (plan section 5.7)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from toxagent.validation.wire import ClaimCandidate, GroundedAnswerCandidate

VALID_CLAIM_ID = "clm_" + "a" * 32
VALID_OBS_ID = "obs_" + "b" * 32


def claim(**overrides) -> dict:
    defaults = dict(
        claim_id=VALID_CLAIM_ID, kind="numeric", text="x",
        observation_id=VALID_OBS_ID, field_path="predictions.herg.probability_blocker",
        source_value=0.281, rendered_value="0.281", transform="identity",
    )
    defaults.update(overrides)
    return defaults


def test_a_well_formed_claim_id_is_accepted():
    ClaimCandidate(**claim())


@pytest.mark.parametrize(
    "bad_id",
    [
        "c1",  # exactly what a live Phase 3 run submitted (progress log §4.6)
        "clm-" + "a" * 32,  # wrong separator
        "clm_" + "a" * 31,  # too short
        "clm_" + "A" * 32,  # uppercase hex
        "obs_" + "a" * 32,  # right shape, wrong kind
        "",
    ],
)
def test_a_malformed_claim_id_is_a_schema_violation_not_a_burned_attempt(bad_id):
    """This must fail at ClaimCandidate construction — i.e. at tool-argument
    schema validation, before submit_answer.execute() ever runs — not survive
    to the domain Claim() construction inside _build_answer, where it would
    consume one of the run's two answer attempts on a shape error instead of a
    substantive fix."""
    with pytest.raises(ValidationError, match="claim_id"):
        ClaimCandidate(**claim(claim_id=bad_id))


def test_a_malformed_claim_id_inside_a_full_candidate_fails_at_the_top_level():
    with pytest.raises(ValidationError):
        GroundedAnswerCandidate(
            answer_markdown="x",
            claims=[claim(claim_id="c1")],
        )
