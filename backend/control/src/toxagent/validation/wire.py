"""The candidate wire shape (plan section 5.7).

This is what a model actually submits to ``submit_grounded_answer``: strings
and numbers, not domain objects. Pydantic enforces the outer shape — types,
required fields, the transform grammar, id prefixes — so that a candidate with
the wrong *shape* is rejected before any validator has to reason about it, and
a candidate with the right shape but a wrong *value* (a number that does not
match its observation) reaches the semantic validators as ordinary, well-formed
data.

Deliberately not the domain ``Claim``/``GroundedAnswer`` types: those enforce
invariants a malformed model output cannot always satisfy (e.g. a numeric claim
naming an observation that does not exist), and constructing one from bad input
would raise instead of collecting a violation to send back for correction.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

TRANSFORM_PATTERN = re.compile(r"^(identity|round:[0-6]|percent:[0-6]|difference|ratio)$")
ID_PATTERN = re.compile(r"^[a-z]{3,4}_[0-9a-f]{32}$")


class _Wire(BaseModel):
    model_config = ConfigDict(extra="forbid")


#: Unlike observation_id/citation_ids (whose wrong-kind or nonexistent case is
#: caught later by a dedicated, correctable violation), nothing downstream of
#: this wire model produces a typed violation for a malformed claim_id — the
#: domain ``Claim`` construction in ``_build_answer`` is the first and only
#: place that actually enforces the ``clm_`` shape, and by then a bad id has
#: burned the run's last correction attempt on ``candidate_malformed`` instead
#: of a substantive fix (a live Phase 3 run did exactly this with
#: ``claim_id: "c1"`` on its final candidate; progress log §4.6). Checking the
#: full shape here turns it into an ordinary generation-1 violation instead.
CLAIM_ID_PATTERN = re.compile(r"^clm_[0-9a-f]{32}$")


class ClaimCandidate(_Wire):
    claim_id: str
    kind: Literal["numeric", "classification", "scientific", "comparison", "limitation", "recommendation"] = Field(
        description=(
            "'comparison' is for a difference/ratio between two other claims' values "
            "(pair it with transform + input_claim_ids below) — never 'numeric', whose "
            "field_path must resolve to exactly one predictor field."
        )
    )
    text: str = Field(min_length=1, max_length=2000)
    observation_id: str | None = None
    field_path: str | None = None
    source_value: object = None
    rendered_value: str | None = None
    transform: str = "identity"
    citation_ids: list[str] = Field(default_factory=list)
    input_claim_ids: list[str] = Field(
        default_factory=list,
        description=(
            "For kind=comparison only: exactly the two other claim_ids (in this same "
            "candidate) that transform=difference/ratio is computed from, first minus/over "
            "second."
        ),
    )

    @field_validator("transform")
    @classmethod
    def _known_transform(cls, value: str) -> str:
        if not TRANSFORM_PATTERN.match(value):
            raise ValueError(
                f"transform {value!r} is not in the allowlist "
                "(identity, round:0-6, percent:0-6, difference, ratio)"
            )
        return value

    @field_validator("observation_id")
    @classmethod
    def _observation_id_shape(cls, value: str | None) -> str | None:
        if value is not None and not ID_PATTERN.match(value):
            raise ValueError(f"observation_id {value!r} is not a ToxAgent identifier")
        return value

    @field_validator("claim_id")
    @classmethod
    def _claim_id_shape(cls, value: str) -> str:
        if not CLAIM_ID_PATTERN.match(value):
            raise ValueError(
                f"claim_id {value!r} must be 'clm_' followed by 32 lowercase hex characters "
                "(a UUID4 works: clm_ + uuid4().hex)"
            )
        return value


class LimitationCandidate(_Wire):
    code: str
    text: str = Field(default="", max_length=1000)


class RecommendationCandidate(_Wire):
    text: str = Field(min_length=1, max_length=1000)
    basis_claim_ids: list[str] = Field(default_factory=list)


class GroundedAnswerCandidate(_Wire):
    schema_version: Literal["grounded-answer-v1"] = "grounded-answer-v1"
    answer_markdown: str = Field(min_length=1, max_length=20_000)
    claims: list[ClaimCandidate] = Field(default_factory=list, max_length=64)
    limitations: list[LimitationCandidate] = Field(default_factory=list, max_length=16)
    recommended_next_steps: list[RecommendationCandidate] = Field(default_factory=list, max_length=8)
