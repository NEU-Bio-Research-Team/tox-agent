"""The grounded-answer draft, v2 (P1-4, P1-5).

Three things a model was being asked to do that it should never have been asked
to do, all of which the 2026-09-13 audit observed going wrong:

**Mint a deployment-global identifier.** ``claim_id`` had to be ``clm_`` plus
32 random lowercase hex characters, unique across every answer this deployment
has ever accepted. That is asking a language model to count characters and
generate entropy, and a live run failed on exactly that — a complete,
well-researched answer refused for id length alone, the one correction attempt
spent re-minting ids, the length wrong again, and the run ending with no answer
at all. In v2 a claim carries a ``local_ref``: a short label, unique only
within this one draft. The server issues the real id.

**Restate a value it can read for itself.** ``source_value`` had to equal the
canonical field exactly, and ``rendered_value`` had to be that value under a
transform, rendered to a locale convention. Both are pure functions of an
observation the server already holds. In v2 the model names *which* fact goes
where — ``observation_id`` plus ``field_path`` plus a transform from the
allowlist — and the server resolves, renders and checks. There is nothing left
for the model to get numerically wrong, which is a stronger guarantee than
checking its arithmetic afterwards.

**Copy a concrete example.** The tool description carried a full, valid-looking
claim id. Examples in a prompt are instructions to imitate; that one was
imitated. v2's description has no id in it because v2 has no id field.

What the model still owns is what it is good at: which fact answers the
question, what the prose says, and which limitations apply.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .candidate_wire import ID_PATTERN, TRANSFORM_PATTERN, LimitationCandidate

#: ADS plan section 9.3/W5-06. Allowed values mirror
#: domain/evidence_relation.py's enums exactly — kept as plain string
#: literals here (rather than importing the domain enums) so the wire layer
#: stays free of domain-object coupling, the same separation wire.py/wire_v2.py
#: already keep from domain/answer.py.
_SOURCE_CLASSES = (
    "predictor_fact", "explanation_fact", "external_experimental",
    "external_regulatory", "report_fact", "agent_synthesis",
)
_RELATIONS = ("supports", "contradicts", "contextual", "insufficient", "not_applicable")
_DIRECTNESS = ("direct", "indirect")
_APPLICABILITY = ("ok", "limited", "out_of_domain", "not_applicable")
_STRENGTH = ("weak", "moderate", "strong", "not_assessed")

SCHEMA_VERSION = "grounded-answer-v2"

#: A label, not an identifier. Short, readable, and scoped to one draft — two
#: runs may both use ``claim_1`` and each gets its own global id.
LOCAL_REF_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,31}$")

#: Kinds whose value the server resolves. The model may not send one.
FIELD_BACKED = frozenset({"numeric", "classification"})


class _Wire(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClaimCandidateV2(_Wire):
    local_ref: str = Field(
        description=(
            "A short label for this claim, unique within this draft — e.g. 'herg_p' "
            "or 'claim_1'. Lowercase letters, digits and underscores. The server "
            "issues the permanent identifier."
        )
    )
    kind: Literal[
        "numeric", "classification", "scientific", "comparison", "limitation",
        "recommendation",
    ]
    text: str = Field(min_length=1, max_length=2000)
    observation_id: str | None = None
    field_path: str | None = None
    transform: str = "identity"
    citation_ids: list[str] = Field(default_factory=list)
    input_local_refs: list[str] = Field(
        default_factory=list,
        description=(
            "For kind=comparison only: the two other local_refs in this draft that "
            "the difference or ratio is computed from, first minus/over second."
        ),
    )

    @field_validator("local_ref")
    @classmethod
    def _label_shape(cls, value: str) -> str:
        if not LOCAL_REF_PATTERN.match(value):
            raise ValueError(
                f"local_ref {value!r} must be 1-32 characters of lowercase letters, "
                "digits and underscores, starting with a letter"
            )
        return value

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

    @model_validator(mode="after")
    def _field_backed_claims_name_a_field(self) -> "ClaimCandidateV2":
        if self.kind in FIELD_BACKED and not self.field_path:
            raise ValueError(
                f"a {self.kind} claim must name the field_path its value comes from; "
                "the server reads the value itself"
            )
        return self


class RecommendationCandidateV2(_Wire):
    text: str = Field(min_length=1, max_length=1000)
    basis_local_refs: list[str] = Field(default_factory=list)


class EvidenceRelationInputV2(_Wire):
    """One source's proposed bearing on one proposition (ADS plan section
    9.3/W5-06, ADR 0010). The model proposes; the server independently
    resolves ``source_id`` against a real, session-owned artifact before
    persisting (domain/evidence_relation.py's own docstring: an unresolved
    source is rejected, not partially trusted).

    Deliberately no ``proposition_id`` field: minting a fresh, globally
    unique id is exactly the "make the model generate entropy" failure
    mode ``ClaimCandidateV2``'s ``local_ref`` exists to avoid (see this
    module's own docstring) — the server mints one, grouping relations that
    give the same ``proposition`` text within one draft under the same id.
    """

    proposition: str = Field(min_length=1, max_length=500)
    source_class: Literal[_SOURCE_CLASSES]
    source_id: str
    relation: Literal[_RELATIONS]
    directness: Literal[_DIRECTNESS] = "direct"
    applicability: Literal[_APPLICABILITY] = "ok"
    strength: Literal[_STRENGTH] = "not_assessed"
    reason_codes: list[str] = Field(default_factory=list, max_length=8)
    endpoint: str | None = None
    species: str | None = None
    dose: str | None = None
    use_context: str | None = None
    input_source_refs: list[str] = Field(
        default_factory=list, max_length=8,
        description=(
            "Required when source_class is agent_synthesis, forbidden otherwise: the "
            "artifacts your synthesis was derived from, as 'observation:<id>', "
            "'evidence:<id>' (read with get_evidence_record in this run) or "
            "'report:<id>'. A synthesis is never its own source."
        ),
    )

    @field_validator("input_source_refs")
    @classmethod
    def _typed_refs(cls, value: list[str]) -> list[str]:
        for ref in value:
            kind, _, identifier = ref.partition(":")
            if kind not in ("observation", "evidence", "report") or not ID_PATTERN.match(identifier):
                raise ValueError(
                    f"input_source_refs entry {ref!r} must be observation:<id>, evidence:<id> "
                    "or report:<id> with an id a tool handed you"
                )
        return value

    @field_validator("source_id")
    @classmethod
    def _source_id_shape(cls, value: str) -> str:
        if not ID_PATTERN.match(value):
            raise ValueError(
                f"source_id {value!r} is not a ToxAgent identifier — it must be a real "
                "observation_id, evidence_id or report_id handed to you by a tool, never one "
                "you invent"
            )
        return value

    @model_validator(mode="after")
    def _reason_codes_required_unless_a_null_outcome(self) -> "EvidenceRelationInputV2":
        if self.relation not in ("insufficient", "not_applicable") and not self.reason_codes:
            raise ValueError(
                f"a {self.relation} relation must carry at least one reason code — 'we "
                "assessed this but decline to say why' is not an accepted state"
            )
        if self.source_class == "agent_synthesis" and not self.input_source_refs:
            raise ValueError(
                "an agent_synthesis relation must list input_source_refs — the artifacts "
                "the synthesis was derived from; a synthesis cannot be its own provenance"
            )
        if self.source_class != "agent_synthesis" and self.input_source_refs:
            raise ValueError("input_source_refs is only for an agent_synthesis relation")
        return self


_POSTURE_VALUES = ("proceed", "hold", "deprioritize", "insufficient", "not_applicable")
_POSTURE_SCOPES = ("drug_candidate", "api", "excipient", "solvent", "intermediate", "unknown")
_POSTURE_CONFIDENCE = ("weak", "moderate", "strong", "not_assessed")


class DevelopmentPostureInputV2(_Wire):
    """The decision-support answer's scoped R&D recommendation (ADS plan
    section 10.1/10.2, W6). ``basis_local_refs``/``contrary_local_refs``
    name claims in *this same draft* by their ``local_ref`` — the server
    resolves them to the real ``claim_id`` it issues, the same
    local-ref-not-a-minted-id principle ``ClaimCandidateV2`` and
    ``RecommendationCandidateV2`` already follow.
    """

    value: Literal[_POSTURE_VALUES]
    scope: Literal[_POSTURE_SCOPES]
    confidence_band: Literal[_POSTURE_CONFIDENCE]
    basis_local_refs: list[str] = Field(default_factory=list)
    contrary_local_refs: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1, max_length=2000)
    conditions: list[str] = Field(default_factory=list, max_length=8)
    recommended_next_steps: list[str] = Field(default_factory=list, max_length=8)


class GroundedAnswerDraftV2(_Wire):
    """One complete draft. The final action of a conversational run."""

    schema_version: Literal["grounded-answer-v2"] = SCHEMA_VERSION
    answer_markdown: str = Field(min_length=1, max_length=20_000)
    claims: list[ClaimCandidateV2] = Field(default_factory=list, max_length=64)
    limitations: list[LimitationCandidate] = Field(default_factory=list, max_length=16)
    recommended_next_steps: list[RecommendationCandidateV2] = Field(
        default_factory=list, max_length=8
    )
    #: Decision-support only in practice (other profiles have no reason to
    #: populate this), but not schema-gated by profile — an empty list is a
    #: no-op everywhere else, the same way other optional fields here are.
    evidence_relations: list[EvidenceRelationInputV2] = Field(
        default_factory=list, max_length=16
    )
    #: Decision-support only in practice, same as evidence_relations above.
    development_posture: DevelopmentPostureInputV2 | None = None

    @model_validator(mode="after")
    def _local_refs_are_unique_and_resolvable(self) -> "GroundedAnswerDraftV2":
        refs = [claim.local_ref for claim in self.claims]
        duplicates = sorted({ref for ref in refs if refs.count(ref) > 1})
        if duplicates:
            raise ValueError(f"local_ref repeated within this draft: {duplicates}")
        known = set(refs)
        for claim in self.claims:
            unknown = sorted(set(claim.input_local_refs) - known)
            if unknown:
                raise ValueError(
                    f"claim {claim.local_ref!r} names input_local_refs this draft does "
                    f"not contain: {unknown}"
                )
            if claim.local_ref in claim.input_local_refs:
                raise ValueError(f"claim {claim.local_ref!r} cannot be its own input")
        for step in self.recommended_next_steps:
            unknown = sorted(set(step.basis_local_refs) - known)
            if unknown:
                raise ValueError(
                    f"a recommendation names basis_local_refs this draft does not "
                    f"contain: {unknown}"
                )
        if self.development_posture is not None:
            posture = self.development_posture
            unknown = sorted(
                (set(posture.basis_local_refs) | set(posture.contrary_local_refs)) - known
            )
            if unknown:
                raise ValueError(
                    f"development_posture names local_refs this draft does not contain: "
                    f"{unknown}"
                )
        return self
