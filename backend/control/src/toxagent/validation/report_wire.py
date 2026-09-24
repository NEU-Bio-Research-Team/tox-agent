"""The report draft wire shape (spec section 5, workstream R0).

What a model actually submits to ``submit_report_draft``: strings, ids and
enums, never domain objects — for the same reason ``wire.py`` exists for a
grounded answer. A draft with the wrong *shape* is refused by pydantic before
any validator has to reason about it; a draft with the right shape and a wrong
*value* (a number that does not match its observation, a figure attached to the
wrong endpoint) reaches the deterministic report validator as ordinary,
well-formed data and comes back as a typed, correctable violation.

Two shapes here are narrower than they first look, on purpose:

**A section names content by id; it never carries content.** Figures, tables
and claims are referenced, so a model cannot invent a figure by describing one,
and the validator resolves every reference against what the server actually
produced.

**Values are not in this file.** A claim candidate reuses ``ClaimCandidate``
from ``wire.py``, whose ``source_value`` must equal the canonical field — the
report compiler re-renders values from observations at compile time, so the
draft's job is to say *which* fact goes where, not what the number is.
"""
from __future__ import annotations

import re
from typing import Literal

import hashlib
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..domain.report import REQUIRED_SECTION_IDS
from .wire import CLAIM_ID_PATTERN, ClaimCandidate, LimitationCandidate

ID_PATTERN = re.compile(r"^[a-z]{3,5}_[0-9a-f]{32}$")

SectionId = Literal[
    "executive_summary",
    "substance_profile",
    "predictor_results",
    "explanation_and_visuals",
    "external_evidence",
    "integrated_interpretation",
    "conclusions",
    "recommendations",
    "limitations",
    "references",
    "provenance_appendix",
]

SourceClassName = Literal[
    "structure_fact", "predictor_fact", "explanation_fact",
    "external_evidence", "agent_synthesis", "recommendation",
]

RelationName = Literal["supports", "contradicts", "contextualizes", "insufficient"]

GapReasonName = Literal[
    "endpoint_not_served", "explanation_failed", "explanation_partial",
    "compound_identity_unresolved", "no_relevant_evidence",
    "external_evidence_not_requested", "provider_unavailable", "budget_exhausted",
]


class _Wire(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _typed_id(prefix: str, field_name: str):
    def _check(value: str) -> str:
        if not ID_PATTERN.match(value) or not value.startswith(f"{prefix}_"):
            raise ValueError(
                f"{field_name} must be a {prefix!r} identifier returned by a tool, "
                f"got {value!r}"
            )
        return value

    return _check


class ReportSectionCandidate(_Wire):
    section_id: SectionId
    heading: str = Field(min_length=1, max_length=200)
    body_markdown: str = Field(default="", max_length=20_000)
    claim_ids: list[str] = Field(default_factory=list, max_length=64)
    table_ids: list[str] = Field(default_factory=list, max_length=16)
    figure_ids: list[str] = Field(
        default_factory=list, max_length=32,
        description=(
            "Figure ids from get_or_create_explanation / get_explanation_package only. "
            "A figure cannot be created by naming one here."
        ),
    )
    gap_ids: list[str] = Field(default_factory=list, max_length=32)
    source_classes: list[SourceClassName] = Field(default_factory=list, max_length=6)

    @field_validator("figure_ids")
    @classmethod
    def _figure_id_shape(cls, value: list[str]) -> list[str]:
        return [_typed_id("fig", "section.figure_ids[]")(v) for v in value]


class ReportTableCandidate(_Wire):
    table_id: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=200)
    columns: list[str] = Field(min_length=1, max_length=12)
    rows: list[list[str]] = Field(default_factory=list, max_length=64)
    source_class: SourceClassName
    row_claim_ids: list[list[str]] = Field(default_factory=list, max_length=64)


class ExplanationPackageRef(_Wire):
    """A reference to an explanation the *server* produced.

    Deliberately not the explanation itself: the atoms, the figure bytes and
    the unmapped mass come from the explanation observation, and a model that
    could restate them here could restate them wrongly.
    """

    explanation_id: str
    #: Repeated from the tool result so a mismatch between what the model
    #: thinks it is citing and what the id resolves to is a typed violation
    #: rather than a silently mislabelled figure (spec section 11).
    endpoint: Literal["clintox", "herg", "tox21"]
    task: str | None = None
    narrative_claim_ids: list[str] = Field(default_factory=list, max_length=16)

    @field_validator("explanation_id")
    @classmethod
    def _shape(cls, value: str) -> str:
        return _typed_id("xpl", "explanation_ref.explanation_id")(value)


class EvidenceSynthesisCandidate(_Wire):
    proposition: str = Field(min_length=1, max_length=2000)
    relation: RelationName
    evidence_ids: list[str] = Field(default_factory=list, max_length=16)
    endpoint: str | None = None
    assay: str | None = None
    organism: str | None = None
    dose_context: str | None = None
    quality_notes: list[str] = Field(default_factory=list, max_length=8)
    conflict_id: str | None = None

    @field_validator("evidence_ids")
    @classmethod
    def _evidence_shape(cls, value: list[str]) -> list[str]:
        return [_typed_id("evd", "evidence_synthesis.evidence_ids[]")(v) for v in value]


class ConclusionCandidate(_Wire):
    conclusion_id: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=1, max_length=4000)
    basis_claim_ids: list[str] = Field(min_length=1, max_length=32)
    endpoint: str | None = Field(
        default=None,
        description=(
            "The one endpoint this conclusion is about. Omit it only together with "
            "is_integrated=true, which labels the text as an integrated screening "
            "interpretation rather than a per-endpoint finding."
        ),
    )
    task: str | None = None
    is_integrated: bool = False


class RecommendationCandidate(_Wire):
    recommendation_id: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=1, max_length=2000)
    basis_claim_ids: list[str] = Field(min_length=1, max_length=32)
    action_category: Literal[
        "in_vitro_assay", "in_silico_followup", "literature_review",
        "structure_modification", "data_collection", "no_action_indicated",
    ]
    priority: Literal["high", "medium", "low"]
    rationale: str = Field(min_length=1, max_length=2000)
    conditions: str = Field(default="", max_length=1000)


class GapCandidate(_Wire):
    gap_id: str = Field(min_length=1, max_length=64)
    reason: GapReasonName
    detail: str = Field(min_length=1, max_length=1000)
    section_id: SectionId
    endpoint: str | None = None
    task: str | None = None


class ReportDraftCandidate(_Wire):
    """One complete draft. The final action of a report-builder run."""

    schema_version: Literal["report-draft-v1"] = "report-draft-v1"
    report_build_id: str
    title: str = Field(min_length=1, max_length=300)
    sections: list[ReportSectionCandidate] = Field(
        min_length=len(REQUIRED_SECTION_IDS), max_length=32,
        description=(
            "All eleven required sections, every time. A section whose content is "
            "unavailable carries a gap_id and says so; it is never omitted."
        ),
    )
    claims: list[ClaimCandidate] = Field(default_factory=list, max_length=256)
    tables: list[ReportTableCandidate] = Field(default_factory=list, max_length=32)
    explanations: list[ExplanationPackageRef] = Field(default_factory=list, max_length=32)
    evidence_synthesis: list[EvidenceSynthesisCandidate] = Field(
        default_factory=list, max_length=64
    )
    conclusions: list[ConclusionCandidate] = Field(default_factory=list, max_length=32)
    recommendations: list[RecommendationCandidate] = Field(default_factory=list, max_length=16)
    limitations: list[LimitationCandidate] = Field(default_factory=list, max_length=24)
    gaps: list[GapCandidate] = Field(default_factory=list, max_length=32)

    @model_validator(mode="before")
    @classmethod
    def _canonicalise_claim_ids(cls, data: Any) -> Any:
        """Rewrite model-minted claim ids into the stored ``clm_``+32-hex shape.

        A claim id is a label the model invents, not a fact it reports: nothing
        downstream reads meaning out of the 32 characters, they only have to be
        well-shaped, unique, and referred to consistently from the sections,
        conclusions, recommendations and comparisons that cite them.

        Asking a language model to emit a few dozen exact-length random hex
        strings is asking it to count characters, which it does not reliably do.
        Observed live (run_b9f5971651a0455d8c57f3761f7535c9, 2026-09-09): a
        complete, well-researched draft was refused by ``ClaimCandidate``'s
        field validator for id length alone, the run spent its one correction
        attempt re-minting the ids, got the length wrong again, and ended with
        no document at all. The runtime's own closing words were "hệ thống từ
        chối cả hai lần gửi do lỗi định dạng mã claim".

        Hashing the model's own string is what keeps this honest. It is a pure
        function of what the model wrote, so every reference to the same label
        lands on the same id and referential integrity survives the rewrite;
        two distinct labels stay distinct, so a genuine ``duplicate_claim_id``
        is still caught downstream rather than papered over. No value, citation
        or number is touched — a draft whose *content* is wrong still comes back
        as the typed, correctable violation it should be.
        """
        if not isinstance(data, dict):
            return data
        claims = data.get("claims")
        if not isinstance(claims, list):
            return data

        rewrites: dict[str, str] = {}
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            raw = claim.get("claim_id")
            if not isinstance(raw, str) or not raw or CLAIM_ID_PATTERN.match(raw):
                continue
            digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
            rewrites[raw] = f"clm_{digest}"
        if not rewrites:
            return data

        def substitute(value: Any) -> Any:
            if isinstance(value, str):
                return rewrites.get(value, value)
            if isinstance(value, list):
                return [substitute(item) for item in value]
            if isinstance(value, dict):
                return {key: substitute(item) for key, item in value.items()}
            return value

        return substitute(data)

    @field_validator("report_build_id")
    @classmethod
    def _build_shape(cls, value: str) -> str:
        return _typed_id("rpb", "draft.report_build_id")(value)
