"""ReportArtifact and its build aggregate — the canonical report contract.

Spec: docs/spec/TOXAGENT_REPORT_BUILDER_PLAN.md sections 5 and 10. The artifact
is the source of truth; Markdown, HTML and PDF are renderings of it, and a
renderer that disagrees with the artifact is a renderer bug, never a second
opinion about what the report says.

Three things here are deliberate and load-bearing:

**No aggregate verdict field.** Exactly as ``GroundedAnswer`` has no overall
toxicity score (ADR 0002), ``ReportArtifact`` has no field into which "this
compound is safe" could be written. Endpoint-level conclusions each name their
own endpoint; there is nowhere to put a collapsed one (spec section 3.5).

**Missing data is a value, not an absence.** ``REQUIRED_SECTION_IDS`` must all
be present even when a section reports only a gap, so a failed provider or a
failed explainer cannot make a section quietly disappear (spec sections 3.6,
11 "Missing-data visibility").

**Source class is carried on every piece of content.** A predictor number, an
explanation, an external record and an agent synthesis are four different kinds
of claim about the world, and the report is required never to blur them
(spec section 3.4), so the distinction lives in the type rather than in
whichever sentence a model happened to write.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Final

from .ids import (
    ANALYSIS,
    ATTACHMENT,
    EVIDENCE,
    EXPLANATION,
    FIGURE,
    OBSERVATION,
    REPORT,
    REPORT_BUILD,
    RUN,
    SESSION,
    new_id,
    require_id,
)
from .provenance import content_sha256

#: v2 adds ``references``: the resolved, immutable snapshot of every cited
#: source (REP-01). v1 artifacts carried only ``evidence_id`` strings, which a
#: renderer could print but not turn into anything a reader could open, so the
#: readers below accept both and treat a v1 report as one with no references
#: rather than as one with none *cited*.
SCHEMA_VERSION: Final = "toxagent-report-v2"

#: Every artifact schema version a reader must handle, newest first.
SCHEMA_VERSIONS: Final[tuple[str, ...]] = ("toxagent-report-v2", "toxagent-report-v1")
BUILD_REQUEST_SCHEMA_VERSION: Final = "report-build-request-v1"

#: Spec section 5.3. Stable IDs; the heading a person reads is a rendering
#: decision and may change without changing these.
REQUIRED_SECTION_IDS: Final[tuple[str, ...]] = (
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
)

#: The English-first scope (spec section 5.1). A second language is a product
#: decision with its own limitation wording, not a parameter.
SUPPORTED_REPORT_LANGUAGES: Final[frozenset[str]] = frozenset({"en"})

#: Media types a figure may be. An allowlist rather than a denylist because the
#: renderer inlines these, and "whatever the explainer produced" is how a
#: remote-embed or script payload gets into a report (spec section 11).
ALLOWED_FIGURE_MEDIA_TYPES: Final[frozenset[str]] = frozenset(
    {"image/svg+xml", "image/png"}
)


class BuildStage(str, Enum):
    """Spec section 10. Persisted so a build resumes rather than repeating a
    billable predictor or provider call after a restart."""

    QUEUED = "queued"
    PREPARING_ANALYSIS = "preparing_analysis"
    ASSEMBLING_SUBSTANCE = "assembling_substance"
    ASSEMBLING_PREDICTIONS = "assembling_predictions"
    GENERATING_EXPLANATIONS = "generating_explanations"
    RESEARCHING_EVIDENCE = "researching_evidence"
    SYNTHESIZING = "synthesizing"
    VALIDATING = "validating"
    RENDERING = "rendering"
    COMPLETED = "completed"
    COMPLETED_WITH_GAPS = "completed_with_gaps"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STAGES: Final[frozenset[BuildStage]] = frozenset(
    {
        BuildStage.COMPLETED,
        BuildStage.COMPLETED_WITH_GAPS,
        BuildStage.FAILED,
        BuildStage.CANCELLED,
    }
)

#: The forward path. Every non-terminal stage may additionally fail or be
#: cancelled; that is added below rather than repeated eleven times.
_FORWARD: Final[dict[BuildStage, frozenset[BuildStage]]] = {
    BuildStage.QUEUED: frozenset({BuildStage.PREPARING_ANALYSIS}),
    BuildStage.PREPARING_ANALYSIS: frozenset({BuildStage.ASSEMBLING_SUBSTANCE}),
    BuildStage.ASSEMBLING_SUBSTANCE: frozenset({BuildStage.ASSEMBLING_PREDICTIONS}),
    BuildStage.ASSEMBLING_PREDICTIONS: frozenset(
        # Explanations are skippable by request (include_explanations=false);
        # nothing else on this path is.
        {BuildStage.GENERATING_EXPLANATIONS, BuildStage.RESEARCHING_EVIDENCE,
         BuildStage.SYNTHESIZING}
    ),
    BuildStage.GENERATING_EXPLANATIONS: frozenset(
        {BuildStage.RESEARCHING_EVIDENCE, BuildStage.SYNTHESIZING}
    ),
    BuildStage.RESEARCHING_EVIDENCE: frozenset({BuildStage.SYNTHESIZING}),
    # Validation returns to synthesis for the one permitted correction attempt
    # (spec section 10 stage 4); the attempt cap lives on the aggregate, not
    # in this table, so the transition itself stays legal exactly once.
    BuildStage.SYNTHESIZING: frozenset({BuildStage.VALIDATING}),
    BuildStage.VALIDATING: frozenset({BuildStage.SYNTHESIZING, BuildStage.RENDERING}),
    BuildStage.RENDERING: frozenset(
        {BuildStage.COMPLETED, BuildStage.COMPLETED_WITH_GAPS}
    ),
}

ALLOWED_STAGE_TRANSITIONS: Final[dict[BuildStage, frozenset[BuildStage]]] = {
    stage: (
        frozenset()
        if stage in TERMINAL_STAGES
        else _FORWARD.get(stage, frozenset()) | {BuildStage.FAILED, BuildStage.CANCELLED}
    )
    for stage in BuildStage
}


class SourceClass(str, Enum):
    """Spec section 3.4's table, as a type.

    The report must never blur these, so nothing that carries content in this
    module is constructible without saying which one it is.
    """

    STRUCTURE_FACT = "structure_fact"
    PREDICTOR_FACT = "predictor_fact"
    EXPLANATION_FACT = "explanation_fact"
    EXTERNAL_EVIDENCE = "external_evidence"
    AGENT_SYNTHESIS = "agent_synthesis"
    RECOMMENDATION = "recommendation"


class EvidenceRelation(str, Enum):
    """Spec section 5.6. ``INSUFFICIENT`` is a real outcome: "we looked and
    found nothing that bears on this" is information, and suppressing it in
    favour of a weak match is the failure this enum exists to make nameable."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    CONTEXTUALIZES = "contextualizes"
    INSUFFICIENT = "insufficient"


class GapReason(str, Enum):
    """Why something the report was asked for is not in it.

    A closed set because ``completed_with_gaps`` is a product outcome that has
    to be countable per reason (spec section 14's tracked metrics), and free
    text is not countable.
    """

    ENDPOINT_NOT_SERVED = "endpoint_not_served"
    EXPLANATION_FAILED = "explanation_failed"
    EXPLANATION_PARTIAL = "explanation_partial"
    COMPOUND_IDENTITY_UNRESOLVED = "compound_identity_unresolved"
    NO_RELEVANT_EVIDENCE = "no_relevant_evidence"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    BUDGET_EXHAUSTED = "budget_exhausted"


class ExplanationStatus(str, Enum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ReportFigure:
    """A persisted visual artifact, referenced never inlined.

    ``content_sha256`` is over the stored bytes, so a renderer that resolves
    ``attachment_id`` can prove it drew the same image the artifact was
    validated against (spec section 11 "Figure integrity").
    """

    figure_id: str
    attachment_id: str
    media_type: str
    caption: str
    alt_text: str
    content_sha256: str
    renderer_version: str
    #: What this figure depicts, so a figure can never be silently reattached
    #: to a different endpoint's section.
    endpoint: str | None = None
    task: str | None = None
    observation_id: str | None = None

    def __post_init__(self) -> None:
        require_id(self.figure_id, FIGURE, field="figure.figure_id")
        require_id(self.attachment_id, ATTACHMENT, field="figure.attachment_id")
        if self.observation_id is not None:
            require_id(self.observation_id, OBSERVATION, field="figure.observation_id")
        if self.media_type not in ALLOWED_FIGURE_MEDIA_TYPES:
            raise ValueError(
                f"figure.media_type {self.media_type!r} is not one this product renders "
                f"({sorted(ALLOWED_FIGURE_MEDIA_TYPES)})"
            )
        if not self.caption.strip():
            raise ValueError("figure.caption must not be empty")
        if not self.alt_text.strip():
            # Not a style rule: the HTML and PDF renderings are the report, and
            # a figure with no text alternative makes part of the report
            # unreadable to a reader who cannot see it.
            raise ValueError("figure.alt_text must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "figure_id": self.figure_id,
            "attachment_id": self.attachment_id,
            "media_type": self.media_type,
            "caption": self.caption,
            "alt_text": self.alt_text,
            "content_sha256": self.content_sha256,
            "renderer_version": self.renderer_version,
            "endpoint": self.endpoint,
            "task": self.task,
            "observation_id": self.observation_id,
        }


@dataclass(frozen=True, slots=True)
class ExplanationHighlights:
    """The text half of an explanation package.

    ``unmapped_importance`` is kept even when it is zero, and is ``None`` only
    when the explainer did not report it: "how much of the attribution mass
    landed on nothing we can point at" is exactly the number an explanation
    narrative is tempted to omit (spec section 6.2, eval scenario 5).
    """

    positive_contributors: tuple[dict[str, Any], ...] = ()
    negative_contributors: tuple[dict[str, Any], ...] = ()
    unmapped_importance: float | None = None
    narrative_claim_ids: tuple[str, ...] = ()
    #: How the total importance mass was split between atoms/bonds, the
    #: tokenizer's own sequence markers, and everything else (P1-6). One float
    #: could not tell those apart, and the audit's report described 35.84% of
    #: special-token mass as "no unmapped mass".
    coverage: dict[str, Any] = field(default_factory=dict)

    @property
    def contributor_count(self) -> int:
        return len(self.positive_contributors) + len(self.negative_contributors)

    @property
    def coverage_status(self) -> str:
        return str(self.coverage.get("coverage_status") or "unknown")

    def to_dict(self) -> dict[str, Any]:
        return {
            "positive_contributors": [dict(c) for c in self.positive_contributors],
            "negative_contributors": [dict(c) for c in self.negative_contributors],
            "unmapped_importance": self.unmapped_importance,
            "narrative_claim_ids": list(self.narrative_claim_ids),
            "coverage": dict(self.coverage),
        }


@dataclass(frozen=True, slots=True)
class ExplanationPackage:
    """One endpoint, and for Tox21 one assay (spec section 5.5).

    The figure and the highlights are generated from the same canonical
    explanation payload and point at the same observation. Nothing here merges
    two assays: the twelve Tox21 targets are independent measurements and a
    combined explanation would not mean anything.
    """

    explanation_id: str
    observation_id: str
    endpoint: str
    task: str | None
    method: str | None
    status: ExplanationStatus
    figure: ReportFigure | None = None
    highlights: ExplanationHighlights = field(default_factory=ExplanationHighlights)
    required_limitations: tuple[str, ...] = ("attribution_not_causality",)
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        require_id(self.explanation_id, EXPLANATION, field="explanation.explanation_id")
        require_id(self.observation_id, OBSERVATION, field="explanation.observation_id")
        if "attribution_not_causality" not in self.required_limitations:
            # The one limitation an explanation can never be published without.
            raise ValueError(
                "an explanation package must carry attribution_not_causality: "
                "attribution is model behaviour, not a chemical mechanism"
            )
        if self.status is ExplanationStatus.FAILED and self.figure is not None:
            raise ValueError("a failed explanation has no figure to show")
        if self.figure is not None:
            if self.figure.endpoint not in (None, self.endpoint):
                raise ValueError(
                    "explanation figure names a different endpoint than its package"
                )
            if self.figure.task not in (None, self.task):
                raise ValueError(
                    "explanation figure names a different Tox21 assay than its package"
                )
            if self.figure.observation_id not in (None, self.observation_id):
                raise ValueError(
                    "explanation figure names a different observation than its package"
                )

    @property
    def target(self) -> tuple[str, str | None]:
        return (self.endpoint, self.task)

    def to_dict(self) -> dict[str, Any]:
        return {
            "explanation_id": self.explanation_id,
            "observation_id": self.observation_id,
            "endpoint": self.endpoint,
            "task": self.task,
            "method": self.method,
            "status": self.status.value,
            "figure": self.figure.to_dict() if self.figure else None,
            "extracted_highlights": self.highlights.to_dict(),
            "required_limitations": list(self.required_limitations),
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True, slots=True)
class SubstanceProfile:
    """Spec section 5.4.

    Canonical SMILES comes from the analysis snapshot. Everything else is
    external and carries a field-level source reference; an unresolved field
    stays ``None`` rather than being inferred from a similar compound.
    """

    canonical_smiles: str
    structure_figure_id: str | None = None
    preferred_name: str | None = None
    synonyms: tuple[str, ...] = ()
    identifiers: dict[str, Any] = field(default_factory=dict)
    properties: tuple[dict[str, Any], ...] = ()
    #: field name -> evidence id / provider ref that supplied it.
    source_refs: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.canonical_smiles:
            raise ValueError("substance_profile.canonical_smiles is required")
        if self.structure_figure_id is not None:
            require_id(
                self.structure_figure_id, FIGURE, field="substance_profile.structure_figure_id"
            )

    @property
    def is_identity_resolved(self) -> bool:
        return bool(self.preferred_name or any(self.identifiers.values()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_smiles": self.canonical_smiles,
            "structure_figure_id": self.structure_figure_id,
            "preferred_name": self.preferred_name,
            "synonyms": list(self.synonyms),
            "identifiers": dict(self.identifiers),
            "properties": [dict(p) for p in self.properties],
            "source_refs": dict(self.source_refs),
        }


@dataclass(frozen=True, slots=True)
class EvidenceSynthesis:
    """One external proposition and what the read records did to it.

    Spec section 5.6. ``evidence_ids`` may be empty only for
    ``relation=insufficient`` — a supported or contradicted proposition with no
    record behind it is an uncited claim wearing an evidence label.
    """

    synthesis_id: str
    proposition: str
    relation: EvidenceRelation
    evidence_ids: tuple[str, ...] = ()
    endpoint: str | None = None
    assay: str | None = None
    organism: str | None = None
    dose_context: str | None = None
    quality_notes: tuple[str, ...] = ()
    conflict_id: str | None = None

    def __post_init__(self) -> None:
        if not self.proposition.strip():
            raise ValueError("evidence_synthesis.proposition must not be empty")
        if self.relation is not EvidenceRelation.INSUFFICIENT and not self.evidence_ids:
            raise ValueError(
                f"an evidence synthesis with relation {self.relation.value!r} must name at "
                "least one evidence record it was read from"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "synthesis_id": self.synthesis_id,
            "proposition": self.proposition,
            "relation": self.relation.value,
            "evidence_ids": list(self.evidence_ids),
            "endpoint": self.endpoint,
            "assay": self.assay,
            "organism": self.organism,
            "dose_context": self.dose_context,
            "quality_notes": list(self.quality_notes),
            "conflict_id": self.conflict_id,
        }


#: URL schemes a rendered citation may become a link to. An allowlist, and
#: HTTPS-only: a report is a downloadable document that gets opened later, and
#: ``javascript:``/``data:`` in an href is the one thing a citation must never
#: be able to smuggle past a renderer (spec REP-01, section 11).
SAFE_URL_SCHEMES: Final[frozenset[str]] = frozenset({"https"})


def is_safe_citation_url(url: str | None) -> bool:
    """Whether this URL may be rendered as a clickable link.

    Refusing the link is not refusing the citation: a source with an unusable
    URL still appears in the references with its title, authors and identifier,
    because dropping the source entirely would hide that a claim had a basis.
    """
    if not url:
        return False
    scheme, separator, rest = url.partition("://")
    if not separator or scheme.lower() not in SAFE_URL_SCHEMES:
        return False
    # A host is required. "https://" alone, or a URL whose authority is empty,
    # resolves to nothing and would render as a dead link.
    return bool(rest.split("/", 1)[0].strip())


@dataclass(frozen=True, slots=True)
class ReportReference:
    """One cited source, resolved and frozen into the report (spec REP-01).

    The evidence record itself lives in a mutable, session-scoped table: it can
    be superseded, rejected, or retention-expired. A report is immutable and
    citable, so it carries its own snapshot of what each source *was* when the
    report was written, and that snapshot is inside the content hash. Two
    consequences, both intended:

    - Opening a report never fetches anything. No network call on read means no
      report that reports on its readers, and no citation that quietly changes
      what it says between two readings.
    - ``number`` is assigned by first appearance in report order, so ``[1]`` is
      the same source in the in-app view, the Markdown, the HTML and the PDF.
      Numbering derived independently per renderer is numbering that disagrees.

    ``link_url`` is ``None`` when the canonical URL is missing or not an HTTPS
    URL. The metadata stays; only the anchor goes.
    """

    evidence_id: str
    number: int
    title: str
    provider: str
    #: The URL as recorded. Kept even when unsafe, so an audit can see what the
    #: provider actually returned rather than only that it was refused.
    canonical_url: str | None = None
    authors: tuple[str, ...] = ()
    published_at: str | None = None
    identifier: dict[str, Any] = field(default_factory=dict)
    source_type: str | None = None
    source_quality_tier: str | None = None
    retrieved_at: str | None = None
    #: Set when the record was not in ``accepted`` status at compile time. A
    #: citation that cannot be resolved must be visible, never silently absent.
    unresolved_reason: str | None = None

    def __post_init__(self) -> None:
        require_id(self.evidence_id, EVIDENCE, field="reference.evidence_id")
        if self.number < 1:
            raise ValueError("reference numbering starts at 1")

    @property
    def link_url(self) -> str | None:
        return self.canonical_url if is_safe_citation_url(self.canonical_url) else None

    @property
    def short_form(self) -> str:
        """"Author et al. (year)" where both are known. The form a reader scans."""
        year = (self.published_at or "")[:4]
        if self.authors:
            lead = self.authors[0]
            trailing = " et al." if len(self.authors) > 1 else ""
            return f"{lead}{trailing}" + (f" ({year})" if year else "")
        return f"{self.provider}" + (f" ({year})" if year else "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "number": self.number,
            "title": self.title,
            "provider": self.provider,
            "canonical_url": self.canonical_url,
            "link_url": self.link_url,
            "authors": list(self.authors),
            "published_at": self.published_at,
            "identifier": dict(self.identifier),
            "source_type": self.source_type,
            "source_quality_tier": self.source_quality_tier,
            "retrieved_at": self.retrieved_at,
            "unresolved_reason": self.unresolved_reason,
            "short_form": self.short_form,
        }


@dataclass(frozen=True, slots=True)
class ReportTable:
    """A table the renderers lay out. Rows are already-rendered strings paired
    with the claim that authorised each cell's value, so a renderer never has
    to reach back into an observation and never has to format a number."""

    table_id: str
    title: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    source_class: SourceClass
    #: row index -> the claim ids backing that row.
    row_claim_ids: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        for row in self.rows:
            if len(row) != len(self.columns):
                raise ValueError(
                    f"table {self.table_id!r} has a row of {len(row)} cells against "
                    f"{len(self.columns)} columns"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_id": self.table_id,
            "title": self.title,
            "columns": list(self.columns),
            "rows": [list(row) for row in self.rows],
            "source_class": self.source_class.value,
            "row_claim_ids": [list(ids) for ids in self.row_claim_ids],
        }


@dataclass(frozen=True, slots=True)
class ReportSection:
    """One of the eleven required sections, or a subsection of one.

    ``body_markdown`` is prose only: every number in it is expected to have a
    claim, exactly as in a grounded answer, and the compiler is what renders
    values into it.
    """

    section_id: str
    heading: str
    body_markdown: str = ""
    claim_ids: tuple[str, ...] = ()
    table_ids: tuple[str, ...] = ()
    figure_ids: tuple[str, ...] = ()
    gap_ids: tuple[str, ...] = ()
    source_classes: tuple[SourceClass, ...] = ()

    def __post_init__(self) -> None:
        if not self.section_id:
            raise ValueError("section.section_id is required")

    @property
    def is_empty(self) -> bool:
        return not (
            self.body_markdown.strip()
            or self.claim_ids
            or self.table_ids
            or self.figure_ids
            or self.gap_ids
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_id": self.section_id,
            "heading": self.heading,
            "body_markdown": self.body_markdown,
            "claim_ids": list(self.claim_ids),
            "table_ids": list(self.table_ids),
            "figure_ids": list(self.figure_ids),
            "gap_ids": list(self.gap_ids),
            "source_classes": [s.value for s in self.source_classes],
        }


@dataclass(frozen=True, slots=True)
class ReportGap:
    """Something the report was asked for and does not have.

    A gap is content: it is rendered, it is counted, and a section that has one
    still exists. Spec section 3.6.
    """

    gap_id: str
    reason: GapReason
    detail: str
    section_id: str
    endpoint: str | None = None
    task: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "gap_id": self.gap_id,
            "reason": self.reason.value,
            "detail": self.detail,
            "section_id": self.section_id,
            "endpoint": self.endpoint,
            "task": self.task,
        }


@dataclass(frozen=True, slots=True)
class ReportConclusion:
    """Endpoint-level, or an explicitly labelled integrated interpretation.

    ``endpoint=None`` is permitted only with ``is_integrated=True``, and the
    validator additionally refuses integrated text that reads as a verdict —
    the type stops the shape, the wording check stops the sentence.
    """

    conclusion_id: str
    text: str
    basis_claim_ids: tuple[str, ...]
    endpoint: str | None = None
    task: str | None = None
    is_integrated: bool = False

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("conclusion.text must not be empty")
        if not self.basis_claim_ids:
            raise ValueError("a conclusion must reference the claims it rests on")
        if self.endpoint is None and not self.is_integrated:
            raise ValueError(
                "a conclusion is either about one endpoint or explicitly marked as an "
                "integrated screening interpretation"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "conclusion_id": self.conclusion_id,
            "text": self.text,
            "basis_claim_ids": list(self.basis_claim_ids),
            "endpoint": self.endpoint,
            "task": self.task,
            "is_integrated": self.is_integrated,
        }


@dataclass(frozen=True, slots=True)
class ReportRecommendation:
    """A proposed follow-up action (spec section 5.7).

    Framed as validation work, never as a diagnosis or a safety guarantee;
    ``basis_claim_ids`` is required by the type because a recommendation with
    no basis is the single most quotable thing a report can get wrong.
    """

    recommendation_id: str
    text: str
    basis_claim_ids: tuple[str, ...]
    action_category: str
    priority: str
    rationale: str
    conditions: str = ""

    def __post_init__(self) -> None:
        if not self.basis_claim_ids:
            raise ValueError("a recommendation must reference its basis claims")
        if not self.text.strip():
            raise ValueError("recommendation.text must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommendation_id": self.recommendation_id,
            "text": self.text,
            "basis_claim_ids": list(self.basis_claim_ids),
            "action_category": self.action_category,
            "priority": self.priority,
            "rationale": self.rationale,
            "conditions": self.conditions,
        }


@dataclass(frozen=True, slots=True)
class ReportRendering:
    """One produced file. Bytes live in the object store; this is the row."""

    rendering_id: str
    format: str
    media_type: str
    object_uri: str
    content_sha256: str
    size_bytes: int
    renderer_version: str
    created_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "rendering_id": self.rendering_id,
            "format": self.format,
            "media_type": self.media_type,
            "object_uri": self.object_uri,
            "content_sha256": self.content_sha256,
            "size_bytes": self.size_bytes,
            "renderer_version": self.renderer_version,
            "created_at": self.created_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class ReportArtifact:
    """The canonical report. Immutable; a rebuild is a new version.

    Reaching this type means the deterministic validator already passed, in the
    same way that reaching ``GroundedAnswer`` does. The draft a model submits is
    a separate wire model in ``validation.report_wire``.
    """

    id: str
    report_build_id: str
    session_id: str
    analysis_id: str
    title: str
    status: BuildStage
    subject: SubstanceProfile
    sections: tuple[ReportSection, ...]
    tables: tuple[ReportTable, ...]
    figures: tuple[ReportFigure, ...]
    claims: tuple[Any, ...]
    explanations: tuple[ExplanationPackage, ...]
    evidence_synthesis: tuple[EvidenceSynthesis, ...]
    conclusions: tuple[ReportConclusion, ...]
    recommendations: tuple[ReportRecommendation, ...]
    limitations: tuple[Any, ...]
    gaps: tuple[ReportGap, ...]
    references: tuple[ReportReference, ...]
    provenance: dict[str, Any]
    renderings: tuple[ReportRendering, ...]
    content_sha256: str
    created_at: datetime
    report_language: str = "en"
    schema_version: str = SCHEMA_VERSION
    #: The report this one supersedes, when it is a rebuild (spec section 5.2).
    supersedes_report_id: str | None = None
    version: int = 1

    def __post_init__(self) -> None:
        require_id(self.id, REPORT, field="report.id")
        require_id(self.report_build_id, REPORT_BUILD, field="report.report_build_id")
        require_id(self.session_id, SESSION, field="report.session_id")
        require_id(self.analysis_id, ANALYSIS, field="report.analysis_id")
        if self.supersedes_report_id is not None:
            require_id(self.supersedes_report_id, REPORT, field="report.supersedes_report_id")
        if self.status not in (BuildStage.COMPLETED, BuildStage.COMPLETED_WITH_GAPS):
            raise ValueError(
                "a ReportArtifact exists only for a build that finished; a failed or "
                "cancelled build has no artifact"
            )
        missing = set(REQUIRED_SECTION_IDS) - {s.section_id for s in self.sections}
        if missing:
            raise ValueError(
                f"report is missing required section(s): {sorted(missing)}. A section whose "
                "content is unavailable carries a gap; it does not disappear."
            )
        if self.status is BuildStage.COMPLETED and self.gaps:
            raise ValueError(
                "a report with gaps is completed_with_gaps, not completed; the distinction "
                "is what tells a reader the document is partial"
            )
        if self.status is BuildStage.COMPLETED_WITH_GAPS and not self.gaps:
            raise ValueError("completed_with_gaps with no recorded gap")
        section_ids = [s.section_id for s in self.sections]
        if len(section_ids) != len(set(section_ids)):
            raise ValueError("duplicate section_id within one report")
        numbers = [r.number for r in self.references]
        if sorted(numbers) != list(range(1, len(numbers) + 1)):
            raise ValueError(
                f"reference numbering must be 1..{len(numbers)} with no gaps or repeats, "
                f"got {sorted(numbers)}; a renderer that cannot map [n] back to a source "
                "prints a number that means nothing"
            )
        snapshotted = {r.evidence_id for r in self.references}
        if len(snapshotted) != len(self.references):
            raise ValueError("the same evidence record appears twice in references")
        # v1 artifacts carried no references at all; that is a readable older
        # report, not a broken new one. A v2 report that cites a source it did
        # not snapshot is the REP-01 failure itself.
        if self.schema_version != "toxagent-report-v1":
            missing = sorted(self.cited_evidence_ids - snapshotted)
            if missing:
                raise ValueError(
                    f"cited evidence with no resolved reference snapshot: {missing}. A "
                    "citation the report cannot resolve must be a recorded gap, never an "
                    "id a reader is handed with nothing behind it."
                )

    @classmethod
    def create(
        cls,
        *,
        report_build_id: str,
        session_id: str,
        analysis_id: str,
        title: str,
        subject: SubstanceProfile,
        sections: tuple[ReportSection, ...],
        tables: tuple[ReportTable, ...] = (),
        figures: tuple[ReportFigure, ...] = (),
        claims: tuple[Any, ...] = (),
        explanations: tuple[ExplanationPackage, ...] = (),
        evidence_synthesis: tuple[EvidenceSynthesis, ...] = (),
        conclusions: tuple[ReportConclusion, ...] = (),
        recommendations: tuple[ReportRecommendation, ...] = (),
        limitations: tuple[Any, ...] = (),
        gaps: tuple[ReportGap, ...] = (),
        references: tuple[ReportReference, ...] = (),
        provenance: dict[str, Any] | None = None,
        renderings: tuple[ReportRendering, ...] = (),
        report_language: str = "en",
        supersedes_report_id: str | None = None,
        version: int = 1,
        now: datetime,
    ) -> "ReportArtifact":
        body = {
            "schema_version": SCHEMA_VERSION,
            "analysis_id": analysis_id,
            "title": title,
            "report_language": report_language,
            "subject": subject.to_dict(),
            "sections": [s.to_dict() for s in sections],
            "tables": [t.to_dict() for t in tables],
            "figures": [f.to_dict() for f in figures],
            "claims": [c.to_dict() for c in claims],
            "explanations": [e.to_dict() for e in explanations],
            "evidence_synthesis": [e.to_dict() for e in evidence_synthesis],
            "conclusions": [c.to_dict() for c in conclusions],
            "recommendations": [r.to_dict() for r in recommendations],
            "limitations": [l.to_dict() for l in limitations],
            "gaps": [g.to_dict() for g in gaps],
            # Inside the hash: a report whose citation resolved to a different
            # source than the one it was written against is a different report,
            # and a hash that could not see the difference would certify it.
            "references": [r.to_dict() for r in references],
        }
        return cls(
            id=new_id(REPORT),
            report_build_id=report_build_id,
            session_id=session_id,
            analysis_id=analysis_id,
            title=title,
            # Derived, never passed in: whether the document is partial is a
            # fact about its gaps, not a status a caller may assert.
            status=BuildStage.COMPLETED_WITH_GAPS if gaps else BuildStage.COMPLETED,
            subject=subject,
            sections=sections,
            tables=tables,
            figures=figures,
            claims=claims,
            explanations=explanations,
            evidence_synthesis=evidence_synthesis,
            conclusions=conclusions,
            recommendations=recommendations,
            limitations=limitations,
            gaps=gaps,
            references=references,
            provenance=dict(provenance or {}),
            # Renderings are produced *from* the artifact and are therefore
            # outside its content hash — adding a PDF later must not change
            # what the report says it is.
            renderings=renderings,
            content_sha256=content_sha256(body),
            created_at=now,
            report_language=report_language,
            supersedes_report_id=supersedes_report_id,
            version=version,
        )

    def with_renderings(self, renderings: tuple[ReportRendering, ...]) -> "ReportArtifact":
        """A copy carrying its produced files. ``content_sha256`` is unchanged
        by construction, which is the point: the same report may gain a PDF."""
        from dataclasses import replace

        return replace(self, renderings=renderings)

    @property
    def figure_by_id(self) -> dict[str, ReportFigure]:
        return {f.figure_id: f for f in self.figures}

    @property
    def section_by_id(self) -> dict[str, ReportSection]:
        return {s.section_id: s for s in self.sections}

    @property
    def reference_by_evidence_id(self) -> dict[str, ReportReference]:
        return {r.evidence_id: r for r in self.references}

    @property
    def cited_evidence_ids(self) -> frozenset[str]:
        from_claims = {e for c in self.claims for e in getattr(c, "citation_ids", ())}
        from_synthesis = {e for s in self.evidence_synthesis for e in s.evidence_ids}
        return frozenset(from_claims | from_synthesis)

    @property
    def cited_observation_ids(self) -> frozenset[str]:
        return frozenset(
            c.observation_id for c in self.claims if getattr(c, "observation_id", None)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "report_id": self.id,
            "report_build_id": self.report_build_id,
            "session_id": self.session_id,
            "analysis_id": self.analysis_id,
            "title": self.title,
            "status": self.status.value,
            "report_language": self.report_language,
            "version": self.version,
            "supersedes_report_id": self.supersedes_report_id,
            "subject": self.subject.to_dict(),
            "sections": [s.to_dict() for s in self.sections],
            "tables": [t.to_dict() for t in self.tables],
            "figures": [f.to_dict() for f in self.figures],
            "claims": [c.to_dict() for c in self.claims],
            "explanations": [e.to_dict() for e in self.explanations],
            "evidence_synthesis": [e.to_dict() for e in self.evidence_synthesis],
            "conclusions": [c.to_dict() for c in self.conclusions],
            "recommendations": [r.to_dict() for r in self.recommendations],
            "limitations": [l.to_dict() for l in self.limitations],
            "gaps": [g.to_dict() for g in self.gaps],
            "references": [r.to_dict() for r in self.references],
            "evidence_summary": [e.to_dict() for e in self.evidence_synthesis],
            "provenance": dict(self.provenance),
            "renderings": [r.to_dict() for r in self.renderings],
            "content_sha256": self.content_sha256,
            "created_at": self.created_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class ReportBuildRequest:
    """Spec section 5.1. Validated at admission, then frozen into the build.

    The request is stored rather than re-derived because "which endpoints did
    the user actually ask for" is not recoverable from the finished report: a
    report with two endpoints could be a two-endpoint request that succeeded or
    a three-endpoint request that lost one, and those are different documents.
    """

    session_id: str
    analysis_id: str
    selected_endpoints: tuple[str, ...]
    selected_tox21_tasks: tuple[str, ...] = ()
    report_language: str = "en"
    audience: str = "technical_r_and_d"
    include_explanations: bool = True
    include_external_evidence: bool = True
    output_formats: tuple[str, ...] = ("markdown", "html")
    schema_version: str = BUILD_REQUEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_id(self.session_id, SESSION, field="report_build_request.session_id")
        require_id(self.analysis_id, ANALYSIS, field="report_build_request.analysis_id")
        if self.report_language not in SUPPORTED_REPORT_LANGUAGES:
            raise ValueError(
                f"report_language {self.report_language!r} is not served; this version is "
                f"English-first ({sorted(SUPPORTED_REPORT_LANGUAGES)})"
            )
        if not self.selected_endpoints:
            raise ValueError("a report must name at least one endpoint")
        if len(set(self.selected_endpoints)) != len(self.selected_endpoints):
            raise ValueError("selected_endpoints contains a duplicate")
        if self.selected_tox21_tasks and "tox21" not in self.selected_endpoints:
            raise ValueError("selected_tox21_tasks named without selecting the tox21 endpoint")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "analysis_id": self.analysis_id,
            "report_language": self.report_language,
            "audience": self.audience,
            "selected_endpoints": list(self.selected_endpoints),
            "selected_tox21_tasks": list(self.selected_tox21_tasks),
            "include_explanations": self.include_explanations,
            "include_external_evidence": self.include_external_evidence,
            "output_formats": list(self.output_formats),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReportBuildRequest":
        return cls(
            session_id=payload["session_id"],
            analysis_id=payload["analysis_id"],
            selected_endpoints=tuple(payload.get("selected_endpoints", ())),
            selected_tox21_tasks=tuple(payload.get("selected_tox21_tasks", ())),
            report_language=payload.get("report_language", "en"),
            audience=payload.get("audience", "technical_r_and_d"),
            include_explanations=bool(payload.get("include_explanations", True)),
            include_external_evidence=bool(payload.get("include_external_evidence", True)),
            output_formats=tuple(payload.get("output_formats", ("markdown", "html"))),
        )


class BuildTransitionError(ValueError):
    """An illegal stage move. Named so a caller can distinguish "this build
    cannot go there" from "this build is malformed"."""


@dataclass(frozen=True, slots=True)
class ReportBuild:
    """The resumable aggregate (spec section 10).

    Frozen, and every transition returns a new value, so a stage change is a
    thing that gets written rather than a field that gets mutated somewhere and
    maybe persisted. ``correction_attempts`` lives here rather than in the
    validator because the cap is a property of the build, and a validator that
    counted its own retries would reset on every restart.
    """

    id: str
    session_id: str
    run_id: str
    analysis_id: str
    request: ReportBuildRequest
    stage: BuildStage
    created_at: datetime
    updated_at: datetime
    deadline_at: datetime | None = None
    report_id: str | None = None
    correction_attempts: int = 0
    failure_code: str | None = None
    failure_detail: str | None = None
    #: Stage outputs already produced, so a resumed build skips the expensive
    #: work it already paid for: explanation ids by "endpoint/task", the
    #: substance profile, the accepted evidence it read.
    stage_state: dict[str, Any] = field(default_factory=dict)

    #: Spec section 10 stage 4: one correction attempt, then stop.
    MAX_CORRECTION_ATTEMPTS = 1

    def __post_init__(self) -> None:
        require_id(self.id, REPORT_BUILD, field="report_build.id")
        require_id(self.session_id, SESSION, field="report_build.session_id")
        require_id(self.run_id, RUN, field="report_build.run_id")
        require_id(self.analysis_id, ANALYSIS, field="report_build.analysis_id")
        if self.report_id is not None:
            require_id(self.report_id, REPORT, field="report_build.report_id")

    @classmethod
    def start(
        cls,
        *,
        session_id: str,
        run_id: str,
        request: ReportBuildRequest,
        now: datetime,
        deadline_at: datetime | None = None,
    ) -> "ReportBuild":
        return cls(
            id=new_id(REPORT_BUILD),
            session_id=session_id,
            run_id=run_id,
            analysis_id=request.analysis_id,
            request=request,
            stage=BuildStage.QUEUED,
            created_at=now,
            updated_at=now,
            deadline_at=deadline_at,
        )

    @property
    def is_terminal(self) -> bool:
        return self.stage in TERMINAL_STAGES

    @property
    def corrections_exhausted(self) -> bool:
        return self.correction_attempts >= self.MAX_CORRECTION_ATTEMPTS

    def advance(self, stage: BuildStage, *, now: datetime, **changes: Any) -> "ReportBuild":
        from dataclasses import replace

        allowed = ALLOWED_STAGE_TRANSITIONS[self.stage]
        if stage not in allowed:
            raise BuildTransitionError(
                f"a report build cannot move from {self.stage.value} to {stage.value}"
            )
        if stage is BuildStage.SYNTHESIZING and self.stage is BuildStage.VALIDATING:
            # The correction loop. Counted here so the cap survives a restart
            # and a second failed draft cannot buy a third attempt.
            if self.corrections_exhausted:
                raise BuildTransitionError(
                    "this build has already used its one correction attempt; an invalid "
                    "draft is never accepted, so the build fails instead"
                )
            changes.setdefault("correction_attempts", self.correction_attempts + 1)
        return replace(self, stage=stage, updated_at=now, **changes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_build_id": self.id,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "analysis_id": self.analysis_id,
            "request": self.request.to_dict(),
            "stage": self.stage.value,
            "report_id": self.report_id,
            "correction_attempts": self.correction_attempts,
            "failure_code": self.failure_code,
            "failure_detail": self.failure_detail,
            "deadline_at": self.deadline_at.isoformat() if self.deadline_at else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
