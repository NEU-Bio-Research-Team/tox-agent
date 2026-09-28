"""The parts a report is assembled from: figures, explanations, substance and evidence, references, tables, sections, gaps, conclusion, recommendations, renderings."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..ids import (
    ATTACHMENT,
    EVIDENCE,
    EXPLANATION,
    FIGURE,
    OBSERVATION,
    require_id,
)
from .vocabulary import (
    ALLOWED_FIGURE_MEDIA_TYPES,
    EvidenceRelation,
    ExplanationStatus,
    GapReason,
    SourceClass,
    is_safe_citation_url,
)


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
