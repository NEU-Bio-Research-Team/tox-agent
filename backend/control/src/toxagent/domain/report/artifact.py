"""``ReportArtifact``: the report, as every reader (API, renderers, persistence, browser) understands it."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..ids import (
    ANALYSIS,
    REPORT,
    REPORT_BUILD,
    SESSION,
    new_id,
    require_id,
)
from ..provenance import content_sha256
from .build import BuildStage
from .parts import (
    EvidenceSynthesis,
    ExplanationPackage,
    ReportConclusion,
    ReportFigure,
    ReportGap,
    ReportRecommendation,
    ReportReference,
    ReportRendering,
    ReportSection,
    ReportTable,
    SubstanceProfile,
)
from .vocabulary import REQUIRED_SECTION_IDS, SCHEMA_VERSION


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
