"""A compiled v3 report, as the artifact every reader already understands (PR-12).

PR-11 produced a ``CompiledReport``: the model's prose with every value
substituted from the fact bundle, plus the sections the compiler owns outright.
Nothing could read it. The API, the renderers, the persistence layer and the
browser all speak ``ReportArtifact``, and teaching each of them a second shape
would be four places for the two to drift apart.

So this module is the one translation, and it is deliberately lossless in one
direction only: every sentence, gap, limitation and reference in the compiled
report appears in the artifact, and nothing appears in the artifact that the
compiler did not produce. A field the v2 artifact has and v3 does not fill —
``claims``, ``tables`` — is empty rather than reconstructed, because a claim row
invented here would be a second source for a fact the bundle already owns.

``content_sha256`` is the compiled report's own digest, not a re-hash of the
artifact body. The report is the thing the gates passed; a hash computed over a
translation of it would certify the translation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Sequence

from ..domain.ids import REPORT, new_id
from ..domain.report import (
    BuildStage,
    EvidenceRelation,
    EvidenceSynthesis,
    ExplanationPackage,
    GapReason,
    ReportArtifact,
    ReportConclusion,
    ReportGap,
    ReportRecommendation,
    ReportReference,
    ReportSection,
    SourceClass,
    SubstanceProfile,
)
from .synthesis_compiler import CompiledReport

SCHEMA_VERSION_V3 = "toxagent-report-v3"

#: What kind of content each compiled section carries. A narrative section is
#: the model's synthesis however many facts it quotes: the values in it were
#: rendered by the server, the sentence around them was not.
_COMPILED_SOURCE_CLASS: Mapping[str, tuple[SourceClass, ...]] = {
    "predictor_results": (SourceClass.PREDICTOR_FACT,),
    "limitations": (),
    "references": (SourceClass.EXTERNAL_EVIDENCE,),
    "provenance_appendix": (),
}


@dataclass(frozen=True, slots=True)
class CompiledLimitation:
    """A limitation the compiler worded. Code and sentence from one place."""

    code: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "text": self.text}


def _gap_reason(value: str) -> GapReason | None:
    try:
        return GapReason(value)
    except ValueError:
        return None


def to_artifact(
    report: CompiledReport,
    *,
    session_id: str,
    subject: SubstanceProfile,
    evidence: Sequence[Mapping[str, Any]] = (),
    explanations: Sequence[ExplanationPackage] = (),
    synthesis_sha256: str = "",
    version: int = 1,
    now: datetime,
) -> ReportArtifact:
    """Translate. Raises ``ValueError`` if the result would not be a valid artifact.

    ``evidence`` is the bundle's promoted records, used only to fill reference
    metadata the compiler's reference list does not carry (provider, authors,
    dates). It never adds a reference: numbering comes from the compiled report.
    """
    gaps: list[ReportGap] = []
    unknown_gaps: list[str] = []
    for index, gap in enumerate(report.gaps, start=1):
        reason = _gap_reason(str(gap.get("reason") or ""))
        if reason is None:
            # A gap this schema cannot count is still a gap. Refusing to
            # publish is correct; dropping it would make a partial report look
            # whole.
            unknown_gaps.append(str(gap.get("reason")))
            continue
        gaps.append(
            ReportGap(
                gap_id=f"gap_{index}",
                reason=reason,
                detail=str(gap.get("detail") or ""),
                section_id=str(gap.get("section_id") or gap.get("stage") or "limitations"),
                endpoint=gap.get("endpoint"),
                task=gap.get("task"),
            )
        )
    if unknown_gaps:
        raise ValueError(f"compiled report carries gap reason(s) with no GapReason: {unknown_gaps}")

    gap_ids_by_section: dict[str, list[str]] = {}
    for gap in gaps:
        gap_ids_by_section.setdefault(gap.section_id, []).append(gap.gap_id)

    figure_ids_by_section: dict[str, list[str]] = {}
    for package in explanations:
        if package.figure is not None:
            figure_ids_by_section.setdefault("explanation_and_visuals", []).append(
                package.figure.figure_id
            )

    sections = tuple(
        ReportSection(
            section_id=section.section_id,
            heading=section.heading,
            body_markdown=section.body_markdown,
            figure_ids=tuple(figure_ids_by_section.get(section.section_id, ())),
            gap_ids=tuple(gap_ids_by_section.get(section.section_id, ())),
            source_classes=(
                _COMPILED_SOURCE_CLASS.get(section.section_id, ())
                if section.compiled
                else (SourceClass.AGENT_SYNTHESIS,)
            ),
        )
        for section in report.sections
    )

    records = {str(record.get("evidence_id")): record for record in evidence}
    references = tuple(
        ReportReference(
            evidence_id=str(item["evidence_id"]),
            number=int(item["number"]),
            title=str(item.get("title") or item["evidence_id"]),
            provider=str(records.get(str(item["evidence_id"]), {}).get("provider") or "unknown"),
            canonical_url=item.get("canonical_url"),
            authors=tuple(records.get(str(item["evidence_id"]), {}).get("authors") or ()),
            published_at=records.get(str(item["evidence_id"]), {}).get("published_at"),
            identifier=dict(records.get(str(item["evidence_id"]), {}).get("identifier") or {}),
            source_type=records.get(str(item["evidence_id"]), {}).get("source_type"),
            source_quality_tier=records.get(str(item["evidence_id"]), {}).get(
                "source_quality_tier"
            ),
            retrieved_at=records.get(str(item["evidence_id"]), {}).get("retrieved_at"),
        )
        for item in report.references
    )

    conclusions = tuple(
        ReportConclusion(
            conclusion_id=f"conclusion_{item['local_ref']}",
            text=str(item["text"]),
            # Fact ids, not claim ids: v3 has no claim rows. The field name is
            # the v2 reader's; what it holds resolves against the provenance
            # appendix's fact bundle instead.
            basis_claim_ids=tuple(item.get("basis_fact_ids") or ()),
            endpoint=item.get("endpoint"),
            task=item.get("task"),
            is_integrated=bool(item.get("is_integrated")),
        )
        for item in report.conclusions
    )
    recommendations = tuple(
        ReportRecommendation(
            recommendation_id=f"recommendation_{item['local_ref']}",
            text=str(item["text"]),
            basis_claim_ids=tuple(item.get("basis_fact_ids") or ()),
            action_category=str(item.get("action_category") or ""),
            priority=str(item.get("priority") or ""),
            rationale=str(item.get("rationale") or ""),
            conditions=str(item.get("conditions") or ""),
        )
        for item in report.recommendations
    )
    evidence_synthesis = tuple(
        EvidenceSynthesis(
            synthesis_id=f"synthesis_{index}",
            proposition=str(item.get("summary") or ""),
            relation=EvidenceRelation(str(item.get("relation"))),
            evidence_ids=(str(item["evidence_id"]),) if item.get("evidence_id") else (),
            endpoint=item.get("endpoint"),
            assay=item.get("task"),
        )
        for index, item in enumerate(report.evidence_interpretations, start=1)
    )

    return ReportArtifact(
        id=new_id(REPORT),
        report_build_id=report.report_build_id,
        session_id=session_id,
        analysis_id=report.analysis_id,
        title=report.title,
        # Derived, as ReportArtifact.create derives it: partial is a fact about
        # the gaps, not a status the caller asserts.
        status=BuildStage.COMPLETED_WITH_GAPS if gaps else BuildStage.COMPLETED,
        subject=subject,
        sections=sections,
        tables=(),
        figures=tuple(p.figure for p in explanations if p.figure is not None),
        claims=(),
        explanations=tuple(explanations),
        evidence_synthesis=evidence_synthesis,
        conclusions=conclusions,
        recommendations=recommendations,
        limitations=tuple(
            CompiledLimitation(code=str(item["code"]), text=str(item.get("text") or ""))
            for item in report.limitations
        ),
        gaps=tuple(gaps),
        references=references,
        provenance={
            **dict(report.provenance),
            "schema_version": SCHEMA_VERSION_V3,
            "synthesis_sha256": synthesis_sha256,
        },
        renderings=(),
        content_sha256=report.content_sha256(),
        created_at=now,
        report_language=report.language,
        schema_version=SCHEMA_VERSION_V3,
        version=version,
    )
