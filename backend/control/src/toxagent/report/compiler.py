"""Draft -> ``ReportArtifact`` (spec section 10, stage 5).

The compiler is where a validated draft stops being a model's description of a
report and becomes the report. Three things happen here and nowhere else:

**Values are re-rendered from observations.** A draft's ``rendered_value``
passed validation by *matching* the canonical field; the compiled claim's value
is read from the observation again and formatted by ``AnswerCompiler``'s
formatter. The draft therefore cannot be the source of a number even by being
right about it — a distinction that matters the day a transform is fixed and
every stored report has to re-render consistently.

**References are resolved.** Figure ids, explanation ids and evidence ids
become the actual figures, packages and links carried by the artifact, so a
renderer never has to look anything up and can never fail to. For evidence this
means a *snapshot*: title, authors, provider, identifier, canonical URL and
retrieval date, numbered by first appearance and frozen into the content hash
(REP-01). The evidence table is session-scoped and mutable — records get
superseded, rejected, retention-expired — and a citable document may not depend
on it still saying the same thing next month. It also means opening a report
performs no fetch, so a reader is never reported on by the report.

**Status is derived.** ``completed`` versus ``completed_with_gaps`` is computed
from whether there are gaps, not asserted by anyone.
"""
from __future__ import annotations

from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from ..domain.answer import Claim, ClaimKind, Limitation, LimitationCode
from ..domain.evidence import EvidenceRecord, EvidenceStatus
from ..domain.observation import Observation
from ..domain.report import (
    EvidenceRelation,
    EvidenceSynthesis,
    ExplanationPackage,
    GapReason,
    ReportArtifact,
    ReportConclusion,
    ReportFigure,
    ReportGap,
    ReportRecommendation,
    ReportReference,
    ReportSection,
    ReportTable,
    SourceClass,
    SubstanceProfile,
)
from ..domain.ids import SYNTHESIS, new_id
from ..validation.citations import CITATION_TOKEN
from ..validation.limitations import text_for
from ..validation.report_wire import ReportDraftCandidate

#: Bumped when compiled output for the same draft would change.
COMPILER_VERSION = "toxagent-report-compiler-v2"


def _format(value: float, transform: str) -> str:
    """The same formatter ``answer/compiler.py`` uses. Duplicated deliberately
    rather than imported through a chain of report -> answer modules: the two
    are the same rule today, and a report's rendering must not silently change
    because a conversational-answer transform was tuned."""
    if transform == "identity":
        return str(value)
    if transform.startswith(("round:", "percent:")):
        digits = int(transform.split(":", 1)[1])
        scaled = value * 100 if transform.startswith("percent:") else value
        quantum = Decimal(1).scaleb(-digits)
        text = format(Decimal(str(scaled)).quantize(quantum, rounding=ROUND_HALF_UP), f".{digits}f")
        return text + ("%" if transform.startswith("percent:") else "")
    return str(value)


def citation_order(
    draft: ReportDraftCandidate, sections: Iterable[Any] | None = None
) -> list[str]:
    """Every cited evidence id, in the order a reader first meets it.

    Report order, not claim order: the number a reader sees next to a sentence
    has to ascend down the page. So sections are walked in order, and within a
    section the inline ``[@evd_...]`` tokens come first (they sit in specific
    sentences), then the section's claims' citations, then the evidence
    synthesis. Anything cited but never placed in a section is appended at the
    end rather than dropped — an unplaced citation is still a citation, and
    losing it here would make the artifact fail its own "everything cited is
    snapshotted" check for a reason nobody could see.
    """
    claims_by_id = {c.claim_id: c for c in draft.claims}
    order: list[str] = []

    def add(evidence_id: str) -> None:
        if evidence_id not in order:
            order.append(evidence_id)

    for section in draft.sections:
        # ``finditer`` rather than the set helper: within one section the
        # numbers must ascend in the order the sentences appear, and a set has
        # no order to preserve.
        for match in CITATION_TOKEN.finditer(section.body_markdown or ""):
            add(match.group(1))
        for claim_id in section.claim_ids:
            claim = claims_by_id.get(claim_id)
            if claim is not None:
                for evidence_id in claim.citation_ids:
                    add(evidence_id)
        if section.section_id == "external_evidence":
            for item in draft.evidence_synthesis:
                for evidence_id in item.evidence_ids:
                    add(evidence_id)
    for item in draft.evidence_synthesis:
        for evidence_id in item.evidence_ids:
            add(evidence_id)
    for claim in draft.claims:
        for evidence_id in claim.citation_ids:
            add(evidence_id)
    return order


def compile_references(
    draft: ReportDraftCandidate, evidence_by_id: Mapping[str, EvidenceRecord]
) -> tuple[ReportReference, ...]:
    """Freeze each cited source into the artifact, numbered by first appearance.

    A citation whose record does not resolve — or resolves to something no
    longer citable — still produces a reference, carrying
    ``unresolved_reason``. The validator refuses such a draft before this runs,
    so in practice this path is reached only for a record that changed status
    between validation and compilation; making it visible rather than absent is
    the difference between a reader seeing a broken citation and a reader seeing
    an unsupported sentence.
    """
    references: list[ReportReference] = []
    for number, evidence_id in enumerate(citation_order(draft), start=1):
        record = evidence_by_id.get(evidence_id)
        if record is None:
            references.append(
                ReportReference(
                    evidence_id=evidence_id,
                    number=number,
                    title="(source could not be resolved)",
                    provider="unknown",
                    unresolved_reason="no evidence record with this id exists in the session",
                )
            )
            continue
        references.append(
            ReportReference(
                evidence_id=record.id,
                number=number,
                title=record.title,
                provider=record.provider,
                canonical_url=record.canonical_url,
                # Capped: the point of the snapshot is that a reader can
                # identify the source, and a 400-author consortium paper is
                # identified by its first few names plus its identifier.
                authors=tuple(record.authors[:12]),
                published_at=record.published_at.isoformat() if record.published_at else None,
                identifier={
                    k: v for k, v in record.identifier.to_dict().items() if v
                },
                source_type=record.source_type.value,
                source_quality_tier=record.source_quality_tier.value,
                retrieved_at=record.retrieved_at.isoformat(),
                unresolved_reason=None
                if record.status is EvidenceStatus.ACCEPTED
                else f"the record is {record.status.value}, not accepted",
            )
        )
    return tuple(references)


def compile_report(
    draft: ReportDraftCandidate,
    *,
    session_id: str,
    analysis_id: str,
    report_build_id: str,
    subject: SubstanceProfile,
    observations_by_id: Mapping[str, Observation],
    explanations_by_id: Mapping[str, ExplanationPackage],
    figures_by_id: Mapping[str, ReportFigure],
    evidence_by_id: Mapping[str, EvidenceRecord] = MappingProxyType({}),
    provenance: dict[str, Any],
    report_language: str = "en",
    supersedes_report_id: str | None = None,
    version: int = 1,
    now: datetime,
) -> ReportArtifact:
    claims = tuple(
        _compile_claim(candidate, observations_by_id) for candidate in draft.claims
    )
    packages = tuple(
        explanations_by_id[ref.explanation_id]
        for ref in draft.explanations
        if ref.explanation_id in explanations_by_id
    )
    # Every figure the report actually shows, in section order, deduplicated.
    figure_ids: list[str] = []
    for section in draft.sections:
        for figure_id in section.figure_ids:
            if figure_id not in figure_ids:
                figure_ids.append(figure_id)
    for package in packages:
        if package.figure and package.figure.figure_id not in figure_ids:
            figure_ids.append(package.figure.figure_id)
    # The neutral structure drawing. Carried by the subject rather than named in
    # a section's ``figure_ids`` — a draft is written before it exists — so it
    # has to be collected explicitly or the artifact would reference a figure id
    # from the profile that its own ``figures`` list could not resolve (REP-02).
    if subject.structure_figure_id and subject.structure_figure_id not in figure_ids:
        figure_ids.append(subject.structure_figure_id)
    figures = tuple(
        figures_by_id[figure_id] for figure_id in figure_ids if figure_id in figures_by_id
    )

    sections = tuple(
        ReportSection(
            section_id=section.section_id,
            heading=section.heading,
            body_markdown=section.body_markdown,
            claim_ids=tuple(section.claim_ids),
            table_ids=tuple(section.table_ids),
            figure_ids=tuple(section.figure_ids),
            gap_ids=tuple(section.gap_ids),
            source_classes=tuple(SourceClass(c) for c in section.source_classes),
        )
        for section in draft.sections
    )
    tables = tuple(
        ReportTable(
            table_id=table.table_id,
            title=table.title,
            columns=tuple(table.columns),
            rows=tuple(tuple(row) for row in table.rows),
            source_class=SourceClass(table.source_class),
            row_claim_ids=tuple(tuple(ids) for ids in table.row_claim_ids),
        )
        for table in draft.tables
    )
    synthesis = tuple(
        EvidenceSynthesis(
            synthesis_id=new_id(SYNTHESIS),
            proposition=item.proposition,
            relation=EvidenceRelation(item.relation),
            evidence_ids=tuple(item.evidence_ids),
            endpoint=item.endpoint,
            assay=item.assay,
            organism=item.organism,
            dose_context=item.dose_context,
            quality_notes=tuple(item.quality_notes),
            conflict_id=item.conflict_id,
        )
        for item in draft.evidence_synthesis
    )
    conclusions = tuple(
        ReportConclusion(
            conclusion_id=item.conclusion_id,
            text=item.text,
            basis_claim_ids=tuple(item.basis_claim_ids),
            endpoint=item.endpoint,
            task=item.task,
            is_integrated=item.is_integrated,
        )
        for item in draft.conclusions
    )
    recommendations = tuple(
        ReportRecommendation(
            recommendation_id=item.recommendation_id,
            text=item.text,
            basis_claim_ids=tuple(item.basis_claim_ids),
            action_category=item.action_category,
            priority=item.priority,
            rationale=item.rationale,
            conditions=item.conditions,
        )
        for item in draft.recommendations
    )
    limitations = tuple(
        Limitation(
            code=LimitationCode(item.code),
            # Server-owned wording when the draft left it blank, so a
            # limitation's text cannot be quietly weakened by paraphrase.
            text=item.text or text_for(LimitationCode(item.code), report_language),
        )
        for item in draft.limitations
    )
    gaps = tuple(
        ReportGap(
            gap_id=item.gap_id,
            reason=GapReason(item.reason),
            detail=item.detail,
            section_id=item.section_id,
            endpoint=item.endpoint,
            task=item.task,
        )
        for item in draft.gaps
    )

    return ReportArtifact.create(
        report_build_id=report_build_id,
        session_id=session_id,
        analysis_id=analysis_id,
        title=draft.title,
        subject=subject,
        sections=sections,
        tables=tables,
        figures=figures,
        claims=claims,
        explanations=packages,
        evidence_synthesis=synthesis,
        conclusions=conclusions,
        recommendations=recommendations,
        limitations=limitations,
        gaps=gaps,
        references=compile_references(draft, evidence_by_id),
        provenance={**provenance, "compiler_version": COMPILER_VERSION},
        report_language=report_language,
        supersedes_report_id=supersedes_report_id,
        version=version,
        now=now,
    )


def _compile_claim(candidate, observations_by_id: Mapping[str, Observation]) -> Claim:
    """Re-read the value from the observation rather than trusting the draft.

    For a derived claim there is no single observation to read, so the draft's
    already-validated arithmetic stands — ``validate_derived_numeric`` has
    checked it against the two input claims' canonical values.
    """
    source_value = candidate.source_value
    rendered_value = candidate.rendered_value
    observation = (
        observations_by_id.get(candidate.observation_id) if candidate.observation_id else None
    )
    if (
        candidate.kind in {"numeric", "classification"}
        and observation is not None
        and candidate.field_path
        and observation.has(candidate.field_path)
    ):
        source_value = observation.value_at(candidate.field_path)
        if candidate.kind == "numeric" and isinstance(source_value, (int, float)):
            rendered_value = _format(float(source_value), candidate.transform)
    return Claim(
        claim_id=candidate.claim_id,
        kind=ClaimKind(candidate.kind),
        text=candidate.text,
        observation_id=candidate.observation_id,
        field_path=candidate.field_path,
        source_value=source_value,
        rendered_value=rendered_value,
        transform=candidate.transform,
        citation_ids=tuple(candidate.citation_ids),
        input_claim_ids=tuple(candidate.input_claim_ids),
    )


def claim_links(artifact: ReportArtifact) -> list[dict[str, Any]]:
    """Rows for ``report_claim_links``: which claim sits in which section, and
    what observation it points at. Written alongside the artifact so an audit
    can go from an observation back to every report that used it."""
    section_of: dict[str, str] = {}
    class_of: dict[str, str] = {}
    for section in artifact.sections:
        for claim_id in section.claim_ids:
            section_of.setdefault(claim_id, section.section_id)
            class_of.setdefault(
                claim_id,
                section.source_classes[0].value if section.source_classes
                else SourceClass.AGENT_SYNTHESIS.value,
            )
    return [
        {
            "claim_id": claim.claim_id,
            "section_id": section_of.get(claim.claim_id, ""),
            "kind": claim.kind.value,
            "source_class": class_of.get(claim.claim_id, SourceClass.AGENT_SYNTHESIS.value),
            "observation_id": claim.observation_id,
            "field_path": claim.field_path,
        }
        for claim in artifact.claims
        if claim.claim_id in section_of
    ]


def evidence_links(artifact: ReportArtifact) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for synthesis in artifact.evidence_synthesis:
        for evidence_id in synthesis.evidence_ids:
            if evidence_id in seen:
                continue
            seen.add(evidence_id)
            rows.append(
                {
                    "evidence_id": evidence_id,
                    "relation": synthesis.relation.value,
                    "section_id": "external_evidence",
                }
            )
    for claim in artifact.claims:
        for evidence_id in claim.citation_ids:
            if evidence_id in seen:
                continue
            seen.add(evidence_id)
            rows.append(
                {
                    "evidence_id": evidence_id,
                    "relation": EvidenceRelation.SUPPORTS.value,
                    "section_id": "references",
                }
            )
    return rows
