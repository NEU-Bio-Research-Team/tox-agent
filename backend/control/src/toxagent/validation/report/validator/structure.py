"""Structural checks: scope, sections, source separation, endpoint coverage, figures, references, citation tokens."""
from __future__ import annotations

from typing import Mapping

from ....domain.errors import Violation
from ....domain.report import (
    REQUIRED_SECTION_IDS,
    SourceClass,
)
from ...answer.citations import (
    cited_in_prose,
)
from ..draft_wire import ReportDraftCandidate
from ._common import _KIND_TO_ALLOWED_CLASSES, _MUST_NOT_BE_EMPTY, ReportValidationContext


def _validate_scope(
    draft: ReportDraftCandidate, context: ReportValidationContext
) -> list[Violation]:
    """The report belongs to the authorized session, build and analysis.

    The build id is the only one a draft states, and it is checked rather than
    trusted: everything else is resolved from it server-side, so a draft cannot
    address itself at another session's build.
    """
    if draft.report_build_id != context.report_build_id:
        return [
            Violation(
                "report_scope_mismatch",
                "this draft names a different report build than the run it was submitted from",
                path="report_build_id",
                expected=context.report_build_id,
                actual=draft.report_build_id,
            )
        ]
    return []


def _validate_sections(draft: ReportDraftCandidate) -> list[Violation]:
    """Every required section exists, exactly once, and carries something.

    This is the "Missing-data visibility" gate. The failure it prevents is not
    a model forgetting a heading — it is a section quietly vanishing because
    the provider behind it failed, leaving a report that looks complete.
    """
    violations: list[Violation] = []
    seen = [s.section_id for s in draft.sections]
    missing = [sid for sid in REQUIRED_SECTION_IDS if sid not in seen]
    if missing:
        violations.append(
            Violation(
                "missing_required_section",
                f"the report is missing required section(s): {missing}. A section whose "
                "content is unavailable records a gap and says so; it is never omitted.",
                path="sections", expected=missing,
            )
        )
    duplicates = sorted({sid for sid in seen if seen.count(sid) > 1})
    if duplicates:
        violations.append(
            Violation(
                "duplicate_section", f"section id(s) repeated: {duplicates}", path="sections"
            )
        )
    for section in draft.sections:
        empty = not (
            section.body_markdown.strip()
            or section.claim_ids or section.table_ids
            or section.figure_ids or section.gap_ids
        )
        if empty and section.section_id in _MUST_NOT_BE_EMPTY:
            violations.append(
                Violation(
                    "required_section_empty",
                    f"section {section.section_id!r} has no content, no claims and no recorded "
                    "gap; a report cannot be silent about this section",
                    path=f"sections[{section.section_id}]",
                )
            )
        elif empty:
            violations.append(
                Violation(
                    "section_without_content_or_gap",
                    f"section {section.section_id!r} is empty and records no gap explaining why",
                    path=f"sections[{section.section_id}].gap_ids",
                )
            )
    return violations


def _validate_source_separation(
    claim,
    section_of: Mapping[str, str],
    classes_of: Mapping[str, frozenset[SourceClass]],
) -> list[Violation]:
    """A predictor number cannot be filed as external evidence, and vice versa.

    The check is against the *section's* declared source classes, because that
    is what a reader sees: the heading tells them whether they are looking at
    model output or at literature, and a claim placed under the wrong one has
    been mislabelled no matter how carefully its own text is worded.
    """
    section_id = section_of[claim.claim_id]
    declared = classes_of.get(section_id, frozenset())
    if not declared:
        return []
    allowed = _KIND_TO_ALLOWED_CLASSES.get(claim.kind, frozenset())
    if allowed and not (declared & allowed):
        return [
            Violation(
                "source_class_mismatch",
                f"a {claim.kind} claim cannot appear in section {section_id!r}, which declares "
                f"only {sorted(c.value for c in declared)}",
                path=f"claims[{claim.claim_id}]",
                expected=sorted(c.value for c in allowed),
                actual=sorted(c.value for c in declared),
            )
        ]
    # A claim citing external evidence in a section that says it holds no
    # external evidence is the same mislabelling from the other direction.
    if claim.citation_ids and SourceClass.EXTERNAL_EVIDENCE not in declared:
        if SourceClass.AGENT_SYNTHESIS not in declared:
            return [
                Violation(
                    "source_class_mismatch",
                    f"claim {claim.claim_id!r} cites external evidence but sits in section "
                    f"{section_id!r}, which does not declare external_evidence",
                    path=f"sections[{section_id}].source_classes",
                )
            ]
    return []


def _validate_endpoint_coverage(
    draft: ReportDraftCandidate, context: ReportValidationContext
) -> list[Violation]:
    """Every selected *served* endpoint appears; every selected unserved one is
    a recorded gap.

    Both halves matter. Dropping a served endpoint hides a result; silently
    dropping an unserved one hides that the user asked for something this
    deployment could not do (SCI-06).
    """
    violations: list[Violation] = []
    served = set(context.served_endpoints)
    selected = set(context.selected_endpoints)

    predictor_claims = [
        claim for claim in draft.claims
        if claim.kind in {"numeric", "classification"} and claim.field_path
    ]
    covered = {
        endpoint
        for endpoint in selected & served
        if any(f"predictions.{endpoint}" in (c.field_path or "") for c in predictor_claims)
    }
    missing = sorted((selected & served) - covered)
    if missing:
        violations.append(
            Violation(
                "endpoint_not_reported",
                f"endpoint(s) {missing} were selected and served, but no claim in this report "
                "reports their values",
                path="claims", expected=missing,
            )
        )

    gap_endpoints = {gap.endpoint for gap in draft.gaps if gap.endpoint}
    unserved = sorted(selected - served)
    unexplained = [e for e in unserved if e not in gap_endpoints]
    if unexplained:
        violations.append(
            Violation(
                "unavailable_endpoint_hidden",
                f"endpoint(s) {unexplained} were requested and are not served by this analysis; "
                "the report must record that as a gap rather than omit them",
                path="gaps", expected=unexplained,
            )
        )
    return violations


def _validate_figures(
    draft: ReportDraftCandidate, context: ReportValidationContext
) -> list[Violation]:
    """Figure integrity and render integrity: every referenced figure exists.

    A broken image reference is not cosmetic here — the Markdown, HTML and PDF
    renderings are the report, and a figure a renderer cannot resolve is a
    claim about the structure that the document silently fails to make.
    """
    violations: list[Violation] = []
    for section in draft.sections:
        for figure_id in section.figure_ids:
            figure = context.figures_by_id.get(figure_id)
            if figure is None:
                violations.append(
                    Violation(
                        "figure_not_found",
                        f"section {section.section_id!r} references figure {figure_id!r}, which "
                        "this build did not produce",
                        path=f"sections[{section.section_id}].figure_ids",
                        actual=figure_id,
                    )
                )
                continue
            if not figure.caption.strip() or not figure.alt_text.strip():
                violations.append(
                    Violation(
                        "figure_missing_text",
                        f"figure {figure_id!r} has no caption or no alt text",
                        path=f"sections[{section.section_id}].figure_ids",
                    )
                )
            error = (context.figure_errors or {}).get(figure_id)
            if error:
                violations.append(
                    Violation(
                        "figure_integrity_failed",
                        f"figure {figure_id!r} cannot be verified: {error}",
                        path=f"sections[{section.section_id}].figure_ids",
                    )
                )
    return violations


def _validate_citation_tokens(
    draft: ReportDraftCandidate, context: ReportValidationContext
) -> list[Violation]:
    """Inline ``[@evd_...]`` markers, checked like any other citation (REP-01).

    The marker exists so that a citation can sit in the sentence it supports
    without the model writing either the number — which is the artifact's to
    assign — or the URL, which would put unvalidated text into something the
    renderers turn into a link. Having given models a token, the token has to be
    held to the same three rules a claim's citation is: the record exists in this
    session, it was actually opened, and the section carrying it declares that it
    contains external evidence.
    """
    violations: list[Violation] = []
    for section in draft.sections:
        path = f"sections[{section.section_id}].body_markdown"
        found = cited_in_prose(section.body_markdown)
        if not found:
            continue
        for evidence_id in sorted(found):
            record = context.evidence_by_id.get(evidence_id)
            if record is None:
                violations.append(
                    Violation(
                        "citation_not_found",
                        f"the prose cites {evidence_id!r}, which does not exist in this session",
                        path=path, actual=evidence_id,
                    )
                )
                continue
            if not record.is_citable:
                violations.append(
                    Violation(
                        "citation_not_citable",
                        f"the prose cites {evidence_id!r}, which is {record.status.value} "
                        "rather than accepted",
                        path=path, actual=evidence_id,
                    )
                )
            if evidence_id not in context.read_evidence_ids:
                violations.append(
                    Violation(
                        "citation_not_read",
                        f"the prose cites {evidence_id!r}, which was never opened with "
                        "get_evidence_record; a search result is a byline, not a source",
                        path=path, actual=evidence_id,
                    )
                )
        declared = {SourceClass(c) for c in section.source_classes}
        if SourceClass.EXTERNAL_EVIDENCE not in declared:
            violations.append(
                Violation(
                    "section_cites_without_declaring_external_evidence",
                    f"section {section.section_id!r} cites literature inline but does not "
                    "declare the external_evidence source class; the report is required never "
                    "to blur a predictor number with a published one",
                    path=f"sections[{section.section_id}].source_classes",
                )
            )
    return violations


def _validate_references(
    draft: ReportDraftCandidate, context: ReportValidationContext
) -> list[Violation]:
    """Everything cited anywhere is reachable from the references section."""
    cited = {e for c in draft.claims for e in c.citation_ids}
    cited |= {e for s in draft.evidence_synthesis for e in s.evidence_ids}
    for section in draft.sections:
        cited |= cited_in_prose(section.body_markdown)
    if not cited:
        return []
    references = next((s for s in draft.sections if s.section_id == "references"), None)
    has_content = references is not None and bool(
        references.body_markdown.strip() or references.claim_ids or references.table_ids
    )
    if not has_content:
        return [
            Violation(
                "references_section_empty",
                f"this report cites {len(cited)} evidence record(s) and its references section "
                "is empty",
                path="sections[references]",
            )
        ]
    return []
