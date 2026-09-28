"""Content safety (raw HTML, remote images, guarantees) and semantic consistency."""
from __future__ import annotations

from ....domain.errors import Violation
from ..draft_wire import ReportDraftCandidate
from ..semantics import (
    EvidenceSituation,
    check_evidence_scope_consistency,
    check_evidence_state_gap,
    check_explanation_consistency,
)
from ._common import _HTML_EVENT_ATTR, _RAW_HTML, _REMOTE_IMAGE, ReportValidationContext


def _validate_content_safety(draft: ReportDraftCandidate) -> list[Violation]:
    """No raw HTML, no event handlers, no remote images anywhere in the prose.

    External evidence text is untrusted data. It reaches a draft through a
    model that read it, and the HTML and PDF renderings inline whatever the
    draft says — so this is the boundary at which quoted provider content stops
    being able to act.
    """
    violations: list[Violation] = []
    surfaces: list[tuple[str, str]] = [
        (f"sections[{s.section_id}].body_markdown", s.body_markdown) for s in draft.sections
    ]
    surfaces += [
        (f"conclusions[{i}].text", c.text) for i, c in enumerate(draft.conclusions)
    ]
    surfaces += [
        (f"recommendations[{i}].text", r.text) for i, r in enumerate(draft.recommendations)
    ]
    surfaces += [
        (f"evidence_synthesis[{i}].proposition", e.proposition)
        for i, e in enumerate(draft.evidence_synthesis)
    ]
    for path, text in surfaces:
        if _RAW_HTML.search(text):
            violations.append(
                Violation(
                    "raw_html_in_report",
                    "report prose may not contain raw HTML; the renderings inline it",
                    path=path,
                )
            )
        if _HTML_EVENT_ATTR.search(text):
            violations.append(
                Violation(
                    "html_event_handler_in_report",
                    "report prose may not contain an HTML event handler attribute", path=path,
                )
            )
        if _REMOTE_IMAGE.search(text):
            violations.append(
                Violation(
                    "remote_image_in_report",
                    "a report may only show figures this build produced and stored; a remote "
                    "image would fetch from a third party every time the report is opened",
                    path=path,
                )
            )
    return violations


def _validate_semantic_consistency(
    draft: ReportDraftCandidate, context: ReportValidationContext
) -> list[Violation]:
    """The cross-section gates (P0-2).

    Everything above this line checks one part of the draft against the server's
    data. This checks the draft against *itself*: whether one section denies
    what another section, and the underlying observation, both state. The audit
    report passed every gate above and shipped with two such contradictions.
    """
    situation = EvidenceSituation(
        requested=context.include_external_evidence,
        search_performed=context.evidence_search_performed,
        candidates_found=context.evidence_candidates_found,
        provider_failed=context.evidence_provider_failed,
        promoted=len(context.evidence_by_id),
    )
    referenced = {
        ref.explanation_id: package
        for ref in draft.explanations
        if (package := context.explanations_by_id.get(ref.explanation_id)) is not None
    }
    return [
        *check_explanation_consistency(sections=draft.sections, explanations=referenced),
        *check_evidence_scope_consistency(
            sections=draft.sections, limitations=draft.limitations, situation=situation
        ),
        *check_evidence_state_gap(gaps=draft.gaps, situation=situation),
    ]
