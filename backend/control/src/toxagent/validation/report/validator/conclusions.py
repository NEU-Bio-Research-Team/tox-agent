"""Checks on conclusions, recommendations, limitations and gaps."""
from __future__ import annotations

from ....domain.answer import LimitationCode
from ....domain.errors import Violation
from ...answer.citations import (
    validate_recommendation_basis,
)
from ...limitations import required_for_answer
from ...prohibited_claims import (
    matches_unnegated,
    validate_answer_markdown,
)
from ..draft_wire import ReportDraftCandidate
from ._common import _GUARANTEE, _KNOWN_LIMITATION_CODES, ReportValidationContext, _prefixed


def _validate_conclusions(
    draft: ReportDraftCandidate, context: ReportValidationContext
) -> list[Violation]:
    """Every conclusion names an endpoint or is labelled integrated, and rests
    on claims that exist. No conclusion may be an aggregate safety verdict."""
    violations: list[Violation] = []
    known_claims = {c.claim_id for c in draft.claims}
    for index, conclusion in enumerate(draft.conclusions):
        path = f"conclusions[{index}]"
        if conclusion.endpoint is None and not conclusion.is_integrated:
            violations.append(
                Violation(
                    "conclusion_scope_missing",
                    "a conclusion is either about one endpoint or explicitly marked as an "
                    "integrated screening interpretation",
                    path=f"{path}.endpoint",
                )
            )
        if conclusion.endpoint and conclusion.endpoint not in context.served_endpoints:
            violations.append(
                Violation(
                    "conclusion_endpoint_not_served",
                    f"conclusion names endpoint {conclusion.endpoint!r}, which this analysis "
                    "does not serve",
                    path=f"{path}.endpoint",
                    expected=list(context.served_endpoints),
                )
            )
        unknown = [cid for cid in conclusion.basis_claim_ids if cid not in known_claims]
        if unknown:
            violations.append(
                Violation(
                    "conclusion_basis_unknown",
                    f"conclusion basis references claim(s) this report does not contain: "
                    f"{unknown}",
                    path=f"{path}.basis_claim_ids", actual=unknown,
                )
            )
        violations += _prefixed(validate_answer_markdown(conclusion.text), path)
    return violations


def _validate_recommendations(draft: ReportDraftCandidate) -> list[Violation]:
    """Recommendation basis, plus the wording line specific to recommendations.

    A recommendation is meant to propose an action, so the general clinical
    check is not enough: what it may never do is promise an outcome.
    """
    violations: list[Violation] = []
    known_claims = {c.claim_id for c in draft.claims}
    for index, recommendation in enumerate(draft.recommendations):
        path = f"recommendations[{index}]"
        violations += validate_recommendation_basis(
            index, recommendation.basis_claim_ids, known_claims
        )
        for text, field_name in (
            (recommendation.text, "text"), (recommendation.rationale, "rationale"),
        ):
            violations += _prefixed(validate_answer_markdown(text), f"{path}.{field_name}")
            if matches_unnegated(_GUARANTEE, text):
                violations.append(
                    Violation(
                        "recommendation_guarantees_safety",
                        "a recommendation proposes follow-up work; it cannot assure safety or "
                        "promise an outcome",
                        path=f"{path}.{field_name}",
                    )
                )
        unknown = [
            cid for cid in recommendation.basis_claim_ids if cid not in known_claims
        ]
        if unknown:
            violations.append(
                Violation(
                    "recommendation_basis_unknown",
                    f"recommendation basis references claim(s) this report does not contain: "
                    f"{unknown}",
                    path=f"{path}.basis_claim_ids", actual=unknown,
                )
            )
    return violations


def _validate_limitations(
    draft: ReportDraftCandidate, context: ReportValidationContext
) -> list[Violation]:
    """The declared limitations cover everything the report's own claims require.

    Derived from the claims, exactly as for an answer, so a limitation cannot be
    avoided by leaving it out. ``screening_not_safety_assessment`` is additionally
    unconditional here: a whole document about toxicity endpoints is precisely
    the artifact a reader is most likely to mistake for a safety assessment.
    """
    violations: list[Violation] = []
    for index, limitation in enumerate(draft.limitations):
        if limitation.code not in _KNOWN_LIMITATION_CODES:
            violations.append(
                Violation(
                    "unknown_limitation_code",
                    f"{limitation.code!r} is not a declared limitation code",
                    path=f"limitations[{index}].code",
                )
            )

    observation_limitations = {
        obs_id: obs.required_limitations
        for obs_id, obs in context.observations_by_id.items()
    }
    required = set(
        required_for_answer(
            draft.claims,
            observation_limitations=observation_limitations,
            cited_evidence=any(c.citation_ids for c in draft.claims),
            has_recommendation=bool(draft.recommendations),
        )
    )
    required.add(LimitationCode.SCREENING_NOT_SAFETY_ASSESSMENT.value)
    if draft.explanations:
        required.add(LimitationCode.ATTRIBUTION_NOT_CAUSALITY.value)
    if any(e not in context.served_endpoints for e in context.selected_endpoints):
        required.add(LimitationCode.ENDPOINT_UNAVAILABLE.value)
    if draft.evidence_synthesis or any(c.citation_ids for c in draft.claims):
        required.add(LimitationCode.EVIDENCE_SCOPE_LIMITED.value)

    missing = sorted(required - {l.code for l in draft.limitations})
    if missing:
        violations.append(
            Violation(
                "missing_required_limitation",
                f"this report requires limitation(s) it did not declare: {missing}",
                path="limitations", expected=missing,
            )
        )
    return violations


def _validate_gaps(draft: ReportDraftCandidate) -> list[Violation]:
    """Each gap is attached to a section that exists and is referenced by it."""
    violations: list[Violation] = []
    by_section: dict[str, set[str]] = {
        s.section_id: set(s.gap_ids) for s in draft.sections
    }
    ids = [g.gap_id for g in draft.gaps]
    duplicates = sorted({gid for gid in ids if ids.count(gid) > 1})
    if duplicates:
        violations.append(
            Violation("duplicate_gap_id", f"gap id(s) repeated: {duplicates}", path="gaps")
        )
    for index, gap in enumerate(draft.gaps):
        referenced = by_section.get(gap.section_id)
        if referenced is None:
            violations.append(
                Violation(
                    "gap_section_unknown",
                    f"gap {gap.gap_id!r} names section {gap.section_id!r}, which the report "
                    "does not contain",
                    path=f"gaps[{index}].section_id",
                )
            )
        elif gap.gap_id not in referenced:
            violations.append(
                Violation(
                    "gap_not_shown",
                    f"gap {gap.gap_id!r} is declared but section {gap.section_id!r} does not "
                    "reference it, so a reader would never see it",
                    path=f"sections[{gap.section_id}].gap_ids",
                )
            )
    for section in draft.sections:
        unknown = sorted(set(section.gap_ids) - set(ids))
        if unknown:
            violations.append(
                Violation(
                    "section_references_unknown_gap",
                    f"section {section.section_id!r} references gap(s) the report does not "
                    f"declare: {unknown}",
                    path=f"sections[{section.section_id}].gap_ids",
                )
            )
    return violations
