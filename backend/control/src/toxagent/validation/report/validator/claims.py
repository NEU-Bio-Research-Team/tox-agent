"""Checks on what the draft asserts: claims, explanations and evidence."""
from __future__ import annotations

from ....domain.errors import Violation
from ....domain.observation import ObservationKind
from ....domain.report import (
    EvidenceRelation,
    ExplanationStatus,
    SourceClass,
)
from ....predictor.contract import TOX21_TASKS
from ...answer.citations import (
    cited_in_prose,
    validate_basis,
    validate_citations,
    validate_observation_reference,
)
from ...answer.classification import validate_classification
from ...answer.coverage import (
    cited_fact_values,
    validate_markdown_numeric_coverage,
    validate_no_uncited_links,
)
from ...answer.numeric import validate_derived_numeric, validate_field_backed_numeric
from ...prohibited_claims import (
    validate_answer_markdown,
    validate_claim_wording,
    validate_no_hitcount_severity,
)
from ..draft_wire import ReportDraftCandidate
from ._common import _BASIS_REQUIRED_KINDS, _DERIVED_TRANSFORMS, ReportValidationContext, _prefixed
from .structure import _validate_source_separation


def _validate_claims(
    draft: ReportDraftCandidate, context: ReportValidationContext
) -> list[Violation]:
    """Numeric, classification, basis, citation and wording — the answer
    validator's checks, unchanged, plus source separation."""
    violations: list[Violation] = []
    by_id = {c.claim_id: c for c in draft.claims}
    ids = [c.claim_id for c in draft.claims]
    duplicates = sorted({cid for cid in ids if ids.count(cid) > 1})
    if duplicates:
        violations.append(
            Violation("duplicate_claim_id", f"claim id(s) repeated: {duplicates}", path="claims")
        )

    section_of: dict[str, str] = {}
    for section in draft.sections:
        for claim_id in section.claim_ids:
            section_of.setdefault(claim_id, section.section_id)
    classes_of = {
        section.section_id: frozenset(SourceClass(c) for c in section.source_classes)
        for section in draft.sections
    }

    for claim in draft.claims:
        observation = (
            context.observations_by_id.get(claim.observation_id)
            if claim.observation_id else None
        )
        is_derived = claim.transform in _DERIVED_TRANSFORMS

        if claim.kind == "numeric":
            violations += (
                validate_derived_numeric(claim, by_id) if is_derived
                else validate_field_backed_numeric(claim, observation)
            )
        elif claim.kind == "classification":
            if is_derived:
                violations.append(
                    Violation(
                        "claim_transform_invalid_for_kind",
                        "a classification claim cannot use a difference/ratio transform",
                        path=f"claims[{claim.claim_id}].transform",
                    )
                )
            else:
                violations += validate_classification(claim, observation)
        elif claim.kind in _BASIS_REQUIRED_KINDS:
            if is_derived:
                violations += validate_derived_numeric(claim, by_id)
            has_observation_basis = bool(
                observation is not None
                and (
                    (claim.field_path and observation.has(claim.field_path))
                    or observation.kind is ObservationKind.ATTRIBUTION
                )
            )
            violations += validate_basis(claim, has_observation_basis=has_observation_basis)

        violations += validate_observation_reference(claim, context.observations_by_id)
        violations += validate_citations(
            claim, context.evidence_by_id, read_evidence_ids=context.read_evidence_ids
        )
        violations += validate_claim_wording(claim)

        # Every claim is placed. An unplaced claim is worse than a missing one:
        # it satisfies a conclusion's basis requirement while appearing nowhere
        # a reader could check it.
        if claim.claim_id not in section_of:
            violations.append(
                Violation(
                    "claim_not_placed",
                    f"claim {claim.claim_id!r} is not referenced by any section, so nothing in "
                    "the report shows the reader what it supports",
                    path=f"claims[{claim.claim_id}]",
                )
            )
            continue

        violations += _validate_source_separation(claim, section_of, classes_of)

    for section in draft.sections:
        unknown = [cid for cid in section.claim_ids if cid not in by_id]
        if unknown:
            violations.append(
                Violation(
                    "section_cites_unknown_claim",
                    f"section {section.section_id!r} references claim(s) the draft does not "
                    f"define: {unknown}",
                    path=f"sections[{section.section_id}].claim_ids",
                )
            )

    violations += validate_no_hitcount_severity(
        draft.claims, "\n".join(s.body_markdown for s in draft.sections)
    )
    for section in draft.sections:
        violations += _prefixed(
            validate_answer_markdown(section.body_markdown), f"sections[{section.section_id}]"
        )
        violations += _prefixed(
            validate_no_uncited_links(section.body_markdown), f"sections[{section.section_id}]"
        )
        section_claims = [by_id[cid] for cid in section.claim_ids if cid in by_id]
        # A section cites through its claims and through [@evd_…] in its prose.
        cited = {eid for claim in section_claims for eid in claim.citation_ids}
        cited |= cited_in_prose(section.body_markdown)
        violations += _prefixed(
            validate_markdown_numeric_coverage(
                section.body_markdown, section_claims,
                claimed_values=cited_fact_values(cited, context.evidence_by_id),
            ),
            f"sections[{section.section_id}]",
        )
    return violations


def _validate_explanations(
    draft: ReportDraftCandidate, context: ReportValidationContext
) -> list[Violation]:
    """Explanation linkage: the ref, the package, and the figure agree.

    Eval scenario 12 is the case this exists for — a figure that has drifted
    onto the wrong endpoint, model or observation. A reader cannot catch that
    by looking; the ids can.
    """
    violations: list[Violation] = []
    for index, ref in enumerate(draft.explanations):
        package = context.explanations_by_id.get(ref.explanation_id)
        path = f"explanations[{index}]"
        if package is None:
            violations.append(
                Violation(
                    "explanation_not_found",
                    f"explanation {ref.explanation_id!r} was not produced by this build",
                    path=path, actual=ref.explanation_id,
                )
            )
            continue
        if package.endpoint != ref.endpoint or package.task != ref.task:
            violations.append(
                Violation(
                    "explanation_target_mismatch",
                    f"explanation {ref.explanation_id!r} explains "
                    f"{package.endpoint}/{package.task}, not {ref.endpoint}/{ref.task}",
                    path=path,
                    expected=[package.endpoint, package.task],
                    actual=[ref.endpoint, ref.task],
                )
            )
        if package.figure is not None:
            figure = package.figure
            if figure.endpoint != package.endpoint or figure.task != package.task:
                violations.append(
                    Violation(
                        "figure_target_mismatch",
                        f"the figure on explanation {ref.explanation_id!r} depicts "
                        f"{figure.endpoint}/{figure.task}",
                        path=f"{path}.figure",
                    )
                )
            if figure.observation_id != package.observation_id:
                violations.append(
                    Violation(
                        "figure_observation_mismatch",
                        "the figure and the explanation narrative do not point at the same "
                        "observation, so the image and the text describe different computations",
                        path=f"{path}.figure",
                        expected=package.observation_id,
                        actual=figure.observation_id,
                    )
                )

        # A partial explanation, or one with unmapped attribution mass, must be
        # said out loud somewhere. Silence here overstates how much of the
        # score the picture accounts for (spec section 6.2, eval scenario 5).
        unmapped = package.highlights.unmapped_importance
        if package.status is ExplanationStatus.PARTIAL or (unmapped or 0) > 0:
            disclosed = any(
                "unmapped" in section.body_markdown.lower()
                or "partial" in section.body_markdown.lower()
                for section in draft.sections
                if section.section_id == "explanation_and_visuals"
            ) or any(
                gap.reason == "explanation_partial" for gap in draft.gaps
            )
            if not disclosed:
                violations.append(
                    Violation(
                        "unmapped_importance_suppressed",
                        f"explanation {ref.explanation_id!r} is partial or leaves "
                        f"{unmapped!r} of its attribution unmapped, and the report does not "
                        "say so",
                        path="sections[explanation_and_visuals]",
                    )
                )

    if context.include_explanations:
        required_targets: set[tuple[str, str | None]] = set()
        for endpoint in context.selected_endpoints:
            if endpoint == "tox21":
                required_targets.update(
                    (endpoint, task)
                    for task in (context.selected_tox21_tasks or TOX21_TASKS)
                )
            else:
                required_targets.add((endpoint, None))
        covered = {(ref.endpoint, ref.task) for ref in draft.explanations}
        gap_targets = {
            (gap.endpoint, gap.task)
            for gap in draft.gaps
            if gap.reason in {"explanation_failed", "explanation_partial"}
        }
        missing_targets = sorted(
            required_targets - covered - gap_targets,
            key=lambda target: (target[0], target[1] or ""),
        )
        if missing_targets:
            violations.append(
                Violation(
                    "explanations_requested_but_absent",
                    "this build asked for an explanation for every selected endpoint/assay, "
                    "and some targets have neither a package nor a target-specific gap",
                    path="explanations",
                    expected=[list(target) for target in missing_targets],
                )
            )
    return violations


def _validate_evidence(
    draft: ReportDraftCandidate, context: ReportValidationContext
) -> list[Violation]:
    """Read-before-cite, and evidence context that may not be suppressed."""
    violations: list[Violation] = []
    for index, synthesis in enumerate(draft.evidence_synthesis):
        path = f"evidence_synthesis[{index}]"
        relation = EvidenceRelation(synthesis.relation)
        if relation is not EvidenceRelation.INSUFFICIENT and not synthesis.evidence_ids:
            violations.append(
                Violation(
                    "synthesis_without_evidence",
                    f"a synthesis with relation {relation.value!r} names no evidence record; "
                    "only 'insufficient' may stand on nothing",
                    path=f"{path}.evidence_ids",
                )
            )
        for evidence_id in synthesis.evidence_ids:
            record = context.evidence_by_id.get(evidence_id)
            if record is None:
                violations.append(
                    Violation(
                        "citation_not_found",
                        f"evidence {evidence_id!r} does not exist in this session",
                        path=f"{path}.evidence_ids", actual=evidence_id,
                    )
                )
                continue
            if evidence_id not in context.read_evidence_ids:
                violations.append(
                    Violation(
                        "citation_not_read",
                        f"evidence {evidence_id!r} was never opened with get_evidence_record; a "
                        "search result is a byline, not a source",
                        path=f"{path}.evidence_ids", actual=evidence_id,
                    )
                )

    # A contradiction that exists must be visible. Finding a conflicting record
    # and reporting only the supporting ones is the failure mode that makes a
    # literature section actively misleading (eval scenario 8).
    contradicting = [
        s for s in draft.evidence_synthesis if s.relation == "contradicts"
    ]
    if contradicting:
        evidence_section = next(
            (s for s in draft.sections if s.section_id == "external_evidence"), None
        )
        integrated = next(
            (s for s in draft.sections if s.section_id == "integrated_interpretation"), None
        )
        text = " ".join(
            s.body_markdown.lower() for s in (evidence_section, integrated) if s
        )
        if not any(
            word in text
            for word in ("conflict", "contradict", "disagree", "inconsistent", "at odds")
        ):
            violations.append(
                Violation(
                    "evidence_conflict_suppressed",
                    "the synthesis records contradicting evidence, and neither the external "
                    "evidence nor the integrated interpretation section mentions the conflict",
                    path="sections[external_evidence]",
                )
            )

    if context.include_external_evidence and not draft.evidence_synthesis:
        if not any(gap.reason in {"no_relevant_evidence", "provider_unavailable",
                                  "budget_exhausted"} for gap in draft.gaps):
            violations.append(
                Violation(
                    "research_outcome_missing",
                    "this build asked for external evidence and the report records neither a "
                    "synthesis nor a gap explaining the absence; 'no relevant evidence found' "
                    "is a valid outcome, silence is not",
                    path="evidence_synthesis",
                )
            )
    return violations
