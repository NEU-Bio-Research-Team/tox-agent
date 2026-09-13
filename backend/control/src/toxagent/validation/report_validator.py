"""The deterministic report validator (spec section 11): draft in, decision out.

The acceptance boundary. A ``preflight-report-draft`` skill can improve a
first-pass draft, and does not decide anything; this module does. Every gate in
spec section 11's table is applied here, in one place, and passing all of them
is what produces the compiled artifact — there is no path that stores a draft
without going through every check.

Almost nothing here is new logic. Numbers, classifications, citations,
limitations and prohibited wording are the *same* validators a grounded answer
goes through (``answer_validator.py``), because a claim in a report is a claim,
and a report that could assert a number an answer could not would be a way to
route around the trust chain. What is genuinely new is the structural half:
section coverage, endpoint coverage, explanation-to-figure linkage, source
separation and render integrity — the things a document has and a chat reply
does not.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final, Mapping, Sequence

from ..domain.answer import LimitationCode
from ..domain.errors import Violation
from ..domain.evidence import EvidenceRecord
from ..domain.observation import Observation, ObservationKind
from ..domain.report import (
    REQUIRED_SECTION_IDS,
    EvidenceRelation,
    ExplanationPackage,
    ExplanationStatus,
    ReportFigure,
    SourceClass,
)
from ..predictor.contract import TOX21_TASKS
from .citations import (
    cited_in_prose,
    validate_basis,
    validate_citations,
    validate_recommendation_basis,
)
from .classification import validate_classification
from .coverage import validate_markdown_numeric_coverage, validate_no_uncited_links
from .limitations import required_for_answer
from .numeric import validate_derived_numeric, validate_field_backed_numeric
from .prohibited_claims import (
    matches_unnegated,
    validate_answer_markdown,
    validate_claim_wording,
    validate_no_hitcount_severity,
)
from .report_semantics import (
    EvidenceSituation,
    check_evidence_scope_consistency,
    check_evidence_state_gap,
    check_explanation_consistency,
)
from .report_wire import ReportDraftCandidate

_KNOWN_LIMITATION_CODES: Final = frozenset(c.value for c in LimitationCode)
_DERIVED_TRANSFORMS: Final = frozenset({"difference", "ratio"})
_BASIS_REQUIRED_KINDS: Final = frozenset({"scientific", "comparison"})

#: Which source class a claim of each kind may be filed under. Spec section 3.4
#: forbids blurring the classes; this is that table read backwards, so a
#: predictor number placed in the external-evidence section is a violation
#: rather than a formatting choice.
_KIND_TO_ALLOWED_CLASSES: Final[dict[str, frozenset[SourceClass]]] = {
    "numeric": frozenset(
        {SourceClass.PREDICTOR_FACT, SourceClass.EXPLANATION_FACT,
         SourceClass.STRUCTURE_FACT, SourceClass.EXTERNAL_EVIDENCE}
    ),
    "classification": frozenset({SourceClass.PREDICTOR_FACT}),
    "scientific": frozenset(
        {SourceClass.EXTERNAL_EVIDENCE, SourceClass.EXPLANATION_FACT,
         SourceClass.AGENT_SYNTHESIS, SourceClass.STRUCTURE_FACT}
    ),
    "comparison": frozenset({SourceClass.AGENT_SYNTHESIS, SourceClass.PREDICTOR_FACT}),
    "limitation": frozenset({SourceClass.AGENT_SYNTHESIS}),
    "recommendation": frozenset({SourceClass.RECOMMENDATION}),
}

#: Sections that must say something. The other required sections may legitimately
#: consist of a single recorded gap (there was no relevant literature; the
#: explainer failed), but a report whose predictor results or provenance are
#: blank is not a partial report, it is an empty one.
_MUST_NOT_BE_EMPTY: Final[frozenset[str]] = frozenset(
    {"executive_summary", "substance_profile", "predictor_results",
     "conclusions", "limitations", "provenance_appendix"}
)

#: Raw HTML and remote embeds in report prose. External text is untrusted data
#: (spec section 11 "Content safety"), and the HTML rendering inlines this
#: markdown — an <img src="http://..."> in a section body is a tracking beacon
#: fired by every reader of the report.
_RAW_HTML = re.compile(r"<\s*/?\s*(script|iframe|object|embed|img|svg|style|link|meta)\b", re.IGNORECASE)
_HTML_EVENT_ATTR = re.compile(r"\son[a-z]+\s*=", re.IGNORECASE)
_REMOTE_IMAGE = re.compile(r"!\[[^\]]*\]\(\s*(https?:)?//", re.IGNORECASE)

#: Wording a recommendation may never use. Separate from the general clinical
#: check because a recommendation is *supposed* to propose an action, so the
#: line is about promising an outcome rather than about mentioning a patient.
_GUARANTEE = re.compile(
    r"\b(guarantee[sd]?|assure[sd]?\s+safety|proven\s+safe|safe\s+for\s+(human|clinical)|"
    r"no\s+risk|risk[- ]free|approved\s+for\s+use)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ReportValidationResult:
    violations: tuple[Violation, ...]
    #: Present only when every gate passed. The compiler turns it into the
    #: immutable artifact; nothing else may.
    accepted_draft: ReportDraftCandidate | None = None

    @property
    def ok(self) -> bool:
        return not self.violations and self.accepted_draft is not None


@dataclass(frozen=True)
class ReportValidationContext:
    """Everything the validator needs that is *not* in the draft.

    All of it is server-known: what the build asked for, what the snapshot
    actually served, which observations and evidence records exist, and which
    evidence the run genuinely opened. A draft cannot supply any of it, which
    is what makes these gates checks rather than assertions.
    """

    session_id: str
    report_build_id: str
    analysis_id: str
    selected_endpoints: tuple[str, ...]
    served_endpoints: tuple[str, ...]
    observations_by_id: Mapping[str, Observation]
    evidence_by_id: Mapping[str, EvidenceRecord]
    explanations_by_id: Mapping[str, ExplanationPackage]
    figures_by_id: Mapping[str, ReportFigure]
    figure_errors: Mapping[str, str] = field(default_factory=dict)
    read_evidence_ids: frozenset[str] = frozenset()
    include_explanations: bool = True
    include_external_evidence: bool = True
    #: What actually happened when this build looked for external evidence.
    #: Server-known and never supplied by a draft, which is what lets the
    #: semantic gates check a claim about the search rather than repeat it.
    evidence_search_performed: bool = False
    evidence_candidates_found: int = 0
    evidence_provider_failed: bool = False
    selected_tox21_tasks: tuple[str, ...] = ()
    language: str = "en"


def validate_report_draft(
    draft: ReportDraftCandidate,
    *,
    context: ReportValidationContext,
    now: datetime,
) -> ReportValidationResult:
    """Apply every gate. Order matches spec section 11's table."""
    violations: list[Violation] = []

    violations += _validate_scope(draft, context)
    violations += _validate_sections(draft)
    violations += _validate_claims(draft, context)
    violations += _validate_endpoint_coverage(draft, context)
    violations += _validate_explanations(draft, context)
    violations += _validate_figures(draft, context)
    violations += _validate_evidence(draft, context)
    violations += _validate_conclusions(draft, context)
    violations += _validate_recommendations(draft)
    violations += _validate_limitations(draft, context)
    violations += _validate_gaps(draft)
    violations += _validate_content_safety(draft)
    violations += _validate_citation_tokens(draft, context)
    violations += _validate_references(draft, context)
    violations += _validate_semantic_consistency(draft, context)

    if violations:
        return ReportValidationResult(violations=tuple(violations))
    return ReportValidationResult(violations=(), accepted_draft=draft)


# --- gates ------------------------------------------------------------------


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
        violations += _prefixed(
            validate_markdown_numeric_coverage(section.body_markdown, section_claims),
            f"sections[{section.section_id}]",
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
        for text, field in (
            (recommendation.text, "text"), (recommendation.rationale, "rationale"),
        ):
            violations += _prefixed(validate_answer_markdown(text), f"{path}.{field}")
            if matches_unnegated(_GUARANTEE, text):
                violations.append(
                    Violation(
                        "recommendation_guarantees_safety",
                        "a recommendation proposes follow-up work; it cannot assure safety or "
                        "promise an outcome",
                        path=f"{path}.{field}",
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


def _prefixed(violations: Sequence[Violation], prefix: str) -> list[Violation]:
    """Re-root a reused validator's paths under the report structure, so a
    correction attempt is told which *section* to fix, not just 'answer_markdown'."""
    out: list[Violation] = []
    for violation in violations:
        path = f"{prefix}.{violation.path}" if violation.path else prefix
        out.append(
            Violation(
                violation.code, violation.message, path=path,
                expected=violation.expected, actual=violation.actual,
            )
        )
    return out
