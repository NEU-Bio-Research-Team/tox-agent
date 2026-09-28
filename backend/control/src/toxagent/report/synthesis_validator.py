"""The gates a compiled v3 report must pass before it is published (PR-11).

The compiler already made a whole class of error impossible: a value in prose is
a placeholder the server rendered, so two sections cannot state two different
numbers for one fact. What structure cannot prevent is a sentence that *denies*
a fact, or asserts one nobody measured, or describes a method the build did not
use. That is what these gates are for, and the audit's report failed all three.

Everything here runs on the **compiled** representation, after substitution. A
gate that ran on the raw synthesis would be reading placeholders, not the
sentences a person will actually read.

Ordered from cheapest to most specific, and every violation is typed and names
a path, so a correction attempt knows exactly what to change.
"""
from __future__ import annotations

import re
from typing import Mapping, Sequence

from ..domain.errors import Violation
from ..validation.prohibited_claims import validate_answer_markdown
from ..validation.report_semantics import (
    EvidenceSituation,
    check_evidence_scope_consistency,
    check_evidence_state_gap,
    check_explanation_consistency,
)
from .compiler_v3 import CompiledReport
from .fact_bundle import ReportFactBundle

#: A number that looks like a measurement: a decimal, a Vietnamese decimal
#: comma, or a percentage. Bare integers are excluded — "three atoms", "two
#: assays", "step 1" are prose, not smuggled predictions. Deliberately the same
#: shape the answer validator uses, because a number is a number whichever
#: artifact it appears in.
_MEASUREMENT = re.compile(r"(?<![\w.,])(?<![\w]-)-?\d+(?:[.,]\d+%?|%)(?![\w])(?![.,]\d)")

#: Sections whose text the compiler wrote. Their numbers came from facts by
#: construction, so scanning them would flag the compiler's own renderings.
_COMPILER_OWNED = frozenset(
    {"predictor_results", "limitations", "references", "provenance_appendix"}
)


def validate_compiled_report(
    report: CompiledReport,
    *,
    bundle: ReportFactBundle,
    explanations: Mapping[str, object] | None = None,
    situation: EvidenceSituation | None = None,
) -> list[Violation]:
    """Every gate. Returns an empty list for a report that may be published."""
    violations: list[Violation] = []
    violations += _validate_numbers_come_from_facts(report, bundle)
    violations += _validate_endpoint_coverage(report, bundle)
    violations += _validate_conclusion_scope(report)
    violations += _validate_required_limitations(report, bundle)
    violations += _validate_prose_safety(report)
    violations += _validate_evidence_interpretations(report, bundle)
    violations += _validate_semantics(report, bundle, explanations, situation)
    return violations


# --- numbers ----------------------------------------------------------------


def _validate_numbers_come_from_facts(
    report: CompiledReport, bundle: ReportFactBundle
) -> list[Violation]:
    """A measurement in model-written prose must be a rendered fact.

    The placeholder mechanism means a *correct* number arrives by substitution.
    A number typed directly is therefore either a fact the model restated from
    memory — the thing this whole design removes — or one it invented.
    """
    rendered = {fact.rendered for fact in bundle.facts if fact.rendered}
    violations: list[Violation] = []
    for section in report.sections:
        if section.compiled or section.section_id in _COMPILER_OWNED:
            continue
        # Only what the model wrote. The compiler's own addendum carries
        # renderings by construction, and scanning it would have the gate
        # accuse the compiler of the thing it exists to prevent.
        for token in _unique(_MEASUREMENT.findall(section.authored_markdown)):
            if token in rendered:
                continue
            violations.append(
                Violation(
                    "unreferenced_measurement",
                    f"{token!r} appears in section {section.section_id!r} and is not the "
                    "rendering of any fact in this report's bundle. Write {{fact_id}} and "
                    "let the server render it.",
                    path=f"sections[{section.section_id}].body_markdown",
                    actual=token,
                )
            )
    for item in (*report.conclusions, *report.recommendations):
        for token in _unique(_MEASUREMENT.findall(str(item.get("text", "")))):
            if token in rendered:
                continue
            violations.append(
                Violation(
                    "unreferenced_measurement",
                    f"{token!r} appears in {item.get('local_ref')!r} and is not the "
                    "rendering of any fact in this report's bundle",
                    path=f"conclusions[{item.get('local_ref')}].text",
                    actual=token,
                )
            )
    return violations


def _unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


# --- coverage ---------------------------------------------------------------


def _validate_endpoint_coverage(
    report: CompiledReport, bundle: ReportFactBundle
) -> list[Violation]:
    """Every selected endpoint appears exactly once, or carries a gap.

    An endpoint that was asked for and silently absent reads as a negative
    result. It is not one; it is a missing measurement (SCI-06).
    """
    violations: list[Violation] = []
    assessed = [item.endpoint for item in report.endpoint_assessments]
    duplicates = sorted({e for e in assessed if assessed.count(e) > 1})
    if duplicates:
        violations.append(
            Violation(
                "endpoint_assessed_twice",
                f"endpoint(s) assessed more than once: {duplicates}",
                path="endpoint_assessments",
            )
        )
    selected = [item.endpoint for item in bundle.endpoints]
    missing = sorted(set(selected) - set(assessed))
    if missing:
        violations.append(
            Violation(
                "endpoint_not_assessed",
                f"endpoint(s) selected for this report and neither assessed nor gapped: "
                f"{missing}",
                path="endpoint_assessments",
                expected=missing,
            )
        )
    gap_reasons = {gap.get("reason") for gap in report.gaps}
    for item in report.endpoint_assessments:
        if item.served or item.gap_reason:
            continue
        violations.append(
            Violation(
                "endpoint_unserved_without_gap",
                f"{item.endpoint} was not served and the report records no gap for it",
                path=f"endpoint_assessments[{item.endpoint}]",
            )
        )
    if any(not item.served for item in report.endpoint_assessments):
        if "endpoint_not_served" not in gap_reasons:
            violations.append(
                Violation(
                    "endpoint_unserved_without_gap",
                    "an endpoint is reported as not served and no gap says so",
                    path="gaps",
                )
            )
    return violations


def _validate_conclusion_scope(report: CompiledReport) -> list[Violation]:
    """A per-endpoint conclusion names an endpoint the report assessed."""
    assessed = {item.endpoint for item in report.endpoint_assessments if item.served}
    violations: list[Violation] = []
    for item in report.conclusions:
        endpoint = item.get("endpoint")
        if item.get("is_integrated") or endpoint is None:
            continue
        if endpoint not in assessed:
            violations.append(
                Violation(
                    "conclusion_endpoint_not_assessed",
                    f"conclusion {item.get('local_ref')!r} is about {endpoint!r}, which "
                    "this report did not assess",
                    path=f"conclusions[{item.get('local_ref')}].endpoint",
                    actual=endpoint,
                )
            )
    return violations


# --- limitations ------------------------------------------------------------


def _validate_required_limitations(
    report: CompiledReport, bundle: ReportFactBundle
) -> list[Violation]:
    """Every required limitation is present, and every present one has words.

    The compiler owns both, so a failure here is a compiler bug rather than a
    model error — which is exactly why it is checked: a silent empty limitation
    is a limitation a reader never sees.
    """
    declared = {item["code"] for item in report.limitations}
    missing = sorted(set(bundle.required_limitations) - declared)
    violations: list[Violation] = []
    if missing:
        violations.append(
            Violation(
                "missing_required_limitation",
                f"this report requires limitation(s) it does not carry: {missing}",
                path="limitations",
                expected=missing,
            )
        )
    for item in report.limitations:
        if not str(item.get("text", "")).strip():
            violations.append(
                Violation(
                    "limitation_without_text",
                    f"limitation {item.get('code')!r} has no wording, so a reader never "
                    "sees it",
                    path=f"limitations[{item.get('code')}].text",
                )
            )
    return violations


# --- wording ----------------------------------------------------------------


def _validate_prose_safety(report: CompiledReport) -> list[Violation]:
    """The hard gates, over every sentence a person will read.

    Run on the compiled text rather than the synthesis: a placeholder that
    renders into a sentence can change what that sentence claims.
    """
    violations: list[Violation] = []
    for section in report.sections:
        for violation in validate_answer_markdown(section.body_markdown):
            violations.append(
                Violation(
                    violation.code,
                    violation.message,
                    path=f"sections[{section.section_id}].body_markdown",
                )
            )
    for item in (*report.conclusions, *report.recommendations):
        for violation in validate_answer_markdown(str(item.get("text", ""))):
            violations.append(
                Violation(
                    violation.code,
                    violation.message,
                    path=f"conclusions[{item.get('local_ref')}].text",
                )
            )
    return violations


# --- evidence ---------------------------------------------------------------


def _validate_evidence_interpretations(
    report: CompiledReport, bundle: ReportFactBundle
) -> list[Violation]:
    """An interpretation names a record this build actually promoted.

    Relevance assessment already decided what is citable. This only checks that
    the model is interpreting one of those, not a record it remembers from
    somewhere else (P1-2).
    """
    promoted = {
        str(record.get("evidence_id"))
        for record in bundle.evidence
        if record.get("evidence_id")
    }
    violations: list[Violation] = []
    for item in report.evidence_interpretations:
        evidence_id = str(item.get("evidence_id"))
        if evidence_id not in promoted:
            violations.append(
                Violation(
                    "evidence_not_citable",
                    f"evidence {evidence_id!r} is interpreted here and was not promoted "
                    "by this build's relevance assessment",
                    path="evidence_interpretations",
                    actual=evidence_id,
                )
            )
    if report.evidence_interpretations and not report.references:
        violations.append(
            Violation(
                "interpretation_without_reference",
                "this report interprets external evidence and lists no references",
                path="references",
            )
        )
    return violations


# --- the cross-section gates ------------------------------------------------


def _validate_semantics(
    report: CompiledReport,
    bundle: ReportFactBundle,
    explanations: Mapping[str, object] | None,
    situation: EvidenceSituation | None,
) -> list[Violation]:
    """P0-2, over the compiled sections.

    Reuses the gates the answer path already has rather than restating them:
    one set of rules about denying a fact, applied wherever prose is published.
    """
    if situation is None:
        situation = EvidenceSituation(
            requested=bool(bundle.policy.get("include_external_evidence", True)),
            search_performed=bool(bundle.policy.get("search_performed", False)),
            candidates_found=int(bundle.policy.get("evidence_candidates_found", 0) or 0),
            provider_failed=bool(bundle.policy.get("evidence_provider_failed", False)),
            promoted=len(bundle.evidence),
        )
    return [
        *check_explanation_consistency(
            sections=report.sections, explanations=dict(explanations or {})
        ),
        *check_evidence_scope_consistency(
            sections=report.sections,
            limitations=[_LimitationView(item) for item in report.limitations],
            situation=situation,
        ),
        *check_evidence_state_gap(
            gaps=[_GapView(gap) for gap in report.gaps], situation=situation
        ),
    ]


class _LimitationView:
    """Adapts a compiled limitation dict to what the semantic gates read."""

    __slots__ = ("code", "text")

    def __init__(self, item: Mapping[str, str]) -> None:
        self.code = item.get("code", "")
        self.text = item.get("text", "")


class _GapView:
    __slots__ = ("gap_id", "reason", "section_id")

    def __init__(self, gap: Mapping[str, str]) -> None:
        self.gap_id = gap.get("gap_id", gap.get("reason", ""))
        self.reason = gap.get("reason", "")
        self.section_id = gap.get("section_id", "")
