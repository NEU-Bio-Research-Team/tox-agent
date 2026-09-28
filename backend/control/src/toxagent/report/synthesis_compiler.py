"""Compile a fact bundle and a synthesis into a report (PR-11).

The audit's report was assembled by the model: it gathered the facts, wrote
each one into however many sections mentioned it, chose its own limitations,
and declared its own gaps. Two sections disagreed about one explanation, and a
limitation described a literature search that had not happened.

Here the split is absolute.

**The model owns prose.** Narrative sections, conclusions, recommendations, and
what a piece of evidence means for a prediction.

**The compiler owns everything else.** The predictor results section, the
limitations, the gaps, the references, the provenance appendix, the endpoint
assessments, and every value that appears anywhere — because prose arrives with
``{{fct_...}}`` placeholders and this module substitutes the canonical
rendering from the bundle. The same placeholder in two sections renders the
same string by construction. That is the structural half of P0-2; the semantic
gates are the half that catches what structure cannot.

Nothing here is a network call or a database read. It takes a bundle the stages
assembled and a synthesis the model submitted, and returns either a compiled
report or typed violations.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..domain.errors import Violation
from ..domain.report import REQUIRED_SECTION_IDS, SourceClass
from ..validation.report.synthesis_wire import (
    FACT_PLACEHOLDER,
    ReportSynthesisV3,
)
from .fact_bundle import ExplanationFacts, ReportFact, ReportFactBundle

COMPILER_VERSION = "report-compiler-v3"

#: Sections this module writes from the bundle, in the order a report reads.
COMPILED_SECTIONS = ("predictor_results", "limitations", "references", "provenance_appendix")


@dataclass(frozen=True, slots=True)
class CompiledSection:
    section_id: str
    heading: str
    #: The whole section as a reader sees it: what the model wrote, then
    #: whatever the compiler appended.
    body_markdown: str
    #: Only the part the model authored, after fact substitution. Kept apart
    #: because the gates that ask "did a model type a number here" must not be
    #: shown the compiler's own renderings and conclude the compiler cheated.
    authored_markdown: str = ""
    #: Facts this section's text actually renders, after substitution. Recorded
    #: so a reader — or a renderer, or a diff — can go from a sentence back to
    #: the observation behind it without re-parsing prose.
    fact_ids: tuple[str, ...] = ()
    #: True when the compiler wrote the whole section. A reader can tell which
    #: sentences are a model's interpretation and which are the product's own
    #: rendering.
    compiled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_id": self.section_id,
            "heading": self.heading,
            "body_markdown": self.body_markdown,
            "authored_markdown": self.authored_markdown,
            "fact_ids": list(self.fact_ids),
            "compiled": self.compiled,
        }


@dataclass(frozen=True, slots=True)
class EndpointAssessment:
    """One endpoint's compiled result line. Never an aggregate (ADR 0002)."""

    endpoint: str
    task: str | None
    served: bool
    probability: float | None = None
    rendered_probability: str = ""
    label: str = ""
    threshold: float | None = None
    threshold_source: str = ""
    model_id: str = ""
    explanation_id: str | None = None
    explanation_status: str = "absent"
    coverage_status: str = "unknown"
    gap_reason: str | None = None
    fact_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "task": self.task,
            "served": self.served,
            "probability": self.probability,
            "rendered_probability": self.rendered_probability,
            "label": self.label,
            "threshold": self.threshold,
            "threshold_source": self.threshold_source,
            "model_id": self.model_id,
            "explanation_id": self.explanation_id,
            "explanation_status": self.explanation_status,
            "coverage_status": self.coverage_status,
            "gap_reason": self.gap_reason,
            "fact_ids": list(self.fact_ids),
        }


@dataclass(frozen=True, slots=True)
class CompiledReport:
    report_build_id: str
    analysis_id: str
    title: str
    language: str
    sections: tuple[CompiledSection, ...]
    endpoint_assessments: tuple[EndpointAssessment, ...]
    conclusions: tuple[Mapping[str, Any], ...]
    recommendations: tuple[Mapping[str, Any], ...]
    evidence_interpretations: tuple[Mapping[str, Any], ...]
    limitations: tuple[Mapping[str, str], ...]
    gaps: tuple[Mapping[str, str], ...]
    references: tuple[Mapping[str, Any], ...]
    provenance: Mapping[str, Any] = field(default_factory=dict)
    compiler_version: str = COMPILER_VERSION
    schema_version: str = "toxagent-report-v3"

    @property
    def has_gaps(self) -> bool:
        return bool(self.gaps)

    def content_sha256(self) -> str:
        """Over the scientific content, not the renderings.

        A report whose PDF was re-rendered is the same report; a report whose
        numbers changed is not.
        """
        payload = {
            "schema_version": self.schema_version,
            "title": self.title,
            "sections": [section.to_dict() for section in self.sections],
            "endpoint_assessments": [a.to_dict() for a in self.endpoint_assessments],
            "conclusions": [dict(c) for c in self.conclusions],
            "recommendations": [dict(r) for r in self.recommendations],
            "evidence_interpretations": [dict(e) for e in self.evidence_interpretations],
            "limitations": [dict(l) for l in self.limitations],
            "gaps": [dict(g) for g in self.gaps],
            "references": [dict(r) for r in self.references],
        }
        return "sha256:" + hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class CompilationResult:
    report: CompiledReport | None
    violations: tuple[Violation, ...] = ()

    @property
    def ok(self) -> bool:
        return self.report is not None and not self.violations


# --- substitution -----------------------------------------------------------


def substitute_facts(
    text: str, facts: Mapping[str, ReportFact]
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Replace ``{{fct_...}}`` with the canonical rendering.

    Returns the rendered text, the facts it used, and the placeholders that
    resolved to nothing. The last one is not silently dropped: a sentence whose
    number never arrived would otherwise publish as "The probability is ."
    """
    used: list[str] = []
    missing: list[str] = []

    def replace(match) -> str:
        fact_id = match.group(1)
        fact = facts.get(fact_id)
        if fact is None:
            missing.append(fact_id)
            return match.group(0)
        used.append(fact_id)
        return fact.rendered

    rendered = FACT_PLACEHOLDER.sub(replace, text)
    return rendered, tuple(dict.fromkeys(used)), tuple(dict.fromkeys(missing))


# --- compiler-owned sections ------------------------------------------------


def _predictor_results_section(
    assessments: Sequence[EndpointAssessment], language: str
) -> CompiledSection:
    """A table, rendered from facts. The model never writes this section.

    Three separate rows, never a combined score: the endpoints are different
    measurements and a reader who sums them has been misled by the document.
    """
    header = "| Endpoint | Assay | Probability | Classification | Threshold | Model |"
    divider = "|---|---|---|---|---|---|"
    rows = []
    for item in assessments:
        if not item.served:
            rows.append(
                f"| {item.endpoint} | {item.task or '—'} | not served | — | — | — |"
            )
            continue
        rows.append(
            f"| {item.endpoint} | {item.task or '—'} | {item.rendered_probability or '—'} "
            f"| {item.label or '—'} | {item.threshold if item.threshold is not None else '—'} "
            f"| {item.model_id or '—'} |"
        )
    note = (
        "\n\nEach endpoint is a separate measurement. This product does not "
        "produce a combined score across them."
    )
    return CompiledSection(
        section_id="predictor_results",
        heading="Predictor results",
        body_markdown="\n".join([header, divider, *rows]) + note,
        fact_ids=tuple(fid for item in assessments for fid in item.fact_ids),
        compiled=True,
    )


#: The wording for each mandatory limitation. One place, so two reports cannot
#: describe the same limitation differently — and, critically, so no wording
#: can describe a method the build did not use (P0-2's second contradiction).
LIMITATION_TEXT: Mapping[str, str] = {
    "uncalibrated_probability": (
        "The probabilities are model outputs and are not calibrated to any "
        "clinical or population prevalence."
    ),
    "applicability_is_rule_based": (
        "Applicability is a rule-based check on the elements present, not a "
        "learned in-distribution or out-of-distribution test."
    ),
    "attribution_not_causality": (
        "Attribution shows what moved this model's score. It is not evidence of "
        "a chemical mechanism."
    ),
    "endpoint_unavailable": (
        "At least one requested endpoint was not served by this analysis and is "
        "reported as unavailable rather than as a negative result."
    ),
    "evidence_scope_limited": (
        "The external evidence in this report is limited to what the configured "
        "provider returned and what relevance assessment admitted; absence of a "
        "citation is not evidence of absence."
    ),
    "screening_not_safety_assessment": (
        "This is a screening report. It is not a safety assessment and does not "
        "support a decision about human exposure."
    ),
}

#: Wording used when no search ran at all. Distinct from the sentence above,
#: which describes a search that happened and was limited.
NO_SEARCH_LIMITATION_TEXT = (
    "No external literature was consulted for this build, so nothing here is "
    "corroborated against published data."
)


def _limitations(
    bundle: ReportFactBundle, *, search_performed: bool
) -> tuple[dict[str, str], ...]:
    """Compiler-owned, and worded from what the build actually did.

    The audit's report carried `evidence_scope_limited` phrased as "the search
    covered one provider and did not read full texts" in a build that never
    searched. The code was right; the sentence described a method that was not
    used. Here the code and the sentence come from the same place.
    """
    out: list[dict[str, str]] = []
    for code in bundle.required_limitations:
        if code == "evidence_scope_limited" and not search_performed:
            out.append({"code": code, "text": NO_SEARCH_LIMITATION_TEXT})
            continue
        out.append({"code": code, "text": LIMITATION_TEXT.get(code, "")})
    return tuple(out)


def _gaps(
    bundle: ReportFactBundle, *, search_performed: bool
) -> tuple[Mapping[str, str], ...]:
    """The bundle's gaps, plus the evidence-state gap if nothing recorded one.

    Four states, not one: nobody looked, the search found nothing, the provider
    was down, nothing found was relevant enough. A report that cannot tell them
    apart tells a reader the literature is thin when it may be unexamined
    (P0-2).
    """
    gaps = [dict(gap) for gap in bundle.gaps]
    known = {gap.get("reason") for gap in gaps}
    evidence_reasons = {
        "external_evidence_not_requested",
        "no_relevant_evidence",
        "provider_unavailable",
    }
    if known & evidence_reasons:
        return tuple(gaps)

    requested = bool(bundle.policy.get("include_external_evidence", True))
    if not requested:
        gaps.append(
            {
                "reason": "external_evidence_not_requested",
                "section_id": "external_evidence",
                "detail": "this build did not ask for a literature search",
            }
        )
    elif not search_performed and not bundle.evidence:
        gaps.append(
            {
                "reason": "provider_unavailable",
                "section_id": "external_evidence",
                "detail": "no literature search completed for this build",
            }
        )
    elif not bundle.evidence:
        gaps.append(
            {
                "reason": "no_relevant_evidence",
                "section_id": "external_evidence",
                "detail": "the search returned nothing relevant enough to cite",
            }
        )
    return tuple(gaps)


def _references(bundle: ReportFactBundle) -> tuple[dict[str, Any], ...]:
    """Numbered 1..n over promoted evidence, in a stable order."""
    return tuple(
        {
            "number": index,
            "evidence_id": record.get("evidence_id"),
            "title": record.get("title", ""),
            "canonical_url": record.get("canonical_url"),
            "relevance": record.get("relevance"),
        }
        for index, record in enumerate(bundle.evidence, start=1)
    )


def _provenance_section(bundle: ReportFactBundle) -> CompiledSection:
    lines = [f"- compiler: {COMPILER_VERSION}", f"- fact bundle: {bundle.schema_version}"]
    for key in sorted(bundle.provenance):
        lines.append(f"- {key}: {bundle.provenance[key]}")
    for key in sorted(bundle.policy):
        lines.append(f"- policy.{key}: {bundle.policy[key]}")
    return CompiledSection(
        section_id="provenance_appendix",
        heading="Provenance",
        body_markdown="\n".join(lines),
        compiled=True,
    )


def _references_section(references: Sequence[Mapping[str, Any]]) -> CompiledSection:
    if not references:
        body = "No external sources are cited in this report."
    else:
        body = "\n".join(
            f"{item['number']}. {item.get('title') or item.get('evidence_id')}"
            + (f" — {item['canonical_url']}" if item.get("canonical_url") else "")
            for item in references
        )
    return CompiledSection(
        section_id="references",
        heading="References",
        body_markdown=body,
        compiled=True,
    )


def _limitations_section(limitations: Sequence[Mapping[str, str]]) -> CompiledSection:
    body = "\n".join(f"- {item['text']}" for item in limitations if item.get("text"))
    return CompiledSection(
        section_id="limitations",
        heading="Limitations",
        body_markdown=body or "No limitations were derived for this report.",
        compiled=True,
    )


def _compiler_addendum(section_id: str, bundle: ReportFactBundle) -> str:
    """What the compiler adds to a section the model wrote.

    Only one today, and it is the audit's first contradiction: the explanation
    coverage sentence. A model asked to write it can write it wrongly — the
    audit's did, reporting no contributors and no unmapped mass for an
    explanation that had three and 35.84%. Appending the canonical sentence
    means the correct statement is present whatever the narrative says, and the
    semantic gate refuses a narrative that contradicts it.
    """
    if section_id != "explanation_and_visuals" or not bundle.explanations:
        return ""
    lines = [
        f"- {item.endpoint}{'' if item.task is None else '/' + item.task}: "
        f"{item.summary_sentence}"
        for item in bundle.explanations
    ]
    return "\n\nAttribution coverage, as computed:\n" + "\n".join(lines)


# --- endpoint assessments ---------------------------------------------------


def _assessments(bundle: ReportFactBundle) -> tuple[EndpointAssessment, ...]:
    explanations_by_target: dict[tuple[str, str | None], ExplanationFacts] = {
        (item.endpoint, item.task): item for item in bundle.explanations
    }
    by_path = bundle.by_path()
    out: list[EndpointAssessment] = []
    for endpoint_facts in bundle.endpoints:
        endpoint = endpoint_facts.endpoint
        if not endpoint_facts.served:
            out.append(
                EndpointAssessment(
                    endpoint=endpoint,
                    task=None,
                    served=False,
                    gap_reason=endpoint_facts.gap_reason,
                )
            )
            continue
        probability_fact = next(
            (
                fact
                for fact in endpoint_facts.facts
                if fact.path.split(".")[-1].startswith("probability")
            ),
            None,
        )
        explanation = explanations_by_target.get((endpoint, None))
        out.append(
            EndpointAssessment(
                endpoint=endpoint,
                task=None,
                served=True,
                probability=probability_fact.value if probability_fact else None,
                rendered_probability=probability_fact.rendered if probability_fact else "",
                label=str(_value(by_path, f"predictions.{endpoint}.label") or ""),
                threshold=_value(by_path, f"predictions.{endpoint}.threshold"),
                threshold_source=str(
                    _value(by_path, f"predictions.{endpoint}.threshold_source") or ""
                ),
                model_id=str(_value(by_path, f"predictions.{endpoint}.model_id") or ""),
                explanation_id=explanation.explanation_id if explanation else None,
                explanation_status=explanation.status if explanation else "absent",
                coverage_status=(
                    explanation.coverage.coverage_status if explanation else "unknown"
                ),
                fact_ids=tuple(fact.id for fact in endpoint_facts.facts),
            )
        )
    return tuple(out)


def _value(by_path: Mapping[str, ReportFact], path: str) -> Any:
    fact = by_path.get(path)
    return fact.value if fact else None


# --- the compiler ------------------------------------------------------------


def compile_report(
    *,
    bundle: ReportFactBundle,
    synthesis: ReportSynthesisV3,
    search_performed: bool = False,
) -> CompilationResult:
    """Render one report. Deterministic, and total over a valid synthesis.

    Violations here are about *linkage* — a placeholder that resolves to
    nothing, a basis fact that does not exist, a required section nobody wrote.
    Semantic contradictions are the validator's job, on the output of this.
    """
    violations: list[Violation] = []
    facts = bundle.by_id()

    if synthesis.report_build_id != bundle.report_build_id:
        return CompilationResult(
            None,
            (
                Violation(
                    "synthesis_build_mismatch",
                    "this synthesis names a different report build than the fact bundle",
                    path="report_build_id",
                    expected=bundle.report_build_id,
                    actual=synthesis.report_build_id,
                ),
            ),
        )

    sections: list[CompiledSection] = []
    for index, section in enumerate(synthesis.sections):
        compiled, used, missing = substitute_facts(section.prose_markdown, facts)
        if missing:
            violations.append(
                Violation(
                    "fact_reference_unresolved",
                    f"section {section.section_id!r} references fact(s) the bundle does "
                    f"not contain: {sorted(missing)}",
                    path=f"sections[{index}].prose_markdown",
                    actual=sorted(missing),
                )
            )
        unknown_basis = sorted(set(section.basis_fact_ids) - set(facts))
        if unknown_basis:
            violations.append(
                Violation(
                    "basis_fact_unresolved",
                    f"section {section.section_id!r} names basis fact(s) the bundle does "
                    f"not contain: {unknown_basis}",
                    path=f"sections[{index}].basis_fact_ids",
                    actual=unknown_basis,
                )
            )
        addendum = _compiler_addendum(section.section_id, bundle)
        sections.append(
            CompiledSection(
                section_id=section.section_id,
                heading=section.heading,
                body_markdown=(compiled + addendum) if addendum else compiled,
                authored_markdown=compiled,
                fact_ids=used,
            )
        )

    conclusions, conclusion_violations = _compile_basis_items(
        synthesis.conclusions, facts, kind="conclusion"
    )
    recommendations, recommendation_violations = _compile_basis_items(
        synthesis.recommendations, facts, kind="recommendation"
    )
    violations.extend(conclusion_violations)
    violations.extend(recommendation_violations)

    assessments = _assessments(bundle)
    limitations = _limitations(bundle, search_performed=search_performed)
    references = _references(bundle)
    gaps = _gaps(bundle, search_performed=search_performed)

    sections.append(_predictor_results_section(assessments, bundle.language))
    sections.append(_limitations_section(limitations))
    sections.append(_references_section(references))
    sections.append(_provenance_section(bundle))

    written = {section.section_id for section in sections}
    missing_sections = [sid for sid in REQUIRED_SECTION_IDS if sid not in written]
    if missing_sections:
        violations.append(
            Violation(
                "required_section_missing",
                "a section whose content is unavailable carries a gap; it does not "
                f"disappear. Missing: {missing_sections}",
                path="sections",
                expected=missing_sections,
            )
        )

    if violations:
        return CompilationResult(None, tuple(violations))

    ordered = _in_report_order(sections)
    return CompilationResult(
        CompiledReport(
            report_build_id=bundle.report_build_id,
            analysis_id=bundle.analysis_id,
            title=synthesis.title,
            language=bundle.language,
            sections=ordered,
            endpoint_assessments=assessments,
            conclusions=tuple(conclusions),
            recommendations=tuple(recommendations),
            evidence_interpretations=tuple(
                item.model_dump() for item in synthesis.evidence_interpretations
            ),
            limitations=limitations,
            gaps=gaps,
            references=references,
            provenance={
                **dict(bundle.provenance),
                "compiler_version": COMPILER_VERSION,
                "fact_bundle_version": bundle.schema_version,
            },
        )
    )


def _compile_basis_items(
    items: Sequence[Any], facts: Mapping[str, ReportFact], *, kind: str
) -> tuple[list[dict[str, Any]], list[Violation]]:
    compiled: list[dict[str, Any]] = []
    violations: list[Violation] = []
    for index, item in enumerate(items):
        unknown = sorted(set(item.basis_fact_ids) - set(facts))
        if unknown:
            violations.append(
                Violation(
                    "basis_fact_unresolved",
                    f"{kind} {item.local_ref!r} names basis fact(s) the bundle does not "
                    f"contain: {unknown}",
                    path=f"{kind}s[{index}].basis_fact_ids",
                    actual=unknown,
                )
            )
        text, _, missing = substitute_facts(item.text, facts)
        if missing:
            violations.append(
                Violation(
                    "fact_reference_unresolved",
                    f"{kind} {item.local_ref!r} references fact(s) the bundle does not "
                    f"contain: {sorted(missing)}",
                    path=f"{kind}s[{index}].text",
                    actual=sorted(missing),
                )
            )
        payload = item.model_dump()
        payload["text"] = text
        payload["source_class"] = (
            SourceClass.RECOMMENDATION.value
            if kind == "recommendation"
            else SourceClass.AGENT_SYNTHESIS.value
        )
        compiled.append(payload)
    return compiled, violations


def _in_report_order(sections: Sequence[CompiledSection]) -> tuple[CompiledSection, ...]:
    order = {sid: index for index, sid in enumerate(REQUIRED_SECTION_IDS)}
    return tuple(
        sorted(sections, key=lambda section: order.get(section.section_id, len(order)))
    )
