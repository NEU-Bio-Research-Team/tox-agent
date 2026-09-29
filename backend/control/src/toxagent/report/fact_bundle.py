"""Every fact a report may state, assembled by the server (WS05 5A / PR-10).

The audit's report let the model gather facts itself, restate them section by
section, and mint the identifiers it used to refer to them. Two sections of one
artifact then disagreed about the same explanation, and nothing could catch it
because there was no single place either sentence came from.

A fact bundle is that single place. Everything a report is allowed to assert
about the substance, the predictions, the explanations and the evidence exists
here once, with:

* a **stable id** — derived from the build and the fact's path, so a resumed
  build produces the same ids and a section written before a restart still
  resolves after one;
* a **source class** — predictor, structure, explanation, external evidence or
  request policy. A reader can tell a measured number from a retrieved one;
* a **typed value and a canonical rendering** — the rendering is produced here,
  under the report's locale, so "0,731" is a formatting rule applied once
  rather than a string a model has to type correctly (ADR 0005);
* a **reference** to the observation, explanation or evidence record it came
  from, so every assertion is traceable to something immutable.

The model's job shrinks to choosing which facts answer the question and writing
prose that cites them by id. It does not send facts back, which is what makes
the executive-summary contradiction (P0-2) structurally impossible rather than
merely detected.

Nothing here reaches a network or a database. It projects what the caller has
already loaded, which keeps the part that decides what a report may say
testable without a predictor, a provider or a runtime.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from ..domain.xai_coverage import ExplanationCoverage, compute_coverage
from ..domain.report import ExplanationPackage, ExplanationStatus, SourceClass

#: Bumped when the fact shape changes. A bundle assembled under one version is
#: never merged with one assembled under another.
BUNDLE_SCHEMA_VERSION = "report-fact-bundle-1"

#: Languages whose decimal separator is a comma. Same rule as the answer
#: compiler's, and deliberately the same list: two renderers disagreeing about
#: a decimal point is exactly the drift this module exists to prevent.
_COMMA_DECIMAL_LANGUAGES = frozenset({"vi"})


class FactKind(str, Enum):
    NUMERIC = "numeric"
    CLASSIFICATION = "classification"
    COUNT = "count"
    FRACTION = "fraction"
    BOOLEAN = "boolean"
    TEXT = "text"
    IDENTIFIER = "identifier"


def fact_id(report_build_id: str, path: str) -> str:
    """Deterministic from the build and the path.

    Not random: a stage that is re-run after a worker restart must produce the
    same ids, or a draft written against the first attempt would refer to facts
    that no longer exist.
    """
    digest = hashlib.sha256(f"{report_build_id}|{path}".encode("utf-8")).hexdigest()
    return f"fct_{digest[:32]}"


def render_value(value: Any, kind: FactKind, *, language: str, digits: int = 3) -> str:
    """The one canonical rendering of a fact. Plain: no units, no phrasing."""
    if value is None:
        return ""
    if kind is FactKind.BOOLEAN:
        return "true" if value else "false"
    if kind is FactKind.FRACTION and isinstance(value, (int, float)):
        text = f"{float(value) * 100:.{digits - 1}f}%"
    elif kind is FactKind.NUMERIC and isinstance(value, (int, float)):
        text = f"{float(value):.{digits}f}"
    elif kind is FactKind.COUNT:
        text = str(int(value))
    else:
        return str(value)
    if language in _COMMA_DECIMAL_LANGUAGES:
        text = text.replace(".", ",")
    return text


@dataclass(frozen=True, slots=True)
class ReportFact:
    id: str
    path: str
    label: str
    kind: FactKind
    source_class: SourceClass
    value: Any
    rendered: str
    observation_id: str | None = None
    explanation_id: str | None = None
    evidence_id: str | None = None
    endpoint: str | None = None
    task: str | None = None
    unit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.id,
            "path": self.path,
            "label": self.label,
            "kind": self.kind.value,
            "source_class": self.source_class.value,
            "value": self.value,
            "rendered": self.rendered,
            "observation_id": self.observation_id,
            "explanation_id": self.explanation_id,
            "evidence_id": self.evidence_id,
            "endpoint": self.endpoint,
            "task": self.task,
            "unit": self.unit,
        }


@dataclass(frozen=True, slots=True)
class EndpointFacts:
    """One endpoint's facts, plus whether it was served at all."""

    endpoint: str
    served: bool
    facts: tuple[ReportFact, ...] = ()
    gap_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ExplanationFacts:
    """One explanation, and the coverage accounting nothing else may restate."""

    explanation_id: str
    endpoint: str
    task: str | None
    status: str
    coverage: ExplanationCoverage
    facts: tuple[ReportFact, ...] = ()

    @property
    def summary_sentence(self) -> str:
        """The one wording for this explanation's coverage.

        Every section that mentions contributors or unmapped mass uses this
        string. That is the structural half of the P0-2 fix: two sections
        cannot disagree about a sentence neither of them wrote.
        """
        return self.coverage.summary_sentence()


@dataclass(frozen=True, slots=True)
class ReportFactBundle:
    report_build_id: str
    analysis_id: str
    language: str
    schema_version: str = BUNDLE_SCHEMA_VERSION
    facts: tuple[ReportFact, ...] = ()
    endpoints: tuple[EndpointFacts, ...] = ()
    explanations: tuple[ExplanationFacts, ...] = ()
    #: Evidence that survived relevance assessment, with the assessment.
    evidence: tuple[Mapping[str, Any], ...] = ()
    #: What the request asked for and what policy allowed. Recorded so a report
    #: can be read years later against the rules it was built under.
    policy: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    required_limitations: tuple[str, ...] = ()
    gaps: tuple[Mapping[str, str], ...] = ()

    def by_id(self) -> dict[str, ReportFact]:
        return {fact.id: fact for fact in self.facts}

    def by_path(self) -> dict[str, ReportFact]:
        return {fact.path: fact for fact in self.facts}

    def resolve(self, fact_ids: Iterable[str]) -> tuple[list[ReportFact], list[str]]:
        """Facts for these ids, and the ids that resolved to nothing.

        Both halves returned: a draft citing a fact that does not exist is a
        typed violation, and silently dropping it would publish a sentence with
        no basis.
        """
        index = self.by_id()
        found = [index[fid] for fid in fact_ids if fid in index]
        missing = [fid for fid in fact_ids if fid not in index]
        return found, missing

    def to_model_view(self) -> dict[str, Any]:
        """What the synthesis stage is shown. Ids, labels and renderings only.

        Not the canonical payloads: a model handed the raw prediction object
        can restate a number from it, and then the report has two sources for
        one fact again.
        """
        return {
            "schema_version": self.schema_version,
            "report_build_id": self.report_build_id,
            "analysis_id": self.analysis_id,
            "language": self.language,
            "facts": [
                {
                    "fact_id": fact.id,
                    "label": fact.label,
                    "rendered": fact.rendered,
                    "source_class": fact.source_class.value,
                    "endpoint": fact.endpoint,
                    "task": fact.task,
                }
                for fact in self.facts
            ],
            "explanations": [
                {
                    "explanation_id": item.explanation_id,
                    "endpoint": item.endpoint,
                    "task": item.task,
                    "status": item.status,
                    "coverage_status": item.coverage.coverage_status,
                    # The sentence, not the numbers behind it. A section that
                    # wants to talk about coverage quotes this.
                    "summary": item.summary_sentence,
                }
                for item in self.explanations
            ],
            "evidence": [dict(record) for record in self.evidence],
            "required_limitations": list(self.required_limitations),
            "gaps": [dict(gap) for gap in self.gaps],
            "policy": dict(self.policy),
        }


# --- assembly ---------------------------------------------------------------


#: Which prediction fields become facts, and how each should be read. A closed
#: list, for the same reason ``SECTION_FIELDS`` is one: a report may only state
#: what the product has agreed to show.
_PREDICTION_FIELDS: Mapping[str, tuple[tuple[str, str, FactKind], ...]] = {
    "herg": (
        ("probability_blocker", "hERG blocker probability", FactKind.NUMERIC),
        ("label", "hERG classification", FactKind.CLASSIFICATION),
        ("threshold", "hERG decision threshold", FactKind.NUMERIC),
        ("threshold_source", "hERG threshold source", FactKind.TEXT),
        ("model_id", "hERG model", FactKind.IDENTIFIER),
    ),
    "clintox": (
        (
            "probability_clinical_toxicity",
            "ClinTox clinical-toxicity probability",
            FactKind.NUMERIC,
        ),
        ("label", "ClinTox classification", FactKind.CLASSIFICATION),
        ("threshold", "ClinTox decision threshold", FactKind.NUMERIC),
        ("threshold_source", "ClinTox threshold source", FactKind.TEXT),
        ("model_id", "ClinTox model", FactKind.IDENTIFIER),
    ),
    "tox21": (
        ("model_id", "Tox21 model", FactKind.IDENTIFIER),
        ("task_order_version", "Tox21 assay order", FactKind.IDENTIFIER),
    ),
}

_TOX21_ASSAY_FIELDS: tuple[tuple[str, str, FactKind], ...] = (
    ("probability_activity", "activity probability", FactKind.NUMERIC),
    ("active", "called active", FactKind.BOOLEAN),
    ("threshold", "decision threshold", FactKind.NUMERIC),
)


def _fact(
    *,
    report_build_id: str,
    path: str,
    label: str,
    kind: FactKind,
    source_class: SourceClass,
    value: Any,
    language: str,
    **refs: Any,
) -> ReportFact:
    return ReportFact(
        id=fact_id(report_build_id, path),
        path=path,
        label=label,
        kind=kind,
        source_class=source_class,
        value=value,
        rendered=render_value(value, kind, language=language),
        **refs,
    )


def _prediction_facts(
    *,
    report_build_id: str,
    predictions: Mapping[str, Any],
    observation_ids: Mapping[str, str],
    language: str,
    selected_tox21_tasks: Sequence[str],
) -> dict[str, list[ReportFact]]:
    by_endpoint: dict[str, list[ReportFact]] = {}
    for endpoint, fields in _PREDICTION_FIELDS.items():
        section = predictions.get(endpoint)
        if not isinstance(section, Mapping):
            continue
        observation_id = observation_ids.get(endpoint)
        facts: list[ReportFact] = []
        for name, label, kind in fields:
            if name not in section:
                continue
            facts.append(
                _fact(
                    report_build_id=report_build_id,
                    path=f"predictions.{endpoint}.{name}",
                    label=label,
                    kind=kind,
                    source_class=SourceClass.PREDICTOR_FACT,
                    value=section[name],
                    language=language,
                    observation_id=observation_id,
                    endpoint=endpoint,
                )
            )
        if endpoint == "tox21":
            facts.extend(
                _tox21_facts(
                    report_build_id=report_build_id,
                    section=section,
                    observation_id=observation_id,
                    language=language,
                    selected=selected_tox21_tasks,
                )
            )
        by_endpoint[endpoint] = facts
    return by_endpoint


def _tox21_facts(
    *,
    report_build_id: str,
    section: Mapping[str, Any],
    observation_id: str | None,
    language: str,
    selected: Sequence[str],
) -> list[ReportFact]:
    """One fact per assay field. Twelve independent measurements, never summed.

    ADR 0002 and the audit's own non-goals: there is no aggregate here, and a
    count of active assays is not a severity.
    """
    assays = section.get("tasks") or section.get("assays") or {}
    if not isinstance(assays, Mapping):
        return []
    wanted = tuple(selected) if selected else tuple(assays)
    facts: list[ReportFact] = []
    for task in wanted:
        values = assays.get(task)
        if not isinstance(values, Mapping):
            continue
        for name, label, kind in _TOX21_ASSAY_FIELDS:
            if name not in values:
                continue
            facts.append(
                _fact(
                    report_build_id=report_build_id,
                    path=f"predictions.tox21.{task}.{name}",
                    label=f"Tox21 {task} {label}",
                    kind=kind,
                    source_class=SourceClass.PREDICTOR_FACT,
                    value=values[name],
                    language=language,
                    observation_id=observation_id,
                    endpoint="tox21",
                    task=task,
                )
            )
    return facts


def _substance_facts(
    *, report_build_id: str, substance: Mapping[str, Any], language: str
) -> list[ReportFact]:
    fields = (
        ("canonical_smiles", "Canonical SMILES", FactKind.IDENTIFIER),
        ("preferred_name", "Preferred name", FactKind.TEXT),
        ("inchikey", "InChIKey", FactKind.IDENTIFIER),
        ("molecular_formula", "Molecular formula", FactKind.TEXT),
        ("molecular_weight", "Molecular weight", FactKind.NUMERIC),
    )
    return [
        _fact(
            report_build_id=report_build_id,
            path=f"substance.{name}",
            label=label,
            kind=kind,
            source_class=SourceClass.STRUCTURE_FACT,
            value=substance[name],
            language=language,
        )
        for name, label, kind in fields
        if substance.get(name) not in (None, "")
    ]


def _explanation_facts(
    *,
    report_build_id: str,
    package: ExplanationPackage,
    language: str,
) -> tuple[ExplanationFacts, list[ReportFact]]:
    """Coverage counts owned by the compiler, never by the model.

    ``positive_contributors``/``negative_contributors``/``unmapped_importance``
    become facts with ids. A section that wants to say "three atoms contribute
    negatively" cites the count fact; it does not count the list again, which is
    how the audit's executive summary came to report zero of both.
    """
    highlights = package.highlights
    stored = dict(highlights.coverage or {})
    coverage = (
        ExplanationCoverage(
            mapped_importance_fraction=stored.get("mapped_importance_fraction"),
            special_token_importance_fraction=stored.get(
                "special_token_importance_fraction"
            ),
            other_unmapped_importance_fraction=stored.get(
                "other_unmapped_importance_fraction"
            ),
            unmapped_importance=stored.get("unmapped_importance"),
            coverage_status=str(stored.get("coverage_status") or "unknown"),
            positive_contributor_count=int(stored.get("positive_contributor_count") or 0),
            negative_contributor_count=int(stored.get("negative_contributor_count") or 0),
        )
        if stored
        # An explanation stored before WS06 has no coverage block. Recomputing
        # it from the highlights is better than reporting "unknown", and the
        # two paths agree because both come from the same payload shape.
        else compute_coverage(
            {"unmapped_importance": highlights.unmapped_importance},
            positive_contributor_count=len(highlights.positive_contributors),
            negative_contributor_count=len(highlights.negative_contributors),
        )
    )

    target = package.endpoint if package.task is None else f"{package.endpoint}.{package.task}"
    numbers: tuple[tuple[str, str, FactKind, Any], ...] = (
        (
            "positive_contributor_count",
            "positive contributors",
            FactKind.COUNT,
            coverage.positive_contributor_count,
        ),
        (
            "negative_contributor_count",
            "negative contributors",
            FactKind.COUNT,
            coverage.negative_contributor_count,
        ),
        (
            "mapped_importance_fraction",
            "attribution mass on atoms or bonds",
            FactKind.FRACTION,
            coverage.mapped_importance_fraction,
        ),
        (
            "special_token_importance_fraction",
            "attribution mass on sequence markers",
            FactKind.FRACTION,
            coverage.special_token_importance_fraction,
        ),
        (
            "unmapped_importance",
            "unmapped attribution mass",
            FactKind.FRACTION,
            coverage.unmapped_importance,
        ),
    )
    facts = [
        _fact(
            report_build_id=report_build_id,
            path=f"explanations.{target}.{name}",
            label=f"{target}: {label}",
            kind=kind,
            source_class=SourceClass.EXPLANATION_FACT,
            value=value,
            language=language,
            explanation_id=package.explanation_id,
            observation_id=package.observation_id,
            endpoint=package.endpoint,
            task=package.task,
        )
        for name, label, kind, value in numbers
        if value is not None
    ]
    return (
        ExplanationFacts(
            explanation_id=package.explanation_id,
            endpoint=package.endpoint,
            task=package.task,
            status=package.status.value,
            coverage=coverage,
            facts=tuple(facts),
        ),
        facts,
    )


def assemble(
    *,
    report_build_id: str,
    analysis_id: str,
    predictions: Mapping[str, Any],
    served_endpoints: Sequence[str],
    selected_endpoints: Sequence[str],
    selected_tox21_tasks: Sequence[str] = (),
    substance: Mapping[str, Any] | None = None,
    explanations: Sequence[ExplanationPackage] = (),
    evidence: Sequence[Mapping[str, Any]] = (),
    observation_ids: Mapping[str, str] | None = None,
    required_limitations: Sequence[str] = (),
    policy: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    extra_gaps: Sequence[Mapping[str, str]] = (),
    language: str = "en",
) -> ReportFactBundle:
    """Project what the stages loaded into the facts a report may state.

    Pure. Everything it needs has already been fetched by the stage handlers,
    which is what lets the rule about what a report may say be tested without
    any of the services that produce it.
    """
    observation_ids = dict(observation_ids or {})
    facts: list[ReportFact] = []
    gaps: list[dict[str, str]] = [dict(gap) for gap in extra_gaps]

    if substance:
        facts.extend(
            _substance_facts(
                report_build_id=report_build_id, substance=substance, language=language
            )
        )
    else:
        gaps.append(
            {
                "reason": "compound_identity_unresolved",
                "section_id": "substance_profile",
                "detail": "no compound identity was resolved for this build",
            }
        )

    by_endpoint = _prediction_facts(
        report_build_id=report_build_id,
        predictions=predictions,
        observation_ids=observation_ids,
        language=language,
        selected_tox21_tasks=selected_tox21_tasks,
    )

    endpoint_facts: list[EndpointFacts] = []
    for endpoint in selected_endpoints:
        served = endpoint in served_endpoints and bool(by_endpoint.get(endpoint))
        if served:
            endpoint_facts.append(
                EndpointFacts(endpoint, True, tuple(by_endpoint[endpoint]))
            )
            facts.extend(by_endpoint[endpoint])
        else:
            # Named rather than omitted: an endpoint asked for and not served is
            # a gap the report must record, and the easiest way to lose one is
            # never to compute the difference (SCI-06).
            endpoint_facts.append(
                EndpointFacts(endpoint, False, (), gap_reason="endpoint_not_served")
            )
            gaps.append(
                {
                    "reason": "endpoint_not_served",
                    "section_id": "predictor_results",
                    "detail": f"{endpoint} was requested and this analysis does not serve it",
                }
            )

    explanation_facts: list[ExplanationFacts] = []
    for package in explanations:
        summary, produced = _explanation_facts(
            report_build_id=report_build_id, package=package, language=language
        )
        explanation_facts.append(summary)
        facts.extend(produced)
        if package.status is ExplanationStatus.FAILED:
            gaps.append(
                {
                    "reason": "explanation_failed",
                    "section_id": "explanation_and_visuals",
                    "detail": package.failure_reason
                    or f"the explainer failed for {package.endpoint}",
                }
            )

    return ReportFactBundle(
        report_build_id=report_build_id,
        analysis_id=analysis_id,
        language=language,
        facts=tuple(facts),
        endpoints=tuple(endpoint_facts),
        explanations=tuple(explanation_facts),
        evidence=tuple(dict(record) for record in evidence),
        policy=dict(policy or {}),
        provenance=dict(provenance or {}),
        required_limitations=tuple(required_limitations),
        gaps=tuple(gaps),
    )
