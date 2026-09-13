"""What the model submits at the one LLM boundary in a report build (PR-11).

Everything a report says that is a *fact* now comes from the fact bundle: the
numbers, the classifications, the coverage fractions, the contributor counts,
the applicability status, the mandatory limitations, the gaps. The model's
remaining job is the one it is actually good at — deciding which facts answer
the question, and writing the prose that connects them.

So this schema has no numeric field at all. There is nothing here for a model
to restate incorrectly, because there is nothing here to restate.

Two shapes do the work:

**A fact reference is a placeholder, not a number.** Prose is written with
``{{fct_...}}`` where a value belongs, and the compiler substitutes the
canonical rendering. "The hERG blocker probability is {{fct_abc}}" cannot
disagree with the observation, and the same placeholder in two sections cannot
render two different numbers — which is the structural half of P0-2.

**The model cannot declare a limitation or a gap.** Both were fields in the v1
draft, and the audit's report used them to describe a literature search that
never ran. They are compiler-owned now: derived from what the stages actually
did, not from what the model remembers about them.

The narrative sections are the only ones here. ``predictor_results``,
``limitations``, ``references`` and ``provenance_appendix`` are compiled, and a
synthesis that tries to write one is refused rather than merged.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "toxagent-report-synthesis-v3"

#: ``{{fct_<32 hex>}}``. Doubled braces so a stray single brace in prose is not
#: mistaken for one, and so a reader can see at a glance that it is a slot.
FACT_PLACEHOLDER = re.compile(r"\{\{(fct_[0-9a-f]{32})\}\}")

FACT_ID_PATTERN = re.compile(r"^fct_[0-9a-f]{32}$")

LOCAL_REF_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,31}$")

#: Sections the model writes. Everything else in a report is compiled.
NarrativeSectionId = Literal[
    "executive_summary",
    "substance_profile",
    "explanation_and_visuals",
    "external_evidence",
    "integrated_interpretation",
    "conclusions",
    "recommendations",
]

#: Sections the compiler owns outright. Naming one here is a typed refusal
#: rather than a silent merge, because a model that writes its own limitations
#: section is a model that can describe a search that did not happen.
COMPILED_SECTION_IDS: frozenset[str] = frozenset(
    {"predictor_results", "limitations", "references", "provenance_appendix"}
)


class _Wire(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _fact_ids(value: list[str], field_name: str) -> list[str]:
    for item in value:
        if not FACT_ID_PATTERN.match(item):
            raise ValueError(
                f"{field_name} must contain fact ids from the report's fact bundle, "
                f"got {item!r}"
            )
    return value


class SynthesisSection(_Wire):
    section_id: NarrativeSectionId
    heading: str = Field(min_length=1, max_length=200)
    prose_markdown: str = Field(default="", max_length=20_000)
    basis_fact_ids: list[str] = Field(
        default_factory=list,
        max_length=64,
        description=(
            "The facts this section rests on. Every fact whose value the prose "
            "states must be here and must appear in the prose as {{fact_id}}."
        ),
    )

    @field_validator("basis_fact_ids")
    @classmethod
    def _basis_shape(cls, value: list[str]) -> list[str]:
        return _fact_ids(value, "section.basis_fact_ids")

    @property
    def referenced_fact_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(FACT_PLACEHOLDER.findall(self.prose_markdown)))


class SynthesisConclusion(_Wire):
    local_ref: str
    text: str = Field(min_length=1, max_length=4000)
    basis_fact_ids: list[str] = Field(min_length=1, max_length=32)
    endpoint: str | None = Field(
        default=None,
        description=(
            "The one endpoint this conclusion is about. Omit it only together "
            "with is_integrated=true."
        ),
    )
    task: str | None = None
    is_integrated: bool = False

    @field_validator("local_ref")
    @classmethod
    def _ref_shape(cls, value: str) -> str:
        if not LOCAL_REF_PATTERN.match(value):
            raise ValueError(
                f"local_ref {value!r} must be 1-32 lowercase letters, digits and "
                "underscores, starting with a letter"
            )
        return value

    @field_validator("basis_fact_ids")
    @classmethod
    def _basis_shape(cls, value: list[str]) -> list[str]:
        return _fact_ids(value, "conclusion.basis_fact_ids")

    @model_validator(mode="after")
    def _scope_is_stated(self) -> "SynthesisConclusion":
        if not self.is_integrated and not self.endpoint:
            raise ValueError(
                "a per-endpoint conclusion must name its endpoint; an interpretation "
                "across endpoints must set is_integrated=true. A conclusion whose "
                "scope is unstated reads as though it applied to everything."
            )
        if self.is_integrated and self.endpoint:
            raise ValueError(
                "an integrated conclusion is not about one endpoint; drop the endpoint "
                "or set is_integrated=false"
            )
        return self


class SynthesisRecommendation(_Wire):
    local_ref: str
    text: str = Field(min_length=1, max_length=2000)
    basis_fact_ids: list[str] = Field(min_length=1, max_length=32)
    action_category: Literal[
        "in_vitro_assay",
        "in_silico_followup",
        "literature_review",
        "structure_modification",
        "data_collection",
        "no_action_indicated",
    ]
    priority: Literal["high", "medium", "low"]
    rationale: str = Field(min_length=1, max_length=2000)
    conditions: str = Field(default="", max_length=1000)

    @field_validator("local_ref")
    @classmethod
    def _ref_shape(cls, value: str) -> str:
        if not LOCAL_REF_PATTERN.match(value):
            raise ValueError(f"local_ref {value!r} is not a valid label")
        return value

    @field_validator("basis_fact_ids")
    @classmethod
    def _basis_shape(cls, value: list[str]) -> list[str]:
        return _fact_ids(value, "recommendation.basis_fact_ids")


class EvidenceInterpretation(_Wire):
    """What a promoted evidence record means for one endpoint.

    The model reads the paper and says how it bears on the prediction. It does
    not decide whether the record is citable — relevance assessment already did
    that, server-side, before this stage ran (P1-2).
    """

    evidence_id: str = Field(pattern=r"^evd_[0-9a-f]{32}$")
    endpoint: str
    task: str | None = None
    relation: Literal["supports", "contradicts", "contextualizes", "insufficient"]
    summary: str = Field(min_length=1, max_length=2000)


class ReportSynthesisV3(_Wire):
    """One complete synthesis. The final action of the synthesizing stage."""

    schema_version: Literal["toxagent-report-synthesis-v3"] = SCHEMA_VERSION
    report_build_id: str = Field(pattern=r"^rpb_[0-9a-f]{32}$")
    title: str = Field(min_length=1, max_length=300)
    sections: list[SynthesisSection] = Field(default_factory=list, max_length=16)
    conclusions: list[SynthesisConclusion] = Field(default_factory=list, max_length=32)
    recommendations: list[SynthesisRecommendation] = Field(
        default_factory=list, max_length=16
    )
    evidence_interpretations: list[EvidenceInterpretation] = Field(
        default_factory=list, max_length=64
    )

    @model_validator(mode="after")
    def _sections_are_writable_and_unique(self) -> "ReportSynthesisV3":
        ids = [section.section_id for section in self.sections]
        duplicates = sorted({sid for sid in ids if ids.count(sid) > 1})
        if duplicates:
            raise ValueError(f"section written more than once: {duplicates}")
        compiled = sorted(set(ids) & COMPILED_SECTION_IDS)
        if compiled:
            raise ValueError(
                f"section(s) {compiled} are compiled from the fact bundle and cannot be "
                "written here; the limitations and references a report carries are "
                "derived from what the build actually did"
            )
        return self

    @model_validator(mode="after")
    def _local_refs_are_unique(self) -> "ReportSynthesisV3":
        refs = [item.local_ref for item in (*self.conclusions, *self.recommendations)]
        duplicates = sorted({ref for ref in refs if refs.count(ref) > 1})
        if duplicates:
            raise ValueError(f"local_ref repeated within this synthesis: {duplicates}")
        return self

    @property
    def all_basis_fact_ids(self) -> tuple[str, ...]:
        ids: list[str] = []
        for section in self.sections:
            ids.extend(section.basis_fact_ids)
            ids.extend(section.referenced_fact_ids)
        for item in (*self.conclusions, *self.recommendations):
            ids.extend(item.basis_fact_ids)
        return tuple(dict.fromkeys(ids))
