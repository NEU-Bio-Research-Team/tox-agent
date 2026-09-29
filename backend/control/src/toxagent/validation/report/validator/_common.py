"""The validation context and result, and the patterns and tables several checks share."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Final, Mapping, Sequence

from ....domain.answer import LimitationCode
from ....domain.errors import Violation
from ....domain.evidence import EvidenceRecord
from ....domain.observation import Observation
from ....domain.report import (
    ExplanationPackage,
    ReportFigure,
    SourceClass,
)
from ..draft_wire import ReportDraftCandidate

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
