"""Cross-section semantic gates (P0-2).

The audit's live report passed every schema gate it had and shipped anyway with
an executive summary stating the explanation "reports no mapped positive or
negative contributors and no unmapped mass" — while the explanation package in
the same artifact carried three negative contributors and 0.3584 of unmapped
importance, and the very next section described them correctly. A second
contradiction in the same document: a limitation saying the literature search
covered one provider and did not read full texts, in a build that had
``include_external_evidence=false``, no references at all, and an evidence
section saying no search was performed.

Those are not formatting defects. A reader who stops at the executive summary
takes away the opposite of what the data says, and that is the most dangerous
failure mode a scientific document has.

**The real fix is structural, and it is being built around this module.** Facts
that appear in more than one section must be compiled from one
``ExplanationCoverage`` / fact-bundle entry and referenced by id, never
restated in prose section by section (ADR 0009). ``canonical_explanation_line``
here is that single source of wording.

**This module is the gate that holds until every section is compiled.** It does
not try to understand prose. It checks a small, closed set of *denials*: a
sentence that asserts the absence of a fact the server knows is present. That
is a much narrower question than "does this paragraph agree with the data", and
it is the exact shape both audit contradictions took. Where a denial is true —
the explanation really has no contributors, the search really did not run — no
violation is raised, because then the sentence is correct.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from ..application.xai_coverage import ExplanationCoverage
from ..domain.report import ExplanationPackage
from ..domain.errors import Violation

# --- the four evidence states, which are not interchangeable ----------------

#: No search was asked for. The build set ``include_external_evidence=false``.
EVIDENCE_NOT_REQUESTED = "not_requested"
#: A search ran and returned nothing.
EVIDENCE_ZERO_RESULTS = "zero_results"
#: A search was attempted and the provider could not answer.
EVIDENCE_PROVIDER_FAILED = "provider_failed"
#: A search ran, returned candidates, and none of them was relevant enough to
#: cite.
EVIDENCE_INSUFFICIENT = "insufficient"

EVIDENCE_STATES = (
    EVIDENCE_NOT_REQUESTED,
    EVIDENCE_ZERO_RESULTS,
    EVIDENCE_PROVIDER_FAILED,
    EVIDENCE_INSUFFICIENT,
)


# --- denial patterns --------------------------------------------------------
#
# Each pattern matches a sentence asserting that something is *absent*. They are
# deliberately not patterns for "describes contributors correctly": recognising
# a correct description is open-ended, recognising a flat denial is not.

_NO_CONTRIBUTORS = (
    # "reports no mapped positive or negative contributors"
    re.compile(
        r"\bno\s+(?:mapped\s+|significant\s+|atom[- ]level\s+)*"
        r"(?:positive\s+(?:or|and)\s+negative\s+|positive\s+|negative\s+)?"
        r"(?:atom\s+)?contributor",
        re.IGNORECASE,
    ),
    re.compile(r"\bcontributors?\s*:?\s*none\b", re.IGNORECASE),
    # Vietnamese: "không có đóng góp nào", "không có nguyên tử đóng góp"
    re.compile(r"không\s+có\s+[^.;]{0,40}?đóng\s+góp", re.IGNORECASE),
)

_NO_UNMAPPED = (
    re.compile(
        r"\bno\s+unmapped\s+(?:importance|mass|attribution|contribution)", re.IGNORECASE
    ),
    re.compile(
        r"\bunmapped\s+(?:importance|mass)\s+(?:is|of)\s+(?:zero|0(?:\.0+)?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:all|entire|100\s*%)\s+of\s+the\s+attribution\s+(?:mass\s+)?"
        r"(?:is\s+|was\s+)?map(?:ped|s)",
        re.IGNORECASE,
    ),
    re.compile(r"không\s+có\s+[^.;]{0,40}?chưa\s+ánh\s+xạ", re.IGNORECASE),
)

#: A sentence claiming a literature search actually happened.
_SEARCH_PERFORMED = (
    re.compile(
        r"\b(?:the\s+)?(?:literature\s+|evidence\s+)?search(?:es)?\s+"
        r"(?:covered|was\s+(?:limited\s+to|restricted\s+to|performed|run|conducted)|"
        r"included|returned|retrieved|queried)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:we|this\s+report|the\s+build)\s+(?:searched|queried)\b", re.IGNORECASE
    ),
    re.compile(
        r"\b(?:abstracts?\s+only|full\s+texts?\s+were\s+not\s+(?:read|retrieved))",
        re.IGNORECASE,
    ),
    re.compile(r"(?:đã\s+)?tìm\s+kiếm\s+(?:tài\s+liệu|y\s+văn)", re.IGNORECASE),
)


def _any_match(patterns: Iterable[re.Pattern[str]], text: str) -> str | None:
    for pattern in patterns:
        found = pattern.search(text)
        if found:
            return found.group(0)
    return None


# --- the single source of explanation wording -------------------------------


def canonical_explanation_line(coverage: ExplanationCoverage) -> str:
    """The one sentence about coverage that every section must reuse.

    Exposed here rather than only on the dataclass because this is the seam the
    compiler writes through: a section that needs to state coverage takes this
    string, and a section that writes its own is what the gates below catch.
    """
    return coverage.summary_sentence()


@dataclass(frozen=True, slots=True)
class EvidenceSituation:
    """What actually happened to external evidence in this build.

    Server-known, every field. A draft cannot supply any of it, which is what
    makes the limitation check below a check rather than a restatement.
    """

    requested: bool
    search_performed: bool
    candidates_found: int = 0
    provider_failed: bool = False
    promoted: int = 0

    @property
    def searched(self) -> bool:
        """Whether a search demonstrably happened.

        Promoted records and candidates are proof on their own: a caller that
        has not yet been taught to pass ``search_performed`` must not make a
        report that cites literature look like one that never looked.
        """
        return self.search_performed or self.promoted > 0 or self.candidates_found > 0

    @property
    def found(self) -> int:
        """Candidates the search turned up. A promoted record is one by
        definition, so a caller that only knows what it cited still lands in
        the right state."""
        return max(self.candidates_found, self.promoted)

    @property
    def state(self) -> str:
        if not self.requested:
            return EVIDENCE_NOT_REQUESTED
        if self.provider_failed:
            return EVIDENCE_PROVIDER_FAILED
        if not self.searched or self.found == 0:
            return EVIDENCE_ZERO_RESULTS
        if self.promoted == 0:
            return EVIDENCE_INSUFFICIENT
        return "cited"


# --- gates ------------------------------------------------------------------


def check_explanation_consistency(
    *,
    sections: Sequence[object],
    explanations: Mapping[str, ExplanationPackage],
) -> list[Violation]:
    """No section may deny an explanation fact the server holds.

    A denial is flagged only when *no* explanation in the build makes it true.
    A report covering several endpoints may legitimately say one of them has no
    contributors, and that sentence is correct for that endpoint.
    """
    packages = list(explanations.values())
    if not packages:
        return []

    contributor_counts = [pkg.highlights.contributor_count for pkg in packages]
    unmapped_values = [
        pkg.highlights.unmapped_importance or 0.0
        for pkg in packages
        if pkg.highlights.unmapped_importance is not None
    ]
    every_explanation_has_contributors = all(count > 0 for count in contributor_counts)
    every_explanation_has_unmapped_mass = bool(unmapped_values) and all(
        value > 0 for value in unmapped_values
    )

    violations: list[Violation] = []
    for section in sections:
        body = getattr(section, "body_markdown", "") or ""
        section_id = getattr(section, "section_id", "?")
        if not body:
            continue

        if every_explanation_has_contributors:
            phrase = _any_match(_NO_CONTRIBUTORS, body)
            if phrase:
                violations.append(
                    Violation(
                        "explanation_summary_contradicted",
                        f"section {section_id!r} says {phrase!r}, but every explanation in "
                        f"this build reports contributors "
                        f"({', '.join(str(c) for c in contributor_counts)})",
                        path=f"sections[{section_id}].body_markdown",
                        expected=contributor_counts,
                        actual=phrase,
                    )
                )

        if every_explanation_has_unmapped_mass:
            phrase = _any_match(_NO_UNMAPPED, body)
            if phrase:
                violations.append(
                    Violation(
                        "explanation_summary_contradicted",
                        f"section {section_id!r} says {phrase!r}, but every explanation in "
                        f"this build leaves attribution mass unmapped "
                        f"({', '.join(f'{v:.4f}' for v in unmapped_values)})",
                        path=f"sections[{section_id}].body_markdown",
                        expected=[round(v, 6) for v in unmapped_values],
                        actual=phrase,
                    )
                )
    return violations


def check_evidence_scope_consistency(
    *,
    sections: Sequence[object],
    limitations: Sequence[object],
    situation: EvidenceSituation,
) -> list[Violation]:
    """Nothing may describe a search that did not happen.

    The audit's report carried ``evidence_scope_limited`` worded as "the search
    covered one provider and did not read full texts" in a build that never
    searched. The limitation code itself is not the problem — a report with no
    external evidence has a genuinely limited evidence scope — the wording is:
    it describes a method that was not used.
    """
    if situation.searched:
        return []

    violations: list[Violation] = []
    for index, limitation in enumerate(limitations):
        text = getattr(limitation, "text", "") or ""
        phrase = _any_match(_SEARCH_PERFORMED, text)
        if phrase:
            violations.append(
                Violation(
                    "evidence_scope_limitation_without_search",
                    f"limitation {getattr(limitation, 'code', '?')!r} says {phrase!r}, but "
                    f"this build performed no literature search "
                    f"(evidence state: {situation.state})",
                    path=f"limitations[{index}].text",
                    expected=situation.state,
                    actual=phrase,
                )
            )

    for section in sections:
        if getattr(section, "section_id", None) not in {"external_evidence", "limitations"}:
            continue
        body = getattr(section, "body_markdown", "") or ""
        phrase = _any_match(_SEARCH_PERFORMED, body)
        if phrase:
            violations.append(
                Violation(
                    "evidence_scope_limitation_without_search",
                    f"section {getattr(section, 'section_id', '?')!r} says {phrase!r}, but "
                    f"this build performed no literature search "
                    f"(evidence state: {situation.state})",
                    path=f"sections[{getattr(section, 'section_id', '?')}].body_markdown",
                    expected=situation.state,
                    actual=phrase,
                )
            )
    return violations


def check_evidence_state_gap(
    *, gaps: Sequence[object], situation: EvidenceSituation
) -> list[Violation]:
    """The declared gap names the state the build was actually in.

    "No search was requested", "the search found nothing", "the provider was
    down" and "nothing found was relevant enough to cite" are four different
    facts. Collapsing them into one reason tells a reader the evidence base is
    thin when it may be that nobody looked.
    """
    expected = {
        EVIDENCE_NOT_REQUESTED: "external_evidence_not_requested",
        EVIDENCE_ZERO_RESULTS: "no_relevant_evidence",
        EVIDENCE_PROVIDER_FAILED: "provider_unavailable",
        EVIDENCE_INSUFFICIENT: "no_relevant_evidence",
    }.get(situation.state)
    if expected is None:
        return []

    evidence_gaps = [
        gap
        for gap in gaps
        if getattr(gap, "section_id", None) == "external_evidence"
        or getattr(gap, "reason", None)
        in {"no_relevant_evidence", "provider_unavailable", "external_evidence_not_requested"}
    ]
    if not evidence_gaps:
        return [
            Violation(
                "evidence_state_not_disclosed",
                f"this build's external evidence state is {situation.state!r} and no gap "
                "says so, so a reader cannot tell an empty result from a search that "
                "never ran",
                path="gaps",
                expected=expected,
            )
        ]

    violations: list[Violation] = []
    for gap in evidence_gaps:
        reason = getattr(gap, "reason", None)
        if reason != expected:
            violations.append(
                Violation(
                    "evidence_state_mislabelled",
                    f"gap {getattr(gap, 'gap_id', '?')!r} reports {reason!r}, but this "
                    f"build's evidence state is {situation.state!r}",
                    path=f"gaps[{getattr(gap, 'gap_id', '?')}].reason",
                    expected=expected,
                    actual=reason,
                )
            )
    return violations
