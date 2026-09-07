"""Prohibited wording (plan sections 2.2, 9.2, 16.5).

Deterministic pattern checks over the prose a candidate submits — the answer
markdown and every claim's text. These catch the specific substitutions the
plan calls out by name: a screening probability presented as a safety verdict,
hERG relabelled as clinical toxicity, applicability relabelled as a learned
in-distribution test, and a Tox21 assay count relabelled as a severity. They do
not attempt open-ended semantic correctness — that is explicitly the model
grader and SME's job (plan section 9.3) — only these named, testable patterns.
"""
from __future__ import annotations

import re
from typing import Iterable

from ..domain.errors import Violation
from .wire import ClaimCandidate

#: A verdict this product never issues (plan sections 3.5, 16.5 #9). Matches
#: "is safe", "considered safe", "an toàn", etc.; a bare mention of the word
#: "safe" inside a limitation explaining it is *not* a safety verdict is not
#: flagged, because those sentences pair it with a negation the pattern below
#: requires to be absent.
_SAFETY_VERDICT = re.compile(
    r"\b(is|are|considered|deemed|generally)\s+(safe|unsafe)\b"
    r"|\b(not\s+)?(an\s+toàn|độc\s+hại)\b"
    r"|\bregulatory[- ]ready\b|\bclinically\s+approved\b",
    re.IGNORECASE,
)

_AGGREGATE_VERDICT = re.compile(
    r"\boverall\s+(toxicity|risk|safety)\b|\baggregate\s+(score|risk|toxicity)\b"
    r"|\btotal\s+risk\b|\bcombined\s+(risk|toxicity)\s+score\b"
    r"|\bmức\s+độ\s+độc\s+tính\s+tổng\b",
    re.IGNORECASE,
)

_CLINICAL_OVERREACH = re.compile(
    r"\bclinical(?:[- ]trial)?\s+toxicity\b|\bclinically\s+toxic\b|\bnguy\s+cơ\s+lâm\s+sàng\b",
    re.IGNORECASE,
)

_HERG_LANGUAGE = re.compile(r"\bherg\b|\bcardiotox|\bchannel\s+block", re.IGNORECASE)

_IN_DISTRIBUTION = re.compile(
    r"\bin[- ]distribution\b|\bout[- ]of[- ]distribution\b|\bood\b(?!\w)", re.IGNORECASE
)

_MECHANISM_CLAIM = re.compile(
    r"\b(proves?|demonstrates?|is\s+evidence\s+of)\b[^.]{0,30}\bmechanism\b"
    r"|\bcausal(?:ly)?\s+(proof|evidence)\b",
    re.IGNORECASE,
)

_SEVERITY_FROM_COUNT = re.compile(
    r"\b\d+\s+(active\s+)?assays?\b[^.]{0,60}\b(severe|severity|more\s+toxic|highly\s+toxic|worse)\b"
    r"|\b(severity|how\s+toxic)\b[^.]{0,60}\bnumber\s+of\s+active\s+assays\b",
    re.IGNORECASE,
)

#: Negation cues that, unlike `_SAFETY_VERDICT`'s adjacency trick, sit
#: *before* a matched noun phrase rather than inside it — "does **not**
#: provide an overall toxicity score" still contains the literal substring
#: "overall toxicity". `_negated_before` treats a cue found shortly before
#: the match, and not separated from it by a sentence boundary, as the
#: phrase being denied rather than asserted (audit_5_9.md A-open/§4.7).
_NEGATION_CUE = re.compile(
    r"\b(not|no|never|without|lacks?|isn't|aren't|wasn't|weren't|doesn't|don't|"
    r"does\s+not|do\s+not|did\s+not|didn't|cannot|can't|couldn't|"
    r"none\s+of|no\s+such|not\s+provide[sd]?|không)\b",
    re.IGNORECASE,
)

_NEGATION_WINDOW_CHARS = 48


def _negated_before(text: str, start: int, window: int = _NEGATION_WINDOW_CHARS) -> bool:
    segment = text[max(0, start - window):start]
    boundary = max(segment.rfind("."), segment.rfind("\n"))
    if boundary != -1:
        segment = segment[boundary + 1:]
    return bool(_NEGATION_CUE.search(segment))


def _scan(pattern: re.Pattern, text: str, code: str, message: str, path: str) -> list[Violation]:
    if pattern.search(text):
        return [Violation(code, message, path=path)]
    return []


def matches_unnegated(pattern: re.Pattern, text: str) -> bool:
    """Whether ``pattern`` matches somewhere in ``text`` that a negation cue
    does not immediately precede. Public (not underscore-prefixed) because
    ``evals/graders/hard_gates.py`` reuses these exact patterns to keep its
    hard gates from drifting away from what the validator enforces (plan
    section 16.5) — a caller outside this module needs the same
    negation-awareness `_scan_unless_negated` gives `validate_claim_wording`,
    or it re-flags the same false positives audit_5_9.md's §4.7 fix already
    closed here.
    """
    return any(not _negated_before(text, match.start()) for match in pattern.finditer(text))


def _scan_unless_negated(
    pattern: re.Pattern, text: str, code: str, message: str, path: str
) -> list[Violation]:
    """Like `_scan`, but a match preceded by a negation cue is not a
    violation — the sentence is denying the prohibited claim, not making it.
    """
    if matches_unnegated(pattern, text):
        return [Violation(code, message, path=path)]
    return []


def validate_answer_markdown(answer_markdown: str) -> list[Violation]:
    violations: list[Violation] = []
    violations += _scan(
        _SAFETY_VERDICT, answer_markdown, "safety_verdict_out_of_scope",
        "the answer states a safety verdict this product does not issue", "answer_markdown",
    )
    violations += _scan_unless_negated(
        _AGGREGATE_VERDICT, answer_markdown, "aggregate_verdict_present",
        "the answer states an aggregate toxicity/risk score, which does not exist in this product",
        "answer_markdown",
    )
    return violations


def validate_claim_wording(claim: ClaimCandidate) -> list[Violation]:
    violations: list[Violation] = []
    text = claim.text
    path = f"claims[{claim.claim_id}].text"
    field = claim.field_path or ""

    violations += _scan(
        _SAFETY_VERDICT, text, "safety_verdict_out_of_scope",
        "this claim states a safety verdict this product does not issue", path,
    )
    violations += _scan_unless_negated(
        _AGGREGATE_VERDICT, text, "aggregate_verdict_present",
        "this claim states an aggregate score, which does not exist in this product", path,
    )

    if field.startswith("predictions.herg"):
        violations += _scan_unless_negated(
            _CLINICAL_OVERREACH, text, "endpoint_substitution_language",
            "an hERG claim describes clinical-trial toxicity; hERG blockade and clinical "
            "toxicity are different measurements (SCI-01, SCI-04)",
            path,
        )
    if field.startswith("predictions.clintox") and _HERG_LANGUAGE.search(text):
        violations.append(
            Violation(
                "endpoint_substitution_language",
                "a ClinTox claim describes hERG/cardiotoxicity; they are different measurements "
                "(SCI-01)",
                path=path,
            )
        )
    if field.startswith("applicability") and _IN_DISTRIBUTION.search(text):
        violations.append(
            Violation(
                "applicability_overinterpreted",
                "applicability is a rule-based element check, not a learned in/out-of-distribution "
                "test (SCI-07)",
                path=path,
            )
        )
    if "attribution" in field and _MECHANISM_CLAIM.search(text):
        violations.append(
            Violation(
                "attribution_overinterpreted",
                "attribution explains what moved the model's score; it is not proof of a "
                "chemical mechanism (SCI-09)",
                path=path,
            )
        )
    return violations


def validate_no_hitcount_severity(claims: Iterable[ClaimCandidate], answer_markdown: str) -> list[Violation]:
    """SCI-05: Tox21 assays are independent; a hit count is not a severity."""
    violations: list[Violation] = []
    if _SEVERITY_FROM_COUNT.search(answer_markdown):
        violations.append(
            Violation(
                "hitcount_as_severity",
                "the answer treats a Tox21 active-assay count as a severity measure",
                path="answer_markdown",
            )
        )
    for claim in claims:
        if _SEVERITY_FROM_COUNT.search(claim.text):
            violations.append(
                Violation(
                    "hitcount_as_severity",
                    "this claim treats a Tox21 active-assay count as a severity measure",
                    path=f"claims[{claim.claim_id}].text",
                )
            )
    return violations
