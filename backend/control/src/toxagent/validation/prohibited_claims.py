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

from ..domain.development_posture import DevelopmentPosture
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

#: RC-07/W6: a development posture (proceed/hold/deprioritize) must never be
#: dressed up as a safety verdict. `_SAFETY_VERDICT` already catches the
#: blatant form ("is/are/considered/deemed/generally safe", or any bare
#: mention of Vietnamese "an toàn") — but a posture word paired with "safe"
#: through a *different* verb ("appears safe to proceed", "safely advance")
#: or the adverb form ("safely") slips past it. This is additive to
#: `_SAFETY_VERDICT`, not a replacement: it exists only to catch a safety
#: word sitting near development/posture language, in either order, within
#: one sentence.
_POSTURE_WORD = (
    r"(?:proceed(?:ing)?|develop(?:ment|ing)?|advanc(?:e|ing)|continu(?:e|ing)|"
    r"tiếp\s+tục|phát\s+triển)"
)
_SAFETY_POSTURE_CONFLATION = re.compile(
    rf"\b(safe|unsafe|safely|an\s+toàn)\b[^.\n]{{0,60}}\b{_POSTURE_WORD}\b"
    rf"|\b{_POSTURE_WORD}\b[^.\n]{{0,60}}\b(safe|unsafe|safely|an\s+toàn)\b",
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
    r"none\s+of|no\s+such|not\s+provide[sd]?|không|"
    # "separate measurements rather than an aggregate score" (W9-02b).
    r"rather\s+than|instead\s+of|thay\s+vì)\b",
    re.IGNORECASE,
)

#: A negation that follows the phrase it denies: "an overall toxicity score
#: cannot be provided", "một điểm độc tính tổng hợp không có" (W9-02b). Up to
#: three words may sit between them ("score", "is", "for this molecule").
_NEGATED_AFTER = re.compile(
    r"^\W*(?:[\w-]+\s+){0,3}?(?:cannot|can't|could\s+not|is\s+not|isn't|are\s+not|"
    r"aren't|was\s+not|wasn't|does\s+not|doesn't|do\s+not|don't|will\s+not|won't|"
    r"không|chưa)\b",
    re.IGNORECASE,
)

#: W9-02b: the first measurement of these gates with the flagged sentence
#: stored (W9-A) found almost every flag was a denial whose cue sat more than
#: 48 characters back ("… the stored status is ok, which does not mean the
#: compound is unsafe"). The window is wider, and a clause boundary now ends it
#: as well as a sentence boundary, so "not toxic, but the compound is safe"
#: is still caught.
_NEGATION_WINDOW_CHARS = 80
_CLAUSE_BOUNDARY = re.compile(r"[.;\n]|\b(?:but|however|yet|nhưng|tuy\s+nhiên|song)\b",
                              re.IGNORECASE)


def _same_clause_before(text: str, start: int, window: int) -> str:
    """The text before ``start`` back to the nearest clause boundary."""
    segment = text[max(0, start - window):start]
    last = None
    for last in _CLAUSE_BOUNDARY.finditer(segment):  # noqa: B007 - keeps the last match
        pass
    return segment[last.end():] if last is not None else segment

#: Cues that say the *product* is declining to make the verdict, not that a
#: verdict is being made about something negative (P1-5).
#:
#: `_SAFETY_VERDICT` is deliberately not run through the general
#: `_NEGATION_CUE` list: that list contains bare "no" and "not", and "there is
#: no doubt the compound is safe" would pass a gate whose whole job is to
#: refuse that sentence. These cues are narrower — each one states that no
#: conclusion is being offered — so the sentences they admit are the product
#: describing its own limits, which is what it is supposed to say.
#:
#: The audit's case: "Mô hình không đưa ra kết luận an toàn cho người" was
#: refused as a safety verdict when it is the opposite of one.
_DECLINES_TO_ASSERT = re.compile(
    r"\b(cannot|can't|could\s+not|couldn't|does\s+not|doesn't|do\s+not|don't|did\s+not|"
    r"will\s+not|won't|is\s+not\s+able\s+to|are\s+not\s+able\s+to)\s+(be\s+)?(used\s+to\s+)?"
    r"(say|state|tell|conclude|determine|assert|claim|establish|provide|issue|make|"
    # W9-02b: "does not mean/imply/show …" declines the reading that follows.
    r"mean|imply|indicate|show|prove|demonstrate|infer|confirm|support)\b"
    r"|\bmakes?\s+no\s+(claim|statement|assertion|verdict)\b"
    r"|\bno\s+(such\s+)?(verdict|conclusion|claim)\b"
    # "not a determination / finding / conclusion that …", "not evidence that".
    # Up to three words may qualify the noun: "not a clinical diagnosis".
    r"|\bnot\s+(?:(?:a|an|the)\s+)?(?:[\w-]+\s+){0,3}?(?:determination|finding|conclusion|"
    r"verdict|claim|assessment|statement|confirmation|proof|diagnosis)\b"
    r"|\bnot\s+(?:as\s+)?evidence\s+(?:that|of|for)\b"
    r"|\bkhông\s+(?:tự\s+(?:nó\s+|chúng\s+)?)?(đưa\s+ra|kết\s+luận|khẳng\s+định|tuyên\s+bố|nói|chứng\s+minh|"
    r"xác\s+nhận|suy\s+ra|thay\s+thế|đảm\s+bảo|cho\s+biết)\b"
    r"|\b(không|chưa)\s+(đủ\s+để|thể)\s+(kết\s+luận|suy\s+ra|khẳng\s+định|xác\s+nhận|"
    r"chứng\s+minh|đánh\s+giá)\b"
    # "không phải nguy cơ lâm sàng hay kết luận an toàn": a few words may come
    # first, but the noun that makes it a declined conclusion must be there.
    r"|\b(không|chưa)\s+phải\s+(?:là\s+)?(?:[\w-]+\s+){0,6}?(kết\s+luận|tuyên\s+bố|"
    r"bằng\s+chứng|đánh\s+giá|xác\s+nhận|phán\s+quyết|nhận\s+định)\b"
    # Live, 2026-09-26: "nó không tự nó là phán quyết an toàn", "chưa nên diễn
    # giải đây là kết luận về an toàn".
    r"|\bkhông\s+(?:tự\s+(?:nó\s+|chúng\s+)?)?là\s+(?:một\s+)?(phán\s+quyết|kết\s+luận|"
    r"bằng\s+chứng|đánh\s+giá|khẳng\s+định)\b"
    r"|\b(không|chưa)\s+nên\s+(diễn\s+giải|hiểu|xem|coi)\b",
    re.IGNORECASE,
)


#: Vietnamese nouns that make "an toàn" part of a noun phrase — "dữ liệu an
#: toàn" (safety data), "đánh giá an toàn" (a safety assessment), "biên an
#: toàn" (a safety margin) — rather than a predicate about the compound. A
#: closed list, from the flagged sentences of W9-B. "độ/tính/mức độ an toàn" are
#: deliberately absent: "hợp chất có độ an toàn cao" is a verdict. So are
#: "nghiên cứu" and "thử nghiệm", which read as verbs as often as nouns.
_SAFETY_NOUN_HEAD = re.compile(
    r"(?:dữ\s+liệu|đánh\s+giá|hồ\s+sơ|bối\s+cảnh|biên|ngưỡng|thông\s+tin|"
    r"dược\s+lý|khía\s+cạnh|vấn\s+đề)\s+$",
    re.IGNORECASE,
)

#: A marker that turns the head into a verb: "được đánh giá an toàn" is
#: "assessed as safe", a verdict, not "a safety assessment".
_VERBAL_MARKER = re.compile(
    r"\b(?:được|bị|vẫn|đều|cũng)\s+$", re.IGNORECASE
)

#: What follows "an toàn" when it is the predicate after all: "hồ sơ an toàn
#: tốt", "dữ liệu an toàn cho người" — the phrase is being judged, not named.
_SAFETY_PREDICATE_AFTER = re.compile(
    r"^\s+(?:cho|với|khi|đối\s+với|tốt|thuận\s+lợi|chấp\s+nhận\s+được|đảm\s+bảo)\b",
    re.IGNORECASE,
)


def _safety_noun_phrase(text: str, match: re.Match) -> bool:
    """Whether a matched "an toàn" is part of a noun phrase (W9-B)."""
    if not match.group(0).lower().startswith("an"):
        return False
    before = text[max(0, match.start() - 40):match.start()]
    head = _SAFETY_NOUN_HEAD.search(before)
    if head is None or _VERBAL_MARKER.search(before[:head.start()]):
        return False
    return not _SAFETY_PREDICATE_AFTER.match(text[match.end():match.end() + 24])


def _normalized(text: str) -> str:
    """Typographic apostrophes as ASCII, so "can’t" negates like "can't".
    Same length, so match offsets still point into the original text."""
    return text.replace("\u2019", "'").replace("\u2018", "'")


def _declines_before(text: str, start: int, window: int = _NEGATION_WINDOW_CHARS) -> bool:
    """Whether the product declines to assert, within the same clause."""
    return bool(_DECLINES_TO_ASSERT.search(_same_clause_before(text, start, window)))


#: Characters of context kept on each side of a flagged phrase.
_EXCERPT_CONTEXT = 40


def _excerpt(text: str, match: re.Match) -> str:
    """The flagged phrase with a little of its sentence, for ``Violation.actual``.

    Returned to the model so its one correction rewrites the sentence that was
    flagged rather than guessing, and stored with the rejection event so a gate's
    false-positive rate can be measured from real drafts (W9-02: before this, a
    rejected draft's wording was not recoverable from anything the run kept).
    """
    start = max(0, match.start() - _EXCERPT_CONTEXT)
    end = min(len(text), match.end() + _EXCERPT_CONTEXT)
    return text[start:end].strip()


def _scan_unless_declined(
    pattern: re.Pattern, text: str, code: str, message: str, path: str
) -> list[Violation]:
    norm = _normalized(text)
    for match in pattern.finditer(norm):
        if _safety_noun_phrase(norm, match):
            continue
        if not _declines_before(norm, match.start()):
            return [Violation(code, message, path=path, actual=_excerpt(text, match))]
    return []


def _negated_before(text: str, start: int, window: int = _NEGATION_WINDOW_CHARS) -> bool:
    return bool(_NEGATION_CUE.search(_same_clause_before(text, start, window)))


def _negated_after(text: str, end: int, window: int = 40) -> bool:
    segment = text[end:end + window]
    cut = _CLAUSE_BOUNDARY.search(segment)
    return bool(_NEGATED_AFTER.search(segment[:cut.start()] if cut else segment))


def _negated(text: str, match: re.Match) -> bool:
    return _negated_before(text, match.start()) or _negated_after(text, match.end())


def _scan(pattern: re.Pattern, text: str, code: str, message: str, path: str) -> list[Violation]:
    match = pattern.search(text)
    if match:
        return [Violation(code, message, path=path, actual=_excerpt(text, match))]
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
    norm = _normalized(text)
    return any(not _negated(norm, match) for match in pattern.finditer(norm))


def _scan_unless_negated(
    pattern: re.Pattern, text: str, code: str, message: str, path: str
) -> list[Violation]:
    """Like `_scan`, but a match preceded by a negation cue is not a
    violation — the sentence is denying the prohibited claim, not making it.
    """
    norm = _normalized(text)
    for match in pattern.finditer(norm):
        if not _negated(norm, match):
            return [Violation(code, message, path=path, actual=_excerpt(text, match))]
    return []


def validate_answer_markdown(answer_markdown: str) -> list[Violation]:
    violations: list[Violation] = []
    violations += _scan_unless_declined(
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

    violations += _scan_unless_declined(
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


def validate_posture_wording(
    answer_markdown: str, posture: DevelopmentPosture
) -> list[Violation]:
    """W6/RC-07: scans the answer markdown plus the posture's own rationale
    and conditions for safety/posture conflation — reuses
    `_scan_unless_negated` so a hedged/negated sentence ("not proceeding
    because it cannot be confirmed safe") is not flagged, the same
    negation-awareness every other gate in this module gets."""
    violations: list[Violation] = []
    violations += _scan_unless_negated(
        _SAFETY_POSTURE_CONFLATION, answer_markdown, "safety_posture_conflation",
        f"the answer pairs a safety word with a {posture.value.value} recommendation — a "
        "development posture is never a safety verdict",
        "answer_markdown",
    )
    violations += _scan_unless_negated(
        _SAFETY_POSTURE_CONFLATION, posture.rationale, "safety_posture_conflation",
        f"the posture's rationale pairs a safety word with its {posture.value.value} "
        "recommendation — a development posture is never a safety verdict",
        "development_posture.rationale",
    )
    for index, condition in enumerate(posture.conditions):
        violations += _scan_unless_negated(
            _SAFETY_POSTURE_CONFLATION, condition, "safety_posture_conflation",
            f"a posture condition pairs a safety word with its {posture.value.value} "
            "recommendation — a development posture is never a safety verdict",
            f"development_posture.conditions[{index}]",
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
