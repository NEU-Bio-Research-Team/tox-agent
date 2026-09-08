"""Coverage between the prose and the claims that back it (plan section 9).

Two checks live here, both closing the same hole: a candidate whose
``answer_markdown`` reads as grounded and cited without its ``claims``/
``citation_ids`` actually saying so. Neither is caught elsewhere — the rest of
the validator only inspects ``claims``, never the prose a user actually reads.

* A number in the prose that looks like a predictor value (a decimal or a
  percentage) must equal some claim's ``rendered_value`` verbatim. An empty
  ``claims`` list is not, on its own, a violation anywhere else in this
  module; without this check ``"The hERG probability is 99.99%."`` with
  ``claims=[]`` passes validation outright.
* A raw hyperlink in the prose is rejected outright. This product has no
  sanctioned way for a model to embed a citation in text — evidence flows
  through ``claim.citation_ids`` and a resolved ``EvidenceRecord`` only — so a
  self-authored URL is fabricated provenance, not a shortcut around one.
"""
from __future__ import annotations

import re

from ..domain.errors import Violation
from .wire import ClaimCandidate

#: A probability/percentage-shaped number embedded in free text: a decimal
#: point or Vietnamese comma is required, or a trailing '%' — this is what
#: keeps plain prose integers ("2 lần", "bước 3", "candidate 1/2") from being
#: misread as an unclaimed prediction. Deliberately narrower than
#: ``numeric._CANONICAL_NUMBER``, which matches a whole, already-isolated
#: token; this instead has to find one embedded inside a sentence.
_NUMERIC_TOKEN = re.compile(r"(?<![\w.,])-?\d+(?:[.,]\d+%?|%)(?![\w])")

_MARKDOWN_LINK = re.compile(r"\[[^\]\n]*\]\(\s*\S+\s*\)")
_BARE_URL = re.compile(r"\bhttps?://\S+", re.IGNORECASE)


def validate_markdown_numeric_coverage(
    answer_markdown: str, claims: tuple[ClaimCandidate, ...]
) -> list[Violation]:
    rendered_values = {
        claim.rendered_value for claim in claims if claim.rendered_value
    }
    violations: list[Violation] = []
    seen: set[str] = set()
    for match in _NUMERIC_TOKEN.finditer(answer_markdown):
        token = match.group(0)
        if token in rendered_values or token in seen:
            continue
        seen.add(token)
        violations.append(
            Violation(
                "unclaimed_numeric_value",
                f"{token!r} appears in answer_markdown but no claim's rendered_value equals it "
                "— every predictor-derived number in the prose must come from a claim",
                path="answer_markdown",
                actual=token,
            )
        )
    return violations


def validate_no_uncited_links(answer_markdown: str) -> list[Violation]:
    if _MARKDOWN_LINK.search(answer_markdown) or _BARE_URL.search(answer_markdown):
        return [
            Violation(
                "raw_link_in_answer_markdown",
                "answer_markdown contains a hyperlink; citations must go through a claim's "
                "citation_ids and a resolved evidence record, never a link written into the prose",
                path="answer_markdown",
            )
        ]
    return []
