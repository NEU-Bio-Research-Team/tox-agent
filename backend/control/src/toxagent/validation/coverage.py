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
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Iterable

from ..domain.errors import Violation
from .wire import ClaimCandidate

#: A probability/percentage-shaped number embedded in free text: a decimal
#: point or Vietnamese comma is required, or a trailing '%' — this is what
#: keeps plain prose integers ("2 lần", "bước 3", "candidate 1/2") from being
#: misread as an unclaimed prediction. Deliberately narrower than
#: ``numeric._CANONICAL_NUMBER``, which matches a whole, already-isolated
#: token; this instead has to find one embedded inside a sentence.
#:
#: Two exclusions keep *version strings* out. A report's provenance appendix is
#: required to name the predictor version and the renderer that produced the
#: document, so ``0.1.0.dev0`` and ``weasyprint-66.0`` are content that section
#: cannot be written without — and no claim will ever carry ``rendered_value ==
#: "0.1"``, so treating them as unclaimed predictions made a required section
#: impossible to write. A genuine bare probability there is still caught.
#:
#: - ``(?<![\w]-)`` — not the tail of a hyphenated identifier, so the ``66.0`` in
#:   ``weasyprint-66.0`` is part of a name rather than a measurement. A real
#:   negative number is unaffected: the ``-`` in ``" -0.5"`` has no word
#:   character before it.
#: - ``(?![.,]\d)`` — not one component of a dotted version, so ``0.1.0.dev0``
#:   yields nothing rather than yielding ``0.1``.
_NUMERIC_TOKEN = re.compile(
    r"(?<![\w.,])(?<![\w]-)-?\d+(?:[.,]\d+%?|%)(?![\w])(?![.,]\d)"
)

_MARKDOWN_LINK = re.compile(r"\[[^\]\n]*\]\(\s*\S+\s*\)")
_BARE_URL = re.compile(r"\bhttps?://\S+", re.IGNORECASE)


def faithful_rendering(token: str, value: float) -> bool:
    """Whether ``token`` is ``value`` written to the token's own precision.

    ``"0.800"`` renders 0.7999394536 faithfully, and so does ``"80%"``,
    ``"0,80"`` or the full ``"0.7999394536018372"`` a tool printed; ``"0.81"``
    does not. A token that rounds a non-zero value to zero is never faithful:
    "0%" for an inactive assay's 0.004 is exactly the reading
    ``numeric-11-inactive-assay-not-zero`` exists to refuse.
    """
    text = token.strip()
    percent = text.endswith("%")
    body = (text[:-1] if percent else text).replace(",", ".")
    try:
        written = Decimal(body)
        source = Decimal(repr(float(value)))
    except (InvalidOperation, ValueError):
        return False
    if percent:
        source *= 100
    if written == 0:
        return source == 0
    exponent = written.as_tuple().exponent
    quantum = Decimal(1).scaleb(exponent if isinstance(exponent, int) and exponent < 0 else 0)
    return any(
        source.quantize(quantum, rounding=mode) == written
        for mode in (ROUND_HALF_UP, ROUND_HALF_EVEN)
    )


def validate_markdown_numeric_coverage(
    answer_markdown: str, claims: tuple[ClaimCandidate, ...], *,
    claimed_values: Iterable[float] = (),
) -> list[Violation]:
    """``claimed_values`` are the values of the draft's server-resolved claims
    (grounded-answer v2 only). There the server, not the model, renders each
    claim, so the model cannot know the exact string to repeat in its prose; a
    prose number that faithfully renders one of those values still comes from a
    claim, which is what this check exists to enforce. The v1 and report paths
    pass nothing and keep the exact-string rule."""
    rendered_values = {
        claim.rendered_value for claim in claims if claim.rendered_value
    }
    values = tuple(claimed_values)
    violations: list[Violation] = []
    seen: set[str] = set()
    for match in _NUMERIC_TOKEN.finditer(answer_markdown):
        token = match.group(0)
        if token in rendered_values or token in seen:
            continue
        if any(faithful_rendering(token, value) for value in values):
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
