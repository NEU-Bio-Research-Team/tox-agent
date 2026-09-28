"""Word-boundary phrase matching for the router (P1-9).

The router's term lists were matched with ``term in text``. That is fast, it is
one line, and it is wrong in a way that is invisible until it fires on a real
sentence:

* ``"execute"`` is a substring of **exec**utive. "Summarise the executive
  summary" routed to ``OUT_OF_SCOPE`` — the product refusing to answer a
  question about its own report section.
* ``"contribut"`` matches **contribut**ing, so "what are the contributing
  factors to uncertainty here" became an attribution request.
* ``"cite"`` is inside ex**cite**d and re**cite**; ``"study"`` inside
  **study**ing is fine, but ``"is it"`` is inside "th**is it**em".

Every one of those spends a provider request on the wrong workflow, and two of
them answer about the wrong thing.

This module matches **whole words and whole phrases**. Text is normalized and
split into word tokens, and a phrase matches only as a contiguous run of tokens.
Vietnamese is handled by the same code path: NFC normalization keeps diacritics
inside the word class, so ``tài liệu`` is two tokens and matches as two tokens.

A term list here therefore has to spell out the forms it means. That is a
feature: ``contributor``/``contributors``/``contribution`` are three decisions
someone made, not three accidents of a prefix.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Sequence

#: A word is a run of letters, digits and the marks that belong to them.
#: ``\w`` under Python's Unicode rules already covers Vietnamese once the text
#: is NFC-composed; the explicit class keeps underscores out, since a SMILES or
#: an identifier is not prose.
_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def normalize(text: str) -> str:
    """Case-folded, NFC-composed, whitespace-collapsed.

    NFC first: a decomposed ``tài`` (``a`` + combining grave) and a composed one
    are the same word to a reader and different strings to ``in``, and browsers
    send both.
    """
    composed = unicodedata.normalize("NFC", text or "")
    return " ".join(composed.casefold().split())


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(_WORD.findall(normalize(text)))


def phrase_tokens(phrase: str) -> tuple[str, ...]:
    return tokenize(phrase)


def phrase_positions(tokens: Sequence[str], phrase: str) -> tuple[int, ...]:
    """Start indices where ``phrase`` appears as a contiguous run of words."""
    needle = phrase_tokens(phrase)
    if not needle:
        return ()
    span = len(needle)
    return tuple(
        index
        for index in range(len(tokens) - span + 1)
        if tuple(tokens[index : index + span]) == needle
    )


def contains_phrase(tokens: Sequence[str], phrase: str) -> bool:
    """Whether ``phrase`` appears as a contiguous run of whole words."""
    return bool(phrase_positions(tokens, phrase))


#: Phrases that turn a request into a refusal of that request. Bounded to a
#: short window before the term, because "do not search the literature; the
#: prediction is enough" negates the search and "search the literature, I do
#: not trust the prediction" does not.
NEGATORS: tuple[str, ...] = (
    "do not", "don t", "dont", "without", "no need to", "skip", "avoid",
    "rather than", "instead of",
    "đừng", "không cần", "không phải", "thay vì", "bỏ qua",
)

#: How many tokens may sit between a negator and the term it negates.
NEGATION_WINDOW = 4


def _is_negated(tokens: Sequence[str], start: int, negators: Sequence[str]) -> bool:
    window_start = max(0, start - NEGATION_WINDOW)
    for negator in negators:
        for position in phrase_positions(tokens, negator):
            end = position + len(phrase_tokens(negator))
            if window_start <= position < start and end <= start:
                return True
    return False


def matched_terms(
    text: str, terms: Iterable[str], *, negators: Sequence[str] | None = NEGATORS
) -> tuple[str, ...]:
    """Every term in ``terms`` that appears in ``text`` as whole words.

    Returns the matches rather than a boolean so a routing decision can say
    *which* phrase moved it — a reason code with no evidence behind it is a
    guess that survived review.

    A term whose every occurrence sits just after a negator does not count.
    "Answer from the prediction only, do not search the literature" names
    literature and asks for the opposite; spending a provider request on it is
    both wrong and expensive. The window is short on purpose: negation at a
    distance is a language-model question, and guessing at it here would trade
    one class of false positive for another.
    """
    tokens = tokenize(text)
    if not tokens:
        return ()
    negators = tuple(negators or ())
    found: list[str] = []
    for term in terms:
        positions = phrase_positions(tokens, term)
        if not positions:
            continue
        if negators and all(_is_negated(tokens, start, negators) for start in positions):
            continue
        found.append(term)
    return tuple(found)


def mentions(
    text: str, terms: Iterable[str], *, negators: Sequence[str] | None = NEGATORS
) -> bool:
    return bool(matched_terms(text, terms, negators=negators))
