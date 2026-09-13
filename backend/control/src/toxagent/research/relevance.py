"""Whether a search hit is about this compound and this endpoint (P1-2).

The audit asked for at most two papers on ethanol and hERG. The run persisted
five as durable, ``accepted`` evidence — alpha-asaronol and asthma, cannabinoids
and QT, a breast-cancer benzimidazole, remdesivir pharmacokinetics, a
brain–heart review — and then cited none of them, because the model reading them
could tell they were irrelevant. The database could not.

``accepted`` meant the provider payload parsed and its host was on the
allowlist. That is a statement about bytes. It is not a statement about
subject matter, and the two had been sharing one durable state.

This module adds the missing judgement, deterministically:

* **direct** — the paper is about this compound *and* this endpoint. Citable.
* **contextual** — about this compound, a different endpoint (or the reverse:
  this endpoint, a close analogue). Citable, and a report has to say which.
* **uncertain** — signals conflict, or the metadata is too thin to tell. Never
  promoted automatically; it exists so a borderline paper is set aside for a
  human rather than silently dropped, which is how a ranker quietly loses the
  one useful source.
* **irrelevant** — neither. Kept for audit, never citable.

Every decision carries reason codes and the policy version that produced it. A
relevance call made under one threshold must not be re-read years later as
though it had been made under another.

**"No relevant evidence" is a correct outcome.** Nothing here lowers a
threshold to fill a requested slot. The audit's query genuinely had no direct
literature in reach, and five unrelated papers is a worse answer than none.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

#: Bumped when a rule or a vocabulary changes, and recorded on every
#: assessment.
RELEVANCE_POLICY_VERSION = "evidence-relevance-1"


class Relevance(str, Enum):
    DIRECT = "direct"
    CONTEXTUAL = "contextual"
    UNCERTAIN = "uncertain"
    IRRELEVANT = "irrelevant"

    @property
    def is_citable(self) -> bool:
        return self in (Relevance.DIRECT, Relevance.CONTEXTUAL)


#: What each endpoint is actually about, in the words literature uses. Not a
#: synonym list for the *model*: these are the terms a paper on that biology
#: would contain, which is a different question from what the endpoint is
#: called internally.
ENDPOINT_VOCABULARY: Mapping[str, tuple[str, ...]] = {
    "herg": (
        "herg", "kcnh2", "ikr", "rapid delayed rectifier",
        "potassium channel", "k+ channel", "cardiac repolarisation",
        "cardiac repolarization", "qt prolongation", "qtc", "torsades",
        "patch clamp", "channel block",
    ),
    "clintox": (
        "clinical trial", "clinical toxicity", "drug withdrawal",
        "adverse event", "fda approval", "trial failure", "toxicity in humans",
    ),
    "tox21": (
        "tox21", "nuclear receptor", "stress response", "high-throughput screen",
        "hts assay", "reporter assay", "aryl hydrocarbon", "estrogen receptor",
        "androgen receptor", "mitochondrial membrane potential",
    ),
}

#: Tokens too generic to count as a compound match on their own. "alcohol"
#: appears in a paper about alcoholism, about ethanol, and about any molecule
#: with a hydroxyl group.
_WEAK_COMPOUND_TOKENS = frozenset(
    {"alcohol", "acid", "ester", "salt", "compound", "drug", "agent", "water", "oil"}
)

_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text or "").casefold().split())


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(_WORD.findall(_normalize(text)))


def _contains_phrase(tokens: Sequence[str], phrase: str) -> bool:
    needle = _tokens(phrase)
    if not needle:
        return False
    span = len(needle)
    return any(
        tuple(tokens[i : i + span]) == needle for i in range(len(tokens) - span + 1)
    )


@dataclass(frozen=True, slots=True)
class CompoundIdentity:
    """What the run actually knows the molecule is called.

    Assembled server-side from the analysis snapshot and the compound
    resolver, never from the model: a query plan built out of names the model
    supplied would let it search for whatever it expected to find.
    """

    canonical_smiles: str = ""
    preferred_name: str = ""
    synonyms: tuple[str, ...] = ()
    inchikey: str = ""
    identifiers: Mapping[str, str] = field(default_factory=dict)

    @property
    def names(self) -> tuple[str, ...]:
        """Every name worth matching on, weak generics removed."""
        candidates = [self.preferred_name, *self.synonyms]
        seen: list[str] = []
        for name in candidates:
            normalized = _normalize(name)
            if not normalized or normalized in seen:
                continue
            if normalized in _WEAK_COMPOUND_TOKENS:
                continue
            seen.append(normalized)
        return tuple(seen)


@dataclass(frozen=True, slots=True)
class RelevanceTarget:
    """The endpoint/assay and the proposition the search is for."""

    endpoint: str
    task: str | None = None
    proposition: str = ""

    @property
    def vocabulary(self) -> tuple[str, ...]:
        terms = list(ENDPOINT_VOCABULARY.get(self.endpoint, ()))
        if self.task:
            terms.append(self.task.replace("_", " "))
        return tuple(terms)


@dataclass(frozen=True, slots=True)
class RelevanceAssessment:
    relevance: Relevance
    reason_codes: tuple[str, ...]
    #: The evidence the decision rests on, so a reviewer can check it rather
    #: than re-read the paper to guess what the ranker saw.
    matched_compound_terms: tuple[str, ...] = ()
    matched_endpoint_terms: tuple[str, ...] = ()
    policy_version: str = RELEVANCE_POLICY_VERSION
    #: ``rule`` here; a model- or SME-sourced assessment records itself as
    #: such, because a judgement's provenance changes what it is worth.
    assessor: str = "rule"

    @property
    def is_citable(self) -> bool:
        return self.relevance.is_citable

    def to_dict(self) -> dict[str, Any]:
        return {
            "relevance": self.relevance.value,
            "reason_codes": list(self.reason_codes),
            "matched_compound_terms": list(self.matched_compound_terms),
            "matched_endpoint_terms": list(self.matched_endpoint_terms),
            "policy_version": self.policy_version,
            "assessor": self.assessor,
        }


def _searchable_text(hit: Any) -> str:
    parts = [
        getattr(hit, "title", "") or "",
        getattr(hit, "abstract_or_excerpt", "") or "",
    ]
    facts = getattr(hit, "normalized_facts", None)
    if isinstance(facts, Mapping):
        parts.extend(str(value) for value in facts.values())
    return " ".join(parts)


def _identifier_match(hit: Any, compound: CompoundIdentity) -> str | None:
    """An identifier match is proof, not evidence. Checked first."""
    wanted = {
        _normalize(value)
        for value in (compound.inchikey, *compound.identifiers.values())
        if value
    }
    if not wanted:
        return None
    text = _normalize(_searchable_text(hit))
    identifier = getattr(hit, "identifier", None)
    for attribute in ("doi", "pmid", "pmcid", "inchikey", "cid"):
        value = getattr(identifier, attribute, None) if identifier else None
        if value and _normalize(str(value)) in wanted:
            return str(value)
    for value in wanted:
        if value and value in text:
            return value
    return None


def assess(
    hit: Any,
    *,
    compound: CompoundIdentity,
    target: RelevanceTarget,
) -> RelevanceAssessment:
    """Judge one search hit. Pure, deterministic, and explainable.

    Deliberately conservative about the *compound*: a paper that does not name
    the molecule is irrelevant to a question about that molecule, however good
    the endpoint match. The audit's five false matches were all endpoint-ish and
    none of them was about ethanol.
    """
    text = _searchable_text(hit)
    tokens = _tokens(text)
    if not tokens:
        return RelevanceAssessment(
            Relevance.UNCERTAIN,
            ("metadata_too_thin",),
        )

    matched_names = tuple(
        name for name in compound.names if _contains_phrase(tokens, name)
    )
    identifier = _identifier_match(hit, compound)
    compound_matched = bool(matched_names or identifier)

    matched_endpoint = tuple(
        term for term in target.vocabulary if _contains_phrase(tokens, term)
    )
    endpoint_matched = bool(matched_endpoint)

    compound_terms = tuple(matched_names) + ((identifier,) if identifier else ())

    if compound_matched and endpoint_matched:
        return RelevanceAssessment(
            Relevance.DIRECT,
            ("compound_match", "endpoint_match"),
            matched_compound_terms=compound_terms,
            matched_endpoint_terms=matched_endpoint,
        )
    if compound_matched:
        return RelevanceAssessment(
            Relevance.CONTEXTUAL,
            ("compound_match", "endpoint_mismatch"),
            matched_compound_terms=compound_terms,
        )
    if endpoint_matched:
        # About the biology, about some other molecule. This is exactly the
        # shape of four of the audit's five false matches, and it is the one a
        # naive ranker scores highest.
        return RelevanceAssessment(
            Relevance.IRRELEVANT,
            ("compound_mismatch",),
            matched_endpoint_terms=matched_endpoint,
        )
    return RelevanceAssessment(
        Relevance.IRRELEVANT, ("compound_mismatch", "endpoint_mismatch")
    )


@dataclass(frozen=True, slots=True)
class RetrievalBudget:
    """The user's limit as a hard ceiling, not a target (WS04).

    The audit's request said "at most two papers" and the run persisted five.
    A limit that only shapes a provider's page size is not a limit; this one
    bounds how many candidates are read in detail and how many may ever become
    citable.
    """

    max_reads: int
    max_promotions: int

    @classmethod
    def from_request(cls, limit: int, *, read_multiplier: int = 3) -> "RetrievalBudget":
        """Reading more than you cite is normal and necessary — you cannot know
        a paper is irrelevant without looking. Promoting more than asked for is
        not."""
        bounded = max(0, int(limit))
        return cls(max_reads=bounded * read_multiplier, max_promotions=bounded)


def select_for_promotion(
    assessed: Iterable[tuple[Any, RelevanceAssessment]],
    *,
    budget: RetrievalBudget,
) -> list[tuple[Any, RelevanceAssessment]]:
    """The citable subset, direct before contextual, never over budget.

    Returning fewer than the budget allows is a correct outcome and the common
    one. Nothing here reaches into the irrelevant pile to fill a slot.
    """
    ranked = sorted(
        (pair for pair in assessed if pair[1].is_citable),
        key=lambda pair: 0 if pair[1].relevance is Relevance.DIRECT else 1,
    )
    return ranked[: budget.max_promotions]
