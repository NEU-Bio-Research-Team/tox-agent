"""Citation and basis validation (plan section 9.3).

Every scientific or comparison claim needs at least one of two bases: an
observation field path, or one or more accepted evidence citations. What is
checked here is deterministic and narrow — existence, session scope, and
accepted status — not whether the cited source actually supports the sentence.
That semantic judgment needs a model grader or human review (plan section
9.3); a heuristic that claimed to settle it would be lying about what it
checked.
"""
from __future__ import annotations

from typing import Mapping

from ..domain.errors import Violation
from ..domain.evidence import EvidenceRecord
from .wire import ClaimCandidate

NEEDS_A_BASIS = {"scientific", "comparison"}


def validate_basis(
    claim: ClaimCandidate, *, has_observation_basis: bool
) -> list[Violation]:
    if claim.kind not in NEEDS_A_BASIS:
        return []
    if has_observation_basis or claim.citation_ids:
        return []
    return [
        Violation(
            "claim_has_no_basis",
            f"a {claim.kind} claim needs an observation field_path or at least one citation",
            path=f"claims[{claim.claim_id}]",
        )
    ]


def validate_citations(
    claim: ClaimCandidate,
    evidence_by_id: Mapping[str, EvidenceRecord],
    *,
    read_evidence_ids: frozenset[str] = frozenset(),
) -> list[Violation]:
    """``read_evidence_ids`` is every evidence id this run actually fetched
    through ``get_evidence_record`` (remaining-plan W3-07) — a search result
    already carries title/authors/identifier (tools/definitions/evidence.py's
    ``_SEARCH_RESULT_FIELDS``), enough to construct a citation without ever
    reading what the record says. Citing one anyway is citing a byline, not
    a source; the empty default keeps every caller that does not track reads
    (tests, other validators calling this per-claim) working unchanged."""
    violations: list[Violation] = []
    path = f"claims[{claim.claim_id}].citation_ids"
    for evidence_id in claim.citation_ids:
        record = evidence_by_id.get(evidence_id)
        if record is None:
            violations.append(
                Violation(
                    "citation_not_found",
                    f"evidence {evidence_id!r} does not exist in this session",
                    path=path, actual=evidence_id,
                )
            )
        elif not record.is_citable:
            violations.append(
                Violation(
                    "citation_not_accepted",
                    f"evidence {evidence_id!r} is {record.status.value}, not accepted",
                    path=path, actual=evidence_id,
                )
            )
        elif evidence_id not in read_evidence_ids:
            violations.append(
                Violation(
                    "citation_not_read",
                    f"evidence {evidence_id!r} was cited without ever calling "
                    "get_evidence_record on it in this run",
                    path=path, actual=evidence_id,
                )
            )
    return violations


def validate_recommendation_basis(
    recommendation_index: int, basis_claim_ids: list[str], known_claim_ids: frozenset[str]
) -> list[Violation]:
    unknown = [c for c in basis_claim_ids if c not in known_claim_ids]
    if not unknown:
        return []
    return [
        Violation(
            "recommendation_basis_unknown",
            f"recommended_next_steps[{recommendation_index}] cites claim(s) not in this answer",
            path=f"recommended_next_steps[{recommendation_index}].basis_claim_ids",
            actual=unknown,
        )
    ]
