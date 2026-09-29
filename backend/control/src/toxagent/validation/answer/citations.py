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

import re
from typing import Mapping

from ...domain.errors import Violation
from ...domain.evidence import EvidenceRecord
from .candidate_wire import ClaimCandidate

NEEDS_A_BASIS = {"scientific", "comparison"}

#: A citation marker a model may write inside report prose: ``[@evd_<32 hex>]``.
#: Structured on purpose. The alternative — letting the model write ``[1]`` or a
#: bare URL — makes the number the model's opinion rather than the artifact's
#: fact, and puts unvalidated text into something the renderers turn into a
#: link. Lives here rather than beside the compiler because both the validator
#: that refuses a bad token and the compiler that numbers a good one must read
#: exactly the same pattern (REP-01).
CITATION_TOKEN = re.compile(r"\[@(evd_[0-9a-f]{32})\]")


def cited_in_prose(body_markdown: str | None) -> set[str]:
    """Evidence ids cited inline in one section's prose."""
    return {match.group(1) for match in CITATION_TOKEN.finditer(body_markdown or "")}


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


def validate_observation_reference(
    claim: ClaimCandidate, observations_by_id: Mapping[str, object]
) -> list[Violation]:
    """A claim of any kind that names an observation must name one this run
    read. Numeric and classification claims already say so on their own path;
    the others passed with an unknown id and crashed the compile, which builds
    a domain ``Claim`` that refuses it (live, 2026-09-26: a report draft put a
    ChEMBL ``evd_`` id in ``observation_id`` and every submit raised)."""
    if (not claim.observation_id or claim.kind in {"numeric", "classification"}
            or claim.observation_id in observations_by_id):
        return []
    hint = (" — an evidence record is cited through citation_ids, not observation_id"
            if claim.observation_id.startswith("evd_") else "")
    return [
        Violation(
            "claim_observation_not_found",
            f"observation {claim.observation_id!r} was not read by this run{hint}",
            path=f"claims[{claim.claim_id}].observation_id",
            actual=claim.observation_id,
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
                    f"evidence {evidence_id!r} does not exist in this session"
                    + (" — it is an observation id: name it in observation_id with a "
                       "field_path, not in citation_ids"
                       if evidence_id.startswith("obs_") else ""),
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
