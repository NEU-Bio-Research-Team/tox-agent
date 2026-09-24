"""Deterministic acceptance policy for retrieved evidence (plan sections 5.6,
7.3, 14.2).

A search hit is not evidence. Every record this module touches starts
``retrieved`` and leaves as either ``accepted`` (passed every check below) or
``rejected`` (kept for audit, never citable) — nothing stays ``retrieved`` or
``normalized`` at rest, because nothing downstream reads those states as
"maybe usable". Rejection here is about the *record's* shape (host, a
required minimum field), never about whether its content is scientifically
correct — that judgment stays a model/SME question (plan section 9.3),
answered later by whether a claim citing it survives review.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Sequence
from urllib.parse import urlparse

from ..domain.evidence import EvidenceRecord, EvidenceStatus, SourceQualityTier, SourceType
from .interfaces import SearchHit


def host_is_allowed(url: str | None, allowed_hosts: Sequence[str]) -> bool:
    if not url:
        return False
    host = urlparse(url).hostname or ""
    return host in allowed_hosts


def classify_quality_tier(record: EvidenceRecord) -> SourceQualityTier:
    """A conservative, auditable default, not a judgment of scientific
    weight. Peer-reviewed journal literature is ``authoritative_secondary``;
    anything else this provider can return (e.g. a preprint, or a record
    missing the facts that indicate peer review) is plain ``secondary``.
    Neither tier is ``primary`` — a literature search result reports *about*
    primary data, it is not the predictor's own primary measurement."""
    pub_type = str(record.normalized_facts.get("pub_type") or "").lower()
    is_peer_reviewed_journal = bool(record.normalized_facts.get("journal")) and "preprint" not in pub_type
    return SourceQualityTier.AUTHORITATIVE_SECONDARY if is_peer_reviewed_journal else SourceQualityTier.SECONDARY


def filter_source_types(
    hits: Sequence[SearchHit], requested: Sequence[str] | None
) -> list[SearchHit]:
    if not requested:
        return list(hits)
    wanted = {SourceType(value) for value in requested}
    return [hit for hit in hits if hit.source_type in wanted]


def decide_acceptance(record: EvidenceRecord, *, allowed_hosts: Sequence[str]) -> EvidenceRecord:
    """Run one ``retrieved`` record through policy, returning it at its final
    resting status. Never returns a record still ``retrieved`` or
    ``normalized``."""
    normalized = record.to_status(EvidenceStatus.NORMALIZED)
    if not normalized.title.strip():
        return normalized.to_status(
            EvidenceStatus.REJECTED, reason="the provider returned a record with no title"
        )
    if normalized.canonical_url and not host_is_allowed(normalized.canonical_url, allowed_hosts):
        return normalized.to_status(
            EvidenceStatus.REJECTED,
            reason=f"canonical_url host is not on the research provider allowlist: "
            f"{normalized.canonical_url}",
        )
    tiered = replace(normalized, source_quality_tier=classify_quality_tier(normalized))
    return tiered.to_status(EvidenceStatus.ACCEPTED)
