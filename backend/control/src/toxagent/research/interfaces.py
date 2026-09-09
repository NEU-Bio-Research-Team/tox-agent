"""The research provider contract (plan sections 8.4, 12.1... actually 5.6/14.2).

A provider returns compact, already-typed search hits. Normalization and
acceptance policy live in this package (``normalization.py``, ``policy.py``),
not in the provider, so a second provider can be added later without
re-deriving evidence policy, and so ``get_evidence_record`` never has to call
a provider a second time: everything a stored ``EvidenceRecord`` can offer is
already in the hit a search produced (plan section 8.4's two-tool split is
about model-facing surface area, not about how many network calls happen).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol, Sequence, runtime_checkable

from ..domain.evidence import SourceIdentifier, SourceType


@dataclass(frozen=True)
class SearchHit:
    """One provider search result, already carrying everything normalization
    needs to build an ``EvidenceRecord``."""

    provider_record_id: str
    source_type: SourceType
    title: str
    authors: tuple[str, ...] = ()
    published_at: date | None = None
    canonical_url: str | None = None
    identifier: SourceIdentifier = field(default_factory=SourceIdentifier)
    abstract_or_excerpt: str | None = None
    #: A small, provider-declared subset of facts normalization can trust
    #: without knowing that provider's raw schema (e.g. ``journal``,
    #: ``pub_type``) — kept generic on purpose so a second provider does not
    #: require ``research/normalization.py`` or ``policy.py`` to change.
    normalized_facts: dict[str, Any] = field(default_factory=dict)
    #: The raw provider record, kept only for the content hash and audit —
    #: never sent to a model (plan section 8.4: "Raw payload chỉ cho audit
    #: role, không model-facing mặc định").
    raw: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ResearchProvider(Protocol):
    """Plan section 8.1: schema and execution policy have one source of
    truth. A provider only searches; it does not decide what is citable."""

    name: str

    async def search(
        self,
        *,
        query: str,
        source_types: Sequence[str] | None,
        date_from: date | None,
        limit: int,
    ) -> list[SearchHit]: ...
