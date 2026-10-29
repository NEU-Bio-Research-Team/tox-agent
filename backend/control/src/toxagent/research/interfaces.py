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
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

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


@dataclass(frozen=True)
class FullTextFetch:
    """The outcome of asking a provider for a document's text.

    Never an exception: "the text exists and we could not open it" is a fact
    about the evidence, not a transport error to swallow, and a reader has to
    be able to tell it from "we only ever saw the abstract". ``reason`` is set
    whenever ``retrieved`` is false.
    """

    retrieved: bool
    passages: tuple[Any, ...] = ()
    section_titles: tuple[str, ...] = ()
    reason: str | None = None
    truncated: bool = False
    #: What the parser's caps left out, by passage kind (``research/fulltext``).
    omitted: Mapping[str, int] = field(default_factory=dict)
    #: The provider could not be asked (outage, rate limit, open circuit).
    #: That says nothing about the document, so it is not stored as an
    #: access failure; a later call asks again.
    transient: bool = False
    #: The identifier the text was asked for under, which need not be the
    #: record's own: Europe PMC serves text by PMCID, and a MEDLINE hit names
    #: its PMCID separately. ``None`` when no request was made.
    full_text_id: str | None = None
    #: Where the text came from (#43): the provider that served it, whether it
    #: was replayed from a frozen store, the parser version, and for a capture
    #: the sha256 of the bytes received. Kept with what was read, so a frozen
    #: run's text can be traced to the live capture it replays.
    provenance: Mapping[str, Any] = field(default_factory=dict)


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


@runtime_checkable
class FullTextCapableProvider(Protocol):
    """A provider that can also be asked for a document's text.

    Separate from ``ResearchProvider`` on purpose: a deployment whose provider
    cannot serve text must not look as though it can, so the capability is
    something callers test for rather than assume (plan section 8.4's point
    that whatever a search did not include cannot be shown later).
    """

    name: str

    async def fetch_full_text(
        self, *, provider_record_id: str, identifier: SourceIdentifier | None = None,
    ) -> FullTextFetch:
        """``identifier`` is the stored record's own, so the provider can find
        the text under an id other than ``provider_record_id``."""
        ...
