"""A stand-in research provider.

No network, no real EuropePMC — a scripted runtime test asserting the full
search -> get_evidence_record -> citation flow must not depend on a live
external service to be deterministic. ``StubResearchProvider`` implements the
exact ``ResearchProvider`` protocol a real one does, so the tool handlers built
in ``toxagent/tools/definitions/evidence.py`` cannot tell the difference.
"""
from __future__ import annotations

from datetime import date
from typing import Sequence

from toxagent.domain.evidence import SourceIdentifier, SourceType
from toxagent.research.interfaces import SearchHit

ACCEPTED_HIT = SearchHit(
    provider_record_id="MED:11111111",
    source_type=SourceType.ARTICLE,
    title="hERG blockade screening of a marine natural product series",
    authors=("Doe J", "Roe R"),
    published_at=date(2025, 6, 1),
    canonical_url="https://europepmc.org/article/MED/11111111",
    identifier=SourceIdentifier(pmid="11111111", doi="10.1000/example.doi"),
    abstract_or_excerpt="This study screened a series of compounds for hERG channel blockade.",
    normalized_facts={"journal": "J Med Chem", "pub_type": "Journal Article", "pub_year": "2025"},
    raw={"id": "11111111", "source": "MED"},
)

#: A hit policy must reject: no title at all.
TITLELESS_HIT = SearchHit(
    provider_record_id="MED:22222222",
    source_type=SourceType.ARTICLE,
    title="",
    canonical_url="https://europepmc.org/article/MED/22222222",
)

#: A hit whose canonical_url points off the configured allowlist.
OFF_ALLOWLIST_HIT = SearchHit(
    provider_record_id="MED:33333333",
    source_type=SourceType.ARTICLE,
    title="An article hosted somewhere the allowlist does not cover",
    canonical_url="https://not-a-real-allowlisted-host.example/article/33333333",
)


class StubResearchProvider:
    """``hits`` is what every ``search`` call returns, in order, regardless of
    the query — callers that need per-query behaviour should build one stub
    per test rather than parsing the query string."""

    name = "stub"

    def __init__(self, *, hits: Sequence[SearchHit] = (ACCEPTED_HIT,)) -> None:
        self.hits = list(hits)
        self.calls: list[dict] = []

    async def search(
        self,
        *,
        query: str,
        source_types: Sequence[str] | None,
        date_from: date | None,
        limit: int,
    ) -> list[SearchHit]:
        self.calls.append(
            {"query": query, "source_types": source_types, "date_from": date_from, "limit": limit}
        )
        return list(self.hits)[:limit]
