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


#: A results table a question can only be answered from: the abstract gives no
#: number, the table gives the IC50.
POTENCY_JATS = """<article>
  <front><article-meta><abstract><p>We screened a series for hERG blockade.</p></abstract>
  </article-meta></front>
  <body>
    <sec><title>Results</title>
      <p>Potencies are summarised in Table 1.</p>
      <table-wrap><label>Table 1</label><caption><p>hERG potency by patch clamp</p></caption>
        <table>
          <thead><tr><th>Compound</th><th>hERG IC50 (uM)</th></tr></thead>
          <tbody>
            <tr><td>compound 4</td><td>2.75</td></tr>
            <tr><td>compound 7</td><td>0.38</td></tr>
          </tbody>
        </table>
      </table-wrap>
    </sec>
  </body>
</article>"""


class FullTextStubProvider(StubResearchProvider):
    """A stub that can also serve full text (``FullTextCapableProvider``).

    ``documents`` maps a provider record id to JATS; a record not in it answers
    the way Europe PMC does for a record with no deposited text. Set
    ``outage`` to answer as a provider that cannot be reached."""

    def __init__(self, *, hits: Sequence[SearchHit] = (ACCEPTED_HIT,),
                 documents: dict[str, str] | None = None) -> None:
        super().__init__(hits=hits)
        self.documents = dict(documents or {})
        self.outage = False
        self.fetches: list[str] = []

    async def fetch_full_text(self, *, provider_record_id: str, identifier=None):
        from toxagent.research.fulltext import parse_jats
        from toxagent.research.interfaces import FullTextFetch

        self.fetches.append(provider_record_id)
        if self.outage:
            return FullTextFetch(retrieved=False, reason="provider unreachable", transient=True)
        xml = self.documents.get(provider_record_id)
        if xml is None:
            return FullTextFetch(
                retrieved=False, reason="the record names no PMCID, and Europe PMC serves "
                                        "full text only by PMCID",
            )
        document = parse_jats(xml)
        return FullTextFetch(
            retrieved=True, passages=document.passages,
            section_titles=document.section_titles, truncated=document.truncated,
            omitted=dict(document.omitted), full_text_id="PMC1111111",
        )
