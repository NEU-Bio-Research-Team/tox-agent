"""SearchHit -> EvidenceRecord field mapping (plan section 5.6)."""
from __future__ import annotations

from datetime import date, datetime, timezone

from toxagent.domain.evidence import EvidenceStatus, SourceIdentifier, SourceType
from toxagent.domain.ids import new_id
from toxagent.research.interfaces import SearchHit
from toxagent.research.normalization import hit_to_evidence

SESSION_ID = new_id("ses")
NOW = datetime(2026, 9, 5, tzinfo=timezone.utc)


def test_a_hit_becomes_a_retrieved_evidence_record_with_every_field_carried_over():
    hit = SearchHit(
        provider_record_id="MED:42593910",
        source_type=SourceType.ARTICLE,
        title="Overcoming hERG Cardiotoxicity in a Rational Design Series",
        authors=("Yang T", "Li R"),
        published_at=date(2026, 8, 1),
        canonical_url="https://europepmc.org/article/MED/42593910",
        identifier=SourceIdentifier(doi="10.1021/x", pmid="42593910"),
        abstract_or_excerpt="Liver fibrosis is a major global health challenge...",
        normalized_facts={"journal": "J Med Chem", "pub_type": "Journal Article", "pub_year": "2026"},
        raw={"id": "42593910"},
    )
    record = hit_to_evidence(hit, provider="europepmc", session_id=SESSION_ID, retrieved_at=NOW)

    assert record.session_id == SESSION_ID
    assert record.provider == "europepmc"
    assert record.provider_record_id == "MED:42593910"
    assert record.source_type is SourceType.ARTICLE
    assert record.title == hit.title
    assert record.authors == hit.authors
    assert record.published_at == hit.published_at
    assert record.canonical_url == hit.canonical_url
    assert record.identifier == hit.identifier
    assert record.abstract_or_excerpt == hit.abstract_or_excerpt
    assert record.normalized_facts == hit.normalized_facts
    assert record.status is EvidenceStatus.RETRIEVED
    assert record.retrieved_at == NOW


def test_the_dedupe_key_prefers_the_strongest_identifier():
    hit = SearchHit(
        provider_record_id="MED:1",
        source_type=SourceType.ARTICLE,
        title="A",
        identifier=SourceIdentifier(doi="10.1000/x", pmid="123"),
    )
    record = hit_to_evidence(hit, provider="europepmc", session_id=SESSION_ID, retrieved_at=NOW)
    assert record.dedupe_key == "doi:10.1000/x"


def test_a_hit_with_no_identifier_dedupes_on_provider_and_record_id():
    hit = SearchHit(provider_record_id="MED:1", source_type=SourceType.ARTICLE, title="A")
    record = hit_to_evidence(hit, provider="europepmc", session_id=SESSION_ID, retrieved_at=NOW)
    assert record.dedupe_key == "europepmc:MED:1"
