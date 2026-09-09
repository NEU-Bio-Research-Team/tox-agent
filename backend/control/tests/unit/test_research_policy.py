"""Deterministic evidence acceptance policy (plan sections 5.6, 7.3, 14.2)."""
from __future__ import annotations

from datetime import datetime, timezone

from toxagent.domain.evidence import EvidenceStatus, SourceQualityTier, SourceType
from toxagent.domain.ids import new_id
from toxagent.research.interfaces import SearchHit
from toxagent.research.normalization import hit_to_evidence
from toxagent.research.policy import (
    classify_quality_tier,
    decide_acceptance,
    filter_source_types,
    host_is_allowed,
)

ALLOWED_HOSTS = ("www.ebi.ac.uk", "europepmc.org")
SESSION_ID = new_id("ses")
NOW = datetime(2026, 9, 5, tzinfo=timezone.utc)


def _record(**overrides):
    hit = SearchHit(
        provider_record_id=overrides.pop("provider_record_id", "MED:1"),
        source_type=overrides.pop("source_type", SourceType.ARTICLE),
        title=overrides.pop("title", "A hERG screening study"),
        canonical_url=overrides.pop("canonical_url", "https://europepmc.org/article/MED/1"),
        normalized_facts=overrides.pop(
            "normalized_facts", {"journal": "J Med Chem", "pub_type": "Journal Article"}
        ),
        **overrides,
    )
    return hit_to_evidence(hit, provider="europepmc", session_id=SESSION_ID, retrieved_at=NOW)


def test_host_is_allowed_checks_the_hostname_not_a_substring():
    assert host_is_allowed("https://europepmc.org/article/MED/1", ALLOWED_HOSTS)
    assert not host_is_allowed("https://evil-europepmc.org.attacker.example/x", ALLOWED_HOSTS)
    assert not host_is_allowed(None, ALLOWED_HOSTS)


def test_a_peer_reviewed_journal_record_is_authoritative_secondary():
    record = _record()
    assert classify_quality_tier(record) is SourceQualityTier.AUTHORITATIVE_SECONDARY


def test_a_preprint_is_only_secondary():
    record = _record(normalized_facts={"journal": "bioRxiv", "pub_type": "preprint"})
    assert classify_quality_tier(record) is SourceQualityTier.SECONDARY


def test_a_record_missing_journal_facts_is_only_secondary():
    record = _record(normalized_facts={})
    assert classify_quality_tier(record) is SourceQualityTier.SECONDARY


def test_a_well_formed_record_is_accepted():
    record = _record()
    final = decide_acceptance(record, allowed_hosts=ALLOWED_HOSTS)
    assert final.status is EvidenceStatus.ACCEPTED
    assert final.source_quality_tier is SourceQualityTier.AUTHORITATIVE_SECONDARY


def test_a_titleless_record_is_rejected_with_a_reason():
    record = _record(title="")
    final = decide_acceptance(record, allowed_hosts=ALLOWED_HOSTS)
    assert final.status is EvidenceStatus.REJECTED
    assert final.rejection_reason
    assert not final.is_citable


def test_a_record_off_the_host_allowlist_is_rejected():
    record = _record(canonical_url="https://not-allowed.example/article/1")
    final = decide_acceptance(record, allowed_hosts=ALLOWED_HOSTS)
    assert final.status is EvidenceStatus.REJECTED
    assert "allowlist" in final.rejection_reason


def test_a_record_with_no_canonical_url_is_not_rejected_for_that_reason():
    """Some providers may not always resolve a URL; absence is not itself a
    policy violation, only a *wrong* host is."""
    record = _record(canonical_url=None)
    final = decide_acceptance(record, allowed_hosts=ALLOWED_HOSTS)
    assert final.status is EvidenceStatus.ACCEPTED


def test_filter_source_types_is_a_no_op_when_nothing_was_requested():
    hits = [SearchHit(provider_record_id="a", source_type=SourceType.ARTICLE, title="A")]
    assert filter_source_types(hits, None) == hits
    assert filter_source_types(hits, []) == hits


def test_filter_source_types_drops_hits_outside_the_requested_set():
    article = SearchHit(provider_record_id="a", source_type=SourceType.ARTICLE, title="A")
    database = SearchHit(provider_record_id="b", source_type=SourceType.DATABASE, title="B")
    result = filter_source_types([article, database], ["article"])
    assert result == [article]
