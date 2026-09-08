"""Provider hit -> ``EvidenceRecord`` (plan section 5.6).

Purely a field mapping. Whether the resulting record is fit to cite is a
separate question, decided by ``policy.py`` — normalization never rejects
anything on its own, so a policy change never has to touch this module.
"""
from __future__ import annotations

from datetime import datetime

from ..domain.evidence import EvidenceRecord
from .interfaces import SearchHit


def hit_to_evidence(
    hit: SearchHit, *, provider: str, session_id: str, retrieved_at: datetime
) -> EvidenceRecord:
    return EvidenceRecord.create(
        session_id=session_id,
        provider=provider,
        provider_record_id=hit.provider_record_id,
        source_type=hit.source_type,
        title=hit.title,
        retrieved_at=retrieved_at,
        authors=hit.authors,
        published_at=hit.published_at,
        canonical_url=hit.canonical_url,
        identifier=hit.identifier,
        abstract_or_excerpt=hit.abstract_or_excerpt,
        normalized_facts=dict(hit.normalized_facts),
    )
