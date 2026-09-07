"""EvidenceRecord — external material, normalised, and never trusted.

Plan sections 5.6 and 14.2. Two properties matter more than the field list.
First, a search hit is not evidence: a record moves ``retrieved -> normalized ->
accepted`` before a claim may cite it, and ``rejected`` records stay for audit
without becoming citable. Second, every piece of text that came from outside is
marked ``untrusted_external_content``; it is data for the model to read, never
instructions for it to follow, and no source text can widen tool authority.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
from datetime import date, datetime
from enum import Enum
from typing import Any

from .ids import EVIDENCE, SESSION, new_id, require_id
from .provenance import content_sha256


class SourceType(str, Enum):
    ARTICLE = "article"
    DATABASE = "database"
    REGULATORY = "regulatory"
    VENDOR_DOCUMENTATION = "vendor_documentation"
    OTHER = "other"


class SourceQualityTier(str, Enum):
    PRIMARY = "primary"
    AUTHORITATIVE_SECONDARY = "authoritative_secondary"
    SECONDARY = "secondary"
    UNKNOWN = "unknown"


class EvidenceStatus(str, Enum):
    RETRIEVED = "retrieved"
    NORMALIZED = "normalized"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


ALLOWED_STATUS_TRANSITIONS: dict[EvidenceStatus, frozenset[EvidenceStatus]] = {
    EvidenceStatus.RETRIEVED: frozenset({EvidenceStatus.NORMALIZED, EvidenceStatus.REJECTED}),
    EvidenceStatus.NORMALIZED: frozenset({EvidenceStatus.ACCEPTED, EvidenceStatus.REJECTED}),
    EvidenceStatus.ACCEPTED: frozenset({EvidenceStatus.SUPERSEDED, EvidenceStatus.REJECTED}),
    EvidenceStatus.REJECTED: frozenset(),
    EvidenceStatus.SUPERSEDED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class SourceIdentifier:
    doi: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    cid: str | None = None
    other: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "doi": self.doi, "pmid": self.pmid, "pmcid": self.pmcid,
            "cid": self.cid, "other": self.other,
        }

    @property
    def dedupe_key(self) -> str | None:
        """The strongest stable identifier available. Two records sharing one
        are the same source arriving twice, whatever their titles say."""
        for prefix, value in (
            ("doi", self.doi), ("pmid", self.pmid), ("pmcid", self.pmcid), ("cid", self.cid)
        ):
            if value:
                return f"{prefix}:{value.strip().lower()}"
        return None


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    id: str
    session_id: str
    provider: str
    provider_record_id: str
    source_type: SourceType
    title: str
    retrieved_at: datetime
    status: EvidenceStatus
    content_sha256: str
    authors: tuple[str, ...] = ()
    published_at: date | None = None
    canonical_url: str | None = None
    identifier: SourceIdentifier = field(default_factory=SourceIdentifier)
    abstract_or_excerpt: str | None = None
    normalized_facts: dict[str, Any] = field(default_factory=dict)
    source_quality_tier: SourceQualityTier = SourceQualityTier.UNKNOWN
    raw_payload_ref: str | None = None
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        require_id(self.id, EVIDENCE, field="evidence.id")
        require_id(self.session_id, SESSION, field="evidence.session_id")
        if not self.provider or not self.provider_record_id:
            raise ValueError("evidence must name its provider and the provider's record id")

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        provider: str,
        provider_record_id: str,
        source_type: SourceType,
        title: str,
        retrieved_at: datetime,
        **rest: Any,
    ) -> "EvidenceRecord":
        payload = {
            "provider": provider,
            "provider_record_id": provider_record_id,
            "title": title,
            **{k: (str(v) if isinstance(v, (date, datetime)) else v) for k, v in rest.items()
               if k not in {"status", "content_sha256"}},
        }
        return cls(
            id=new_id(EVIDENCE),
            session_id=session_id,
            provider=provider,
            provider_record_id=provider_record_id,
            source_type=source_type,
            title=title,
            retrieved_at=retrieved_at,
            status=EvidenceStatus.RETRIEVED,
            content_sha256=content_sha256(_hashable(payload)),
            **{k: v for k, v in rest.items() if k not in {"status", "content_sha256"}},
        )

    def to_status(self, target: EvidenceStatus, *, reason: str | None = None) -> "EvidenceRecord":
        if target not in ALLOWED_STATUS_TRANSITIONS[self.status]:
            raise ValueError(f"evidence cannot go {self.status.value} -> {target.value}")
        if target is EvidenceStatus.REJECTED and not reason:
            raise ValueError("a rejected evidence record must record why")
        return replace(self, status=target, rejection_reason=reason or self.rejection_reason)

    @property
    def is_citable(self) -> bool:
        return self.status is EvidenceStatus.ACCEPTED

    @property
    def dedupe_key(self) -> str:
        return self.identifier.dedupe_key or f"{self.provider}:{self.provider_record_id}"

    def model_view(self, fields: tuple[str, ...] | None = None) -> dict[str, Any]:
        """What a model may see. Raw provider payloads are audit-only, and every
        free-text field carries the untrusted marker (plan section 14.2)."""
        available = {
            "evidence_id": self.id,
            "title": self.title,
            "authors": list(self.authors[:8]),
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "source_type": self.source_type.value,
            "source_quality_tier": self.source_quality_tier.value,
            "identifier": self.identifier.to_dict(),
            "canonical_url": self.canonical_url,
            "abstract_or_excerpt": self.abstract_or_excerpt,
            "normalized_facts": self.normalized_facts,
            # A model deciding whether to cite this needs to know a rejected
            # record is not citable *before* it spends a correction attempt
            # discovering that from submit_grounded_answer's own violation.
            "status": self.status.value,
            "rejection_reason": self.rejection_reason,
        }
        if fields:
            keep = set(fields) | {"evidence_id", "status"}
            available = {k: v for k, v in available.items() if k in keep}
        available["untrusted_external_content"] = True
        return available


def _hashable(payload: Any) -> Any:
    """Make a value canonical-JSON encodable without losing information.

    Dates become ISO strings and enums become their values so that the content
    hash of a record is stable across a process restart; a dataclass becomes its
    field mapping so a nested ``SourceIdentifier`` hashes as the data it is.
    """
    if is_dataclass(payload) and not isinstance(payload, type):
        return {f.name: _hashable(getattr(payload, f.name)) for f in fields(payload)}
    if isinstance(payload, dict):
        return {k: _hashable(v) for k, v in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_hashable(v) for v in payload]
    if isinstance(payload, (date, datetime)):
        return payload.isoformat()
    if isinstance(payload, Enum):
        return payload.value
    return payload
