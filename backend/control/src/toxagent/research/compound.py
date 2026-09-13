"""The compound-information contract (spec sections 5.4, 9).

Names, synonyms, registry identifiers and bulk physicochemical properties are
*external facts*. They are not predictor output and they are not agent
knowledge, so they enter the same way literature does: through a named
provider, with a retrieval timestamp, a canonical URL and a content hash, and
they carry a field-level source reference into the report.

What this contract deliberately does not do is guess. A compound the provider
cannot resolve comes back as ``CompoundRecord.unresolved(...)`` — every
identity field ``None`` — because the failure mode that matters here is a
report confidently naming a *similar* molecule (spec section 5.4: "the agent
must not infer them from a similar compound").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from ..domain.provenance import content_sha256


@dataclass(frozen=True, slots=True)
class CompoundProperty:
    """One external property, with the field name a report can cite."""

    name: str
    value: Any
    unit: str | None = None
    source_field: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "source_field": self.source_field,
        }


@dataclass(frozen=True, slots=True)
class CompoundRecord:
    """A normalized compound-information record.

    ``resolved`` is explicit rather than inferred from empty fields: "the
    provider answered and knows nothing about this structure" and "the provider
    could not be reached" are different report gaps
    (``compound_identity_unresolved`` versus ``provider_unavailable``).
    """

    provider: str
    query_smiles: str
    resolved: bool
    retrieved_at: datetime
    preferred_name: str | None = None
    synonyms: tuple[str, ...] = ()
    identifiers: dict[str, Any] = field(default_factory=dict)
    properties: tuple[CompoundProperty, ...] = ()
    canonical_url: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    unresolved_reason: str | None = None

    @classmethod
    def unresolved(
        cls, *, provider: str, query_smiles: str, retrieved_at: datetime, reason: str
    ) -> "CompoundRecord":
        return cls(
            provider=provider,
            query_smiles=query_smiles,
            resolved=False,
            retrieved_at=retrieved_at,
            unresolved_reason=reason,
        )

    @property
    def content_sha256(self) -> str:
        return content_sha256(self.raw or self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "query_smiles": self.query_smiles,
            "resolved": self.resolved,
            "retrieved_at": self.retrieved_at.isoformat(),
            "preferred_name": self.preferred_name,
            "synonyms": list(self.synonyms),
            "identifiers": dict(self.identifiers),
            "properties": [p.to_dict() for p in self.properties],
            "canonical_url": self.canonical_url,
            "unresolved_reason": self.unresolved_reason,
        }

    def model_view(self) -> dict[str, Any]:
        """What a model is shown. No raw payload: the provider's own schema is
        for the audit trail, not for a prompt (same rule as ``SearchHit.raw``)."""
        view = self.to_dict()
        view["source_ref"] = f"{self.provider}:{self.identifiers.get('pubchem_cid') or 'unresolved'}"
        return view


@runtime_checkable
class CompoundProvider(Protocol):
    """Resolves a canonical SMILES to identity and selected properties.

    One method, by structure only. It deliberately takes no free-text name: a
    name lookup would let a model steer identity resolution towards the
    compound it expected rather than the one that was analysed.
    """

    name: str

    async def resolve(self, *, canonical_smiles: str) -> CompoundRecord: ...
