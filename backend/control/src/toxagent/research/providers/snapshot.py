"""A frozen evidence provider, for benchmarks against a live model (Wave 3).

A live benchmark run has a real model and usually a real predictor, but its
evidence came from EuropePMC on the day it ran: a different day is a different
suite, and an injection test cannot rely on the internet happening to serve an
abstract that says "ignore previous instructions". This provider serves the
``evidence`` section of an ``eval-fixture-v1`` file instead, verified against
its content hash, and can inject the provider faults a task declares.

It is a deployment option, never a default: ``TOXAGENT_RESEARCH_PROVIDER=snapshot``
with ``TOXAGENT_RESEARCH_SNAPSHOT_PATH`` naming the fixture, and optionally
``TOXAGENT_RESEARCH_SNAPSHOT_FAULT`` in ``rate_limited | timeout | empty |
unavailable``. The effective-product manifest records ``research_provider:
snapshot``, so a run on frozen evidence cannot be mistaken for a live one.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Sequence

from ...domain.errors import EvidenceUnavailable, ProviderRateLimited
from ...domain.evidence import SourceIdentifier, SourceType
from ...domain.provenance import content_sha256
from ..interfaces import SearchHit

FAULTS = ("rate_limited", "timeout", "empty", "unavailable")


def _hit(record: dict[str, Any]) -> SearchHit:
    published = record.get("published_at")
    identifier = record.get("identifier") or {}
    return SearchHit(
        provider_record_id=record["provider_record_id"],
        source_type=SourceType(record.get("source_type", "article")),
        title=record.get("title", ""),
        authors=tuple(record.get("authors") or ()),
        published_at=date.fromisoformat(published) if published else None,
        canonical_url=record.get("canonical_url"),
        identifier=SourceIdentifier(**{k: v for k, v in identifier.items() if v}),
        abstract_or_excerpt=record.get("abstract_or_excerpt"),
        normalized_facts=dict(record.get("normalized_facts") or {}),
        raw={"snapshot_record": record["provider_record_id"]},
    )


class SnapshotResearchProvider:
    name = "snapshot"

    def __init__(self, path: Path, *, fault: str = "") -> None:
        document = json.loads(Path(path).read_text())
        recorded = document.get("content_sha256")
        actual = content_sha256({k: v for k, v in document.items() if k != "content_sha256"})
        if recorded != actual:
            raise ValueError(
                f"evidence snapshot {path} content_sha256 does not match its content; "
                "refusing to serve evidence that is not the recorded snapshot"
            )
        if fault and fault not in FAULTS:
            raise ValueError(f"unknown snapshot fault {fault!r}; expected one of {FAULTS}")
        evidence = document.get("evidence") or {}
        self.snapshot_name = document.get("name", Path(path).stem)
        self.snapshot_sha256 = recorded
        self._index: dict[str, list[str]] = {
            key.casefold(): list(ids) for key, ids in (evidence.get("search") or {}).items()
        }
        self._records: dict[str, dict[str, Any]] = dict(evidence.get("records") or {})
        self._fault = fault

    async def search(
        self,
        *,
        query: str,
        source_types: Sequence[str] | None,
        date_from: date | None,
        limit: int,
    ) -> list[SearchHit]:
        if self._fault == "rate_limited":
            raise ProviderRateLimited("snapshot provider: injected rate limit", retry_after_ms=30_000)
        if self._fault in ("timeout", "unavailable"):
            raise EvidenceUnavailable(f"snapshot provider: injected {self._fault}")
        if self._fault == "empty":
            return []
        lowered = query.casefold()
        ids: list[str] = []
        for key, record_ids in self._index.items():
            if key in lowered:
                ids.extend(i for i in record_ids if i not in ids)
        hits = [_hit(self._records[i]) for i in ids if i in self._records]
        if source_types:
            hits = [h for h in hits if h.source_type.value in source_types]
        if date_from:
            hits = [h for h in hits if h.published_at is None or h.published_at >= date_from]
        return hits[:limit]

    async def aclose(self) -> None:
        return None
