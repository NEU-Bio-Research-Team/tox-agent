"""Full text captured once and replayed offline, for frozen benchmarks (#43).

A frozen evidence provider (``snapshot``, ``corpus``) froze what a search
returns -- titles and abstracts -- and nothing else, so the text tool was
never registered on a frozen deployment and a question whose answer sits in a
paper's results table could not be asked of the main study at all.

This store is the missing half: each document's located passages, captured
once from the real provider, with what the capture was (provider, identifier,
the sha256 of the bytes it received, the parser version that split them) and
the outcome (text read, or the provider's statement that it has none). A
frozen provider given a store serves ``fetch_full_text`` from it and never
from the network: a record the store does not hold is answered as not held,
not fetched.

File format (``frozen-full-text-v1``)::

    {"schema_version": "frozen-full-text-v1",
     "documents": {"<provider_record_id>": <document>, ...},
     "content_sha256": "<hash of everything else in the file>"}

A document::

    {"outcome": "retrieved" | "access_failed", "reason": str | null,
     "full_text_id": "PMC...", "parser_version": "jats-passages-v1",
     "capture": {"provider": "europepmc", "captured_at": "...",
                 "source_sha256": "<sha256 of the XML received>"},
     "passages": [{"locator", "section", "kind", "text", "truncated", "cells"}],
     "section_titles": [...], "omitted": {...}, "truncated": false,
     "document_sha256": "<hash of the document without this field>"}

Every document is re-hashed on load and the whole file is checked against its
``content_sha256`` -- and, for a sidecar file, against the deployment's pin.
Bytes that changed are refused rather than served.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..domain.provenance import content_sha256
from .fulltext import PARSER_VERSION, Passage
from .interfaces import FullTextFetch

SCHEMA_VERSION = "frozen-full-text-v1"
OUTCOMES = ("retrieved", "access_failed")
_PASSAGE_KEYS = frozenset({"locator", "section", "kind", "text", "truncated", "cells"})


class FullTextStoreError(ValueError):
    """The store cannot be trusted as the text it claims to be."""


def document_sha256(document: Mapping[str, Any]) -> str:
    return content_sha256({k: v for k, v in document.items() if k != "document_sha256"})


def freeze_document(
    fetch: FullTextFetch, *, provider: str, captured_at: str,
) -> dict[str, Any]:
    """One live fetch as a frozen document. A transient failure says nothing
    about the document and cannot be frozen."""
    if fetch.transient:
        raise FullTextStoreError(
            "a transient failure is not a fact about the document and is never frozen"
        )
    document: dict[str, Any] = {
        "outcome": "retrieved" if fetch.retrieved else "access_failed",
        "reason": None if fetch.retrieved else (fetch.reason or "the provider returned no text"),
        "full_text_id": fetch.full_text_id,
        "parser_version": fetch.provenance.get("parser_version") or PARSER_VERSION,
        "capture": {
            "provider": provider,
            "captured_at": captured_at,
            "source_sha256": fetch.provenance.get("source_sha256"),
        },
        "passages": [
            {"locator": p.locator, "section": p.section, "kind": p.kind, "text": p.text,
             "truncated": bool(getattr(p, "truncated", False)),
             "cells": list(getattr(p, "cells", ()) or ())}
            for p in fetch.passages
        ],
        "section_titles": list(fetch.section_titles),
        "omitted": dict(fetch.omitted),
        "truncated": bool(fetch.truncated),
    }
    document["document_sha256"] = document_sha256(document)
    return document


def build_store(documents: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    body = {"schema_version": SCHEMA_VERSION,
            "documents": {k: dict(v) for k, v in sorted(documents.items())}}
    return {**body, "content_sha256": content_sha256(body)}


def write_store(documents: Mapping[str, Mapping[str, Any]], path: Path) -> str:
    """Write a store and return the sha256 of its bytes -- the deployment's pin
    (TOXAGENT_RESEARCH_FULL_TEXT_SHA256)."""
    data = json.dumps(build_store(documents), sort_keys=True, indent=1,
                      ensure_ascii=False).encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _check_document(record_id: str, document: Mapping[str, Any]) -> None:
    where = f"full text of {record_id}"
    if document.get("outcome") not in OUTCOMES:
        raise FullTextStoreError(f"{where}: outcome must be one of {OUTCOMES}")
    if document_sha256(document) != document.get("document_sha256"):
        raise FullTextStoreError(f"{where} does not match its document_sha256")
    passages = document.get("passages") or []
    if document["outcome"] == "retrieved" and not passages:
        raise FullTextStoreError(f"{where}: a retrieved text holds at least one passage")
    if document["outcome"] == "access_failed" and not document.get("reason"):
        raise FullTextStoreError(f"{where}: a failed access records why")
    seen: set[str] = set()
    for index, passage in enumerate(passages):
        unknown = sorted(set(passage) - _PASSAGE_KEYS)
        if unknown or not passage.get("locator"):
            raise FullTextStoreError(f"{where}: passage {index} is malformed ({unknown})")
        if passage["locator"] in seen:
            raise FullTextStoreError(f"{where}: locator {passage['locator']} appears twice")
        seen.add(passage["locator"])


@dataclass(frozen=True)
class FrozenFullTextStore:
    name: str
    sha256: str
    _documents: Mapping[str, Mapping[str, Any]]

    @classmethod
    def from_document(cls, data: Mapping[str, Any], *, name: str) -> "FrozenFullTextStore":
        if data.get("schema_version") != SCHEMA_VERSION:
            raise FullTextStoreError(f"{name} is not a {SCHEMA_VERSION} store")
        body = {k: v for k, v in data.items() if k != "content_sha256"}
        if content_sha256(body) != data.get("content_sha256"):
            raise FullTextStoreError(f"{name} does not match its content_sha256")
        documents = dict(data.get("documents") or {})
        for record_id, document in documents.items():
            _check_document(record_id, document)
        return cls(name=name, sha256=str(data["content_sha256"]), _documents=documents)

    @classmethod
    def load(cls, path: Path, *, sha256: str) -> "FrozenFullTextStore":
        if not sha256.strip():
            raise FullTextStoreError(
                "a full-text store must be pinned by its sha256 "
                "(TOXAGENT_RESEARCH_FULL_TEXT_SHA256)"
            )
        data = Path(path).read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != sha256.strip().lower():
            raise FullTextStoreError(
                f"{Path(path).name} is not the pinned full-text store (sha256 {actual}, "
                f"pinned {sha256.strip().lower()})"
            )
        store = cls.from_document(json.loads(data), name=Path(path).name)
        return cls(name=store.name, sha256=actual, _documents=store._documents)

    def __len__(self) -> int:
        return len(self._documents)

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "sha256": self.sha256, "documents": len(self)}

    def fetch(self, provider_record_id: str, *, served_by: str) -> FullTextFetch:
        """The frozen answer for one record. A record the store does not hold
        is answered as such -- definitively, because a frozen run must not
        reach the network to find out otherwise."""
        document = self._documents.get(provider_record_id)
        replay = {"served_by": served_by, "replay": True, "store": self.name,
                  "store_sha256": self.sha256}
        if document is None:
            return FullTextFetch(
                retrieved=False,
                reason="the frozen full-text store holds no text for this record",
                provenance={**replay, "parser_version": None},
            )
        capture = document.get("capture") or {}
        provenance = {
            **replay,
            "parser_version": document.get("parser_version"),
            "captured_from": capture.get("provider"),
            "captured_at": capture.get("captured_at"),
            "source_sha256": capture.get("source_sha256"),
            "document_sha256": document.get("document_sha256"),
        }
        if document["outcome"] == "access_failed":
            return FullTextFetch(
                retrieved=False, reason=document.get("reason"),
                full_text_id=document.get("full_text_id"), provenance=provenance,
            )
        return FullTextFetch(
            retrieved=True,
            passages=tuple(
                Passage(locator=p["locator"], section=p.get("section", ""), kind=p["kind"],
                        text=p.get("text", ""), truncated=bool(p.get("truncated")),
                        cells=tuple(p.get("cells") or ()))
                for p in document["passages"]
            ),
            section_titles=tuple(document.get("section_titles") or ()),
            omitted=dict(document.get("omitted") or {}),
            truncated=bool(document.get("truncated")),
            full_text_id=document.get("full_text_id"),
            provenance=provenance,
        )


class FrozenFullTextProvider:
    """A frozen search provider that can also serve frozen full text.

    A wrapper rather than a flag on the providers it wraps: a deployment with
    no store must not *look* text-capable (the text tool is registered by
    ``isinstance(provider, FullTextCapableProvider)``), and one with a store
    must never fall back to the network. Everything but ``fetch_full_text``
    is the inner provider's.
    """

    def __init__(self, inner, store: FrozenFullTextStore, *, transient_fault: bool = False) -> None:
        self._inner = inner
        self._store = store
        self._transient_fault = transient_fault
        self.name = inner.name
        self.full_text_store = store

    def __getattr__(self, item: str):
        return getattr(self._inner, item)

    async def search(self, **kwargs):
        return await self._inner.search(**kwargs)

    async def fetch_full_text(self, *, provider_record_id: str, identifier=None) -> FullTextFetch:
        del identifier  # the store is keyed by the record the search returned
        if self._transient_fault:
            # The injected provider outage the snapshot provider declares
            # reaches the text too: it is transient, never an access failure.
            return FullTextFetch(retrieved=False, reason="frozen provider: injected outage",
                                 transient=True)
        return self._store.fetch(provider_record_id, served_by=self.name)

    async def aclose(self) -> None:
        await self._inner.aclose()
