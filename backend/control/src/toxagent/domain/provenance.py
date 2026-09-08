"""Canonical serialisation and content hashing.

Two things depend on canonical bytes being canonical: ``content_sha256`` on the
immutable records, and the idempotency keys that stop a retried request from
producing a second snapshot of the same molecule. Both would be silently wrong
if key order or float formatting drifted, so there is exactly one encoder and
everything uses it.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(payload: Any) -> str:
    """Deterministic JSON: sorted keys, no incidental whitespace, UTF-8 kept.

    ``ensure_ascii=False`` because Vietnamese content is stored as-is; the hash
    is over the UTF-8 encoding of this string, which is stable across platforms.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def idempotency_key(*parts: Any) -> str:
    """A stable key over an ordered tuple of scope components.

    Used for ``create_analysis_snapshot`` (canonical SMILES + endpoints +
    resolved policy + artifact hashes) and for answer candidates.
    """
    return content_sha256(list(parts))
