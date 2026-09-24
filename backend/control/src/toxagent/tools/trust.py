"""Trust envelopes for content a model reads but must never obey (P1-04).

Provider content — a title, an abstract, an author list — reached the model as
plain fields beside a single ``untrusted_external_content: true`` flag. The
system prompt says not to follow instructions found in an abstract; a flag at
the end of a dict is a weak way to say which strings that means.

An envelope makes the boundary structural. Every free-text field a provider
supplied is moved into ``untrusted`` as::

    {"trust": "untrusted_external", "provenance": {"provider": ..., "record_id": ...},
     "content_type": "abstract", "instructions_allowed": false, "content": "...",
     "signals": ["imperative_to_assistant", "tool_reference"]}

Identifiers and server-computed metadata (evidence_id, status, source type,
dates, quality tier) stay at the top level: the server produced those and they
are safe to act on. ``signals`` are deterministic, non-blocking observations
for audit and trace grading — never a filter, because a keyword list is not a
security boundary and treating it as one is how an injection that avoids the
keywords gets through.

Applied behind the ``trust_envelope_v1`` rollout flag, to the model view only.
Canonical and UI views are unchanged.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

SCHEMA_VERSION = "trust-envelope-v1"

UNTRUSTED_EXTERNAL = "untrusted_external"
TRUSTED_INTERNAL = "trusted_internal"
USER_SUPPLIED = "user_supplied"

#: Provider free-text fields, by the content type they are enveloped as.
EVIDENCE_TEXT_FIELDS: dict[str, str] = {
    "title": "title",
    "abstract_or_excerpt": "abstract",
    "authors": "author_list",
    "normalized_facts": "provider_metadata",
    "rejection_reason": "provider_error",
}

_SIGNALS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("imperative_to_assistant", re.compile(
        r"\b(ignore|disregard|forget)\b.{0,40}\b(instruction|prompt|rule|previous)", re.I | re.S)),
    ("role_marker", re.compile(r"(^|\n|\s)(system|assistant|developer)\s*:", re.I)),
    ("tool_reference", re.compile(
        r"\b(bash|shell|curl|wget|exec|execute|webfetch|websearch|call the \w+ tool)\b", re.I)),
    ("url", re.compile(r"https?://|\b[\w-]+\.(example|com|net|org|io)\b", re.I)),
    ("verdict_demand", re.compile(
        r"\b(state|declare|say|conclude)\b.{0,40}\b(safe|regulatory|approved)\b", re.I | re.S)),
)


def signals(text: str) -> list[str]:
    return [name for name, pattern in _SIGNALS if pattern.search(text or "")]


def _as_text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "; ".join(str(v) for v in value)
    if isinstance(value, dict):
        return "; ".join(f"{k}: {v}" for k, v in value.items())
    return "" if value is None else str(value)


def envelope(
    content: Any, *, content_type: str, provider: str, record_id: str,
    trust: str = UNTRUSTED_EXTERNAL,
) -> dict[str, Any]:
    text = _as_text(content)
    return {
        "trust": trust,
        "provenance": {"provider": provider, "record_id": record_id},
        "content_type": content_type,
        "instructions_allowed": False,
        "content": content,
        "signals": signals(text),
    }


def wrap_evidence_view(view: dict[str, Any], *, provider: str, record_id: str) -> dict[str, Any]:
    """Move provider free text into envelopes; keep server metadata on top."""
    wrapped = {k: v for k, v in view.items() if k not in EVIDENCE_TEXT_FIELDS}
    wrapped.pop("untrusted_external_content", None)
    untrusted = [
        {"field": field, **envelope(view[field], content_type=content_type,
                                    provider=provider, record_id=record_id)}
        for field, content_type in EVIDENCE_TEXT_FIELDS.items()
        if view.get(field) not in (None, "", [], {})
    ]
    wrapped["untrusted"] = untrusted
    wrapped["trust_envelope"] = SCHEMA_VERSION
    return wrapped


def signal_summary(views: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Counts of each signal across enveloped views, for audit/provenance."""
    counts: dict[str, int] = {}
    for view in views:
        for item in view.get("untrusted", ()):
            for name in item.get("signals", ()):
                counts[name] = counts.get(name, 0) + 1
    return counts
