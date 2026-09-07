"""Prefixed identifiers.

Every entity id carries its type in the string itself: ``obs_9f2c...``. The
alternative — bare UUIDs — costs nothing until the day a claim cites an
``evidence_id`` where an ``observation_id`` belongs, and the lookup returns
nothing rather than telling you the reference was the wrong kind of thing. Here
that is a ``ValueError`` at the boundary.

The prefix is part of the stored primary key, so an id read out of a database
row, a log line, or a model-authored answer is self-describing without a
schema to consult.
"""
from __future__ import annotations

import re
from typing import Final
from uuid import uuid4

SESSION: Final = "ses"
MESSAGE: Final = "msg"
PART: Final = "prt"
RUN: Final = "run"
ANALYSIS: Final = "ana"
OBSERVATION: Final = "obs"
EVIDENCE: Final = "evd"
ANSWER: Final = "ans"
CLAIM: Final = "clm"
EVENT: Final = "evt"
RUNTIME_BINDING: Final = "rtb"
ATTACHMENT: Final = "att"
TOOL_CALL: Final = "call"
CAPABILITY: Final = "cap"
RUNTIME_USAGE: Final = "use"

PREFIXES: Final[frozenset[str]] = frozenset(
    {
        SESSION, MESSAGE, PART, RUN, ANALYSIS, OBSERVATION, EVIDENCE, ANSWER,
        CLAIM, EVENT, RUNTIME_BINDING, ATTACHMENT, TOOL_CALL, CAPABILITY, RUNTIME_USAGE,
    }
)

_PATTERN: Final = re.compile(r"^([a-z]{3,4})_([0-9a-f]{32})$")


def new_id(prefix: str) -> str:
    """Mint an identifier. ``prefix`` must be one of the declared kinds."""
    if prefix not in PREFIXES:
        raise ValueError(f"unknown id prefix {prefix!r}; declare it in domain.ids")
    return f"{prefix}_{uuid4().hex}"


def prefix_of(value: str) -> str:
    match = _PATTERN.match(value)
    if match is None:
        raise ValueError(f"not a ToxAgent identifier: {value!r}")
    return match.group(1)


def is_id(value: object, prefix: str | None = None) -> bool:
    if not isinstance(value, str):
        return False
    match = _PATTERN.match(value)
    if match is None:
        return False
    return prefix is None or match.group(1) == prefix


def require_id(value: object, prefix: str, *, field: str = "id") -> str:
    """Validate an identifier's kind, naming the field for the error message."""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string, got {type(value).__name__}")
    match = _PATTERN.match(value)
    if match is None:
        raise ValueError(f"{field} is not a ToxAgent identifier: {value!r}")
    actual = match.group(1)
    if actual != prefix:
        raise ValueError(
            f"{field} must be a {prefix!r} identifier, got a {actual!r} one: {value!r}"
        )
    return value
