"""Dotted field paths into a canonical payload.

A numeric claim says "this number is ``predictions.herg.probability_blocker``
of observation obs_…". Validation then has to resolve exactly that path against
exactly that payload, with no coercion and no partial matches: resolving a path
that does not exist must raise, not return ``None``, because ``None`` compared
against a model-authored number is a comparison that can accidentally pass.

Segments are literal mapping keys or list indices. Tox21 task names contain
hyphens (``SR-p53``) but no dots, so a plain dot split is unambiguous.
"""
from __future__ import annotations

from typing import Any, Iterator

SEPARATOR = "."


class FieldPathError(KeyError):
    def __init__(self, path: str, at: str, reason: str) -> None:
        super().__init__(f"{path!r} does not resolve: {reason} at {at!r}")
        self.path, self.at, self.reason = path, at, reason


def split(path: str) -> list[str]:
    if not path or path.startswith(SEPARATOR) or path.endswith(SEPARATOR):
        raise FieldPathError(path, path, "empty or dangling segment")
    segments = path.split(SEPARATOR)
    if any(not s for s in segments):
        raise FieldPathError(path, path, "empty segment")
    return segments


def resolve(payload: Any, path: str) -> Any:
    """Return the value at ``path``. Raises ``FieldPathError`` if absent."""
    current = payload
    for segment in split(path):
        if isinstance(current, dict):
            if segment not in current:
                raise FieldPathError(path, segment, "no such key")
            current = current[segment]
        elif isinstance(current, (list, tuple)):
            if not segment.isdigit():
                raise FieldPathError(path, segment, "list index must be digits")
            index = int(segment)
            if index >= len(current):
                raise FieldPathError(path, segment, "index out of range")
            current = current[index]
        else:
            raise FieldPathError(path, segment, f"cannot descend into {type(current).__name__}")
    return current


def exists(payload: Any, path: str) -> bool:
    try:
        resolve(payload, path)
    except (FieldPathError, ValueError):
        return False
    return True


def walk(payload: Any, prefix: str = "") -> Iterator[str]:
    """Every leaf path in a payload. Used to build the allowlist a tool slice
    may expose and to report what a bad path could have meant."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            yield from walk(value, f"{prefix}{SEPARATOR}{key}" if prefix else str(key))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            yield from walk(value, f"{prefix}{SEPARATOR}{index}" if prefix else str(index))
    elif prefix:
        yield prefix
