"""Helpers shared by more than one store."""
from __future__ import annotations

from datetime import datetime
from typing import Any


def _parse_ts(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))
