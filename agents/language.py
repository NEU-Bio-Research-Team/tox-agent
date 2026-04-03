from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")


def normalize_language(language: str | None) -> str:
    value = str(language or "").strip().lower()
    if value.startswith("vi"):
        return "vi"
    if value.startswith("en"):
        return "en"
    return "vi"


def is_vietnamese(language: str | None) -> bool:
    return normalize_language(language) == "vi"


def choose_text(language: str | None, vi: T, en: T) -> T:
    return vi if is_vietnamese(language) else en
