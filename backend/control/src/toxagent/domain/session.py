"""Session — the product-owned conversation and its ownership.

Plan section 5.1. A session belongs to exactly one subject for its whole life,
its sequence counter is the ordering authority for every event a client will
ever see, and none of that lives in a runtime process (PROD-04).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum

from .ids import SESSION, new_id, require_id


class SessionStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETION_PENDING = "deletion_pending"
    DELETED = "deleted"


class Language(str, Enum):
    VI = "vi"
    EN = "en"


class TitleSource(str, Enum):
    DETERMINISTIC = "deterministic"
    MODEL = "model"
    MANUAL = "manual"


@dataclass(frozen=True, slots=True)
class Session:
    id: str
    owner_id: str
    status: SessionStatus
    preferred_language: Language
    created_at: datetime
    updated_at: datetime
    version: int
    title: str | None = None
    active_analysis_id: str | None = None
    context_epoch: int = 0
    event_sequence: int = 0
    title_source: TitleSource | None = None
    title_status: str = "pending"
    title_updated_at: datetime | None = None

    def __post_init__(self) -> None:
        require_id(self.id, SESSION, field="session.id")
        if not self.owner_id:
            raise ValueError("session.owner_id is required and never changes")
        if self.context_epoch < 0:
            raise ValueError("session.context_epoch must not go backwards")
        if self.version < 1:
            raise ValueError("session.version starts at 1")

    @classmethod
    def create(
        cls,
        owner_id: str,
        *,
        now: datetime,
        preferred_language: Language = Language.EN,
        title: str | None = None,
    ) -> "Session":
        return cls(
            id=new_id(SESSION),
            owner_id=owner_id,
            status=SessionStatus.ACTIVE,
            preferred_language=preferred_language,
            title=title,
            created_at=now,
            updated_at=now,
            version=1,
        )

    @property
    def is_writable(self) -> bool:
        return self.status is SessionStatus.ACTIVE

    def with_active_analysis(self, analysis_id: str, *, now: datetime) -> "Session":
        return replace(
            self,
            active_analysis_id=analysis_id,
            updated_at=now,
            version=self.version + 1,
        )

    def with_title(self, title: str, *, source: TitleSource, now: datetime) -> "Session":
        normalized = " ".join(title.split())
        if not normalized or len(normalized) > 120 or any(ord(char) < 32 for char in normalized):
            raise ValueError("session title must be 1–120 printable characters")
        return replace(
            self, title=normalized, title_source=source, title_status="ready",
            title_updated_at=now, updated_at=now, version=self.version + 1,
        )

    def archived(self, *, now: datetime) -> "Session":
        """Archiving hides a session. It never deletes the audit trail; a real
        deletion goes through the retention workflow (plan section 13.4)."""
        return replace(
            self, status=SessionStatus.ARCHIVED, updated_at=now, version=self.version + 1
        )
