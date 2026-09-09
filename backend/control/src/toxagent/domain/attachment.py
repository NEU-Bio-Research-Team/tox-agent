"""Attachment — bytes that live in the object store, referenced by metadata.

Plan section 5.9. A model receives metadata or a scoped reference, never a
base64 payload by default: an inlined blob costs prompt budget, defeats the
projection caps, and turns an ACL decision into a string concatenation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from .ids import ATTACHMENT, SESSION, new_id, require_id


class RetentionClass(str, Enum):
    TRANSIENT = "transient"
    SESSION = "session"
    AUDIT = "audit"


@dataclass(frozen=True, slots=True)
class Attachment:
    id: str
    owner_id: str
    session_id: str
    media_type: str
    object_uri: str
    sha256: str
    size_bytes: int
    retention_class: RetentionClass
    created_at: datetime
    expires_at: datetime | None = None

    def __post_init__(self) -> None:
        require_id(self.id, ATTACHMENT, field="attachment.id")
        require_id(self.session_id, SESSION, field="attachment.session_id")
        if self.size_bytes < 0:
            raise ValueError("attachment.size_bytes must be non-negative")

    @classmethod
    def create(
        cls,
        *,
        owner_id: str,
        session_id: str,
        media_type: str,
        object_uri: str,
        sha256: str,
        size_bytes: int,
        retention_class: RetentionClass,
        now: datetime,
        expires_at: datetime | None = None,
    ) -> "Attachment":
        return cls(
            id=new_id(ATTACHMENT),
            owner_id=owner_id,
            session_id=session_id,
            media_type=media_type,
            object_uri=object_uri,
            sha256=sha256,
            size_bytes=size_bytes,
            retention_class=retention_class,
            created_at=now,
            expires_at=expires_at,
        )

    def model_view(self) -> dict[str, Any]:
        return {
            "attachment_id": self.id,
            "media_type": self.media_type,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }
