"""Messages and their typed parts.

Plan section 5.2. Assistant text is persisted as bounded chunks, never one row
per token (plan section 13.3) — the part is the unit of persistence, and a
delta is a stream optimisation that may be dropped without losing anything.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Sequence

from .ids import MESSAGE, PART, SESSION, new_id, require_id


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM_EVENT = "system_event"


class PartType(str, Enum):
    TEXT = "text"
    ANALYSIS_REF = "analysis_ref"
    ANSWER_REF = "answer_ref"
    TOOL_CALL = "tool_call"
    ERROR = "error"
    #: Metadata only (mime type, byte size and, once accepted, an opaque
    #: attachment id) — this is what gets persisted and read back. Uploaded
    #: bytes live in ObjectStore, never in product message state; OCR reaches
    #: them through AttachmentStore + ObjectStore (W4-07).
    IMAGE_REF = "image_ref"


@dataclass(frozen=True, slots=True)
class MessagePart:
    id: str
    message_id: str
    index: int
    type: PartType
    content: dict[str, Any]
    version: int = 1

    def __post_init__(self) -> None:
        require_id(self.id, PART, field="part.id")
        require_id(self.message_id, MESSAGE, field="part.message_id")
        if self.index < 0:
            raise ValueError("part.index must be non-negative")

    @classmethod
    def create(
        cls, message_id: str, index: int, type: PartType, content: dict[str, Any]
    ) -> "MessagePart":
        return cls(
            id=new_id(PART), message_id=message_id, index=index, type=type, content=content
        )


@dataclass(frozen=True, slots=True)
class Message:
    id: str
    session_id: str
    role: Role
    sequence: int
    created_at: datetime
    client_message_id: str | None = None
    parts: tuple[MessagePart, ...] = ()

    def __post_init__(self) -> None:
        require_id(self.id, MESSAGE, field="message.id")
        require_id(self.session_id, SESSION, field="message.session_id")
        if self.sequence < 0:
            raise ValueError("message.sequence must be non-negative")

    @classmethod
    def create(
        cls,
        session_id: str,
        role: Role,
        sequence: int,
        *,
        now: datetime,
        client_message_id: str | None = None,
        parts: Sequence[tuple[PartType, dict[str, Any]]] = (),
    ) -> "Message":
        """Parts are given as ``(type, content)`` pairs: their ids and indices
        belong to the message, so a caller cannot mint a part id that does not
        match the message it ends up in."""
        message_id = new_id(MESSAGE)
        return cls(
            id=message_id,
            session_id=session_id,
            role=role,
            sequence=sequence,
            created_at=now,
            client_message_id=client_message_id,
            parts=tuple(
                MessagePart.create(message_id, index, part_type, content)
                for index, (part_type, content) in enumerate(parts)
            ),
        )

    def text(self) -> str:
        return "\n".join(
            str(p.content.get("text", "")) for p in self.parts if p.type is PartType.TEXT
        ).strip()
