"""The event envelope clients reconstruct state from.

Plan section 6.4. Every event carries a per-session ``sequence`` assigned inside
the same transaction as the state change it describes, so "I have seen up to 42"
is a complete statement of what a client knows. Delivery is at-least-once;
dedupe on ``event_id``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from .ids import EVENT, SESSION, new_id, require_id


class EventType(str, Enum):
    SESSION_CREATED = "session.created"
    MESSAGE_CREATED = "message.created"
    RUN_QUEUED = "run.queued"
    RUN_STARTED = "run.started"
    RUN_VALIDATING = "run.validating"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    RUN_CANCELLED = "run.cancelled"
    PART_CREATED = "part.created"
    PART_UPDATED = "part.updated"
    TOOL_STARTED = "tool.started"
    TOOL_COMPLETED = "tool.completed"
    TOOL_FAILED = "tool.failed"
    OBSERVATION_CREATED = "observation.created"
    ANALYSIS_CREATED = "analysis.created"
    EVIDENCE_CREATED = "evidence.created"
    ANSWER_ACCEPTED = "answer.accepted"
    ANSWER_REJECTED = "answer.rejected"
    RUNTIME_RECOVERY_STARTED = "runtime.recovery_started"
    RUNTIME_USAGE_REPORTED = "runtime.usage_reported"


@dataclass(frozen=True, slots=True)
class Event:
    event_id: str
    session_id: str
    sequence: int
    type: EventType
    entity_type: str
    entity_id: str
    occurred_at: datetime
    entity_version: int = 1
    run_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_id(self.event_id, EVENT, field="event.event_id")
        require_id(self.session_id, SESSION, field="event.session_id")
        if self.sequence < 1:
            raise ValueError("event.sequence is 1-based and monotonic per session")

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        sequence: int,
        type: EventType,
        entity_type: str,
        entity_id: str,
        occurred_at: datetime,
        entity_version: int = 1,
        run_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> "Event":
        return cls(
            event_id=new_id(EVENT),
            session_id=session_id,
            sequence=sequence,
            type=type,
            entity_type=entity_type,
            entity_id=entity_id,
            occurred_at=occurred_at,
            entity_version=entity_version,
            run_id=run_id,
            payload=payload or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "sequence": self.sequence,
            "type": self.type.value,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "entity_version": self.entity_version,
            "run_id": self.run_id,
            "occurred_at": self.occurred_at.isoformat(),
            "payload": self.payload,
        }
