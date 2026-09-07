"""Store interfaces the application depends on.

The application layer imports these Protocols and never a driver. Two rules are
expressed in the signatures rather than in prose:

* **Ownership is a query parameter, not a filter applied afterwards.** Reads
  take ``owner_id`` and return ``None`` for a session belonging to someone else,
  so a handler cannot forget to check and cannot leak the difference between
  "not yours" and "does not exist" (plan section 14.1).
* **Immutable records have no update method.** ``AnalysisStore``,
  ``ObservationStore`` and ``AnswerStore`` can add and read. There is nothing to
  call to rewrite a snapshot.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, Sequence, runtime_checkable

from ..domain.analysis import AnalysisSnapshot
from ..domain.answer import Claim, GroundedAnswer
from ..domain.attachment import Attachment
from ..domain.evidence import EvidenceRecord, EvidenceStatus
from ..domain.events import Event, EventType
from ..domain.message import Message
from ..domain.observation import Observation
from ..domain.run import Run
from ..domain.runtime import RuntimeBinding
from ..domain.usage import RuntimeUsageEvent
from ..domain.session import Session


@runtime_checkable
class SessionStore(Protocol):
    async def add(self, session: Session, *, client_session_id: str | None = None) -> None: ...
    async def get(self, session_id: str, *, owner_id: str) -> Session | None: ...
    async def get_unscoped(self, session_id: str) -> Session | None:
        """For internal callers that already hold an authorised capability."""
    async def find_by_client_id(self, owner_id: str, client_session_id: str) -> Session | None: ...
    async def update(self, session: Session, *, expected_version: int) -> None:
        """Optimistic update. Raises ``Conflict`` if the version moved."""
    async def list_for_owner(self, owner_id: str, *, limit: int, offset: int) -> Sequence[Session]: ...


@runtime_checkable
class MessageStore(Protocol):
    async def add(self, message: Message) -> None: ...
    async def get(self, message_id: str) -> Message | None: ...
    async def find_by_client_id(self, session_id: str, client_message_id: str) -> Message | None: ...
    async def list_for_session(
        self, session_id: str, *, after_sequence: int = 0, limit: int = 100
    ) -> Sequence[Message]: ...
    async def next_sequence(self, session_id: str) -> int: ...
    async def append_part(
        self, message_id: str, index: int, part_type: str, content: dict[str, Any]
    ) -> str:
        """Append one bounded assistant chunk. Returns the part id."""


@runtime_checkable
class RunStore(Protocol):
    async def add(self, run: Run) -> None: ...
    async def get(self, run_id: str) -> Run | None: ...
    async def update(self, run: Run, *, expected_version: int) -> None: ...
    async def list_for_session(self, session_id: str, *, limit: int = 50) -> Sequence[Run]: ...
    async def list_non_terminal(self, *, limit: int = 1000) -> Sequence[Run]: ...
    async def request_cancel(self, run_id: str) -> bool: ...
    async def cancel_requested(self, run_id: str) -> bool: ...


@runtime_checkable
class AnalysisStore(Protocol):
    async def add(self, snapshot: AnalysisSnapshot) -> None: ...
    async def get(self, analysis_id: str, *, session_id: str) -> AnalysisSnapshot | None: ...
    async def find_by_idempotency_key(
        self, session_id: str, idempotency_key: str
    ) -> AnalysisSnapshot | None: ...
    async def list_for_session(self, session_id: str, *, limit: int = 50) -> Sequence[AnalysisSnapshot]: ...


@runtime_checkable
class ObservationStore(Protocol):
    async def add(self, observation: Observation, *, analysis_id: str | None = None) -> None: ...
    async def get(self, observation_id: str, *, session_id: str) -> Observation | None: ...
    async def list_for_run(self, run_id: str) -> Sequence[Observation]: ...
    async def list_for_analysis(self, analysis_id: str) -> Sequence[Observation]: ...


@runtime_checkable
class EvidenceStore(Protocol):
    async def add(self, record: EvidenceRecord) -> None: ...
    async def get(self, evidence_id: str, *, session_id: str) -> EvidenceRecord | None: ...
    async def find_by_dedupe_key(self, session_id: str, dedupe_key: str) -> EvidenceRecord | None: ...
    async def set_status(
        self, evidence_id: str, status: EvidenceStatus, *, reason: str | None = None
    ) -> None: ...
    async def list_for_session(
        self, session_id: str, *, status: EvidenceStatus | None = None, limit: int = 50, offset: int = 0
    ) -> Sequence[EvidenceRecord]: ...


@runtime_checkable
class AnswerStore(Protocol):
    async def add(self, answer: GroundedAnswer) -> None: ...
    async def get(self, answer_id: str, *, session_id: str) -> GroundedAnswer | None: ...
    async def get_for_run(self, run_id: str) -> GroundedAnswer | None: ...
    async def candidate_generations(self, run_id: str) -> int: ...
    async def claims_for(self, answer_id: str) -> Sequence[Claim]: ...
    async def claim_id_exists(self, claim_id: str) -> bool: ...


@runtime_checkable
class RuntimeBindingStore(Protocol):
    async def add(self, binding: RuntimeBinding) -> None: ...
    async def get(self, binding_id: str) -> RuntimeBinding | None: ...
    async def active_for_session(self, session_id: str) -> RuntimeBinding | None: ...
    async def set_status(self, binding_id: str, status: str, *, now: datetime) -> None: ...


@runtime_checkable
class RuntimeUsageStore(Protocol):
    async def add(self, event: RuntimeUsageEvent) -> None: ...
    async def list_for_run(self, run_id: str) -> Sequence[RuntimeUsageEvent]: ...


@runtime_checkable
class ToolCallStore(Protocol):
    async def finish(
        self, call_id: str, *, status: str, error_code: str | None,
        observation_ids: list[str], duration_ms: int, now: datetime,
    ) -> None: ...
    async def count_for_run(self, run_id: str) -> int: ...
    async def duplicate_count(self, run_id: str, tool_name: str, arguments_sha256: str) -> int: ...
    async def list_for_run(self, run_id: str) -> Sequence[dict[str, Any]]: ...
    async def try_reserve(
        self, *, call_id: str, session_id: str, run_id: str, tool_name: str,
        arguments_sha256: str, now: datetime, max_calls: int | None, max_identical: int,
    ) -> bool: ...
    async def record_denied(
        self, *, call_id: str, session_id: str, run_id: str, tool_name: str,
        arguments_sha256: str, error_code: str, now: datetime,
    ) -> None: ...


@runtime_checkable
class CapabilityTokenStore(Protocol):
    async def issue(
        self, *, jti: str, session_id: str, run_id: str, runtime_binding_id: str | None,
        allowed_tools: list[str], issued_at: datetime, expires_at: datetime,
    ) -> None: ...
    async def is_valid(self, jti: str, *, now: datetime) -> bool: ...
    async def revoke(self, jti: str, *, now: datetime) -> None: ...


@runtime_checkable
class AttachmentStore(Protocol):
    async def add(self, attachment: Attachment) -> None: ...
    async def get(self, attachment_id: str, *, owner_id: str) -> Attachment | None: ...


@runtime_checkable
class OutboxReader(Protocol):
    """Read side of the outbox, used by the SSE dispatcher (plan section 13.3)."""

    async def read_after(
        self, session_id: str, after_sequence: int, *, limit: int = 200
    ) -> Sequence[Event]: ...
    async def mark_dispatched(self, event_ids: Sequence[str], *, now: datetime) -> None: ...
    async def latest_sequence(self, session_id: str) -> int: ...


@runtime_checkable
class UnitOfWork(Protocol):
    """One transaction. State changes and their events commit together.

    ``emit`` queues an event; sequences are assigned from the session counter at
    flush time, inside the same transaction, which is what makes the feed's
    ordering an actual guarantee rather than a hope about wall-clock time.
    """

    sessions: SessionStore
    messages: MessageStore
    runs: RunStore
    analyses: AnalysisStore
    observations: ObservationStore
    evidence: EvidenceStore
    answers: AnswerStore
    runtime_bindings: RuntimeBindingStore
    runtime_usage: RuntimeUsageStore
    tool_calls: ToolCallStore
    capability_tokens: CapabilityTokenStore
    attachments: AttachmentStore

    def emit(
        self, *, session_id: str, type: EventType, entity_type: str, entity_id: str,
        run_id: str | None = None, entity_version: int = 1, payload: dict[str, Any] | None = None,
    ) -> None: ...

    async def commit(self) -> None: ...
    async def rollback(self) -> None: ...


@runtime_checkable
class Database(Protocol):
    def unit_of_work(self) -> Any:
        """Async context manager yielding a :class:`UnitOfWork`."""

    def outbox(self) -> OutboxReader: ...
    async def create_schema(self) -> None: ...
    async def dispose(self) -> None: ...
