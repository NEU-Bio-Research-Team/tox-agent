"""Admission: from a client message to a queued run (plan sections 6.2, 7).

Admission control happens before anything expensive: size caps, ownership,
idempotency, the concurrency cap, and routing. A request that the router cannot
resolve produces a clarification message and a completed run — not a queued one
that later fails — because a clarification is an answer, not an error.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from sqlalchemy.exc import DBAPIError

from ..config import PolicySettings
from ..domain.attachment import Attachment, RetentionClass
from ..domain.errors import (
    AdmissionBusy,
    AnalysisNotFound,
    AttachmentUnavailable,
    Conflict,
    InvalidRequest,
    SessionNotFound,
)
from ..domain.events import EventType
from ..domain.message import Message, PartType, Role
from ..domain.run import Intent, Lane, Run, RunStatus
from ..domain.session import Session
from ..persistence.object_store import ObjectStore
from .policy import Actor
from .router import Clarification, RouteRequest, route
from .run_scheduler import RunContext, RunScheduler


def _now() -> datetime:
    return datetime.now(timezone.utc)


#: One `capability_unavailable` answer per gated intent. Keyed by `Intent`
#: rather than inlined at each call site, so a new gated capability adds one
#: entry here instead of another branch in `_answer_without_a_runtime`.
_CAPABILITY_UNAVAILABLE_MESSAGE: dict[Intent, str] = {
    Intent.EVIDENCE_RESEARCH: (
        "This deployment does not yet support searching external literature. "
        "Ask a question about the analysis already on screen instead."
    ),
    Intent.STRUCTURE_RECOGNITION: (
        "This deployment does not yet support recognising a chemical structure from an "
        "image. Submit a SMILES string directly, or draw the structure instead."
    ),
}


@dataclass(frozen=True)
class MessageSubmission:
    text: str = ""
    client_message_id: str | None = None
    intent_hint: str = "auto"
    smiles: str | None = None
    batch_smiles: tuple[str, ...] = ()
    endpoints: tuple[str, ...] | None = None
    threshold_overrides: Mapping[str, Any] | None = None
    include_attribution: bool = False
    analysis_id: str | None = None
    #: Set by the API layer, which decodes and size-checks the upload before
    #: this dataclass is built. `SubmitMessage.execute` persists these bytes
    #: to the object store (remaining-plan W4-07) and hands the run only an
    #: `attachment_id` — they never ride `RunContext` in memory. Only
    #: `image_mime_type`/`image_size_bytes` end up in the stored `image_ref`
    #: message part; the bytes themselves are addressed by the attachment
    #: row, never inlined into product state a client reads back.
    image_mime_type: str | None = None
    image_size_bytes: int = 0
    image_bytes: bytes | None = None


@dataclass(frozen=True)
class Accepted:
    message_id: str
    run_id: str
    run_status: RunStatus
    selected_intent: Intent
    lane: Lane
    events_url: str
    clarification: Clarification | None = None
    duplicate_of: str | None = None

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "message_id": self.message_id,
            "run_id": self.run_id,
            "run_status": self.run_status.value,
            "selected_intent": self.selected_intent.value,
            "lane": self.lane.value,
            "events_url": self.events_url,
        }
        if self.clarification is not None:
            body["clarification"] = self.clarification.to_dict()
        if self.duplicate_of is not None:
            body["duplicate_of_message_id"] = self.duplicate_of
        return body


class SubmitMessage:
    def __init__(
        self,
        database,
        settings: PolicySettings,
        scheduler: RunScheduler,
        *,
        evidence_research_available: bool = False,
        structure_recognition_available: bool = False,
        object_store: ObjectStore | None = None,
    ) -> None:
        self._db = database
        self._settings = settings
        self._scheduler = scheduler
        # Only ever needed when an image is actually about to be queued for a
        # real STRUCTURE_RECOGNITION run (never in the capability_unavailable
        # branch below — nothing would ever read it back) — but the object
        # store must exist whenever OCR itself is configured, so this is
        # asserted at construction, not discovered as a crash mid-request.
        if structure_recognition_available and object_store is None:
            raise ValueError(
                "structure_recognition_available requires an object_store to persist uploads into"
            )
        self._object_store = object_store
        # A deployment with no research provider configured (plan Phase 5;
        # ``TOXAGENT_RESEARCH_PROVIDER=""``) must say so deterministically
        # instead of dispatching a runtime turn a model has no tool to
        # actually fulfil it with.
        self._evidence_research_available = evidence_research_available
        # Same shape as evidence_research: a deployment with no OCR service
        # configured (TOXAGENT_OCR_URL unset — see config.OcrSettings and
        # api/app.py) must say so deterministically rather than queuing a run
        # whose registered handler doesn't exist.
        self._structure_recognition_available = structure_recognition_available

    async def execute(
        self, *, actor: Actor, session_id: str, submission: MessageSubmission
    ) -> Accepted:
        self._validate_envelope(submission)

        async with self._db.unit_of_work() as uow:
            # This database lock is the multi-instance admission authority.
            # In-memory scheduler tasks cannot prevent two API processes from
            # both observing an empty run list for the same session.
            try:
                session = await uow.sessions.get_for_admission(
                    session_id,
                    owner_id=actor.subject_id,
                    lock_timeout_ms=self._settings.admission_lock_timeout_ms,
                )
            except DBAPIError as exc:
                # PostgreSQL's lock_timeout has SQLSTATE 55P03. It is a
                # transient hot-session condition, not a 500; clients can
                # retry the exact idempotency key safely after the other
                # process commits its admission decision.
                if getattr(exc.orig, "sqlstate", None) == "55P03":
                    raise AdmissionBusy(
                        "another request is being admitted for this session; retry shortly",
                        retry_after_ms=self._settings.admission_lock_timeout_ms,
                    ) from exc
                raise
            if session is None:
                raise SessionNotFound("no such session", session_id=session_id)
            if not session.is_writable:
                raise Conflict(f"session is {session.status.value}", session_id=session_id)

            if submission.client_message_id:
                existing = await uow.messages.find_by_client_id(
                    session_id, submission.client_message_id
                )
                if existing is not None:
                    return await self._replay(uow, session, existing)

            if submission.analysis_id:
                target = await uow.analyses.get(submission.analysis_id, session_id=session_id)
                if target is None:
                    raise AnalysisNotFound(
                        "no such analysis in this session", analysis_id=submission.analysis_id
                    )

            await self._enforce_concurrency(uow, session_id)

            decision = route(
                RouteRequest(
                    text=submission.text,
                    molecule_smiles=submission.smiles,
                    batch_smiles=submission.batch_smiles,
                    has_image=submission.image_mime_type is not None,
                    intent_hint=submission.intent_hint,
                    has_active_analysis=session.active_analysis_id is not None,
                    analysis_id=submission.analysis_id,
                    requested_endpoints=submission.endpoints or (),
                    include_attribution=submission.include_attribution,
                )
            )

            # Do not accept a queued OCR run unless its source image has
            # already made it to durable byte storage.  The object write is
            # intentionally before message/run creation: if storage is down,
            # the request fails without leaving a queued run that no worker
            # can ever fulfil.  The attachment row, message and run then
            # commit in this one database transaction.  A database rollback
            # after a successful object write can leave a transient orphan;
            # W4-10's idempotent TTL cleanup owns that recoverable case.
            evidence_research_unavailable = (
                decision.intent is Intent.EVIDENCE_RESEARCH
                and not self._evidence_research_available
            )
            structure_recognition_unavailable = (
                decision.intent is Intent.STRUCTURE_RECOGNITION
                and not self._structure_recognition_available
            )
            capability_unavailable = evidence_research_unavailable or structure_recognition_unavailable

            attachment_id: str | None = None
            if submission.image_bytes is not None and not capability_unavailable:
                # `has_image` always routes to STRUCTURE_RECOGNITION.  Keep
                # the intent assertion explicit so a future router change
                # cannot silently persist an upload for an unrelated lane.
                assert decision.intent is Intent.STRUCTURE_RECOGNITION
                assert self._object_store is not None
                attachment_id = await self._persist_attachment(
                    uow, actor, session_id, submission
                )

            sequence = await uow.messages.next_sequence(session_id)
            message = Message.create(
                session_id, Role.USER, sequence, now=_now(),
                client_message_id=submission.client_message_id,
                parts=self._user_parts(submission, attachment_id=attachment_id),
            )
            await uow.messages.add(message)
            uow.emit(
                session_id=session_id, type=EventType.MESSAGE_CREATED,
                entity_type="message", entity_id=message.id,
                payload={"role": "user", "sequence": sequence},
            )

            # Measured ~1-2s per image on CPU (toxocr/), so this would already
            # fit run_deadline_s — the separate, larger deadline exists only
            # as a margin against a cold model load or a contended host.
            deadline_s = (
                self._settings.structure_recognition_deadline_s
                if decision.intent is Intent.STRUCTURE_RECOGNITION
                else self._settings.run_deadline_s
            )
            run = Run.create(
                session_id, message.id, decision.lane, decision.intent, now=_now(),
                deadline=timedelta(seconds=deadline_s),
            )
            await uow.runs.add(run)
            uow.emit(
                session_id=session_id, type=EventType.RUN_QUEUED, entity_type="run",
                entity_id=run.id, run_id=run.id,
                payload={"intent": decision.intent.value, "lane": decision.lane.value},
            )

            if (
                decision.intent in (Intent.CLARIFICATION_REQUIRED, Intent.OUT_OF_SCOPE)
                or capability_unavailable
            ):
                await self._answer_without_a_runtime(
                    uow, session, run, decision, capability_unavailable=capability_unavailable
                )
                await uow.commit()
                return Accepted(
                    message.id, run.id, RunStatus.COMPLETED, decision.intent, decision.lane,
                    self._events_url(session_id), decision.clarification,
                )

            await uow.commit()

        # An explicit analysis_id always wins. Otherwise, when this request is
        # about to snapshot a *new* molecule, leave the target unresolved
        # here — session.active_analysis_id still names the *old* one until
        # that snapshot commits, and pinning it now would answer against the
        # molecule this run is about to replace. The gateway resolves
        # the eventual target itself once the snapshot (if any) has landed.
        if submission.analysis_id:
            target_analysis_id: str | None = submission.analysis_id
        elif decision.needs_snapshot_first:
            target_analysis_id = None
        else:
            target_analysis_id = session.active_analysis_id

        self._scheduler.submit(
            RunContext(
                actor=actor,
                session_id=session_id,
                run_id=run.id,
                intent=decision.intent,
                text=submission.text,
                smiles=submission.smiles,
                batch_smiles=submission.batch_smiles,
                endpoints=submission.endpoints,
                threshold_overrides=submission.threshold_overrides,
                analysis_id=target_analysis_id,
                needs_snapshot_first=decision.needs_snapshot_first,
                language=session.preferred_language.value,
                attachment_id=attachment_id,
            )
        )
        return Accepted(
            message.id, run.id, RunStatus.QUEUED, decision.intent, decision.lane,
            self._events_url(session_id),
        )

    # --- admission ---------------------------------------------------------

    def _validate_envelope(self, submission: MessageSubmission) -> None:
        size = len(submission.text.encode("utf-8"))
        if size > self._settings.max_message_bytes:
            raise InvalidRequest(
                f"message of {size} bytes exceeds the {self._settings.max_message_bytes}-byte limit"
            )
        if len(submission.batch_smiles) > self._settings.max_batch_size:
            raise InvalidRequest(
                f"batch of {len(submission.batch_smiles)} exceeds the "
                f"{self._settings.max_batch_size}-molecule limit"
            )
        if submission.image_size_bytes > self._settings.max_image_bytes:
            raise InvalidRequest(
                f"image of {submission.image_size_bytes} bytes exceeds the "
                f"{self._settings.max_image_bytes}-byte limit"
            )

    async def _enforce_concurrency(self, uow, session_id: str) -> None:
        runs = await uow.runs.list_for_session(session_id, limit=20)
        active = [r for r in runs if not r.is_terminal]
        if len(active) >= self._settings.max_concurrent_runs_per_session:
            raise Conflict(
                "a run is already in flight for this session",
                active_run_ids=[r.id for r in active],
            )

    async def _persist_attachment(
        self, uow, actor: Actor, session_id: str, submission: MessageSubmission
    ) -> str:
        """Persist the bytes before an OCR run can be queued.

        The key is content-addressed, so byte-identical re-uploads reuse the
        same object.  The database row is staged in the caller's transaction
        only after ``put`` succeeds; hence no committed attachment can point
        to an object that was never stored.
        """
        assert submission.image_bytes is not None and submission.image_mime_type is not None
        digest = hashlib.sha256(submission.image_bytes).hexdigest()
        try:
            ref = await self._object_store.put(  # type: ignore[union-attr]
                f"attachments/{digest}", submission.image_bytes, content_type=submission.image_mime_type
            )
        except OSError as exc:
            raise AttachmentUnavailable("could not persist the uploaded image") from exc
        attachment = Attachment.create(
            owner_id=actor.subject_id,
            session_id=session_id,
            media_type=submission.image_mime_type,
            object_uri=ref.key,
            sha256=digest,
            size_bytes=submission.image_size_bytes,
            retention_class=RetentionClass.TRANSIENT,
            now=_now(),
        )
        await uow.attachments.add(attachment)
        return attachment.id

    @staticmethod
    def _user_parts(submission: MessageSubmission, *, attachment_id: str | None = None):
        parts: list[tuple[PartType, dict[str, Any]]] = []
        if submission.text.strip():
            parts.append((PartType.TEXT, {"text": submission.text}))
        if submission.smiles:
            parts.append((PartType.ANALYSIS_REF, {"smiles": submission.smiles}))
        if submission.batch_smiles:
            parts.append((PartType.ANALYSIS_REF, {"batch_smiles": list(submission.batch_smiles)}))
        if submission.image_mime_type:
            image_ref: dict[str, Any] = {
                "mime_type": submission.image_mime_type,
                "size_bytes": submission.image_size_bytes,
            }
            if attachment_id is not None:
                # This is an opaque product id, not an ObjectStore key or a
                # fetchable URL.  It lets recovery/audit locate the metadata
                # without exposing blob storage to a normal message reader.
                image_ref["attachment_id"] = attachment_id
            parts.append(
                (
                    PartType.IMAGE_REF,
                    image_ref,
                )
            )
        return tuple(parts)

    async def _replay(self, uow, session: Session, existing: Message) -> Accepted:
        """An idempotency key that has been seen returns the original run.

        Re-submitting must never buy a second prediction; a client retrying
        after a dropped response is the common case, not an error.
        """
        runs = await uow.runs.list_for_session(session.id, limit=50)
        run = next((r for r in runs if r.trigger_message_id == existing.id), None)
        if run is None:
            raise Conflict("the original run for this message id is missing", message_id=existing.id)
        return Accepted(
            existing.id, run.id, run.status, run.intent, run.lane,
            self._events_url(session.id), duplicate_of=existing.id,
        )

    async def _answer_without_a_runtime(
        self, uow, session: Session, run: Run, decision, *, capability_unavailable: bool = False
    ) -> None:
        """Clarifications, out-of-scope requests, and a capability this
        deployment hasn't wired yet are all deterministic answers — none of
        them should spend a runtime turn a model has no way to fulfil."""
        content: dict[str, Any] = {"reason": decision.reason}
        if capability_unavailable:
            content.update(
                {
                    "code": "capability_unavailable",
                    # Lets clients explain which optional capability is
                    # absent without inferring it from prose. This remains a
                    # deployment fact rather than a permanent product limit.
                    "capability": decision.intent.value,
                    "question": "",
                    "message": _CAPABILITY_UNAVAILABLE_MESSAGE.get(
                        decision.intent, _CAPABILITY_UNAVAILABLE_MESSAGE[Intent.EVIDENCE_RESEARCH]
                    ),
                }
            )
        elif decision.clarification is not None:
            content.update(decision.clarification.to_dict())
        else:
            content.update(
                {
                    "code": "out_of_scope",
                    "question": "",
                    "message": (
                        "This request is outside what ToxAgent does. It reports hERG, Tox21 and "
                        "ClinTox model outputs with their provenance and proposes verification "
                        "steps; it does not give clinical, dosing or regulatory advice."
                    ),
                }
            )
        sequence = await uow.messages.next_sequence(session.id)
        reply = Message.create(
            session.id, Role.ASSISTANT, sequence, now=_now(),
            parts=((PartType.TEXT, content),),
        )
        await uow.messages.add(reply)
        started = run.transition(RunStatus.RUNNING, now=_now())
        completed = started.transition(RunStatus.COMPLETED, now=_now())
        await uow.runs.update(completed, expected_version=run.version)
        uow.emit(
            session_id=session.id, type=EventType.MESSAGE_CREATED, entity_type="message",
            entity_id=reply.id, run_id=run.id, payload={"role": "assistant"},
        )
        uow.emit(
            session_id=session.id, type=EventType.RUN_COMPLETED, entity_type="run",
            entity_id=run.id, run_id=run.id, payload={"intent": run.intent.value},
        )

    @staticmethod
    def _events_url(session_id: str) -> str:
        return f"/v1/sessions/{session_id}/events"
