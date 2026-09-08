"""Structure recognition: image -> SMILES -> the same deterministic analysis
pipeline a typed SMILES already goes through.

Optical structure recognition (OCSR) is served by a separate deployable
(../../../toxocr/) — this layer never imports a vision/OCR model directly,
extending ADR 0001's three-boundary topology to a third boundary. When no OCR
service is configured, SubmitMessage never reaches this class at all — see
its `structure_recognition_available` gate.

remaining-plan W4-07: the uploaded bytes live in the object store, not in
this run's memory — `submit_message.py` persists them and hands this class
only an `attachment_id`, scoped to the same actor that uploaded it (an
owner check, not just an existence check, the same discipline
`AnalysisStore`/`SessionStore` already apply elsewhere).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from ..domain.errors import AttachmentNotFound
from ..domain.events import EventType
from ..domain.message import Message, PartType, Role
from ..domain.run import RunStatus
from ..persistence.object_store import ObjectNotFound, ObjectRef, ObjectStore
from ..predictor.ocr_client import OcrClient, OcrError, OcrResult, OcrUnavailable
from .create_analysis import CreateAnalysis
from .policy import Actor
from .runs import advance


def _now() -> datetime:
    return datetime.now(timezone.utc)


class RecognizeStructure:
    def __init__(
        self,
        database,
        ocr_client: OcrClient,
        create_analysis: CreateAnalysis,
        object_store: ObjectStore,
    ) -> None:
        self._db = database
        self._ocr = ocr_client
        self._create_analysis = create_analysis
        self._object_store = object_store

    async def execute(
        self,
        *,
        actor: Actor,
        session_id: str,
        run_id: str,
        attachment_id: str,
        endpoints: tuple[str, ...] | None,
        threshold_overrides: Mapping[str, Any] | None,
    ) -> None:
        try:
            image_bytes, image_mime_type = await self._read_attachment(actor, attachment_id)
        except AttachmentNotFound:
            # Should not happen from a real upload (submit_message.py writes
            # the attachment in the same request that queues this run) — this
            # is a defensive, honest completion for whatever could still
            # produce it (e.g. the attachment's TTL cleanup, once W4-10
            # exists, races an unusually delayed run), not a silent 500.
            await self._complete_with_message(
                session_id, run_id,
                "The uploaded image is no longer available. Please upload it again.",
                reason="attachment_unavailable",
            )
            return

        try:
            result = await self._ocr.recognize(image_bytes, image_mime_type)
        except OcrUnavailable:
            await self._complete_with_message(
                session_id, run_id,
                "The structure recognition service could not be reached just now. "
                "Try again shortly, or submit a SMILES string directly.",
                reason="service_unavailable",
            )
            return
        except OcrError:
            await self._complete_with_message(
                session_id, run_id,
                "No chemical structure could be recognised in this image. "
                "Try a clearer image, or submit a SMILES string directly.",
                reason="no_structure_detected",
            )
            return

        # Persist the recognition result before starting prediction.  An
        # Analysis snapshot retains the recognised input SMILES, but not the
        # OCR confidence; the structured message is therefore the durable
        # UI/audit record for this hand-off. It remains truthful if the
        # subsequent predictor call fails.
        await self._record_recognition(session_id, run_id, result)

        # CreateAnalysis owns the run from here on — the exact same
        # deterministic pipeline a typed SMILES already goes through, so a
        # recognised structure is indistinguishable downstream from one the
        # user typed themselves (same validators, same snapshot, same
        # provenance).
        await self._create_analysis.execute(
            actor=actor, session_id=session_id, run_id=run_id,
            smiles=result.smiles, endpoints=endpoints, threshold_overrides=threshold_overrides,
        )

    async def _read_attachment(self, actor: Actor, attachment_id: str) -> tuple[bytes, str]:
        async with self._db.unit_of_work() as uow:
            attachment = await uow.attachments.get(attachment_id, owner_id=actor.subject_id)
        if attachment is None:
            raise AttachmentNotFound("no such attachment for this actor", attachment_id=attachment_id)
        try:
            data = await self._object_store.get(ObjectRef(key=attachment.object_uri))
        except ObjectNotFound as exc:
            # The metadata row is still useful for audit/TTL cleanup, but it
            # cannot make OCR possible when the underlying transient blob has
            # disappeared.  Normalize both absence modes into the same
            # user-safe completion above; never leak storage keys or a raw
            # provider/filesystem exception into the run transcript.
            raise AttachmentNotFound(
                "uploaded attachment bytes are no longer available", attachment_id=attachment_id
            ) from exc
        return data, attachment.media_type

    async def _record_recognition(self, session_id: str, run_id: str, result: OcrResult) -> None:
        """Write a bounded recognition result, never image bytes or a blob URI."""
        content: dict[str, Any] = {
            "code": "structure_recognized",
            "smiles": result.smiles,
            "canonical_smiles": result.canonical_smiles,
        }
        if result.confidence is not None:
            content["confidence"] = result.confidence

        async with self._db.unit_of_work() as uow:
            sequence = await uow.messages.next_sequence(session_id)
            message = Message.create(
                session_id,
                Role.ASSISTANT,
                sequence,
                now=_now(),
                parts=((PartType.TEXT, content),),
            )
            await uow.messages.add(message)
            uow.emit(
                session_id=session_id,
                type=EventType.MESSAGE_CREATED,
                entity_type="message",
                entity_id=message.id,
                run_id=run_id,
                payload={"role": "assistant", "kind": "structure_recognized"},
            )
            await uow.commit()

    async def _complete_with_message(
        self,
        session_id: str,
        run_id: str,
        message: str,
        *,
        reason: str = "recognition_failed",
    ) -> None:
        async with self._db.unit_of_work() as uow:
            run = await uow.runs.get(run_id)
            sequence = await uow.messages.next_sequence(session_id)
            reply = Message.create(
                session_id, Role.ASSISTANT, sequence, now=_now(),
                parts=(
                    (
                        PartType.TEXT,
                        {
                            "code": "structure_recognition_failed",
                            "question": "",
                            "message": message,
                            "reason": reason,
                        },
                    ),
                ),
            )
            await uow.messages.add(reply)
            uow.emit(
                session_id=session_id, type=EventType.MESSAGE_CREATED, entity_type="message",
                entity_id=reply.id, run_id=run_id, payload={"role": "assistant"},
            )
            if run is not None and not run.is_terminal:
                await advance(
                    uow, run, RunStatus.COMPLETED, payload={"reason": "structure_recognition_failed"}
                )
            await uow.commit()
