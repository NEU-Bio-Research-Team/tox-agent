"""After dispatch: consuming runtime events, recording usage, committing the turn, cleaning up."""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Awaitable, Callable

from ...application.runs.scheduler import RunContext
from ...application.runs.transitions import advance
from ...domain.errors import DeadlineExceeded, RuntimeProtocolError, RuntimeUnavailable
from ...domain.events import EventType
from ...domain.message import Message, PartType, Role
from ...domain.run import Intent, RunStatus
from ...domain.runtime import BindingStatus, RuntimeBinding
from ...domain.usage import RuntimeUsageEvent
from ...platform.flags import is_enabled
from ..provider import (
    RuntimeEvent,
    RuntimeEventType,
    RuntimeSession,
)
from ..usage_normalizer import RuntimeUsageNormalizer
from ._common import _DIAGNOSTIC_DELTA_PREVIEW_CHARS, _now, log


class CompletionMixin:
    """Consuming a turn's events and committing its outcome."""

    async def _persist_started_run(self, context: RunContext, binding: RuntimeBinding) -> None:
        async with self._db.unit_of_work() as uow:
            current = await uow.runs.get(context.run_id)
            if current is None or current.session_id != context.session_id or current.is_terminal:
                raise RuntimeProtocolError("the run changed before runtime binding could be stored")
            await uow.runtime_bindings.add(binding)
            if current.status is RunStatus.RUNNING:
                # An orchestrated build started its run before any runtime was
                # involved: the deterministic stages are work too. Attach the
                # binding without pretending the run started a second time —
                # and attach it, because recovery after a lost runtime is only
                # permitted for a run that demonstrably had one.
                from dataclasses import replace
                await uow.runs.update(
                    replace(
                        current, runtime_binding_id=binding.id, version=current.version + 1
                    ),
                    expected_version=current.version,
                )
                await uow.commit()
                return
            await advance(
                uow,
                current,
                RunStatus.RUNNING,
                runtime_binding_id=binding.id,
                payload={"runtime": binding.manifest()},
            )
            await uow.commit()

    async def _consume_events(
        self,
        runtime_session: RuntimeSession,
        context: RunContext,
        binding: RuntimeBinding,
        deadline: datetime,
        has_product: Callable[[RunContext], Awaitable[bool]] | None = None,
    ) -> bool:
        """Wait for a normalized terminal event; return whether the binding was lost.

        Tool lifecycle is already persisted by ToolRunner.  MESSAGE_DELTA is
        deliberately not mirrored into product state — persisting it before
        ``submit_grounded_answer`` would let an ungrounded number survive in
        the transcript even when the validator correctly refuses the final
        candidate. A short, bounded tail of it is kept in memory only for the
        diagnostic log below; it is never written to the database or exposed
        over the API.
        """
        has_product = has_product or self._has_answer
        stream = self._provider.events(runtime_session, after=None)
        # One normalizer per runtime session. It is the fast path only: the
        # partial unique index on (runtime_binding_id, source_event_id) is what
        # holds when a worker is replaced mid-run and the replacement re-reads
        # the stream from the beginning.
        normalizer = (
            RuntimeUsageNormalizer(
                runtime_session.runtime_session_id, provider=self._provider.kind
            )
            if is_enabled("normalized_usage_v2")
            else None
        )
        lost = False
        delta_tail = ""
        while True:
            remaining = (deadline - _now()).total_seconds()
            if remaining <= 0:
                if await has_product(context):
                    return lost
                raise DeadlineExceeded("the runtime turn exceeded its deadline")
            try:
                event = await asyncio.wait_for(anext(stream), timeout=remaining)
            except asyncio.TimeoutError:
                # `remaining` bounds this one wait, not the whole turn — the
                # deadline itself is re-checked at the top of the loop, which
                # is where DeadlineExceeded actually gets raised. Found live
                # (2026-09-05): letting this propagate raw meant the run
                # scheduler's catch-all logged a turn that simply ran out of
                # time as failure_code "internal_error" instead of the typed
                # "deadline_exceeded" this exact case exists for.
                continue
            except StopAsyncIteration:
                if await has_product(context):
                    return lost
                raise RuntimeProtocolError(  # noqa: B904 - the stream end is the cause
                    "the runtime event stream ended without a final answer"
                )

            if event.type is RuntimeEventType.MESSAGE_DELTA:
                delta_tail = (delta_tail + str(event.payload.get("text", "")))[
                    -_DIAGNOSTIC_DELTA_PREVIEW_CHARS:
                ]
                continue
            if event.type is RuntimeEventType.USAGE_REPORTED:
                await self._record_usage_event(context, binding, event, normalizer)
                continue
            if event.type is RuntimeEventType.TURN_IDLE:
                if delta_tail and not await has_product(context):
                    log.warning(
                        "run %s reached TURN_IDLE with no submit_grounded_answer call; "
                        "last %d chars the runtime wrote instead: %r",
                        context.run_id, len(delta_tail), delta_tail,
                    )
                return lost
            if event.type is RuntimeEventType.SESSION_LOST:
                lost = True
                # A persisted, validated answer is sufficient product state.
                # Do not turn it into a failed report merely because the
                # provider died after the authoritative tool call completed.
                if await has_product(context):
                    return lost
                raise RuntimeUnavailable("the runtime session was lost", **event.payload)
            if event.type is RuntimeEventType.TURN_FAILED:
                if await has_product(context):
                    return lost
                raise RuntimeProtocolError("the runtime turn failed", **event.payload)

    async def _record_usage_event(
        self,
        context: RunContext,
        binding: RuntimeBinding,
        event: RuntimeEvent,
        normalizer: "RuntimeUsageNormalizer | None" = None,
    ) -> None:
        """Persist a report as a fact, without inventing a total.

        A zero is faithfully retained; an absent/malformed field is ``None``.
        With normalization on, a report that restates a total already held
        establishes no new fact, and nothing is written or emitted — which is
        the difference between 21 rows and the 7 the audit's Q&A run actually
        measured (P1-1). Providers disagree on whether a later report is a
        delta or a cumulative snapshot, so the row says which this one is
        rather than leaving a reader to assume.
        """
        normalized = None
        if normalizer is not None:
            normalized = normalizer.accept(event.payload)
            if normalized is None:
                return
        usage = RuntimeUsageEvent.from_provider_payload(
            session_id=context.session_id,
            run_id=context.run_id,
            runtime_binding_id=binding.id,
            provider_id=binding.provider_id,
            model_id=binding.model_id,
            payload=event.payload,
            reported_at=event.occurred_at,
            source_event_id=normalized.source_event_id if normalized else None,
            source_event_type=normalized.source_event_type if normalized else None,
            provider_message_id=normalized.provider_message_id if normalized else None,
            provider_step_id=normalized.provider_step_id if normalized else None,
            revision=normalized.revision if normalized else None,
            semantics=normalized.semantics.value if normalized else "unknown",
            is_normalized=normalized is not None,
            raw_payload_hash=normalized.raw_payload_hash if normalized else None,
        )
        async with self._db.unit_of_work() as uow:
            await uow.runtime_usage.add(usage)
            uow.emit(
                session_id=context.session_id,
                type=EventType.RUNTIME_USAGE_REPORTED,
                entity_type="runtime_usage",
                entity_id=usage.id,
                run_id=context.run_id,
                payload={
                    "usage_event_id": usage.id,
                    "provider_id": usage.provider_id,
                    "model_id": usage.model_id,
                },
            )
            await uow.commit()

    async def _has_answer(self, context: RunContext) -> bool:
        async with self._db.unit_of_work() as uow:
            if context.intent is Intent.BUILD_REPORT:
                builds = await uow.reports.list_builds_for_session(context.session_id, limit=50)
                build = next((item for item in builds if item.id == context.report_build_id), None)
                return bool(build and build.report_id)
            return await uow.answers.get_for_run(context.run_id) is not None

    async def _commit_product_and_complete(self, context: RunContext) -> None:
        if context.intent is Intent.BUILD_REPORT:
            await self._commit_report_and_complete(context)
            return
        await self._commit_answer_and_complete(context)

    async def _commit_answer_and_complete(self, context: RunContext) -> None:
        async with self._db.unit_of_work() as uow:
            run = await uow.runs.get(context.run_id)
            if run is None:
                raise RuntimeProtocolError("the runtime run disappeared before completion")
            answer = await uow.answers.get_for_run(context.run_id)
            if answer is None:
                raise RuntimeProtocolError(
                    "the runtime reached a terminal event without submit_grounded_answer"
                )
            if run.status is not RunStatus.RUNNING:
                raise RuntimeProtocolError(
                    "the run changed before its accepted answer could be committed",
                    status=run.status.value,
                )
            parts: list[tuple[PartType, dict]] = [
                (PartType.TEXT, {"text": answer.answer_markdown}),
                (PartType.ANSWER_REF, {"answer_id": answer.id}),
            ]
            if context.intent is Intent.DECISION_SUPPORT and is_enabled("scientific_case_v1"):
                # Stored before the run completes (_commit_with_case), so it
                # exists here for every completed case run.
                dossier = await uow.scientific_cases.get_dossier(
                    context.run_id, session_id=context.session_id
                )
                if dossier is not None:
                    parts.append((PartType.DOSSIER_REF, {
                        "case_id": dossier["case_id"], "run_id": context.run_id,
                        "case_revision": dossier.get("case_revision"),
                    }))
            sequence = await uow.messages.next_sequence(context.session_id)
            reply = Message.create(
                context.session_id,
                Role.ASSISTANT,
                sequence,
                now=_now(),
                parts=tuple(parts),
            )
            await uow.messages.add(reply)
            uow.emit(
                session_id=context.session_id,
                type=EventType.MESSAGE_CREATED,
                entity_type="message",
                entity_id=reply.id,
                run_id=context.run_id,
                payload={"role": "assistant", "answer_id": answer.id},
            )
            await advance(
                uow,
                run,
                RunStatus.COMPLETED,
                payload={"answer_id": answer.id, "is_fallback": answer.is_fallback},
            )
            await uow.commit()

    async def _revoke_token_quietly(self, token: str) -> None:
        try:
            claims = await self._capability_tokens.verify(token)
            await self._capability_tokens.revoke(claims.jti)
        except Exception:  # noqa: BLE001 - expiry/revocation must not hide a run result
            return

    async def _close_quietly(self, runtime_session: RuntimeSession) -> bool:
        try:
            return (await self._provider.close(runtime_session)).closed
        except Exception:  # noqa: BLE001 - status below records that the state is unknown
            return False

    async def _cancel_quietly(self, runtime_session: RuntimeSession, receipt) -> None:
        try:
            await self._provider.cancel(runtime_session, receipt)
        except Exception:  # noqa: BLE001 - scheduler records the exact terminal state
            return

    async def _set_binding_status_quietly(self, binding_id: str, status: BindingStatus) -> None:
        try:
            async with self._db.unit_of_work() as uow:
                await uow.runtime_bindings.set_status(binding_id, status.value, now=_now())
                await uow.commit()
        except Exception:
            # The worker will already have written a typed run failure if this
            # operation was reached on an error path.  Do not replace it with
            # a secondary persistence exception that says less.
            return

    async def _mark_potentially_billed_quietly(self, run_id: str) -> None:
        """Read-then-write, deliberately separate from the terminal
        ``advance()`` call the scheduler makes right after this (plan section
        6.6 / remaining-plan W2-12/15): this runs from ``execute()``'s own
        ``finally``, before the exception that triggered it has even reached
        the scheduler's exception handler, and a terminal transition is not
        this method's job. ``Run.mark_potentially_billed`` bumps version, so
        the scheduler's own fresh ``uow.runs.get()`` immediately afterward
        sees it and its transition preserves the flag (nothing in
        ``advance()``/``transition()`` touches it)."""
        try:
            async with self._db.unit_of_work() as uow:
                run = await uow.runs.get(run_id)
                if run is None:
                    return
                await uow.runs.update(run.mark_potentially_billed(), expected_version=run.version)
                await uow.commit()
        except Exception:
            # Best-effort audit enrichment; never let it mask the real
            # terminal failure the scheduler is about to record.
            return
