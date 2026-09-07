"""Tool execution: policy first, then the handler.

Everything a tool call must satisfy before any work happens lives here, in one
order, for every tool: visibility, per-run budget, duplicate detection, argument
validation, deadline. A handler is only reached once all of that has passed, so
no handler needs to re-implement any of it and none of them can forget.

Timeouts are enforced against both the tool's own hard timeout and the run
deadline, whichever is nearer. A tool that outlives its run would be writing
into a run that has already reported an outcome.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError

from ..domain.errors import ToolDenied, ToxAgentError
from ..domain.events import EventType
from ..domain.ids import TOOL_CALL, new_id
from ..domain.provenance import content_sha256
from . import envelope
from .definitions.answer import ANSWER_TOOL_NAME
from .registry import ToolContext, ToolRegistry

log = logging.getLogger("toxagent.tools")

#: Identical arguments to the same tool this many times in one run is a loop,
#: not a retry (plan section 14.5).
MAX_IDENTICAL_CALLS = 2


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ToolRunner:
    def __init__(
        self,
        registry: ToolRegistry,
        database,
        *,
        max_calls_per_run: int = 12,
    ) -> None:
        self._registry = registry
        self._db = database
        self._max_calls = max_calls_per_run

    async def call(
        self, context: ToolContext, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        call_id = new_id(TOOL_CALL)
        context = ToolContext(**{**context.__dict__, "call_id": call_id})
        started = time.perf_counter()

        definition = self._registry.get(tool_name)
        if definition is None or not self._registry.is_visible(tool_name, context.profile):
            # Same answer for "no such tool" and "not in your profile": a model
            # that can tell them apart can enumerate the tools it is denied.
            return envelope.failed(
                call_id=call_id, tool_name=tool_name, code="tool_denied",
                message=f"{tool_name} is not available to this run",
            )

        arguments_hash = content_sha256({"tool": tool_name, "arguments": arguments})

        try:
            payload = definition.input_model.model_validate(arguments)
        except ValidationError as exc:
            return envelope.failed(
                call_id=call_id, tool_name=tool_name, code="invalid_request",
                message="the arguments do not match this tool's schema",
                # ``ctx`` carries the raw ValueError raised by a field
                # validator, and ``url`` is noise for a model. Both must go:
                # the MCP envelope is JSON, and an exception object is not
                # serializable, which would turn a correctable validation
                # failure into a dead tool call.
                details={
                    "errors": exc.errors(include_url=False, include_context=False)[:5]
                },
            )

        try:
            await self._reserve(context, tool_name, arguments_hash)
        except ToxAgentError as exc:
            return envelope.failed(
                call_id=call_id, tool_name=tool_name, code=exc.code, message=exc.message,
                retryable=exc.retryable, details=exc.detail,
            )

        timeout = self._budget(context, definition.hard_timeout_s)
        try:
            output = await asyncio.wait_for(definition.handler(context, payload), timeout)
        except asyncio.TimeoutError:
            return await self._finish_error(
                context, tool_name, started, "tool_timeout",
                f"{tool_name} exceeded its {definition.hard_timeout_s:g}s budget", retryable=True,
            )
        except asyncio.CancelledError:
            await self._finish_error(
                context, tool_name, started, "cancelled", "the run was cancelled"
            )
            raise
        except ToxAgentError as exc:
            return await self._finish_error(
                context, tool_name, started, exc.code, exc.message,
                retryable=exc.retryable, details=exc.detail,
                retry_after_ms=getattr(exc, "retry_after_ms", None),
            )
        except Exception as exc:  # noqa: BLE001 — never leaks as a success body
            log.exception("tool %s failed unexpectedly", tool_name)
            return await self._finish_error(
                context, tool_name, started, "internal_error", type(exc).__name__
            )

        duration_ms = int((time.perf_counter() - started) * 1000)
        result = envelope.completed(
            call_id=call_id, tool_name=tool_name, canonical=output.canonical,
            model_view=output.model_view, ui_view=output.ui_view,
            observation_ids=output.observation_ids, provenance=output.provenance,
            attachments=output.attachments, duration_ms=duration_ms,
        )
        await self._record_finish(context, tool_name, output.observation_ids, duration_ms)
        return result

    # --- admission -----------------------------------------------------

    async def _reserve(self, context: ToolContext, tool_name: str, arguments_hash: str) -> None:
        """Atomically admit and reserve one call, or deny it with an audit
        trail (plan section 14.5).

        The budget check and the reservation happen as one statement
        (``SqlToolCallStore.try_reserve``) so several concurrent calls cannot
        each observe room under the budget before any of them has actually
        taken a slot — the exact race a separate "count, then insert" allows.
        ``submit_grounded_answer`` is exempt from the call-count budget
        (``max_calls=None``): a run that spent its budget on read tools must
        still be able to attempt the answer it's required to submit. Every
        denial — terminal run, cancelled run, budget, or duplicate — leaves an
        audit row and a ``TOOL_FAILED`` event, not just a response the model
        sees and the database never records.
        """
        max_calls = None if tool_name == ANSWER_TOOL_NAME else self._max_calls
        denial: ToolDenied | None = None
        async with self._db.unit_of_work() as uow:
            run = await uow.runs.get(context.run_id)
            if run is None or run.is_terminal:
                denial = ToolDenied(
                    "this run is no longer accepting tool calls", run_id=context.run_id
                )
            elif await uow.runs.cancel_requested(context.run_id):
                denial = ToolDenied("this run has been cancelled", run_id=context.run_id)
            else:
                reserved = await uow.tool_calls.try_reserve(
                    call_id=context.call_id, session_id=context.session_id,
                    run_id=context.run_id, tool_name=tool_name,
                    arguments_sha256=arguments_hash, now=_now(),
                    max_calls=max_calls, max_identical=MAX_IDENTICAL_CALLS,
                )
                if reserved:
                    uow.emit(
                        session_id=context.session_id, type=EventType.TOOL_STARTED,
                        entity_type="tool_call", entity_id=context.call_id,
                        run_id=context.run_id, payload={"tool_name": tool_name},
                    )
                    await uow.commit()
                    return

                total = await uow.tool_calls.count_for_run(context.run_id)
                repeats = await uow.tool_calls.duplicate_count(
                    context.run_id, tool_name, arguments_hash
                )
                if max_calls is not None and total >= max_calls:
                    denial = ToolDenied(
                        f"this run has reached its budget of {max_calls} tool calls",
                        tool_calls=total, max_calls=max_calls, remaining_calls=0,
                    )
                else:
                    denial = ToolDenied(
                        f"{tool_name} has already been called {repeats} times with these exact "
                        "arguments in this run",
                        tool_name=tool_name,
                    )

            await uow.tool_calls.record_denied(
                call_id=context.call_id, session_id=context.session_id,
                run_id=context.run_id, tool_name=tool_name, arguments_sha256=arguments_hash,
                error_code=denial.code, now=_now(),
            )
            uow.emit(
                session_id=context.session_id, type=EventType.TOOL_FAILED,
                entity_type="tool_call", entity_id=context.call_id, run_id=context.run_id,
                payload={"tool_name": tool_name, "error_code": denial.code},
            )
            await uow.commit()
        raise denial

    def _budget(self, context: ToolContext, hard_timeout_s: float) -> float:
        remaining = (context.deadline_at - _now()).total_seconds()
        return max(0.1, min(hard_timeout_s, remaining))

    # --- bookkeeping -------------------------------------------------------

    async def _record_finish(
        self, context: ToolContext, tool_name: str, observation_ids: tuple[str, ...], duration_ms: int
    ) -> None:
        async with self._db.unit_of_work() as uow:
            await uow.tool_calls.finish(
                context.call_id, status="completed", error_code=None,
                observation_ids=list(observation_ids), duration_ms=duration_ms, now=_now(),
            )
            uow.emit(
                session_id=context.session_id, type=EventType.TOOL_COMPLETED,
                entity_type="tool_call", entity_id=context.call_id, run_id=context.run_id,
                payload={
                    "tool_name": tool_name, "observation_ids": list(observation_ids),
                    "duration_ms": duration_ms,
                },
            )
            await uow.commit()

    async def _finish_error(
        self,
        context: ToolContext,
        tool_name: str,
        started: float,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
        retry_after_ms: int | None = None,
    ) -> dict[str, Any]:
        duration_ms = int((time.perf_counter() - started) * 1000)
        async with self._db.unit_of_work() as uow:
            await uow.tool_calls.finish(
                context.call_id, status="error", error_code=code, observation_ids=[],
                duration_ms=duration_ms, now=_now(),
            )
            uow.emit(
                session_id=context.session_id, type=EventType.TOOL_FAILED,
                entity_type="tool_call", entity_id=context.call_id, run_id=context.run_id,
                payload={"tool_name": tool_name, "error_code": code},
            )
            await uow.commit()
        return envelope.failed(
            call_id=context.call_id, tool_name=tool_name, code=code, message=message,
            retryable=retryable, details=details, retry_after_ms=retry_after_ms,
            duration_ms=duration_ms,
        )
