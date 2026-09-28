"""The server drives a report build, stage by stage (WS05 / PR-09).

Today ``BUILD_REPORT`` goes straight into the generic runtime gateway and the
model calls the whole workflow itself. The audit measured what that costs: a
minimal hERG report loaded 18,149 input tokens on its first turn, finished at
28,408, took 155.5 seconds, called tools out of order, hit a version conflict,
and used the validator as a reactive planner. The gateway meanwhile advanced the
build through five stages at one timestamp and emitted one stage event, so the
audit trail said work had happened that had not started.

This module is the other shape. The control plane owns the order of work; the
model is invoked at exactly one stage boundary, and every stage is separately
checkpointed, separately retried and separately reported.

**It is a skeleton on purpose.** The handlers are injected, so this file holds
the loop, the checkpointing, the event discipline and the recovery rule, and
nothing about substances, predictions or evidence. That keeps the part that has
to be right — never repeat a billable stage, never emit an event for work that
has not begun — testable without a predictor, a provider or a runtime. It is off
by default behind ``report_orchestrator_v2``; PR-10 fills in the handlers.

Two rules the old path broke, stated once:

**An event means work.** ``REPORT_STAGE_CHANGED`` is emitted when a stage
actually starts and when it actually settles. Never as a batch, never before
dispatch, never for a stage nothing is going to do.

**A checkpoint is permission to skip, not a note.** A resumed build reuses a
settled stage's output refs and does not call its handler again. The whole
reason a 155-second build is recoverable rather than repeatable is that its
explanation stage does not run twice.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Mapping, Protocol

from ...domain.errors import DeadlineExceeded
from ...domain.report import BuildStage, ReportBuild
from .stages import (
    PlannedStage,
    StageCheckpoint,
    StageStatus,
    gaps_from_checkpoints,
    plan_stages,
    read_checkpoints,
    remaining_stages,
    reusable_outputs,
    stage_progress,
    write_checkpoint,
)

log = logging.getLogger("toxagent.report.orchestrator")


def _now() -> datetime:
    return datetime.now(timezone.utc)


class StageSkipped(Exception):
    """A handler declining to do work, with a reason.

    Distinct from an exception: "the provider had nothing for this compound" is
    an outcome the report has to state, not a failure to recover from.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class StageContext:
    """What a handler is given. Everything server-known, nothing model-supplied."""

    build: ReportBuild
    #: Output refs from the stages that already settled, keyed by stage value.
    completed: Mapping[str, Mapping[str, Any]]
    attempt: int
    deadline_at: datetime | None


#: A handler returns the refs of what it produced, raises ``StageSkipped`` to
#: decline, or raises anything else to fail the stage.
StageHandler = Callable[[StageContext], Awaitable[Mapping[str, Any]]]


class BuildStore(Protocol):
    """The persistence seam, narrow on purpose.

    A protocol rather than the unit-of-work, so the loop below can be tested
    against an in-memory double without a database — which is the difference
    between a recovery test that runs in milliseconds and one nobody runs.
    """

    async def load(self, build_id: str) -> ReportBuild | None: ...
    async def save(self, build: ReportBuild) -> None: ...


class EventSink(Protocol):
    async def emit(self, build: ReportBuild, event: str, payload: dict) -> None: ...


@dataclass(frozen=True, slots=True)
class OrchestrationResult:
    build: ReportBuild
    #: Stages that actually ran in this pass. Empty on a resume that had
    #: nothing left to do, which is a correct and important outcome.
    executed: tuple[str, ...]
    reused: tuple[str, ...]
    gaps: tuple[dict[str, str], ...]


class ReportOrchestrator:
    def __init__(
        self,
        *,
        store: BuildStore,
        handlers: Mapping[BuildStage, StageHandler],
        events: EventSink,
        clock: Callable[[], datetime] = _now,
    ) -> None:
        self._store = store
        self._handlers = dict(handlers)
        self._events = events
        self._clock = clock

    async def run(self, build_id: str) -> OrchestrationResult:
        build = await self._store.load(build_id)
        if build is None:
            raise LookupError(f"no such report build: {build_id}")
        if build.is_terminal:
            # A terminal build is finished, including a cancelled one. Running
            # it again would re-emit events for work that is over.
            checkpoints = read_checkpoints(build.stage_state)
            return OrchestrationResult(
                build, (), tuple(checkpoints), gaps_from_checkpoints(checkpoints.values())
            )

        planned = plan_stages(build.request)
        checkpoints = read_checkpoints(build.stage_state)
        todo = remaining_stages(planned, checkpoints)
        reused = tuple(
            item.stage.value for item in planned if item not in todo
        )

        executed: list[str] = []
        for item in todo:
            self._check_deadline(build)
            build = await self._run_stage(build, item)
            checkpoints = read_checkpoints(build.stage_state)
            executed.append(item.stage.value)
            if checkpoints[item.stage.value].status is StageStatus.FAILED:
                break

        checkpoints = read_checkpoints(build.stage_state)
        return OrchestrationResult(
            build=build,
            executed=tuple(executed),
            reused=reused,
            gaps=gaps_from_checkpoints(checkpoints.values()),
        )

    # --- one stage ----------------------------------------------------------

    async def _run_stage(self, build: ReportBuild, item: PlannedStage) -> ReportBuild:
        now = self._clock()
        checkpoints = read_checkpoints(build.stage_state)
        previous = checkpoints.get(item.stage.value)
        attempt = (previous.attempt + 1) if previous is not None else 1
        checkpoint = StageCheckpoint(
            stage=item.stage.value, status=StageStatus.RUNNING
        ).started(now=now, attempt=attempt)

        if not item.will_run:
            # Walked, and correctly did nothing. Recorded as skipped with its
            # reason rather than omitted, so the report can tell "nobody
            # looked" from "the search found nothing" (P0-2).
            build = await self._settle(
                build, checkpoint.skipped(now=now, reason=item.skip_reason), item
            )
            return build

        build = await self._advance_to(build, item.stage, now=now)
        build = await self._persist(build, checkpoint)
        await self._events.emit(
            build,
            "report.stage_changed",
            {
                "stage": item.stage.value,
                "status": StageStatus.RUNNING.value,
                "attempt": attempt,
                **stage_progress(plan_stages(build.request), read_checkpoints(build.stage_state)),
            },
        )

        handler = self._handlers.get(item.stage)
        if handler is None:
            return await self._settle(
                build,
                checkpoint.skipped(
                    now=self._clock(),
                    reason=f"no handler is registered for {item.stage.value}",
                ),
                item,
            )

        context = StageContext(
            build=build,
            completed=reusable_outputs(read_checkpoints(build.stage_state)),
            attempt=attempt,
            deadline_at=build.deadline_at,
        )
        try:
            output = await handler(context)
        except StageSkipped as skipped:
            return await self._settle(
                build, checkpoint.skipped(now=self._clock(), reason=skipped.reason), item
            )
        except DeadlineExceeded:
            raise
        except Exception as exc:  # noqa: BLE001 - recorded, then re-raised by the caller's policy
            log.warning(
                "report stage failed",
                extra={
                    "report_build_id": build.id,
                    "stage": item.stage.value,
                    "attempt": attempt,
                    "error": type(exc).__name__,
                },
            )
            return await self._settle(
                build,
                checkpoint.failed(
                    now=self._clock(), reason=f"{type(exc).__name__}: {exc}"
                ),
                item,
            )

        return await self._settle(
            build, checkpoint.completed(now=self._clock(), output=output), item
        )

    async def _settle(
        self, build: ReportBuild, checkpoint: StageCheckpoint, item: PlannedStage
    ) -> ReportBuild:
        build = await self._persist(build, checkpoint)
        await self._events.emit(
            build,
            "report.stage_changed",
            {
                "stage": checkpoint.stage,
                "status": checkpoint.status.value,
                "attempt": checkpoint.attempt,
                "detail": checkpoint.detail,
                **stage_progress(plan_stages(build.request), read_checkpoints(build.stage_state)),
            },
        )
        return build

    async def _persist(
        self, build: ReportBuild, checkpoint: StageCheckpoint
    ) -> ReportBuild:
        build = replace(
            build,
            stage_state=write_checkpoint(build.stage_state, checkpoint),
            updated_at=self._clock(),
        )
        await self._store.save(build)
        return build

    async def _advance_to(
        self, build: ReportBuild, stage: BuildStage, *, now: datetime
    ) -> ReportBuild:
        """Move the build's own stage pointer, one legal step at a time.

        A skipped stage does not move the pointer: the pointer says where work
        is happening, and nothing is happening in a stage that was switched
        off. The aggregate's transition table still forbids an illegal move, so
        this cannot invent a path.
        """
        if build.stage is stage:
            return build
        try:
            return build.advance(stage, now=now)
        except Exception:
            # The pointer is a convenience for readers; the checkpoints are the
            # truth. A build whose pointer cannot legally reach this stage is
            # still allowed to record that the stage ran, rather than failing
            # the whole build over a bookkeeping field.
            log.warning(
                "report build stage pointer could not advance",
                extra={
                    "report_build_id": build.id,
                    "from": build.stage.value,
                    "to": stage.value,
                },
            )
            return build

    def _check_deadline(self, build: ReportBuild) -> None:
        if build.deadline_at is not None and self._clock() >= build.deadline_at:
            raise DeadlineExceeded(
                "the report build deadline elapsed between stages",
                report_build_id=build.id,
            )
