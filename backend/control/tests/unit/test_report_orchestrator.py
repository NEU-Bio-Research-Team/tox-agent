"""The server owns the order of work, and an event means work (PR-09).

The audit's build advanced through five stages at one timestamp before the
runtime was dispatched, emitted a single stage event, and then went quiet for
155 seconds. Recovery meant repeating everything, because nothing recorded what
had already been paid for.

These run against in-memory doubles: the loop, the checkpointing, the event
discipline and the recovery rule are the parts that have to be right, and they
should be testable without a predictor, a provider or a runtime.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from toxagent.application.report.orchestrator import (
    ReportOrchestrator,
    StageContext,
    StageSkipped,
)
from toxagent.application.report.stages import (
    CHECKPOINT_KEY,
    PIPELINE,
    StageStatus,
    plan_stages,
    read_checkpoints,
    stage_progress,
)
from toxagent.domain.errors import DeadlineExceeded
from toxagent.domain.report import BuildStage, ReportBuild, ReportBuildRequest

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc)
SESSION = "ses_" + "1" * 32
RUN = "run_" + "2" * 32
ANALYSIS = "ana_" + "3" * 32


def _request(**overrides) -> ReportBuildRequest:
    payload = {
        "session_id": SESSION,
        "analysis_id": ANALYSIS,
        "selected_endpoints": ("herg",),
        "include_explanations": True,
        "include_external_evidence": True,
    }
    payload.update(overrides)
    return ReportBuildRequest(**payload)


def _build(**overrides) -> ReportBuild:
    build = ReportBuild.start(
        session_id=SESSION,
        run_id=RUN,
        request=overrides.pop("request", _request()),
        now=NOW,
        deadline_at=overrides.pop("deadline_at", NOW + timedelta(hours=1)),
    )
    if overrides:
        from dataclasses import replace

        build = replace(build, **overrides)
    return build


class FakeStore:
    def __init__(self, build: ReportBuild) -> None:
        self.build = build
        self.saves = 0

    async def load(self, build_id: str):
        return self.build if build_id == self.build.id else None

    async def save(self, build: ReportBuild) -> None:
        self.build = build
        self.saves += 1


class FakeEvents:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def emit(self, build: ReportBuild, event: str, payload: dict) -> None:
        self.events.append((event, payload))

    def stages(self, status: str | None = None) -> list[str]:
        return [
            payload["stage"]
            for _, payload in self.events
            if status is None or payload.get("status") == status
        ]


def _handlers(calls: list[str], **overrides):
    """A handler per stage that records that it ran."""

    def make(stage: BuildStage):
        async def handler(context: StageContext) -> dict[str, Any]:
            calls.append(stage.value)
            return {"ref": f"{stage.value}-output"}

        return handler

    handlers = {stage: make(stage) for stage in PIPELINE}
    handlers.update(overrides)
    return handlers


async def _run(build: ReportBuild, handlers, clock=lambda: NOW):
    store, events = FakeStore(build), FakeEvents()
    orchestrator = ReportOrchestrator(
        store=store, handlers=handlers, events=events, clock=clock
    )
    return await orchestrator.run(build.id), store, events


# --- the happy path ----------------------------------------------------------


async def test_every_planned_stage_runs_once_in_order() -> None:
    calls: list[str] = []
    result, _, _ = await _run(_build(), _handlers(calls))
    assert calls == [stage.value for stage in PIPELINE]
    assert result.executed == tuple(stage.value for stage in PIPELINE)
    assert result.reused == ()


async def test_each_stage_emits_a_start_and_a_settle() -> None:
    """The audit's build emitted one event for five stages of work it had not
    done. Every stage now says when it begins and when it ends."""
    calls: list[str] = []
    _, _, events = await _run(_build(), _handlers(calls))
    running = events.stages("running")
    completed = events.stages("completed")
    assert running == [stage.value for stage in PIPELINE]
    assert completed == [stage.value for stage in PIPELINE]


async def test_no_event_is_emitted_before_its_stage_begins() -> None:
    """The first event must be the first stage starting, not a batch of five."""
    seen: list[tuple[str, str]] = []

    async def recording(context: StageContext) -> dict:
        seen.append(("handler", context.build.stage.value))
        return {}

    handlers = {stage: recording for stage in PIPELINE}
    _, _, events = await _run(_build(), handlers)
    first = events.events[0][1]
    assert first["stage"] == BuildStage.PREPARING_ANALYSIS.value
    assert first["status"] == "running"


async def test_a_checkpoint_records_what_the_stage_produced() -> None:
    calls: list[str] = []
    _, store, _ = await _run(_build(), _handlers(calls))
    checkpoints = read_checkpoints(store.build.stage_state)
    prepared = checkpoints[BuildStage.PREPARING_ANALYSIS.value]
    assert prepared.status is StageStatus.COMPLETED
    assert prepared.output_refs == {"ref": "preparing_analysis-output"}
    assert prepared.output_hash.startswith("sha256:")
    assert prepared.started_at and prepared.completed_at


# --- recovery ---------------------------------------------------------------


async def test_a_resumed_build_does_not_repeat_a_settled_stage() -> None:
    """This is the whole reason a 155-second build is recoverable rather than
    repeatable: the explanation stage does not run twice."""
    first_calls: list[str] = []
    _, store, _ = await _run(_build(), _handlers(first_calls))

    second_calls: list[str] = []
    result, _, events = await _run(store.build, _handlers(second_calls))
    assert second_calls == []
    assert result.executed == ()
    assert set(result.reused) == {stage.value for stage in PIPELINE}
    assert events.events == []


async def test_recovery_after_a_partial_run_continues_where_it_stopped() -> None:
    boom = {"count": 0}

    async def explode(context: StageContext) -> dict:
        boom["count"] += 1
        raise RuntimeError("the predictor was unavailable")

    first_calls: list[str] = []
    handlers = _handlers(first_calls, **{BuildStage.GENERATING_EXPLANATIONS: explode})
    result, store, _ = await _run(_build(), handlers)

    # The failing stage stops the pass; nothing after it ran.
    assert first_calls == [
        BuildStage.PREPARING_ANALYSIS.value,
        BuildStage.ASSEMBLING_SUBSTANCE.value,
        BuildStage.ASSEMBLING_PREDICTIONS.value,
    ]
    assert result.executed[-1] == BuildStage.GENERATING_EXPLANATIONS.value

    # The retry reruns only the failed stage and everything after it.
    second_calls: list[str] = []
    await _run(store.build, _handlers(second_calls))
    assert second_calls == [
        BuildStage.GENERATING_EXPLANATIONS.value,
        BuildStage.RESEARCHING_EVIDENCE.value,
        BuildStage.SYNTHESIZING.value,
        BuildStage.VALIDATING.value,
        BuildStage.RENDERING.value,
    ]


async def test_a_retry_increments_the_attempt_rather_than_hiding_it() -> None:
    async def explode(context: StageContext) -> dict:
        raise RuntimeError("nope")

    handlers = _handlers([], **{BuildStage.ASSEMBLING_SUBSTANCE: explode})
    _, store, _ = await _run(_build(), handlers)
    _, store, _ = await _run(store.build, handlers)
    checkpoint = read_checkpoints(store.build.stage_state)[
        BuildStage.ASSEMBLING_SUBSTANCE.value
    ]
    assert checkpoint.attempt == 2
    assert checkpoint.status is StageStatus.FAILED
    assert "nope" in checkpoint.detail


async def test_a_checkpoint_from_an_unknown_schema_is_not_trusted() -> None:
    build = _build()
    from dataclasses import replace

    build = replace(
        build,
        stage_state={
            CHECKPOINT_KEY: {
                BuildStage.PREPARING_ANALYSIS.value: {
                    "stage": BuildStage.PREPARING_ANALYSIS.value,
                    "status": "completed",
                    "schema_version": "stage-checkpoint-99",
                }
            }
        },
    )
    calls: list[str] = []
    await _run(build, _handlers(calls))
    assert BuildStage.PREPARING_ANALYSIS.value in calls


# --- skipping is an outcome, not an absence ---------------------------------


async def test_a_stage_switched_off_is_recorded_as_skipped_not_omitted() -> None:
    """'Nobody looked' and 'the search found nothing' are different facts, and
    a report that cannot tell them apart is the P0-2 contradiction."""
    calls: list[str] = []
    build = _build(request=_request(include_external_evidence=False))
    result, store, events = await _run(build, _handlers(calls))
    assert BuildStage.RESEARCHING_EVIDENCE.value not in calls
    checkpoint = read_checkpoints(store.build.stage_state)[
        BuildStage.RESEARCHING_EVIDENCE.value
    ]
    assert checkpoint.status is StageStatus.SKIPPED
    assert "did not ask for external evidence" in checkpoint.detail
    assert BuildStage.RESEARCHING_EVIDENCE.value in events.stages("skipped")


async def test_a_skipped_evidence_stage_produces_the_right_gap() -> None:
    build = _build(request=_request(include_external_evidence=False))
    result, _, _ = await _run(build, _handlers([]))
    reasons = {gap["reason"] for gap in result.gaps}
    assert "external_evidence_not_requested" in reasons
    assert "no_relevant_evidence" not in reasons


async def test_a_failed_evidence_stage_produces_a_different_gap() -> None:
    async def explode(context: StageContext) -> dict:
        raise RuntimeError("provider down")

    handlers = _handlers([], **{BuildStage.RESEARCHING_EVIDENCE: explode})
    result, _, _ = await _run(_build(), handlers)
    assert {gap["reason"] for gap in result.gaps} == {"provider_unavailable"}


async def test_a_handler_may_decline_with_a_reason() -> None:
    async def decline(context: StageContext) -> dict:
        raise StageSkipped("this compound has no resolvable identity")

    handlers = _handlers([], **{BuildStage.ASSEMBLING_SUBSTANCE: decline})
    _, store, _ = await _run(_build(), handlers)
    checkpoint = read_checkpoints(store.build.stage_state)[
        BuildStage.ASSEMBLING_SUBSTANCE.value
    ]
    assert checkpoint.status is StageStatus.SKIPPED
    assert checkpoint.detail == "this compound has no resolvable identity"


async def test_a_missing_handler_is_skipped_with_a_reason_not_silently() -> None:
    _, store, _ = await _run(_build(), {})
    checkpoint = read_checkpoints(store.build.stage_state)[
        BuildStage.PREPARING_ANALYSIS.value
    ]
    assert checkpoint.status is StageStatus.SKIPPED
    assert "no handler is registered" in checkpoint.detail


# --- a handler sees what earlier stages produced ----------------------------


async def test_a_later_stage_receives_the_earlier_stages_output_refs() -> None:
    seen: dict[str, Any] = {}

    async def capture(context: StageContext) -> dict:
        seen.update(context.completed)
        return {}

    handlers = _handlers([], **{BuildStage.SYNTHESIZING: capture})
    await _run(_build(), handlers)
    assert seen[BuildStage.ASSEMBLING_PREDICTIONS.value] == {
        "ref": "assembling_predictions-output"
    }


# --- deadlines and terminal builds ------------------------------------------


async def test_the_deadline_is_checked_between_stages() -> None:
    build = _build(deadline_at=NOW - timedelta(seconds=1))
    with pytest.raises(DeadlineExceeded):
        await _run(build, _handlers([]))


async def test_a_terminal_build_is_not_run_again() -> None:
    from dataclasses import replace

    build = replace(_build(), stage=BuildStage.CANCELLED)
    calls: list[str] = []
    result, _, events = await _run(build, _handlers(calls))
    assert calls == []
    assert result.executed == ()
    assert events.events == []


# --- what a progress UI gets -------------------------------------------------


async def test_progress_carries_codes_and_counts_never_prose() -> None:
    calls: list[str] = []
    _, store, events = await _run(_build(), _handlers(calls))
    progress = stage_progress(
        plan_stages(store.build.request), read_checkpoints(store.build.stage_state)
    )
    assert progress["total"] == len(PIPELINE)
    assert progress["completed"] == len(PIPELINE)
    assert progress["current_stage"] is None
    assert {row["status"] for row in progress["stages"]} == {"completed"}
    # Every payload is machine-readable: codes, counts, attempts. No sentence a
    # model wrote reaches a progress feed.
    for _, payload in events.events:
        assert set(payload) <= {
            "stage", "status", "attempt", "detail",
            "total", "completed", "current_stage", "stages",
        }


async def test_progress_names_the_stage_that_is_running() -> None:
    captured: list[dict] = []

    async def capture(context: StageContext) -> dict:
        captured.append(
            stage_progress(
                plan_stages(context.build.request),
                read_checkpoints(context.build.stage_state),
            )
        )
        return {}

    handlers = _handlers([], **{BuildStage.ASSEMBLING_PREDICTIONS: capture})
    await _run(_build(), handlers)
    assert captured[0]["current_stage"] == BuildStage.ASSEMBLING_PREDICTIONS.value
    assert captured[0]["completed"] == 2


async def test_an_unknown_build_is_a_lookup_error() -> None:
    store, events = FakeStore(_build()), FakeEvents()
    orchestrator = ReportOrchestrator(store=store, handlers={}, events=events)
    with pytest.raises(LookupError):
        await orchestrator.run("rpb_" + "9" * 32)
