"""Recovery eligibility is strict before persistence creates the new run."""
from __future__ import annotations

from datetime import datetime, timezone

from toxagent.application.run_scheduler import _can_recover_runtime_loss
from toxagent.domain.ids import new_id
from toxagent.domain.run import Intent, Lane, Run, RunStatus

NOW = datetime(2026, 9, 4, tzinfo=timezone.utc)


def _run(**overrides) -> Run:
    bound = overrides.pop("bound", False)
    run = Run.create(
        new_id("ses"), new_id("msg"), Lane.AGENTIC, Intent.REPORT_QA, now=NOW,
        **overrides,
    )
    if bound:
        run = run.transition(RunStatus.RUNNING, now=NOW, runtime_binding_id=new_id("rtb"))
    return run


def test_only_a_bound_first_attempt_lost_runtime_is_recoverable():
    run = _run(bound=True)
    assert _can_recover_runtime_loss(
        run, status=RunStatus.FAILED, failure_code="runtime_unavailable"
    )


def test_runtime_unavailable_before_a_binding_is_not_a_hidden_retry():
    run = _run()
    assert not _can_recover_runtime_loss(
        run, status=RunStatus.FAILED, failure_code="runtime_unavailable"
    )


def test_a_recovery_run_never_recovers_again_automatically():
    original = _run(bound=True)
    recovery = Run.create(
        original.session_id,
        original.trigger_message_id,
        original.lane,
        original.intent,
        now=NOW,
        recovery_of_run_id=original.id,
    ).transition(RunStatus.RUNNING, now=NOW, runtime_binding_id=new_id("rtb"))
    assert not _can_recover_runtime_loss(
        recovery, status=RunStatus.FAILED, failure_code="runtime_unavailable"
    )
