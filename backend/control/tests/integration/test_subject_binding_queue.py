"""A queued run keeps the subject it was bound to (#44).

The binding is written into the run's durable envelope when the run is
accepted. The worker that claims it later executes it with that binding --
not with whatever the session selects by then -- and a worker that predates
bindings leaves the run for one that understands it rather than running it
against the wrong subject.
"""
from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from toxagent.application.runs import envelope
from toxagent.application.runs.scheduler import RunContext, RunScheduler
from toxagent.application.runs.transitions import advance
from toxagent.domain.activity import TargetIdentity
from toxagent.domain.ids import SCREENING, new_id
from toxagent.domain.run import Intent, RunStatus
from toxagent.domain.subject import RunSubjectBinding, SubjectIdentity, SubjectKind
from tests.integration.test_external_workers import _accept, _seed, _status, _until

pytestmark = pytest.mark.anyio


def _binding(target_id: str) -> RunSubjectBinding:
    return RunSubjectBinding(
        kind=SubjectKind.ACTIVITY_SCREENING, subject_ref={"screening_id": new_id(SCREENING)},
        identity=SubjectIdentity(canonical_smiles="CCO", target=TargetIdentity(target_id=target_id),
                                 assay_context="IC50|wild_type"),
    )


def _recording(db, seen: list[RunContext]):
    async def handler(context: RunContext) -> None:
        seen.append(context)
        async with db.unit_of_work() as uow:
            run = await uow.runs.get(context.run_id)
            await advance(uow, run, RunStatus.COMPLETED)
            await uow.commit()

    return handler


async def test_the_claiming_worker_runs_the_subject_bound_at_admission(db):
    seen: list[RunContext] = []
    bound = _binding("EGFR")
    context = replace(await _seed(db, Intent.DECISION_SUPPORT), subject=bound.to_dict())
    api = RunScheduler(db, external=True)
    await _accept(db, api, context)

    async with db.unit_of_work() as uow:
        job = await uow.run_jobs.get(context.run_id)
    assert job["envelope"]["v"] == 2
    assert job["envelope"]["subject"]["binding_sha256"] == bound.binding_sha256

    worker = RunScheduler(db, external=True)
    worker.register(Intent.DECISION_SUPPORT, _recording(db, seen))
    assert await worker.adopt() == 1
    await _until(lambda: _done(db, context.run_id))
    (executed,) = seen
    assert RunSubjectBinding.from_dict(executed.subject) == bound


async def test_a_worker_that_cannot_read_bindings_leaves_the_run(db, monkeypatch):
    seen: list[RunContext] = []
    context = replace(await _seed(db, Intent.DECISION_SUPPORT), subject=_binding("KCNH2").to_dict())
    await _accept(db, RunScheduler(db, external=True), context)

    old = RunScheduler(db, external=True)
    old.register(Intent.DECISION_SUPPORT, _recording(db, seen))
    monkeypatch.setattr(envelope, "ENVELOPE_VERSION", 1)
    assert await old.adopt() == 0
    await asyncio.sleep(0.05)
    assert seen == [] and await _status(db, context.run_id) is RunStatus.QUEUED

    monkeypatch.setattr(envelope, "ENVELOPE_VERSION", 2)
    current = RunScheduler(db, external=True)
    current.register(Intent.DECISION_SUPPORT, _recording(db, seen))
    assert await current.adopt() == 1
    await _until(lambda: _done(db, context.run_id))
    assert seen[0].subject["identity"]["target"]["target_id"] == "KCNH2"


async def _done(db, run_id: str) -> bool:
    return await _status(db, run_id) is RunStatus.COMPLETED
