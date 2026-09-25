"""ScientificCaseV1 persistence: snapshot, append-only log, dossiers (ADR 0012)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from toxagent.application import scientific_case_service as service
from toxagent.domain import scientific_case as sc
from toxagent.domain.errors import Conflict
from toxagent.domain.message import Message, Role
from toxagent.domain.run import Intent, Lane, Run
from toxagent.domain.session import Session

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 9, 25, tzinfo=timezone.utc)
EVD = "evd_" + "4" * 32


async def seeded(db, *, runs: int = 1):
    session = Session.create("user-1", now=NOW)
    created = []
    async with db.unit_of_work() as uow:
        await uow.sessions.add(session)
        for index in range(runs):
            message = Message.create(session.id, Role.USER, index + 1, now=NOW)
            run = Run.create(session.id, message.id, Lane.AGENTIC, Intent.DECISION_SUPPORT, now=NOW)
            await uow.messages.add(message)
            await uow.runs.add(run)
            created.append(run)
        await uow.commit()
    return session, created


async def open_case(db, session, run, *, analysis_id="ana_" + "a" * 32, goal="Is A a hERG concern?"):
    async with db.unit_of_work() as uow:
        case = await service.open_or_continue(
            uow, session_id=session.id, analysis_id=analysis_id, run_id=run.id, goal=goal,
            subject_refs=[f"analysis:{analysis_id}"],
        )
        await uow.commit()
    return case


async def test_the_first_run_opens_a_case_with_its_goal_as_the_question(db):
    session, (run,) = await seeded(db)
    case = await open_case(db, session, run)
    assert case.id.startswith("scase_")
    assert case.question == "Is A a hERG concern?"
    assert [r.run_id for r in case.runs] == [run.id]
    assert case.revision == 2


async def test_a_second_run_on_the_same_subject_continues_the_case(db):
    session, (first, second) = await seeded(db, runs=2)
    opened = await open_case(db, session, first)
    continued = await open_case(db, session, second, goal="Here is our patch clamp result")
    assert continued.id == opened.id
    assert continued.question == "Is A a hERG concern?"
    assert [r.run_id for r in continued.runs] == [first.id, second.id]


async def test_another_subject_opens_another_case(db):
    session, (first, second) = await seeded(db, runs=2)
    a = await open_case(db, session, first)
    b = await open_case(db, session, second, analysis_id="ana_" + "b" * 32)
    assert a.id != b.id


async def test_reattaching_the_same_run_writes_nothing(db):
    session, (run,) = await seeded(db)
    first = await open_case(db, session, run)
    again = await open_case(db, session, run)
    assert again.revision == first.revision


async def test_the_stored_snapshot_is_the_replay_of_the_stored_log(db):
    session, (run,) = await seeded(db)
    case = await open_case(db, session, run)
    async with db.unit_of_work() as uow:
        await service.apply_updates(uow, case_id=case.id, session_id=session.id, updates=[
            service.update("add_hypothesis", actor="model", run_id=run.id, statement="A blocks hERG",
                           kind="mechanism", refutation_condition="IC50 above 30 µM"),
            service.update("record_evidence", actor="model", run_id=run.id, claim="IC50 0.1 µM",
                           source_class="external_experimental", source_ref=f"evidence:{EVD}",
                           stance="supports", directness="direct", hypothesis_ids=["h1"]),
        ])
        await uow.commit()
    async with db.unit_of_work() as uow:
        stored = await uow.scientific_cases.get(case.id, session_id=session.id)
        log = await uow.scientific_cases.events(case.id, session_id=session.id)
    assert [u.revision for u in log] == [1, 2, 3, 4]
    assert [u.op for u in log] == ["open", "attach_run", "add_hypothesis", "record_evidence"]
    assert sc.replay(log) == stored
    assert stored.evidence[0].source_ref == f"evidence:{EVD}"


async def test_a_refused_update_writes_nothing_at_all(db):
    session, (run,) = await seeded(db)
    case = await open_case(db, session, run)
    async with db.unit_of_work() as uow:
        with pytest.raises(sc.InvalidCaseUpdate):
            await service.apply_updates(uow, case_id=case.id, session_id=session.id, updates=[
                service.update("add_hypothesis", actor="model", run_id=run.id, statement="ok",
                               kind="other", refutation_condition="x"),
                service.update("revise_hypothesis", actor="model", run_id=run.id,
                               hypothesis_id="h1", status="supported", reason="no evidence"),
            ])
    async with db.unit_of_work() as uow:
        stored = await uow.scientific_cases.get(case.id, session_id=session.id)
    assert stored.hypotheses == ()


async def test_a_stale_writer_loses_with_a_conflict(db):
    session, (run,) = await seeded(db)
    case = await open_case(db, session, run)
    stale = sc.apply(case, service.update("add_hypothesis", actor="model", run_id=run.id,
                                          statement="x", kind="other", refutation_condition="y"))
    async with db.unit_of_work() as uow:
        await service.apply_updates(uow, case_id=case.id, session_id=session.id, updates=[
            service.update("add_hypothesis", actor="model", run_id=run.id, statement="first",
                           kind="other", refutation_condition="y"),
        ])
        await uow.commit()
    async with db.unit_of_work() as uow:
        with pytest.raises(Conflict):
            await uow.scientific_cases.append(
                stale, [service.update("add_hypothesis", actor="model", statement="x",
                                       kind="other", refutation_condition="y")],
                expected_revision=case.revision, now=NOW,
            )


async def test_a_case_is_invisible_from_another_session(db):
    session, (run,) = await seeded(db)
    other, _ = await seeded(db)
    case = await open_case(db, session, run)
    async with db.unit_of_work() as uow:
        assert await uow.scientific_cases.get(case.id, session_id=other.id) is None
        assert await uow.scientific_cases.events(case.id, session_id=other.id) == []
        assert await uow.scientific_cases.case_id_for_run(run.id, session_id=other.id) is None


async def test_bookkeeping_skips_a_refused_update_and_keeps_the_rest(db):
    session, (run,) = await seeded(db)
    await open_case(db, session, run)
    relations = [
        {"proposition": "p", "source_class": "external_experimental", "source_id": EVD,
         "relation": "contextual", "directness": "direct"},
        {"proposition": "q", "source_class": "agent_synthesis", "source_id": "syn_x",
         "relation": "supports"},
    ]
    case = await service.advance(
        db, session_id=session.id, run_id=run.id,
        updates_for=service.answer_relation_updates(relations, run_id=run.id),
    )
    assert [e.source_ref for e in case.evidence] == [f"evidence:{EVD}"]
    assert case.evidence[0].actor == "server"


async def test_a_dossier_is_stored_once_per_run_and_read_back(db):
    session, (run,) = await seeded(db)
    case = await open_case(db, session, run)
    dossier = sc.compile_dossier(case, run_id=run.id, stop_reason="sufficient", answer_id=None)
    async with db.unit_of_work() as uow:
        await uow.scientific_cases.put_dossier(dossier, now=NOW)
        await uow.commit()
    async with db.unit_of_work() as uow:
        assert await uow.scientific_cases.get_dossier(run.id, session_id=session.id) == dossier
        assert await uow.scientific_cases.latest_dossier(case.id, session_id=session.id) == dossier
    async with db.unit_of_work() as uow:
        with pytest.raises(Exception):
            await uow.scientific_cases.put_dossier(dossier, now=NOW)
            await uow.commit()


def test_the_server_records_only_what_it_knows_is_uncertain():
    from types import SimpleNamespace

    ood = SimpleNamespace(
        predictor_response={"applicability": {"status": "out_of_domain", "reasons": ["contains boron"]}},
        unavailable_endpoints=("clintox",),
    )
    updates = service.analysis_uncertainties(ood, run_id="run_" + "6" * 32)
    kinds = [(u.payload["kind"], u.payload["severity"]) for u in updates]
    assert kinds == [("applicability_domain", "high"), ("missing_endpoint", "medium")]
    assert "contains boron" in updates[0].payload["description"]
    fine = SimpleNamespace(predictor_response={"applicability": {"status": "ok"}},
                           unavailable_endpoints=())
    assert service.analysis_uncertainties(fine, run_id="run_x") == []
    assert service.analysis_uncertainties(None, run_id="run_x") == []


async def test_recording_the_same_server_uncertainty_every_turn_is_idempotent(db):
    from types import SimpleNamespace

    session, (first, second) = await seeded(db, runs=2)
    snapshot = SimpleNamespace(
        predictor_response={"applicability": {"status": "limited", "reasons": ["rare element"]}},
        unavailable_endpoints=(),
    )
    for run in (first, second):
        async with db.unit_of_work() as uow:
            case = await service.open_or_continue(
                uow, session_id=session.id, analysis_id="ana_" + "a" * 32, run_id=run.id,
                goal="q", subject_refs=[],
                extra_updates=service.analysis_uncertainties(snapshot, run_id=run.id),
            )
            await uow.commit()
    assert [u.kind for u in case.uncertainties] == ["applicability_domain"]
