"""search_toxicology_evidence and get_evidence_record (plan section 8.4, Phase 5).

Exercises the tool handlers the same way ``tests/integration/test_tool_runner.py``
exercises the analysis tools — through the real registry/runner, a real
database, and a stub provider standing in only for the external network call.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from toxagent.application.create_analysis import CreateAnalysis
from toxagent.application.policy import Actor
from toxagent.config import PolicySettings, ResearchSettings
from toxagent.domain.evidence import SourceType
from toxagent.domain.events import EventType
from toxagent.domain.message import Message, Role
from toxagent.domain.run import Intent, Lane, Run, RunStatus
from toxagent.domain.session import Session
from toxagent.research.interfaces import SearchHit
from toxagent.tools.bootstrap import build_registry
from toxagent.tools.registry import ToolContext
from toxagent.tools.runner import ToolRunner
from tests.support.predictor import ASPIRIN, StubPredictor
from tests.support.research import ACCEPTED_HIT, OFF_ALLOWLIST_HIT, TITLELESS_HIT, StubResearchProvider

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 9, 5, tzinfo=timezone.utc)
ACTOR = Actor(subject_id="user-1")
RESEARCH_SETTINGS = ResearchSettings(allowed_hosts=("www.ebi.ac.uk", "europepmc.org"))


async def scenario(db, *, hits=(ACCEPTED_HIT,), profile="evidence_research", owner="user-1"):
    actor = Actor(subject_id=owner)
    session = Session.create(owner, now=NOW)
    message = Message.create(session.id, Role.USER, 1, now=NOW)
    run = Run.create(session.id, message.id, Lane.AGENTIC, Intent.EVIDENCE_RESEARCH, now=NOW)
    async with db.unit_of_work() as uow:
        await uow.sessions.add(session)
        await uow.messages.add(message)
        await uow.runs.add(run)
        uow.emit(
            session_id=session.id, type=EventType.SESSION_CREATED,
            entity_type="session", entity_id=session.id,
        )
        await uow.commit()
    predictor = StubPredictor().client()
    analysis_service = CreateAnalysis(db, predictor, PolicySettings())
    result = await analysis_service.execute(
        actor=actor, session_id=session.id, run_id=run.id, smiles=ASPIRIN, owns_run=False,
    )
    provider = StubResearchProvider(hits=hits)
    registry = build_registry(
        db, predictor, analysis_service, PolicySettings(),
        research_provider=provider, research_settings=RESEARCH_SETTINGS,
    )
    runner = ToolRunner(registry, db, max_calls_per_run=20)
    context = ToolContext(
        session_id=session.id, run_id=run.id, actor=actor, profile=profile,
        deadline_at=datetime.now(timezone.utc) + timedelta(seconds=60),
    )
    return runner, context, session, result.snapshot.id, provider


async def test_a_search_returns_compact_metadata_for_an_accepted_hit(db):
    runner, context, _, analysis_id, provider = await scenario(db)
    result = await runner.call(
        context, "search_toxicology_evidence",
        {"analysis_id": analysis_id, "query": "hERG blockade", "limit": 5},
    )
    assert result["status"] == "completed"
    view = result["model_view"]
    assert view["returned"] == 1
    assert view["rejected"] == 0
    hit_view = view["results"][0]
    assert hit_view["title"] == ACCEPTED_HIT.title
    assert hit_view["untrusted_external_content"] is True
    # Compact: no abstract in the search result itself (plan section 8.4).
    assert "abstract_or_excerpt" not in hit_view
    assert provider.calls[0]["query"] == "hERG blockade"


async def test_a_rejected_hit_is_stored_but_not_returned_to_the_model(db):
    runner, context, _, analysis_id, _ = await scenario(db, hits=(TITLELESS_HIT, ACCEPTED_HIT))
    result = await runner.call(
        context, "search_toxicology_evidence",
        {"analysis_id": analysis_id, "query": "x", "limit": 5},
    )
    view = result["model_view"]
    assert view["returned"] == 1
    assert view["rejected"] == 1
    assert len(view["results"]) == 1
    assert view["results"][0]["title"] == ACCEPTED_HIT.title


async def test_a_hit_off_the_host_allowlist_is_rejected(db):
    runner, context, _, analysis_id, _ = await scenario(db, hits=(OFF_ALLOWLIST_HIT,))
    result = await runner.call(
        context, "search_toxicology_evidence",
        {"analysis_id": analysis_id, "query": "x", "limit": 5},
    )
    view = result["model_view"]
    assert view["returned"] == 0
    assert view["rejected"] == 1


async def test_repeating_a_search_reuses_the_same_record_instead_of_duplicating_it(db):
    runner, context, session, analysis_id, _ = await scenario(db)
    first = await runner.call(
        context, "search_toxicology_evidence",
        {"analysis_id": analysis_id, "query": "hERG", "limit": 5},
    )
    second = await runner.call(
        context, "search_toxicology_evidence",
        {"analysis_id": analysis_id, "query": "hERG channel", "limit": 5},
    )
    first_id = first["model_view"]["results"][0]["evidence_id"]
    second_id = second["model_view"]["results"][0]["evidence_id"]
    assert first_id == second_id
    assert second["model_view"]["reused_from_this_session"] == 1

    async with runner._db.unit_of_work() as uow:  # noqa: SLF001 - test-only introspection
        stored = await uow.evidence.list_for_session(session.id)
    assert len(stored) == 1


async def test_get_evidence_record_returns_the_abstract_and_is_marked_untrusted(db):
    runner, context, _, analysis_id, _ = await scenario(db)
    search = await runner.call(
        context, "search_toxicology_evidence",
        {"analysis_id": analysis_id, "query": "hERG", "limit": 5},
    )
    evidence_id = search["model_view"]["results"][0]["evidence_id"]
    detail = await runner.call(context, "get_evidence_record", {"evidence_id": evidence_id})
    assert detail["status"] == "completed"
    view = detail["model_view"]
    assert view["abstract_or_excerpt"] == ACCEPTED_HIT.abstract_or_excerpt
    assert view["untrusted_external_content"] is True
    assert view["status"] == "accepted"


async def test_get_evidence_record_honours_a_field_selection(db):
    runner, context, _, analysis_id, _ = await scenario(db)
    search = await runner.call(
        context, "search_toxicology_evidence",
        {"analysis_id": analysis_id, "query": "hERG", "limit": 5},
    )
    evidence_id = search["model_view"]["results"][0]["evidence_id"]
    detail = await runner.call(
        context, "get_evidence_record", {"evidence_id": evidence_id, "fields": ["title"]}
    )
    view = detail["model_view"]
    assert set(view) == {"title", "evidence_id", "status", "untrusted_external_content"}


async def test_get_evidence_record_for_an_unknown_id_is_a_typed_not_found(db):
    runner, context, _, _, _ = await scenario(db)
    result = await runner.call(context, "get_evidence_record", {"evidence_id": "evd_" + "0" * 32})
    assert result["status"] == "error"
    assert result["error"]["code"] == "evidence_not_found"


async def test_evidence_from_one_session_is_invisible_to_another(db):
    runner_a, context_a, _, analysis_id_a, _ = await scenario(db, owner="user-1")
    search = await runner_a.call(
        context_a, "search_toxicology_evidence",
        {"analysis_id": analysis_id_a, "query": "hERG", "limit": 5},
    )
    evidence_id = search["model_view"]["results"][0]["evidence_id"]

    runner_b, context_b, _, _, _ = await scenario(db, owner="user-2")
    leaked = await runner_b.call(context_b, "get_evidence_record", {"evidence_id": evidence_id})
    assert leaked["status"] == "error"
    assert leaked["error"]["code"] == "evidence_not_found"


async def test_search_is_denied_outside_the_evidence_research_and_audit_profiles(db):
    runner, context, _, analysis_id, _ = await scenario(db, profile="report_qa")
    denied = await runner.call(
        context, "search_toxicology_evidence", {"analysis_id": analysis_id, "query": "x"}
    )
    assert denied["error"]["code"] == "tool_denied"


async def test_get_evidence_record_is_visible_to_the_audit_readonly_profile(db):
    runner, context, _, analysis_id, _ = await scenario(db, profile="evidence_research")
    search = await runner.call(
        context, "search_toxicology_evidence",
        {"analysis_id": analysis_id, "query": "hERG", "limit": 5},
    )
    evidence_id = search["model_view"]["results"][0]["evidence_id"]

    audit_context = ToolContext(
        session_id=context.session_id, run_id=context.run_id, actor=ACTOR,
        profile="audit_readonly", deadline_at=context.deadline_at,
    )
    result = await runner.call(audit_context, "get_evidence_record", {"evidence_id": evidence_id})
    assert result["status"] == "completed"

    submit_denied = await runner.call(
        audit_context, "submit_grounded_answer",
        {"schema_version": "grounded-answer-v1", "answer_markdown": "x", "claims": [],
         "limitations": [], "recommended_next_steps": []},
    )
    assert submit_denied["error"]["code"] == "tool_denied"
