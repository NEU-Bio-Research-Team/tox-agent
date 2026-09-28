"""Per-run evidence search/read budgets for decision_support (ADS plan
section 9.2, W4-04).

These are decision_support-only guardrails distinct from the generic
whole-run tool-call cap (``tests/integration/test_evidence_tools.py`` and
``ToolRunner``'s own budget already cover that): evidence_research's entire
purpose is intensive search and must not be throttled by this.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from toxagent.application.prediction.create_analysis import CreateAnalysis
from toxagent.application.policy import Actor
from toxagent.config import PolicySettings, ResearchSettings
from toxagent.domain.events import EventType
from toxagent.domain.message import Message, Role
from toxagent.domain.run import Intent, Lane, Run
from toxagent.domain.session import Session
from toxagent.tools.bootstrap import build_registry
from toxagent.tools.definitions.evidence import (
    DECISION_SUPPORT_MAX_EVIDENCE_READS_PER_RUN,
    DECISION_SUPPORT_MAX_SEARCHES_PER_RUN,
)
from toxagent.tools.registry import ToolContext
from toxagent.tools.runner import ToolRunner
from tests.support.predictor import ASPIRIN, StubPredictor
from tests.support.research import ACCEPTED_HIT, StubResearchProvider

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 9, 16, tzinfo=timezone.utc)
ACTOR = Actor(subject_id="user-1")
RESEARCH_SETTINGS = ResearchSettings(allowed_hosts=("www.ebi.ac.uk", "europepmc.org"))


async def scenario(db, *, profile="decision_support"):
    session = Session.create(ACTOR.subject_id, now=NOW)
    message = Message.create(session.id, Role.USER, 1, now=NOW)
    run = Run.create(session.id, message.id, Lane.AGENTIC, Intent.DECISION_SUPPORT, now=NOW)
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
        actor=ACTOR, session_id=session.id, run_id=run.id, smiles=ASPIRIN, owns_run=False,
    )
    provider = StubResearchProvider(hits=(ACCEPTED_HIT,))
    registry = build_registry(
        db, predictor, analysis_service, PolicySettings(),
        research_provider=provider, research_settings=RESEARCH_SETTINGS,
    )
    # Deliberately generous whole-run cap so only the tool-specific budget
    # below is what denies the over-limit call.
    runner = ToolRunner(registry, db, max_calls_per_run=100)
    context = ToolContext(
        session_id=session.id, run_id=run.id, actor=ACTOR, profile=profile,
        deadline_at=datetime.now(timezone.utc) + timedelta(seconds=60),
    )
    return runner, context, result.snapshot.id


async def test_the_nth_plus_one_search_is_denied_for_decision_support(db):
    runner, context, analysis_id = await scenario(db, profile="decision_support")
    for i in range(DECISION_SUPPORT_MAX_SEARCHES_PER_RUN):
        result = await runner.call(
            context, "search_toxicology_evidence",
            {"analysis_id": analysis_id, "query": f"hERG blockade {i}", "limit": 5},
        )
        assert result["status"] == "completed", result

    denied = await runner.call(
        context, "search_toxicology_evidence",
        {"analysis_id": analysis_id, "query": "one search too many", "limit": 5},
    )
    assert denied["status"] == "error"
    assert denied["error"]["code"] == "tool_denied"
    assert "budget" in denied["error"]["message"]


async def test_evidence_research_profile_is_not_throttled_by_the_decision_support_budget(db):
    runner, context, analysis_id = await scenario(db, profile="evidence_research")
    for i in range(DECISION_SUPPORT_MAX_SEARCHES_PER_RUN + 2):
        result = await runner.call(
            context, "search_toxicology_evidence",
            {"analysis_id": analysis_id, "query": f"hERG blockade {i}", "limit": 5},
        )
        assert result["status"] == "completed", result


async def test_the_nth_plus_one_evidence_read_is_denied_for_decision_support(db):
    runner, context, analysis_id = await scenario(db, profile="decision_support")
    search_result = await runner.call(
        context, "search_toxicology_evidence",
        {"analysis_id": analysis_id, "query": "hERG blockade", "limit": 5},
    )
    evidence_id = search_result["model_view"]["results"][0]["evidence_id"]

    # Distinct arguments each call: ToolRunner's own generic duplicate-call
    # cap (MAX_IDENTICAL_CALLS=2) would otherwise deny a repeat of the exact
    # same arguments before this budget gets a chance to.
    field_pool = [
        "title", "authors", "published_at", "source_type", "source_quality_tier",
        "identifier", "canonical_url", "abstract_or_excerpt", "normalized_facts",
    ]
    assert DECISION_SUPPORT_MAX_EVIDENCE_READS_PER_RUN <= len(field_pool)
    for i in range(DECISION_SUPPORT_MAX_EVIDENCE_READS_PER_RUN):
        result = await runner.call(
            context, "get_evidence_record",
            {"evidence_id": evidence_id, "fields": field_pool[: i + 1]},
        )
        assert result["status"] == "completed", result

    denied = await runner.call(
        context, "get_evidence_record", {"evidence_id": evidence_id, "fields": field_pool},
    )
    assert denied["status"] == "error"
    assert denied["error"]["code"] == "tool_denied"
