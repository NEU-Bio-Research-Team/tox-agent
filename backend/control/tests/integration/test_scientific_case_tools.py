"""get_scientific_case / update_scientific_case through the real runner (ADR 0012)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from toxagent.application import scientific_case_service as service
from toxagent.application.create_analysis import CreateAnalysis
from toxagent.application.policy import Actor
from toxagent.config import PolicySettings
from toxagent.domain.message import Message, Role
from toxagent.domain.run import Intent, Lane, Run
from toxagent.domain.session import Session
from toxagent.tools.bootstrap import build_registry
from toxagent.tools.registry import ToolContext
from toxagent.tools.runner import ToolRunner
from tests.support.predictor import ASPIRIN, StubPredictor

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 9, 25, tzinfo=timezone.utc)
ACTOR = Actor(subject_id="user-1")
CASE_TOOLS = {"get_scientific_case", "update_scientific_case"}


async def scenario(db, *, attach: bool = True):
    session = Session.create("user-1", now=NOW)
    message = Message.create(session.id, Role.USER, 1, now=NOW)
    run = Run.create(session.id, message.id, Lane.AGENTIC, Intent.DECISION_SUPPORT, now=NOW)
    async with db.unit_of_work() as uow:
        await uow.sessions.add(session)
        await uow.messages.add(message)
        await uow.runs.add(run)
        await uow.commit()
    stub = StubPredictor()
    client = stub.client()
    analysis = CreateAnalysis(db, client, PolicySettings())
    registry = build_registry(db, client, analysis)
    runner = ToolRunner(registry, db, max_calls_per_run=20)
    context = ToolContext(
        session_id=session.id, run_id=run.id, actor=ACTOR, profile="decision_support",
        deadline_at=datetime.now(timezone.utc) + timedelta(seconds=60),
    )
    result = await analysis.execute(
        actor=ACTOR, session_id=session.id, run_id=run.id, smiles=ASPIRIN, owns_run=False,
    )
    if attach:
        async with db.unit_of_work() as uow:
            await service.open_or_continue(
                uow, session_id=session.id, analysis_id=result.snapshot.id, run_id=run.id,
                goal="Is aspirin a hERG concern for this programme?",
                subject_refs=[f"analysis:{result.snapshot.id}"],
            )
            await uow.commit()
    return runner, context, registry, result


@pytest.fixture
def case_flag(monkeypatch):
    monkeypatch.setenv("TOXAGENT_FLAG_SCIENTIFIC_CASE_V1", "1")


async def _prediction_observation(runner, context, analysis_id) -> tuple[str, str]:
    result = await runner.call(context, "get_analysis_slice", {
        "analysis_id": analysis_id, "section": "herg", "fields": ["probability_blocker"],
    })
    value = result["model_view"]["values"]["probability_blocker"]
    return value["observation_id"], value["field_path"]


async def test_with_the_flag_off_the_case_tools_do_not_exist(db):
    runner, context, registry, _ = await scenario(db, attach=False)
    assert not CASE_TOOLS & {tool.name for tool in registry.visible_for("decision_support")}
    denied = await runner.call(context, "get_scientific_case", {})
    assert denied["error"]["code"] == "tool_denied"


async def test_the_flag_adds_exactly_the_case_tools_and_nothing_else(db, monkeypatch):
    _, _, off, _ = await scenario(db, attach=False)
    monkeypatch.setenv("TOXAGENT_FLAG_SCIENTIFIC_CASE_V1", "1")
    _, _, on, _ = await scenario(db, attach=False)
    off_names = [t.name for t in off.visible_for("decision_support")]
    on_names = [t.name for t in on.visible_for("decision_support")]
    assert set(on_names) - set(off_names) == CASE_TOOLS
    assert [d for d in on.descriptors("decision_support") if d["name"] not in CASE_TOOLS] == \
        off.descriptors("decision_support")


async def test_a_model_frames_and_grounds_the_case(db, case_flag):
    runner, context, _, analysis = await scenario(db)
    observation_id, field_path = await _prediction_observation(
        runner, context, analysis.snapshot.id
    )
    read = await runner.call(context, "get_scientific_case", {})
    assert read["status"] == "completed", read
    assert read["model_view"]["question"].startswith("Is aspirin")
    updated = await runner.call(context, "update_scientific_case", {"operations": [
        {"op": "add_hypothesis", "statement": "Aspirin blocks hERG at relevant exposure",
         "hypothesis_kind": "model_signal",
         "refutation_condition": "A patch-clamp IC50 far above free Cmax"},
        {"op": "add_hypothesis", "statement": "The score reflects the model, not the compound",
         "hypothesis_kind": "alternative_explanation",
         "refutation_condition": "Direct hERG data agreeing with the score"},
        {"op": "record_evidence", "claim": "The model's hERG blocker probability for aspirin",
         "source_class": "predictor_fact", "source_ref": f"observation:{observation_id}",
         "locator": field_path, "stance": "contextual", "directness": "direct",
         "hypothesis_ids": ["h1"]},
        {"op": "record_uncertainty", "uncertainty_kind": "missing_exposure",
         "description": "No free Cmax for the intended use", "severity": "blocking",
         "hypothesis_ids": ["h1"]},
        {"op": "record_action", "action": "counterevidence_search",
         "purpose": "look for direct hERG data showing no block", "decision": "continue",
         "hypothesis_ids": ["h1"]},
        {"op": "propose_next_test", "test": "Manual patch clamp at three concentrations",
         "rationale": "Separates a real block from a model artefact",
         "discriminates": ["h1", "h2"], "expected_readouts": ["IC50 < 10 µM supports h1"]},
    ]})
    assert updated["status"] == "completed", updated
    view = updated["model_view"]
    assert view["issued_ids"]["hypotheses"] == ["h1", "h2"]
    assert view["issued_ids"]["evidence"] == ["e1"]
    assert view["coverage"]["with_any_source"] == 1
    # A model score is a source, not independent direct evidence.
    assert view["coverage"]["with_independent_direct_evidence"] == 0
    assert view["coverage"]["with_counterevidence_considered"] == 1


async def test_an_evidence_ref_must_exist_and_be_of_the_right_kind(db, case_flag):
    runner, context, _, analysis = await scenario(db)
    observation_id, _ = await _prediction_observation(runner, context, analysis.snapshot.id)
    await runner.call(context, "update_scientific_case", {"operations": [
        {"op": "add_hypothesis", "statement": "h", "hypothesis_kind": "other",
         "refutation_condition": "r"},
    ]})
    wrong_kind = await runner.call(context, "update_scientific_case", {"operations": [
        {"op": "record_evidence", "claim": "c", "source_class": "explanation_fact",
         "source_ref": f"observation:{observation_id}", "stance": "contextual"},
    ]})
    assert wrong_kind["error"]["code"] == "invalid_request"
    assert "prediction observation" in wrong_kind["error"]["message"]
    missing = await runner.call(context, "update_scientific_case", {"operations": [
        {"op": "record_evidence", "claim": "c", "source_class": "external_experimental",
         "source_ref": "evidence:evd_" + "9" * 32, "stance": "supports", "hypothesis_ids": ["h1"]},
    ]})
    assert "not an accepted evidence record" in missing["error"]["message"]


async def test_a_refused_operation_applies_none_of_the_batch(db, case_flag):
    runner, context, _, _ = await scenario(db)
    refused = await runner.call(context, "update_scientific_case", {"operations": [
        {"op": "add_hypothesis", "statement": "h", "hypothesis_kind": "other",
         "refutation_condition": "r"},
        {"op": "revise_hypothesis", "hypothesis_id": "h1", "status": "supported",
         "reason": "no evidence"},
    ]})
    assert refused["error"]["code"] == "invalid_request"
    assert refused["error"]["message"].startswith("no operation was applied")
    read = await runner.call(context, "get_scientific_case", {})
    assert read["model_view"]["hypotheses"] == []


async def test_a_run_without_a_case_is_told_so(db, case_flag):
    runner, context, _, _ = await scenario(db, attach=False)
    result = await runner.call(context, "get_scientific_case", {})
    assert "not attached to a scientific case" in result["error"]["message"]
