"""A deterministic vertical slice through the runtime gateway.

The scripted adapter is not a mock of ToolRunner: the script calls the exact
registry/runner used by MCP, so this test proves the product state machine,
tool authorization and grounded-answer boundary without a live model binary.
"""
from __future__ import annotations

import base64

import pytest

from toxagent.config import ResearchSettings
from toxagent.domain.run import Intent
from toxagent.harness.adapters.scripted import ScriptedRuntimeProvider
from toxagent.harness.gateway import AgentRuntimeGateway
from toxagent.persistence.object_store import InMemoryObjectStore, ObjectNotFound, ObjectRef
from tests.support.api import AUTH, OTHER_AUTH, api_client, settings, wait_for_run
from tests.support.ocr import stub_no_structure_detected, stub_success, stub_unavailable
from tests.support.predictor import ASPIRIN, StubPredictor
from tests.support.research import ACCEPTED_HIT, StubResearchProvider

pytestmark = pytest.mark.anyio

# W4-08 validates file signatures at the API boundary.  These are deliberately
# minimal headers: OCR parsing is owned by the stubbed toxocr boundary in this
# suite, so a full raster payload would only obscure the control-plane contract.
PNG_BYTES = b"\x89PNG\r\n\x1a\ncontrol-plane-test"


async def _install_scripted_runtime(client, script) -> None:
    """Attach a test-only in-process provider at the same composition seam an
    OpenCode/DSH adapter will use in deployment."""
    app = client.app
    provider = ScriptedRuntimeProvider(app.state.tool_registry, app.state.tool_runner, script)
    gateway = AgentRuntimeGateway(
        app.state.database,
        app.state.tool_registry,
        app.state.capability_tokens,
        provider,
        app.state.settings.runtime,
        create_analysis=app.state.create_analysis,
    )

    async def run_agentic(context) -> None:
        await gateway.execute(context)

    for intent in (Intent.REPORT_QA, Intent.ATTRIBUTION, Intent.EVIDENCE_RESEARCH):
        app.state.scheduler.register(intent, run_agentic)


async def _new_session(client) -> str:
    response = await client.post("/v1/sessions", json={}, headers=AUTH)
    assert response.status_code == 201, response.text
    return response.json()["session_id"]


async def _analyse(client, session_id: str, smiles: str = ASPIRIN) -> str:
    submitted = await client.post(
        f"/v1/sessions/{session_id}/messages",
        json={"molecule": {"smiles": smiles}},
        headers=AUTH,
    )
    assert submitted.status_code == 202, submitted.text
    await wait_for_run(client, session_id, submitted.json()["run_id"])
    state = await client.get(f"/v1/sessions/{session_id}", headers=AUTH)
    return state.json()["active_analysis"]["analysis_id"]


_analyse_aspirin = _analyse


async def test_a_scripted_runtime_can_commit_a_grounded_report_answer(db):
    analysis_id = ""

    async def script(turn) -> None:
        # This is intentionally unsafe prose from the runtime.  The gateway
        # receives its MESSAGE_DELTA but must never persist it before answer
        # validation succeeds.
        turn.say("This compound is safe to use.")
        turn.report_usage(
            {"input": 0, "output": 5, "cache": {"read": 0}},
            cost={"amount": "0.00125000", "currency": "usd"},
        )
        slice_result = await turn.call_tool(
            "get_analysis_slice",
            {
                "analysis_id": analysis_id,
                "section": "herg",
                "fields": ["probability_blocker"],
            },
        )
        assert slice_result["status"] == "completed"
        value = slice_result["model_view"]["values"]["probability_blocker"]
        submitted = await turn.call_tool(
            "submit_grounded_answer",
            {
                "schema_version": "grounded-answer-v1",
                "answer_markdown": "The predicted hERG blocker probability is 0.731.",
                "claims": [
                    {
                        "claim_id": "clm_" + "1" * 32,
                        "kind": "numeric",
                        "text": "The predicted hERG blocker probability is 0.731.",
                        "observation_id": value["observation_id"],
                        "field_path": value["field_path"],
                        "source_value": value["value"],
                        "rendered_value": "0.731",
                        "transform": "round:3",
                    }
                ],
                "limitations": [{"code": "uncalibrated_probability", "text": ""}],
                "recommended_next_steps": [],
            },
        )
        assert submitted["status"] == "completed"

    async with api_client(db, StubPredictor()) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        analysis_id = await _analyse_aspirin(client, session_id)

        submitted = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={
                "intent_hint": "ask_report",
                "content": [{"type": "text", "text": "What is the hERG result?"}],
            },
            headers=AUTH,
        )
        assert submitted.status_code == 202, submitted.text
        run = await wait_for_run(client, session_id, submitted.json()["run_id"])
        assert run["status"] == "completed"
        assert run["runtime"]["runtime_kind"] == "scripted"
        assert run["runtime"]["tool_schema_hash"]
        assert [call["tool_name"] for call in run["tool_calls"]] == [
            "get_analysis_slice",
            "submit_grounded_answer",
        ]
        assert run["usage"]["status"] == "reported"
        assert run["usage"]["events"][0]["provider_id"] == "scripted"
        assert run["usage"]["events"][0]["tokens"] == {
            "input": 0,
            "output": 5,
            "reasoning": None,
            "cache_read": 0,
            "cache_write": None,
            "total": None,
        }
        assert run["usage"]["events"][0]["cost"] == {
            "amount": "0.00125000", "currency": "USD"
        }

        messages = await client.get(f"/v1/sessions/{session_id}/messages", headers=AUTH)
        assistant = [m for m in messages.json()["messages"] if m["role"] == "assistant"]
        assert len(assistant) == 1
        assert assistant[0]["parts"][0]["content"]["text"] == (
            "The predicted hERG blocker probability is 0.731."
        )
        assert assistant[0]["parts"][1]["type"] == "answer_ref"
        assert "safe to use" not in str(assistant)


async def test_a_runtime_without_a_validated_answer_fails_without_leaking_its_delta(db):
    async def script(turn) -> None:
        turn.say("The compound is safe despite the missing answer tool call.")

    async with api_client(db, StubPredictor()) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        await _analyse_aspirin(client, session_id)
        submitted = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={
                "intent_hint": "ask_report",
                "content": [{"type": "text", "text": "Explain the report."}],
            },
            headers=AUTH,
        )
        run = await wait_for_run(client, session_id, submitted.json()["run_id"])
        assert run["status"] == "failed"
        assert run["failure_code"] == "runtime_protocol_error"
        assert run["usage"] == {"status": "unknown", "events": []}

        messages = await client.get(f"/v1/sessions/{session_id}/messages", headers=AUTH)
        assert "safe despite" not in str(messages.json()["messages"])


async def test_a_claims_observation_is_readable_and_tool_calls_carry_timestamps(db):
    analysis_id = ""
    observation_id = ""

    async def script(turn) -> None:
        nonlocal observation_id
        slice_result = await turn.call_tool(
            "get_analysis_slice",
            {"analysis_id": analysis_id, "section": "herg", "fields": ["probability_blocker"]},
        )
        value = slice_result["model_view"]["values"]["probability_blocker"]
        observation_id = value["observation_id"]
        await turn.call_tool(
            "submit_grounded_answer",
            {
                "schema_version": "grounded-answer-v1",
                "answer_markdown": "The predicted hERG blocker probability is 0.731.",
                "claims": [
                    {
                        "claim_id": "clm_" + "2" * 32,
                        "kind": "numeric",
                        "text": "The predicted hERG blocker probability is 0.731.",
                        "observation_id": observation_id,
                        "field_path": value["field_path"],
                        "source_value": value["value"],
                        "rendered_value": "0.731",
                        "transform": "round:3",
                    }
                ],
                "limitations": [{"code": "uncalibrated_probability", "text": ""}],
                "recommended_next_steps": [],
            },
        )

    async with api_client(db, StubPredictor()) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        analysis_id = await _analyse_aspirin(client, session_id)

        submitted = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={
                "intent_hint": "ask_report",
                "content": [{"type": "text", "text": "What is the hERG result?"}],
            },
            headers=AUTH,
        )
        run = await wait_for_run(client, session_id, submitted.json()["run_id"])
        assert run["status"] == "completed"

        for call in run["tool_calls"]:
            assert call["started_at"] is not None
            assert call["ended_at"] is not None
            assert call["started_at"] <= call["ended_at"]

        observation = await client.get(
            f"/v1/sessions/{session_id}/observations/{observation_id}", headers=AUTH
        )
        assert observation.status_code == 200
        body = observation.json()
        assert body["observation_id"] == observation_id
        assert body["kind"] == "prediction"
        assert "canonical_payload" not in body
        assert body["model_projection"]["observation_id"] == observation_id

        # An observation belongs to the session it was created in; a
        # different owner's session cannot read it even by a correct id.
        other_session = await client.post("/v1/sessions", json={}, headers=OTHER_AUTH)
        other_session_id = other_session.json()["session_id"]
        leaked = await client.get(
            f"/v1/sessions/{other_session_id}/observations/{observation_id}", headers=OTHER_AUTH
        )
        assert leaked.status_code == 404


ETHANOL = "CCO"


async def test_evidence_research_answers_deterministically_when_unavailable(db):
    """audit_5_9.md A06, and now the general case: a deployment with no
    research provider configured (``TOXAGENT_RESEARCH_PROVIDER=""``) must not
    let this intent reach a runtime turn a model has no way to fulfil — even
    with a scripted runtime installed that *would* otherwise happily accept
    the turn. Phase 5 exists now (see the test below), so this is exercising
    the "not configured" deployment fact, not "not built yet"."""
    runtime_was_called = False

    async def script(turn) -> None:
        nonlocal runtime_was_called
        runtime_was_called = True

    no_research = settings(research=ResearchSettings(provider=""))
    async with api_client(db, StubPredictor(), config=no_research) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        await _analyse(client, session_id, ASPIRIN)

        submitted = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={
                "intent_hint": "research_evidence",
                "content": [{"type": "text", "text": "find literature about this"}],
            },
            headers=AUTH,
        )
        assert submitted.status_code == 202, submitted.text
        body = submitted.json()
        assert body["run_status"] == "completed"

        messages = await client.get(f"/v1/sessions/{session_id}/messages", headers=AUTH)
        assistant = [m for m in messages.json()["messages"] if m["role"] == "assistant"]
        assert assistant[-1]["parts"][0]["content"]["code"] == "capability_unavailable"

    assert runtime_was_called is False


async def test_structure_recognition_answers_deterministically_when_unavailable(db):
    """This client is built without an `ocr_client` (contrast the happy-path
    test below, which passes one) — the same "unconfigured" state a real
    deployment is in when `TOXAGENT_OCR_URL` is unset (ADR 0006). An uploaded
    structure image must then get the same conversational treatment as
    evidence_research when unconfigured: a completed run with a
    `capability_unavailable` answer, never a queued run a runtime has no tool
    to fulfil."""
    runtime_was_called = False

    async def script(turn) -> None:
        nonlocal runtime_was_called
        runtime_was_called = True

    async with api_client(db, StubPredictor()) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)

        submitted = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={
                "image": {
                    "mime_type": "image/png",
                    "data_base64": base64.b64encode(PNG_BYTES).decode(),
                }
            },
            headers=AUTH,
        )
        assert submitted.status_code == 202, submitted.text
        body = submitted.json()
        assert body["run_status"] == "completed"

        messages = await client.get(f"/v1/sessions/{session_id}/messages", headers=AUTH)
        assistant = [m for m in messages.json()["messages"] if m["role"] == "assistant"]
        assert assistant[-1]["parts"][0]["content"]["code"] == "capability_unavailable"

        user_messages = [m for m in messages.json()["messages"] if m["role"] == "user"]
        image_part = next(p for p in user_messages[-1]["parts"] if p["type"] == "image_ref")
        assert image_part["content"]["mime_type"] == "image/png"
        assert image_part["content"]["size_bytes"] > 0

    assert runtime_was_called is False


async def test_structure_recognition_analyses_the_smiles_an_ocr_service_returns(db):
    """The happy path once an OCR service *is* configured (unlike the
    always-unavailable test above): recognition hands off to the exact same
    CreateAnalysis pipeline a typed SMILES goes through — real predictor call,
    real Analysis snapshot, same as if the user had pasted the SMILES."""
    ocr = stub_success(ASPIRIN)
    objects = InMemoryObjectStore()
    uploaded = PNG_BYTES + b"-happy-path"
    async with api_client(
        db, StubPredictor(), ocr_client=ocr, object_store=objects
    ) as client:
        session_id = await _new_session(client)

        submitted = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={
                "image": {
                    "mime_type": "image/png",
                    "data_base64": base64.b64encode(uploaded).decode(),
                }
            },
            headers=AUTH,
        )
        assert submitted.status_code == 202, submitted.text
        assert submitted.json()["selected_intent"] == "structure_recognition"
        await wait_for_run(client, session_id, submitted.json()["run_id"])

        state = await client.get(f"/v1/sessions/{session_id}", headers=AUTH)
        active_analysis = state.json()["active_analysis"]
        assert active_analysis is not None
        assert active_analysis["canonical_smiles"] == ASPIRIN
        assert "herg" in active_analysis["sections"]

        messages = await client.get(f"/v1/sessions/{session_id}/messages", headers=AUTH)
        recognition = [m for m in messages.json()["messages"] if m["role"] == "assistant"][-1]
        recognition_content = recognition["parts"][0]["content"]
        assert recognition_content == {
            "code": "structure_recognized",
            "smiles": ASPIRIN,
            "canonical_smiles": ASPIRIN,
            "confidence": 0.91,
        }
        user = [m for m in messages.json()["messages"] if m["role"] == "user"][-1]
        image_part = next(p for p in user["parts"] if p["type"] == "image_ref")
        attachment_id = image_part["content"]["attachment_id"]

        async with db.unit_of_work() as uow:
            attachment = await uow.attachments.get(attachment_id, owner_id="user-1")
        assert attachment is not None
        assert attachment.session_id == session_id
        assert attachment.sha256
        assert await objects.get(ObjectRef(attachment.object_uri)) == uploaded

    assert len(ocr.calls) == 1


class _UnavailableObjectStore(InMemoryObjectStore):
    async def put(self, key: str, data: bytes, *, content_type: str) -> ObjectRef:
        raise OSError("object storage is unavailable")


class _MissingBlobObjectStore(InMemoryObjectStore):
    async def get(self, ref: ObjectRef) -> bytes:
        raise ObjectNotFound(ref.key)


async def test_image_storage_failure_does_not_create_a_queued_ocr_run(db):
    """W4-07: durable bytes are a precondition to accepting an OCR run."""
    async with api_client(
        db,
        StubPredictor(),
        ocr_client=stub_success(ASPIRIN),
        object_store=_UnavailableObjectStore(),
    ) as client:
        session_id = await _new_session(client)
        submitted = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={
                "image": {
                    "mime_type": "image/png",
                    "data_base64": base64.b64encode(PNG_BYTES + b"-store-down").decode(),
                }
            },
            headers=AUTH,
        )
        assert submitted.status_code == 503, submitted.text
        assert submitted.json()["error"]["code"] == "attachment_unavailable"

        messages = await client.get(f"/v1/sessions/{session_id}/messages", headers=AUTH)
        assert messages.json()["messages"] == []
        state = await client.get(f"/v1/sessions/{session_id}", headers=AUTH)
        assert state.json()["recent_runs"] == []


async def test_missing_attachment_blob_completes_ocr_run_without_leaking_store_error(db):
    """TTL/object-store loss is an honest user-facing completion, never 500."""
    ocr = stub_success(ASPIRIN)
    async with api_client(
        db, StubPredictor(), ocr_client=ocr, object_store=_MissingBlobObjectStore()
    ) as client:
        session_id = await _new_session(client)
        submitted = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={
                "image": {
                    "mime_type": "image/png",
                    "data_base64": base64.b64encode(PNG_BYTES + b"-expired").decode(),
                }
            },
            headers=AUTH,
        )
        assert submitted.status_code == 202, submitted.text
        await wait_for_run(client, session_id, submitted.json()["run_id"])

        messages = await client.get(f"/v1/sessions/{session_id}/messages", headers=AUTH)
        assistant = [m for m in messages.json()["messages"] if m["role"] == "assistant"]
        content = assistant[-1]["parts"][0]["content"]
        assert content["code"] == "structure_recognition_failed"
        assert "upload it again" in content["message"]

    assert ocr.calls == []


async def test_structure_recognition_answers_gracefully_when_no_structure_is_found(db):
    async with api_client(db, StubPredictor(), ocr_client=stub_no_structure_detected()) as client:
        session_id = await _new_session(client)

        submitted = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={"image": {"mime_type": "image/png", "data_base64": base64.b64encode(PNG_BYTES + b"-noise").decode()}},
            headers=AUTH,
        )
        assert submitted.status_code == 202, submitted.text
        await wait_for_run(client, session_id, submitted.json()["run_id"])

        messages = await client.get(f"/v1/sessions/{session_id}/messages", headers=AUTH)
        assistant = [m for m in messages.json()["messages"] if m["role"] == "assistant"]
        content = assistant[-1]["parts"][0]["content"]
        assert content["code"] == "structure_recognition_failed"

        state = await client.get(f"/v1/sessions/{session_id}", headers=AUTH)
        assert state.json()["active_analysis"] is None


async def test_structure_recognition_answers_gracefully_when_the_ocr_service_is_unreachable(db):
    async with api_client(db, StubPredictor(), ocr_client=stub_unavailable()) as client:
        session_id = await _new_session(client)

        submitted = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={"image": {"mime_type": "image/png", "data_base64": base64.b64encode(PNG_BYTES + b"-unavailable").decode()}},
            headers=AUTH,
        )
        await wait_for_run(client, session_id, submitted.json()["run_id"])

        messages = await client.get(f"/v1/sessions/{session_id}/messages", headers=AUTH)
        assistant = [m for m in messages.json()["messages"] if m["role"] == "assistant"]
        assert assistant[-1]["parts"][0]["content"]["code"] == "structure_recognition_failed"


async def test_an_oversized_image_is_rejected_before_a_run_is_created(db):
    async with api_client(db, StubPredictor()) as client:
        session_id = await _new_session(client)
        oversized = base64.b64encode(PNG_BYTES + b"x" * (5_000_000 + 1 - len(PNG_BYTES))).decode()

        submitted = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={"image": {"mime_type": "image/png", "data_base64": oversized}},
            headers=AUTH,
        )
        assert submitted.status_code == 400, submitted.text
        assert submitted.json()["error"]["code"] == "invalid_request"


async def test_evidence_research_searches_reads_and_cites_a_configured_provider(db):
    """The Phase 5 happy path: search_toxicology_evidence and
    get_evidence_record are real tools once a provider is configured, and a
    claim citing what they returned is accepted with evidence_scope_limited
    declared (plan section 9.4)."""
    analysis_id = ""
    provider = StubResearchProvider(hits=[ACCEPTED_HIT])
    evidence_id = ""

    async def script(turn) -> None:
        nonlocal evidence_id
        search_result = await turn.call_tool(
            "search_toxicology_evidence",
            {"analysis_id": analysis_id, "query": "hERG blockade screening", "limit": 5},
        )
        assert search_result["status"] == "completed"
        results = search_result["model_view"]["results"]
        assert len(results) == 1
        evidence_id = results[0]["evidence_id"]

        detail = await turn.call_tool("get_evidence_record", {"evidence_id": evidence_id})
        assert detail["status"] == "completed"
        assert detail["model_view"]["untrusted_external_content"] is True
        assert detail["model_view"]["status"] == "accepted"

        submitted = await turn.call_tool(
            "submit_grounded_answer",
            {
                "schema_version": "grounded-answer-v1",
                "answer_markdown": (
                    "One retrieved study screened a related series for hERG channel blockade."
                ),
                "claims": [
                    {
                        "claim_id": "clm_" + "3" * 32,
                        "kind": "scientific",
                        "text": (
                            "A retrieved study screened a related compound series for hERG "
                            "channel blockade."
                        ),
                        "citation_ids": [evidence_id],
                    }
                ],
                "limitations": [{"code": "evidence_scope_limited", "text": ""}],
                "recommended_next_steps": [],
            },
        )
        assert submitted["status"] == "completed"

    async with api_client(db, StubPredictor(), research_provider=provider) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        analysis_id = await _analyse_aspirin(client, session_id)

        submitted = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={
                "intent_hint": "research_evidence",
                "content": [{"type": "text", "text": "find literature about this molecule"}],
            },
            headers=AUTH,
        )
        assert submitted.status_code == 202, submitted.text
        run = await wait_for_run(client, session_id, submitted.json()["run_id"])
        assert run["status"] == "completed", run
        assert [c["tool_name"] for c in run["tool_calls"]] == [
            "search_toxicology_evidence", "get_evidence_record", "submit_grounded_answer",
        ]

        evidence = await client.get(f"/v1/sessions/{session_id}/evidence", headers=AUTH)
        assert evidence.status_code == 200
        assert [e["evidence_id"] for e in evidence.json()["evidence"]] == [evidence_id]
        assert provider.calls == [
            {"query": "hERG blockade screening", "source_types": None, "date_from": None, "limit": 5}
        ]


async def test_citing_evidence_without_reading_it_first_is_a_correctable_violation(db):
    """W3-07 (remaining-plan): search_toxicology_evidence's result already
    carries title/identifier (tools/definitions/evidence.py's
    _SEARCH_RESULT_FIELDS) — enough for a model to construct a citation
    without ever calling get_evidence_record. A claim citing straight from a
    search result, skipping the read, must be rejected and correctable, the
    same policy as any other validation failure — not silently accepted."""
    analysis_id = ""
    provider = StubResearchProvider(hits=[ACCEPTED_HIT])
    evidence_id = ""

    async def script(turn) -> None:
        nonlocal evidence_id
        search_result = await turn.call_tool(
            "search_toxicology_evidence",
            {"analysis_id": analysis_id, "query": "hERG blockade screening", "limit": 5},
        )
        evidence_id = search_result["model_view"]["results"][0]["evidence_id"]

        # No get_evidence_record call here — cite straight from the search hit.
        first = await turn.call_tool(
            "submit_grounded_answer",
            {
                "schema_version": "grounded-answer-v1",
                "answer_markdown": "One retrieved study is relevant here.",
                "claims": [
                    {
                        "claim_id": "clm_" + "5" * 32,
                        "kind": "scientific",
                        "text": "A retrieved study is relevant here.",
                        "citation_ids": [evidence_id],
                    }
                ],
                "limitations": [{"code": "evidence_scope_limited", "text": ""}],
                "recommended_next_steps": [],
            },
        )
        assert first["status"] == "error"
        assert first["error"]["code"] == "answer_validation_failed"
        violations = first["error"]["details"]["violations"]
        assert any(v["code"] == "citation_not_read" for v in violations)

        # The correction: read it, then cite it — same claim, now accepted.
        await turn.call_tool("get_evidence_record", {"evidence_id": evidence_id})
        retried = await turn.call_tool(
            "submit_grounded_answer",
            {
                "schema_version": "grounded-answer-v1",
                "answer_markdown": "One retrieved study is relevant here.",
                "claims": [
                    {
                        "claim_id": "clm_" + "5" * 32,
                        "kind": "scientific",
                        "text": "A retrieved study is relevant here.",
                        "citation_ids": [evidence_id],
                    }
                ],
                "limitations": [{"code": "evidence_scope_limited", "text": ""}],
                "recommended_next_steps": [],
            },
        )
        assert retried["status"] == "completed"
        assert retried["model_view"]["is_fallback"] is False

    async with api_client(db, StubPredictor(), research_provider=provider) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        analysis_id = await _analyse_aspirin(client, session_id)

        submitted = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={
                "intent_hint": "research_evidence",
                "content": [{"type": "text", "text": "find literature about this molecule"}],
            },
            headers=AUTH,
        )
        assert submitted.status_code == 202, submitted.text
        run = await wait_for_run(client, session_id, submitted.json()["run_id"])
        assert run["status"] == "completed", run


async def test_accepted_evidence_is_pinned_into_a_later_turns_prompt(db):
    """plan section 10.4 step 5: pinned references are analysis *and*
    evidence — a later turn should not have to re-search for something this
    session already found and accepted."""
    analysis_id = ""
    later_prompt = ""

    async def script(turn) -> None:
        nonlocal later_prompt
        if "find literature" in turn.user_message:
            search_result = await turn.call_tool(
                "search_toxicology_evidence",
                {"analysis_id": analysis_id, "query": "hERG blockade", "limit": 5},
            )
            evidence_id = search_result["model_view"]["results"][0]["evidence_id"]
            await turn.call_tool(
                "submit_grounded_answer",
                {
                    "schema_version": "grounded-answer-v1",
                    "answer_markdown": "One retrieved study is relevant.",
                    "claims": [
                        {
                            "claim_id": "clm_" + "4" * 32,
                            "kind": "scientific",
                            "text": "One retrieved study screened a related series for hERG blockade.",
                            "citation_ids": [evidence_id],
                        }
                    ],
                    "limitations": [{"code": "evidence_scope_limited", "text": ""}],
                    "recommended_next_steps": [],
                },
            )
        else:
            later_prompt = turn.system_prompt
            turn.say("noop")

    provider = StubResearchProvider(hits=[ACCEPTED_HIT])
    async with api_client(db, StubPredictor(), research_provider=provider) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        analysis_id = await _analyse_aspirin(client, session_id)

        first = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={
                "intent_hint": "research_evidence",
                "content": [{"type": "text", "text": "find literature about this molecule"}],
            },
            headers=AUTH,
        )
        await wait_for_run(client, session_id, first.json()["run_id"])

        second = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={
                "intent_hint": "ask_report",
                "content": [{"type": "text", "text": "Summarise what we know so far."}],
            },
            headers=AUTH,
        )
        await wait_for_run(client, session_id, second.json()["run_id"])

    assert ACCEPTED_HIT.title in later_prompt


async def test_an_explicit_analysis_id_overrides_a_different_stale_active_one(db):
    """audit_5_9.md A02, repro 1: analysis A exists, then B becomes active;
    a request naming ``analysis_id=A`` explicitly used to still get B pinned
    into its prompt because the gateway only ever read
    ``session.active_analysis_id``."""
    captured_prompt = ""

    async def script(turn) -> None:
        nonlocal captured_prompt
        captured_prompt = turn.system_prompt
        turn.say("noop")

    async with api_client(db, StubPredictor()) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        analysis_a = await _analyse(client, session_id, ASPIRIN)
        await _analyse(client, session_id, ETHANOL)  # B becomes active

        submitted = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={
                "intent_hint": "ask_report",
                "content": [{"type": "text", "text": "What is the hERG result?"}],
                "analysis_id": analysis_a,
            },
            headers=AUTH,
        )
        assert submitted.status_code == 202, submitted.text
        await wait_for_run(client, session_id, submitted.json()["run_id"])

    assert ASPIRIN in captured_prompt
    assert ETHANOL not in captured_prompt


async def test_a_foreign_analysis_id_is_rejected_before_a_run_is_queued(db):
    async with api_client(db, StubPredictor()) as client:
        session_id = await _new_session(client)
        response = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={
                "intent_hint": "ask_report",
                "content": [{"type": "text", "text": "What is the hERG result?"}],
                "analysis_id": "ana_" + "0" * 32,
            },
            headers=AUTH,
        )
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "analysis_not_found"


async def test_a_new_molecule_with_ask_report_always_snapshots_even_with_an_active_analysis(db):
    """audit_5_9.md A02, repro 2: a new molecule submitted alongside an
    explicit ``ask_report`` hint used to make zero predictor calls and stay
    pinned to whatever was already active, because
    ``needs_snapshot_first`` was gated on ``not has_active_analysis``."""
    captured_prompt = ""

    async def script(turn) -> None:
        nonlocal captured_prompt
        captured_prompt = turn.system_prompt
        turn.say("noop")

    async with api_client(db, StubPredictor()) as client:
        await _install_scripted_runtime(client, script)
        session_id = await _new_session(client)
        await _analyse(client, session_id, ASPIRIN)  # A is active

        submitted = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={
                "intent_hint": "ask_report",
                "content": [{"type": "text", "text": "What is the hERG result?"}],
                "molecule": {"smiles": ETHANOL},
            },
            headers=AUTH,
        )
        assert submitted.status_code == 202, submitted.text
        # The script never calls submit_grounded_answer, so the turn itself
        # ends in runtime_protocol_error — irrelevant here: the snapshot is
        # taken deterministically *before* the runtime is even dispatched, so
        # it must exist regardless of what the "model" does afterward.
        await wait_for_run(client, session_id, submitted.json()["run_id"])

        state = await client.get(f"/v1/sessions/{session_id}", headers=AUTH)
        assert state.json()["active_analysis"]["canonical_smiles"] == ETHANOL

    assert ETHANOL in captured_prompt
    assert ASPIRIN not in captured_prompt
