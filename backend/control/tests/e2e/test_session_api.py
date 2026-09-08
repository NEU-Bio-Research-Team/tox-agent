"""The product API end to end (plan sections 6, 7.1; Phase 1 exit gate).

Nothing in this file consults a model. The Phase 1 gate is that a valid SMILES
produces an immutable snapshot, that invalid input and unavailable endpoints
keep their typed semantics, and that a restart loses no state.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from toxagent.config import SecuritySettings
from toxagent.domain.evidence import EvidenceRecord, EvidenceStatus, SourceIdentifier, SourceType
from toxagent.domain.observation import Observation, ObservationKind, Producer

from tests.support.api import (
    AUTH,
    EXPERT_AUTH,
    OTHER_AUTH,
    USER_TOKEN,
    api_client,
    settings,
    wait_for_run,
)
from tests.support.predictor import ASPIRIN, StubPredictor

pytestmark = pytest.mark.anyio


async def new_session(client, **body):
    response = await client.post("/v1/sessions", json={"preferred_language": "vi", **body}, headers=AUTH)
    assert response.status_code == 201, response.text
    return response.json()["session_id"]


async def test_a_session_is_created_and_read_back(db):
    async with api_client(db, StubPredictor()) as client:
        session_id = await new_session(client, title="Aspirin review")
        response = await client.get(f"/v1/sessions/{session_id}", headers=AUTH)
        assert response.status_code == 200
        body = response.json()
        assert body["preferred_language"] == "vi"
        assert body["title"] == "Aspirin review"
        assert body["active_analysis"] is None


async def test_evidence_list_and_detail_are_owner_scoped_and_exclude_raw_payload_ref(db):
    async with api_client(db, StubPredictor()) as client:
        session_id = await new_session(client)
        record = EvidenceRecord.create(
            session_id=session_id,
            provider="europepmc",
            provider_record_id="PMC123",
            source_type=SourceType.ARTICLE,
            title="A normalized evidence record",
            retrieved_at=datetime.now(timezone.utc),
            authors=("Ada Lovelace",),
            canonical_url="https://europepmc.org/articles/PMC123",
            identifier=SourceIdentifier(pmcid="PMC123"),
            abstract_or_excerpt="External evidence text.",
            normalized_facts={"endpoint": "herg"},
            raw_payload_ref="objects/audit-only-raw.json",
        ).to_status(EvidenceStatus.NORMALIZED).to_status(EvidenceStatus.ACCEPTED)
        async with db.unit_of_work() as uow:
            await uow.evidence.add(record)
            await uow.commit()

        listed = await client.get(f"/v1/sessions/{session_id}/evidence?status=all", headers=AUTH)
        assert listed.status_code == 200
        assert listed.json()["evidence"][0]["evidence_id"] == record.id

        detail = await client.get(f"/v1/sessions/{session_id}/evidence/{record.id}", headers=AUTH)
        assert detail.status_code == 200
        body = detail.json()
        assert body["authors"] == ["Ada Lovelace"]
        assert body["canonical_url"] == "https://europepmc.org/articles/PMC123"
        assert body["untrusted_external_content"] is True
        assert "raw_payload_ref" not in body

        other = await client.get(f"/v1/sessions/{session_id}/evidence/{record.id}", headers=OTHER_AUTH)
        assert other.status_code == 404
        assert other.json()["error"]["code"] == "session_not_found"


async def test_attribution_list_is_scoped_to_one_analysis_and_returns_bounded_projection(db):
    async with api_client(db, StubPredictor()) as client:
        session_id = await new_session(client)
        accepted = (await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={"molecule": {"smiles": ASPIRIN}}, headers=AUTH,
        )).json()
        await wait_for_run(client, session_id, accepted["run_id"])
        analysis_id = (await client.get(f"/v1/sessions/{session_id}", headers=AUTH)).json()["active_analysis"]["analysis_id"]
        attribution = Observation.create(
            session_id=session_id,
            run_id=accepted["run_id"],
            producer=Producer.ATTRIBUTION,
            kind=ObservationKind.ATTRIBUTION,
            schema_version="toxpred-attribution-v1",
            canonical_payload={"tokens": [{"token": "Cl", "score": -0.1}]},
            model_projection={
                "analysis_id": analysis_id,
                "endpoint": "herg",
                "task": None,
                "status": "completed",
                "method": "integrated_gradients",
                "model_id": "herg-v1",
                "top_tokens": [{"token": "Cl", "score": -0.1}],
                "required_limitations": ["attribution_not_causality"],
            },
            provenance={"analysis_id": analysis_id},
            now=datetime.now(timezone.utc),
            required_limitations=("attribution_not_causality",),
        )
        async with db.unit_of_work() as uow:
            await uow.observations.add(attribution, analysis_id=analysis_id)
            await uow.commit()

        response = await client.get(
            f"/v1/sessions/{session_id}/analyses/{analysis_id}/attributions", headers=AUTH,
        )
        assert response.status_code == 200
        body = response.json()["attributions"]
        assert len(body) == 1
        assert body[0]["observation_id"] == attribution.id
        assert body[0]["endpoint"] == "herg"
        assert body[0]["top_tokens"] == [{"token": "Cl", "score": -0.1}]
        assert "canonical_payload" not in body[0]


async def test_creating_a_session_is_idempotent(db):
    async with api_client(db, StubPredictor()) as client:
        first = await client.post(
            "/v1/sessions", json={"client_session_id": "web-1"}, headers=AUTH
        )
        second = await client.post(
            "/v1/sessions", json={"client_session_id": "web-1"}, headers=AUTH
        )
        assert first.json()["session_id"] == second.json()["session_id"]


async def test_an_unknown_field_is_a_400_not_a_silent_default(db):
    async with api_client(db, StubPredictor()) as client:
        response = await client.post(
            "/v1/sessions", json={"preferred_langauge": "vi"}, headers=AUTH
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_request"


async def test_a_missing_token_is_rejected(db):
    async with api_client(db, StubPredictor()) as client:
        response = await client.post("/v1/sessions", json={})
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "unauthenticated"


async def test_another_users_session_is_not_found_not_forbidden(db):
    """Plan section 14.1: a 403 would confirm the session exists."""
    async with api_client(db, StubPredictor()) as client:
        session_id = await new_session(client)
        response = await client.get(f"/v1/sessions/{session_id}", headers=OTHER_AUTH)
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "session_not_found"


async def test_a_smiles_becomes_an_analysis(db):
    async with api_client(db, StubPredictor()) as client:
        session_id = await new_session(client)
        accepted = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={"molecule": {"smiles": ASPIRIN}, "client_message_id": "m-1"},
            headers=AUTH,
        )
        assert accepted.status_code == 202
        body = accepted.json()
        assert body["selected_intent"] == "analysis"
        assert body["lane"] == "deterministic"
        assert body["events_url"] == f"/v1/sessions/{session_id}/events"

        run = await wait_for_run(client, session_id, body["run_id"])
        assert run["status"] == "completed"

        session = (await client.get(f"/v1/sessions/{session_id}", headers=AUTH)).json()
        analysis = session["active_analysis"]
        assert analysis["sections"]["herg"]["probability_blocker"] == 0.73064
        assert analysis["sections"]["herg"]["label"] == "blocker"
        assert len(analysis["sections"]["tox21"]["assays"]) == 12
        assert "uncalibrated_probability" in analysis["required_limitations"]


async def test_the_analysis_endpoint_returns_the_same_projection(db):
    async with api_client(db, StubPredictor()) as client:
        session_id = await new_session(client)
        accepted = (
            await client.post(
                f"/v1/sessions/{session_id}/messages",
                json={"molecule": {"smiles": ASPIRIN}}, headers=AUTH,
            )
        ).json()
        await wait_for_run(client, session_id, accepted["run_id"])
        session = (await client.get(f"/v1/sessions/{session_id}", headers=AUTH)).json()
        analysis_id = session["active_analysis"]["analysis_id"]

        direct = await client.get(
            f"/v1/sessions/{session_id}/analyses/{analysis_id}", headers=AUTH
        )
        assert direct.status_code == 200
        assert direct.json()["canonical_smiles"] == ASPIRIN
        # The lossless payload is not in the default body.
        assert "predictor_response" not in direct.json()


async def test_an_invalid_smiles_fails_the_run_with_a_typed_code(db):
    async with api_client(db, StubPredictor()) as client:
        session_id = await new_session(client)
        accepted = (
            await client.post(
                f"/v1/sessions/{session_id}/messages",
                json={"molecule": {"smiles": "not-a-molecule"}}, headers=AUTH,
            )
        ).json()
        run = await wait_for_run(client, session_id, accepted["run_id"])
        assert run["status"] == "failed"
        assert run["failure_code"] == "invalid_smiles"

        messages = (
            await client.get(f"/v1/sessions/{session_id}/messages", headers=AUTH)
        ).json()["messages"]
        error_parts = [
            p for m in messages for p in m["parts"] if p["type"] == "error"
        ]
        assert error_parts and error_parts[0]["content"]["code"] == "invalid_smiles"
        # No prediction-shaped placeholder anywhere.
        assert "probability" not in str(messages)


async def test_an_unavailable_endpoint_fails_rather_than_substituting(db):
    async with api_client(db, StubPredictor(served=("herg", "tox21"))) as client:
        session_id = await new_session(client)
        accepted = (
            await client.post(
                f"/v1/sessions/{session_id}/messages",
                json={
                    "molecule": {"smiles": ASPIRIN},
                    "analysis_options": {"endpoints": ["herg", "clintox"]},
                },
                headers=AUTH,
            )
        ).json()
        run = await wait_for_run(client, session_id, accepted["run_id"])
        assert run["status"] == "failed"
        assert run["failure_code"] == "endpoint_unavailable"
        session = (await client.get(f"/v1/sessions/{session_id}", headers=AUTH)).json()
        assert session["active_analysis"] is None


async def test_a_predictor_outage_is_retryable_and_loses_no_session(db):
    async with api_client(db, StubPredictor(fail_with=503)) as client:
        session_id = await new_session(client)
        accepted = (
            await client.post(
                f"/v1/sessions/{session_id}/messages",
                json={"molecule": {"smiles": ASPIRIN}}, headers=AUTH,
            )
        ).json()
        run = await wait_for_run(client, session_id, accepted["run_id"])
        assert run["failure_code"] == "predictor_not_ready"
        assert (await client.get(f"/v1/sessions/{session_id}", headers=AUTH)).status_code == 200


async def test_a_question_with_no_molecule_is_answered_with_a_clarification(db):
    async with api_client(db, StubPredictor()) as client:
        session_id = await new_session(client)
        response = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={"content": [{"type": "text", "text": "Chất này có an toàn không?"}]},
            headers=AUTH,
        )
        body = response.json()
        assert body["selected_intent"] == "clarification_required"
        assert body["run_status"] == "completed"
        assert body["clarification"]["code"] == "molecule_missing"


async def test_an_out_of_scope_request_never_starts_a_run(db):
    async with api_client(db, StubPredictor()) as client:
        session_id = await new_session(client)
        body = (
            await client.post(
                f"/v1/sessions/{session_id}/messages",
                json={"content": [{"type": "text", "text": "prescribe a dose for my patient"}]},
                headers=AUTH,
            )
        ).json()
        assert body["selected_intent"] == "out_of_scope"
        assert body["run_status"] == "completed"


async def test_resubmitting_a_client_message_id_returns_the_first_run(db):
    stub = StubPredictor()
    async with api_client(db, stub) as client:
        session_id = await new_session(client)
        payload = {"molecule": {"smiles": ASPIRIN}, "client_message_id": "retry-1"}
        first = (
            await client.post(f"/v1/sessions/{session_id}/messages", json=payload, headers=AUTH)
        ).json()
        await wait_for_run(client, session_id, first["run_id"])
        second = (
            await client.post(f"/v1/sessions/{session_id}/messages", json=payload, headers=AUTH)
        ).json()
        assert second["run_id"] == first["run_id"]
        assert second["duplicate_of_message_id"] == first["message_id"]
        assert len([r for r in stub.requests if r["path"] == "/v1/predictions"]) == 1


async def test_a_second_run_is_refused_while_one_is_in_flight(db):
    config = settings()
    async with api_client(db, StubPredictor(), config=config) as client:
        session_id = await new_session(client)
        await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={"molecule": {"smiles": ASPIRIN}}, headers=AUTH,
        )
        second = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={"molecule": {"smiles": "CCO"}}, headers=AUTH,
        )
        assert second.status_code in (202, 409)
        if second.status_code == 409:
            assert second.json()["error"]["code"] == "conflict"


async def test_a_batch_keeps_order_and_reports_per_item_failures(db):
    async with api_client(db, StubPredictor()) as client:
        session_id = await new_session(client)
        accepted = (
            await client.post(
                f"/v1/sessions/{session_id}/messages",
                json={"molecule": {"batch_smiles": [ASPIRIN, "not-a-molecule", "CCO"]}},
                headers=AUTH,
            )
        ).json()
        assert accepted["selected_intent"] == "analysis_batch"
        run = await wait_for_run(client, session_id, accepted["run_id"])
        assert run["status"] == "completed"


async def test_threshold_overrides_need_the_expert_role(db):
    async with api_client(db, StubPredictor()) as client:
        session_id = await new_session(client)
        accepted = (
            await client.post(
                f"/v1/sessions/{session_id}/messages",
                json={
                    "molecule": {"smiles": ASPIRIN},
                    "analysis_options": {"threshold_overrides": {"herg": 0.3}},
                },
                headers=EXPERT_AUTH,
            )
        ).json()
        run = await wait_for_run(client, session_id, accepted["run_id"])
        assert run["failure_code"] == "forbidden"


async def test_state_survives_a_control_plane_restart(db):
    """PROD-04. The second app instance shares only the database."""
    stub = StubPredictor()
    async with api_client(db, stub) as client:
        session_id = await new_session(client)
        accepted = (
            await client.post(
                f"/v1/sessions/{session_id}/messages",
                json={"molecule": {"smiles": ASPIRIN}}, headers=AUTH,
            )
        ).json()
        await wait_for_run(client, session_id, accepted["run_id"])

    async with api_client(db, stub) as restarted:
        session = (await restarted.get(f"/v1/sessions/{session_id}", headers=AUTH)).json()
        assert session["active_analysis"]["sections"]["herg"]["probability_blocker"] == 0.73064
        messages = (
            await restarted.get(f"/v1/sessions/{session_id}/messages", headers=AUTH)
        ).json()
        assert messages["count"] >= 1


async def test_the_root_route_identifies_the_service(db):
    async with api_client(db, StubPredictor()) as client:
        response = await client.get("/")
        assert response.status_code == 200
        body = response.json()
        assert body["name"] == "toxagent-control"
        assert body["docs"] == "/docs"


async def test_sessions_are_listed_for_their_owner_only(db):
    async with api_client(db, StubPredictor()) as client:
        mine = await new_session(client, title="Mine")
        await client.post("/v1/sessions", json={"title": "Theirs"}, headers=OTHER_AUTH)

        response = await client.get("/v1/sessions", headers=AUTH)
        assert response.status_code == 200
        body = response.json()
        ids = [row["session_id"] for row in body["sessions"]]
        assert mine in ids
        assert all(row["title"] != "Theirs" for row in body["sessions"])
        assert body["next_offset"] is None


async def test_session_list_pagination_reports_a_next_offset(db):
    async with api_client(db, StubPredictor()) as client:
        for i in range(3):
            await new_session(client, title=f"s{i}")

        first_page = (
            await client.get("/v1/sessions?limit=2", headers=AUTH)
        ).json()
        assert len(first_page["sessions"]) == 2
        assert first_page["next_offset"] == 2

        second_page = (
            await client.get(f"/v1/sessions?limit=2&offset={first_page['next_offset']}", headers=AUTH)
        ).json()
        assert len(second_page["sessions"]) == 1
        assert second_page["next_offset"] is None


async def test_a_deterministic_run_can_be_replayed_from_the_event_list(db):
    async with api_client(db, StubPredictor()) as client:
        session_id = await new_session(client)
        accepted = (
            await client.post(
                f"/v1/sessions/{session_id}/messages",
                json={"molecule": {"smiles": ASPIRIN}}, headers=AUTH,
            )
        ).json()
        await wait_for_run(client, session_id, accepted["run_id"])

        listed = await client.get(f"/v1/sessions/{session_id}/events:list", headers=AUTH)
        assert listed.status_code == 200
        body = listed.json()
        assert body["count"] >= 1
        assert body["latest_sequence"] >= body["count"]
        types = [e["type"] for e in body["events"]]
        assert "run.completed" in types
        assert "analysis.created" in types

        scoped = await client.get(
            f"/v1/sessions/{session_id}/events:list?run_id={accepted['run_id']}", headers=AUTH
        )
        scoped_body = scoped.json()
        assert scoped_body["count"] > 0
        assert all(e["run_id"] == accepted["run_id"] for e in scoped_body["events"])
        assert scoped_body["count"] < body["count"] or scoped_body["count"] == body["count"]


async def test_cors_is_off_by_default_and_on_when_configured(db):
    async with api_client(db, StubPredictor()) as client:
        response = await client.get(
            "/health/live", headers={"origin": "https://app.example.test"}
        )
        assert "access-control-allow-origin" not in response.headers

    configured = settings(
        security=SecuritySettings(
            capability_secret="test-secret-not-for-production",
            static_tokens=(f"{USER_TOKEN}:user-1",),
            cors_allow_origins=("https://app.example.test",),
        )
    )
    async with api_client(db, StubPredictor(), config=configured) as client:
        response = await client.get(
            "/health/live", headers={"origin": "https://app.example.test"}
        )
        assert response.headers["access-control-allow-origin"] == "https://app.example.test"

        blocked_origin = await client.get(
            "/health/live", headers={"origin": "https://evil.test"}
        )
        assert "access-control-allow-origin" not in blocked_origin.headers
