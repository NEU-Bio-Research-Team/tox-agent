"""The change feed (plan sections 6.4, 13.3; PROD-05).

The property under test is not "SSE works" but "losing the stream loses
nothing": every event is a committed outbox row, resuming from a sequence
replays exactly the tail, and REST alone reconstructs the session.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from tests.support.api import AUTH, OTHER_AUTH, api_client, wait_for_run
from tests.support.predictor import ASPIRIN, StubPredictor
from toxagent.streaming.events import EventNotifier
from toxagent.streaming.sse import event_stream

pytestmark = pytest.mark.anyio


async def drain(outbox, notifier, session_id, after=0, limit=50):
    """Read what the feed would emit right now, without holding a connection."""
    events = []
    stream = event_stream(
        outbox, notifier, session_id, after_sequence=after,
        poll_seconds=0.01, max_idle_seconds=0.01,
    )
    async for frame in stream:
        if frame["event"] == "heartbeat":
            break
        events.append(frame)
        if len(events) >= limit:
            break
    return events


async def test_every_state_change_reaches_the_feed_in_order(db):
    notifier = EventNotifier()
    async with api_client(db, StubPredictor()) as client:
        session_id = (await client.post("/v1/sessions", json={}, headers=AUTH)).json()["session_id"]
        accepted = (
            await client.post(
                f"/v1/sessions/{session_id}/messages",
                json={"molecule": {"smiles": ASPIRIN}}, headers=AUTH,
            )
        ).json()
        await wait_for_run(client, session_id, accepted["run_id"])

        frames = await drain(db.outbox(), notifier, session_id)
        types = [f["event"] for f in frames]
        assert types[0] == "session.created"
        assert "message.created" in types
        assert "run.queued" in types
        assert "analysis.created" in types
        assert "run.completed" in types

        sequences = [int(f["id"]) for f in frames]
        assert sequences == sorted(sequences)
        assert sequences == list(range(1, len(sequences) + 1))


async def test_resuming_from_a_sequence_replays_only_the_tail(db):
    notifier = EventNotifier()
    async with api_client(db, StubPredictor()) as client:
        session_id = (await client.post("/v1/sessions", json={}, headers=AUTH)).json()["session_id"]
        accepted = (
            await client.post(
                f"/v1/sessions/{session_id}/messages",
                json={"molecule": {"smiles": ASPIRIN}}, headers=AUTH,
            )
        ).json()
        await wait_for_run(client, session_id, accepted["run_id"])

        everything = await drain(db.outbox(), notifier, session_id)
        tail = await drain(db.outbox(), notifier, session_id, after=2)
        assert [f["id"] for f in tail] == [f["id"] for f in everything[2:]]


async def test_an_event_envelope_carries_what_a_client_needs_to_reconcile(db):
    notifier = EventNotifier()
    async with api_client(db, StubPredictor()) as client:
        session_id = (await client.post("/v1/sessions", json={}, headers=AUTH)).json()["session_id"]
        frames = await drain(db.outbox(), notifier, session_id)
        payload = json.loads(frames[0]["data"])
        assert set(payload) >= {
            "event_id", "session_id", "sequence", "type", "entity_type", "entity_id",
            "entity_version", "run_id", "occurred_at", "payload",
        }
        assert payload["session_id"] == session_id


async def test_the_feed_of_a_foreign_session_is_not_reachable(db):
    async with api_client(db, StubPredictor()) as client:
        session_id = (await client.post("/v1/sessions", json={}, headers=AUTH)).json()["session_id"]
        response = await client.get(f"/v1/sessions/{session_id}/events", headers=OTHER_AUTH)
        assert response.status_code == 404


async def test_a_notifier_wakes_a_waiting_subscriber():
    notifier = EventNotifier()
    waiting = asyncio.create_task(notifier.wait("ses_x", timeout=2.0))
    await asyncio.sleep(0.01)
    notifier.notify(["ses_x"])
    assert await waiting is True
    assert notifier.subscriber_count == 0


async def test_a_notification_for_another_session_does_not_wake_this_one():
    notifier = EventNotifier()
    waiting = asyncio.create_task(notifier.wait("ses_a", timeout=0.05))
    await asyncio.sleep(0.01)
    notifier.notify(["ses_b"])
    assert await waiting is False


async def test_rest_alone_reconstructs_the_session_after_a_lost_stream(db):
    """The stream is an optimisation; this is the guarantee."""
    async with api_client(db, StubPredictor()) as client:
        session_id = (await client.post("/v1/sessions", json={}, headers=AUTH)).json()["session_id"]
        accepted = (
            await client.post(
                f"/v1/sessions/{session_id}/messages",
                json={"content": [{"type": "text", "text": "phân tích chất này"}],
                      "molecule": {"smiles": ASPIRIN}},
                headers=AUTH,
            )
        ).json()
        await wait_for_run(client, session_id, accepted["run_id"])

        session = (await client.get(f"/v1/sessions/{session_id}", headers=AUTH)).json()
        messages = (
            await client.get(f"/v1/sessions/{session_id}/messages", headers=AUTH)
        ).json()
        run = (
            await client.get(f"/v1/sessions/{session_id}/runs/{accepted['run_id']}", headers=AUTH)
        ).json()

        assert session["active_analysis"]["canonical_smiles"] == ASPIRIN
        assert messages["messages"][0]["parts"][0]["content"]["text"] == "phân tích chất này"
        assert run["status"] == "completed"
        assert session["latest_event_sequence"] >= 4
