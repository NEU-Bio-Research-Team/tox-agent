"""One request, one id, on the response and in the logs (K12).

Correlating a failure with the request that caused it was a matter of reading
timestamps, because nothing tied a log line to a request. The id is
correlation only: it grants nothing, so a client may supply its own — which is
what makes a browser's trace and a server's log joinable — and a forged one
buys the forger nothing but a confusing log.
"""
from __future__ import annotations

import io
import json
import logging

import pytest

from toxagent import observability as obs
from tests.support.api import AUTH, api_client
from tests.support.predictor import StubPredictor

pytestmark = pytest.mark.anyio


@pytest.fixture(autouse=True)
def restore_logging():
    yield
    logging.getLogger().handlers = []
    obs.forget_secrets()


async def test_every_response_carries_an_id_and_two_requests_do_not_share_one(db):
    async with api_client(db, StubPredictor()) as client:
        first = await client.get("/v1/predict/capabilities", headers=AUTH)
        second = await client.get("/v1/predict/capabilities", headers=AUTH)
    assert first.headers["x-request-id"]
    assert first.headers["x-request-id"] != second.headers["x-request-id"]


async def test_a_client_supplied_id_is_echoed_so_the_two_traces_join(db):
    async with api_client(db, StubPredictor()) as client:
        response = await client.get(
            "/v1/predict/capabilities", headers={**AUTH, "x-request-id": "browser-trace-42"}
        )
    assert response.headers["x-request-id"] == "browser-trace-42"


async def test_a_supplied_id_cannot_inject_structure_into_a_log_line(db):
    """It ends up inside a JSON string in a log file. A newline or a quote in
    it would let a caller write log records of their own."""
    async with api_client(db, StubPredictor()) as client:
        response = await client.get(
            "/v1/predict/capabilities",
            headers={**AUTH, "x-request-id": 'a"b\\c'},
        )
    echoed = response.headers["x-request-id"]
    assert echoed == "abc"


async def test_an_id_that_is_only_punctuation_falls_back_to_a_generated_one(db):
    async with api_client(db, StubPredictor()) as client:
        response = await client.get("/v1/predict/capabilities", headers={**AUTH, "x-request-id": "!!!"})
    assert response.headers["x-request-id"].isalnum()
    assert len(response.headers["x-request-id"]) == 32


async def test_a_supplied_id_is_bounded(db):
    async with api_client(db, StubPredictor()) as client:
        response = await client.get(
            "/v1/predict/capabilities", headers={**AUTH, "x-request-id": "x" * 500}
        )
    assert len(response.headers["x-request-id"]) == 64


async def test_a_log_line_written_during_the_request_inherits_its_id(db):
    """Including one a dependency wrote.

    The id is bound in a ContextVar rather than passed down, so the
    predictor client's own HTTP log — a library nobody threaded an id
    through — comes out correlated with the product request that caused it.
    That is the property the id exists for, and it is why the binding is a
    ContextVar and not a thread-local: one thread serves many requests here.
    """
    async with api_client(db, StubPredictor()) as client:
        # Attached after startup, because the app installs its own handler in
        # the lifespan — which is the configuration under test.
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(obs.JsonFormatter())
        logging.getLogger().addHandler(handler)
        try:
            logging.getLogger("toxagent.test").info("outside any request")
            response = await client.get(
                "/v1/predict/capabilities", headers={**AUTH, "x-request-id": "trace-1"}
            )
        finally:
            logging.getLogger().removeHandler(handler)

    assert response.status_code == 200
    lines = [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]
    correlated = [line for line in lines if line.get("request_id") == "trace-1"]
    assert correlated, f"nothing was correlated with the request: {lines}"
    assert all(
        line.get("request_id") is None
        for line in lines if line["message"] == "outside any request"
    ), "a line written outside the request must not borrow its id"


async def test_the_binding_does_not_leak_into_the_next_request(db):
    """`reset` in a finally, not a bare `set`: otherwise the second request on
    a reused task would be logged under the first one's id."""
    async with api_client(db, StubPredictor()) as client:
        await client.get("/v1/predict/capabilities", headers={**AUTH, "x-request-id": "trace-1"})
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(obs.JsonFormatter())
        logging.getLogger().addHandler(handler)
        try:
            logging.getLogger("toxagent.test").info("between requests")
        finally:
            logging.getLogger().removeHandler(handler)

    line = json.loads(stream.getvalue().strip())
    assert "request_id" not in line
