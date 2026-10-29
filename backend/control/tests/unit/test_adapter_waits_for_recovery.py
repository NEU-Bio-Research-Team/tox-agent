"""The benchmark adapter waits for the run the server recovered, not the lost one.

Live, arm E on a05 (2026-10-29): a run failed runtime_unavailable, the server
started its one recovery run, the adapter returned the failed run and sent the
next turn -- and the product refused it with 409, a run already in flight.
"""
from __future__ import annotations

import httpx
import pytest

from evals.investigation.adapters.toxagent_api import ToxAgentAPI

pytestmark = pytest.mark.anyio

SESSION = "ses_1"


def _transport(runs: dict[str, dict], active: dict | None) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == f"/v1/sessions/{SESSION}":
            return httpx.Response(200, json={"session_id": SESSION, "active_run": active})
        run_id = path.rsplit("/", 1)[-1]
        return httpx.Response(200, json=runs[run_id])

    return httpx.MockTransport(handler)


async def test_a_run_the_server_recovered_is_waited_for_through_its_recovery():
    runs = {
        "run_lost": {"run_id": "run_lost", "status": "failed",
                     "failure_code": "runtime_unavailable"},
        "run_retry": {"run_id": "run_retry", "status": "completed",
                      "recovery_of_run_id": "run_lost"},
    }
    active = {"run_id": "run_retry", "recovery_of_run_id": "run_lost"}
    async with ToxAgentAPI("http://x", "t", transport=_transport(runs, active),
                           poll_delay_s=0) as api:
        run = await api.wait(SESSION, "run_lost")
    assert run["run_id"] == "run_retry"


async def test_any_other_terminal_run_is_returned_as_it_is():
    runs = {"run_a": {"run_id": "run_a", "status": "failed",
                      "failure_code": "phase_budget_exhausted"}}
    async with ToxAgentAPI("http://x", "t", transport=_transport(runs, None),
                           poll_delay_s=0) as api:
        run = await api.wait(SESSION, "run_a")
    assert run["run_id"] == "run_a"


async def test_a_lost_run_with_no_recovery_is_returned_after_a_bounded_look():
    runs = {"run_lost": {"run_id": "run_lost", "status": "failed",
                         "failure_code": "runtime_unavailable"}}
    async with ToxAgentAPI("http://x", "t", transport=_transport(runs, None),
                           poll_delay_s=0) as api:
        run = await api.wait(SESSION, "run_lost")
    assert run["run_id"] == "run_lost"
