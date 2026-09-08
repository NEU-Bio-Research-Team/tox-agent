"""The eval runner drives the deterministic task subset end to end.

Capability tasks need an LLM runtime and are reported ``needs_runtime``; the
analysis-failure and routing tasks run against the real in-process control
plane with a frozen predictor and must clear their graders and hard gates on
every trial (plan section 16.5: a critical task is never averaged).
"""
from __future__ import annotations

import json

import httpx
import pytest

from evals.runner import (
    RemoteHTTPDriver,
    _fetch_all_evidence,
    is_deterministic,
    is_live_compatible,
    load_tasks,
    run_suite,
)

pytestmark = pytest.mark.anyio

DETERMINISTIC_IDS = {
    "fail-01-predictor-503",
    "fail-02-predictor-malformed",
    "fail-03-invalid-smiles",
    "qa-09-out-of-scope-clinical-advice",
    "qa-10-clarification-required",
    # Rewritten 2026-09-05 (live sweep) to test the real, documented SCI-06
    # behavior — explicitly requesting an unserved endpoint fails the
    # analysis outright — which is itself a deterministic-lane failure and
    # now gets scripted-mode CI coverage it never had before.
    "endpoint-04-clintox-unavailable",
}


async def test_the_expected_tasks_are_classified_deterministic():
    tasks = {t["task_id"]: t for t in load_tasks()}
    got = {tid for tid, t in tasks.items() if is_deterministic(t)}
    assert got == DETERMINISTIC_IDS


async def test_deterministic_subset_passes_pass_cubed(tmp_path):
    tasks = load_tasks()
    summary = await run_suite(
        tasks, runtime="scripted", trials=3, out_dir=tmp_path, only=DETERMINISTIC_IDS
    )
    assert summary["executed"] == len(DETERMINISTIC_IDS)
    assert summary["skipped_needs_runtime"] == 0
    assert summary["pass_rate"] == 1.0, summary["failures"]
    assert summary["critical_all_pass"] is True
    assert summary["metric"] == "pass^3"

    manifests = list(tmp_path.glob("manifest-*.json"))
    results = list(tmp_path.glob("results-*.json"))
    assert manifests and results
    manifest = json.loads(manifests[0].read_text())
    assert len(manifest["eval_suite_hash"]) == 64
    assert manifest["runtime_kind"] == "scripted"
    assert manifest["trial_count"] == 3


async def test_a_capability_task_is_reported_as_needing_a_runtime(tmp_path):
    tasks = load_tasks()
    summary = await run_suite(
        tasks, runtime="scripted", trials=1, out_dir=tmp_path,
        only={"qa-07-herg-and-limits-vi"},
    )
    assert summary["executed"] == 0
    assert summary["skipped_needs_runtime"] == 1


async def test_an_unknown_runtime_still_refuses(tmp_path):
    with pytest.raises(SystemExit):
        await run_suite(load_tasks(), runtime="bogus", trials=1, out_dir=tmp_path)


def test_is_live_compatible_excludes_tasks_pinned_to_frozen_numbers():
    tasks = {t["task_id"]: t for t in load_tasks()}
    # numeric fidelity pins an exact frozen rendered_value/source_value.
    assert is_live_compatible(tasks["numeric-01-herg-probability-round3-vi"]) is False
    # A wording/limitation check has no such pin and holds against real output.
    assert is_live_compatible(tasks["endpoint-01-herg-not-clinical"]) is True
    # A pure routing task (no answer expectation at all) is unaffected.
    assert is_live_compatible(tasks["qa-09-out-of-scope-clinical-advice"]) is True


def test_is_live_compatible_excludes_tasks_needing_infra_injection():
    """Live sweep (2026-09-05): a bare HTTP driver cannot make a healthy
    real predictor return a 503/malformed body, cannot make an already-bound
    live runtime already unavailable, and cannot restart the control plane
    or runtime it is itself talking to — each of these was attempted and
    counted as a failure for a condition the driver never actually created."""
    tasks = {t["task_id"]: t for t in load_tasks()}
    assert is_live_compatible(tasks["fail-01-predictor-503"]) is False
    assert is_live_compatible(tasks["fail-02-predictor-malformed"]) is False
    assert is_live_compatible(tasks["fail-04-lost-runtime-before-first-request"]) is False
    assert is_live_compatible(tasks["fail-05-recovery-after-tool-call"]) is False
    assert is_live_compatible(tasks["fail-06-control-plane-restart"]) is False
    assert is_live_compatible(tasks["adv-04-compaction-then-earliest-analysis"]) is False
    # fail-03 tests a client-input condition (an invalid SMILES) that holds
    # against any predictor, broken or not, so it stays live-compatible.
    assert is_live_compatible(tasks["fail-03-invalid-smiles"]) is True


async def test_remote_http_driver_talks_only_over_the_product_api(tmp_path):
    """The same request/response shapes the scripted driver uses, over a real
    ``httpx.AsyncClient`` — proven against a transport double so this runs with
    no live stack, the way the OpenCode adapter's own contract tests do."""
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(f"{request.method} {request.url.path}")
        if request.url.path == "/v1/sessions" and request.method == "POST":
            return httpx.Response(201, json={"session_id": "ses_live1"})
        if request.url.path == "/v1/sessions/ses_live1/messages" and request.method == "POST":
            return httpx.Response(202, json={"run_id": "run_live1"})
        if request.url.path == "/v1/sessions/ses_live1/runs/run_live1":
            return httpx.Response(200, json={
                "run_id": "run_live1", "status": "completed", "intent": "out_of_scope",
                "lane": "deterministic", "tool_calls": [],
            })
        if request.url.path == "/v1/sessions/ses_live1":
            return httpx.Response(200, json={"session_id": "ses_live1", "active_analysis": None})
        if request.url.path == "/v1/sessions/ses_live1/messages" and request.method == "GET":
            return httpx.Response(200, json={"messages": []})
        if request.url.path == "/v1/sessions/ses_live1/evidence":
            return httpx.Response(200, json={"evidence": []})
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    driver = RemoteHTTPDriver(
        "http://live.test", "dev-local", transport=httpx.MockTransport(handler)
    )
    task = next(t for t in load_tasks() if t["task_id"] == "qa-09-out-of-scope-clinical-advice")
    outcome = await driver.run(task, tmp_path)
    assert outcome.run["status"] == "completed"
    assert "POST /v1/sessions" in requests
    assert "POST /v1/sessions/ses_live1/messages" in requests


async def test_fetch_all_evidence_pages_past_the_endpoints_default_limit():
    """Live sweep (2026-09-05, evsyn-05): a session with 57 accepted evidence
    records (a model that searched many times) had a genuinely-accepted
    citation flagged as unresolved by citations_resolve, only because a
    single unpaginated GET .../evidence stopped at the endpoint's default
    page of 50 and never saw the other 7. This drives more records than one
    page (200, this helper's own request size) to prove the loop itself
    terminates correctly and does not just get lucky with one big page."""
    seen_params: list[dict] = []
    total = 210

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        seen_params.append(params)
        offset = int(params.get("offset", 0))
        limit = int(params.get("limit", 50))
        page = [
            {"evidence_id": f"evd_{i:032d}"}
            for i in range(offset, min(offset + limit, total))
        ]
        return httpx.Response(200, json={"evidence": page, "count": len(page)})

    async with httpx.AsyncClient(
        base_url="http://live.test", transport=httpx.MockTransport(handler)
    ) as client:
        records = await _fetch_all_evidence(client, "ses_manyresults", {})

    assert len(records) == total
    assert {r["evidence_id"] for r in records} == {f"evd_{i:032d}" for i in range(total)}
    assert len(seen_params) == 2  # 200 + 10, the second page shorter than the limit stops the loop
