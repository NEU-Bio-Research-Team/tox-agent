"""A ChatGPT sign-in's model list is a claim; the check is the evidence.

Two things the product got wrong before this existed:

* the picker offered whatever the runtime's catalogue named, and a model the
  account cannot use (``gpt-5.4-mini``) was saved into the session and failed
  every run -- ``check_model`` asks the model a real question first;
* the list was the runtime's own filtered view, frozen at the binary's build,
  so a model released later (5.6, 6.1) never appeared -- ``host_models`` also
  reads the unfiltered catalogue.

No browser, no OpenAI account: the host is a MockTransport.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from toxagent.connections.chatgpt import (
    CHECK_TIMEOUT_S,
    ModelCheck,
    catalogue_models,
    check_model,
    host_models,
    merge_models,
)
from toxagent.connections.models import ModelOption
from toxagent.domain.errors import RuntimeUnavailable
from toxagent.domain.runtime import AuthMode

pytestmark = pytest.mark.anyio

OAUTH = json.dumps({"type": "oauth", "refresh": "r", "access": "a", "expires": 1_900_000_000_000})
CONNECTION = SimpleNamespace(
    id="conn_1", owner_id="owner-a", provider_id="openai",
    auth_mode=AuthMode.CHATGPT_SUBSCRIPTION, credential_ref=None,
)

RUNTIME_VIEW = {"providers": [{"id": "openai", "models": {
    "gpt-5.3-codex-spark": {"name": "GPT-5.3 Codex Spark"},
    "gpt-5.5": {"name": "GPT-5.5"},
}}]}
CATALOGUE = {"all": [{"id": "openai", "models": {
    "gpt-5.5": {"name": "GPT-5.5", "release_date": "2026-04-23"},
    "gpt-5.6": {"name": "GPT-5.6", "release_date": "2026-07-09"},
    "gpt-6.1-sol": {"name": "GPT-6.1 Sol", "release_date": "2026-09-29"},
    "gpt-4.1": {"name": "GPT-4.1", "release_date": "2025-04-14"},
    "gpt-image-2": {"name": "gpt-image-2", "release_date": "2026-04-21"},
    "gpt-realtime-2.1": {"name": "Realtime", "release_date": "2026-07-06"},
    "o3": {"name": "o3", "release_date": "2025-04-16"},
}}]}


class Host:
    """A private host as ``check_model`` and ``host_models`` see it."""

    def __init__(self, tmp: Path, *, reply=None, reply_status=200, catalogue=CATALOGUE,
                 catalogue_status=200, runtime_view=RUNTIME_VIEW, timeout=False, session_status=200):
        self.workspace = tmp / "workspace"
        self.calls: list[tuple[str, str, object]] = []
        self.reply, self.reply_status = reply, reply_status
        self.catalogue, self.catalogue_status = catalogue, catalogue_status
        self.runtime_view, self.timeout, self.session_status = runtime_view, timeout, session_status
        self.client = httpx.AsyncClient(base_url="http://host", transport=httpx.MockTransport(self._handle))

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        self.calls.append((request.method, request.url.path, body))
        path = request.url.path
        if path == "/config/providers":
            return httpx.Response(200, json=self.runtime_view)
        if path == "/provider":
            return httpx.Response(self.catalogue_status, json=self.catalogue)
        if path == "/session" and request.method == "POST":
            return httpx.Response(self.session_status, json={"id": "ses_check"})
        if path.endswith("/message"):
            if self.timeout:
                raise httpx.ReadTimeout("slow", request=request)
            return httpx.Response(self.reply_status, json=self.reply)
        if request.method == "DELETE":
            return httpx.Response(200, json=True)
        return httpx.Response(404)


class Pool:
    def __init__(self, host: Host | None = None, *, busy: str | None = None):
        self.host, self.busy = host, busy
        self.acquired: list = []
        self.released: list = []

    async def acquire(self, credential):
        if self.busy:
            raise RuntimeUnavailable(self.busy)
        self.acquired.append(credential)
        return self.host

    async def release(self, host):
        self.released.append(host)

    async def forget_connection(self, connection_id):
        self.forgotten = getattr(self, "forgotten", []) + [connection_id]


def _answered(**info) -> dict:
    return {"info": {"role": "assistant", "id": "msg_1", **info}, "parts": [{"type": "text", "text": "OK"}]}


async def _check(tmp_path, host: Host, model_id="gpt-5.6") -> tuple[ModelCheck, Pool]:
    pool = Pool(host)
    return await check_model(pool, CONNECTION, OAUTH, model_id), pool


# -- the check ---------------------------------------------------------------------


async def test_a_model_that_answers_is_usable_and_the_host_is_asked_as_a_run_asks(tmp_path):
    host = Host(tmp_path, reply=_answered())
    result, pool = await _check(tmp_path, host)
    assert result == ModelCheck("gpt-5.6", "usable", "the model answered")
    # The host was declared with the model being checked, so a model the
    # runtime's catalogue predates is still reachable.
    assert pool.acquired[0].model_id == "gpt-5.6" and pool.acquired[0].secret == OAUTH
    sent = next(body for method, path, body in host.calls if path.endswith("/message"))
    assert sent["model"] == {"providerID": "openai", "modelID": "gpt-5.6"}
    assert sent["agent"] == "toxagent"
    # The throwaway session is removed and the lease returned.
    assert ("DELETE", "/session/ses_check", None) in host.calls
    assert pool.released == [host]


async def test_the_sentence_openai_sent_for_gpt_5_4_mini_is_an_unsupported_model(tmp_path):
    refusal = "The 'gpt-5.4-mini' model is not supported when using Codex with a ChatGPT account."
    host = Host(tmp_path, reply=_answered(error={"name": "APIError", "data": {"message": refusal, "statusCode": 400}}))
    result, pool = await _check(tmp_path, host, "gpt-5.4-mini")
    assert result.status == "unsupported" and result.message == refusal
    assert pool.released == [host]


async def test_a_rejected_sign_in_says_so_rather_than_blaming_the_model(tmp_path):
    host = Host(tmp_path, reply=_answered(error={"name": "APIError", "data": {"message": "Unauthorized", "statusCode": 401}}))
    result, _ = await _check(tmp_path, host)
    assert result.status == "signin_rejected"


async def test_an_unclassified_provider_error_says_nothing_about_the_model(tmp_path):
    host = Host(tmp_path, reply=_answered(error={"name": "APIError", "data": {"message": "upstream exploded", "statusCode": 500}}))
    result, _ = await _check(tmp_path, host)
    assert result.status == "unavailable" and result.message == "upstream exploded"


@pytest.mark.parametrize("reply", [
    {}, {"info": {}}, {"info": {"role": "user"}, "parts": [{"type": "text", "text": "OK"}]},
    {"info": {"role": "assistant"}, "parts": []},
    {"info": {"role": "assistant"}, "parts": [{"type": "text", "text": "  "}]},
    {"info": {"role": "assistant", "error": {}}, "parts": [{"type": "text", "text": "OK"}]},
])
async def test_missing_or_malformed_answer_is_not_usable(tmp_path, reply):
    result, pool = await _check(tmp_path, Host(tmp_path, reply=reply))
    assert result.status == "unavailable"
    assert len(pool.released) == 1


async def test_a_connection_held_by_a_run_is_busy_not_unsupported():
    pool = Pool(busy="this AI connection changed while one of its runs is still active; retry when that run ends")
    result = await check_model(pool, CONNECTION, OAUTH, "gpt-5.6")
    assert result.status == "busy" and "still active" in result.message
    assert pool.released == []  # nothing was acquired


async def test_a_host_that_never_answers_is_unavailable_and_still_released(tmp_path):
    host = Host(tmp_path, timeout=True)
    result, pool = await _check(tmp_path, host)
    assert result.status == "unavailable" and str(int(CHECK_TIMEOUT_S)) in result.message
    assert pool.released == [host]


@pytest.mark.parametrize("kwargs", [{"reply_status": 500, "reply": {}}, {"session_status": 500}])
async def test_a_runtime_error_is_unavailable(tmp_path, kwargs):
    result, pool = await _check(tmp_path, Host(tmp_path, **kwargs))
    assert result.status == "unavailable"
    assert len(pool.released) == 1


@pytest.mark.parametrize("model_id", ["", "  ", "bad model", "x" * 129, "-leading"])
async def test_a_malformed_model_id_is_refused_before_a_host_starts(model_id):
    pool = Pool()
    with pytest.raises(ValueError):
        await check_model(pool, CONNECTION, OAUTH, model_id)
    assert pool.acquired == []


# -- the list ----------------------------------------------------------------------


def test_the_catalogue_adds_newer_chat_models_newest_first_and_drops_non_chat():
    known = catalogue_models(CATALOGUE)
    assert [m.id for m in known] == ["gpt-6.1-sol", "gpt-5.6", "gpt-5.5", "o3", "gpt-4.1"]
    assert {m.availability for m in known} == {"catalogue"}


def test_the_runtime_offer_comes_first_and_a_model_is_listed_once():
    offered = [ModelOption("gpt-5.5", "GPT-5.5", "runtime")]
    merged = merge_models(offered, catalogue_models(CATALOGUE))
    assert [m.id for m in merged][:3] == ["gpt-5.5", "gpt-6.1-sol", "gpt-5.6"]
    assert [m.id for m in merged].count("gpt-5.5") == 1
    assert merged[0].availability == "runtime" and merged[1].availability == "catalogue"


async def test_a_sign_in_lists_what_the_runtime_offers_and_what_only_the_catalogue_knows(tmp_path):
    models = await host_models(Host(tmp_path))
    ids = [m.id for m in models]
    assert ids[:2] == ["gpt-5.3-codex-spark", "gpt-5.5"]
    # 5.6 and 6.1 were the models that never appeared.
    assert "gpt-5.6" in ids and "gpt-6.1-sol" in ids
    assert {m.id: m.availability for m in models}["gpt-5.5"] == "runtime"
    assert {m.id: m.availability for m in models}["gpt-6.1-sol"] == "catalogue"


@pytest.mark.parametrize("catalogue", [{"all": []}, {}, "nonsense", {"all": [{"id": "openai", "models": []}]}])
async def test_an_empty_or_malformed_catalogue_leaves_the_runtime_list(tmp_path, catalogue):
    models = await host_models(Host(tmp_path, catalogue=catalogue))
    assert [m.id for m in models] == ["gpt-5.3-codex-spark", "gpt-5.5"]


async def test_a_catalogue_that_errors_leaves_the_runtime_list(tmp_path):
    models = await host_models(Host(tmp_path, catalogue_status=500, catalogue={"error": "x"}))
    assert [m.id for m in models] == ["gpt-5.3-codex-spark", "gpt-5.5"]
