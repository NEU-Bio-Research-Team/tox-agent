"""A small client for the ToxAgent product API, as the study uses it.

The study drives the product exactly as a researcher's browser would — create
a session, submit the structure, send each message, wait — and then reads back
everything the product recorded. It never reaches past ``/v1``.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

TERMINAL = ("completed", "failed", "cancelled")


class ToxAgentAPI:
    def __init__(self, base_url: str, token: str, *, transport: httpx.AsyncBaseTransport | None = None,
                 timeout_s: float = 120.0, poll_delay_s: float = 1.0,
                 run_timeout_s: float = 1200.0) -> None:
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout_s,
                                          transport=transport)
        self._auth = {"authorization": f"Bearer {token}"}
        self._poll_delay_s = poll_delay_s
        self._run_timeout_s = run_timeout_s

    async def __aenter__(self) -> "ToxAgentAPI":
        return self

    async def __aexit__(self, *exc) -> None:
        await self._client.aclose()

    async def get(self, path: str, **params: Any) -> dict[str, Any] | None:
        response = await self._client.get(path, headers=self._auth, params=params or None)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    async def post(self, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        response = await self._client.post(path, json=body or {}, headers=self._auth)
        if response.status_code >= 400:
            raise RuntimeError(f"POST {path} -> {response.status_code}: {response.text[:500]}")
        return response.json()

    async def effective_product(self) -> dict[str, Any]:
        return await self.get("/v1/system/effective-product") or {}

    async def new_session(self, language: str = "en") -> str:
        return (await self.post("/v1/sessions", {"preferred_language": language}))["session_id"]

    async def wait(self, session_id: str, run_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self._run_timeout_s
        while True:
            run = await self.get(f"/v1/sessions/{session_id}/runs/{run_id}")
            if run and run.get("status") in TERMINAL:
                return run
            if time.monotonic() > deadline:
                raise TimeoutError(f"run {run_id} did not finish in {self._run_timeout_s:.0f}s")
            await asyncio.sleep(self._poll_delay_s)

    async def analyse(self, session_id: str, smiles: str) -> dict[str, Any]:
        """Submit the structure; return the finished analysis run and its projection."""
        accepted = await self.post(f"/v1/sessions/{session_id}/messages",
                                   {"molecule": {"smiles": smiles}})
        run = await self.wait(session_id, accepted["run_id"])
        session = await self.get(f"/v1/sessions/{session_id}") or {}
        active = (session.get("active_analysis") or {}).get("analysis_id")
        analysis = await self.get(f"/v1/sessions/{session_id}/analyses/{active}") if active else None
        return {"run": run, "analysis": analysis}

    async def send(self, session_id: str, text: str) -> dict[str, Any]:
        accepted = await self.post(
            f"/v1/sessions/{session_id}/messages",
            {"intent_hint": "auto", "content": [{"type": "text", "text": text}]},
        )
        return await self.wait(session_id, accepted["run_id"])

    async def evidence(self, session_id: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        offset = 0
        while True:
            page = (await self.get(f"/v1/sessions/{session_id}/evidence", status="all",
                                   limit=200, offset=offset) or {}).get("evidence", [])
            records.extend(page)
            if len(page) < 200:
                return records
            offset += 200

    async def answer_for_run(self, session_id: str, run_id: str) -> dict[str, Any] | None:
        messages = (await self.get(f"/v1/sessions/{session_id}/messages") or {}).get("messages", [])
        for message in messages:
            if message.get("run_id") not in (None, run_id):
                continue
            for part in message.get("parts", []):
                if part.get("type") == "answer_ref":
                    answer = await self.get(
                        f"/v1/sessions/{session_id}/answers/{part['content']['answer_id']}"
                    )
                    if answer and answer.get("run_id") == run_id:
                        return answer
        return None


def snapshot_from_analysis(analysis: dict[str, Any] | None) -> dict[str, Any] | None:
    """The predictor facts a platform arm is given: values, labels, thresholds,
    model ids and applicability — no ids, hashes or policy internals."""
    if not analysis:
        return None
    return {
        "canonical_smiles": analysis.get("canonical_smiles"),
        "requested_endpoints": analysis.get("requested_endpoints"),
        "unavailable_endpoints": analysis.get("unavailable_endpoints"),
        "predictions": analysis.get("sections"),
        "applicability": analysis.get("applicability"),
    }
