"""An in-process control plane for end-to-end tests.

Runs the real app — real routes, real workflows, real database — with the
predictor swapped for a stub and a static development token for auth. Lifespan
is entered explicitly so the app assembles exactly as it does under uvicorn.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx

from toxagent.api.app import create_app
from toxagent.config import (
    OcrSettings,
    PolicySettings,
    PredictSettings,
    PredictorSettings,
    ResearchSettings,
    RuntimeSettings,
    SecuritySettings,
    Settings,
)
from toxagent.persistence.sql.database import Database

USER_TOKEN = "dev-user-token"
EXPERT_TOKEN = "dev-expert-token"
OTHER_TOKEN = "dev-other-token"
AUTH = {"authorization": f"Bearer {USER_TOKEN}"}
OTHER_AUTH = {"authorization": f"Bearer {OTHER_TOKEN}"}
EXPERT_AUTH = {"authorization": f"Bearer {EXPERT_TOKEN}"}


def settings(**overrides) -> Settings:
    policy = overrides.pop("policy", None) or PolicySettings()
    return Settings(
        database_url="sqlite+aiosqlite:///:memory:",
        predictor=PredictorSettings(base_url="http://predictor.test"),
        policy=policy,
        predict=overrides.pop("predict", None) or PredictSettings(),
        runtime=overrides.pop("runtime", None) or RuntimeSettings(),
        research=overrides.pop("research", None) or ResearchSettings(),
        ocr=overrides.pop("ocr", None) or OcrSettings(),
        security=overrides.pop("security", None)
        or SecuritySettings(
            capability_secret="test-secret-not-for-production",
            static_tokens=(
                f"{USER_TOKEN}:user-1",
                f"{EXPERT_TOKEN}:user-1:expert",
                f"{OTHER_TOKEN}:user-2",
            ),
        ),
        **overrides,
    )


@asynccontextmanager
async def api_client(
    database: Database,
    predictor,
    *,
    config: Settings | None = None,
    research_provider=None,
    ocr_client=None,
    object_store=None,
) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(
        config or settings(),
        database=database,
        predictor=predictor.client(),
        research_provider=research_provider,
        ocr_client=ocr_client,
        object_store=object_store,
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://control.test"
        ) as client:
            client.app = app  # tests occasionally need the scheduler
            yield client


async def wait_for_run(client: httpx.AsyncClient, session_id: str, run_id: str, *, tries: int = 200):
    """Poll the run until it is terminal.

    Runs are asynchronous by design (202 + a run id), so a test that asserts on
    the outcome has to wait for it the same way a client would.
    """
    import asyncio

    for _ in range(tries):
        response = await client.get(f"/v1/sessions/{session_id}/runs/{run_id}", headers=AUTH)
        response.raise_for_status()
        body = response.json()
        if body["status"] in ("completed", "failed", "cancelled"):
            return body
        await asyncio.sleep(0.01)
    raise AssertionError(f"run {run_id} never reached a terminal state")
