"""Frozen fixtures (plan section 16.3, "Frozen mode").

A fixture supplies content-hashed predictor, attribution and evidence responses
so a task runs with no internet and the same inputs every trial. The runner
turns ``fixture["predictor"]`` into an httpx transport in the exact shape of the
pinned ToxPred contract; ``fixture["evidence"]`` feeds the research provider
double.

Regenerate the stored ``content_sha256`` after editing a fixture::

    python -m evals.frozen --rehash
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import httpx

from toxagent.domain.provenance import content_sha256

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
FIXTURE_VERSION = "eval-fixture-v1"

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def fixture_digest(fixture: dict[str, Any]) -> str:
    """Hash of everything but the recorded hash itself."""
    body = {k: v for k, v in fixture.items() if k != "content_sha256"}
    return content_sha256(body)


def load_fixture(name: str) -> dict[str, Any]:
    path = FIXTURES_DIR / f"{name}.json"
    fixture = json.loads(path.read_text())
    if fixture.get("fixture_version") != FIXTURE_VERSION:
        raise ValueError(f"{name}: fixture_version must be {FIXTURE_VERSION!r}")
    recorded = fixture.get("content_sha256")
    actual = fixture_digest(fixture)
    if recorded != actual:
        raise ValueError(
            f"{name}: content_sha256 {recorded!r} != {actual!r}; run "
            "`python -m evals.frozen --rehash` after an intentional edit"
        )
    return fixture


class FrozenPredictor:
    """A data-driven ToxPred double. Serves exactly the fixture's payloads;
    unknown SMILES get a typed ``invalid_smiles``, an unserved endpoint is
    dropped from the response the way the real predictor does."""

    def __init__(self, spec: dict[str, Any]) -> None:
        self._served = tuple(spec.get("served_endpoints", ("herg", "tox21")))
        self._ready = spec.get("ready", True)
        self._fail_with = spec.get("fail_with")
        self._malformed = spec.get("malformed", False)
        self._predictions = spec.get("predictions", {})
        self._attributions = spec.get("attributions", {})
        self.requests: list[dict[str, Any]] = []

    # The runner passes ``.client()`` straight to ``create_app(predictor=...)``.
    def client(self):
        from toxagent.config import PredictorSettings
        from toxagent.predictor.client import PredictorClient

        return PredictorClient(
            PredictorSettings(base_url="http://predictor.frozen"),
            transport=httpx.MockTransport(self._handle),
        )

    def _handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        self.requests.append({"path": request.url.path, "body": body})
        path = request.url.path

        if path == "/health/live":
            return httpx.Response(200, json={"status": "alive"})
        if path == "/health/ready":
            return httpx.Response(
                200 if self._ready else 503,
                json={
                    "ready": self._ready,
                    "reasons": [] if self._ready else ["frozen: predictor marked not ready"],
                    "served_endpoints": list(self._served),
                },
            )
        if path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "model_id": "frozen", "capabilities": list(self._served),
                            "loaded": True, "required": True, "detail": "", "blocked_reason": None,
                        }
                    ],
                    "served_endpoints": list(self._served),
                },
            )

        if self._fail_with is not None:
            return httpx.Response(
                self._fail_with, json={"error": "model_not_ready", "message": "frozen failure"}
            )
        if self._malformed:
            return httpx.Response(200, json={"canonical_smiles": "CCO"})

        if path == "/v1/predictions":
            return self._predict(body.get("smiles", ""), body.get("endpoints"))
        if path == "/v1/predictions:batch":
            results, errors = [], []
            for index, smiles in enumerate(body.get("smiles", [])):
                if smiles in self._predictions:
                    payload = self._project(self._predictions[smiles], body.get("endpoints"))
                    results.append(payload)
                else:
                    errors.append(
                        {"index": index, "input_smiles": smiles,
                         "error": "invalid_smiles", "detail": "frozen: not in fixture"}
                    )
            return httpx.Response(
                200, json={"results": results, "errors": errors, "count": len(body.get("smiles", []))}
            )
        if path == "/v1/attributions":
            key = f"{body.get('smiles')}|{body.get('endpoint')}|{body.get('task') or ''}"
            if body.get("endpoint") not in self._served:
                return httpx.Response(
                    503, json={"error": "model_not_ready", "message": f"{body.get('endpoint')} not served"}
                )
            record = self._attributions.get(key)
            if record is None:
                return httpx.Response(
                    404, json={"error": "not_found", "message": f"no frozen attribution for {key}"}
                )
            return httpx.Response(200, json=record)

        return httpx.Response(404, json={"error": "not_found", "message": path})

    def _predict(self, smiles: str, endpoints) -> httpx.Response:
        if not smiles or smiles not in self._predictions:
            return httpx.Response(
                400,
                json={
                    "error": "invalid_smiles",
                    "message": f"frozen: {smiles!r} is not in this fixture",
                    "detail": {"smiles": smiles, "reason": "not frozen"},
                },
            )
        return httpx.Response(200, json=self._project(self._predictions[smiles], endpoints))

    def _project(self, payload: dict[str, Any], endpoints) -> dict[str, Any]:
        requested = tuple(endpoints) if endpoints else self._served
        keep = {e for e in requested if e in self._served}
        out = json.loads(json.dumps(payload))  # deep copy
        out["predictions"] = {k: v for k, v in out.get("predictions", {}).items() if k in keep}
        return out


def _rehash() -> int:
    changed = 0
    for path in sorted(FIXTURES_DIR.glob("*.json")):
        fixture = json.loads(path.read_text())
        digest = fixture_digest(fixture)
        if fixture.get("content_sha256") != digest:
            fixture["content_sha256"] = digest
            path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
            print(f"rehashed {path.name}")
            changed += 1
    print(f"{changed} fixture(s) updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(_rehash() if "--rehash" in sys.argv else 0)
