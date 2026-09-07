"""A stand-in ToxPred.

Serves payloads in the exact shape of the pinned contract, so the client and
everything above it are exercised against the real schema without needing model
artifacts. Behaviours the workflows must handle — an unserved endpoint, a
missing model, a malformed body — are selectable rather than mocked ad hoc at
each call site.
"""
from __future__ import annotations

import json
from typing import Any

import httpx

from toxagent.config import PredictorSettings
from toxagent.predictor.client import PredictorClient
from toxagent.predictor.contract import TOX21_TASKS

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"

#: The real shape captured from a live ToxPred `POST /v1/predictions` (audit
#: A01/A14): `predictor_version`, not `service_version`, and `artifacts` is a
#: *list of dicts*, not a flat mapping — a fixture using the old shape is
#: exactly why the client's mapping bug went uncaught by this contract test.
#: `git_commit` is kept only because other fixtures/tests still assert on it;
#: the real payload does not currently emit one.
PROVENANCE = {
    "request_id": "ce2ffd98-ff45-4148-93de-c5e528b78970",
    "predictor_version": "0.1.0.dev0",
    "policy_version": "tox-policy-v1",
    "tox21_task_order_version": "tox21-12task-v1",
    "models": ["herg-tox21-chemberta-v1"],
    "artifacts": [
        {
            "model_id": "herg-tox21-chemberta-v1",
            "weights_sha256": "c851e81541f8975f66589879ba9bd35c3068c3fbd57417bb7939214183f62690",
            "tokenizer_sha256": "ba6a21b7958b8aebf1f3ac341a883c430ae9906cba797b4f186ac79dcd00d785",
        }
    ],
    "truncated_input": False,
    "git_commit": "562b988de9714106fd842bb503072cfe8cd2852a",
}


def herg_section(probability: float = 0.73064, threshold: float = 0.5) -> dict[str, Any]:
    return {
        "probability_blocker": probability,
        "label": "blocker" if probability >= threshold else "non_blocker",
        "threshold": threshold,
        "threshold_source": "model_default",
        "model_id": "pretrained_2head_herg_chemberta",
    }


def tox21_section(active_tasks: tuple[str, ...] = ("SR-MMP",)) -> dict[str, Any]:
    return {
        "task_order_version": "tox21-12task-v1",
        "assays": {
            task: {
                "probability_activity": 0.82 if task in active_tasks else 0.07,
                "active": task in active_tasks,
                "threshold": 0.5,
                "threshold_source": "model_default",
            }
            for task in TOX21_TASKS
        },
        "model_id": "pretrained_2head_herg_chemberta",
    }


def prediction(
    smiles: str = ASPIRIN,
    *,
    endpoints: tuple[str, ...] = ("herg", "tox21"),
    probability: float = 0.73064,
    applicability_status: str = "ok",
) -> dict[str, Any]:
    predictions: dict[str, Any] = {}
    if "herg" in endpoints:
        predictions["herg"] = herg_section(probability)
    if "tox21" in endpoints:
        predictions["tox21"] = tox21_section()
    if "clintox" in endpoints:
        predictions["clintox"] = {
            "probability_clinical_toxicity": 0.21,
            "label": "negative",
            "threshold": 0.5,
            "threshold_source": "model_default",
            "model_id": "smilesgnn_clintox",
        }
    return {
        "input_smiles": smiles,
        "canonical_smiles": smiles,
        "predictions": predictions,
        "applicability": {
            "status": applicability_status,
            "method": "element_rules_v1",
            "reasons": [] if applicability_status == "ok" else ["contains boron"],
        },
        "provenance": PROVENANCE,
    }


class StubPredictor:
    """Configurable predictor. ``served`` is what this deployment can answer."""

    def __init__(
        self,
        *,
        served: tuple[str, ...] = ("herg", "tox21"),
        ready: bool = True,
        fail_with: int | None = None,
        malformed: bool = False,
        probability: float = 0.73064,
        explain_status: str = "completed",
    ) -> None:
        self.served = served
        self.ready = ready
        self.fail_with = fail_with
        self.malformed = malformed
        self.probability = probability
        self.explain_status = explain_status
        self.requests: list[dict[str, Any]] = []

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def client(self, **settings: Any) -> PredictorClient:
        return PredictorClient(
            PredictorSettings(base_url="http://predictor.test", **settings),
            transport=self.transport(),
        )

    def _handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        self.requests.append({"path": request.url.path, "body": body})

        if request.url.path == "/health/live":
            return httpx.Response(200, json={"status": "alive"})
        if request.url.path == "/health/ready":
            return httpx.Response(
                200 if self.ready else 503,
                json={
                    "ready": self.ready,
                    "reasons": [] if self.ready else ["herg artifact missing"],
                    "served_endpoints": list(self.served),
                },
            )
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "model_id": "pretrained_2head_herg_chemberta",
                            "capabilities": ["herg", "tox21"],
                            "loaded": True, "required": True, "detail": "",
                            "blocked_reason": None,
                        }
                    ],
                    "served_endpoints": list(self.served),
                },
            )

        if self.fail_with is not None:
            return httpx.Response(
                self.fail_with,
                json={"error": "model_not_ready", "message": "artifact unavailable"},
            )
        if self.malformed:
            return httpx.Response(200, json={"canonical_smiles": "CCO"})

        if request.url.path == "/v1/predictions":
            smiles = body.get("smiles", "")
            if not smiles or " " in smiles or smiles in ("not-a-molecule", "???"):
                return httpx.Response(
                    400,
                    json={
                        "error": "invalid_smiles",
                        "message": f"could not parse {smiles!r}",
                        "detail": {"smiles": smiles, "reason": "rdkit returned None"},
                    },
                )
            requested = tuple(body.get("endpoints") or self.served)
            return httpx.Response(
                200,
                json=prediction(
                    smiles,
                    endpoints=tuple(e for e in requested if e in self.served),
                    probability=self.probability,
                ),
            )

        if request.url.path == "/v1/predictions:batch":
            results, errors = [], []
            for index, smiles in enumerate(body.get("smiles", [])):
                if smiles in ("not-a-molecule", ""):
                    errors.append(
                        {
                            "index": index, "input_smiles": smiles,
                            "error": "invalid_smiles", "detail": "rdkit returned None",
                        }
                    )
                else:
                    results.append(prediction(smiles, endpoints=self.served))
            return httpx.Response(
                200,
                json={"results": results, "errors": errors, "count": len(body.get("smiles", []))},
            )

        if request.url.path == "/v1/attributions":
            endpoint = body.get("endpoint")
            if endpoint not in self.served:
                return httpx.Response(
                    503,
                    json={"error": "model_not_ready", "message": f"{endpoint} is not served"},
                )
            return httpx.Response(
                200,
                json={
                    "status": "completed",
                    "input_smiles": body.get("smiles"),
                    "canonical_smiles": body.get("smiles"),
                    "endpoint": endpoint,
                    "task": body.get("task"),
                    "probability": self.probability,
                    "tokens": [
                        {"token": "C", "score": 0.12}, {"token": "c1ccccc1", "score": 0.44},
                    ],
                    "metadata": {
                        "method": "integrated_gradients_v1",
                        "model_id": "pretrained_2head_herg_chemberta",
                        "deterministic": True, "duration_ms": 812.0,
                        "timeout_ms": 30000, "note": None,
                    },
                },
            )

        if request.url.path == "/v1/explanations":
            endpoint = body.get("endpoint")
            if endpoint not in self.served:
                return httpx.Response(
                    503,
                    json={"error": "model_not_ready", "message": f"{endpoint} is not served"},
                )
            smiles = body.get("smiles")
            note = (
                "attribution took 210000 ms, over the 180000 ms budget"
                if self.explain_status == "partial"
                else None
            )
            return httpx.Response(
                200,
                json={
                    "status": self.explain_status,
                    "endpoint": endpoint,
                    "task": body.get("task"),
                    "input_smiles": smiles,
                    "canonical_smiles": smiles,
                    "atom_order_version": "rdkit-output-order-v1",
                    "probability": self.probability,
                    "atoms": [
                        {"atom_index": 0, "symbol": "C", "importance": 0.5,
                         "relative_importance": 0.4},
                        {"atom_index": 1, "symbol": "C", "importance": 0.3,
                         "relative_importance": 0.25},
                    ],
                    "unmapped_importance": 0.35,
                    "tokens": [
                        {"token": "C", "position": 1, "importance": 0.5, "offsets": [0, 1]},
                    ],
                    "method": "grad_x_embedding_l2_v1+token_atom_align_v1",
                    "metadata": {
                        "model_id": "pretrained_2head_herg_chemberta",
                        "deterministic": True, "duration_ms": 900.0, "note": note,
                    },
                },
            )

        return httpx.Response(404, json={"error": "not_found", "message": request.url.path})
