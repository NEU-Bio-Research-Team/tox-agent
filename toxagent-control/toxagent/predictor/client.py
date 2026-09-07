"""The ToxPred client.

The only place the control plane talks to the predictor. Three things happen
here and nowhere else: HTTP status codes become typed product errors, the
response is validated against the pinned contract before anything stores it, and
the predictor's provenance is copied out verbatim (SCI-10).

There is no retry on ``/v1/predictions`` beyond a connect-level one. A predictor
that answered slowly has still done the forward pass, and a duplicate request
buys a second bill for the same number.
"""
from __future__ import annotations

import re
from typing import Any, Mapping

import httpx

from ..config import PredictorSettings
from ..domain.analysis import PredictorProvenance
from ..domain.errors import (
    EndpointUnavailable,
    InvalidSmiles,
    PredictorNotReady,
    PredictorProtocolError,
)
from .schemas import (
    AttributionResponse,
    BatchPredictionResponse,
    ExplanationResponse,
    ModelsResponse,
    PredictionResponse,
    ReadinessResponse,
)

#: toxpred/application/predictor.py's exact, deterministic phrasing for "this
#: build will never serve that endpoint" — not free-form text, a fixed
#: f-string this one code path always produces.
_ENDPOINT_NOT_SERVED_BY_BUILD = re.compile(r"endpoint '([^']+)' is not served by this build")


def _artifact_hash_strings(artifacts: Any) -> list[str]:
    """Reduce the predictor's ``artifacts``/``artifact_hashes`` provenance
    field to the flat list of hash strings the ``PredictorProvenance``
    dataclass actually names. The real ToxPred contract sends a *list of
    dicts* (``[{"model_id": ..., "weights_sha256": ..., "tokenizer_sha256":
    ..., ...}]``); a naive ``str()`` on each element turned that into a
    Python dict repr instead of structured hashes. The full artifact dicts
    are never lost — they stay in ``raw`` regardless."""
    if isinstance(artifacts, Mapping):
        return [f"{k}:{v}" for k, v in sorted(artifacts.items())]
    hashes: list[str] = []
    for artifact in artifacts or []:
        if not isinstance(artifact, Mapping):
            hashes.append(str(artifact))
            continue
        model_id = artifact.get("model_id", "unknown")
        for key, value in sorted(artifact.items()):
            if key.endswith("_sha256") or key.endswith("_hash"):
                hashes.append(f"{model_id}:{key}={value}")
    return hashes


class PredictorClient:
    def __init__(
        self, settings: PredictorSettings, *, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.base_url,
            timeout=httpx.Timeout(
                settings.read_timeout_s, connect=settings.connect_timeout_s
            ),
            transport=transport,
            headers={"accept": "application/json"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    @property
    def base_url_id(self) -> str:
        return self._settings.base_url_id

    # --- health and inventory ---------------------------------------------

    async def ready(self) -> ReadinessResponse:
        response = await self._request("GET", "/health/ready")
        return self._parse(ReadinessResponse, response)

    async def models(self) -> ModelsResponse:
        response = await self._request("GET", "/v1/models")
        return self._parse(ModelsResponse, response)

    async def served_endpoints(self) -> tuple[str, ...]:
        return tuple((await self.models()).served_endpoints)

    # --- prediction --------------------------------------------------------

    async def predict(
        self,
        smiles: str,
        endpoints: tuple[str, ...] | None = None,
        *,
        threshold_overrides: Mapping[str, Any] | None = None,
    ) -> PredictionResponse:
        body: dict[str, Any] = {"smiles": smiles}
        if endpoints:
            body["endpoints"] = list(endpoints)
        if threshold_overrides:
            body["threshold_overrides"] = dict(threshold_overrides)
        response = await self._request("POST", "/v1/predictions", json=body)
        result = self._parse_prediction(response.json())
        self._assert_requested_endpoints_served(result, endpoints)
        return result

    async def predict_batch(
        self,
        smiles: list[str],
        endpoints: tuple[str, ...] | None = None,
        *,
        threshold_overrides: Mapping[str, Any] | None = None,
    ) -> BatchPredictionResponse:
        if len(smiles) > self._settings.max_batch_size:
            raise PredictorProtocolError(
                f"batch of {len(smiles)} exceeds the predictor's documented maximum "
                f"of {self._settings.max_batch_size}",
            )
        body: dict[str, Any] = {"smiles": smiles}
        if endpoints:
            body["endpoints"] = list(endpoints)
        if threshold_overrides:
            body["threshold_overrides"] = dict(threshold_overrides)
        response = await self._request("POST", "/v1/predictions:batch", json=body)
        payload = response.json()
        result = self._parse(BatchPredictionResponse, response)
        # Keep each item's raw payload too, so a batch member can be snapshotted
        # with the same losslessness as a single prediction.
        for parsed, raw in zip(result.results, payload.get("results", [])):
            parsed._raw = raw
        return result

    async def attribution(
        self, smiles: str, endpoint: str, task: str | None = None
    ) -> AttributionResponse:
        body: dict[str, Any] = {"smiles": smiles, "endpoint": endpoint}
        if task is not None:
            body["task"] = task
        response = await self._request(
            "POST", "/v1/attributions", json=body,
            timeout=httpx.Timeout(
                self._settings.attribution_read_timeout_s,
                connect=self._settings.connect_timeout_s,
            ),
        )
        return self._parse(AttributionResponse, response)

    async def explain(
        self, smiles: str, endpoint: str, task: str | None = None
    ) -> ExplanationResponse:
        """Atom-level explanation via ToxPred ``POST /v1/explanations``. Same
        generous read budget as ``attribution`` — it is the same backward pass
        plus a deterministic token->atom walk."""
        body: dict[str, Any] = {"smiles": smiles, "endpoint": endpoint}
        if task is not None:
            body["task"] = task
        response = await self._request(
            "POST", "/v1/explanations", json=body,
            timeout=httpx.Timeout(
                self._settings.attribution_read_timeout_s,
                connect=self._settings.connect_timeout_s,
            ),
        )
        return self._parse(ExplanationResponse, response)

    # --- provenance --------------------------------------------------------

    def provenance_of(self, response: PredictionResponse) -> PredictorProvenance:
        """Lossless copy. The raw provenance mapping is kept as-is alongside the
        fields the control plane indexes on."""
        raw = dict(response.provenance)
        return PredictorProvenance(
            base_url_id=self._settings.base_url_id,
            # The pinned ToxPred contract's own field is `predictor_version`;
            # `service_version`/`version` are kept first only in case a future
            # or alternate deployment renames it.
            service_version=raw.get("service_version") or raw.get("predictor_version")
            or raw.get("version"),
            git_commit=raw.get("git_commit") or raw.get("commit"),
            artifact_hashes=tuple(
                _artifact_hash_strings(raw.get("artifacts") or raw.get("artifact_hashes") or [])
            ),
            raw=raw,
        )

    # --- plumbing ----------------------------------------------------------

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.ConnectError as exc:
            raise PredictorNotReady(f"cannot reach the predictor at {self._settings.base_url}") from exc
        except httpx.TimeoutException as exc:
            raise PredictorNotReady("the predictor did not answer within its budget") from exc
        except httpx.HTTPError as exc:
            raise PredictorProtocolError(f"predictor transport failure: {exc}") from exc
        self._raise_for_status(response)
        return response

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        body = self._body(response)
        code = str(body.get("error") or "")
        message = str(body.get("message") or response.text[:400])
        detail = body.get("detail") if isinstance(body.get("detail"), dict) else {}

        if response.status_code == 400 and code == "invalid_smiles":
            # SCI-08: a molecule the predictor cannot parse is a validation
            # failure, never a prediction of zero risk.
            raise InvalidSmiles(message, **detail)
        if response.status_code in (400, 422):
            raise PredictorProtocolError(message, status=response.status_code, predictor_code=code)
        if response.status_code == 503:
            # ToxPred has one status/code for every "a model is missing,
            # corrupt or unloaded" case (toxpred/api/errors.py's
            # artifact_error_handler) — a predictor still warming up at
            # startup and an endpoint this build will *never* serve (e.g.
            # ClinTox, missing its tokenizer) both come back as 503
            # model_not_ready. Only the message tells them apart, but the
            # phrasing for the permanent case is deterministic, not
            # free-form text — application/predictor.py always raises it
            # verbatim as "endpoint '<name>' is not served by this build
            # (available: [...])" for exactly this case. Retrying it, which
            # PredictorNotReady's retryable=True invites, can never help;
            # found live (progress log §3.13) reporting a permanently
            # unavailable endpoint as "not ready yet" instead of the
            # SCI-06-typed EndpointUnavailable.
            not_served = _ENDPOINT_NOT_SERVED_BY_BUILD.search(message)
            if code == "model_not_ready" and not_served:
                raise EndpointUnavailable(message, requested=[not_served.group(1)])
            raise PredictorNotReady(message, predictor_code=code)
        raise PredictorProtocolError(
            f"predictor returned {response.status_code}", status=response.status_code, body=message
        )

    @staticmethod
    def _body(response: httpx.Response) -> dict[str, Any]:
        try:
            body = response.json()
        except ValueError:
            return {}
        return body if isinstance(body, dict) else {}

    @staticmethod
    def _parse_prediction(payload: Any) -> PredictionResponse:
        try:
            return PredictionResponse.parse_lossless(payload)
        except ValueError as exc:
            raise PredictorProtocolError(
                f"the predictor's response does not match the pinned contract: {exc}"
            ) from exc

    @staticmethod
    def _parse(model, response: httpx.Response):
        try:
            return model.model_validate(response.json())
        except ValueError as exc:  # includes ValidationError and JSON errors
            raise PredictorProtocolError(
                f"the predictor's response does not match the pinned contract: {exc}"
            ) from exc

    @staticmethod
    def _assert_requested_endpoints_served(
        result: PredictionResponse, requested: tuple[str, ...] | None
    ) -> None:
        """SCI-06. An endpoint this build does not serve fails loudly here, at
        the point where the caller can still be told which one — rather than
        producing a snapshot whose missing section is discovered later by a
        model, which would then be free to reach for a different endpoint."""
        if not requested:
            return
        missing = sorted(set(requested) - set(result.served_endpoints()))
        if missing:
            raise EndpointUnavailable(
                f"the predictor does not serve {', '.join(missing)}; there is no substitute",
                requested=list(requested),
                served=list(result.served_endpoints()),
                unavailable=missing,
            )
