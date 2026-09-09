"""The ToxOCR client.

A third external boundary alongside the predictor and the runtime (ADR 0001):
this control plane never imports a vision/OCR model, only talks to whatever
serves ``POST /v1/structure-recognition`` over HTTP — see ../../../toxocr/.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import httpx


class OcrError(Exception):
    """The service answered, but found no structure — never retryable."""


class OcrUnavailable(Exception):
    """The service could not be reached or answered something unexpected —
    a deployment fact, not something about the image."""


@dataclass(frozen=True)
class OcrResult:
    smiles: str
    canonical_smiles: str
    confidence: float | None


class OcrClient:
    def __init__(self, base_url: str, *, timeout_s: float = 60.0, connect_timeout_s: float = 5.0) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_s, connect=connect_timeout_s),
            headers={"accept": "application/json"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def recognize(self, image_bytes: bytes, mime_type: str) -> OcrResult:
        try:
            response = await self._client.post(
                "/v1/structure-recognition",
                json={"mime_type": mime_type, "data_base64": base64.b64encode(image_bytes).decode()},
            )
        except httpx.TimeoutException as exc:
            raise OcrUnavailable("the OCR service did not answer within its budget") from exc
        except httpx.HTTPError as exc:
            raise OcrUnavailable(f"cannot reach the OCR service: {exc}") from exc

        if response.status_code == 422:
            raise OcrError(self._message(response, "no structure was detected in the image"))
        if not response.is_success:
            raise OcrUnavailable(f"the OCR service returned {response.status_code}")

        try:
            body: dict[str, Any] = response.json()
            confidence_raw = body.get("confidence")
            # The OCR service contract allows this field to be absent. Be
            # strict about its numeric range at the boundary so a malformed
            # remote response cannot become a misleading UI percentage.
            confidence = (
                float(confidence_raw)
                if isinstance(confidence_raw, (int, float))
                and not isinstance(confidence_raw, bool)
                and 0.0 <= float(confidence_raw) <= 1.0
                else None
            )
            return OcrResult(
                smiles=str(body["smiles"]),
                canonical_smiles=str(body["canonical_smiles"]),
                confidence=confidence,
            )
        except (ValueError, KeyError) as exc:
            raise OcrUnavailable(f"the OCR service's response did not match the expected contract: {exc}") from exc

    @staticmethod
    def _message(response: httpx.Response, default: str) -> str:
        try:
            body = response.json()
        except ValueError:
            return default
        return str(((body or {}).get("error") or {}).get("message") or default)
