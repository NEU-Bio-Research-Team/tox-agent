"""A stand-in ToxOCR client.

Implements the same interface as ``predictor.ocr_client.OcrClient`` — a single
``recognize`` coroutine — so RecognizeStructure and the intent-routing tests
can exercise "OCR is configured" without a real MolScribe model or a running
toxocr/ service.
"""
from __future__ import annotations

from toxagent.predictor.ocr_client import OcrError, OcrResult, OcrUnavailable


class StubOcrClient:
    """One fixed outcome for every call — parametrise per test rather than
    branching on the image bytes, which are opaque to this stub anyway."""

    def __init__(
        self,
        *,
        result: OcrResult | None = None,
        raise_error: Exception | None = None,
    ) -> None:
        self.result = result
        self.raise_error = raise_error
        self.calls: list[tuple[bytes, str]] = []

    async def recognize(self, image_bytes: bytes, mime_type: str) -> OcrResult:
        self.calls.append((image_bytes, mime_type))
        if self.raise_error is not None:
            raise self.raise_error
        assert self.result is not None
        return self.result

    async def aclose(self) -> None:
        pass


def stub_success(smiles: str) -> StubOcrClient:
    return StubOcrClient(result=OcrResult(smiles=smiles, canonical_smiles=smiles, confidence=0.91))


def stub_no_structure_detected() -> StubOcrClient:
    return StubOcrClient(raise_error=OcrError("no structure was detected in the image"))


def stub_unavailable() -> StubOcrClient:
    return StubOcrClient(raise_error=OcrUnavailable("cannot reach the OCR service"))
