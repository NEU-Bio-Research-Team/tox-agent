"""Loads MolScribe once and exposes image -> SMILES.

Ported from this codebase's pre-refactor agent-layer (model_server/main.py at
tag archive/agent-layer-165319beede5), which ran this exact library and
checkpoint successfully before the predictor-only rebuild moved OCR out of
the predictor's own dependency footprint (docs/refactor/PREDICTOR_ONLY_STATUS_VI.md).
This service is where that capability belongs now.
"""
from __future__ import annotations

import io
import hashlib
from pathlib import Path
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from ..settings import Settings


class ImageDecodeError(Exception):
    """The bytes are not a decodable PNG/JPEG/WebP image."""


class StructureNotDetected(Exception):
    """MolScribe ran, but produced nothing a chemistry toolkit accepts."""


@dataclass(frozen=True)
class RecognitionResult:
    smiles: str
    canonical_smiles: str
    confidence: float | None


class MolScribePredictor:
    """One model instance per process. Loading (and, without an explicit
    ``checkpoint_path``, downloading the checkpoint from HuggingFace) happens
    at most once, behind a lock — two requests racing the cold-start path
    must not both pay for it."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._model: Any = None
        self._device = "unloaded"
        self._checkpoint_fingerprint: str | None = None

    def is_ready(self) -> bool:
        return self._model is not None

    def runtime_status(self) -> tuple[str, str | None]:
        return self._device, self._checkpoint_fingerprint

    def preload(self) -> None:
        self._load()

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            import torch
            from molscribe import MolScribe

            if self._settings.device not in {"cpu", "cuda"}:
                raise RuntimeError("TOXOCR_DEVICE must be exactly 'cpu' or 'cuda'")
            if self._settings.device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("TOXOCR_DEVICE=cuda was requested but CUDA is unavailable")

            checkpoint_path = self._settings.checkpoint_path
            if not checkpoint_path:
                if not self._settings.allow_checkpoint_download:
                    raise RuntimeError("a local TOXOCR_CHECKPOINT_PATH is required when downloads are disabled")
                from huggingface_hub import hf_hub_download

                checkpoint_path = hf_hub_download(
                    repo_id=self._settings.checkpoint_repo_id,
                    filename=self._settings.checkpoint_filename,
                )
            checkpoint = Path(checkpoint_path)
            if not checkpoint.is_file():
                raise RuntimeError(f"MolScribe checkpoint does not exist: {checkpoint}")
            # ``hashlib.file_digest`` starts at Python 3.11; this service is
            # intentionally pinned to Python 3.10 for MolScribe compatibility.
            hasher = hashlib.sha256()
            with checkpoint.open("rb") as checkpoint_file:
                for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
            if self._settings.checkpoint_sha256 and digest != self._settings.checkpoint_sha256:
                raise RuntimeError("MolScribe checkpoint SHA-256 does not match TOXOCR_CHECKPOINT_SHA256")
            device = torch.device(self._settings.device)
            self._model = MolScribe(checkpoint_path, device=device)
            self._device = str(device)
            self._checkpoint_fingerprint = f"sha256:{digest}"
            return self._model

    @staticmethod
    def _decode_image(raw_bytes: bytes) -> np.ndarray:
        try:
            with Image.open(io.BytesIO(raw_bytes)) as image:
                normalized = ImageOps.autocontrast(image.convert("RGB"))
                return np.asarray(normalized, dtype=np.uint8)
        except UnidentifiedImageError as exc:
            raise ImageDecodeError("image content could not be decoded") from exc

    def recognize(self, raw_bytes: bytes) -> RecognitionResult:
        """Synchronous and CPU/GPU-bound — callers run this in a thread."""
        model = self._load()
        image_rgb = self._decode_image(raw_bytes)
        output = model.predict_image(image_rgb, return_confidence=True)
        smiles = str((output or {}).get("smiles") or "").strip()
        if not smiles:
            raise StructureNotDetected("no SMILES sequence was detected in the image")

        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise StructureNotDetected("extracted text is not a valid SMILES sequence")
        canonical_smiles = Chem.MolToSmiles(mol)

        confidence_raw = output.get("confidence")
        confidence = (
            float(max(0.0, min(1.0, confidence_raw)))
            if isinstance(confidence_raw, (int, float))
            else None
        )
        return RecognitionResult(smiles=smiles, canonical_smiles=canonical_smiles, confidence=confidence)
