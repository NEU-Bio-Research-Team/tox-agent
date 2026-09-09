"""Runtime settings. The one place that reads the environment."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _int(name: str, default: int) -> int:
    raw = _env(name)
    return int(raw) if raw else default


def _bool(name: str, default: bool) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    #: "cpu" everywhere this has been run so far; "cuda" is honoured only
    #: when a GPU is actually available.
    device: str = "cpu"
    #: The checkpoint MolScribe's own README documents for general use.
    checkpoint_repo_id: str = "yujieq/MolScribe"
    checkpoint_filename: str = "swin_base_char_aux_1m.pth"
    #: Explicit local path override — set this to skip the HuggingFace
    #: download entirely (e.g. a pre-provisioned deployment).
    checkpoint_path: str = ""
    checkpoint_sha256: str = ""
    allow_checkpoint_download: bool = True
    max_image_bytes: int = 5_000_000
    #: Load (and, absent `checkpoint_path`, download) the model at process
    #: startup rather than on the first request, so `/health/ready` means
    #: what it says.
    eager_load: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            device=_env("TOXOCR_DEVICE", cls.device),
            checkpoint_repo_id=_env("TOXOCR_CHECKPOINT_REPO_ID", cls.checkpoint_repo_id),
            checkpoint_filename=_env("TOXOCR_CHECKPOINT_FILENAME", cls.checkpoint_filename),
            checkpoint_path=_env("TOXOCR_CHECKPOINT_PATH"),
            checkpoint_sha256=_env("TOXOCR_CHECKPOINT_SHA256").lower(),
            allow_checkpoint_download=_bool("TOXOCR_ALLOW_CHECKPOINT_DOWNLOAD", cls.allow_checkpoint_download),
            max_image_bytes=_int("TOXOCR_MAX_IMAGE_BYTES", cls.max_image_bytes),
            eager_load=_bool("TOXOCR_EAGER_LOAD", cls.eager_load),
        )
