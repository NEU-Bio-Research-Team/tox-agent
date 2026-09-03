"""Artifact manifest and integrity verification.

Rules this enforces, from the refactor plan:

1. A directory existing is not evidence of a valid artifact.
2. Every declared file is checksummed before the model is loaded.
3. A missing or corrupt required artifact fails loudly. There is no silent
   substitution of another model — the behaviour that let
   ``DEFAULT_TOX_TYPE_MODEL_KEY="tox21_ensemble_3_best"`` point at
   ``models/dualhead_ensemble3/``, a directory holding a metrics JSON and no
   weights at all.
4. Thresholds and the tokenizer are part of the same model release as the
   weights, so they are checksummed alongside them.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

CHUNK = 1 << 20


class ArtifactError(RuntimeError):
    """Raised when an artifact is missing, incomplete or fails verification."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class ArtifactFile:
    relative_path: str
    sha256: str
    bytes: int | None = None


@dataclass(frozen=True)
class ArtifactSpec:
    model_id: str
    provider: str
    capabilities: frozenset[str]
    root: Path
    files: tuple[ArtifactFile, ...]
    required: bool = True
    base_model: Mapping[str, Any] = field(default_factory=dict)
    feature_schema_version: str = "unknown"
    notes: str = ""

    def verify(self) -> None:
        """Check every declared file exists and matches its checksum.

        Raises ArtifactError listing *all* problems rather than the first, so a
        broken deployment is diagnosed in one pass.
        """
        problems: list[str] = []
        if not self.root.is_dir():
            raise ArtifactError(
                f"[{self.model_id}] artifact root is not a directory: {self.root}"
            )
        for entry in self.files:
            path = self.root / entry.relative_path
            if not path.is_file():
                problems.append(f"missing file: {entry.relative_path}")
                continue
            if entry.bytes is not None and path.stat().st_size != entry.bytes:
                problems.append(
                    f"size mismatch: {entry.relative_path} "
                    f"(expected {entry.bytes}, got {path.stat().st_size})"
                )
                continue
            actual = sha256_file(path)
            if actual != entry.sha256:
                problems.append(
                    f"checksum mismatch: {entry.relative_path}\n"
                    f"      expected {entry.sha256}\n"
                    f"      actual   {actual}"
                )
        if problems:
            raise ArtifactError(
                f"[{self.model_id}] artifact verification failed ({len(problems)} problem(s)):\n  - "
                + "\n  - ".join(problems)
            )

    def path(self, relative: str) -> Path:
        p = self.root / relative
        if not p.exists():
            raise ArtifactError(f"[{self.model_id}] declared file absent: {relative}")
        return p


def load_manifest(manifest_path: Path, models_root: Path | None = None) -> dict[str, ArtifactSpec]:
    """Parse an artifact manifest into ArtifactSpecs. Does not read weights."""
    manifest_path = Path(manifest_path)
    raw = yaml.safe_load(manifest_path.read_text()) or {}
    if int(raw.get("schema_version", 0)) != 1:
        raise ArtifactError(
            f"unsupported manifest schema_version: {raw.get('schema_version')!r}"
        )

    base = Path(models_root) if models_root else (manifest_path.parent / raw.get("models_root", "."))
    base = base.resolve()

    specs: dict[str, ArtifactSpec] = {}
    for entry in raw.get("models") or []:
        model_id = entry["model_id"]
        if model_id in specs:
            raise ArtifactError(f"duplicate model_id in manifest: {model_id}")
        files = tuple(
            ArtifactFile(
                relative_path=f["path"], sha256=f["sha256"], bytes=f.get("bytes")
            )
            for f in entry.get("files") or []
        )
        if not files:
            raise ArtifactError(f"[{model_id}] manifest declares no files")
        capabilities = frozenset(entry.get("capabilities") or ())
        if not capabilities:
            raise ArtifactError(f"[{model_id}] manifest declares no capabilities")
        specs[model_id] = ArtifactSpec(
            model_id=model_id,
            provider=entry["provider"],
            capabilities=capabilities,
            root=(base / entry["artifact_dir"]).resolve(),
            files=files,
            required=bool(entry.get("required", True)),
            base_model=entry.get("base_model") or {},
            feature_schema_version=str(entry.get("feature_schema_version", "unknown")),
            notes=str(entry.get("notes", "")),
        )
    if not specs:
        raise ArtifactError("manifest declares no models")
    return specs
