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
class TokenizerRequirement:
    """The identity a model's tokenizer must have to be admitted.

    I07: the ClinTox v1 checkpoint's embedding matrix is (69, 96) — a 69-token
    vocabulary derived from its training corpus, which the 80-token SMILES
    tokenizers on disk do not match. The requirement lived in a prose
    `blocked_reason` and in a provider docstring, so "is this the right
    tokenizer?" was a question only a person reading two files could answer,
    and the only safe implementation was to refuse every tokenizer including a
    correct one.

    Declaring it makes admission a check. `vocab_size` can be verified against
    the checkpoint that is here today. `sha256` and `vocab_sha256` are what
    proves a *particular* file is the training artifact rather than a
    different tokenizer of the same size; until they are recorded from the
    training run, no tokenizer can be admitted, and the reason a deployment is
    given says exactly which of these is missing rather than "restore it".
    """

    relative_path: str
    vocab_size: int | None = None
    #: sha256 of the tokenizer file itself.
    sha256: str | None = None
    #: sha256 over the canonical token-to-id mapping, so a tokenizer
    #: re-serialised by a different library version is still recognisable.
    vocab_sha256: str | None = None
    #: Where in the checkpoint the vocabulary size can be read back.
    checkpoint_embedding_key: str | None = None

    @property
    def identity_recorded(self) -> bool:
        """Whether the manifest can tell one tokenizer of this size from another."""
        return bool(self.sha256 or self.vocab_sha256)


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
    model_config_path: Path | None = None
    """Optional architecture config living outside the artifact directory."""
    blocked_reason: str = ""
    tokenizer: "TokenizerRequirement | None" = None
    """Set when the model needs a tokenizer whose identity has to be proved."""
    declared_thresholds: Mapping[str, float] = field(default_factory=dict)
    """Operating points chosen in the manifest rather than calibrated with the
    weights. Surfaced as ``threshold_source="manifest_declared"`` so a reader can
    tell them from the artifact's own calibrated values."""

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
    """Parse a manifest or manifest index into artifact specifications.

    A registry index may contain ``includes`` pointing at per-model manifests.
    Every included release is parsed independently (so its relative config
    paths remain local to that release), while the index owns the common
    artifact root.  The older single-file ``models`` form remains supported
    for external deployments that have not split their registry yet.
    """
    manifest_path = Path(manifest_path)
    raw = yaml.safe_load(manifest_path.read_text()) or {}
    if int(raw.get("schema_version", 0)) != 1:
        raise ArtifactError(
            f"unsupported manifest schema_version: {raw.get('schema_version')!r}"
        )

    base = Path(models_root) if models_root else (manifest_path.parent / raw.get("models_root", "."))
    base = base.resolve()

    specs: dict[str, ArtifactSpec] = {}
    for include in raw.get("includes") or ():
        child_path = (manifest_path.parent / str(include)).resolve()
        if not child_path.is_file():
            raise ArtifactError(f"manifest include does not exist: {include}")
        child_specs = load_manifest(child_path, models_root=base)
        for model_id, spec in child_specs.items():
            if model_id in specs:
                raise ArtifactError(f"duplicate model_id across manifests: {model_id}")
            specs[model_id] = spec

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
            model_config_path=(
                (manifest_path.parent / entry["model_config"]).resolve()
                if entry.get("model_config") else None
            ),
            blocked_reason=str(entry.get("blocked_reason", "")).strip(),
            tokenizer=(
                TokenizerRequirement(
                    relative_path=str(entry["tokenizer"]["path"]),
                    vocab_size=(
                        int(entry["tokenizer"]["vocab_size"])
                        if entry["tokenizer"].get("vocab_size") is not None else None
                    ),
                    sha256=entry["tokenizer"].get("sha256") or None,
                    vocab_sha256=entry["tokenizer"].get("vocab_sha256") or None,
                    checkpoint_embedding_key=(
                        entry["tokenizer"].get("checkpoint_embedding_key") or None
                    ),
                )
                if entry.get("tokenizer") else None
            ),
            declared_thresholds={
                str(k): float(v) for k, v in (entry.get("declared_thresholds") or {}).items()
            },
        )
    if not specs:
        raise ArtifactError("manifest declares no models")
    return specs
