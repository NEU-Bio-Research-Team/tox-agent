"""ClinTox SMILES-GNN legacy artifact admission guard.

The v1 checkpoint lacks its exact tokenizer and is therefore never admitted.
The old backend-dependent inference path is intentionally not part of the
standalone wheel. A reproducible retrain must ship as ``clintox-smilesgnn-v2``
with its own provider, tokenizer, manifest, calibrator and evaluation.

Two things differ from the code path it wraps:

* ``predict`` returns raw probabilities. The wrapped ``predict_batch`` returns a
  DataFrame that has already thresholded, sorted by score and rendered labels;
  none of that survives here, because the label belongs to the policy layer.
* An unfeaturisable molecule raises instead of becoming a ``"Parse error"`` row
  with ``P(toxic) = None``, which a caller can misread as a low score.

Availability
------------
This provider needs ``tokenizer.pkl`` next to the checkpoint. That file is
absent from the repository and is excluded by ``.gitignore`` (``*.pkl``), and
the checkpoint's embedding matrix is (69, 96) — a 69-token vocabulary derived
from the ClinTox training corpus, which the other SMILES tokenizers on disk (80
tokens) do not match. Until it is restored or the model is retrained, ``load()``
raises ``ArtifactError`` and the registry leaves the provider unregistered
rather than serving a different model in its place.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..artifacts import ArtifactError, ArtifactSpec
from ..registry import ModelHealth
from .contracts import ClinToxRawOutput, ProviderBatchResult

MODEL_ID = "clintox-smilesgnn-v1"
CAPABILITIES = frozenset({"clintox"})

TOKENIZER_FILENAME = "tokenizer.pkl"


class ClinToxSmilesGnnProvider:
    model_id = MODEL_ID
    capabilities = CAPABILITIES

    def __init__(
        self,
        spec: ArtifactSpec,
        config_path: Path,
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        self._spec = spec
        self._config_path = Path(config_path)
        self._device = device
        self._batch_size = int(batch_size)
        self._model = None
        self._wrapped = None
        self._tokenizer = None
        self._detail = "not loaded"

    # -- availability ------------------------------------------------------
    @property
    def tokenizer_path(self) -> Path:
        return self._spec.root / TOKENIZER_FILENAME

    def availability(self) -> tuple[bool, str]:
        """Why this provider can or cannot load, without loading it."""
        if not self._spec.root.is_dir():
            return False, f"artifact directory missing: {self._spec.root}"
        if not (self._spec.root / "best_model.pt").is_file():
            return False, "checkpoint missing: best_model.pt"
        if not self.tokenizer_path.is_file():
            return False, (
                f"tokenizer missing: {TOKENIZER_FILENAME}. The checkpoint was trained with a "
                "69-token vocabulary derived from the ClinTox corpus; without that vocabulary "
                "the token ids cannot be reproduced and the embedding weights are unusable. "
                "Restore it from the training run, or retrain with scripts/train_hybrid.py and "
                "commit the tokenizer alongside the weights."
            )
        if not self._config_path.is_file():
            return False, f"model config missing: {self._config_path}"
        return True, "ready to load"

    # -- lifecycle ---------------------------------------------------------
    def load(self) -> None:
        available, reason = self.availability()
        if not available:
            self._detail = reason
            raise ArtifactError(f"[{self.model_id}] {reason}")

        raise ArtifactError(
            f"[{self.model_id}] legacy v1 admission is disabled: even a file named "
            f"{TOKENIZER_FILENAME!r} is not sufficient proof that its vocabulary mapping "
            "matches the missing training artifact. Restore and hash-verify the exact "
            "69-token vocabulary, or retrain and release clintox-smilesgnn-v2."
        )

    def health(self) -> ModelHealth:
        if self._model is None:
            _, reason = self.availability()
            detail = self._detail if self._detail != "not loaded" else reason
            return ModelHealth(self.model_id, False, self.capabilities, detail)
        return ModelHealth(self.model_id, True, self.capabilities, self._detail)

    # -- inference ---------------------------------------------------------
    def predict(self, canonical_smiles: list[str]) -> ProviderBatchResult[ClinToxRawOutput]:
        raise ArtifactError(f"[{self.model_id}] predict() called before load()")


def make_factory(config_path: Path, device: str = "cpu"):
    def factory(spec: ArtifactSpec) -> ClinToxSmilesGnnProvider:
        return ClinToxSmilesGnnProvider(spec, config_path=config_path, device=device)

    return factory
