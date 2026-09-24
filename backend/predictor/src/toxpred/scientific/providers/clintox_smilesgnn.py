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

I07: those admission criteria used to live only in prose here and in a
``blocked_reason`` string, which had two consequences. Nothing checked the
checkpoint actually had the vocabulary the prose claimed, so a swapped
checkpoint would have been described wrongly and confidently. And because a
file named ``tokenizer.pkl`` proves nothing about its contents, the only safe
implementation was to refuse *every* tokenizer, correct ones included — so the
endpoint could never be restored without editing this file.

``ArtifactSpec.tokenizer`` now carries the criteria as data:
``vocab_size``, checked against the checkpoint itself; ``sha256`` and
``vocab_sha256``, which identify a particular file as the training artifact
rather than a coincidentally sized one. Those two are null in the manifest,
because the training run did not record them and this repository has never
held the file — so nothing is admitted, exactly as before. What changed is
that ``admission_report()`` says which criterion is unmet, and that supplying
it is a manifest change rather than a code change.

Even a fully verified tokenizer would not make this provider serve: the v1
inference path depended on the old backend and is deliberately not in the
standalone wheel. Restoring the endpoint means a v2 release, and this guard's
job is to make sure nothing is served in the meantime.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..artifacts import ArtifactError, ArtifactSpec, sha256_file
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
        requirement = self._spec.tokenizer
        name = requirement.relative_path if requirement else TOKENIZER_FILENAME
        return self._spec.root / name

    def checkpoint_vocab_size(self) -> int | None:
        """The vocabulary size the weights themselves were trained with.

        Read from the embedding matrix named in the manifest, so the claim
        "this checkpoint is a 69-token model" is checked rather than asserted.
        Returns None when the checkpoint cannot be read or names no such
        tensor — the caller reports that as an unmet criterion, never as a
        pass.
        """
        requirement = self._spec.tokenizer
        if requirement is None or not requirement.checkpoint_embedding_key:
            return None
        checkpoint = self._spec.root / "best_model.pt"
        if not checkpoint.is_file():
            return None
        try:
            import torch

            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except Exception:  # noqa: BLE001 — an unreadable checkpoint is not a size
            return None
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            return None
        tensor = state.get(requirement.checkpoint_embedding_key)
        shape = getattr(tensor, "shape", None)
        if shape is None or len(shape) < 1:
            return None
        return int(shape[0])

    def admission_report(self) -> tuple[bool, list[str]]:
        """Every admission criterion this artifact fails, not just the first.

        A deployment that is told one problem at a time fixes them one restart
        at a time.
        """
        unmet: list[str] = []
        if not self._spec.root.is_dir():
            return False, [f"artifact directory missing: {self._spec.root}"]
        if not (self._spec.root / "best_model.pt").is_file():
            unmet.append("checkpoint missing: best_model.pt")
        if not self._config_path.is_file():
            unmet.append(f"model config missing: {self._config_path}")

        requirement = self._spec.tokenizer
        if requirement is None:
            unmet.append(
                "the manifest declares no tokenizer requirement, so there is nothing to "
                "verify a restored tokenizer against"
            )
            return False, unmet

        declared = requirement.vocab_size
        actual = self.checkpoint_vocab_size()
        if declared is not None and actual is not None and declared != actual:
            unmet.append(
                f"the checkpoint's {requirement.checkpoint_embedding_key} has {actual} rows, "
                f"but the manifest declares a {declared}-token vocabulary; one of them "
                "describes a different model"
            )

        if not requirement.identity_recorded:
            unmet.append(
                f"the manifest records no sha256 or vocab_sha256 for "
                f"{requirement.relative_path}, so a file of the right size could not be told "
                "from the training artifact. Record them from the training run, or release "
                "clintox-smilesgnn-v2 with its own tokenizer, calibration and evaluation"
            )

        if not self.tokenizer_path.is_file():
            unmet.append(
                f"tokenizer missing: {requirement.relative_path}. The checkpoint was trained "
                f"with a {declared}-token vocabulary derived from the ClinTox corpus; without "
                "it the token ids cannot be reproduced and the embedding weights are unusable"
            )
        elif requirement.sha256:
            actual_sha = sha256_file(self.tokenizer_path)
            if actual_sha != requirement.sha256:
                unmet.append(
                    f"tokenizer checksum mismatch for {requirement.relative_path}: expected "
                    f"{requirement.sha256}, got {actual_sha}"
                )
        return not unmet, unmet

    def availability(self) -> tuple[bool, str]:
        """Why this provider can or cannot load, without loading it."""
        admissible, unmet = self.admission_report()
        if admissible:
            return True, "every declared admission criterion is met"
        return False, "; ".join(unmet)

    # -- lifecycle ---------------------------------------------------------
    def load(self) -> None:
        admissible, unmet = self.admission_report()
        if not admissible:
            self._detail = "; ".join(unmet)
            raise ArtifactError(
                f"[{self.model_id}] not admitted ({len(unmet)} unmet criterion(s)):\n  - "
                + "\n  - ".join(unmet)
            )

        # Reached only once the manifest records the tokenizer's identity and
        # a file matching it is present. It still does not serve: the v1
        # inference path depended on the old backend and is not in this wheel,
        # and inventing one here would be a new model wearing v1's name.
        raise ArtifactError(
            f"[{self.model_id}] the tokenizer is verified, but v1 has no inference path in "
            "this package: the original depended on the retired backend. Release "
            "clintox-smilesgnn-v2 with its own provider, calibration and evaluation."
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
