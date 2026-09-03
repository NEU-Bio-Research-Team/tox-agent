"""ClinTox SMILES-GNN provider: clinical-trial toxicity.

The scientific code is NOT reimplemented here. The architecture, the graph
featurisation and the state-dict load all stay in ``backend/`` and are called
through ``backend.inference.load_model``; this class only supplies the artifact
boundary, the raw-probability contract and typed unavailability.

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

        self._spec.verify()

        from backend.inference import load_model

        model, tokenizer, wrapped = load_model(
            self._spec.root,
            self._config_path,
            device=self._device,
            enforce_workspace_mode=False,
        )
        vocab_size = len(tokenizer.token_to_id)
        embedding = model.state_dict().get("smiles_encoder.token_embedding.weight")
        if embedding is not None and embedding.shape[0] != vocab_size:
            raise ArtifactError(
                f"[{self.model_id}] tokenizer vocabulary ({vocab_size}) does not match the "
                f"checkpoint embedding ({embedding.shape[0]}). This tokenizer belongs to a "
                "different training run; using it would silently remap every token."
            )

        self._model = model
        self._tokenizer = tokenizer
        self._wrapped = wrapped
        self._detail = f"loaded (vocab {vocab_size})"

    def health(self) -> ModelHealth:
        if self._model is None:
            _, reason = self.availability()
            detail = self._detail if self._detail != "not loaded" else reason
            return ModelHealth(self.model_id, False, self.capabilities, detail)
        return ModelHealth(self.model_id, True, self.capabilities, self._detail)

    # -- inference ---------------------------------------------------------
    def predict(self, canonical_smiles: list[str]) -> list[dict[str, Any]]:
        import torch

        if self._wrapped is None or self._tokenizer is None:
            raise ArtifactError(f"[{self.model_id}] predict() called before load()")
        if not canonical_smiles:
            return []

        from torch.utils.data import DataLoader

        from backend.graph_data import smiles_to_pyg_data
        from backend.inference import _collate, _HybridDataset

        graphs = []
        for smiles in canonical_smiles:
            try:
                data = smiles_to_pyg_data(smiles, label=0)
            except Exception as exc:  # noqa: BLE001
                raise ArtifactError(
                    f"[{self.model_id}] cannot featurise {smiles!r}: {exc}"
                ) from exc
            if data is None:
                raise ArtifactError(
                    f"[{self.model_id}] cannot featurise {smiles!r}: RDKit produced no graph"
                )
            graphs.append(data)

        dataset = _HybridDataset(graphs, list(canonical_smiles), self._tokenizer)
        loader = DataLoader(
            dataset, batch_size=self._batch_size, shuffle=False, collate_fn=_collate
        )

        probabilities: list[float] = []
        with torch.inference_mode():
            for batch in loader:
                batch = batch.to(self._device)
                logits = self._wrapped(batch).squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
                probabilities.extend(
                    probs.tolist() if probs.ndim > 0 else [float(probs)]
                )

        if len(probabilities) != len(canonical_smiles):
            raise ArtifactError(
                f"[{self.model_id}] produced {len(probabilities)} scores for "
                f"{len(canonical_smiles)} inputs"
            )
        return [
            {
                "model_id": self.model_id,
                "clintox_probability_toxicity": float(p),
            }
            for p in probabilities
        ]


def make_factory(config_path: Path):
    def factory(spec: ArtifactSpec) -> ClinToxSmilesGnnProvider:
        return ClinToxSmilesGnnProvider(spec, config_path=config_path)

    return factory
