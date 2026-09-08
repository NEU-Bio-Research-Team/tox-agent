"""ChemBERTa dual-head provider: hERG blockade + 12 Tox21 assay activities.

Ported from ``backend/pretrained_mol_model.py`` and the dual-head branch of
``backend/inference.py``, with three deliberate changes:

* the hERG head's output is returned as ``herg_probability_blocker`` and is
  never routed into a clinical field;
* no thresholding happens here — the provider returns raw probabilities and the
  policy layer decides labels;
* the checkpoint's task order is checked against the frozen constant at load,
  so a re-trained artifact with a different column order fails loudly instead of
  silently relabelling assays.

Training-time code paths (optimiser state, loss, gradient hooks) are not ported.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ...domain.endpoints import TOX21_TASKS, validate_task_order
from ..artifacts import ArtifactError, ArtifactSpec
from ..registry import ModelHealth
from .contracts import HergTox21RawOutput, ProviderBatchResult, TokenizationProvenance

MODEL_ID = "herg-tox21-chemberta-v1"
CAPABILITIES = frozenset({"herg", "tox21"})


class HergTox21ChembertaProvider:
    """Lazy-loading provider. Constructing it must not read the checkpoint."""

    model_id = MODEL_ID
    capabilities = CAPABILITIES

    def __init__(self, spec: ArtifactSpec, device: str = "cpu", batch_size: int = 32) -> None:
        self._spec = spec
        self._device = device
        self._batch_size = int(batch_size)
        self._model = None
        self._tokenizer = None
        self._max_length = 128
        self._herg_threshold: float | None = None
        self._tox21_thresholds: dict[str, float] | None = None
        self._detail = "not loaded"
        self._base_model_revision: str | None = None

    # -- artifact-declared thresholds -------------------------------------
    @property
    def artifact_herg_threshold(self) -> float:
        if self._herg_threshold is None:
            raise ArtifactError(f"[{self.model_id}] load() must run before reading thresholds")
        return self._herg_threshold

    @property
    def artifact_tox21_thresholds(self) -> dict[str, float]:
        if self._tox21_thresholds is None:
            raise ArtifactError(f"[{self.model_id}] load() must run before reading thresholds")
        return dict(self._tox21_thresholds)

    # -- lifecycle ---------------------------------------------------------
    def load(self) -> None:
        import torch
        from transformers import AutoTokenizer

        from ..models.chemberta_dual_head import create_chemberta_dual_head_model

        self._spec.verify()

        ckpt_path = self._spec.path("best_model.pt")
        checkpoint = torch.load(ckpt_path, map_location=self._device, weights_only=False)
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ArtifactError(
                f"[{self.model_id}] checkpoint is not a dual-head bundle "
                f"(expected a dict with 'model_state_dict', got {type(checkpoint).__name__})"
            )

        task_names = list(checkpoint.get("task_names") or [])
        validate_task_order(task_names)

        self._max_length = int(checkpoint.get("max_length", 128))
        base_model = str(checkpoint.get("pretrained_model") or "")
        declared = str((self._spec.base_model or {}).get("id") or "")
        if declared and base_model and declared != base_model:
            raise ArtifactError(
                f"[{self.model_id}] base model disagreement: manifest says {declared!r}, "
                f"checkpoint says {base_model!r}"
            )

        # Prefer the vendored architecture config: the checkpoint carries every
        # backbone weight, so with the config on disk nothing needs to reach
        # Hugging Face and the service starts with no network at all.
        base_config_dir = None
        vendored = self._spec.root / "base_model"
        if (vendored / "config.json").is_file():
            base_config_dir = str(vendored)
            self._base_model_revision = (
                (vendored / "REVISION").read_text().strip()
                if (vendored / "REVISION").is_file() else None
            )

        model_config = dict(checkpoint.get("model_config") or {})
        model = create_chemberta_dual_head_model(
            pretrained_model=base_model,
            num_tox21_tasks=len(TOX21_TASKS),
            dropout=float(model_config.get("dropout", 0.1)),
            use_herg_mlp=bool(model_config.get("use_herg_mlp", True)),
            herg_hidden_dim=int(model_config.get("herg_hidden_dim", 192)),
            base_config_dir=base_config_dir,
        )
        missing, unexpected = model.load_state_dict(
            checkpoint["model_state_dict"], strict=False
        )
        if missing or unexpected:
            raise ArtifactError(
                f"[{self.model_id}] state dict does not match the architecture.\n"
                f"  missing:    {list(missing)[:8]}\n"
                f"  unexpected: {list(unexpected)[:8]}"
            )
        model.to(self._device)
        model.eval()

        self._model = model
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self._spec.path("tokenizer")), local_files_only=True
        )
        self._herg_threshold = self._read_herg_threshold()
        self._tox21_thresholds = self._read_tox21_thresholds()
        offline = " (offline, vendored config)" if base_config_dir else " (config fetched from Hugging Face)"
        self._detail = f"loaded from {ckpt_path.name}{offline}"

    def _read_herg_threshold(self) -> float:
        import json

        payload = json.loads(self._spec.path("herg_threshold.json").read_text())
        value = payload.get("threshold")
        if value is None:
            raise ArtifactError(f"[{self.model_id}] herg_threshold.json has no 'threshold'")
        return float(value)

    def _read_tox21_thresholds(self) -> dict[str, float]:
        import json

        payload = json.loads(self._spec.path("tox21_task_thresholds.json").read_text())
        table = payload.get("task_thresholds")
        if not isinstance(table, dict):
            raise ArtifactError(
                f"[{self.model_id}] tox21_task_thresholds.json has no 'task_thresholds' map"
            )
        absent = [t for t in TOX21_TASKS if t not in table]
        if absent:
            raise ArtifactError(f"[{self.model_id}] no threshold for Tox21 task(s): {absent}")
        return {task: float(table[task]) for task in TOX21_TASKS}

    def health(self) -> ModelHealth:
        return ModelHealth(
            model_id=self.model_id,
            loaded=self._model is not None,
            capabilities=self.capabilities,
            detail=self._detail,
        )

    # -- inference ---------------------------------------------------------
    def predict(self, canonical_smiles: list[str]) -> ProviderBatchResult[HergTox21RawOutput]:
        """Raw sigmoid outputs per molecule. Input must already be canonical."""
        import torch

        if self._model is None or self._tokenizer is None:
            raise ArtifactError(f"[{self.model_id}] predict() called before load()")
        if not canonical_smiles:
            return ProviderBatchResult(self.model_id, ())

        results: list[HergTox21RawOutput] = []
        for start in range(0, len(canonical_smiles), self._batch_size):
            chunk = canonical_smiles[start : start + self._batch_size]
            enc = self._tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            )
            untruncated = self._tokenizer(
                chunk,
                padding=False,
                truncation=False,
                add_special_tokens=True,
            )
            input_ids = enc["input_ids"].to(self._device)
            attention_mask = enc["attention_mask"].to(self._device)
            with torch.inference_mode():
                heads = self._model.forward_heads(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                herg = torch.sigmoid(heads["herg_logits"]).cpu().numpy().reshape(-1)
                tox21 = torch.sigmoid(heads["tox21_logits"]).cpu().numpy()

            for i in range(len(chunk)):
                row = tox21[i].reshape(-1)
                if row.shape[0] != len(TOX21_TASKS):
                    raise ArtifactError(
                        f"[{self.model_id}] Tox21 head produced {row.shape[0]} outputs, "
                        f"expected {len(TOX21_TASKS)}"
                    )
                input_token_count = len(untruncated["input_ids"][i])
                encoded_token_count = int(attention_mask[i].sum().item())
                results.append(
                    HergTox21RawOutput(
                        model_id=self.model_id,
                        herg_probability_blocker=float(herg[i]),
                        tox21_probability_activity={
                            task: float(row[j]) for j, task in enumerate(TOX21_TASKS)
                        },
                        tokenization=TokenizationProvenance(
                            input_token_count=input_token_count,
                            encoded_token_count=encoded_token_count,
                            max_length=self._max_length,
                            truncated=input_token_count > self._max_length,
                        ),
                    )
                )
        return ProviderBatchResult(self.model_id, tuple(results))


    # -- attribution -------------------------------------------------------
    ATTRIBUTION_METHOD = "grad_x_input_v2"

    def token_attribution(
        self, canonical_smiles: str, *, head: str, task_index: int | None = None,
        method: str = "grad_x_input", ig_steps: int = 32,
    ) -> dict[str, Any]:
        """Gradient x input-embedding norm, per token, for one head.

        Deterministic: a single backward pass with no sampling and no dropout,
        so repeated calls on the same input give identical scores. Returns
        numbers only — rendering is not this layer's business.
        """
        import torch

        if self._model is None or self._tokenizer is None:
            raise ArtifactError(f"[{self.model_id}] token_attribution() called before load()")
        if head not in {"herg", "tox21"}:
            raise ValueError(f"unknown head {head!r}")
        if head == "tox21" and task_index is None:
            raise ValueError("task_index is required when attributing the tox21 head")
        if method not in {"grad_x_input", "integrated_gradients"}:
            raise ValueError(f"unknown attribution method {method!r}")
        if method == "integrated_gradients" and not 8 <= ig_steps <= 256:
            raise ValueError("integrated gradients steps must be in [8, 256]")

        enc = self._tokenizer(
            [canonical_smiles], padding=True, truncation=True,
            max_length=self._max_length, return_tensors="pt",
            return_offsets_mapping=True,
        )
        input_ids = enc["input_ids"].to(self._device)
        attention_mask = enc["attention_mask"].to(self._device)
        # Char spans into ``canonical_smiles`` for each token, so a downstream
        # explainer can project token importances onto atom indices without
        # re-tokenising. Fast tokenisers only; ChemBERTa's is one.
        offsets = enc["offset_mapping"][0].tolist()

        embedding_layer = self._model.backbone.get_input_embeddings()
        actual_embeddings = embedding_layer(input_ids).detach()

        def target_for(embeddings):
            backbone_out = self._model.backbone(
                inputs_embeds=embeddings, attention_mask=attention_mask
            )
            cls = backbone_out.last_hidden_state[:, 0, :]
            logits = (
                self._model.herg_head(cls) if head == "herg" else self._model.tox21_head(cls)
            )
            return logits.reshape(-1)[0 if head == "herg" else int(task_index)]

        if method == "grad_x_input":
            embeddings = actual_embeddings.clone().requires_grad_(True)
            self._model.zero_grad(set_to_none=True)
            target = target_for(embeddings)
            target.backward()
            if embeddings.grad is None:
                raise ArtifactError(f"[{self.model_id}] attribution produced no gradient")
            contributions = embeddings.grad * embeddings
        else:
            baseline = torch.zeros_like(actual_embeddings)
            accumulated = torch.zeros_like(actual_embeddings)
            target = target_for(actual_embeddings)
            for alpha in torch.linspace(0.0, 1.0, ig_steps, device=actual_embeddings.device):
                interpolated = (
                    baseline + alpha * (actual_embeddings - baseline)
                ).detach().requires_grad_(True)
                self._model.zero_grad(set_to_none=True)
                current = target_for(interpolated)
                current.backward()
                if interpolated.grad is None:
                    raise ArtifactError(
                        f"[{self.model_id}] integrated gradients produced no gradient"
                    )
                accumulated += interpolated.grad.detach()
            contributions = (actual_embeddings - baseline) * accumulated / float(ig_steps)

        signed = contributions.sum(dim=-1).detach().cpu().numpy().reshape(-1)
        magnitudes = contributions.abs().sum(dim=-1).detach().cpu().numpy().reshape(-1)
        tokens = self._tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        mask = attention_mask[0].cpu().numpy().reshape(-1)

        kept = [
            {
                "token": tok,
                "position": i,
                "signed_contribution": float(signed[i]),
                "magnitude": float(magnitudes[i]),
                "importance": float(magnitudes[i]),
                "offsets": [int(offsets[i][0]), int(offsets[i][1])],
            }
            for i, tok in enumerate(tokens)
            if mask[i] == 1
        ]
        total = sum(t["magnitude"] for t in kept) or 1.0
        for t in kept:
            t["relative_importance"] = t["magnitude"] / total
        return {
            "method": (
                "integrated_gradients_v1"
                if method == "integrated_gradients"
                else self.ATTRIBUTION_METHOD
            ),
            "target": "logit",
            "mapping_version": "token-offset-v2",
            "model_id": self.model_id,
            "probability": float(torch.sigmoid(target.detach()).item()),
            "tokens": kept,
        }


def factory(spec: ArtifactSpec, device: str = "cpu") -> HergTox21ChembertaProvider:
    return HergTox21ChembertaProvider(spec, device=device)
