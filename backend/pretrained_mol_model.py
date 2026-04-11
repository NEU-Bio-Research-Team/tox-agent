"""
Pre-trained molecular transformer models for toxicity prediction.

This module ports the upstream XSMILES pretrained foundation wrapper and
adds a dual-head variant for joint hERG + Tox21 training.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


CHECKPOINT_DEFAULTS: Dict[str, Dict[str, object]] = {
    "DeepChem/ChemBERTa-77M-MTR": {
        "max_length": 128,
        "trust_remote_code": False,
        "cls_source": "last_hidden_state",
    },
    "ibm/MoLFormer-XL-both-10pct": {
        "max_length": 202,
        "trust_remote_code": True,
        "cls_source": "pooler_output",
    },
    "seyonec/PubChem10M_SMILES_BPE_450k": {
        "max_length": 128,
        "trust_remote_code": False,
        "cls_source": "last_hidden_state",
    },
}


def get_checkpoint_defaults(checkpoint: str) -> Dict[str, object]:
    """Return per-checkpoint defaults with a safe fallback."""
    return CHECKPOINT_DEFAULTS.get(
        checkpoint,
        {
            "max_length": 128,
            "trust_remote_code": False,
            "cls_source": "last_hidden_state",
        },
    )


def _resolve_hidden_size(backbone: nn.Module) -> int:
    hidden_size = getattr(getattr(backbone, "config", None), "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(getattr(backbone, "config", None), "d_model", None)
    if hidden_size is None:
        raise ValueError("Cannot determine hidden size from backbone config")
    return int(hidden_size)


def _extract_encoder_layers(backbone: nn.Module) -> List[nn.Module]:
    """Best-effort extraction of transformer block list across HF architectures."""
    candidates = [
        getattr(getattr(backbone, "encoder", None), "layer", None),
        getattr(getattr(backbone, "encoder", None), "layers", None),
        getattr(getattr(backbone, "transformer", None), "layer", None),
        getattr(getattr(backbone, "transformer", None), "layers", None),
        getattr(backbone, "layers", None),
    ]

    for layers in candidates:
        if isinstance(layers, nn.ModuleList):
            return list(layers)
        if isinstance(layers, (list, tuple)) and all(isinstance(x, nn.Module) for x in layers):
            return list(layers)

    return []


class PretrainedMolPredictor(nn.Module):
    """Single-head pretrained molecular predictor (upstream-compatible)."""

    def __init__(
        self,
        pretrained_model: str,
        num_tasks: int = 1,
        dropout: float = 0.1,
        trust_remote_code: bool = False,
        cls_source: str = "last_hidden_state",
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            pretrained_model,
            trust_remote_code=trust_remote_code,
        )
        self.cls_source = str(cls_source)
        hidden_size = _resolve_hidden_size(self.backbone)
        self.dropout = nn.Dropout(float(dropout))
        self.head = nn.Linear(hidden_size, int(num_tasks))
        self.num_tasks = int(num_tasks)

    def _get_cls(self, backbone_output) -> torch.Tensor:
        if (
            self.cls_source == "pooler_output"
            and hasattr(backbone_output, "pooler_output")
            and backbone_output.pooler_output is not None
        ):
            return backbone_output.pooler_output
        return backbone_output.last_hidden_state[:, 0, :]

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self._get_cls(out)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        cls = self.encode(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.head(self.dropout(cls))
        if logits.shape[-1] == 1:
            return logits.squeeze(-1)
        return logits

    def get_token_importance(
        self,
        smiles: str,
        tokenizer: AutoTokenizer,
        task_idx: int = 0,
        device: str = "cpu",
        max_length: int = 128,
    ) -> Tuple[List[str], np.ndarray]:
        """Gradient x embedding-norm token importance for one SMILES."""
        self.eval()

        enc = tokenizer(
            smiles,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        embeds = self.backbone.embeddings.word_embeddings(input_ids)
        embeds = embeds.detach().requires_grad_(True)

        out = self.backbone(inputs_embeds=embeds, attention_mask=attention_mask)
        cls = self._get_cls(out)
        logits = self.head(self.dropout(cls))
        score = logits[0, task_idx] if logits.dim() == 2 else logits[0]
        score.backward()

        importance = embeds.grad[0].norm(dim=-1).detach().cpu().numpy()
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        return tokens, importance


class PretrainedMolDualHeadPredictor(nn.Module):
    """Shared pretrained backbone with separate hERG and Tox21 heads."""

    def __init__(
        self,
        pretrained_model: str,
        num_tox21_tasks: int = 12,
        dropout: float = 0.1,
        trust_remote_code: bool = False,
        cls_source: str = "last_hidden_state",
        herg_hidden_dim: Optional[int] = None,
        use_herg_mlp: bool = True,
    ):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(
            pretrained_model,
            trust_remote_code=trust_remote_code,
        )
        self.cls_source = str(cls_source)
        hidden_size = _resolve_hidden_size(self.backbone)

        self.dropout = nn.Dropout(float(dropout))
        self.tox21_head = nn.Linear(hidden_size, int(num_tox21_tasks))

        if use_herg_mlp:
            mid_dim = int(herg_hidden_dim or max(32, hidden_size // 2))
            self.herg_head = nn.Sequential(
                nn.Linear(hidden_size, mid_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(mid_dim, 1),
            )
        else:
            self.herg_head = nn.Linear(hidden_size, 1)

        self.num_tox21_tasks = int(num_tox21_tasks)
        self.hidden_size = int(hidden_size)

    def _get_cls(self, backbone_output) -> torch.Tensor:
        if (
            self.cls_source == "pooler_output"
            and hasattr(backbone_output, "pooler_output")
            and backbone_output.pooler_output is not None
        ):
            return backbone_output.pooler_output
        return backbone_output.last_hidden_state[:, 0, :]

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self._get_cls(out)

    def forward_tox21(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        cls = self.encode(input_ids=input_ids, attention_mask=attention_mask)
        return self.tox21_head(self.dropout(cls))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        cls = self.encode(input_ids=input_ids, attention_mask=attention_mask)
        return self.herg_head(self.dropout(cls)).squeeze(-1)

    def forward_heads(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        cls = self.encode(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(cls)
        return {
            "herg_logits": self.herg_head(cls).squeeze(-1),
            "tox21_logits": self.tox21_head(cls),
        }

    def freeze_backbone_layers(
        self,
        freeze_layers: int = 0,
        freeze_embeddings: bool = False,
    ) -> Dict[str, object]:
        """Freeze first N backbone blocks and optionally embeddings."""
        report: Dict[str, object] = {
            "freeze_layers_requested": int(max(0, freeze_layers)),
            "freeze_layers_applied": 0,
            "total_backbone_layers": 0,
            "freeze_embeddings": bool(freeze_embeddings),
            "num_frozen_params": 0,
        }

        layer_modules = _extract_encoder_layers(self.backbone)
        report["total_backbone_layers"] = len(layer_modules)

        num_to_freeze = min(len(layer_modules), int(max(0, freeze_layers)))
        for idx in range(num_to_freeze):
            for p in layer_modules[idx].parameters():
                p.requires_grad_(False)
        report["freeze_layers_applied"] = int(num_to_freeze)

        if bool(freeze_embeddings) and hasattr(self.backbone, "embeddings"):
            for p in self.backbone.embeddings.parameters():
                p.requires_grad_(False)

        num_frozen = 0
        for p in self.parameters():
            if not p.requires_grad:
                num_frozen += int(p.numel())
        report["num_frozen_params"] = int(num_frozen)

        return report


def create_pretrained_mol_model(
    pretrained_model: str,
    num_tasks: int = 12,
    dropout: float = 0.1,
) -> PretrainedMolPredictor:
    """Factory for the single-head predictor."""
    defaults = get_checkpoint_defaults(pretrained_model)
    return PretrainedMolPredictor(
        pretrained_model=pretrained_model,
        num_tasks=int(num_tasks),
        dropout=float(dropout),
        trust_remote_code=bool(defaults["trust_remote_code"]),
        cls_source=str(defaults["cls_source"]),
    )


def create_pretrained_dual_head_model(
    pretrained_model: str,
    num_tox21_tasks: int = 12,
    dropout: float = 0.1,
    herg_hidden_dim: Optional[int] = None,
    use_herg_mlp: bool = True,
) -> PretrainedMolDualHeadPredictor:
    """Factory for shared-backbone dual-head model (hERG + Tox21)."""
    defaults = get_checkpoint_defaults(pretrained_model)
    return PretrainedMolDualHeadPredictor(
        pretrained_model=pretrained_model,
        num_tox21_tasks=int(num_tox21_tasks),
        dropout=float(dropout),
        trust_remote_code=bool(defaults["trust_remote_code"]),
        cls_source=str(defaults["cls_source"]),
        herg_hidden_dim=herg_hidden_dim,
        use_herg_mlp=bool(use_herg_mlp),
    )
