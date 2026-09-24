"""Inference-only ChemBERTa dual-head architecture.

Parameter/module names intentionally match the frozen training architecture so
the same state dict loads byte-for-byte.  Optimizers, training helpers and
dataset code do not belong in the serving package.
"""
from __future__ import annotations

from typing import Optional


def _defaults(model_id: str) -> tuple[bool, str]:
    if model_id == "ibm/MoLFormer-XL-both-10pct":
        return True, "pooler_output"
    return False, "last_hidden_state"


def create_chemberta_dual_head_model(
    *, pretrained_model: str, num_tox21_tasks: int = 12, dropout: float = 0.1,
    herg_hidden_dim: Optional[int] = None, use_herg_mlp: bool = True,
    base_config_dir: Optional[str] = None,
):
    import torch.nn as nn
    from transformers import AutoConfig, AutoModel

    trust_remote_code, cls_source = _defaults(pretrained_model)

    class ChembertaDualHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if base_config_dir:
                config = AutoConfig.from_pretrained(base_config_dir, local_files_only=True)
                self.backbone = AutoModel.from_config(config)
            else:
                self.backbone = AutoModel.from_pretrained(
                    pretrained_model, trust_remote_code=trust_remote_code
                )
            self.cls_source = cls_source
            hidden_size = getattr(self.backbone.config, "hidden_size", None)
            if hidden_size is None:
                hidden_size = getattr(self.backbone.config, "d_model", None)
            if hidden_size is None:
                raise ValueError("cannot determine backbone hidden size")
            self.dropout = nn.Dropout(float(dropout))
            self.tox21_head = nn.Linear(int(hidden_size), int(num_tox21_tasks))
            if use_herg_mlp:
                mid = int(herg_hidden_dim or max(32, int(hidden_size) // 2))
                self.herg_head = nn.Sequential(
                    nn.Linear(int(hidden_size), mid), nn.GELU(),
                    nn.Dropout(float(dropout)), nn.Linear(mid, 1),
                )
            else:
                self.herg_head = nn.Linear(int(hidden_size), 1)
            self.num_tox21_tasks = int(num_tox21_tasks)
            self.hidden_size = int(hidden_size)

        def _get_cls(self, output):
            if self.cls_source == "pooler_output" and getattr(output, "pooler_output", None) is not None:
                return output.pooler_output
            return output.last_hidden_state[:, 0, :]

        def encode(self, input_ids, attention_mask):
            return self._get_cls(self.backbone(input_ids=input_ids, attention_mask=attention_mask))

        def forward(self, input_ids, attention_mask):
            return self.herg_head(self.dropout(self.encode(input_ids, attention_mask))).squeeze(-1)

        def forward_tox21(self, input_ids, attention_mask):
            return self.tox21_head(self.dropout(self.encode(input_ids, attention_mask)))

        def forward_heads(self, input_ids, attention_mask):
            cls = self.dropout(self.encode(input_ids, attention_mask))
            return {
                "herg_logits": self.herg_head(cls).squeeze(-1),
                "tox21_logits": self.tox21_head(cls),
            }

    return ChembertaDualHead()
