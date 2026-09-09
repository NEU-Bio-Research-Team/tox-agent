"""
GPS Graph Transformer for molecular toxicity prediction.

GPS (General, Powerful, Scalable Graph Transformer) combines local MPNN
message-passing with global multi-head self-attention within each layer.

Reference:
    Rampášek et al., "Recipe for a General, Powerful, Scalable Graph
    Transformer", NeurIPS 2022.

Architecture per layer:
    h_local  = GINEConv(h, edge_index, edge_attr)
    h_global = MultiheadAttention(h, batch_mask)
    h_out    = LayerNorm(h + h_local + h_global)
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GPSConv
from torch_geometric.nn.conv import GINEConv
from torch_geometric.nn import global_mean_pool, global_max_pool


class GPSPredictor(nn.Module):
    """
    GPS Graph Transformer with a multi-task classification head.

    Node and edge features are first projected to a shared hidden dimension.
    Each GPS layer then applies local GINEConv message-passing and global
    multi-head attention in parallel, with residual connections and LayerNorm
    handled internally by GPSConv.

    Global graph representation: mean-pooling ‖ max-pooling (dim = 2 × hidden).
    """

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_channels: int = 128,
        num_layers: int = 4,
        heads: int = 4,
        dropout: float = 0.2,
        num_tasks: int = 1,
    ):
        super().__init__()

        # ── Input projections ────────────────────────────────────────────
        self.node_proj = nn.Linear(node_feat_dim, hidden_channels)
        self.edge_proj = nn.Linear(edge_feat_dim, hidden_channels)

        # ── GPS layers ───────────────────────────────────────────────────
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels * 2, hidden_channels),
            )
            local_conv = GINEConv(mlp, train_eps=True)
            self.convs.append(
                GPSConv(
                    channels=hidden_channels,
                    conv=local_conv,
                    heads=heads,
                    dropout=dropout,
                    attn_type="multihead",
                )
            )

        # ── Classification head ──────────────────────────────────────────
        self.norm = nn.LayerNorm(hidden_channels * 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_tasks),
        )
        self.num_tasks = num_tasks

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        x = self.node_proj(x)
        edge_attr = self.edge_proj(edge_attr)

        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=-1)

        x = self.norm(x)
        out = self.head(x)

        return out.squeeze(-1) if out.shape[-1] == 1 else out


def create_gps_model(
    node_feat_dim: int,
    edge_feat_dim: int,
    hidden_channels: int = 128,
    num_layers: int = 4,
    heads: int = 4,
    dropout: float = 0.2,
    num_tasks: int = 1,
) -> GPSPredictor:
    """Factory function for GPSPredictor."""
    return GPSPredictor(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        heads=heads,
        dropout=dropout,
        num_tasks=num_tasks,
    )
