"""
Pre-trained GIN backbone utilities (Hu et al., NeurIPS 2020).

Implements:
- Hu et al. graph featurization (atom/chirality + bond type/direction indices)
- Backbone + predictor architecture compatible with released checkpoints
- Helper to download and load pretrained backbone weights
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops

# Must match Hu et al. dimensions for checkpoint compatibility.
NUM_ATOM_TYPE = 120
NUM_CHIRALITY_TAG = 3
NUM_BOND_TYPE = 6
NUM_BOND_DIR = 3

_ATOMIC_NUMS = list(range(1, 119))
_CHIRALITY = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
]
_BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
_BOND_DIRS = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
]

_GH_BASE = "https://github.com/snap-stanford/pretrain-gnns/raw/master/chem/model_gin"
PRETRAINED_URLS = {
    "masking": f"{_GH_BASE}/supervised_masking.pth",
    "contextpred": f"{_GH_BASE}/supervised_contextpred.pth",
    "infomax": f"{_GH_BASE}/supervised_infomax.pth",
    "edgepred": f"{_GH_BASE}/supervised_edgepred.pth",
}


def _safe_idx(values: list, value) -> int:
    try:
        return values.index(value)
    except ValueError:
        return len(values)


def mol_to_graph_hu2020(smiles: str, label: Optional[np.ndarray] = None) -> Optional[Data]:
    """Convert a SMILES string into Hu et al. graph features."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    node_rows = []
    for atom in mol.GetAtoms():
        atom_idx = _safe_idx(_ATOMIC_NUMS, atom.GetAtomicNum())
        chir_idx = min(_safe_idx(_CHIRALITY, atom.GetChiralTag()), 2)
        node_rows.append([atom_idx, chir_idx])
    x = torch.tensor(node_rows, dtype=torch.long)

    if mol.GetNumBonds() == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 2), dtype=torch.long)
    else:
        rows, cols, attrs = [], [], []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            b_type = _safe_idx(_BOND_TYPES, bond.GetBondType())
            b_dir = _safe_idx(_BOND_DIRS, bond.GetBondDir())
            rows.extend([i, j])
            cols.extend([j, i])
            attrs.extend([[b_type, b_dir], [b_type, b_dir]])
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        edge_attr = torch.tensor(attrs, dtype=torch.long)

    y = None
    if label is not None:
        arr = np.asarray(label, dtype=np.float32)
        y = torch.tensor(arr).unsqueeze(0)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, smiles=smiles)


def smiles_list_to_hu_dataset(smiles_list: List[str], labels: np.ndarray) -> List[Data]:
    dataset: List[Data] = []
    for idx, smiles in enumerate(smiles_list):
        item = mol_to_graph_hu2020(smiles, label=labels[idx])
        if item is not None:
            dataset.append(item)
    return dataset


class HuGINConv(MessagePassing):
    """Hu et al. GINConv implementation with edge embeddings and self-loops."""

    def __init__(self, emb_dim: int):
        super().__init__(aggr="add")
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim),
        )
        self.eps = nn.Parameter(torch.zeros(1))
        self.edge_embedding1 = nn.Embedding(NUM_BOND_TYPE, emb_dim)
        self.edge_embedding2 = nn.Embedding(NUM_BOND_DIR, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_index_sl, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        self_loop_attr = torch.zeros(x.size(0), 2, dtype=torch.long, device=edge_attr.device)
        self_loop_attr[:, 0] = 4  # dedicated self-loop type
        edge_attr_sl = torch.cat([edge_attr, self_loop_attr], dim=0)

        edge_emb = self.edge_embedding1(edge_attr_sl[:, 0]) + self.edge_embedding2(edge_attr_sl[:, 1])
        return self.propagate(edge_index_sl, x=x, edge_attr=edge_emb)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp((1.0 + self.eps) * aggr_out)


class HuGNNBackbone(nn.Module):
    """5-layer Hu et al. GIN backbone."""

    def __init__(
        self,
        emb_dim: int = 300,
        num_layers: int = 5,
        drop_ratio: float = 0.5,
        jk: str = "last",
    ):
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.num_layers = int(num_layers)
        self.drop_ratio = float(drop_ratio)
        self.jk = str(jk)

        self.x_embedding1 = nn.Embedding(NUM_ATOM_TYPE, self.emb_dim)
        self.x_embedding2 = nn.Embedding(NUM_CHIRALITY_TAG, self.emb_dim)

        self.gnns = nn.ModuleList([HuGINConv(self.emb_dim) for _ in range(self.num_layers)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(self.emb_dim) for _ in range(self.num_layers)])

    def forward(self, x, edge_index, edge_attr):
        h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        h_list = [h]
        for layer_idx, (conv, bn) in enumerate(zip(self.gnns, self.batch_norms)):
            h = bn(conv(h_list[layer_idx], edge_index, edge_attr))
            if layer_idx < self.num_layers - 1:
                h = F.dropout(F.relu(h), p=self.drop_ratio, training=self.training)
            else:
                h = F.dropout(h, p=self.drop_ratio, training=self.training)
            h_list.append(h)

        if self.jk == "last":
            return h_list[-1]
        return torch.stack(h_list[1:], dim=0).sum(dim=0)


class GNNPretrainedPredictor(nn.Module):
    """Hu backbone + graph pooling + multi-task classification head."""

    def __init__(self, backbone: HuGNNBackbone, num_tasks: int, head_dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.head_dropout = float(head_dropout)
        self.head = nn.Linear(backbone.emb_dim, int(num_tasks))

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.backbone(x, edge_index, edge_attr)
        g = global_mean_pool(h, batch)
        g = F.dropout(g, p=self.head_dropout, training=self.training)
        return self.head(g)


def download_hu_pretrained(strategy: str = "masking", cache_dir: str = "data/pretrained_gnns") -> Optional[Path]:
    if strategy not in PRETRAINED_URLS:
        raise ValueError(f"Unknown strategy '{strategy}'. Options: {list(PRETRAINED_URLS)}")

    url = PRETRAINED_URLS[strategy]
    destination = Path(cache_dir) / url.split("/")[-1]
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        return destination

    try:
        urllib.request.urlretrieve(url, str(destination))
        return destination
    except Exception:
        return None


def create_pretrained_gin_model(
    num_tasks: int = 12,
    strategy: str = "masking",
    cache_dir: str = "data/pretrained_gnns",
    emb_dim: int = 300,
    num_layers: int = 5,
    drop_ratio: float = 0.5,
    jk: str = "last",
    head_dropout: float = 0.1,
    freeze_backbone: bool = False,
) -> GNNPretrainedPredictor:
    """Construct predictor and load Hu et al. pretrained backbone when available."""
    backbone = HuGNNBackbone(
        emb_dim=int(emb_dim),
        num_layers=int(num_layers),
        drop_ratio=float(drop_ratio),
        jk=str(jk),
    )
    predictor = GNNPretrainedPredictor(
        backbone=backbone,
        num_tasks=int(num_tasks),
        head_dropout=float(head_dropout),
    )

    weights_path = download_hu_pretrained(strategy=str(strategy), cache_dir=str(cache_dir))
    if weights_path is not None:
        try:
            state = torch.load(weights_path, map_location="cpu")
            predictor.backbone.load_state_dict(state, strict=False)
        except Exception:
            pass

    if freeze_backbone:
        for param in predictor.backbone.parameters():
            param.requires_grad = False

    return predictor
