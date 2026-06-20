# BM1 Explainer Benchmark — Phân Tích Toàn Diện

> Phân tích này dựa trên inspection trực tiếp toàn bộ repository `tox-agent` (thay vì chỉ GitHub remote),
> mapping từng phần của plan research vào trạng thái thực tế của codebase.

---

## 1. Quy Trình BM1 Benchmark Tổng Quan

BM1 benchmark đánh giá chất lượng *explanation* (heatmap/importance score) của GNNExplainer
khi áp dụng lên model GATv2 của ToxAgent. Pipeline gồm 4 tầng:

```
[Data Layer]         Tox21 (labels) + SMARTS structural alerts (rationale ground-truth)
      ↓
[Prediction Layer]   GATv2 forward pass → filter molecules correct + high-confidence
      ↓
[Explanation Layer]  GNNExplainer → atom/bond importance scores
      ↓
[Scoring Layer]      So sánh heatmap với rationale mask → Attribution-AUROC, Hit@k, Fidelity...
```

**Điểm quan trọng**: Tầng Explanation hoàn toàn tách biệt với tầng Prediction.
Model đã được train sẵn; benchmark chỉ đánh giá *lý do* model đưa ra dự đoán có hợp lý không.

---

## 2. Repo Hiện Tại Đã Có Gì?

### ✅ Đã có đầy đủ

| Thành phần | Trạng thái | Chi tiết |
|---|---|---|
| **GNNExplainer implementation** | ✅ **Đã có, production-ready** | `backend/gnn_explainer.py` (1258 dòng) — chứa `explain_molecule()`, `batch_explain()`, wrapper class, visualization. Hỗ trợ cả GNNExplainer, Gradient Saliency, Dual-head gradient |
| **GATv2 Tox21 model** | ✅ Đã train sẵn | `models/tox21_gatv2_model/` — checkpoint full |
| **XSmiles model (SMILES + GATv2 hybrid)** | ✅ Đã train sẵn | `models/smilesgnn_model/`, `smilesgnn_herg_exp_model/`, `smilesgnn_herg_exp_model_tuned_v1/` |
| **Tox21 pretrained GIN model** | ✅ Đã train sẵn | `models/tox21_pretrained_gin_model/` |
| **Ensemble models (3, 5, 6, weighted)** | ✅ Đã train sẵn | `models/tox21_ensemble/`, `tox21_ensemble5/`, `tox21_ensemble6/`, `tox21_weighted_ensemble/` |
| **Tox21 dataset** | ✅ Có sẵn | `data/tox21.csv.gz` — 12 endpoints, có scaffold split trong `data/tox21-featurized/RawFeaturizer/ScaffoldSplitter/` |
| **Prediction endpoint** | ✅ Có sẵn | `POST /predict`, `POST /predict/batch` trong `model_server/main.py` |
| **Explain endpoint** | ✅ Có sẵn | `POST /explain` — nhận SMILES, trả về atom/bond importance + heatmap base64 |
| **Analyze endpoint (3-in-1)** | ✅ Có sẵn | `POST /analyze` — clinical + mechanism + explanation + OOD trong một request |
| **RDKit + PyG** | ✅ Trong dependencies | `requirements.txt` + Dockerfile cài `torch-geometric`, `rdkit-pypi` |
| **Explain CLI script** | ✅ Có sẵn | `scripts/explain_smilesgnn.py` — chạy CLI-based GNNExplainer |
| **Visualization** | ✅ Có sẵn | `backend/gnn_explainer.py` — `visualize_explanation()` và `plot_element_importance()` |

### ❌ Chưa có / Cần tạo cho BM1

| Thành phần | Trạng thái | Ghi chú |
|---|---|---|
| **SMARTS structural alerts file** | ❌ Chưa có | Cần tạo từ RDKit FilterCatalog (PAINS + BRENK) hoặc ToxAlerts |
| **Rationale mask builder script** | ❌ Chưa có | Script match SMARTS → molecule → `rationale_atom_mask` |
| **Explain test set với rationales** | ❌ Chưa có | CSV chứa SMILES + label + rationale atom indices |
| **Benchmark metrics script** | ❌ Chưa có | Attribution-AUROC, Hit@k, Fidelity, Stability, Sparsity |
| **Batch explain runner** | ❌ Chưa có | Chạy GNNExplainer trên toàn bộ explain test set |
| **Output directory cho benchmark** | ❌ Chưa có | `outputs/explainer_benchmark/` và `figures/` |

---

## 3. Kiến Trúc Model & Explainer Mapping

Repo có **3 họ kiến trúc model** khác nhau, mỗi họ cần explainer riêng:

### Nhánh 1 — XSmiles (SMILESGNN Hybrid)

| Thuộc tính | Giá trị |
|---|---|
| **Kiến trúc** | Transformer encoder (SMILES) + GATv2 (graph) → Cross-attention fusion → Predictor head |
| **Explainer** | `GNNExplainer` (qua `SMILESGNNExplainerWrapper`) |
| **Cơ chế** | SMILES embedding frozen → GNNExplainer chỉ optimize mask trên graph pathway |
| **File** | `backend/gnn_explainer.py` — `explain_molecule()` |
| **Endpoints** | `POST /explain`, `POST /analyze` |
| **Unit giải thích** | Atom node + Bond edge |
| **Checkpoint** | `models/smilesgnn_model/`, `models/smilesgnn_herg_exp_model/` |

### Nhánh 2 — Pretrained GIN (Tox21)

| Thuộc tính | Giá trị |
|---|---|
| **Kiến trúc** | Hu et al. pretrained GIN + fine-tune head (12 Tox21 tasks) |
| **Explainer** | `GNNExplainer` (edge-only, node_mask=None) |
| **Cơ chế** | Hu pretrained GIN dùng categorical atom indices → node_mask gây lỗi forward pass. Chỉ mask edge → back-project to atom importance |
| **File** | `backend/gnn_explainer.py` — `explain_tox21_task_pretrained_gin()` |
| **Unit giải thích** | Bond edge → Atom (back-project) |
| **Checkpoint** | `models/tox21_pretrained_gin_model/` |

### Nhánh 3 — Dual-head Transformer (ChemBERTa, PubChem, MolFormer)

| Thuộc tính | Giá trị |
|---|---|
| **Kiến trúc** | Transformer backbone + clinical head + Tox21 head |
| **Explainer** | `Gradient Saliency` (token-level) |
| **Cơ chế** | Gradient của input embeddings → token importance → project to atom spans via offset mapping |
| **File** | `backend/gnn_explainer.py` — `explain_tox21_task_pretrained_dual_head_gradient()` |
| **Unit giải thích** | Token → Atom (via character offset) |
| **Checkpoint** | `models/pretrained_2head_herg_chemberta_model/`, `pretrained_2head_herg_pubchem_model/`, `pretrained_2head_herg_molformer_model/` |

### Nhánh 4 — GATv2 thuần (Tox21)

| Thuộc tính | Giá trị |
|---|---|
| **Kiến trúc** | GATv2 + Set2Set pooling + 12-task head |
| **Explainer** | `GNNExplainer` hoặc `Gradient Saliency` |
| **File** | `backend/gnn_explainer.py` — `explain_tox21_task()`, `explain_tox21_task_gradient()` |
| **Checkpoint** | `models/tox21_gatv2_model/` |

### Nhánh 5 — AttentiveFP / GPS / Fingerprint XGB

| Thuộc tính | Giá trị |
|---|---|
| **Kiến trúc** | AttentiveFP, Graph GPS, ECFP+XGBoost |
| **Explainer** | Chưa tích hợp explainer riêng |
| **Ghi chú** | XGBoost có SHAP TreeExplainer nhưng chưa implement |

**Kết luận cho BM1**: Theo kế hoạch thầy, BM1 chỉ scope cho **GNNExplainer trên GATv2**.
Điều này map vào:
- `POST /explain` (dùng XSmiles + GNNExplainer) — đã có sẵn
- `explain_tox21_task()` (dùng GATv2 thuần + GNNExplainer) — đã có sẵn
- `explain_tox21_task_pretrained_gin()` (dùng pretrained GIN + GNNExplainer) — đã có sẵn

---

## 4. Schema Output Hiện Tại

### `ExplainResponse` (từ `POST /explain`)

| Field | Type | Mô tả |
|---|---|---|
| `smiles` | `str` | SMILES đầu vào |
| `p_toxic` | `float` | Xác suất độc tính |
| `label` | `str` | TOXIC / NON_TOXIC |
| `top_atoms` | `List[AtomImportance]` | Top 10 atom quan trọng nhất |
| `top_bonds` | `List[BondImportance]` | Top 10 bond quan trọng nhất |
| `heatmap_base64` | `str` | PNG heatmap encoded base64 |
| `molecule_png_base64` | `Optional[str]` | PNG molecule depiction |
| `chemical_interpretation` | `str` | Giải thích ngắn gọn |
| `explainer_note` | `str` | Ghi chú về giới hạn explainer |

### `AtomImportance`

| Field | Type | Mô tả |
|---|---|---|
| `atom_idx` | `int` | Index trong molecule |
| `element` | `str` | Ký hiệu nguyên tố (C, N, O, ...) |
| `importance` | `float` | GNNExplainer score [0, 1] |
| `is_in_ring` | `bool` | Có trong ring không |
| `is_aromatic` | `Optional[bool]` | Có aromatic không |

### `BondImportance`

| Field | Type | Mô tả |
|---|---|---|
| `bond_idx` | `int` | Index của bond |
| `atom_pair` | `str` | Ví dụ: "C(5) - N(8)" |
| `bond_type` | `str` | Loại bond (1.0, 2.0, 1.5...) |
| `importance` | `float` | GNNExplainer score [0, 1] |

**Lưu ý**: Schema hiện tại chỉ trả về **top 10** atom và bond. BM1 cần full importance array
cho tất cả atom/bond để tính metrics. Cần sửa schema hoặc dùng batch script bypass API.

---

## 5. Workflow Inference Hiện Tại

### API Flow (production)

```
User POST /explain {smiles: "CCO", epochs: 200}
→ model_server/main.py — explain()
→ _ensure_models_loaded() — lazy load checkpoint
→ predict_batch() — lấy prediction trước
→ smiles_to_pyg_data() — featurize molecule
→ explain_molecule() — SMILESGNNExplainerWrapper → GNNExplainer(200 epochs)
→ Render heatmap PNG → Build AtomImportance/BondImportance lists
→ Return ExplainResponse JSON
```

### Batch Inference (scripts)

```
$ python scripts/explain_smilesgnn.py --smiles "CCO" --epochs 200
→ Gọi trực tiếp explain_molecule() không qua API
→ Output: console + file
```

### Agent Flow (ADK)

```
POST /agent/analyze
→ InputValidator → Orchestrator → ScreeningAgent → ResearcherAgent → WriterAgent
→ ScreeningAgent gọi POST /analyze (3-in-1: clinical + mechanism + explanation + OOD)
→ ScreeningAgent gọi tools/lookup_structural_alerts() để SMARTS matching
```

---

## 6. Data Layer Hiện Tại

### Dataset có sẵn

| File | Path | Dung lượng | Ghi chú |
|---|---|---|---|
| Tox21 | `data/tox21.csv.gz` | ~1MB | 12 endpoints, ~8000 compounds |
| ClinTox | `data/clintox.csv.gz` | ~500KB | FDA-approved vs failed |
| hERG Karim | `data/herg_karim.tab` | ~200KB | hERG blockade assay |
| Test set | `test_data/full_test_set.csv` | ~5KB | 50+ molecules |
| Screening lib | `test_data/screening_library.csv` | ~10KB | 100+ compounds |
| Toxic compounds | `test_data/toxic_compounds.csv` | ~2KB | Known toxins |
| Reference panel | `test_data/reference_panel.csv` | ~2KB | Reference standards |

### Model Artifacts

Thư mục `models/` chứa **31 checkpoint directories**:
- **XSmiles models**: `smilesgnn_model/`, `smilesgnn_herg_exp_model/`, `smilesgnn_herg_exp_model_tuned_v1/`
- **Dual-head models**: `pretrained_2head_herg_chemberta_model/`, `pretrained_2head_herg_pubchem_model/`, `pretrained_2head_herg_molformer_model/`
- **Tox21 GNN models**: `tox21_gatv2_model/`, `tox21_pretrained_gin_model/`, `tox21_attentivefp_model/`, `tox21_gps_model/`
- **Ensemble models**: `tox21_ensemble/`, `tox21_ensemble5/`, `tox21_ensemble6/`, `tox21_weighted_ensemble/`
- **Dual-head ensembles**: `dualhead_ensemble3/`, `dualhead_ensemble5/`, `dualhead_ensemble6/`, `dualhead_weighted_ensemble3/`
- **Others**: `gatv2_model/`, `gin_model/`, `gatv2_transfer_model/`
- **Ranking**: `dualhead_model_ranking.csv`, `dualhead_model_ranking.json`

---

## 7. BM1 Implementation Checklist (Cập Nhật)

Dựa trên inspection thực tế, đây là checklist chi tiết:

### Phase 0 — Kiểm tra & Chuẩn bị (1-2 giờ)

- [x] **`model_server/main.py`** — Đã xác định: `explain()` endpoint ở ~line 4117-4217. GNNExplainer đã implement đầy đủ
- [x] **`model_server/schemas.py`** — Đã xác định: `ExplainResponse` có `top_atoms: List[AtomImportance]` và `top_bonds: List[BondImportance]` nhưng **chỉ top 10** — cần full array cho benchmark
- [x] **`model_server/scripts/download_model_artifacts.py`** — Model artifacts download từ cloud storage
- [x] **`requirements.txt`** — `torch-geometric`, `rdkit-pypi`, `scikit-learn` đã có
- [ ] **Kiểm tra model checkpoint có tồn tại local không** — `models/tox21_gatv2_model/` đã có, nhưng cần xác nhận weights load được

### Phase 1 — Thu thập & Chuẩn hóa Data

- [ ] **Tạo explain test set**: Lọc từ `data/tox21.csv.gz` endpoint SR-p53 — molecules model predict đúng + confidence > 0.8
- [ ] **Build SMARTS alert list**: Dùng RDKit `FilterCatalog` (PAINS + BRENK) — không cần download ngoài
- [ ] **Viết `build_rationale.py`**: Match SMARTS → tạo `rationale_atom_mask` binary array
- [ ] **Validate filter**: 5-60 heavy atoms, valid SMILES, có label

### Phase 2 — Chạy GNNExplainer Batch

- [ ] **Viết `run_gnnexplainer_batch.py`**: Bypass API, load model checkpoint trực tiếp
  - Có thể extend `scripts/explain_smilesgnn.py` đã có
  - Dùng `explain_tox21_task()` từ `backend/gnn_explainer.py` — đã implement sẵn batch logic qua `batch_explain()`
- [ ] **Output**: `gnnexplainer_atom_scores.csv`, `gnnexplainer_bond_scores.csv`, `gnnexplainer_runtime.csv`

### Phase 3 — Tính Metrics

- [ ] **Attribution-AUROC**: `sklearn.metrics.roc_auc_score(rationale_mask, atom_scores)` — chỉ với molecules có rationale
- [ ] **Hit@k**: Overlap giữa top-k atom và rationale atom indices
- [ ] **Fidelity**: Zero-out top-k node features → re-run model → P_orig - P_masked
- [ ] **Stability**: Jaccard similarity top-k giữa 5 lần chạy (seed 1-5)
- [ ] **Sparsity**: (num_atoms - num_highlighted) / num_atoms
- [ ] **Negative control**: Molecule không có SMARTS alert → explainer không được highlight nhiều atom

### Phase 4 — Tổng hợp & Report

- [ ] Merge metrics vào `gnnexplainer_metrics.csv`
- [ ] Render heatmap figures → `outputs/explainer_benchmark/figures/`
- [ ] Viết section BM1 trong report template

---

## 8. Lưu Ý Quan Trọng

### Về việc Benchmark Explainer

1. **Không cần train model lại** — tất cả checkpoints đã có sẵn trong `models/`
2. **Không cần gọi API** — có thể gọi trực tiếp `explain_tox21_task()` từ Python
3. **Schema hiện tại chỉ trả top-10** — cần sửa `ExplainResponse` để trả full importance array
   hoặc dùng batch script bypass API
4. **GNNExplainer chạy 200 epochs/molecule** — 1000 molecules mất ~30-60 phút trên GPU

### Về ChemBERTa

- ChemBERTa dùng **gradient saliency** chứ không phải GNNExplainer
- BM1 theo kế hoạch thầy chỉ scope cho GNNExplainer → **chỉ áp dụng cho GATv2**
- Nếu cần benchmark ChemBERTa, cần BM1b riêng với Attribution-AUROC trên token-level

### Về Tox21 Endpoints

12 task trong Tox21, endpoint **SR-p53** được ưu tiên vì:
- Nhiều positive sample nhất
- Liên quan DNA damage → dễ match SMARTS
- Cơ chế sinh học rõ ràng

### File cần tạo mới

| File | Mục đích |
|---|---|
| `data/toxalerts_smarts.csv` | SMARTS patterns từ RDKit FilterCatalog |
| `scripts/build_rationale.py` | Match SMARTS → rationale mask |
| `scripts/run_gnnexplainer_batch.py` | Batch explain + save scores |
| `scripts/compute_explainer_metrics.py` | Attribution-AUROC, Hit@k, Fidelity, Stability, Sparsity |
| `outputs/explainer_benchmark/metrics.csv` | Kết quả tổng hợp |
| `outputs/explainer_benchmark/figures/` | Heatmap visualizations |

---

## 9. Tài Nguyên Tham Khảo

### File quan trọng nhất cần đọc

| File | Lý do |
|---|---|
| `backend/gnn_explainer.py` | Core explainer — `explain_molecule()`, `batch_explain()`, `explain_tox21_task()` |
| `model_server/main.py` (~line 4117-4217) | `explain()` endpoint handler |
| `backend/inference.py` | `load_model()`, `predict_batch()` |
| `scripts/explain_smilesgnn.py` | CLI explain script — có thể extend |
| `backend/graph_data.py` | `smiles_to_pyg_data()` — featurization |
| `backend/featurization.py` | Feature engineering |
| `backend/data.py` | `get_task_names()` — Tox21 task list |
| `backend/graph_models.py` | `create_gatv2_model()` |
| `backend/graph_models_hybrid.py` | `create_hybrid_model()` |
| `config/` | Training/inference config YAML files |

### Tox21 Task Names

Cần xác nhận từ `backend/data.py` — `get_task_names()` trả về 12 tasks bao gồm SR-p53, NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD, NR-PPAR-gamma, SR-ARE, SR-ATAD5, SR-HSE, SR-MMP.
