Để trả lời đầy đủ và chính xác, mình đã khám phá repo [NEU-Bio-Research-Team/tox-agent](https://github.com/NEU-Bio-Research-Team/tox-agent) và cấu trúc `model_server`. Dưới đây là toàn bộ phân tích — bắt đầu từ checklist nghiên cứu mà mình lập ra để không bỏ sót điểm nào, sau đó giải thích quy trình tổng quát, rồi map vào trạng thái thực tế của repo.

***

## Checklist nghiên cứu mình đã tự lập

Trước khi trả lời, mình xác định 5 câu hỏi gốc của bạn và lập checklist tương ứng:

**Câu hỏi 1:** BM1 Explainer Benchmark là quy trình gồm những bước nào? ✅ cần giải thích từ đầu
**Câu hỏi 2:** Hệ thống có nhiều model (GATv2, ChemBERTa...) → mỗi model dùng explainer khác nhau ra sao? ✅ cần phân tích
**Câu hỏi 3:** Repo hiện tại đã có sẵn gì? ✅ đã kiểm tra cấu trúc repo qua GitHub MCP
**Câu hỏi 4:** Cần thu thập thêm data gì? ✅ cần xác định nguồn
**Câu hỏi 5:** Workflow xử lý data hiện tại trong ToxAgent là gì? ✅ cần kiểm tra `model_server/main.py` và `scripts/`

***

## Quy trình BM1 Benchmark: Tổng quan

Benchmark cho một Explainer **không phải là train/test model prediction** — đó là bước đánh giá xem *explanation* (heatmap/score) có phù hợp với ground-truth structural risk không. Pipeline gồm 4 tầng lớn:

```
[Data Layer]         Tox21 (labels) + ToxAlerts/SMARTS (rationale ground-truth)
      ↓
[Prediction Layer]   GNN/ChemBERTa dự đoán → lọc molecule đúng + confidence cao
      ↓
[Explanation Layer]  Chạy Explainer → atom/bond importance scores (heatmap)
      ↓
[Scoring Layer]      So heatmap với rationale mask → Attribution-AUROC, Hit@k, Fidelity...
```

Điểm quan trọng bạn cần nắm vững: **Tầng Explanation hoàn toàn tách biệt với tầng Prediction**. Model dự đoán độc tính đã được train xong; benchmark chỉ đánh giá *lý do tại sao* model đưa ra dự đoán đó có hợp lý không .

***

## Vấn đề nhiều loại Model → nhiều loại Explainer

Đây là điểm bạn băn khoăn đúng. Hệ thống ToxAgent của bạn có ít nhất 2 kiến trúc model khác nhau, và mỗi loại đòi hỏi explainer riêng:

| Model | Kiến trúc | Explainer phù hợp | Đơn vị explanation |
|---|---|---|---|
| GATv2 / GNN | Graph Neural Network | **GNNExplainer**, PGExplainer, SubgraphX | Atom node, Bond edge |
| ChemBERTa | Transformer (sequence) | **SHAP** (token-level), LIME, Integrated Gradients | Token (SMILES char/fragment) |
| Random Forest / ECFP | Tree/fingerprint | **SHAP TreeExplainer** | Bit position trong fingerprint |

**GNNExplainer** chỉ applicable cho GATv2 vì nó hoạt động bằng cách học một mask \(m \in [0,1]^{|E|}\) tối thiểu hóa entropy của phân phối dự đoán sau khi mask graph. ChemBERTa không có graph structure, nên SHAP token-level hoặc attention rollout mới có ý nghĩa ở mức atom. Kế hoạch thầy bạn đưa ra (BM1) chỉ focus vào GNNExplainer, tức là **chỉ áp dụng cho nhánh GATv2 của ToxAgent** — bạn cần xác nhận điều này với thầy.

***

## Trạng thái hiện tại trong Repo

Từ cấu trúc repo đã khảo sát , đây là những gì đã có và chưa có:

### ✅ Đã có sẵn

- **`model_server/main.py`** (241KB — file rất lớn): Đây là nơi chứa toàn bộ prediction logic, API routes, và nhiều khả năng đã có GATv2 inference + GNNExplainer endpoint. File này cần được đọc kỹ để xác định endpoint nào đã expose explanation.
- **`model_server/schemas.py`**: Định nghĩa input/output schema, bao gồm nhiều khả năng đã có field `atom_weights` hoặc `bond_weights`.
- **`model_server/scripts/download_model_artifacts.py`**: Script download model đã train từ cloud (GCS hoặc tương đương) — tức model GATv2 đã được train và lưu ở đâu đó.
- **`outputs/`** directory: Đã có thư mục output, nhưng chưa rõ nội dung.
- **`test_data/`** directory: Có thể đã có một số molecule test.
- **`results/`** directory: Có thể đã có kết quả từ experiment trước.
- **`scripts/`** directory ở root: Có thể chứa data processing scripts.
- **`environment.yml` và `requirements.txt`**: Dependencies đã được quản lý — cần kiểm tra có `torch-geometric`, `rdkit`, `captum` hay `torch-explain` không.

### ❌ Chưa có / Cần tạo thêm

- **`data/explainer_benchmark/`** — Toàn bộ thư mục này chưa có trong repo 
- **`toxalerts_smarts.csv`** — File SMARTS alerts chưa thấy trong repo
- **`tox21_explain_test_with_rationales.csv`** — Chưa có
- **Script tạo rationale mask từ SMARTS** — Chưa thấy
- **Script tính Attribution-AUROC, Fidelity, Stability** — Chưa thấy
- **Benchmark runner** để loop qua explain_test và tính metrics — Chưa có

***

## Nguồn Data cần thu thập

Bạn cần 2 nguồn data độc lập, với cách thu thập như sau:

### Lớp 1 — Tox21 (Prediction Data)

Tox21 là dataset công khai, bạn có thể download từ:
- **DeepChem** (cách nhanh nhất): `deepchem.molnet.load_tox21()` — tự động chia scaffold split, có label cho 12 endpoints bao gồm SR-p53.
- **Tox21 Challenge website** (raw): https://tripod.nih.gov/tox21/challenge/ — file SDF/CSV gốc.
- **MoleculeNet benchmark**: Tích hợp trong DeepChem, đã được chuẩn hóa.

Endpoint ưu tiên theo kế hoạch thầy: **SR-p53** (stress response p53 signaling), vì endpoint này có nhiều positive sample và liên quan đến cơ chế DNA damage rõ ràng — dễ match SMARTS hơn.

### Lớp 2 — ToxAlerts/SMARTS (Rationale Ground-Truth)

- **ToxAlerts database**: https://toxalerts.mista.io — web tool, có thể export SMARTS. Tuy nhiên không có bulk download API dễ dùng.
- **SMARTS từ ChEMBL Structural Alerts**: Download từ ChEMBL FTP `ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/` — file `chembl_structural_alerts.csv` chứa hàng nghìn SMARTS pattern có annotation theo assay.
- **BRENK et al. / Lilly MedChem Rules**: Các bộ SMARTS có sẵn trên GitHub, ví dụ repo `PatWalters/rd_filters` có `alert_collection.csv` với ~480 SMARTS.
- **RDKit contrib**: `rdkit/Contrib/SA_Score` và `FilterCatalog` trong RDKit có PAINS alerts.

**Recommendation thực tế**: Dùng **RDKit's `FilterCatalog` với `PAINS_A/B/C` và `BRENK`** trước — không cần download file ngoài, built-in trong RDKit. Sau đó mới bổ sung từ ToxAlerts nếu cần coverage cao hơn.

***

## Chuẩn hóa Data từ nhiều nguồn

Tất cả data từ các nguồn khác nhau phải được đưa về schema chung:

```
compound_id | smiles | canonical_smiles | inchikey | endpoint | label | split | has_rationale | rationale_atom_indices | rationale_bond_indices
```

Quy trình chuẩn hóa bằng RDKit:
1. **Parse SMILES** → `Chem.MolFromSmiles()` → loại invalid
2. **Canonical SMILES** → `Chem.MolToSmiles(mol, canonical=True)`
3. **Dedup bằng InChIKey** → `Chem.InchiInfo.MolToInchi()` → `InchiInfo.InchiToInchiKey()` → giữ một record nếu label nhất quán, loại nếu conflict
4. **Lọc kích thước** → `mol.GetNumHeavyAtoms()` → chỉ giữ 5-60
5. **SMARTS matching** → `mol.GetSubstructMatches(pattern)` → lấy atom indices → build `rationale_atom_mask` dạng binary array độ dài bằng số atom

***

## Workflow xử lý Data hiện tại trong ToxAgent

Từ việc quan sát cấu trúc thư mục , workflow inference hiện tại trong ToxAgent đi theo luồng:

```
User SMILES input (API)
→ model_server/main.py (FastAPI endpoint)
→ SMILES parsing + featurization (node/edge features)
→ GATv2 forward pass
→ predicted_probability + predicted_label
→ [GNNExplainer nếu được gọi] → atom_scores, bond_scores
→ JSON response với visualization data
```

File `model_server/schemas.py` sẽ tiết lộ chính xác format response . Điểm cần lưu ý: pipeline hiện tại là **inference-only** (molecule đơn lẻ từ user), không phải batch evaluation. BM1 cần bạn viết thêm một **batch benchmark script** độc lập, không đi qua API — gọi trực tiếp model và GNNExplainer trên toàn bộ `explain_test.csv`.

***

## BM1 Implementation Checklist — Thứ tự thực hiện

Dưới đây là checklist theo thứ tự dependency, từ trên xuống dưới:

**Phase 0 — Kiểm tra repo (1-2 giờ)**
- [ ] Đọc `model_server/main.py` → xác định GNNExplainer đã được implement chưa, nếu có thì ở endpoint nào
- [ ] Đọc `model_server/schemas.py` → xem output schema có `atom_weights`/`bond_weights` không
- [ ] Kiểm tra `model_server/scripts/download_model_artifacts.py` → xem model lưu ở đâu (GCS bucket nào)
- [ ] Kiểm tra `requirements.txt` → xác nhận `torch-geometric`, `rdkit-pypi`, `scikit-learn` đã có

**Phase 1 — Thu thập & chuẩn hóa Data**
- [ ] Download Tox21 qua DeepChem với scaffold split → lưu `tox21_clean.csv`, `tox21_train/valid/test.csv`
- [ ] Build SMARTS alert list từ RDKit FilterCatalog (PAINS + BRENK) → lưu `toxalerts_smarts.csv`
- [ ] Viết `build_rationale.py`: match SMARTS vào mỗi molecule test → tạo `rationale_atom_mask` + `rationale_bond_mask`
- [ ] Lọc `explain_test.csv` (valid SMILES, có label, model predict đúng, confidence > threshold, 5-60 atoms)

**Phase 2 — Chạy GNNExplainer batch**
- [ ] Viết `run_gnnexplainer_batch.py` — bypass API, load model checkpoint trực tiếp
- [ ] Confirm GATv2 model được load từ checkpoint đúng
- [ ] Chạy GNNExplainer với 200 epochs mỗi molecule, seed cố định
- [ ] Lưu `gnnexplainer_atom_scores.csv`, `gnnexplainer_bond_scores.csv`, `gnnexplainer_runtime.csv`

**Phase 3 — Tính Metrics**
- [ ] **Attribution-AUROC**: `sklearn.metrics.roc_auc_score(rationale_atom_mask, atom_scores)` — chỉ tính khi `has_rationale=True`
- [ ] **Hit@k**: Count overlap giữa `top_k_atoms` và `rationale_atom_indices` / k
- [ ] **Fidelity**: Zero-out node features của top-k atoms → re-run model → \(P_{orig} - P_{masked}\)
- [ ] **Stability**: Chạy lại GNNExplainer với seed 1-5 → Jaccard overlap top-k giữa các lần chạy
- [ ] **Sparsity**: `(num_atoms - num_highlighted_atoms) / num_atoms`
- [ ] **Negative control**: Kiểm tra molecule không có SMARTS alert → Explainer không được highlight nhiều atom

**Phase 4 — Tổng hợp & Report**
- [ ] Merge tất cả metrics vào `gnnexplainer_metrics.csv`
- [ ] Render heatmap figures vào `outputs/explainer_benchmark/figures/`
- [ ] Viết section BM1 trong report theo template thầy đã cung cấp

***

## Điểm quan trọng về ChemBERTa

Bạn đang lo ngại rằng ChemBERTa không dùng GNNExplainer — **lo ngại này hoàn toàn đúng**. Nếu ToxAgent có endpoint dùng ChemBERTa để predict, bạn cần một **BM1b** riêng với SHAP token-level attribution. Tuy nhiên, kế hoạch thầy đưa ra chỉ đề cập GNNExplainer → **BM1 này chỉ scope cho GATv2**. ChemBERTa explanation benchmark là một task riêng, có thể để sau nếu thầy không yêu cầu trong BM1 .