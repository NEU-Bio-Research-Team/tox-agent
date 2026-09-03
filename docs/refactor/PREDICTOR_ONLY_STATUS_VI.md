# Predictor-only refactor — kết quả đối chiếu và trạng thái thực thi

> **Plan gốc:** `TOXAGENT_PREDICTOR_ONLY_REFACTOR_PLAN_VI.md`
> **Nhánh:** `docs/harness-master-plan`
> **Baseline commit:** `e6882b2`
> **Ngày đối chiếu:** 2026-09-03

---

## 1. Kết luận ngắn

**Plan đúng và nên làm.** Mọi khẳng định kỹ thuật trong mục 2.2 của plan đều được xác
minh trực tiếp trên code, không có điểm nào sai.

**Nhưng phạm vi phải sửa ở một chỗ quan trọng:** plan giả định giữ **hai** model để
serve. Trên thực tế chỉ **một** model serve được. Model ClinTox không load được và
không thể phục hồi từ repo này. Chi tiết ở mục 3.

Đây chính là stop condition mà plan tự đặt ra:

> *"Dừng deletion/cutover nếu: Selected checkpoint không load reproducibly trong
> container khóa version."*

Nên phần đã thực thi dừng đúng trước bước xoá, theo thứ tự plan quy định ở mục 13
("Sprint đầu không nên bắt đầu bằng xóa folder").

---

## 2. Đối chiếu từng khẳng định của plan

| # | Khẳng định trong plan | Kết quả kiểm chứng | Bằng chứng |
|---|---|---|---|
| — | 688 file trên nhánh | **Đúng, chính xác** | `git ls-files \| wc -l` = 688 |
| — | Phân bố thư mục (benchmark 183, frontend 152, .agents 84, models 51, agents 13, model_server 8, backend 27, src 25) | **Đúng toàn bộ** | `git ls-files \| awk -F/ '{print $1}' \| sort \| uniq -c` |
| — | `model_server/main.py` ~6.279 dòng | **Đúng** (6.278) | `wc -l` |
| 1 | `predict_pretrained_dual_head_outputs` ánh xạ hERG logit thành `clinical.p_toxic` | **Đúng** | `backend/inference.py:851-853`: `p_toxic = float(herg_probs[local_idx])` rồi `is_toxic = p_toxic >= clinical_threshold`, xuất dưới key `"clinical"` |
| 2 | API threshold default 0.35 vs artifact 0.413345 | **Đúng, và nặng hơn plan mô tả** | Artifact: `herg_threshold.json` = `0.4133453071117401`. Hằng số code = 0.35, nhưng `config/workspace_mode.yaml` đặt `threshold_policy: safety_first` → giá trị **thực tế đang chạy là 0.30** |
| 3 | Tox21 có 12 threshold riêng nhưng API dùng `mechanism_threshold` chung | **Đúng** | `tox21_task_thresholds.json` có 12 giá trị (0.35–0.94); `model_server/schemas.py:121` chỉ có một `mechanism_threshold` |
| 4 | `DEFAULT_TOX_TYPE_MODEL_KEY=tox21_ensemble_3_best` trỏ ensemble không tái lập được | **Đúng** | `main.py:202` → `main.py:321` map sang `models/dualhead_ensemble3/`, thư mục này chỉ chứa `dualhead_metrics.json`, **không có file weight nào** |
| 5 | Requirements/Dockerfile không đồng nhất | **Đúng** | `requirements.txt` hướng dẫn `torch==2.4.0`; `model_server/Dockerfile:52` cài `torch==2.6.0`; base image `python:3.10-slim`; môi trường dev hiện tại là torch 2.11.0 + numpy 2.2.6 trong khi requirements ghim `numpy<2.0` |
| 6 | Data loader đổi scaffold split → random split khi fallback DeepChem → PyTDC | **Đúng** | `backend/data.py:233-243`: nhánh fallback chạy `df.sample(frac=1)` rồi cắt 80/10/10, comment ngay trong code: *"For scaffold split, would need additional processing"* |
| 7 | Test suite không khóa predictor contract | **Đúng** | `tests/` chỉ có `test_report_chat_agent.py` (219 dòng) + 2 smoke ADK/agent |
| 8 | Backend workflow chỉ deploy `agent_test`, chưa có CI | **Đúng** | `.github/workflows/backend-autodeploy.yml:6-8` |

**Kết luận mục 2: plan không cần sửa phần chẩn đoán.**

---

## 3. Phát hiện mới — blocker mà plan chưa biết

### 3.1 Model ClinTox không serve được

Plan liệt kê `clintox-smilesgnn-v1` ở trạng thái **"Giữ"** (mục 2.1) và đặt endpoint
`clintox` vào contract đích (mục 4.3). Điều này **không khả thi**:

```
$ python -c "from backend.inference import load_model; load_model(...)"
FileNotFoundError: Tokenizer not found: models/smilesgnn_model/tokenizer.pkl
Run 'python scripts/train_hybrid.py' first.
```

Chuỗi bằng chứng:

1. `backend/inference.py:277` bắt buộc phải có `models/smilesgnn_model/tokenizer.pkl`.
2. Thư mục `models/smilesgnn_model/` chỉ có `best_model.pt`, `smilesgnn_model_metrics.txt`,
   `training_curves.png`. Không có tokenizer.
3. `.gitignore` dòng 49 loại trừ `*.pkl` → file này **chưa từng được commit**
   (`git log --all -- "*tokenizer.pkl"` trả về rỗng).
4. Checkpoint có `smiles_encoder.token_embedding.weight` shape **(69, 96)** — vocabulary
   69 token, sinh ra từ tập huấn luyện ClinTox. Config khai `smiles_vocab_size: 100`,
   tức là vocab thật là **data-derived**, không phải hằng số.
5. Hai tokenizer SMILES khác còn trên đĩa (`smilesgnn_herg_exp_model_tuned_v1`,
   `smilesgnn_multitask_model`) đều có vocab **80 token** → không khớp.

Không có vocabulary thì token id không tái tạo được, nên trọng số embedding vô nghĩa.
**Model này chỉ là một file weight không dùng được.**

### 3.2 Hệ quả: "clinical toxicity" mà sản phẩm đang trả về chính là hERG

Ghép ba sự thật ở trên lại:

| | |
|---|---|
| Model ClinTox | Không load được → không chạy |
| Cái gì điền vào field `clinical`? | Output head hERG của ChemBERTa (`inference.py:851`) |
| Threshold áp lên nó | 0.30 (`safety_first`), thay vì 0.4133 mà model được hiệu chuẩn |

Nghĩa là con số "độc tính lâm sàng" trên UI **là xác suất chẹn kênh hERG, đọc ở sai
điểm vận hành**. Đây không phải lỗi đặt tên — nó là một khẳng định khoa học sai, và
đúng là lý do quan trọng nhất để làm refactor này.

### 3.3 Ba điểm cần sửa trong plan

| Plan viết | Nên sửa thành |
|---|---|
| Giữ 2 model, `clintox-smilesgnn-v1` = "Giữ" | Giữ **1** model. ClinTox chuyển sang trạng thái *blocked*, chờ quyết định |
| Contract v1 có endpoint `clintox` (mục 4.3) | v1 chỉ có `herg` + `tox21`. Type `ClinToxPrediction` đã viết sẵn và có test, nhưng không đăng ký provider |
| "Áp threshold được đóng gói cùng artifact" (mục 1, ý 4) | Đúng cho ChemBERTa. ClinTox **không** có threshold trong artifact — nếu khôi phục được model thì phải hiệu chuẩn lại và đóng gói threshold cùng nó |

### 3.4 Lựa chọn cho ClinTox — cần anh/chị quyết

1. **Huấn luyện lại** bằng `scripts/train_hybrid.py`, lần này commit cả tokenizer + threshold
   vào artifact package. Tốn thời gian, nhưng lấy lại được endpoint `clintox`.
2. **Tìm lại tokenizer.pkl** từ máy đã train hoặc backup ngoài Git. Nếu tìm được thì
   phải verify: nạp vào và so raw probability với một baseline nào đó — hiện **không có**
   baseline ClinTox nào để đối chiếu, nên độ tin cậy thấp.
3. **Bỏ ClinTox khỏi phạm vi v1.** Predictor chỉ phục vụ hERG + Tox21, nói rõ trong
   model card. Đây là phương án trung thực nhất với hiện trạng và không chặn tiến độ.

Khuyến nghị: **(3) trước mắt, (1) song song** nếu đội vẫn cần tín hiệu clinical.

---

## 4. Đã thực thi những gì

Toàn bộ đều **thêm mới, không xoá gì**. Code cũ chạy y nguyên.

### Phase 0 — Đóng băng baseline ✅

| File | Nội dung |
|---|---|
| `benchmarks/build_golden_panel.py` | Sinh panel, tự từ chối ghi nếu có cấu trúc không parse được bằng RDKit |
| `benchmarks/fixtures/golden_panel.json` | **50 ca**: 42 hợp lệ + 6 invalid + 2 envelope-reject |
| `benchmarks/capture_baseline.py` | Ghi fingerprint runtime, SHA-256 mọi artifact, và **raw probability** |
| `benchmarks/manifests/baseline-e6882b2.json` | Runtime + checksum 11 file artifact |
| `benchmarks/golden/baseline_predictions.json` | Raw output cho 42 phân tử |

Panel phủ đúng yêu cầu của plan: 10 hERG blocker đã biết, 10 thuốc thông dụng,
8 hợp chất tham chiếu, 3 ca stereochemistry, 2 muối, 2 ion, 5 nguyên tố hiếm/cơ kim,
2 ca input dài, 6 SMILES sai, 2 ca chuỗi rỗng.

**Baseline có ý nghĩa khoa học** (không phải chỉ là số):

| Nhóm | hERG probability trung bình |
|---|---|
| 10 blocker đã biết (amiodarone, verapamil, astemizole…) | **0,682** |
| 10 thuốc thông dụng (aspirin, caffeine, ibuprofen…) | **0,078** |

Tox21 NR-ER: estradiol 0,973 · bisphenol A 0,989 · ethanol 0,199 — đúng chiều sinh học.
Latency CPU p50 3,0 ms / p95 4,5 ms. Ca trùng lặp cho kết quả **giống hệt từng bit**.

### Phase 1 — Khóa semantic contract ✅

| File | Nội dung |
|---|---|
| `toxpred/domain/endpoints.py` | `TOX21_TASKS` — 12 task theo **đúng thứ tự trong checkpoint**, versioned `tox21-12task-v1`; `validate_task_order()` fail loud khi artifact lệch |
| `toxpred/domain/policy.py` | `PredictionPolicySnapshot` — artifact là nguồn mặc định **duy nhất**; override phải mang `threshold_source="request_override"` |
| `toxpred/domain/prediction.py` | `ClinToxPrediction` / `HergPrediction` / `Tox21Prediction` là **ba type tách biệt**, mỗi type có tên field riêng |
| `toxpred/domain/molecule.py` | `Molecule`, `InvalidSmilesError` |

`HergPrediction` dùng field `probability_blocker`, label `blocker`/`non_blocker`.
`PredictionResult.to_dict()` dựng payload từ chính các type đó, nên **không có đường nào**
để một xác suất hERG xuất hiện dưới key `clinical`. Có test khẳng định điều này.

### Phase 2 — Artifact registry ✅

| File | Nội dung |
|---|---|
| `toxpred/scientific/artifacts.py` | Manifest schema + verify SHA-256 + size; báo **tất cả** lỗi trong một lần |
| `toxpred/scientific/registry.py` | `ModelProvider` protocol, `ModelRegistry`, tách liveness/readiness, **không fallback** |
| `artifacts/manifest.yaml` | Khai báo `herg-tox21-chemberta-v1` với checksum thật của 10 file |

Manifest ghi rõ hai artifact **không** đăng ký (ClinTox và `dualhead_ensemble3`) kèm lý do,
để lần sau không phải phát hiện lại.

### Phase 3B — Provider ChemBERTa ✅ + **parity PASS**

`toxpred/scientific/providers/herg_tox21_chemberta.py`:

- trả **raw probability**, không tự threshold — label do policy layer quyết
- `validate_task_order()` chạy lúc load
- `load_state_dict(strict=False)` rồi **fail nếu có missing/unexpected key**
- tokenizer đọc `local_files_only=True`
- không port code training

```
cases: 42
max |delta| hERG : 2.384e-07
max |delta| Tox21: 5.662e-07
PARITY PASS (tolerance 1e-06)
```

### Phase 4 (một phần) — Application service ✅

`toxpred/application/predictor.py`: `ToxicityPredictor.predict()` / `predict_batch()`.
Batch **giữ nguyên thứ tự input**, lỗi báo **theo từng item**, model chỉ chạy **một lần**
cho cả hai endpoint hERG + Tox21.

`toxpred/scientific/applicability.py`: port từ `ood_guard`, đổi tên thành
`element_rules_v1`. Trạng thái `ok` **nói rõ** rằng nó không xác nhận được sự tương đồng
phân bố — khác với `ood_risk: "LOW"` của bản cũ.

### Test suite ✅ — 90/90 pass

```
tests/unit/test_endpoints.py            6   thứ tự task Tox21
tests/unit/test_policy.py               9   threshold, biên >=, nguồn override
tests/unit/test_prediction_contract.py 14   hERG không thể thành clinical
tests/unit/test_artifacts.py           13   checksum, file thiếu, thư mục rỗng
tests/unit/test_import_boundaries.py   35   luật phụ thuộc theo AST
tests/unit/test_resolver.py            13   SMILES + applicability
tests/golden/test_provider_parity.py    6   đối chiếu baseline (cần artifact thật)
```

`test_import_boundaries.py` chặn tĩnh: `domain/` không được import torch, RDKit,
FastAPI, yaml, `backend`; không module nào trong `toxpred/` được import `agents`,
`services`, `model_server`, `src`, Firebase, DeepChem, MolScribe. Có cả test khẳng định
`import toxpred` **không** kéo theo torch/transformers/rdkit.

---

## 5. Chưa làm — và vì sao

| Phase | Trạng thái | Lý do |
|---|---|---|
| 3A — ClinTox provider | **Blocked** | Mục 3.1 |
| 4 — FastAPI app + `/v1/*` | Chưa | Nên làm cùng lúc với quyết định về ClinTox, vì nó định hình danh sách endpoint |
| 5 — Attribution | Chưa | Plan cho phép không chặn core |
| 6 — Benchmark khoa học đầy đủ | Chưa | Cần frozen split manifest; `backend/data.py` vẫn có fallback đổi split (mục 2 #6) |
| 7A — Cutover | Chưa | Cần Phase 4 |
| **7B — Xoá legacy** | **Cố ý chưa làm** | Plan cấm xoá trước cutover; stop condition đang bật |
| 8 — Slim deps/Docker/CI | Chưa | Sau 7B |

**Không có file nào bị xoá trong lần thực thi này.**

---

## 6. Một sai lệch có chủ ý so với plan

Plan đặt package ở `src/toxpred/`. Repo **đã có** `src/` là một package thật với
`__init__.py` và 20+ compatibility wrapper (`from backend.inference import *`). Đặt vào đó
sẽ khiến package chỉ import được dưới tên `src.toxpred`.

Package được đặt ở **`toxpred/`** tại repo root. Khi Phase 7B xoá `src/`, có thể chuyển
sang layout `src/` nếu muốn — không ảnh hưởng gì đến code đã viết.

---

## 7. Về mục tiêu "80–120 file"

Đạt được. Ước tính sau khi hoàn tất Phase 7B:

| Nhóm | File |
|---|---|
| `toxpred/` | ~22 |
| `tests/` | ~12 |
| `benchmarks/` | ~6 |
| `artifacts/`, `configs/`, `deploy/`, `.github/` | ~8 |
| `docs/` + README + pyproject | ~8 |
| **Tổng** | **~56** |

Thấp hơn cả khoảng plan dự kiến, vì chỉ còn một provider thay vì hai.

---

## 8. Việc tiếp theo

**Cần quyết định (chặn Phase 4):**

1. ClinTox: chọn phương án 1, 2 hay 3 ở mục 3.4?
2. Có consumer nào ngoài frontend đang gọi `/predict`, `/analyze` không? Ảnh hưởng tới
   độ dài deprecation window.

**Làm được ngay, không cần chờ:**

3. Ghim revision `DeepChem/ChemBERTa-77M-MTR` và vendor file config vào artifact —
   checkpoint đã chứa đủ trọng số backbone (61 tensor gồm `backbone.embeddings.*` và
   `backbone.pooler.*`), nên chỉ còn thiếu config kiến trúc là startup hết phụ thuộc mạng.
4. Bỏ `report_state` rehydration và tắt `mechanism_threshold` chung trong API cũ.
5. Sửa `backend/data.py` để fallback PyTDC **fail loud** thay vì âm thầm đổi sang random
   split — đây là rủi ro rò rỉ train/test đang tồn tại, độc lập với refactor.
